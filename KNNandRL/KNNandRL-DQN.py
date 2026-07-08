import json
import math
import os
import time
from datetime import datetime
from itertools import combinations

import matplotlib
import networkx as nx
import numpy as np
import xlrd

from DQN import DQN
from dijkstra import calculate_speed, get_shortest_path


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(BASE_DIR, f"DQN-{datetime.now().strftime('%m%d-%H%M')}")
SHOW_PLOTS = os.environ.get("KNN_DQN_SHOW_PLOTS", "0") == "1"
SAVE_PLOTS = os.environ.get("KNN_DQN_SAVE_PLOTS", "1") == "1"
SAVE_BEST_EPISODE = True

if not SHOW_PLOTS:
    matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402

# Only used to read the channel-rate matrix.
DATA_PATH = os.path.join(BASE_DIR, "..", "data", "15e_user400.xls")
SHEET_INDEX = 4
RESULT_PATH = os.path.join(OUTPUT_DIR, "result.json")

NODE_NUM = 15
ALL_CONNECTIONS_EDGE_NUM = NODE_NUM * (NODE_NUM - 1) // 2
SCORE1_LAMBDA = 1

SCORE1_WEIGHT = 0.5
SCORE2_WEIGHT = -0.005

EPISODE_NUM = 100
CUT_EDGE_RATE = 0.7
ALPHA = 0.3
NO_PARTITION_REWARD = -0.5
REWARD_OFFSET = 0.31

DQN_BATCH_SIZE = 32
DQN_LR = 0.01
DQN_EPSILON = 0.9
DQN_GAMMA = 0.9
DQN_TARGET_REPLACE_ITER = 100
DQN_MEMORY_CAPACITY = 2000
DQN_HIDDEN_DIM = 50

NORMALIZE_STATE = True


def read_xls_to_map(file_path, sheet_index):
    workbook = xlrd.open_workbook(file_path)
    sheet = workbook.sheet_by_index(sheet_index)

    data_map = {}
    for r in range(sheet.nrows):
        for c in range(sheet.ncols):
            data_map[(r, c)] = sheet.cell_value(r, c)
    return data_map


def get_speed(v_map, u, v):
    if (u, v) in v_map:
        return v_map[(u, v)]
    return v_map[(v, u)]


def build_distance_map(v_map, node_count):
    distance_map = {}
    for i in range(node_count):
        for j in range(node_count):
            if i == j:
                distance_map[(i, j)] = math.inf
                continue
            speed = get_speed(v_map, i, j)
            if speed <= 0:
                distance_map[(i, j)] = math.inf
            else:
                distance_map[(i, j)] = 1 / speed
    return distance_map


def knn_graph_from_map(distance_map, k=3, node_count=30):
    graph = nx.Graph()
    graph.add_nodes_from(range(node_count))

    for i in range(node_count):
        distances = [
            (j, distance_map[(i, j)])
            for j in range(node_count)
            if i != j
        ]
        distances = [
            (j, d)
            for j, d in distances
            if d >= 0 and math.isfinite(d)
        ]
        distances.sort(key=lambda x: x[1])

        for j, d in distances[:k]:
            graph.add_edge(i, j, weight=d)
    return graph


def count_islands(graph: nx.Graph) -> tuple[float, int, int]:
    island_list = list(nx.connected_components(graph))
    island_count = len(island_list)
    conn_score_k = (
        1 / island_count
        - SCORE1_LAMBDA * graph.number_of_edges() / ALL_CONNECTIONS_EDGE_NUM
    )
    return conn_score_k, island_count, graph.number_of_edges()


def distance_preservation_score(graph: nx.Graph, v_map: dict):
    island_list = list(nx.connected_components(graph))
    print("island_list: ", island_list)
    island_avg_list = []

    for island in island_list:
        island_nodes = list(island)
        if len(island_nodes) < 2:
            island_avg_list.append(0)
            continue

        graph_d = nx.Graph()
        island_v = [
            get_speed(v_map, island_nodes[i], island_nodes[j])
            for i in range(len(island_nodes))
            for j in range(i + 1, len(island_nodes))
        ]
        turn_graph_edges_to_positive = max(island_v) + 1

        for i in range(len(island_nodes)):
            for j in range(i + 1, len(island_nodes)):
                if not graph.has_edge(island_nodes[i], island_nodes[j]):
                    continue
                u, v = island_nodes[i], island_nodes[j]
                d = -get_speed(v_map, u, v) + turn_graph_edges_to_positive
                graph_d.add_edge(u, v, weight=d)

        sum_dk = 0
        for i in range(len(island_nodes)):
            for j in range(i + 1, len(island_nodes)):
                u, v = island_nodes[i], island_nodes[j]
                path = nx.shortest_path(graph_d, source=u, target=v, weight="weight")
                dijk = sum(
                    -get_speed(v_map, path[n], path[n + 1])
                    for n in range(len(path) - 1)
                )
                d_orig = -get_speed(v_map, u, v)
                sum_dk += abs(dijk - d_orig) / d_orig

        sum_avg_dk = sum_dk / (len(island_nodes) * (len(island_nodes) - 1))
        island_avg_list.append(sum_avg_dk)

    return sum(island_avg_list) / len(island_avg_list)


def build_base_state(edges, v_map):
    edge_speeds = np.array([get_speed(v_map, u, v) for u, v in edges], dtype=np.float32)
    if not NORMALIZE_STATE:
        return edge_speeds.tolist()

    positive_speeds = edge_speeds[edge_speeds > 0]
    scale = float(np.max(positive_speeds)) if len(positive_speeds) > 0 else 1.0
    return (edge_speeds / scale).tolist()


def available_actions_from_state(state):
    return np.flatnonzero(np.asarray(state, dtype=np.float32) > 0).tolist()


def calculate_partition_reward(temp_graph, v_map):
    groups = list(nx.connected_components(temp_graph))
    group_num = len(groups)

    if group_num == 1:
        return NO_PARTITION_REWARD, groups

    groups_avg_speed = []
    for group in groups:
        if len(group) < 2:
            continue

        pairs = [(u, v) for u, v in combinations(group, 2)]
        sum_speed = 0
        for u, v in pairs:
            shortest_path = get_shortest_path(v_map, NODE_NUM, u, v)
            sum_speed += calculate_speed(shortest_path, v_map)
        groups_avg_speed.append(sum_speed / len(pairs))

    if not groups_avg_speed or min(groups_avg_speed) <= 0:
        return NO_PARTITION_REWARD, groups

    mid_term = sum(groups_avg_speed) / group_num
    reward = (
        mid_term
        - ALPHA * mid_term ** 2 / min(groups_avg_speed)
        - REWARD_OFFSET
    )
    return reward, groups


def finish_plot(file_name=None, force_save=False, show_plot=True):
    if (SAVE_PLOTS or force_save) and file_name:
        output_path = os.path.join(OUTPUT_DIR, file_name)
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        plt.savefig(output_path, dpi=200, bbox_inches="tight")
    if show_plot and SHOW_PLOTS:
        plt.show()
    else:
        plt.close()


def plot_graph(graph, title, file_name=None, force_save=False, show_plot=True):
    plt.figure(figsize=(8, 6))
    pos = nx.spring_layout(graph, seed=42)
    nx.draw(
        graph,
        pos,
        node_size=200,
        node_color="skyblue",
        with_labels=True,
        edge_color="gray",
    )
    plt.title(title)
    finish_plot(file_name, force_save=force_save, show_plot=show_plot)


def save_best_episode_graphs(graphs, best_episode, best_step, title_prefix):
    if not SAVE_BEST_EPISODE or best_episode is None:
        return

    folder_name = f"best episode-{best_episode}"
    os.makedirs(os.path.join(OUTPUT_DIR, folder_name), exist_ok=True)
    for step, step_graph in enumerate(graphs, start=1):
        suffix = "-best" if step == best_step else ""
        plot_graph(
            step_graph,
            f"{title_prefix}\nepisode={best_episode}, step={step}",
            os.path.join(folder_name, f"step{step}{suffix}.png"),
            force_save=True,
            show_plot=False,
        )


def plot_rewards(best_reward_each_episode):
    plt.figure(figsize=(8, 5))
    plt.plot(best_reward_each_episode, marker="o")
    plt.xlabel("Episode")
    plt.ylabel("Best Reward")
    plt.title("Best Reward per Episode")
    plt.grid(True)
    plt.tight_layout()
    finish_plot("dqn_best_reward.png")


def choose_best_knn_graph(distance_map, v_map):
    scores = []
    k_values = list(range(1, NODE_NUM))
    result_str = ""

    for k in k_values:
        result_str += "k=" + str(k) + "\n"
        graph = knn_graph_from_map(distance_map, k=k, node_count=NODE_NUM)
        score1 = count_islands(graph)
        score2 = distance_preservation_score(graph, v_map)
        overall_score = SCORE1_WEIGHT * score1[0] + SCORE2_WEIGHT * score2
        scores.append(overall_score)
        print(
            f"k={k}: score1={score1}, score2={score2}, "
            f"overall_score={overall_score}"
        )
        result_str += (
            f"score1={score1}, score2={score2}, "
            f"overall_score={overall_score}\n"
        )

    print(result_str)
    best_k = k_values[int(np.argmax(scores))]
    graph = knn_graph_from_map(distance_map, k=best_k, node_count=NODE_NUM)
    return best_k, graph


def run_dqn_partition(graph, v_map):
    edges = list(graph.edges())
    if len(edges) == 0:
        return graph.copy(), [], [], None, None, -math.inf, []

    rl = DQN(
        n_states=len(edges),
        n_actions=len(edges),
        learning_rate=DQN_LR,
        reward_decay=DQN_GAMMA,
        e_greedy=DQN_EPSILON,
        target_replace_iter=DQN_TARGET_REPLACE_ITER,
        memory_capacity=DQN_MEMORY_CAPACITY,
        batch_size=DQN_BATCH_SIZE,
        hidden_dim=DQN_HIDDEN_DIM,
        mask_invalid_actions=True,
    )

    base_state = build_base_state(edges, v_map)
    cut_edge_num = round(len(edges) * CUT_EDGE_RATE)

    overall_best_reward = -math.inf
    overall_best_graph = graph.copy()
    overall_best_episode = None
    overall_best_step = None
    overall_best_episode_graphs = []
    best_reward_each_episode = []
    last_loss = None

    for episode in range(EPISODE_NUM):
        best_reward_in_this_episode = -math.inf
        best_graph_in_this_episode = graph.copy()
        best_step_in_this_episode = None
        episode_graphs = []
        state = base_state.copy()
        temp_graph = graph.copy()

        for cut_edge_count in range(cut_edge_num):
            available_actions = available_actions_from_state(state)
            if not available_actions:
                break

            action = rl.choose_action(state, available_actions=available_actions)

            next_state = state.copy()
            next_state[action] = 0

            u, v = edges[action]
            if temp_graph.has_edge(u, v):
                temp_graph.remove_edge(u, v)
            episode_graphs.append(temp_graph.copy())

            reward, _ = calculate_partition_reward(temp_graph, v_map)
            print(str(reward), end="\t")

            rl.store_transition(state, action, reward, next_state)
            loss = rl.learn()
            if loss is not None:
                last_loss = loss

            state = next_state
            if reward > best_reward_in_this_episode:
                best_reward_in_this_episode = reward
                best_graph_in_this_episode = temp_graph.copy()
                best_step_in_this_episode = cut_edge_count + 1

        print(f"\nbest reward: {best_reward_in_this_episode}")
        if last_loss is not None:
            print(f"last DQN loss: {last_loss}")
        best_reward_each_episode.append(best_reward_in_this_episode)

        if best_reward_in_this_episode > overall_best_reward:
            overall_best_reward = best_reward_in_this_episode
            overall_best_graph = best_graph_in_this_episode.copy()
            overall_best_episode = episode + 1
            overall_best_step = best_step_in_this_episode
            overall_best_episode_graphs = [step_graph.copy() for step_graph in episode_graphs]
        print(f"current overall best reward: {overall_best_reward}")

    return (
        overall_best_graph,
        best_reward_each_episode,
        edges,
        overall_best_episode,
        overall_best_step,
        overall_best_reward,
        overall_best_episode_graphs,
    )


def KNN_and_DQN():
    start_time = time.perf_counter()

    v_map = read_xls_to_map(DATA_PATH, sheet_index=SHEET_INDEX)
    distance_map = build_distance_map(v_map, NODE_NUM)

    k, graph = choose_best_knn_graph(distance_map, v_map)
    print(f"Testing with k={k}")
    distance_preservation_score(graph, v_map)
    print("G.edges: ")
    print(graph.edges())

    plot_graph(
        graph,
        f"KNN Graph (k={k}, nodes={NODE_NUM})",
        "dqn_initial_knn_graph.png",
    )

    (
        overall_best_graph,
        best_reward_each_episode,
        _,
        overall_best_episode,
        overall_best_step,
        overall_best_reward,
        overall_best_episode_graphs,
    ) = run_dqn_partition(graph, v_map)

    best_group_info = (
        f"best episode={overall_best_episode}, step={overall_best_step}, "
        f"reward={overall_best_reward:.6f}"
    )
    print(f"Best group info: {best_group_info}")
    plot_graph(
        overall_best_graph,
        f"KNN-DQN Graph (k={k}, nodes={NODE_NUM})\n{best_group_info}",
        "dqn_best_graph.png",
    )
    save_best_episode_graphs(
        overall_best_episode_graphs,
        overall_best_episode,
        overall_best_step,
        f"KNN-DQN Graph (k={k}, nodes={NODE_NUM})",
    )
    plot_rewards(best_reward_each_episode)

    components = nx.connected_components(overall_best_graph)
    node_group = [list(component) for component in components]

    elapsed_time = time.perf_counter() - start_time
    print(f"KNN + DQN algorithm running time: {elapsed_time:.6f} seconds")
    print(f"Output directory: {OUTPUT_DIR}")
    return node_group


def KNN_and_RL():
    return KNN_and_DQN()


if __name__ == "__main__":
    node_group = KNN_and_DQN()
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    with open(RESULT_PATH, "w", encoding="utf-8") as f:
        json.dump(node_group, f)
