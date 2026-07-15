"""KNN graph construction followed by discrete asynchronous A3C partitioning.

The A3C action is the index of one currently available KNN edge. Executing an
action removes that edge. This file only keeps the shared-network, shared-Adam,
n-step return, and asynchronous-worker ideas from the supplied discrete A3C
example; it does not depend on Gym or contain the CartPole demonstration.
"""

import json
import math
import os
import time
import traceback
from datetime import datetime
from itertools import combinations

os.environ.setdefault("OMP_NUM_THREADS", "1")

import matplotlib
import networkx as nx
import numpy as np
import torch
import torch.multiprocessing as mp
import torch.nn as nn
import torch.nn.functional as F
import xlrd

from dijkstra import calculate_speed, get_shortest_path


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(BASE_DIR)
OUTPUT_DIR = os.path.join(BASE_DIR, f"A3C-{datetime.now().strftime('%m%d-%H%M')}")
RESULT_PATH = os.path.join(OUTPUT_DIR, "result.json")
SHOW_PLOTS = os.environ.get("KNN_A3C_SHOW_PLOTS", "0") == "1"
SAVE_PLOTS = os.environ.get("KNN_A3C_SAVE_PLOTS", "1") == "1"
SAVE_BEST_EPISODE = True

if not SHOW_PLOTS:
    matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402


# 只用于读取速度矩阵。
DATA_PATH = os.path.join(PROJECT_ROOT, "data", "15e_user400.xls")
SHEET_INDEX = 4

NODE_NUM = 15
ALL_CONNECTIONS_EDGE_NUM = NODE_NUM * (NODE_NUM - 1) // 2
SCORE1_LAMBDA = 1
SCORE1_WEIGHT = 0.5
SCORE2_WEIGHT = -0.005
KNN_MIN_C = 1.0
KNN_EARLY_STOP_DECLINES = 3

EPISODE_NUM = 100
CUT_EDGE_RATE = 0.7
ALPHA = 0.3
NO_PARTITION_REWARD = -0.5
NORMALIZE_STATE = True

# A3C parameters. Workers can be overridden on the server, for example:
# KNN_A3C_WORKERS=8 python KNNandRL/KNNandRL-A3C.py
A3C_WORKER_NUM = max(
    1,
    int(os.environ.get("KNN_A3C_WORKERS", min(4, os.cpu_count() or 1))),
)
A3C_UPDATE_GLOBAL_ITER = 5
A3C_GAMMA = 0.9
A3C_LR = 1e-4
A3C_BETAS = (0.92, 0.999)
A3C_HIDDEN_DIM = 128
A3C_VALUE_COEF = 0.5
A3C_ENTROPY_COEF = 0.01
A3C_MAX_GRAD_NORM = 40.0
A3C_SEED = 42


def read_xls_to_map(file_path, sheet_index):
    """读取指定 .xls 工作表并返回以 (row, col) 为键的速度字典。"""
    workbook = xlrd.open_workbook(file_path)
    sheet = workbook.sheet_by_index(sheet_index)

    data_map = {}
    for row in range(sheet.nrows):
        for col in range(sheet.ncols):
            data_map[(row, col)] = sheet.cell_value(row, col)
    return data_map


def knn_graph_from_map(distance_map, k=3, node_count=30):
    graph = nx.Graph()
    graph.add_nodes_from(range(node_count))

    for source in range(node_count):
        distances = [
            (target, distance_map[(source, target)])
            for target in range(node_count)
            if source != target
        ]
        distances = [(target, distance) for target, distance in distances if distance >= 0]
        distances.sort(key=lambda item: item[1])
        for target, distance in distances[:k]:
            graph.add_edge(source, target, weight=distance)
    return graph


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
    position = nx.spring_layout(graph, seed=42)
    nx.draw(
        graph,
        position,
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
    finish_plot("a3c_best_reward.png")


def count_islands(graph: nx.Graph) -> tuple[float, int, int]:
    """KNN 指标1：连通分量数量和边数之间的平衡。"""
    island_count = nx.number_connected_components(graph)
    connectivity_score = (
        1 / island_count
        - SCORE1_LAMBDA * graph.number_of_edges() / ALL_CONNECTIONS_EDGE_NUM
    )
    return connectivity_score, island_count, graph.number_of_edges()


def distance_preservation_score(graph: nx.Graph, v_map: dict):
    """KNN 指标2：各连通分量内的距离保持程度。"""
    islands = list(nx.connected_components(graph))
    print("island_list: ", islands)
    island_averages = []

    for island in islands:
        island_nodes = list(island)
        island_speeds = [
            v_map[(island_nodes[i], island_nodes[j])]
            for i in range(len(island_nodes))
            for j in range(i + 1, len(island_nodes))
        ]
        turn_edges_positive = max(island_speeds) + 1
        distance_graph = nx.Graph()
        distance_graph.add_nodes_from(island_nodes)

        for i in range(len(island_nodes)):
            for j in range(i + 1, len(island_nodes)):
                source, target = island_nodes[i], island_nodes[j]
                if not graph.has_edge(source, target):
                    continue
                distance = -v_map[(source, target)] + turn_edges_positive
                distance_graph.add_edge(source, target, weight=distance)

        total_difference = 0.0
        for i in range(len(island_nodes)):
            for j in range(i + 1, len(island_nodes)):
                source, target = island_nodes[i], island_nodes[j]
                path = nx.shortest_path(
                    distance_graph,
                    source=source,
                    target=target,
                    weight="weight",
                )
                knn_distance = sum(
                    -v_map[(path[index], path[index + 1])]
                    for index in range(len(path) - 1)
                )
                original_distance = -v_map[(source, target)]
                total_difference += abs(knn_distance - original_distance) / original_distance

        average_difference = total_difference / (
            len(island_nodes) * (len(island_nodes) - 1)
        )
        island_averages.append(average_difference)

    return sum(island_averages) / len(island_averages)


def calculate_partition_reward(temp_graph, v_map):
    groups = list(nx.connected_components(temp_graph))
    group_num = len(groups)
    if group_num == 1:
        return NO_PARTITION_REWARD, groups

    groups_avg_speed = []
    for group in groups:
        if len(group) < 2:
            continue
        pairs = list(combinations(group, 2))
        sum_speed = 0.0
        for source, target in pairs:
            shortest_path = get_shortest_path(v_map, NODE_NUM, source, target)
            sum_speed += calculate_speed(shortest_path, v_map)
        groups_avg_speed.append(sum_speed / len(pairs))

    if not groups_avg_speed or min(groups_avg_speed) <= 0:
        return NO_PARTITION_REWARD, groups

    middle_term = sum(groups_avg_speed) / group_num
    reward = middle_term - ALPHA * middle_term ** 2 / min(groups_avg_speed)
    return reward, groups


def build_base_state(edges, v_map):
    edge_speeds = np.asarray([v_map[edge] for edge in edges], dtype=np.float32)
    if not NORMALIZE_STATE:
        return edge_speeds

    positive_speeds = edge_speeds[edge_speeds > 0]
    scale = float(np.max(positive_speeds)) if len(positive_speeds) > 0 else 1.0
    return edge_speeds / scale


def available_actions_from_state(state):
    return np.flatnonzero(np.asarray(state, dtype=np.float32) > 0).tolist()


def initialize_layers(layers):
    """采用示例 A3C 的正态权重和零偏置初始化。"""
    for layer in layers:
        nn.init.normal_(layer.weight, mean=0.0, std=0.1)
        nn.init.constant_(layer.bias, 0.0)


class ActorCritic(nn.Module):
    """离散动作 Actor-Critic；Actor 输出每条边的 categorical logits。"""

    def __init__(self, state_dim, action_dim, hidden_dim=A3C_HIDDEN_DIM):
        super().__init__()
        self.actor_hidden = nn.Linear(state_dim, hidden_dim)
        self.actor_output = nn.Linear(hidden_dim, action_dim)
        self.critic_hidden = nn.Linear(state_dim, hidden_dim)
        self.critic_output = nn.Linear(hidden_dim, 1)
        initialize_layers(
            [
                self.actor_hidden,
                self.actor_output,
                self.critic_hidden,
                self.critic_output,
            ]
        )

    def forward(self, states):
        actor_features = torch.tanh(self.actor_hidden(states))
        logits = self.actor_output(actor_features)
        critic_features = torch.tanh(self.critic_hidden(states))
        values = self.critic_output(critic_features).squeeze(-1)
        return logits, values

    @staticmethod
    def mask_logits(logits, available_actions):
        masked_logits = torch.full_like(logits, -1e9)
        masked_logits[..., available_actions] = logits[..., available_actions]
        return masked_logits

    @staticmethod
    def mask_batch_logits(logits, action_masks):
        masked_logits = torch.full_like(logits, -1e9)
        for row, available_actions in enumerate(action_masks):
            masked_logits[row, available_actions] = logits[row, available_actions]
        return masked_logits

    def choose_action(self, state, available_actions):
        self.eval()
        state_tensor = torch.as_tensor(state, dtype=torch.float32).unsqueeze(0)
        with torch.no_grad():
            logits, _ = self.forward(state_tensor)
            masked_logits = self.mask_logits(logits, available_actions)
            distribution = torch.distributions.Categorical(logits=masked_logits)
            action = distribution.sample()
        return int(action.item())

    def loss_func(self, states, actions, value_targets, action_masks):
        self.train()
        logits, values = self.forward(states)
        masked_logits = self.mask_batch_logits(logits, action_masks)
        distribution = torch.distributions.Categorical(logits=masked_logits)

        td_error = value_targets - values
        critic_loss = td_error.pow(2).mean()
        actor_loss = -(
            distribution.log_prob(actions) * td_error.detach()
        ).mean()
        entropy = distribution.entropy().mean()
        total_loss = (
            actor_loss
            + A3C_VALUE_COEF * critic_loss
            - A3C_ENTROPY_COEF * entropy
        )
        return total_loss


class SharedAdam(torch.optim.Adam):
    """Adam optimizer whose moment estimates are shared by A3C workers."""

    def __init__(
        self,
        params,
        lr=A3C_LR,
        betas=A3C_BETAS,
        eps=1e-8,
        weight_decay=0,
    ):
        super().__init__(
            params,
            lr=lr,
            betas=betas,
            eps=eps,
            weight_decay=weight_decay,
        )
        for group in self.param_groups:
            for parameter in group["params"]:
                state = self.state[parameter]
                state["step"] = torch.zeros((), dtype=torch.float32)
                state["exp_avg"] = torch.zeros_like(parameter.data)
                state["exp_avg_sq"] = torch.zeros_like(parameter.data)
                state["step"].share_memory_()
                state["exp_avg"].share_memory_()
                state["exp_avg_sq"].share_memory_()


def push_and_pull(
    optimizer,
    local_net,
    global_net,
    done,
    next_state,
    states,
    actions,
    rewards,
    action_masks,
):
    """计算 n-step return，将本地梯度推送到全局网络后同步本地网络。"""
    if not states:
        return None

    if done:
        bootstrap_value = 0.0
    else:
        next_state_tensor = torch.as_tensor(
            next_state,
            dtype=torch.float32,
        ).unsqueeze(0)
        with torch.no_grad():
            _, next_value = local_net(next_state_tensor)
        bootstrap_value = float(next_value.item())

    value_targets = []
    discounted_value = bootstrap_value
    for reward in reversed(rewards):
        discounted_value = reward + A3C_GAMMA * discounted_value
        value_targets.append(discounted_value)
    value_targets.reverse()

    state_tensor = torch.as_tensor(np.asarray(states), dtype=torch.float32)
    action_tensor = torch.as_tensor(actions, dtype=torch.long)
    target_tensor = torch.as_tensor(value_targets, dtype=torch.float32)

    loss = local_net.loss_func(
        state_tensor,
        action_tensor,
        target_tensor,
        action_masks,
    )

    optimizer.zero_grad()
    local_net.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(local_net.parameters(), A3C_MAX_GRAD_NORM)
    for local_parameter, global_parameter in zip(
        local_net.parameters(),
        global_net.parameters(),
    ):
        if local_parameter.grad is not None:
            global_parameter._grad = local_parameter.grad
    optimizer.step()
    local_net.load_state_dict(global_net.state_dict())
    return float(loss.item())


class A3CWorker(mp.Process):
    """独立探索 KNN 删边轨迹并异步更新共享 Actor-Critic 的 worker。"""

    def __init__(
        self,
        worker_id,
        global_net,
        optimizer,
        global_episode,
        result_queue,
        base_graph,
        edges,
        v_map,
        base_state,
        cut_edge_num,
    ):
        super().__init__(name=f"a3c-worker-{worker_id:02d}")
        self.worker_id = worker_id
        self.global_net = global_net
        self.optimizer = optimizer
        self.global_episode = global_episode
        self.result_queue = result_queue
        self.base_graph = base_graph
        self.edges = edges
        self.v_map = v_map
        self.base_state = base_state
        self.cut_edge_num = cut_edge_num
        self.local_net = ActorCritic(len(edges), len(edges))
        self.local_net.load_state_dict(global_net.state_dict())

    def reserve_episode(self):
        with self.global_episode.get_lock():
            if self.global_episode.value >= EPISODE_NUM:
                return None
            episode = self.global_episode.value + 1
            self.global_episode.value += 1
        return episode

    def run_episode(self, episode):
        state = self.base_state.copy()
        temp_graph = self.base_graph.copy()
        episode_actions = []
        episode_reward = 0.0
        best_reward = -math.inf
        best_step = None
        last_loss = None

        buffer_states = []
        buffer_actions = []
        buffer_rewards = []
        buffer_masks = []

        for cut_edge_count in range(self.cut_edge_num):
            available_actions = available_actions_from_state(state)
            if not available_actions:
                break

            action = self.local_net.choose_action(state, available_actions)
            next_state = state.copy()
            next_state[action] = 0.0

            source, target = self.edges[action]
            if temp_graph.has_edge(source, target):
                temp_graph.remove_edge(source, target)

            reward, groups = calculate_partition_reward(temp_graph, self.v_map)
            episode_reward += reward
            episode_actions.append(action)
            buffer_states.append(state.copy())
            buffer_actions.append(action)
            buffer_rewards.append(reward)
            buffer_masks.append(available_actions)

            next_available_actions = available_actions_from_state(next_state)
            done = (
                cut_edge_count + 1 >= self.cut_edge_num
                or not next_available_actions
            )

            if len(groups) > 1 and reward > best_reward:
                best_reward = reward
                best_step = cut_edge_count + 1

            if len(buffer_states) >= A3C_UPDATE_GLOBAL_ITER or done:
                last_loss = push_and_pull(
                    self.optimizer,
                    self.local_net,
                    self.global_net,
                    done,
                    next_state,
                    buffer_states,
                    buffer_actions,
                    buffer_rewards,
                    buffer_masks,
                )
                buffer_states = []
                buffer_actions = []
                buffer_rewards = []
                buffer_masks = []

            state = next_state
            if done:
                break

        return {
            "kind": "episode",
            "worker": self.name,
            "episode": episode,
            "episode_reward": episode_reward,
            "best_reward": best_reward,
            "best_step": best_step,
            "actions": episode_actions,
            "last_loss": last_loss,
        }

    def run(self):
        torch.set_num_threads(1)
        torch.manual_seed(A3C_SEED + self.worker_id + 1)
        np.random.seed(A3C_SEED + self.worker_id + 1)
        try:
            while True:
                episode = self.reserve_episode()
                if episode is None:
                    break
                result = self.run_episode(episode)
                self.result_queue.put(result)
                print(
                    f"{self.name} episode={episode}, "
                    f"episode_reward={result['episode_reward']:.6f}, "
                    f"best_reward={result['best_reward']:.6f}, "
                    f"last_loss={result['last_loss']}"
                )
        except Exception:
            self.result_queue.put(
                {
                    "kind": "error",
                    "worker": self.name,
                    "traceback": traceback.format_exc(),
                }
            )
        finally:
            self.result_queue.put({"kind": "done", "worker": self.name})


def replay_actions(base_graph, edges, actions):
    """重放一个 episode 的删边动作，并返回每一步对应的图。"""
    graph = base_graph.copy()
    graphs = []
    for action in actions:
        source, target = edges[action]
        if graph.has_edge(source, target):
            graph.remove_edge(source, target)
        graphs.append(graph.copy())
    return graphs


def run_a3c_partition(graph, v_map):
    edges = list(graph.edges())
    if not edges:
        return graph.copy(), None, None, -math.inf, [], []

    base_state = build_base_state(edges, v_map)
    cut_edge_num = round(len(edges) * CUT_EDGE_RATE)
    if cut_edge_num <= 0:
        return graph.copy(), None, None, -math.inf, [], []

    torch.manual_seed(A3C_SEED)
    np.random.seed(A3C_SEED)
    global_net = ActorCritic(len(edges), len(edges))
    global_net.share_memory()
    optimizer = SharedAdam(global_net.parameters())
    global_episode = mp.Value("i", 0)
    result_queue = mp.Queue()

    worker_count = min(A3C_WORKER_NUM, EPISODE_NUM)
    print(
        f"A3C training: workers={worker_count}, episodes={EPISODE_NUM}, "
        f"update_interval={A3C_UPDATE_GLOBAL_ITER}, edges={len(edges)}, "
        f"cut_edges_per_episode={cut_edge_num}"
    )
    workers = [
        A3CWorker(
            worker_id,
            global_net,
            optimizer,
            global_episode,
            result_queue,
            graph,
            edges,
            v_map,
            base_state,
            cut_edge_num,
        )
        for worker_id in range(worker_count)
    ]
    for worker in workers:
        worker.start()

    episode_results = []
    worker_errors = []
    finished_workers = 0
    while finished_workers < worker_count:
        message = result_queue.get()
        if message["kind"] == "episode":
            episode_results.append(message)
        elif message["kind"] == "error":
            worker_errors.append(message)
        elif message["kind"] == "done":
            finished_workers += 1

    for worker in workers:
        worker.join()
    result_queue.close()
    result_queue.join_thread()

    if worker_errors:
        details = "\n".join(
            f"{error['worker']}:\n{error['traceback']}"
            for error in worker_errors
        )
        raise RuntimeError(f"A3C worker failed:\n{details}")
    if len(episode_results) != EPISODE_NUM:
        raise RuntimeError(
            f"A3C completed {len(episode_results)} episodes, "
            f"expected {EPISODE_NUM}."
        )

    episode_results.sort(key=lambda item: item["episode"])
    best_reward_each_episode = [
        result["best_reward"]
        if result["best_reward"] != -math.inf
        else -3
        for result in episode_results
    ]
    best_result = max(episode_results, key=lambda item: item["best_reward"])

    overall_best_episode = best_result["episode"]
    overall_best_step = best_result["best_step"]
    overall_best_reward = best_result["best_reward"]
    episode_graphs = replay_actions(graph, edges, best_result["actions"])
    if overall_best_step is None:
        overall_best_graph = graph.copy()
    else:
        overall_best_graph = episode_graphs[overall_best_step - 1].copy()

    return (
        overall_best_graph,
        overall_best_episode,
        overall_best_step,
        overall_best_reward,
        episode_graphs,
        best_reward_each_episode,
    )


def choose_best_knn_graph(distance_map, v_map):
    scores = []
    min_k = max(1, math.ceil(KNN_MIN_C * math.log(NODE_NUM)))
    max_k = NODE_NUM - 1
    k_values = list(range(min_k, max_k + 1))
    evaluated_k_values = []
    result_text = ""
    if not k_values:
        raise ValueError(
            f"No valid KNN k value for NODE_NUM={NODE_NUM}, "
            f"KNN_MIN_C={KNN_MIN_C}."
        )

    print(
        f"KNN search k range: {min_k}..{max_k} "
        f"(k_min=ceil({KNN_MIN_C} * log({NODE_NUM})))"
    )
    for k in k_values:
        result_text += f"k={k}\n"
        graph = knn_graph_from_map(distance_map, k=k, node_count=NODE_NUM)
        score1 = count_islands(graph)
        score2 = distance_preservation_score(graph, v_map)
        overall_score = SCORE1_WEIGHT * score1[0] + SCORE2_WEIGHT * score2
        scores.append(overall_score)
        evaluated_k_values.append(k)
        print(
            f"k={k}: score1={score1}, score2={score2}, "
            f"overall_score={overall_score}"
        )
        result_text += (
            f"score1={score1}, score2={score2}, "
            f"overall_score={overall_score}\n"
        )

        if len(scores) > KNN_EARLY_STOP_DECLINES:
            recent_scores = scores[-(KNN_EARLY_STOP_DECLINES + 1):]
            if all(
                recent_scores[index] > recent_scores[index + 1]
                for index in range(KNN_EARLY_STOP_DECLINES)
            ):
                print(
                    f"KNN early stopped at k={k} after "
                    f"{KNN_EARLY_STOP_DECLINES} consecutive score declines."
                )
                break

    print(result_text)
    best_k = evaluated_k_values[int(np.argmax(scores))]
    return best_k, knn_graph_from_map(distance_map, k=best_k, node_count=NODE_NUM)


def KNN_and_A3C():
    start_time = time.perf_counter()
    v_map = read_xls_to_map(DATA_PATH, sheet_index=SHEET_INDEX)
    distance_map = {
        (source, target): 1 / v_map[(source, target)]
        for source in range(NODE_NUM)
        for target in range(NODE_NUM)
    }

    k, graph = choose_best_knn_graph(distance_map, v_map)
    print(f"Testing with k={k}")
    distance_preservation_score(graph, v_map)
    print("G.edges: ")
    print(graph.edges())
    plot_graph(
        graph,
        f"KNN Graph (k={k}, nodes={NODE_NUM})",
        "a3c_knn_graph.png",
    )

    (
        overall_best_graph,
        overall_best_episode,
        overall_best_step,
        overall_best_reward,
        overall_best_episode_graphs,
        best_reward_each_episode,
    ) = run_a3c_partition(graph, v_map)

    best_group_info = (
        f"best episode={overall_best_episode}, step={overall_best_step}, "
        f"reward={overall_best_reward:.6f}"
    )
    print(f"Best group info: {best_group_info}")
    plot_graph(
        overall_best_graph,
        f"KNN Graph (k={k}, nodes={NODE_NUM})\n{best_group_info}",
        "a3c_best_graph.png",
    )
    save_best_episode_graphs(
        overall_best_episode_graphs,
        overall_best_episode,
        overall_best_step,
        f"KNN A3C Graph (k={k}, nodes={NODE_NUM})",
    )
    plot_rewards(best_reward_each_episode)

    node_group = [
        sorted(component)
        for component in nx.connected_components(overall_best_graph)
    ]
    elapsed_time = time.perf_counter() - start_time
    print(f"KNN + A3C algorithm running time: {elapsed_time:.6f} seconds")
    print(f"Output directory: {OUTPUT_DIR}")
    return node_group


def KNN_and_RL():
    return KNN_and_A3C()


if __name__ == "__main__":
    mp.freeze_support()
    groups = KNN_and_A3C()
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    with open(RESULT_PATH, "w", encoding="utf-8") as result_file:
        json.dump(groups, result_file)
