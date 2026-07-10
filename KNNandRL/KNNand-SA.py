import json
import math
import os
import random
import time
from datetime import datetime

import matplotlib
import xlrd
import networkx as nx
from itertools import combinations
from dijkstra import get_shortest_path, calculate_speed

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(BASE_DIR, f"SA-{datetime.now().strftime('%m%d-%H%M')}")
RESULT_PATH = os.path.join(OUTPUT_DIR, "result.json")
SHOW_PLOTS = os.environ.get("KNN_SA_SHOW_PLOTS", "0") == "1"
SAVE_PLOTS = os.environ.get("KNN_SA_SAVE_PLOTS", "1") == "1"
SAVE_BEST_EPISODE = True

if not SHOW_PLOTS:
    matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402

# 只用于读取速度矩阵
DATA_PATH = "../data/15e_user400.xls"
SHEET_INDEX = 4

NODE_NUM = 15
ALL_CONNECTIONS_EDGE_NUM = NODE_NUM * (NODE_NUM - 1) // 2
SCORE1_LAMBDA = 1  # 指标1平衡系数

SCORE1_WEIGHT = 0.5  # 指标1对总分的权重
SCORE2_WEIGHT = -0.005  # 指标2对总分的权重
KNN_MIN_C = 1.0
KNN_EARLY_STOP_DECLINES = 3

EPISODE_NUM = 100
CUT_EDGE_RATE = 0.7  # 在RL每轮episode删除边时，删除的边数/总边数的比例
ALPHA = 0.3
NO_PARTITION_REWARD = -0.5

SA_INITIAL_TEMPERATURE = 10.0
SA_MIN_TEMPERATURE = 1e-6
SA_COOLING_RATE = 0.95
SA_MAX_ATTEMPT_MULTIPLIER = 4


def read_xls_to_map(file_path, sheet_index):
    """
    读取指定 .xls 文件，将单元格数据存入一个 map（dict）
    键： (row, col)
    值： 单元格内容
    """
    # 打开 Excel 文件
    workbook = xlrd.open_workbook(file_path)
    sheet = workbook.sheet_by_index(sheet_index)

    data_map = {}
    for r in range(sheet.nrows):
        for c in range(sheet.ncols):
            data_map[(r, c)] = sheet.cell_value(r, c)

    return data_map


def knn_graph_from_map(distance_map, k=3, node_count=30):
    G = nx.Graph()
    G.add_nodes_from(range(node_count))

    for i in range(node_count):
        # 获取当前节点 i 到其他节点的所有距离
        distances = [(j, distance_map[(i, j)])
                     for j in range(node_count) if i != j]
        # 过滤非法或未定义的距离
        distances = [(j, d) for j, d in distances if d >= 0]
        # 按距离升序排序
        distances.sort(key=lambda x: x[1])
        # 取前 k 个最近邻
        nearest_neighbors = distances[:k]

        for j, d in nearest_neighbors:
            # 添加无向边（networkx 会自动处理重复边）
            G.add_edge(i, j, weight=d)

    return G


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
        graph, pos,
        node_size=200,
        node_color="skyblue",
        with_labels=True,
        edge_color="gray"
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
    plt.plot(best_reward_each_episode, marker='o')
    plt.xlabel("Episode")
    plt.ylabel("Best Reward")
    plt.title("Best Reward per Episode")
    plt.grid(True)
    plt.tight_layout()
    finish_plot("sa_best_reward.png")


def count_islands(G: nx.Graph) -> tuple[float, int, int]:
    """指标1：获取连通分量"""
    components = nx.connected_components(G)
    island_list = list(components)

    # 计算孤岛数量
    island_count = len(island_list)  # Ck

    connScoreK = 1 / island_count - \
                 SCORE1_LAMBDA * G.number_of_edges() / ALL_CONNECTIONS_EDGE_NUM

    return connScoreK, island_count, G.number_of_edges()


def distance_preservation_score(G: nx.Graph, v_map: dict):
    """指标2：距离保持"""
    components = nx.connected_components(G)
    island_list = list(components)
    print("island_list: ", island_list)
    island_avg_list = []
    # 遍历孤岛，取每个孤岛的平均 Dk
    for island in island_list:
        island_nodes = list(island)
        # 取速度负数等比距离作为 d，转换到 nx.Graph 中快速找到最短路径
        G_d = nx.Graph()
        # 获取孤岛的所有节点对的速度值
        island_v = [v_map[(u, v)] for i in range(len(island_nodes)) for j in range(
            i + 1, len(island_nodes)) for u, v in [(island_nodes[i], island_nodes[j])]]
        # 取负后为了使用最短路径算法，全部加上一个正数以避免负权边
        turn_graph_edges_to_positive = max(island_v) + 1
        for i in range(len(island_nodes)):
            for j in range(i + 1, len(island_nodes)):
                if not G.has_edge(island_nodes[i], island_nodes[j]):
                    # 如果孤岛内节点间没有边，则跳过
                    continue
                u, v = island_nodes[i], island_nodes[j]
                d = -v_map[(u, v)] + turn_graph_edges_to_positive
                G_d.add_edge(u, v, weight=d)
        sumDk = 0
        for i in range(len(island_nodes)):
            for j in range(i + 1, len(island_nodes)):
                u, v = island_nodes[i], island_nodes[j]
                path = nx.shortest_path(
                    G_d, source=u, target=v, weight='weight')
                # 最短路径取负
                dijk = sum(-v_map[(path[n], path[n + 1])]
                           for n in range(len(path) - 1))
                dOrig = -v_map[(u, v)]
                sumDk += abs(dijk - dOrig) / dOrig
        sumAvgDk = sumDk / (len(island_nodes) * (len(island_nodes) - 1))
        island_avg_list.append(sumAvgDk)
    # print("island_avg_list: ",island_avg_list)
    overall_avgDk = sum(island_avg_list) / len(island_avg_list)
    return overall_avgDk


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
    reward = mid_term - ALPHA * mid_term ** 2 / min(groups_avg_speed)
    return reward, groups


def accept_sa_move(candidate_reward, current_reward, temperature):
    if candidate_reward >= current_reward:
        return True
    probability = math.exp((candidate_reward - current_reward) / max(temperature, SA_MIN_TEMPERATURE))
    return random.random() < probability


def KNN_and_SA():
    start_time = time.perf_counter()

    file_path = DATA_PATH  # 你的文件路径
    v_map = read_xls_to_map(file_path, sheet_index=SHEET_INDEX)  # ! 这里直接读取了速度数据，没有x100
    distance_map = {(i, j): 1 / v_map[(i, j)] for i in range(NODE_NUM)
                    for j in range(NODE_NUM)}  # 这个 map 仅做速度的相反排序使用，速度越大其值越小，不是严格的距离

    scores = []
    min_k = max(1, math.ceil(KNN_MIN_C * math.log(NODE_NUM)))
    max_k = NODE_NUM - 1
    k_values = list(range(min_k, max_k + 1))
    evaluated_k_values = []
    resultStr = ""
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
        resultStr += "k=" + str(k) + "\n"
        graph = knn_graph_from_map(distance_map, k=k, node_count=NODE_NUM)
        score1 = count_islands(graph)
        score2 = distance_preservation_score(graph, v_map)
        overall_score = SCORE1_WEIGHT * score1[0] + SCORE2_WEIGHT * score2
        scores.append(overall_score)
        evaluated_k_values.append(k)
        print(f"k={k}: score1={score1}, score2={score2}, "
              f"overall_score={overall_score}")
        resultStr += f"score1={score1}, score2={score2}, " + \
                     f"overall_score={overall_score}\n"
        if len(scores) > KNN_EARLY_STOP_DECLINES:
            recent_scores = scores[-(KNN_EARLY_STOP_DECLINES + 1):]
            if all(
                recent_scores[i] > recent_scores[i + 1]
                for i in range(KNN_EARLY_STOP_DECLINES)
            ):
                print(
                    f"KNN early stopped at k={k} after "
                    f"{KNN_EARLY_STOP_DECLINES} consecutive score declines."
                )
                break
    print(resultStr)

    k = evaluated_k_values[scores.index(max(scores))]
    print(f"Testing with k={k}")
    graph = knn_graph_from_map(distance_map, k=k, node_count=NODE_NUM)
    distance_preservation_score(graph, v_map)
    print("G.edges: ")
    print(graph.edges())
    # 画一个拓扑图
    plot_graph(graph, f"KNN Graph (k={k}, nodes={NODE_NUM})", "sa_knn_graph.png")

    # SA部分
    edges = list(graph.edges())
    cut_edge_num = round(len(edges) * CUT_EDGE_RATE)
    max_attempts_per_episode = max(cut_edge_num, cut_edge_num * SA_MAX_ATTEMPT_MULTIPLIER)

    overall_best_reward = -math.inf
    overall_best_graph = graph.copy()
    overall_best_episode = None
    overall_best_step = None
    overall_best_episode_graphs = []
    best_reward_each_episode = []

    for episode in range(EPISODE_NUM):
        best_reward_in_this_episode = -math.inf
        best_graph_in_this_episode = graph.copy()
        best_step_in_this_episode = None
        episode_graphs = []
        temp_graph = graph.copy()
        remaining_actions = list(range(len(edges)))
        current_reward, current_groups = calculate_partition_reward(temp_graph, v_map)
        temperature = SA_INITIAL_TEMPERATURE
        accepted_count = 0
        attempt_count = 0

        while accepted_count < cut_edge_num and remaining_actions and attempt_count < max_attempts_per_episode:
            attempt_count += 1
            action = random.choice(remaining_actions)
            candidate_graph = temp_graph.copy()
            u, v = edges[action]
            if candidate_graph.has_edge(u, v):
                candidate_graph.remove_edge(u, v)

            candidate_reward, candidate_groups = calculate_partition_reward(candidate_graph, v_map)
            accepted = accept_sa_move(candidate_reward, current_reward, temperature)
            if accepted:
                temp_graph = candidate_graph
                current_reward = candidate_reward
                current_groups = candidate_groups
                remaining_actions.remove(action)
                accepted_count += 1
                episode_graphs.append(temp_graph.copy())

                if len(current_groups) > 1 and current_reward > best_reward_in_this_episode:
                    best_reward_in_this_episode = current_reward
                    best_graph_in_this_episode = temp_graph.copy()
                    best_step_in_this_episode = accepted_count

            print(str(candidate_reward), end="\t")
            temperature = max(SA_MIN_TEMPERATURE, temperature * SA_COOLING_RATE)

        print(f"\nbest reward: {best_reward_in_this_episode}")
        if best_reward_in_this_episode != -math.inf:
            best_reward_each_episode.append(best_reward_in_this_episode)
        else:
            best_reward_each_episode.append(-3)
        # 更新全局最优图
        if best_reward_in_this_episode > overall_best_reward:
            overall_best_reward = best_reward_in_this_episode
            overall_best_graph = best_graph_in_this_episode.copy()
            overall_best_episode = episode + 1
            overall_best_step = best_step_in_this_episode
            overall_best_episode_graphs = [step_graph.copy() for step_graph in episode_graphs]
        print(f"current overall best reward: {overall_best_reward}")

    # 画最终episode的best reward对应的拓扑图
    best_group_info = (
        f"best episode={overall_best_episode}, step={overall_best_step}, "
        f"reward={overall_best_reward:.6f}"
    )
    print(f"Best group info: {best_group_info}")
    plot_graph(
        overall_best_graph,
        f"KNN Graph (k={k}, nodes={NODE_NUM})\n{best_group_info}",
        "sa_best_graph.png",
    )
    save_best_episode_graphs(
        overall_best_episode_graphs,
        overall_best_episode,
        overall_best_step,
        f"KNN SA Graph (k={k}, nodes={NODE_NUM})",
    )
    # 画best reward随episode的变化趋势
    plot_rewards(best_reward_each_episode)
    components = nx.connected_components(overall_best_graph)
    node_group = [list(component) for component in components]

    elapsed_time = time.perf_counter() - start_time
    print(f"KNN + SA algorithm running time: {elapsed_time:.6f} seconds")
    print(f"Output directory: {OUTPUT_DIR}")
    return node_group


def KNN_and_RL():
    return KNN_and_SA()


if __name__ == '__main__':
    node_group = KNN_and_SA()
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    with open(RESULT_PATH, "w", encoding="utf-8") as f:
        json.dump(node_group, f)
