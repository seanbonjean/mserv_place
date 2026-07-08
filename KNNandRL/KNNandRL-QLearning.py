import json
import math
import os
import time

import matplotlib
import xlrd
import networkx as nx
from QLearning import QLearningTable
from itertools import combinations
from dijkstra import get_shortest_path, calculate_speed

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
SHOW_PLOTS = os.environ.get("KNN_QLEARNING_SHOW_PLOTS", "0") == "1"
SAVE_PLOTS = os.environ.get("KNN_QLEARNING_SAVE_PLOTS", "1") == "1"
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

EPISODE_NUM = 100
Q_TABLE_ENTRY_PRINT_TIMES = 10
CUT_EDGE_RATE = 0.7  # 在RL每轮episode删除边时，删除的边数/总边数的比例
ALPHA = 0.3


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
        plt.savefig(os.path.join(BASE_DIR, file_name), dpi=200, bbox_inches="tight")
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
    os.makedirs(os.path.join(BASE_DIR, folder_name), exist_ok=True)
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
    finish_plot("q_learning_best_reward.png")


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


def KNN_and_RL():
    start_time = time.perf_counter()

    file_path = DATA_PATH  # 你的文件路径
    v_map = read_xls_to_map(file_path, sheet_index=SHEET_INDEX)  # ! 这里直接读取了速度数据，没有x100
    distance_map = {(i, j): 1 / v_map[(i, j)] for i in range(NODE_NUM)
                    for j in range(NODE_NUM)}  # 这个 map 仅做速度的相反排序使用，速度越大其值越小，不是严格的距离

    scores = []
    resultStr = ""
    for k in range(1, 15):
        resultStr += "k=" + str(k) + "\n"
        graph = knn_graph_from_map(distance_map, k=k, node_count=NODE_NUM)
        score1 = count_islands(graph)
        score2 = distance_preservation_score(graph, v_map)
        scores.append(SCORE1_WEIGHT * score1[0] + SCORE2_WEIGHT * score2)
        print(f"k={k}: score1={score1}, score2={score2}, "
              f"overall_score={scores[k - 1]}")
        resultStr += f"score1={score1}, score2={score2}, " + \
                     f"overall_score={scores[k - 1]}\n"
    print(resultStr)

    k = scores.index(max(scores)) + 1
    print(f"Testing with k={k}")
    graph = knn_graph_from_map(distance_map, k=k, node_count=NODE_NUM)
    distance_preservation_score(graph, v_map)
    print("G.edges: ")
    print(graph.edges())
    # 画一个拓扑图
    plot_graph(graph, f"KNN Graph (k={k}, nodes={NODE_NUM})", "q_learning_knn_graph.png")

    # RL部分
    edges = list(graph.edges())
    RL = QLearningTable(actions=list(range(len(edges))))
    base_state = list([v_map[e] for e in edges])
    cut_edge_num = round(len(edges) * CUT_EDGE_RATE)

    overall_best_reward = -math.inf
    overall_best_graph = None
    overall_best_episode = None
    overall_best_step = None
    overall_best_episode_graphs = []
    best_reward_each_episode = []
    q_table_print_times = max(1, Q_TABLE_ENTRY_PRINT_TIMES)
    q_table_print_episodes = {
        max(1, round(EPISODE_NUM * i / q_table_print_times))
        for i in range(1, q_table_print_times + 1)
    }

    for episode in range(EPISODE_NUM):
        best_reward_in_this_episode = -math.inf
        best_graph_in_this_episode = None
        best_step_in_this_episode = None
        episode_graphs = []
        state = base_state.copy()
        temp_graph = graph.copy()  # 用于计算reward的临时图
        for cut_edge_count in range(cut_edge_num):
            action = RL.choose_action(str(state))

            # 获取next_state
            next_state = state.copy()
            # 执行动作（删除边）
            next_state[action] = 0

            # 计算reward
            u, v = edges[action]
            if temp_graph.has_edge(u, v):
                temp_graph.remove_edge(u, v)
            episode_graphs.append(temp_graph.copy())
            groups = list(nx.connected_components(temp_graph))
            group_num = len(groups)  # 组数（连通分量的个数）

            # ! 如果没分出组，不仅reward为0，还更耗时计算，因此直接跳过
            if group_num == 1:
                RL.learn(str(state), action, -0.5, str(next_state))  # ! 还多给了点惩罚
                state = next_state
                continue
            # groups_node_num = [len(group) for group in groups]  # 各组内节点数
            # 计算各组内平均速度
            # groups_avg_speed = [sum(v_map[(u, v)] for u, v in combinations(group, 2) if (u, v) in edges) /
            #                     len(group) for group in groups]
            groups_avg_speed = list()
            for group in groups:
                if len(group) < 2:
                    continue
                pairs = [(u, v) for u, v in combinations(group, 2)]
                sum_speed = 0
                for u, v in pairs:
                    shortest_path = get_shortest_path(v_map, NODE_NUM, u, v)
                    sum_speed += calculate_speed(shortest_path, v_map)
                groups_avg_speed.append(sum_speed / len(pairs))
            # 先不带candidate node算
            mid_term = sum(groups_avg_speed) / group_num
            reward = mid_term - ALPHA * mid_term ** 2 / min(groups_avg_speed)
            print(str(reward), end="\t")

            # 学习
            RL.learn(str(state), action, reward, str(next_state))
            state = next_state
            # 更新本episode最优图
            if reward > best_reward_in_this_episode:
                best_reward_in_this_episode = reward
                best_graph_in_this_episode = temp_graph.copy()
                best_step_in_this_episode = cut_edge_count + 1
        # 显示结果：该轮episode最好的reward
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
        if episode + 1 in q_table_print_episodes:
            print(f"episode {episode + 1}/{EPISODE_NUM}, Q-table entries: {len(RL.q_table)}")

    # 画最终episode的best reward对应的拓扑图
    best_group_info = (
        f"best episode={overall_best_episode}, step={overall_best_step}, "
        f"reward={overall_best_reward:.6f}"
    )
    print(f"Best group info: {best_group_info}")
    plot_graph(
        overall_best_graph,
        f"KNN Graph (k={k}, nodes={NODE_NUM})\n{best_group_info}",
        "q_learning_best_graph.png",
    )
    save_best_episode_graphs(
        overall_best_episode_graphs,
        overall_best_episode,
        overall_best_step,
        f"KNN Q-Learning Graph (k={k}, nodes={NODE_NUM})",
    )
    # 画best reward随episode的变化趋势
    plot_rewards(best_reward_each_episode)
    components = nx.connected_components(overall_best_graph)
    node_group = [list(component) for component in components]

    elapsed_time = time.perf_counter() - start_time
    print(f"KNN + Q-Learning algorithm running time: {elapsed_time:.6f} seconds")
    return node_group


if __name__ == '__main__':
    node_group = KNN_and_RL()
    f = open("result.json", "w")
    json.dump(node_group, f)
