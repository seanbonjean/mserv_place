import json
import math
import os
import time
from datetime import datetime

import matplotlib
import xlrd
import networkx as nx
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from itertools import combinations
from dijkstra import get_shortest_path, calculate_speed

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(BASE_DIR, f"PPO-{datetime.now().strftime('%m%d-%H%M')}")
RESULT_PATH = os.path.join(OUTPUT_DIR, "result.json")
SHOW_PLOTS = os.environ.get("KNN_PPO_SHOW_PLOTS", "0") == "1"
SAVE_PLOTS = os.environ.get("KNN_PPO_SAVE_PLOTS", "1") == "1"
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

PPO_LR = 0.0003
PPO_GAMMA = 0.9
PPO_CLIP_EPS = 0.2
PPO_UPDATE_EPOCHS = 4
PPO_ENTROPY_COEF = 0.01
PPO_VALUE_COEF = 0.5
PPO_HIDDEN_DIM = 64
NORMALIZE_STATE = True


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
    finish_plot("ppo_best_reward.png")


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


def build_base_state(edges, v_map):
    edge_speeds = np.array([v_map[e] for e in edges], dtype=np.float32)
    if not NORMALIZE_STATE:
        return edge_speeds.tolist()

    positive_speeds = edge_speeds[edge_speeds > 0]
    scale = float(np.max(positive_speeds)) if len(positive_speeds) > 0 else 1.0
    return (edge_speeds / scale).tolist()


def available_actions_from_state(state):
    return np.flatnonzero(np.asarray(state, dtype=np.float32) > 0).tolist()


class ActorCritic(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=PPO_HIDDEN_DIM):
        super().__init__()
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.actor = nn.Linear(hidden_dim, action_dim)
        self.critic = nn.Linear(hidden_dim, 1)

    def forward(self, states):
        x = F.relu(self.fc1(states))
        x = F.relu(self.fc2(x))
        return self.actor(x), self.critic(x).squeeze(-1)


class PPOAgent:
    def __init__(self, state_dim, action_dim):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.policy = ActorCritic(state_dim, action_dim).to(self.device)
        self.optimizer = torch.optim.Adam(self.policy.parameters(), lr=PPO_LR)

    def choose_action(self, state, available_actions):
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        with torch.no_grad():
            logits, value = self.policy(state_tensor)
            logits = self.mask_logits(logits, available_actions)
            dist = torch.distributions.Categorical(logits=logits)
            action = dist.sample()
            log_prob = dist.log_prob(action)
        return int(action.item()), float(log_prob.item()), float(value.item())

    def update(self, trajectory):
        if not trajectory:
            return None

        states = torch.FloatTensor([item["state"] for item in trajectory]).to(self.device)
        actions = torch.LongTensor([item["action"] for item in trajectory]).to(self.device)
        old_log_probs = torch.FloatTensor([item["log_prob"] for item in trajectory]).to(self.device)
        rewards = [item["reward"] for item in trajectory]
        masks = [item["available_actions"] for item in trajectory]

        returns = []
        discounted_return = 0.0
        for reward in reversed(rewards):
            discounted_return = reward + PPO_GAMMA * discounted_return
            returns.insert(0, discounted_return)
        returns = torch.FloatTensor(returns).to(self.device)

        last_loss = None
        for _ in range(PPO_UPDATE_EPOCHS):
            logits, values = self.policy(states)
            masked_logits = self.mask_batch_logits(logits, masks)
            dist = torch.distributions.Categorical(logits=masked_logits)
            log_probs = dist.log_prob(actions)
            entropy = dist.entropy().mean()

            advantages = returns - values.detach()
            if advantages.numel() > 1:
                advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

            ratio = torch.exp(log_probs - old_log_probs)
            surrogate1 = ratio * advantages
            surrogate2 = torch.clamp(ratio, 1 - PPO_CLIP_EPS, 1 + PPO_CLIP_EPS) * advantages
            actor_loss = -torch.min(surrogate1, surrogate2).mean()
            critic_loss = F.mse_loss(values, returns)
            loss = actor_loss + PPO_VALUE_COEF * critic_loss - PPO_ENTROPY_COEF * entropy

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            last_loss = float(loss.item())

        return last_loss

    def mask_logits(self, logits, available_actions):
        masked_logits = torch.full_like(logits, -1e9)
        masked_logits[:, available_actions] = logits[:, available_actions]
        return masked_logits

    def mask_batch_logits(self, logits, masks):
        masked_logits = torch.full_like(logits, -1e9)
        for row, available_actions in enumerate(masks):
            masked_logits[row, available_actions] = logits[row, available_actions]
        return masked_logits


def KNN_and_PPO():
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
    plot_graph(graph, f"KNN Graph (k={k}, nodes={NODE_NUM})", "ppo_knn_graph.png")

    # PPO部分
    edges = list(graph.edges())
    agent = PPOAgent(state_dim=len(edges), action_dim=len(edges))
    base_state = build_base_state(edges, v_map)
    cut_edge_num = round(len(edges) * CUT_EDGE_RATE)

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
        trajectory = []
        state = base_state.copy()
        temp_graph = graph.copy()
        for cut_edge_count in range(cut_edge_num):
            available_actions = available_actions_from_state(state)
            if not available_actions:
                break

            action, log_prob, _ = agent.choose_action(state, available_actions)
            next_state = state.copy()
            next_state[action] = 0

            u, v = edges[action]
            if temp_graph.has_edge(u, v):
                temp_graph.remove_edge(u, v)
            episode_graphs.append(temp_graph.copy())

            reward, groups = calculate_partition_reward(temp_graph, v_map)
            print(str(reward), end="\t")

            trajectory.append({
                "state": state.copy(),
                "action": action,
                "log_prob": log_prob,
                "reward": reward,
                "available_actions": available_actions,
            })

            state = next_state
            if len(groups) > 1 and reward > best_reward_in_this_episode:
                best_reward_in_this_episode = reward
                best_graph_in_this_episode = temp_graph.copy()
                best_step_in_this_episode = cut_edge_count + 1

        loss = agent.update(trajectory)
        print(f"\nbest reward: {best_reward_in_this_episode}")
        if loss is not None:
            print(f"last PPO loss: {loss}")
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
        "ppo_best_graph.png",
    )
    save_best_episode_graphs(
        overall_best_episode_graphs,
        overall_best_episode,
        overall_best_step,
        f"KNN PPO Graph (k={k}, nodes={NODE_NUM})",
    )
    # 画best reward随episode的变化趋势
    plot_rewards(best_reward_each_episode)
    components = nx.connected_components(overall_best_graph)
    node_group = [list(component) for component in components]

    elapsed_time = time.perf_counter() - start_time
    print(f"KNN + PPO algorithm running time: {elapsed_time:.6f} seconds")
    print(f"Output directory: {OUTPUT_DIR}")
    return node_group


def KNN_and_RL():
    return KNN_and_PPO()


if __name__ == '__main__':
    node_group = KNN_and_PPO()
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    with open(RESULT_PATH, "w", encoding="utf-8") as f:
        json.dump(node_group, f)
