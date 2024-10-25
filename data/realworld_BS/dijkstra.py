import heapq


def dijkstra(graph: dict, start: int, end: int) -> list:
    """
    使用 Dijkstra 算法计算从 start 到 end 的最短路径。
    :param graph: 图的邻接矩阵表示
    :param start: 起始节点
    :param end: 目标节点
    :return: 最短路径上的节点序列
    """
    # 初始化距离字典和前驱节点字典
    distances = {node: float('inf') for node in graph}
    previous_nodes = {node: None for node in graph}
    distances[start] = 0
    queue = []
    heapq.heappush(queue, (0, start))

    while queue:
        current_distance, current_node = heapq.heappop(queue)

        if current_distance > distances[current_node]:
            continue

        for neighbor, weight in graph[current_node].items():
            distance = current_distance + weight
            if distance < distances[neighbor]:
                distances[neighbor] = distance
                previous_nodes[neighbor] = current_node
                heapq.heappush(queue, (distance, neighbor))

    # 构建最短路径
    path = []
    node = end
    while node is not None:
        path.append(node)
        node = previous_nodes[node]
    path = path[::-1]

    if len(path) <= 1:
        raise ValueError("不存在可能的路径，检查是否有节点孤岛？路径字典中是否只提供了单向的有向边（只有(i, j)没有(j, i)）？")
    return path


def get_shortest_path(speeds: dict, edgenode_count: int,
                      start: int, end: int) -> list:
    """
    预处理，删去不互联的边缘节点(值为0)，把传输速率的倒数作为边的权重(1/rate)
    """
    # 剔除值为 0 的项
    speeds_remove_zero = {key: value for key,
    value in speeds.items() if value != 0}

    # 将传输速率的倒数作为边的权重
    speeds_cost = {(i, j): 1 / rate for (i, j),
    rate in speeds_remove_zero.items()}

    # 将 cost 转换为邻接列表形式
    adjacency_list = {i: {} for i in range(edgenode_count)}
    for (i, j), cost in speeds_cost.items():
        adjacency_list[i][j] = cost

    # 计算最短路径
    return dijkstra(adjacency_list, start, end)


def calculate_speed(shortest_path: list, channel: dict) -> float:
    """
    根据最短路径和传输速率表，计算该链路的传输速率“倒数和”，
    再倒数，即为该链路的传输速率
    """
    speed = 0
    for i in range(len(shortest_path) - 1):
        speed += 1 / channel[(shortest_path[i], shortest_path[i + 1])]
    return 1.0 / speed


if __name__ == "__main__":
    # 测试数据
    channel = {
        (0, 1): 10,
        (0, 2): 5,
        (0, 3): 0,  # 不可达的情况
        (1, 0): 10,
        (1, 2): 8,
        (1, 3): 6,
        (2, 0): 5,
        (2, 1): 8,
        (2, 3): 7,
        (3, 0): 0,  # 不可达的情况
        (3, 1): 6,
        (3, 2): 7,
    }
    edgenode_count = 4

    # 计算从节点 0 到节点 3 的最短路径
    shortest_path = get_shortest_path(channel, edgenode_count, 0, 3)
    print("最短路径:", shortest_path)

    speed = calculate_speed(shortest_path, channel)
    print("传输速率:", speed)
