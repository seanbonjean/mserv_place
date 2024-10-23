import random


def random_mserv_place(edge_nodes: list, mservs: list) -> None:
    """
    random的解，每个微服务随机放在某个边缘节点上
    """
    for mserv in mservs:
        is_placed = False
        fail_count = 0
        while not is_placed:  # 循环直到遇到有边缘节点能放下
            rand_node = random.randint(0, len(edge_nodes) - 1)
            is_placed = edge_nodes[rand_node].place_mserv(mserv)
            fail_count += 1
            # 如果每个边缘节点都已经满了，防止死循环
            if fail_count > len(edge_nodes) * 2:
                for edge_node in edge_nodes:
                    if edge_node.place_mserv(mserv):
                        return
                raise Exception("没有足够的边缘节点来放置微服务")


def baseline_mserv_place(edge_nodes: list, mservs: list, users: list, channel: dict) -> None:
    """
    基线解
    统计所有用户需求的微服务频数，把只有某一用户使用到的微服务（频数=1）和多个用户都使用到的微服务（频数>1），分成两类
    频数=1的微服务，直接放在该用户所在的边缘节点，如果放不下就放在最近放得下的边缘节点
    频数>1的微服务，根据频数决定各自放置的总数，范围[1~频数]，然后进行greedy放置，依照节点算力最大的方式greedy
    """
    mserv_frequency = [0] * len(mservs)  # 存储各微服务出现的频数
    mserv_lastseen_user = {}  # 对应最后一次发现使用该微服务的用户，方便为频数=1的微服务指定放置位置，对频数>1的微服务来讲这个数据没有用处
    # 统计微服务频数
    for user in users:
        for mserv_num in user.mserv_dependency:
            mserv_frequency[mserv_num] += 1
            mserv_lastseen_user[mserv_num] = user
    # 根据频数分类
    singleuser_mservs = []
    multiuser_mservs = []
    for mserv_num, mserv_freq in enumerate(mserv_frequency):
        if mserv_freq == 0:
            print(f"注意：微服务{mserv_num}未被任何用户使用！")
        elif mserv_freq == 1:
            singleuser_mservs.append(
                (mserv_num, mservs[mserv_num], mserv_lastseen_user[mserv_num]))  # 记录微服务序号、微服务对象、对应的唯一用户对象组成的元组
        elif mserv_freq > 1:
            multiuser_mservs.append((mserv_num, mservs[mserv_num], mserv_freq))  # 记录微服务序号、微服务对象、对应的频数组成的元组
        else:
            raise Exception("微服务频数统计错误")
    # 频数=1的微服务放置
    for mserv_num, mserv, user in singleuser_mservs:
        if not edge_nodes[user.serv_node].place_mserv(mserv):  # 如果用户所在边缘节点放不下，就放在最近放得下的边缘节点
            alternative_nodes = sorted(enumerate(edge_nodes),
                                       key=lambda x: channel[(user.serv_node, x[0])],
                                       reverse=True)  # 按照与用户所在边缘节点的传输速率进行排序
            for node_num, node in alternative_nodes:
                if user.serv_node != node_num:  # 不重复往用户所在边缘节点放
                    if node.place_mserv(mserv):
                        break
            else:
                raise Exception("没有足够的边缘节点来放置微服务")
    # 频数>1的微服务放置
    sorted_nodes = sorted(enumerate(edge_nodes), key=lambda x: x[1].computing_power, reverse=True)  # 按照算力最大的方式greedy
    for mserv_num, mserv, mserv_freq in multiuser_mservs:
        place_count = (mserv_freq - 2) // 3 + 1  # 根据频数分段决定放置数量
        # greedy放置
        node_pointer = 0  # 指向欲放置的边缘节点，避免重复检测节点是否放得下（多个同样的微服务，上一个在前几个节点已经放不下了，这个肯定也放不下，不用重复检测）
        while place_count > 0:
            node_num, node = sorted_nodes[node_pointer]
            if node.place_mserv(mserv):
                place_count -= 1
            else:
                node_pointer += 1
            if node_pointer >= len(edge_nodes):
                raise Exception("没有足够的边缘节点来放置微服务")
