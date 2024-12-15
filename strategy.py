import random
from placed_functions import count_mserv_user
from utils import *
from values import CONSTANTS
from copy import deepcopy


def get_mserv_distri(edge_nodes: list) -> dict:
    """
    统计各微服务的分布情况，即：某个微服务都有哪些节点上有
    return: dict，以微服务序号为键，以对应边缘节点list为值
    """
    mserv_distribution = {}
    for node in edge_nodes:
        for mserv in node.placed_mservs:
            mserv_distribution.setdefault(mserv.num, [])
            mserv_distribution[mserv.num].append(node)
    return mserv_distribution


def random_mserv_place(edge_nodes: list, mservs: list) -> None:
    """
    random的解，每个微服务随机放在某个边缘节点上
    """
    # for mserv in mservs:
    #     is_placed = False
    #     fail_count = 0
    #     while not is_placed:  # 循环直到遇到有边缘节点能放下
    #         rand_node = random.randint(0, len(edge_nodes) - 1)
    #         is_placed = edge_nodes[rand_node].place_mserv(mserv)
    #         fail_count += 1
    #         # 如果每个边缘节点都已经满了，防止死循环
    #         if fail_count > len(edge_nodes) * 2:
    #             for edge_node in edge_nodes:
    #                 if edge_node.place_mserv(mserv):
    #                     return
    #             raise Exception("没有足够的边缘节点来放置微服务")
    # 思路：一轮轮放置微服务，每次放置所有的微服务一个，若随机到的节点多次放不下，则遍历找到能放的点。如果全部都放不下或放完则踢出循环
    avg_cost=CONSTANTS.MAX_DEPLOY_COST//len(mservs)
    copy_mserv_list=deepcopy(mservs)
    mserv_place_num_list=[0]*len(mservs)
    while copy_mserv_list:
        for mserv in copy_mserv_list:
            is_placed = False
            fail_count = 0
            while not is_placed:
                rand_node = random.randint(0, len(edge_nodes) - 1)
                is_placed = edge_nodes[rand_node].place_mserv(mserv)
                if is_placed:
                    mserv_place_num_list[mserv.num]+=1
                    break
                fail_count += 1
                if fail_count > len(edge_nodes):
                    for edge_node in edge_nodes:
                        if edge_node.place_mserv(mserv):
                            mserv_place_num_list[mserv.num] += 1
                            break
                    else:
                        copy_mserv_list.remove(mserv)
                        break


def random_task_routing(edge_nodes: list, mservs: list, users: list, channel: dict) -> None:
    mserv_distri = get_mserv_distri(edge_nodes)  # 统计各微服务的分布节点
    # 随机寻找路径
    for user in users:
        # 遍历所有用户请求
        for req_mserv_num in user.mserv_dependency:
            chosen_node = random.choice(mserv_distri[req_mserv_num])
            # 检查
            for mserv in chosen_node.placed_mservs:
                if mserv.num == req_mserv_num:
                    break
            else:
                raise Exception("程序有误，在mserv_distribution[m]中选取的节点，没有找到对应的微服务m")
            user.routing_route.append(chosen_node)
        user.calc_makespan(mservs, channel)
        user.print_makespan()


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
            singleuser_mservs.append((mservs[mserv_num], mserv_lastseen_user[mserv_num]))  # 记录微服务对象和对应的唯一用户对象组成的元组
        elif mserv_freq > 1:
            multiuser_mservs.append((mservs[mserv_num], mserv_freq))  # 记录微服务对象和对应的频数组成的元组
        else:
            raise Exception("微服务频数统计错误")
    # 频数=1的微服务放置
    for mserv, user in singleuser_mservs:
        if not edge_nodes[user.serv_node].place_mserv(mserv):  # 如果用户所在边缘节点放不下，就放在最近放得下的边缘节点
            alternative_nodes = sorted(edge_nodes,
                                       key=lambda x: channel[(user.serv_node, x.num)],
                                       reverse=True)  # 按照与用户所在边缘节点的传输速率进行排序
            for node in alternative_nodes:
                if user.serv_node != node.num:  # 不重复往用户所在边缘节点放
                    if node.place_mserv(mserv):
                        break
            else:
                raise Exception("没有足够的边缘节点来放置微服务")
    # 频数>1的微服务放置
    sorted_nodes = sorted(edge_nodes, key=lambda x: x.computing_power, reverse=True)  # 按照算力最大的方式greedy
    # 首先每种微服务都至少放置1个
    for mserv, mserv_freq in multiuser_mservs:
        for node in sorted_nodes:
            if node.place_mserv(mserv):
                break
    # 再根据剩余cost裕度尽量多放
    mserv_user_count = count_mserv_user(mservs, users)
    for mserv, mserv_freq in multiuser_mservs:
        # ----------------------- 在这里修改减去的数量 -----------------------
        place_count = mserv_user_count[mserv][0] - 1  # 根据频数决定放置数量
        # greedy放置
        for node in sorted_nodes:
            current_cost = sum(map(lambda x: sum(map(lambda y: y.place_cost, x.placed_mservs)), edge_nodes))
            if CONSTANTS.MAX_DEPLOY_COST - current_cost >= mserv.place_cost:
                if node.place_mserv(mserv):
                    place_count -= 1
                if place_count == 0:
                    break
            else:
                break
        else:
            raise Exception("没有足够的边缘节点来放置微服务")


def baseline_task_routing(edge_nodes: list, mservs: list, users: list, channel: dict) -> None:
    """
    基线解，寻找距离最近的微服务
    """
    mserv_distri = get_mserv_distri(edge_nodes)  # 统计各微服务的分布节点
    # 寻找距离最近
    for user in users:
        prev_node_num = user.serv_node  # 记录上一个节点编号，即数据当前停留在的节点
        # 遍历所有用户请求
        for req_mserv_num in user.mserv_dependency:
            is_routed = False
            # 先检查上个节点本地有没有这个微服务
            for mserv in edge_nodes[prev_node_num].placed_mservs:
                if mserv.num == req_mserv_num:
                    user.routing_route.append(edge_nodes[prev_node_num])
                    is_routed = True
                    break
            if is_routed:
                continue
            # 如果上个节点本地没有，则寻找距离最近的微服务所在节点
            chosen_node = max(mserv_distri[req_mserv_num], key=lambda x: channel[(prev_node_num, x.num)])
            # 检查
            for mserv in chosen_node.placed_mservs:
                if mserv.num == req_mserv_num:
                    break
            else:
                raise Exception("程序有误，在mserv_distribution[m]中选取的节点，没有找到对应的微服务m")
            user.routing_route.append(chosen_node)
            prev_node_num = chosen_node.num
        user.calc_makespan(mservs, channel)
        user.print_makespan()
