import math
import time
from gurobipy import *
# from multiprocessing import Pool, cpu_count
# from concurrent.futures import ProcessPoolExecutor
# import os
from functools import partial
from joblib import Parallel, delayed
import FuzzyAHP as fahp
from values import CONSTANTS
from placed_functions import *

def get_reliance_node(user_node: int, target_nodes: list, channel: dict) -> int:
    """
    如果用户所在节点到某节点的速率最快，认为用户是依赖该节点的
    """
    max_speed = -2  # 因为node自己到自己的channel定为了-1，所以初始化-2，确保能选上自己（node自己上的用户依赖自己上的微服务）
    max_node = -1
    for other_node in target_nodes:
        if channel[(user_node, other_node)] > max_speed:
            max_speed = channel[(user_node, other_node)]
            max_node = other_node
    if max_node == -1:
        raise ValueError
    return max_node


def build_placed_nodes(current_deploy: list) -> list:
    placed_nodes = []
    for mserv_num in range(len(current_deploy)):
        placed_nodes.append(flatten_list(current_deploy[mserv_num]))

    return placed_nodes


def flatten_list(lst: list) -> list:
    """将多维列表展平为1维列表"""
    flattened_list = []
    for group in lst:
        flattened_list += group
    return flattened_list


def parallel_task(mserv_num: int, partition: list, current_deploy: list, fixed_place_mservs: list,
                  receive: dict, data: tuple) -> tuple:
    """对需要并行执行的代码进行封装"""
    mserv_min_zeta = (math.inf, -1, -1, -1)  # 针对该微服务的最小zeta记录器

    edge_nodes, mservs, users, channel = data
    mserv_partition = partition[mserv_num]  # 该微服务的分组
    mserv_deploy = current_deploy[mserv_num]  # 该微服务的放置情况

    partition_nodes = flatten_list(mserv_partition)
    deployed = flatten_list(mserv_deploy)

    # 该微服务至少放1个，如果已经只剩1个了就退出
    if len(deployed) <= 1:
        return math.inf, -1, -1, -1

    for group_num, group in enumerate(mserv_deploy):
        for node_num in group:
            if node_num in fixed_place_mservs[mserv_num]:  # 如果该节点已经固定放置，则跳过
                continue
            user_node_list = []  # 原先依赖该节点上的微服务，现在由于该节点取消放置微服务从而受影响的用户节点列表
            for user_node_num in receive[mserv_num]:  # 取“有用户请求该微服务”的节点
                if user_node_num == node_num:  # 如果用户节点就是自己，虽然当前自己上还有微服务，但下面就要计算去掉时的影响，所以应该算上自己
                    user_node_list.append(user_node_num)
                    continue
                if user_node_num in deployed:  # 如果这个用户所在节点上有微服务，它会直接依赖自己节点上的微服务
                    continue
                if get_reliance_node(user_node_num, deployed, channel) == node_num:  # 如果用户请求节点依赖该节点，则先添入待选列表
                    user_node_list.append(user_node_num)
            zeta = 0
            for user_node_num in user_node_list:
                deployed_discard = [node for node in deployed if node != node_num]  # 已放置微服务的节点列表，除去当前计算zeta的节点
                new_reliance_node = get_reliance_node(user_node_num, deployed_discard, channel)
                zeta += receive[mserv_num][user_node_num] / channel[(user_node_num, new_reliance_node)]
                zeta += mservs[mserv_num].request_resource / edge_nodes[new_reliance_node].computing_power
                if user_node_num != node_num:
                    zeta -= receive[mserv_num][user_node_num] / channel[(user_node_num, node_num)]
                zeta -= mservs[mserv_num].request_resource / edge_nodes[node_num].computing_power
            if zeta < mserv_min_zeta[0]:  # 记录最小的zeta
                mserv_min_zeta = (zeta, mserv_num, group_num, node_num)

    return mserv_min_zeta


def calc_objFunc(place_strategy: list, data: tuple) -> tuple:
    """
    计算目标函数值
    寻路方式使用gurobi求解，gurobi求解过程中无makespan约束
    :param place_strategy: 微服务放置策略，第一维list以微服务种类序号索引，第二维set是边缘节点序号的集合
    :param data: 打包传入信息edge_nodes: list, mservs: list, users: list, channel: dict
    :return: 目标函数值 ( C + T )、总的C和各自的T
    """
    edge_nodes, mservs, users, channel = data
    edgenode_count = len(edge_nodes)
    microservice_count = len(mservs)
    task_count = len(users)

    # 计算总cost
    total_cost = 0
    for mserv_num, node_set in enumerate(place_strategy):
        total_cost += mservs[mserv_num].place_cost * len(node_set)  # 该微服务放置单价*放置个数

    # gurobi寻路
    model = Model()
    y = {}  # y(i, k, h)路径中使用了第k个节点上的第i个微服务，并用h标识不同用户
    for i in range(microservice_count):
        for k in range(edgenode_count):
            for h in range(task_count):
                y[(i, k, h)] = model.addVar(
                    vtype=GRB.BINARY, name=f"y_{i}_{k}_{h}")
    model.update()
    # 一个f中，每个微服务只能指定一个节点提供
    for user in users:
        h = user.num
        f = user.mserv_dependency
        for i in f:
            model.addConstr(quicksum(y[(i, k, h)] for k in range(edgenode_count)) == 1)
    # y(i, k) <= x(i, k)
    for i in range(microservice_count):
        for k in range(edgenode_count):
            for h in range(task_count):
                if k in place_strategy[i]:
                    model.addConstr(y[(i, k, h)] <= 1)
                else:
                    model.addConstr(y[(i, k, h)] == 0)
    # 计算 makespan
    makespan = [LinExpr()] * task_count
    for user in users:
        h = user.num
        f = user.mserv_dependency
        makespan[h] = quicksum(  # d_in
            y[(f[0], k, h)] * (
                    (user.request_datasize / channel[(user.serv_node, k)] if user.serv_node != k else 0) +
                    mservs[f[0]].request_resource / edge_nodes[k].computing_power
            ) for k in range(edgenode_count)
        ) + quicksum(
            y[(f[i], k, h)] * (  # 中间的sigma求和
                    mservs[f[i - 1]].send_datasize * quicksum(  # 上一个节点到本节点的信道速率（的倒数，因为gurobi不允许除号后面有决策变量）
                y[(f[i - 1], k_prev, h)] * (1 / channel[(k_prev, k)] if k_prev != k else 0)
                for k_prev in range(edgenode_count)
            ) + mservs[f[i]].request_resource / edge_nodes[k].computing_power
            )
            for i in range(1, len(f)) for k in range(edgenode_count)
        ) + quicksum(  # d_out
            y[(f[-1], k, h)] * (
                mservs[f[-1]].send_datasize / channel[(k, user.serv_node)] if k != user.serv_node else 0)
            for k in range(edgenode_count)
        )
    model.setObjective(quicksum(makespan[h] for h in range(task_count)))
    model.modelSense = GRB.MINIMIZE
    model.Params.OutputFlag = 0  # 1表示开启输出（默认值），0为关闭
    model.optimize()
    if model.status == GRB.OPTIMAL:
        return total_cost + model.objVal, total_cost, list(map(lambda x: x.getValue(), makespan))
    return math.inf, 0, [0] * task_count

def get_adjacent_mservs(mserv_num: int, users: list) -> list:
    """
    获取指定微服务前后相邻的微服务
    """
    adjacent_mservs = []
    for user in users:
        if mserv_num in user.mserv_dependency:
            mserv_index = user.mserv_dependency.index(mserv_num)
            # 加入前一个微服务
            if mserv_index>0:
                adjacent_mservs.append(user.mserv_dependency[mserv_index-1])
            # 加入后一个微服务
            if mserv_index<len(user.mserv_dependency)-1:
                adjacent_mservs.append(user.mserv_dependency[mserv_index+1])
    adjacent_mservs = list(set(adjacent_mservs)) # 去重
    return adjacent_mservs

def greedy_combine(edge_nodes: list, mservs: list, users: list, channel: dict, connect: dict, zeta_min_select_proportion:float) -> None:
    data = edge_nodes, mservs, users, channel
    edgenode_count = len(edge_nodes)
    microservice_count = len(mservs)
    task_count = len(users)

    def calc_globalFactor() -> list:
        """
        计算global factor
        """
        node_sets = [set(mserv_req_nums[mserv_num].keys()) for mserv_num in range(microservice_count)]
        Dmi = [0] * microservice_count  # 每种微服务的平均节点间距离
        for mserv_num in range(microservice_count):
            node_pairs = itertools.combinations(node_sets[mserv_num], 2)
            link_num = 0
            for pair in node_pairs:
                Dmi[mserv_num] += 1 / channel[pair]  # TODO: 直接改成距离
                link_num += 1
            if link_num != 0:
                Dmi[mserv_num] /= link_num
            else:
                Dmi[mserv_num] = 0
        g_factor = [len(node_sets[mserv_num]) / edgenode_count * Dmi[mserv_num] for mserv_num in
                    range(microservice_count)]
        return g_factor

    def calc_localFactor() -> list:
        """
        利用模糊AHP，计算local factor
        """
        # 模糊比较矩阵，根据模糊准则确定模糊值
        criteria_matrix = [
            [fahp.FN(1, 1, 1), fahp.FN(1, 2, 3), fahp.FN(3, 4, 5), fahp.FN(2, 3, 4)],
            [fahp.FN(1 / 3, 1 / 2, 1), fahp.FN(1, 1, 1), fahp.FN(2, 3, 4), fahp.FN(1, 2, 3)],
            [fahp.FN(1 / 5, 1 / 4, 1 / 3), fahp.FN(1 / 4, 1 / 3, 1 / 2), fahp.FN(1, 1, 1), fahp.FN(1 / 3, 1 / 2, 1)],
            [fahp.FN(1 / 4, 1 / 3, 1 / 2), fahp.FN(1 / 3, 1 / 2, 1), fahp.FN(1, 2, 3), fahp.FN(1, 1, 1)],
        ]  # TODO: 模糊值待定
        weights = fahp.fuzzyAHP(criteria_matrix)

        mserv_properties = [[] for _ in range(microservice_count)]  # 第一维是微服务种类序号，第二维是节点序号和“微服务属性元组”组成的嵌套元组
        node_sets = [set(mserv_req_nums[mserv_num].keys()) for mserv_num in range(microservice_count)]
        for mserv_num, node_set in enumerate(node_sets):
            for node_num in node_set:
                req_user_num = 0
                cost_price = mservs[mserv_num].place_cost
                order = 0  # order的计量方式：比如有一个first的请求给权值为3，last给2，中间给1，然后对权值累加和求平均
                req_storage = mservs[mserv_num].memory_demand
                for user in user_distri[node_num]:
                    if mserv_num in user.mserv_dependency:
                        req_user_num += 1
                        if mserv_num == user.mserv_dependency[0]:
                            order += 3
                        elif mserv_num == user.mserv_dependency[-1]:
                            order += 2
                        else:
                            order += 1
                order /= req_user_num
                mserv_properties[mserv_num].append((node_num, (req_user_num, cost_price * 0.07, order, req_storage)))
        l_factor = [{} for _ in range(edgenode_count)]  # 第一维是微服务种类序号，第二维是节点序号和factor值
        for mserv_num, properties in enumerate(mserv_properties):
            for node_num, property in properties:
                factor = (  # TODO: 属性数据间的数量级还要调整
                        weights[0] * property[0]
                        + weights[1] * 1 / property[1]
                        + weights[2] * property[2]
                        + weights[3] * 1 / property[3]
                )
                l_factor[node_num][mserv_num] = factor
        return l_factor

    def memory_checkAndMigrate(pre_place_mservs: list, l_factor: list) -> list:
        """
        寻找内存超出节点限制的节点，从local factor最小的微服务开始迁移至最近（速率最快）且内存足够接收该微服务的节点
        """
        # 根据local factor，将内存超限的节点上factor相对小的微服务移至附近通信速率最快的且内存足够的节点
        pre_place_mservs_SeqInNodes = [set() for _ in range(edgenode_count)]
        # 先将拟放置微服务记录器按节点顺序重新分组
        for mserv_num, place_nodes in enumerate(pre_place_mservs):
            for node_num in place_nodes:
                pre_place_mservs_SeqInNodes[node_num].add(mserv_num)
        # 遍历寻找内存超限的节点，移出factor值较小的微服务
        for node_num, place_mservs in enumerate(pre_place_mservs_SeqInNodes):
            if sum(map(lambda x: mservs[x].memory_demand, place_mservs)) > edge_nodes[node_num].memory:
                mservNums_factorSorted = sorted(list(place_mservs),
                                                key=lambda x: l_factor[node_num][x])  # 按factor值升序排列的微服务序号列表
                targetNode_list = sorted(list(range(edgenode_count)), key=lambda x: channel[(node_num, x)],
                                         reverse=True)  # 按通信速率降序排列的目标节点列表
                while sum(map(lambda x: mservs[x].memory_demand, place_mservs)) > edge_nodes[
                    node_num].memory:  # 一直移除直到内存足够
                    for targetNode_num in targetNode_list:
                        # 如果目标节点没有该微服务，且目标节点有足够内存放下该微服务
                        if mservNums_factorSorted[0] not in pre_place_mservs_SeqInNodes[targetNode_num] and \
                                edge_nodes[targetNode_num].memory - \
                                sum(map(lambda x: mservs[x].memory_demand, pre_place_mservs_SeqInNodes[targetNode_num])) \
                                >= mservs[mservNums_factorSorted[0]].memory_demand:
                            pre_place_mservs_SeqInNodes[targetNode_num].add(mservNums_factorSorted[0])
                            break
                    else:
                        return []
                        # raise Exception("没有任何节点能接收移出的微服务")
                    place_mservs.remove(mservNums_factorSorted[0])
                    del mservNums_factorSorted[0]
                pre_place_mservs_SeqInNodes[node_num] = place_mservs
        # 按微服务顺序重新分组回到原格式
        migrated_pre_place_mservs = [set() for _ in range(microservice_count)]  # 清除移动前的方案
        for node_num, place_mservs in enumerate(pre_place_mservs_SeqInNodes):
            for mserv_num in place_mservs:
                migrated_pre_place_mservs[mserv_num].add(node_num)
        return migrated_pre_place_mservs

    # upper bound 根据总cost的限制来计算，当其他微服务都只放置一个的情况下，该微服务最多能放置的个数即为该微服务的上界。对所有微服务同样处理即得完整的上界
    upper_bound = []
    for mserv in mservs:
        remain_cost = CONSTANTS.MAX_DEPLOY_COST - sum(map(lambda x: x.place_cost, mservs))  # 其他微服务都放一个，剩余可用的cost
        max_placeNum = remain_cost // mserv.place_cost  # 剩余cost最多可以放这个微服务的个数（向下取整）
        upper_bound.append(min(max_placeNum, edgenode_count))  # 放置数量不会超过边缘节点总数

    # 统计边缘节点下属用户情况
    user_distri = {}
    for user in users:
        user_distri.setdefault(user.serv_node, [])
        user_distri[user.serv_node].append(user)
    # 统计各边缘节点上有用户请求到的微服务，以及请求该微服务的用户个数。第一维是边缘节点序号，第二维是以微服务序号为键、请求用户个数为值的字典
    mserv_req_nums_SeqInNodes = [{} for _ in range(edgenode_count)]
    for node_num, node_users in user_distri.items():  # 边缘节点序号、节点上的用户列表
        for user in node_users:  # 对该节点上的所有用户遍历
            for mserv_num in user.mserv_dependency:  # 该用户请求的所有微服务的序号
                mserv_req_nums_SeqInNodes[node_num].setdefault(mserv_num, 0)  # 如果该微服务没有被记录过，则记录该微服务并初始化请求用户计数为0
                mserv_req_nums_SeqInNodes[node_num][mserv_num] += 1  # 请求用户计数+1
    # 统计各种微服务都有哪些边缘节点上有用户请求到，以及这些节点上请求该微服务的用户个数。第一维是微服务种类序号，第二维是以边缘节点序号为键、请求用户个数为值的字典
    mserv_req_nums = [{} for _ in range(microservice_count)]
    for node_num, mserv_count_dict in enumerate(mserv_req_nums_SeqInNodes):  # 边缘节点序号、节点上请求到的微服务序号及用户个数字典
        for mserv_num, req_count in mserv_count_dict.items():  # 该节点上有用户请求的微服务序号、请求用户个数
            mserv_req_nums[mserv_num][node_num] = req_count

    mserv_user_count = count_mserv_user(mservs, users)  # 核对过，和mserv_req_nums是一样的
    mserv_receive_data_count = count_mserv_receive_dataflow(mservs, users)
    ksi = 0.6
    upper_bound_dict = {key: value for key, value in enumerate(upper_bound)}
    partition_and_pre_deploy_info = place_mserv(upper_bound_dict, ksi, edge_nodes, mservs, users, channel, connect,
                                                mserv_receive_data_count, mserv_user_count)
    partition = []
    initial_deploy = []
    for mserv_num, info in enumerate(partition_and_pre_deploy_info):
        partition.append(info['devide_node_group 被划分的节点群'])
        initial_deploy.append(info['mserv_place_node_list 微服务放置节点'])

    """
    # 针对每种微服务，如果请求它的节点数量比upper bound中计算的“能放的最多数量”要少，则这些节点上全部拟放置该微服务；
    # 否则，按照请求用户个数由大到小排序，依次拟放置该微服务（注意：拟放置是指合并前的预先放置，不是真正放置下去，它没有考虑总cost限制和内存限制）
    pre_place_mservs = [set() for _ in range(microservice_count)]  # 拟放置微服务情况记录，第一维是微服务种类序号，第二维是边缘节点序号的集合
    for mserv_num in range(microservice_count):
        node_req_count = mserv_req_nums[mserv_num]  # 这个微服务有哪些节点上请求到（键），有几个用户请求（值）
        max_placeNum = upper_bound[mserv_num]  # 该微服务最多放置的个数
        if len(node_req_count) > max_placeNum:  # 如果该微服务请求的节点数大于其上界
            node_req_count_sorted = sorted(node_req_count.items(), key=lambda x: x[1], reverse=True)  # 按请求用户个数由大到小排序
            place_count = 0  # 放置计数器
            for node_num, req_count in node_req_count_sorted:  # 从用户个数最多的节点开始放置，直到额度用完
                pre_place_mservs[mserv_num].add(node_num)  # 在该节点上拟放置该微服务
                place_count += 1
                if place_count == max_placeNum:  # 额度用完就退出
                    break
        else:
            for node_num in node_req_count:  # 如果该微服务请求的节点数小于其上界，则这些节点上都拟放置该微服务
                pre_place_mservs[mserv_num].add(node_num)
    """

    """
    # 初始放置最优解测试
    temp = [(0, 1), (0, 5), (1, 4), (1, 7), (2, 2), (2, 6), (3, 0), (3, 1), (3, 8), (4, 3), (4, 7), (5, 4), (5, 8), (6, 2), (6, 8), (7, 3), (7, 7), (8, 4), (8, 8), (9, 1), (9, 8), (10, 4), (10, 7), (11, 2), (11, 3), (11, 8), (12, 5), (12, 9), (13, 2), (13, 9), (14, 2), (14, 8)]
    pre_place_mservs = [set() for _ in range(microservice_count)]
    for mserv_num, node_num in temp:
        pre_place_mservs[mserv_num].add(node_num)
    """

    print("微服务请求情况")
    print("微服务序号 | 请求总个数 | 请求在各节点上的分布")
    for mserv_num, req_nums in enumerate(mserv_req_nums):
        print(mserv_num, sum(req_nums.values()), req_nums)
    print()

    """
    print("拟放置微服务情况")
    print("微服务序号 | 拟放置节点")
    for mserv_num, placed_nodes in enumerate(pre_place_mservs):
        print(mserv_num, placed_nodes)

    print()
    g_factor = calc_globalFactor()
    print("global factor")
    print("微服务序号 | global factor")
    for mserv_num, gf in enumerate(g_factor):
        print(f"{mserv_num}: {gf}")
    """

    # 贪婪合并，寻找梯度(delta)最大的合并方案
    max_delta = (math.inf, -1, -1)  # 初始化最大delta记录器
    # max_delta元组形式：(delta值, 微服务对应的mserv_num, 合并去掉的节点node_num)
    min_zeta = (math.inf, -1, -1, -1)
    # min_zeta元组形式：(zeta值, 微服务对应的mserv_num, 节点所在的group, 合并去掉的节点node_num)
    current_deploy = initial_deploy
    fixed_place_mservs = [set() for _ in range(microservice_count)]  # 固定放置不动的微服务列表
    loop_count = 0
    start_time = time.time()
    time_elapsed = 0.0
    allow_stop_flag = False
    break_flag = False
    while max_delta[0] > -150 or not allow_stop_flag:  # 迭代停止条件：合并后，目标函数值没有变化，甚至反向变大
        loop_count += 1
        print()
        print(f"第{loop_count}轮合并")
        max_delta = (-math.inf, -1, -1)
        placed_nodes = build_placed_nodes(current_deploy)
        obj_before, _, _ = calc_objFunc(placed_nodes, data)
        if obj_before == math.inf:
            raise Exception("合并前即无解")

        parallel_task_fixed = partial(parallel_task, partition=partition, current_deploy=current_deploy,
                                      fixed_place_mservs=fixed_place_mservs, receive=mserv_receive_data_count,
                                      data=data)  # 固定相同的参数
        mserv_min_zetas = Parallel(n_jobs=-1)(
            delayed(parallel_task_fixed)(i) for i in range(microservice_count))  # 计算各并行任务中，各微服务的最小zeta
        min_zeta = min(mserv_min_zetas, key=lambda x: x[0]) # zeta值,微服务序号,分组序号,节点序号
        print(
            "各微服务各自实例的实例最小zeta列表（列表按mserv序号索引，每个元素是一个微服务实例的属性，为元组(zeta值,微服务序号,分组序号,节点序号)）")
        print(mserv_min_zetas)
        # 选择一定比例的微服务进行合并，先将其排序
        mserv_min_zetas_sorted = sorted(mserv_min_zetas, key=lambda x: x[0])
        # 取前面一定比例的最小zeta值微服务进行合并，此为合并数
        mserv_merge_num = round(zeta_min_select_proportion * len(mserv_min_zetas_sorted))
        """
        process_num = os.cpu_count()  # 允许同时存在的进程数，令其值为CPU核心数
        with ProcessPoolExecutor(max_workers=process_num) as executor:
            parallel_task_fixed = partial(parallel_task, pre_place_mservs, obj_before, data)  # 固定相同的参数
            mserv_max_deltas = list(executor.map(parallel_task_fixed, range(microservice_count)))  # 计算各并行任务中，各微服务的最大delta
        max_delta = max(mserv_max_deltas, key=lambda x: x[0])
        """
        """
        if __name__ == 'greedy_combine':
            process_num = cpu_count()  # 允许同时存在的进程数，令其值为CPU核心数
            with Pool(processes=process_num) as pool:
                parallel_task_fixed = partial(parallel_task, pre_place_mservs, obj_before, data)  # 固定相同的参数
                mserv_max_deltas = pool.map(parallel_task_fixed, range(microservice_count))  # 计算各并行任务中，各微服务的最大delta
            max_delta = max(mserv_max_deltas, key=lambda x: x[0])
        """
        """
        for mserv_num in range(microservice_count):  # TODO: 改成多线程并行
            pre_place_node_set = pre_place_mservs[mserv_num]  # 拟放置该微服务的边缘节点集合
            if len(pre_place_node_set) <= 1:
                continue
            for node_num in pre_place_node_set:  # 逐个尝试，去掉一个节点
                obj_after, _, _ = calc_objFunc(pre_place_mservs[:mserv_num] +
                                               [pre_place_node_set - {node_num}] +
                                               pre_place_mservs[mserv_num + 1:])
                delta = obj_before - obj_after
                if delta > max_delta[0]:  # 记录最大的delta
                    max_delta = (delta, mserv_num, node_num)
        """
        # 如果全都只剩1个微服务，无法再继续合并
        merged_mservs=[]
        if min_zeta[1] == -1:
            print("无法再继续合并")
            break_flag = True
        else:
            # 更新放置情况集合
            # current_deploy[min_zeta[1]][min_zeta[2]].remove(min_zeta[3])
            # 移除升序排列后的zeta列表按比例前几个的微服务所在节点
            wait_to_merge_mserv=mserv_min_zetas_sorted[:mserv_merge_num]
            wait_to_merge_mserv=[t for t in wait_to_merge_mserv if t[0] != math.inf]
            # print("mserv_min_zetas_sorted: ", mserv_min_zetas_sorted)
            skip_mservs=[]
            # merged_mservs=[]
            # print("min_zta_list: ", wait_to_merge_mserv)
            for mserv_min_zeta in wait_to_merge_mserv:
                if mserv_min_zeta[1] in skip_mservs:
                    continue
                # print(f"第{mserv_min_zeta[1]}个微服务在节点{mserv_min_zeta[3]}上被合并去掉")
                # print(f"curren_deploy: {current_deploy}")
                current_deploy[mserv_min_zeta[1]][mserv_min_zeta[2]].remove(mserv_min_zeta[3])
                skip_mservs.extend(get_adjacent_mservs(mserv_min_zeta[1], users)) # 将相邻微服务加入跳过列表
                merged_mservs.append(mserv_min_zeta)
            # pre_place_mservs = (pre_place_mservs[:max_delta[1]] +
            #                    [pre_place_mservs[max_delta[1]] - {max_delta[2]}] +
            #                    pre_place_mservs[max_delta[1] + 1:])
            for mserv_min_zeta in merged_mservs:
                print(f"第{mserv_min_zeta[1]}个微服务在节点{mserv_min_zeta[3]}上被合并去掉")
            print("合并后的放置情况集合：")
            print(current_deploy)
            print("当前微服务固定情况：")
            print(fixed_place_mservs)
        # 判断主问题、子问题是否满足约束
        placed_nodes = build_placed_nodes(current_deploy)
        obj, cost, makespans = calc_objFunc(placed_nodes, data)
        max_delta = (obj_before - obj, min_zeta[1], min_zeta[3])
        print("目标函数值(C+T)：", obj)
        if cost > CONSTANTS.MAX_DEPLOY_COST:
            print("主问题cost超过最大限制，主问题infeasible")
        else:
            print("除内存限制外，主问题feasible，cost=", cost)
            total_memory = 0  # 所有节点提供的内存总量
            for node in edge_nodes:
                total_memory += node.memory
            total_memory_demand = 0  # 拟放置的所有微服务需求的内存总量
            for mserv_num, node_set in enumerate(placed_nodes):
                total_memory_demand += mservs[mserv_num].memory_demand * len(node_set)
            if total_memory_demand <= total_memory:
                print("节点总内存 > 当前拟放置微服务总需求内存，考虑进行内存检查")
                print("***进行内存检查与微服务迁移***")
                l_factor = calc_localFactor()
                migrated_pre_place_mservs = memory_checkAndMigrate(placed_nodes, l_factor)
                if not migrated_pre_place_mservs:
                    print("迁移过程中没有节点内存足够接收移出的微服务")
                else:
                    allow_stop_flag = True
                    obj, cost, makespans = calc_objFunc(migrated_pre_place_mservs, data)
                    print("迁移后目标函数值(C+T)：", obj)
                    print("迁移后主问题cost=", cost)
            else:
                print("节点总内存 < 当前拟放置微服务总需求内存，不进行内存检查，认为主问题infeasible")

            violate_list = []
            for user_num, makespan in enumerate(makespans):
                if makespan > CONSTANTS.MAX_MAKESPAN:
                    violate_list.append((user_num, makespan))
            if violate_list:
                print("子问题makespan超过最大限制，子问题infeasible，总makespan=", sum(makespans))
                print("其中如下用户的makespan超出限制：")
                for user_num, makespan in violate_list:
                    print(f"用户{user_num}的makespan为{makespan}")
                print(f"将第{min_zeta[1]}个微服务在节点{min_zeta[3]}上进行固定")
                # TODO: 确认是否这里也要批量放入
                # fixed_place_mservs[min_zeta[1]].add(min_zeta[3])  # 记录固定微服务
                for mserv_min_zeta in merged_mservs:
                    fixed_place_mservs[mserv_min_zeta[1]].add(mserv_min_zeta[3])
                # TODO: 预计修改
                # current_deploy[min_zeta[1]][min_zeta[2]].append(min_zeta[3])  # 固定的微服务加回拟放置集合
                for mserv_min_zeta in merged_mservs:
                    current_deploy[mserv_min_zeta[1]][mserv_min_zeta[2]].append(mserv_min_zeta[3])
            else:
                print("子问题feasible，总makespan=", sum(makespans))

        print()
        time_elapsed = time.time() - start_time
        print(f"已用时{time_elapsed:.2f} sec")
        if break_flag:
            break
