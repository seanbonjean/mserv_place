import math
from gurobipy import *
from values import CONSTANTS


def greedy_combine(edge_nodes: list, mservs: list, users: list, channel: dict) -> None:
    edgenode_count = len(edge_nodes)
    microservice_count = len(mservs)
    task_count = len(users)

    def calc_objFunc(place_strategy: list) -> tuple:
        """
        计算目标函数值
        寻路方式使用gurobi求解，gurobi求解过程中无makespan约束
        :param place_strategy: 微服务放置策略，第一维list以微服务种类序号索引，第二维set是边缘节点序号的集合
        :return: 目标函数值 ( C + T )、总的C和各自的T
        """
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

    # 针对每种微服务，如果请求它的节点数量比upper bound中计算的“能放的最多数量”要少，则这些节点上全部拟放置该微服务；
    # 否则，按照请求用户个数由大到小排序，依次拟放置该微服务（注意：拟放置是指合并前的预先放置，不是真正放置下去，它没有考虑总cost限制）
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

    for mserv_num, req_nums in enumerate(mserv_req_nums):
        print(mserv_num, sum(req_nums.values()), req_nums)

    # 贪婪合并，寻找梯度(delta)最大的合并方案
    max_delta = (math.inf, -1, -1)  # 初始化最大delta记录器
    # max_delta元组形式：(delta值, 微服务对应的mserv_num, 合并去掉的节点node_num)
    loop_count = 0
    while max_delta[0] > 0:  # 迭代停止条件：合并后，目标函数值没有变化，甚至反向变大
        loop_count += 1
        print()
        print(f"第{loop_count}轮合并")
        max_delta = (-math.inf, -1, -1)
        obj_before, _, _ = calc_objFunc(pre_place_mservs)
        if obj_before == math.inf:
            raise Exception("合并前即无解")
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
        # 如果全都只剩1个微服务，无法再继续合并
        if max_delta[1] == -1:
            print("无法再继续合并")
        else:
            # 更新放置情况集合
            pre_place_mservs = (pre_place_mservs[:max_delta[1]] +
                                [pre_place_mservs[max_delta[1]] - {max_delta[2]}] +
                                pre_place_mservs[max_delta[1] + 1:])
            print(f"第{max_delta[1]}个微服务在节点{max_delta[2]}上被合并去掉")
            print("合并后的放置情况集合：")
            print(pre_place_mservs)
        # 判断主问题、子问题是否满足约束
        obj, cost, makespans = calc_objFunc(pre_place_mservs)
        print("目标函数值(C+T)：", obj)
        if cost > CONSTANTS.MAX_DEPLOY_COST:
            print("主问题cost超过最大限制，主问题infeasible")
        else:
            print("主问题feasible，cost=", cost)
        violate_list = []
        for user_num, makespan in enumerate(makespans):
            if makespan > CONSTANTS.MAX_MAKESPAN:
                violate_list.append((user_num, makespan))
        if violate_list:
            print("子问题makespan超过最大限制，子问题infeasible")
            print("其中如下用户的makespan超出限制：")
            for user_num, makespan in violate_list:
                print(f"用户{user_num}的makespan为{makespan}")
        else:
            print("子问题feasible，总makespan=", sum(makespans))
