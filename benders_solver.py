import math
import random
import time
from gurobipy import *
from values import CONSTANTS


def benders_solve(edge_nodes: list, mservs: list, users: list, channel: dict) -> None:
    """
    通过benders分解问题，循环求解主、子问题，逼近原问题最优解
    """
    edgenode_count = len(edge_nodes)
    microservice_count = len(mservs)
    task_count = len(users)
    Th_max = CONSTANTS.MAX_MAKESPAN
    C_max = CONSTANTS.MAX_DEPLOY_COST

    model_m = Model("main_problem")
    model_s = Model("sub_problem")

    # 添加决策变量
    x = {}  # x(i, k)在第k个节点放置第i个微服务
    # q 体现子问题求解的好坏
    y = {}  # y(i, k, h)路径中使用了第k个节点上的第i个微服务，并用h标识不同用户
    # x(i, k)
    for i in range(microservice_count):
        for k in range(edgenode_count):
            x[(i, k)] = model_m.addVar(vtype=GRB.BINARY, name=f"x_{i}_{k}")
    # q
    q = model_m.addVar(vtype=GRB.CONTINUOUS, name="q")
    # y(i, k, h)
    for i in range(microservice_count):
        for k in range(edgenode_count):
            for h in range(task_count):
                y[(i, k, h)] = model_s.addVar(
                    vtype=GRB.BINARY, name=f"y_{i}_{k}_{h}", lb=0, ub=1)

    # 更新模型
    model_m.update()
    model_s.update()

    # 添加主问题约束
    # C < C_max
    Cost = quicksum(x[(i, k)] * mservs[i].place_cost for i in range(microservice_count) for k in range(edgenode_count))
    model_m.addConstr(Cost <= C_max)

    # 边缘节点上放置的微服务总内存占用不超过节点内存容量上限
    for k in range(edgenode_count):
        model_m.addConstr(
            quicksum(x[(i, k)] * mservs[i].memory_demand for i in range(microservice_count)) <= edge_nodes[k].memory
        )

    # !!!新：只要有用户请求了的微服务，必须至少放置一个
    mserv_frequency = [0] * len(mservs)
    # 统计微服务频数
    for user in users:
        for mserv_num in user.mserv_dependency:
            mserv_frequency[mserv_num] += 1
    for mserv_num, mserv_freq in enumerate(mserv_frequency):
        if mserv_freq > 1:
            model_m.addConstr(quicksum(x[(mserv_num, k)] for k in range(edgenode_count)) == random.choice((1, 2)))
        elif mserv_freq == 1:
            model_m.addConstr(quicksum(x[(mserv_num, k)] for k in range(edgenode_count)) == 1)

    """
    # !!!新：heuristic initial cut
    user_distri = {}  # 统计边缘节点下属用户情况
    for user in users:
        user_distri.setdefault(user.serv_node, [])
        user_distri[user.serv_node].append(user)
    mserv_sets = [set() for _ in range(edgenode_count)]  # 统计各边缘节点上有用户请求到的微服务
    for node_num, node_users in user_distri.items():
        for user in node_users:
            for mserv_num in user.mserv_dependency:
                mserv_sets[node_num].add(mserv_num)
    node_sets = [set() for _ in range(microservice_count)]  # 统计各微服务的边缘节点分布情况
    for node_num, mserv_set in enumerate(mserv_sets):
        for mserv_num in mserv_set:
            node_sets[mserv_num].add(node_num)
    """

    # 添加子问题约束
    # 一个f中，每个微服务只能指定一个节点提供
    for user in users:
        h = user.num
        f = user.mserv_dependency
        for i in f:
            model_s.addConstr(quicksum(y[(i, k, h)] for k in range(edgenode_count)) == 1)

    # Th < Th_max
    makespan = [0] * task_count
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
        model_s.addConstr(makespan[h] <= Th_max)

    # 目标函数
    model_m.setObjective(Cost + q)
    model_m.modelSense = GRB.MINIMIZE
    model_s.setObjective(quicksum(makespan[h] for h in range(task_count)))
    model_s.modelSense = GRB.MINIMIZE

    # 循环求解
    L = 0
    loop_count = 0
    start_time = time.time()
    need_rm_constrs = []  # 存储该轮中更新的子问题约束的引用，即可在进入下一轮之前删除该轮添加的约束
    while not (time.time() - start_time > 20):
        loop_count += 1
        print('\n' * 3 + '*' * 10 + f"第{loop_count}轮循环" + '*' * 10)
        print('\n' + '*' * 10 + "主问题求解" + '*' * 10)
        model_m.optimize()
        x_result = {}  # 保存主问题的一个解，传递给子问题
        for i in range(microservice_count):
            for k in range(edgenode_count):
                x_result[(i, k)] = 1 if x[(i, k)].X > 0.9 else 0
        # 更新子问题约束
        for constr in need_rm_constrs:
            model_s.remove(constr)  # 删除上一轮中添加的约束
        need_rm_constrs = []
        # y(i, k) <= x(i, k)
        for i in range(microservice_count):
            for k in range(edgenode_count):
                for h in range(task_count):
                    constr = model_s.addConstr(y[(i, k, h)] <= x_result[(i, k)])
                    need_rm_constrs.append(constr)  # 存储引用
        """
        # 子问题决策变量回到松弛前
        for i in range(microservice_count):
            for k in range(edgenode_count):
                for h in range(task_count):
                    y[(i, k, h)].vtype = GRB.BINARY
        model_s.update()
        """
        print('\n' * 3 + '*' * 10 + "判断子问题松弛前是否有解" + '*' * 10)
        # 判断子问题松弛前是否有解
        model_s.Params.TimeLimit = 10  # 设置时间限制（单位：s）
        model_s.optimize()
        if model_s.SolCount == 0:
            # 如果子问题松弛前无解，则添加feasible cut
            model_m.addConstr(quicksum(
                1 - x[(i, k)] if x_result[(i, k)] == 1 else x[(i, k)] for i in range(microservice_count)
                for k in range(edgenode_count)
            ) >= 1)
            continue
        model_s.Params.TimeLimit = math.inf  # 取消时间限制
        """
        # 松弛子问题
        for i in range(microservice_count):
            for k in range(edgenode_count):
                for h in range(task_count):
                    y[(i, k, h)].vtype = GRB.CONTINUOUS
                    y[(i, k, h)].lb = 0
                    y[(i, k, h)].ub = 1
        model_s.update()
        """
        model_s_relaxed = model_s.relax()  # 松弛子问题
        print('\n' * 3 + '*' * 10 + "子问题求解" + '*' * 10)
        model_s_relaxed.optimize()

        """
        # 测试是否被松弛
        for i in range(microservice_count):
            for k in range(edgenode_count):
                for h in range(task_count):
                    if 1.0 > y[(i, k, h)].X > 0.0:
                        print(f'y[{i}, {k}, {h}] = {y[(i, k, h)].X}')
                        raise Exception('被松弛了')
        """

        # 将松弛后y的解修正为BINARY
        y_result = {}
        for h in range(task_count):
            for i in range(microservice_count):
                max_y = -1
                max_y_nok = -1
                for k in range(edgenode_count):
                    # 记录所有k中最大的y
                    if y[(i, k, h)].X > max_y:
                        max_y = y[(i, k, h)].X
                        max_y_nok = k
                for k in range(edgenode_count):
                    # 将最大的y设为1，其余设为0
                    if k == max_y_nok:
                        y_result[(i, k, h)] = 1
                    else:
                        y_result[(i, k, h)] = 0
        makespan_opt = [0] * task_count
        for user in users:
            h = user.num
            f = user.mserv_dependency
            makespan_opt[h] = sum(  # d_in
                y_result[(f[0], k, h)] * (
                        (user.request_datasize / channel[(user.serv_node, k)] if user.serv_node != k else 0) +
                        mservs[f[0]].request_resource / edge_nodes[k].computing_power
                ) for k in range(edgenode_count)
            ) + sum(
                y_result[(f[i], k, h)] * (  # 中间的sigma求和
                        mservs[f[i - 1]].send_datasize * sum(  # 上一个节点到本节点的信道速率（的倒数，因为gurobi不允许除号后面有决策变量）
                    y_result[(f[i - 1], k_prev, h)] * (1 / channel[(k_prev, k)] if k_prev != k else 0)
                    for k_prev in range(edgenode_count)
                ) + mservs[f[i]].request_resource / edge_nodes[k].computing_power
                )
                for i in range(1, len(f)) for k in range(edgenode_count)
            ) + sum(  # d_out
                y_result[(f[-1], k, h)] * (
                    mservs[f[-1]].send_datasize / channel[(k, user.serv_node)] if k != user.serv_node else 0)
                for k in range(edgenode_count)
            )
            if makespan_opt[h] > Th_max:
                # 如果不满足deadline约束，则添加feasible cut
                model_m.addConstr(quicksum(
                    1 - x[(i, k)] if x_result[(i, k)] == 1 else x[(i, k)] for i in range(microservice_count)
                    for k in range(edgenode_count)
                ) >= 1)
                break
        else:
            # 添加optimal cut
            phi = sum(makespan_opt[h] for h in range(task_count))
            model_m.addConstr(q >= phi - (phi - L) * quicksum(
                1 - x[(i, k)] if x_result[(i, k)] == 1 else x[(i, k)] for i in range(microservice_count)
                for k in range(edgenode_count)
            ))

            obj_val = model_m.objVal - q.X + phi
            print('\n' * 3 + '*' * 10 + str(obj_val) + '*' * 10 + '\n' * 3)
