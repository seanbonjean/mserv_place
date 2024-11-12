import itertools
import math
import random
import time
from gurobipy import *
from values import CONSTANTS
import FuzzyAHP as fahp


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

    # !!!新：heuristic initial cut
    # Step1: 计算global factor
    user_distri = {}  # 统计边缘节点下属用户情况
    for user in users:
        user_distri.setdefault(user.serv_node, [])
        user_distri[user.serv_node].append(user)
    mserv_sets = [set() for _ in range(edgenode_count)]  # 统计各边缘节点上有用户请求到的微服务，第一维是边缘节点序号，第二维是微服务序号（set集合）
    for node_num, node_users in user_distri.items():
        for user in node_users:
            for mserv_num in user.mserv_dependency:
                mserv_sets[node_num].add(mserv_num)
    node_sets = [set() for _ in range(microservice_count)]  # 统计各微服务的边缘节点分布情况，第一维是微服务种类序号，第二维是边缘节点序号（set集合）
    for node_num, mserv_set in enumerate(mserv_sets):
        for mserv_num in mserv_set:
            node_sets[mserv_num].add(node_num)
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
    g_factor = [len(node_sets[mserv_num]) / edgenode_count * Dmi[mserv_num] for mserv_num in range(microservice_count)]

    # Step2: 计算local factor
    # 模糊比较矩阵，根据模糊准则确定模糊值
    criteria_matrix = [
        [fahp.FN(1, 1, 1), fahp.FN(1, 2, 3), fahp.FN(2, 3, 4), fahp.FN(3, 4, 5)],
        [fahp.FN(1 / 3, 1 / 2, 1), fahp.FN(1, 1, 1), fahp.FN(1, 2, 3), fahp.FN(2, 3, 4)],
        [fahp.FN(1 / 4, 1 / 3, 1 / 2), fahp.FN(1 / 3, 1 / 2, 1), fahp.FN(1, 1, 1), fahp.FN(1, 2, 3)],
        [fahp.FN(1 / 5, 1 / 4, 1 / 3), fahp.FN(1 / 4, 1 / 3, 1 / 2), fahp.FN(1 / 3, 1 / 2, 1), fahp.FN(1, 1, 1)]
    ]  # TODO: 模糊值待定
    weights = fahp.fuzzyAHP(criteria_matrix)

    mserv_properties = [[] for _ in range(microservice_count)]  # 第一维是微服务种类序号，第二维是节点序号和“微服务属性元组”组成的嵌套元组
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
    l_factor = [[] for _ in range(microservice_count)]  # 第一维是微服务种类序号，第二维是节点序号和factor值
    for mserv_num, properties in enumerate(mserv_properties):
        for node_num, property in properties:
            factor = (  # TODO: 属性数据间的数量级还要调整
                    weights[0] * property[0]
                    + weights[1] * 1 / property[1]
                    + weights[2] * property[2]
                    + weights[3] * 1 / property[3]
            )
            l_factor[mserv_num].append((node_num, factor))

    # Step3: combine the global and local factor
    alpha = 0.5
    theta_demand_factor = [
        g_factor[mserv_num] * alpha +
        sum(map(lambda x: x[1], l_factor[mserv_num])) * (1 - alpha)
        for mserv_num in range(microservice_count)
    ]

    # P就是max(theta_mi)
    # 0<epsilon<1任意给
    # n是theta的个数
    # 先拿微服务部署最大总成本来，把每个微服务都放置一个，剩下的总成本限制再拿来做背包








    """
    model_m.addConstr(quicksum(x[0, k] for k in range(edgenode_count)) <= 3)
    model_m.addConstr(quicksum(x[1, k] for k in range(edgenode_count)) <= 2)
    model_m.addConstr(quicksum(x[2, k] for k in range(edgenode_count)) <= 3)
    model_m.addConstr(quicksum(x[3, k] for k in range(edgenode_count)) <= 3)
    model_m.addConstr(quicksum(x[4, k] for k in range(edgenode_count)) <= 2)
    model_m.addConstr(quicksum(x[5, k] for k in range(edgenode_count)) <= 2)
    model_m.addConstr(quicksum(x[6, k] for k in range(edgenode_count)) <= 3)
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
    current_best_main_obj = math.inf
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

        # 只有当次主问题的cost+q比之前的所有cost+q都小，才给optimal cut
        if model_m.objVal > current_best_main_obj:
            # feasible cut
            model_m.addConstr(quicksum(
                1 - x[(i, k)] if x_result[(i, k)] == 1 else x[(i, k)] for i in range(microservice_count)
                for k in range(edgenode_count)
            ) >= 1)
            continue
        current_best_main_obj = model_m.objVal

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
            print('\n' * 3 + '*' * 10 + '!' * 3 + str(obj_val) + '*' * 10 + '\n' * 3)
