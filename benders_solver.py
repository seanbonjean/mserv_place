import itertools
import math
import random
import time
from gurobipy import *

from utils import print_grb_solve_status
from values import CONSTANTS
import FuzzyAHP as fahp
from knapsack import fptas_knapsack


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

    """
    # !!!新：只要有用户请求了的微服务，必须至少放置一个
    mserv_frequency = [0] * len(mservs)
    # 统计微服务频数
    for user in users:
        for mserv_num in user.mserv_dependency:
            mserv_frequency[mserv_num] += 1
    for mserv_num, mserv_freq in enumerate(mserv_frequency):
        if mserv_freq > 1:
            model_m.addConstr(quicksum(x[(mserv_num, k)] for k in range(edgenode_count)) == random.choice((1, 2)))
            # model_m.addConstr(quicksum(x[(mserv_num, k)] for k in range(edgenode_count)) <= 5)
        elif mserv_freq == 1:
            model_m.addConstr(quicksum(x[(mserv_num, k)] for k in range(edgenode_count)) == 1)
    """
    """
    model_m.addConstr(quicksum(x[(0, k)] for k in range(edgenode_count)) == 1)
    model_m.addConstr(quicksum(x[(1, k)] for k in range(edgenode_count)) == 1)
    model_m.addConstr(quicksum(x[(2, k)] for k in range(edgenode_count)) == 2)
    model_m.addConstr(quicksum(x[(3, k)] for k in range(edgenode_count)) == 1)
    model_m.addConstr(quicksum(x[(4, k)] for k in range(edgenode_count)) == 2)
    model_m.addConstr(quicksum(x[(5, k)] for k in range(edgenode_count)) == 2)
    model_m.addConstr(quicksum(x[(6, k)] for k in range(edgenode_count)) == 1)
    model_m.addConstr(quicksum(x[(7, k)] for k in range(edgenode_count)) == 1)
    """
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
        [fahp.FN(1, 1, 1), fahp.FN(1, 2, 3), fahp.FN(3, 4, 5), fahp.FN(2, 3, 4)],
        [fahp.FN(1 / 3, 1 / 2, 1), fahp.FN(1, 1, 1), fahp.FN(2, 3, 4), fahp.FN(1, 2, 3)],
        [fahp.FN(1 / 5, 1 / 4, 1 / 3), fahp.FN(1 / 4, 1 / 3, 1 / 2), fahp.FN(1, 1, 1), fahp.FN(1 / 3, 1 / 2, 1)],
        [fahp.FN(1 / 4, 1 / 3, 1 / 2), fahp.FN(1 / 3, 1 / 2, 1), fahp.FN(1, 2, 3), fahp.FN(1, 1, 1)],
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

    # 先拿微服务部署最大总成本来，把每个微服务都放置一个，剩下的总成本限制再拿来做背包
    remain_cost = math.floor(CONSTANTS.MAX_DEPLOY_COST - sum(map(lambda x: x.place_cost, mservs)))
    place_costs = [math.floor(mserv.place_cost) for mserv in mservs]
    _, place_method = fptas_knapsack(place_costs, theta_demand_factor, remain_cost, 0.5)

    for i in range(microservice_count):
        if place_method.get(i) is None:
            place_method[i] = 0
        place_method[i] += 1
        # TODO: 需要修改策略，让1少点
        model_m.addConstr(quicksum(x[i, k] for k in range(edgenode_count)) <= place_method[i])  # upper bound
        model_m.addConstr(quicksum(x[i, k] for k in range(edgenode_count)) >= 1)  # lower bound

    """
    model_m.addConstr(quicksum(x[0, k] for k in range(edgenode_count)) <= 3)
    model_m.addConstr(quicksum(x[1, k] for k in range(edgenode_count)) <= 2)
    model_m.addConstr(quicksum(x[2, k] for k in range(edgenode_count)) <= 3)
    model_m.addConstr(quicksum(x[3, k] for k in range(edgenode_count)) <= 3)
    model_m.addConstr(quicksum(x[4, k] for k in range(edgenode_count)) <= 2)
    model_m.addConstr(quicksum(x[5, k] for k in range(edgenode_count)) <= 2)
    model_m.addConstr(quicksum(x[6, k] for k in range(edgenode_count)) <= 3)
    model_m.addConstr(quicksum(x[7, k] for k in range(edgenode_count)) <= 3)
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
        # model_s.addConstr(makespan[h] <= Th_max)

    # 目标函数
    model_m.setObjective(Cost + q)
    model_m.modelSense = GRB.MINIMIZE
    model_s.setObjective(quicksum(makespan[h] for h in range(task_count)))
    model_s.modelSense = GRB.MINIMIZE

    model_m.Params.OutputFlag = 0  # 1表示开启输出（默认值），0为关闭
    model_s.Params.OutputFlag = 0
    # 循环求解
    L = 0
    loop_count = 0
    start_time = time.time()
    current_best_main_obj = math.inf
    current_best_total_obj = math.inf
    need_rm_constrs = []  # 存储该轮中更新的子问题约束的引用，即可在进入下一轮之前删除该轮添加的约束
    # feasible cut专用赋初值区
    add_which = 0
    step_flag = False
    mserv_place_count = [1] * microservice_count
    need_rm_constrs2 = []
    # feasible cut专用赋初值区 end
    while not (time.time() - start_time > math.inf):
        loop_count += 1
        print('\n' * 3 + '*' * 10 + '!' * 3 + str(current_best_total_obj) + '*' * 10 + '\n' * 3)

        print('\n' * 3 + '*' * 10 + f"第{loop_count}轮循环" + '*' * 10)
        print('\n' + '*' * 10 + "主问题求解" + '*' * 10)
        model_m.optimize()
        print(f"主问题：", end='')
        print_grb_solve_status(model_m.Status)

        def feasible_cut():
            """
            添加feasible cut
            :global add_which = 0: 指示目前想要添加哪个微服务的数量
            :global step_flag = False: 指示当次循环是否是进入下一个总放置数量level前的获取基准线的步骤（见下文代码）
            :mserv_place_count = [1] * microservice_count: 记录该level下，每种微服务放置的数量
            :need_rm_constrs2 = []: 记录上一轮中需要删除的约束
            """
            """
            model_m.addConstr(quicksum(
                1 - x[(i, k)] if x_result[(i, k)] == 1 else x[(i, k)] for i in range(microservice_count)
                for k in range(edgenode_count)
            ) >= 1)
            """
            nonlocal add_which
            nonlocal step_flag
            nonlocal mserv_place_count

            for constr in need_rm_constrs2:
                model_m.remove(constr)  # 删除上一轮中添加的约束
            need_rm_constrs2.clear()

            # 进入下一个level
            if step_flag:
                # 更新放置基准线
                mserv_place_count = [0] * microservice_count
                for i in range(microservice_count):
                    for k in range(edgenode_count):
                        mserv_place_count[i] += x_result[(i, k)]
                # 开始尝试下一个level
                step_flag = False

            if add_which < microservice_count:
                # 尝试各类微服务多放1个
                for i in range(microservice_count):
                    # 添加微服务放置数量约束
                    if i == add_which:
                        place_count = mserv_place_count[i] + 1
                    else:
                        place_count = mserv_place_count[i]
                    constr = model_m.addConstr(quicksum(x[i, k] for k in range(edgenode_count)) == place_count)
                    need_rm_constrs2.append(constr)
                add_which += 1
            else:
                # TODO: 参考theta的值来选最终落点，但也不能一直往同一个一直加
                # 如果都多放过，子问题还是无解，则增加放置总数，进入下一个level
                add_which /= microservice_count
                # 进入下一个level前，需要先获取该level的基准线（未多放1个时，gurobi给的放置情况）
                total_x = sum(x_result.values())  # 统计当前放置总数
                # 这个值是当前level基准线的放置数量+1，而下一个level基准线的放置数量就是+1后的这个值
                constr = model_m.addConstr(quicksum(
                    x[(i, k)] for i in range(microservice_count) for k in range(edgenode_count)) == total_x)
                need_rm_constrs2.append(constr)
                step_flag = True

        if model_m.Status == 2:
            print("主问题目标函数值：", model_m.objVal)

            x_result = {}  # 保存主问题的一个解，传递给子问题
            mserv_numbers = [0] * microservice_count  # 统计每种微服务放置的数量
            for i in range(microservice_count):
                for k in range(edgenode_count):
                    x_result[(i, k)] = 1 if x[(i, k)].X > 0.9 else 0
                    mserv_numbers[i] += x_result[(i, k)]
            print(f"微服务放置数量：{mserv_numbers}")
        else:
            feasible_cut()
            continue
        """
        # 只有当次主问题的cost+q比之前的所有cost+q都小，才给optimal cut
        if model_m.objVal >= current_best_main_obj:
            # feasible cut
            model_m.addConstr(quicksum(
                1 - x[(i, k)] if x_result[(i, k)] == 1 else x[(i, k)] for i in range(microservice_count)
                for k in range(edgenode_count)
            ) >= 1)
            continue
        current_best_main_obj = model_m.objVal
        """
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
        # model_s.Params.TimeLimit = 10  # 设置时间限制（单位：s）
        model_s.optimize()
        print(f"子问题：", end='')
        print_grb_solve_status(model_s.Status)
        if model_s.Status == 2:
            print("子问题目标函数值：", model_s.objVal)
        if model_s.SolCount == 0:
            # 如果子问题松弛前无解，则添加feasible cut
            feasible_cut()
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
        print(f"子问题松弛后：", end='')
        print_grb_solve_status(model_s_relaxed.Status)
        if model_s_relaxed.Status == 2:
            print("子问题松弛后目标函数值：", model_s_relaxed.objVal)

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
                max_y = 0.00001
                max_y_nok = -1
                for k in range(edgenode_count):
                    # 记录所有k中最大的y
                    if y[(i, k, h)].X > max_y:
                        max_y = y[(i, k, h)].X
                        max_y_nok = k
                if max_y_nok == -1:
                    # 说明用户h未请求微服务i，则将所有k设为0
                    for k in range(edgenode_count):
                        y_result[(i, k, h)] = 0
                else:
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
                feasible_cut()
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
            if obj_val < current_best_total_obj:
                current_best_total_obj = obj_val
