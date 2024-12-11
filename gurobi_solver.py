import math
from gurobipy import *

from values import CONSTANTS


def gurobi_solve(edge_nodes: list, mservs: list, users: list, channel: dict) -> None:
    """
    开始gurobi求解
    """
    edgenode_count = len(edge_nodes)
    microservice_count = len(mservs)
    task_count = len(users)
    Th_max = CONSTANTS.MAX_MAKESPAN
    C_max = CONSTANTS.MAX_DEPLOY_COST

    model = Model("microservice-placement")
    model.setParam(GRB.Param.TimeLimit, 5)
    # model.setParam(GRB.Param.TimeLimit, 60 * 60 * 10.0) # 10 hour

    # 添加决策变量
    x = {}  # x(i, k)在第k个节点放置第i个微服务
    y = {}  # y(i, k, h)路径中使用了第k个节点上的第i个微服务，并用h标识不同用户
    # x(i, k)
    for i in range(microservice_count):
        for k in range(edgenode_count):
            x[(i, k)] = model.addVar(vtype=GRB.BINARY, name=f"x_{i}_{k}")
    # y(i, k, h)
    for i in range(microservice_count):
        for k in range(edgenode_count):
            for h in range(task_count):
                y[(i, k, h)] = model.addVar(
                    vtype=GRB.BINARY, name=f"y_{i}_{k}_{h}")

    # 更新模型
    model.update()

    # 添加限制条件
    # 一个f中，每个微服务只能指定一个节点提供
    for user in users:
        h = user.num
        f = user.mserv_dependency
        for i in f:
            model.addConstr(quicksum(y[(i, k, h)] for k in range(edgenode_count)) == 1)
    """
        # 不能去找不需要的微服务
        for i in range(microservice_count):
            if i not in f:
                model.addConstr(quicksum(y[(i, k, h)] for k in range(edgenode_count)) == 0)
    """

    # Th < Th_max
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
        model.addConstr(makespan[h] <= Th_max)

    # C < C_max
    Cost = quicksum(x[(i, k)] * mservs[i].place_cost for i in range(microservice_count) for k in range(edgenode_count))
    model.addConstr(Cost <= C_max)

    # 边缘节点上放置的微服务总内存占用不超过节点内存容量上限
    for k in range(edgenode_count):
        model.addConstr(
            quicksum(x[(i, k)] * mservs[i].memory_demand for i in range(microservice_count)) <= edge_nodes[k].memory
        )

    # y(i, k) <= x(i, k)
    for i in range(microservice_count):
        for k in range(edgenode_count):
            for h in range(task_count):
                model.addConstr(y[(i, k, h)] <= x[(i, k)])

    # 目标函数
    model.setObjective(Cost + quicksum(makespan[h] for h in range(task_count)))

    # 保存LP模型并运行
    model.write("model.rlp")
    model.modelSense = GRB.MINIMIZE
    model.optimize()

    # 输出结果
    if model.status != GRB.OPTIMAL and model.status != GRB.TIME_LIMIT:
        print(f"No optimum solution found. Status: {model.status}")
    else:
        print("Optimal solution found")
        x_result = [(i, k) for i in range(microservice_count) for k in range(edgenode_count) if x[(i, k)].X > 0.9]
        y_result = [(i, k, h) for i in range(microservice_count) for k in range(edgenode_count) for h in
                    range(task_count) if y[(i, k, h)].X > 0.9]
        print(f"Result = {model.objVal:.4f}")
        print(f"GAP = {model.MIPGap:.4f} %%")
        print(f"Time = {model.Runtime:.4f} seg")

        print("x(i, k): ")
        print(x_result)
        print("y(i, k, h): ")
        print(y_result)

        total_makespan = sum(map(lambda x: x.getValue(), makespan))
        print(f"{model.objVal} = {Cost.getValue()} + {total_makespan}")
