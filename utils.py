def print_mserv_place_state(edge_nodes: list) -> None:
    """打印微服务放置状态"""
    print('-' * 10 + "微服务放置状态" + '-' * 10)
    for node in edge_nodes:
        print(f"边缘节点{node.num:>3}放置了{len(node.placed_mservs):>3}个微服务：", end=' ')
        for mserv in node.placed_mservs:
            print(mserv.num, end=' ')
        print()
    print('-' * 30)


def print_objective(edge_nodes: list, users: list) -> None:
    """打印目标函数结果"""
    total_place_cost = sum(map(lambda x: sum(map(lambda y: y.place_cost, x.placed_mservs)), edge_nodes))  # 总放置成本
    total_makespan = sum(map(lambda x: x.makespan, users))  # 总完成时间
    total_objective = total_place_cost + total_makespan  # 总目标函数值
    print('-' * 10 + "目标函数结果" + '-' * 10)
    print(f"总放置成本：{total_place_cost}")
    print(f"总完成时间：{total_makespan}")
    print(f"总目标函数值：{total_objective}")
    print('-' * 30)


def print_grb_solve_status(status: int) -> None:
    """
    打印Gurobi求解状态
    https://www.gurobi.com/documentation/current/refman/optimization_status_codes.html#sec:StatusCodes
    """
    if status == 1:
        print("LOADED")
    elif status == 2:
        print("OPTIMAL")
    elif status == 3:
        print("INFEASIBLE")
    elif status == 4:
        print("INF_OR_UNBD")
    elif status == 5:
        print("UNBOUNDED")
    elif status == 6:
        print("CUTOFF")
    elif status == 7:
        print("ITERATION_LIMIT")
    elif status == 8:
        print("NODE_LIMIT")
    elif status == 9:
        print("TIME_LIMIT")
    elif status == 10:
        print("SOLUTION_LIMIT")
    elif status == 11:
        print("INTERRUPTED")
    elif status == 12:
        print("NUMERIC")
    elif status == 13:
        print("SUBOPTIMAL")
    elif status == 14:
        print("INPROGRESS")
    elif status == 15:
        print("USER_OBJ_LIMIT")
    elif status == 16:
        print("WORK_LIMIT")
    elif status == 17:
        print("MEM_LIMIT")
    else:
        print("未知状态")
