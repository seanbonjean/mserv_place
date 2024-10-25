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
