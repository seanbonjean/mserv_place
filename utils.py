def print_mserv_place_state(edge_nodes: list) -> None:
    """打印微服务放置状态"""
    print('-' * 10 + "微服务放置状态" + '-' * 10)
    for node in edge_nodes:
        print(f"边缘节点{node.num:>3}放置了{len(node.placed_mservs):>3}个微服务：", end=' ')
        for mserv in node.placed_mservs:
            print(mserv.num, end=' ')
        print()
    print('-' * 30)
