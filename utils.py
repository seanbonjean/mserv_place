def print_mserv_place_state(edge_nodes: list) -> None:
    """打印微服务放置状态"""
    for node_num, node in enumerate(edge_nodes):
        print(f"边缘节点{node_num}放置了{len(node.placed_mserv)}个微服务：", end=' ')
        for mserv in node.placed_mserv:
            print(mserv.mserv_num, end=' ')
        print()
