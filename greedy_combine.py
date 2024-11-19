from values import CONSTANTS


def greedy_combine(edge_nodes: list, mservs: list, users: list, channel: dict) -> None:
    edgenode_count = len(edge_nodes)
    microservice_count = len(mservs)

    # upper bound 根据总cost的限制来计算，当其他微服务都只放置一个的情况下，该微服务最多能放置的个数即为该微服务的上界。对所有微服务同样处理即得完整的上界
    upper_bound = []
    for mserv in mservs:
        remain_cost = CONSTANTS.MAX_DEPLOY_COST - sum(map(lambda x: x.place_cost, mservs))  # 其他微服务都放一个，剩余可用的cost
        max_placeNum = remain_cost // mserv.place_cost  # 剩余cost最多可以放这个微服务的个数（向下取整）
        upper_bound.append(max_placeNum)

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
