from values import load_data
from objects import *
from collections import defaultdict
import copy


# edge_nodes, mservs, users, channelrate_dict = load_data(data_path)

# 统计各边缘节点上有用户请求到的微服务，以及请求该微服务的用户个数（第一维是微服务种类序号、第二维是以边缘节点序号为键、请求用户个数为值的字典）
def count_mserv_user(mservs: list, users: list) -> dict:
    # 统计每个微服务被各边缘节点上多少个用户请求
    mserv_user_count = {mserv_item: defaultdict(int) for mserv_item in map(lambda item: item.num, mservs)}
    for user in users:
        for mserv_item in user.mserv_dependency:
            mserv_user_count[mserv_item][user.serv_node] += 1
    return mserv_user_count


# 统计各个节点上的接收数据总量（第一维是微服务种类序号、第二维是以边缘节点序号为键、数据总量d为值的字典）
def count_mserv_receive_dataflow(mservs: list, users: list) -> dict:
    # 统计每个微服务在各边缘节点接收数据量
    mserv_receive_data_count = {mserv_item: defaultdict(int) for mserv_item in map(lambda item: item.num, mservs)}
    for user in users:
        for mserv_item_index in range(len(user.mserv_dependency)):
            if mserv_item_index == 0:
                mserv_receive_data_count[user.mserv_dependency[mserv_item_index]][
                    user.serv_node] += user.request_datasize
            else:
                mserv_receive_data_count[user.mserv_dependency[mserv_item_index]][user.serv_node] += mservs[
                    user.mserv_dependency[mserv_item_index - 1]].send_datasize
    return mserv_receive_data_count


def getNodeByNodeNum(node_num: int, edge_nodes: list):
    for node in edge_nodes:
        if node.num == node_num:
            return node
    return None


def place_mserv(upper_bound: dict, ksi: float, edge_nodes: list, mservs: list, users: list, channelrate_dict: list,
                channel_connectivity: dict, mserv_receive_data_count: dict, mserv_user_count: dict) -> dict:
    print("mserv placed start")
    mserv_process_info = []  # 元素为 dict，包含信息：{微服务序号 num、放置节点列表 node_list、枢纽节点列表 hub_nodes、节点群列表 node_group、节点群被分配的微服务数目 allocate_mserv_to_node_group_result}
    mservs_count = len(mservs)
    edge_nodes_count = len(edge_nodes)

    # 对于每个微服务
    for mserv in mservs:
        # 1. 找到“节点上有用户请求该微服务”的所有节点，根据一个通信速率阈值ξ，将大于ξ的链路相连，组成若干个节点群
        node_group = []  # 节点群列表，其中的元素也是列表，一个这样的列表为划分的一个节点群，列表中的元素为节点序号
        print(mserv_user_count[mserv.num])
        mserv_exits_nodes = list(mserv_user_count[mserv.num].keys())
        print(f"------ now mserv num:{mserv.num} ------")

        def allocate_node_to_group(around_node: int, single_node_group: list):
            i = 0
            while i < len(mserv_exits_nodes):
                if mserv_exits_nodes[i] == around_node:
                    i += 1
                    continue
                unallocated_node = mserv_exits_nodes[i]
                if channelrate_dict[(around_node, unallocated_node)] > ksi:
                    single_node_group.append(unallocated_node)
                    mserv_exits_nodes.pop(i)  # 删除当前节点
                    single_node_group = allocate_node_to_group(unallocated_node, single_node_group)
                else:
                    i += 1  # 只有在不删除元素时才递增索引
            if len(single_node_group) == 0:  # 即使递归结束后仍然没有一个元素与其通信阈值大于ksi，则自成一个节点群
                single_node_group.append(around_node)
                mserv_exits_nodes.remove(around_node)
            return single_node_group

        while len(mserv_exits_nodes) > 0:
            # 只剩最后一个节点的情况，直接加入节点群
            if len(mserv_exits_nodes) == 1:
                node_group.append(mserv_exits_nodes)
                break
            single_node_group = allocate_node_to_group(mserv_exits_nodes[0], [])
            node_group.append(single_node_group)
        print("node group (节点群列表，其中单个元素为组成对应节点群的节点的列表): ", node_group)
        original_node_group = copy.deepcopy(node_group)
        # 2. 对每个节点，再向自身所处节点群中延伸一个与自身通信速率最快的节点，该节点上无需有用户请求该微服务
        # -------------------------------
        # hub_nodes = []
        # for single_node_group in node_group:
        #     single_hub_node = []
        #     for node in single_node_group:
        #         max_rate = 0
        #         hub_node = None
        #         # 只有一个节点的情况
        #         if len(single_node_group) == 1:
        #             single_hub_node.append(node)
        #             break
        #         for node2 in single_node_group:
        #             if node==node2:
        #                 continue
        #             if channelrate_dict[(node, node2)] > max_rate:
        #                 max_rate = channelrate_dict[(node, node2)]
        #                 hub_node = node2
        #         if hub_node not in single_hub_node:
        #             single_hub_node.append(hub_node)
        #     hub_nodes.append(single_hub_node)
        # print("hub node 枢纽节点列表: ",hub_nodes) # 枢纽节点
        # ------------------------------- 上面是旧代码
        # 先对每个节点按“连通权值”排序（连通权值定义为：自身节点到其他所有节点的通信速率之和）
        connected_weight_order_map = defaultdict(int)
        for i in range(edge_nodes_count):
            for j in range(edge_nodes_count):
                if i == j:
                    continue
                connected_weight_order_map[edge_nodes[i].num] += channelrate_dict[
                    (edge_nodes[i].num, edge_nodes[j].num)]
        # sorted_makespan_map = dict(sorted(makespan_map.items(), key=lambda item: item[1])[:allocate_mserv_to_node_group_result[node_group_index]])
        connected_weight_order_map = dict(connected_weight_order_map)
        connected_weight_order_map = dict(
            sorted(connected_weight_order_map.items(), key=lambda item: item[1], reverse=True))
        print("connected_weight_order_map 连通权值及其降序排列结果 (format: [{节点序号: 连通权值}, ...]): ",
              connected_weight_order_map)
        # 然后，对每个节点群，先寻找与群内节点直连条数>=3的“无请求节点”
        hub_nodes = []
        for single_node_group in node_group:
            if len(single_node_group) < 3:  # 节点群内节点数小于3,该节点群不可能存在直连条数>=3的“无请求节点”，直接跳过
                continue
            for connected_node_index in range(edge_nodes_count):
                # 选出不在节点群中的节点
                if connected_node_index in single_node_group:
                    continue
                connectivity_node_list = []
                for single_group_node in single_node_group:
                    if channel_connectivity[(single_group_node, connected_node_index)] == 1:
                        connectivity_node_list.append(single_group_node)
                if len(connectivity_node_list) >= 3:
                    # 计算这些“无请求节点”的一组Δa=放枢纽 – 放va（va的个数就是第1步时节点群内节点数，即有用户请求节点的节点数），计算Δa的顺序按va的“连通性”从小到大排序的顺序，只要计算到有某个Δa<0就把该无用户请求的节点放入这个节点群，如果遍历后全都>0就不放入
                    # 公式左边 放枢纽 sigma 计算（似乎只需要计算一次就行）
                    hub_node_tran_time = sum([mserv_receive_data_count[mserv.num][other_node] / channelrate_dict[
                        (connected_node_index, other_node)] for other_node in connectivity_node_list])
                    for other_node in connectivity_node_list:
                        delta = hub_node_tran_time - sum([mserv_receive_data_count[mserv.num][group_node] /
                                                          channelrate_dict[(group_node, other_node)] for group_node in
                                                          connectivity_node_list if group_node != other_node])
                        if delta < 0:
                            single_node_group.append(connected_node_index)
                            hub_nodes.append(connected_node_index)
                            break
        print("hub node 枢纽节点列表: ", hub_nodes)  # 枢纽节点
        print("node group after add hub nodes (加入枢纽节点后的节点群列表，其中单个元素为组成对应节点群的节点的列表): ",
              node_group)

        # 3. 统计各节点群中，、求该微服务的用户个数，计算各节点群请求百分比，将upper bound中的放置预算按比例分配给各节点群
        # 各节点群中微服务的用户个数
        # 按比例分配微服务，差值调整
        def allocate_devices_with_adjustment(num_devices, node_percentages):
            allocations = {}
            total_allocated = 0
            for node, percentage in node_percentages.items():
                allocated_devices = round(num_devices * percentage)
                allocations[node] = allocated_devices
                total_allocated += allocated_devices
            diff = num_devices - total_allocated
            if diff > 0:
                sorted_nodes = sorted(node_percentages.items(), key=lambda item: item[1], reverse=True)
                for node, _ in sorted_nodes:
                    if diff == 0:
                        break
                    allocations[node] += 1
                    diff -= 1
            elif diff < 0:
                sorted_nodes = sorted(node_percentages.items(), key=lambda item: item[1])
                for node, _ in sorted_nodes:
                    if diff == 0:
                        break
                    allocations[node] -= 1
                    diff += 1
            return allocations

        total_user_map = {}
        total_user_count = 0
        for node_group_index, single_node_group in enumerate(node_group):
            node_total_user_count = sum([mserv_user_count[mserv.num][node] for node in single_node_group])
            total_user_count += node_total_user_count
            total_user_map[node_group_index] = node_total_user_count
        for node_group_index in total_user_map.keys():
            total_user_map[node_group_index] /= total_user_count
        print("请求微服务的节点的用户数量: ", total_user_map)
        allocate_mserv_to_node_group_result = allocate_devices_with_adjustment(upper_bound[mserv.num], total_user_map)
        print("差值调整后分配到各节点的微服务数量: ", allocate_mserv_to_node_group_result)
        # 4. 如果预算数等于节点数就直接放满，否则对各节点群，先依次往每个节点上只放置一个微服务，计算节点群内的对于该微服务的总makespan（makespan=传输时延+计算时延），然后各节点就有一个【makespan参考值】，按该值从小到大排序，选最小的那几个节点放置预算个
        allocate_mserv_result = []
        for node_group_index, single_node_group in enumerate(node_group):
            # 如果预算数等于节点数就直接放满
            if allocate_mserv_to_node_group_result[node_group_index] >= len(single_node_group):
                for node in single_node_group:
                    allocate_mserv_result.append(node)
                continue
            else:
                # 依次往每个节点上只放置一个微服务，计算节点群内的对于该微服务的总makespan
                makespan_map = {}
                for node in single_node_group:
                    makespan_map[node] = 0
                    for node2 in single_node_group:
                        if node == node2:
                            continue
                        # makespan=传输时延+计算时延
                        trans_time = mserv.send_datasize / channelrate_dict[(node, node2)]
                        compute_time = mserv.request_resource / next(
                            (nodeItem for nodeItem in edge_nodes if nodeItem.num == node), None).computing_power
                        makespan_map[node] += (trans_time + compute_time)
                print("makespan_map (format: [{节点序号: makespan值}, ...]): ", makespan_map)
                # 选最小的那几个节点放置预算个
                sorted_makespan_map = dict(sorted(makespan_map.items(), key=lambda item: item[1])[
                                           :allocate_mserv_to_node_group_result[
                                               node_group_index]])  # 按照value升序排序并取前allocate_mserv_to_node_group_result[node_group_index]个值
                print("排序并筛选出的前n项需要放置服务的 makespan_map 键值 (format: [{节点序号: makespan值}, ...]): ",
                      sorted_makespan_map)
                # 选取前allocate_mserv_to_node_group_result[node_group_index]个节点(即放置被分配预算个数)
                allocate_mserv_result.append(list(sorted_makespan_map.keys()))
        print("该微服务被分配到的节点列表: ", allocate_mserv_result)
        mserv_process_info.append({
            'num 微服务序号': mserv.num,  # 微服务序号
            'devide_node_group 被划分的节点群': node_group,  # 节点群列表
            'mserv_place_node_list 微服务放置节点': allocate_mserv_result,  # 放置微服务的节点列表
            'allocate_mserv_to_node_group_num_result 被分配到给各节点群的微服务数量': allocate_mserv_to_node_group_result,
            # 节点群被分配的微服务数目
            'hub_nodes 枢纽节点列表': hub_nodes,  # 枢纽节点列表
            'original_node_group 原始未加入枢纽节点前的节点群': original_node_group  # 原始节点群
        })
    print("mserv placed end 放置微服务结束，结果如下: ")
    print(mserv_process_info)
    return mserv_process_info


if __name__ == '__main__':
    data_path = "data/input1208.xls"
    edge_nodes, mservs, users, channelrate_dict, channel_connectivity = load_data(data_path)
    mserv_user_count = count_mserv_user(mservs, users)
    print("mserv_user_count (微服务用户数统计。二维字典，第一维是微服务种类序号，第二维是以边缘节点序号): ", {k: (
        dict(v) if isinstance(v, defaultdict) else {kk: (dict(vv) if isinstance(vv, defaultdict) else vv) for kk, vv in
                                                    v.items()} if isinstance(v, dict) else v) for k, v in
                                                                                                           mserv_user_count.items()})
    mserv_receive_data_count = count_mserv_receive_dataflow(mservs, users)
    print("mserv_receive_data_count (微服务数据接收量统计。二维字典，第一维是微服务种类序号，第二维是以边缘节点序号): ", {
        k: (dict(v) if isinstance(v, defaultdict) else {kk: (dict(vv) if isinstance(vv, defaultdict) else vv) for kk, vv
                                                        in v.items()} if isinstance(v, dict) else v) for k, v in
        mserv_receive_data_count.items()})
    upper_bound = {0: 2, 1: 2, 2: 4, 3: 3, 4: 2, 5: 3, 6: 3, 7: 3, 8: 5, 9: 3, 10: 5, 11: 4, 12: 2, 13: 2, 14: 4}
    ksi = 0.6  # 通信速率阈值 ξ
    place_mserv(upper_bound=upper_bound, ksi=ksi, edge_nodes=edge_nodes, mservs=mservs, users=users,
                channelrate_dict=channelrate_dict, channel_connectivity=channel_connectivity,
                mserv_receive_data_count=mserv_receive_data_count, mserv_user_count=mserv_user_count)
