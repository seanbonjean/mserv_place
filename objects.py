class MicroService:
    """
    微服务
    """

    def __init__(self, num: int, place_cost: float, request_resource: float,
                 send_datasize: float, memory_demand: float) -> None:
        self.num = num  # 微服务编号
        self.place_cost = place_cost  # 放置成本
        self.request_resource = request_resource  # 索取的计算资源
        self.send_datasize = send_datasize  # 发送出去的数据大小（到达下一个微服务/回到用户的数据大小）
        self.memory_demand = memory_demand  # 内存需求(GB)


class User:
    """
    用户
    """

    def __init__(self, num: int, request_datasize: float, serv_node: int,
                 mserv_dependency: list) -> None:
        self.num = num  # 用户编号
        self.request_datasize = request_datasize  # 用户发送出去的请求的数据大小（到达第一个微服务前的数据大小）
        self.serv_node = serv_node  # 服务该用户的边缘节点序号
        self.mserv_dependency = mserv_dependency  # 微服务依赖关系列表

        self.makespan = 0  # 完成时间
        self.routing_route = []  # 由边缘节点组成的路径（对应节点上有该用户请求的微服务）
        # !!!注意：routing_route不用包含self.serv_node对应的edge_node

    def calc_makespan(self, mservs: list, channel: dict) -> None:
        """
        计算完成时间
        """
        self.makespan = 0
        prev_node_num = self.serv_node
        if len(self.mserv_dependency) != len(self.routing_route):
            raise Exception("程序有误，微服务依赖数量和路径长度不一致")
        for route_step, node in enumerate(self.routing_route):
            # 如果是第一步，它的数据量应为用户发送的请求的数据量（也就是D_in）
            if route_step == 0:
                data_size = self.request_datasize
            else:
                prev_mserv = mservs[self.mserv_dependency[route_step - 1]]  # 上一个微服务
                data_size = prev_mserv.send_datasize  # 上一个微服务发送过来的数据大小
            # 如果前后节点号一致，说明前后两个微服务在同一节点上，此时不存在传输时延，所以对data_size=0特殊处理
            if prev_node_num == node.num:
                data_size = 0
                channel_rate = 1
            else:
                channel_rate = channel[(prev_node_num, node.num)]
            mserv = mservs[self.mserv_dependency[route_step]]  # 当前微服务
            self.makespan += data_size / channel_rate + mserv.request_resource / node.computing_power
            prev_node_num = node.num
        # 计算d_out
        last_node_num = prev_node_num
        if last_node_num != self.serv_node:
            last_mserv = mservs[self.mserv_dependency[-1]]
            self.makespan += last_mserv.send_datasize / channel[(last_node_num, self.serv_node)]
        else:
            self.makespan += 0

    def print_makespan(self) -> None:
        print(
            f"用户{self.num:<5}"
            f"完成时间：{self.makespan:>10.2f}，"
            + ' ' * 2 +
            f"微服务依赖：{self.mserv_dependency}，"
        )
        print(
            ' ' * 8 +
            f"他的数据从本地节点{self.serv_node:<3}开始，"
            f"走过了路径：{list(map(lambda x: x.num, self.routing_route))}"
        )
        print()


class EdgeNode:
    """
    边缘节点
    """

    def __init__(self, num: int, computing_power: float, memory: float) -> None:
        self.num = num  # 边缘节点编号
        self.computing_power = computing_power  # 算力
        self.memory = memory  # 内存大小(GB)

        self.memory_used = 0  # 已使用的内存大小
        self.placed_mservs = []  # 放置在该边缘节点的微服务列表

    def place_mserv(self, mserv: MicroService) -> bool:
        """
        放置微服务的动作，返回True才是放置成功
        """
        # 判断是否已有相同微服务放置在节点上
        for placed_mserv in self.placed_mservs:
            if mserv.num == placed_mserv.num:
                return False
        # 判断内存容量是否满足放置条件
        self.memory_used = sum(map(lambda x: x.memory_demand, self.placed_mservs))
        if self.memory - self.memory_used >= mserv.memory_demand:
            self.placed_mservs.append(mserv)
            self.memory_used += mserv.memory_demand
            return True
        return False
