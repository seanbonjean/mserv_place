class MicroService:
    """
    微服务
    """

    def __init__(self, mserv_num: int, place_cost: float, request_resource: float,
                 send_datasize: float, memory_demand: float) -> None:
        self.mserv_num = mserv_num  # 微服务编号
        self.place_cost = place_cost  # 放置成本
        self.request_resource = request_resource  # 索取的计算资源
        self.send_datasize = send_datasize  # 发送出去的数据大小（到达下一个微服务/回到用户的数据大小）
        self.memory_demand = memory_demand  # 内存需求(GB)


class User:
    """
    用户
    """

    def __init__(self, request_datasize: float, serv_node: int, mserv_dependency: list) -> None:
        self.request_datasize = request_datasize  # 用户发送出去的请求的数据大小（到达第一个微服务前的数据大小）
        self.serv_node = serv_node  # 服务该用户的边缘节点序号
        self.mserv_dependency = mserv_dependency  # 微服务依赖关系列表

        self.makespan = 0  # 完成时间

    def calc_makespan(self, edge_nodes: list) -> None:
        pass


class EdgeNode:
    """
    边缘节点
    """

    def __init__(self, computing_power: float, memory: float) -> None:
        self.computing_power = computing_power  # 算力
        self.memory = memory  # 内存大小(GB)

        self.memory_used = 0  # 已使用的内存大小
        self.placed_mserv = []  # 放置在该边缘节点的微服务列表

    def place_mserv(self, mserv: MicroService) -> bool:
        """
        放置微服务的动作，返回True才是放置成功
        """
        # 判断内存容量是否满足放置条件
        self.memory_used = sum(map(lambda x: x.memory_demand, self.placed_mserv))
        if self.memory - self.memory_used >= mserv.memory_demand:
            self.placed_mserv.append(mserv)
            self.memory_used += mserv.memory_demand
            return True
        return False
