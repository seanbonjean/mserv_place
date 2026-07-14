# 目的：使用 Gurobi 评估 KNN+RL 产生的节点分组对微服务部署搜索空间和求解结果的影响。
# 本程序不运行 SoCL 的候选节点选择、预部署或合并算法；RL 分组只约束部署变量 x，
# 具体部署节点和用户路由仍由 Gurobi 在原成本、时延及内存约束下统一优化。
# 特别说明：本程序没有引入 candidate/hub 节点，也不执行 candidate-node election；
# 每个微服务的候选部署节点仅来自其请求节点与 RL 分组的交集 G'(m_i)。
# Gurobi 直接求解最终部署，因此不要求每个活跃组至少保留一个实例；分组 quota
# 仅作为实例数量上界，每个有请求的微服务只需在所有组中全局至少部署一个实例。

import argparse
import json
import math
import os
import sys
from collections import defaultdict

from gurobipy import GRB, LinExpr, Model, quicksum


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(BASE_DIR)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from values import CONSTANTS, load_data  # noqa: E402


DEFAULT_DATA_PATH = os.path.join(PROJECT_ROOT, "data", "30e_15m_50u.xls")
DEFAULT_GROUP_PATH = os.path.join(BASE_DIR, "SA-0713-1529", "result.json")


def load_and_validate_groups(group_path: str, node_ids: set[int]) -> list[list[int]]:
    """读取全局 RL 分组，并确认其恰好构成全部边缘节点的一个划分。"""
    with open(group_path, "r", encoding="utf-8") as file:
        raw_groups = json.load(file)

    if not isinstance(raw_groups, list) or not raw_groups:
        raise ValueError("RL group file must contain a non-empty list of node groups.")

    groups = []
    seen_nodes = set()
    duplicate_nodes = set()
    for group_index, raw_group in enumerate(raw_groups):
        if not isinstance(raw_group, list) or not raw_group:
            raise ValueError(f"RL group {group_index} must be a non-empty list.")

        group = []
        for node_num in raw_group:
            if isinstance(node_num, bool) or not isinstance(node_num, int):
                raise ValueError(
                    f"Node id {node_num!r} in RL group {group_index} is not an integer."
                )
            if node_num in seen_nodes:
                duplicate_nodes.add(node_num)
            seen_nodes.add(node_num)
            group.append(node_num)
        groups.append(sorted(group))

    missing_nodes = sorted(node_ids - seen_nodes)
    unknown_nodes = sorted(seen_nodes - node_ids)
    if duplicate_nodes or missing_nodes or unknown_nodes:
        raise ValueError(
            "Invalid RL partition: "
            f"duplicates={sorted(duplicate_nodes)}, "
            f"missing={missing_nodes}, unknown={unknown_nodes}."
        )
    return groups


def count_requests_by_service_and_node(mservs: list, users: list) -> dict[int, dict[int, int]]:
    """统计每种微服务在各用户接入节点上的请求数量。"""
    service_ids = {mserv.num for mserv in mservs}
    request_counts = {service_id: defaultdict(int) for service_id in service_ids}

    for user in users:
        for service_id in user.mserv_dependency:
            if service_id not in service_ids:
                raise ValueError(
                    f"User {user.num} refers to unknown microservice {service_id}."
                )
            request_counts[service_id][user.serv_node] += 1

    return {service_id: dict(counts) for service_id, counts in request_counts.items()}


def calculate_paper_upper_bounds(
    mservs: list,
    request_counts: dict[int, dict[int, int]],
    max_deploy_cost: float,
) -> dict[int, int]:
    """
    按论文计算每种微服务的实例数量上界。

    对 m_i，先为其他每种微服务各保留一个实例的成本，再计算剩余预算最多
    可容纳多少个 m_i，最后使用 |V(m_i)| 截断。
    """
    total_one_copy_cost = sum(mserv.place_cost for mserv in mservs)
    upper_bounds = {}

    for mserv in mservs:
        if mserv.place_cost <= 0:
            raise ValueError(
                f"Microservice {mserv.num} has non-positive placement cost "
                f"{mserv.place_cost}."
            )

        other_services_min_cost = total_one_copy_cost - mserv.place_cost
        available_budget = max_deploy_cost - other_services_min_cost
        budget_bound = max(
            0,
            math.floor(available_budget / mserv.place_cost + 1e-12),
        )
        request_node_count = len(request_counts[mserv.num])
        upper_bounds[mserv.num] = min(request_node_count, budget_bound)

    return upper_bounds


def allocate_group_quotas(
    service_id: int,
    total_upper_bound: int,
    service_groups: list[dict],
) -> list[int]:
    """
    按组请求比例将微服务总 upper bound 整数化分配到各活跃组。

    quota 只作为 group upper bound，可以为 0，不构成任何组下界。quota 不超过
    组内 G'(m_i) 节点数，所有组 quota 之和等于该微服务的总 upper bound。
    """
    group_count = len(service_groups)
    if group_count == 0:
        if total_upper_bound != 0:
            raise ValueError(
                f"Microservice {service_id} has no active group but upper bound is "
                f"{total_upper_bound}."
            )
        return []

    capacities = [len(group_info["nodes"]) for group_info in service_groups]
    demands = [group_info["request_count"] for group_info in service_groups]

    if total_upper_bound > sum(capacities):
        raise ValueError(
            f"Microservice {service_id}: upper bound {total_upper_bound} exceeds "
            f"the total G'(m_i) candidate capacity {sum(capacities)}."
        )

    total_demand = sum(demands)
    if total_demand <= 0:
        raise ValueError(
            f"Microservice {service_id} has active groups but no recorded requests."
        )

    ideal_quotas = [
        total_upper_bound * demand / total_demand
        for demand in demands
    ]
    quotas = [0] * group_count
    remaining = total_upper_bound

    while remaining > 0:
        candidates = [
            group_index
            for group_index in range(group_count)
            if quotas[group_index] < capacities[group_index]
        ]
        if not candidates:
            raise ValueError(
                f"Microservice {service_id}: no group has capacity for the "
                f"remaining {remaining} quota."
            )

        selected_group = max(
            candidates,
            key=lambda group_index: (
                ideal_quotas[group_index] - quotas[group_index],
                demands[group_index],
                -service_groups[group_index]["global_group_index"],
            ),
        )
        quotas[selected_group] += 1
        remaining -= 1

    return quotas


def build_service_partition_info(
    mservs: list,
    global_groups: list[list[int]],
    request_counts: dict[int, dict[int, int]],
    upper_bounds: dict[int, int],
) -> dict[int, dict]:
    """将全局 RL 分组转换为每个微服务的 G'(m_i) 分组和部署 quota。"""
    partition_info = {}

    for mserv in mservs:
        service_id = mserv.num
        request_nodes = set(request_counts[service_id])
        service_groups = []

        for global_group_index, global_group in enumerate(global_groups):
            group_nodes = sorted(request_nodes.intersection(global_group))
            if not group_nodes:
                continue
            group_request_count = sum(
                request_counts[service_id][node_num]
                for node_num in group_nodes
            )
            service_groups.append(
                {
                    "global_group_index": global_group_index,
                    "nodes": group_nodes,
                    "request_count": group_request_count,
                }
            )

        quotas = allocate_group_quotas(
            service_id,
            upper_bounds[service_id],
            service_groups,
        )
        for group_info, quota in zip(service_groups, quotas):
            group_info["quota"] = quota

        partition_info[service_id] = {
            "request_nodes": sorted(request_nodes),
            "upper_bound": upper_bounds[service_id],
            "groups": service_groups,
        }

    return partition_info


def inverse_channel_rate(channel: dict, source: int, target: int) -> float:
    """返回传输速率的倒数；同节点传输时间为零。"""
    if source == target:
        return 0.0
    rate = channel.get((source, target))
    if rate is None or rate <= 0:
        raise ValueError(
            f"Invalid channel rate from node {source} to node {target}: {rate}."
        )
    return 1.0 / rate


def status_name(status: int) -> str:
    """将常见 Gurobi 状态码转换为便于阅读的名称。"""
    names = {
        GRB.LOADED: "LOADED",
        GRB.OPTIMAL: "OPTIMAL",
        GRB.INFEASIBLE: "INFEASIBLE",
        GRB.INF_OR_UNBD: "INF_OR_UNBD",
        GRB.UNBOUNDED: "UNBOUNDED",
        GRB.CUTOFF: "CUTOFF",
        GRB.ITERATION_LIMIT: "ITERATION_LIMIT",
        GRB.NODE_LIMIT: "NODE_LIMIT",
        GRB.TIME_LIMIT: "TIME_LIMIT",
        GRB.SOLUTION_LIMIT: "SOLUTION_LIMIT",
        GRB.INTERRUPTED: "INTERRUPTED",
        GRB.NUMERIC: "NUMERIC",
        GRB.SUBOPTIMAL: "SUBOPTIMAL",
        GRB.INPROGRESS: "INPROGRESS",
        GRB.USER_OBJ_LIMIT: "USER_OBJ_LIMIT",
    }
    return names.get(status, f"STATUS_{status}")


def print_partition_summary(partition_info: dict[int, dict]) -> None:
    """打印每个微服务的 G'(m_i)、upper bound 和组 quota。"""
    print("\nRL partition converted for each microservice:")
    for service_id in sorted(partition_info):
        info = partition_info[service_id]
        print(
            f"  microservice {service_id}: request_nodes={info['request_nodes']}, "
            f"paper_upper_bound={info['upper_bound']}"
        )
        for group_info in info["groups"]:
            print(
                "    "
                f"global_group={group_info['global_group_index']}, "
                f"nodes={group_info['nodes']}, "
                f"requests={group_info['request_count']}, "
                f"quota={group_info['quota']}"
            )


def gurobi_evaluate_rl_groups(
    data_path: str,
    group_path: str,
    time_limit: float | None = None,
    model_output_path: str | None = None,
) -> dict:
    """在 RL 分组部署约束下，用 Gurobi 联合求解微服务部署和全局路由。"""
    edge_nodes, mservs, users, channel, _ = load_data(data_path)

    node_by_num = {node.num: node for node in edge_nodes}
    mserv_by_num = {mserv.num: mserv for mserv in mservs}
    user_ids = [user.num for user in users]
    if len(node_by_num) != len(edge_nodes):
        raise ValueError("Duplicate edge-node ids were found in the input data.")
    if len(mserv_by_num) != len(mservs):
        raise ValueError("Duplicate microservice ids were found in the input data.")
    if len(set(user_ids)) != len(user_ids):
        raise ValueError("Duplicate user ids were found in the input data.")

    node_ids = set(node_by_num)
    for user in users:
        if user.serv_node not in node_ids:
            raise ValueError(
                f"User {user.num} is attached to unknown node {user.serv_node}."
            )

    max_deploy_cost = CONSTANTS.MAX_DEPLOY_COST
    max_makespan = CONSTANTS.MAX_MAKESPAN
    global_groups = load_and_validate_groups(group_path, node_ids)
    request_counts = count_requests_by_service_and_node(mservs, users)
    upper_bounds = calculate_paper_upper_bounds(
        mservs,
        request_counts,
        max_deploy_cost,
    )
    partition_info = build_service_partition_info(
        mservs,
        global_groups,
        request_counts,
        upper_bounds,
    )

    print(f"Data path: {os.path.abspath(data_path)}")
    print(f"RL group path: {os.path.abspath(group_path)}")
    print(f"Global RL groups: {global_groups}")
    print_partition_summary(partition_info)

    candidate_nodes = {
        service_id: info["request_nodes"]
        for service_id, info in partition_info.items()
    }

    model = Model("rl-grouped-microservice-placement")
    model.Params.NonConvex = 2
    if time_limit is not None:
        if time_limit <= 0:
            raise ValueError("time_limit must be positive.")
        model.Params.TimeLimit = time_limit

    x = {}
    for service_id, nodes in candidate_nodes.items():
        for node_num in nodes:
            x[(service_id, node_num)] = model.addVar(
                vtype=GRB.BINARY,
                name=f"x_{service_id}_{node_num}",
            )

    y = {}
    for user in users:
        for service_id in user.mserv_dependency:
            for node_num in candidate_nodes[service_id]:
                key = (service_id, node_num, user.num)
                if key not in y:
                    y[key] = model.addVar(
                        vtype=GRB.BINARY,
                        name=f"y_{service_id}_{node_num}_{user.num}",
                    )

    model.update()

    # 每个用户请求的微服务从所有候选分组中的已部署实例里全局选择一个；
    # 这里不把路由限制在用户接入节点所属的 RL 分组内。
    for user in users:
        for service_id in dict.fromkeys(user.mserv_dependency):
            model.addConstr(
                quicksum(
                    y[(service_id, node_num, user.num)]
                    for node_num in candidate_nodes[service_id]
                )
                == 1,
                name=f"assign_{user.num}_{service_id}",
            )

    # RL 分组只提供部署数量上界：每组最多部署 quota 个实例，但允许部署 0 个；
    # 各组 quota 总和为论文 upper bound。最终部署只保留微服务全局下界 1。
    for service_id, info in partition_info.items():
        if not info["groups"]:
            continue
        for group_info in info["groups"]:
            group_index = group_info["global_group_index"]
            group_instance_count = quicksum(
                x[(service_id, node_num)]
                for node_num in group_info["nodes"]
            )
            model.addConstr(
                group_instance_count <= group_info["quota"],
                name=f"group_max_{service_id}_{group_index}",
            )

        service_instance_count = quicksum(
            x[(service_id, node_num)]
            for node_num in candidate_nodes[service_id]
        )
        model.addConstr(
            service_instance_count >= 1,
            name=f"service_min_{service_id}",
        )
        model.addConstr(
            service_instance_count <= info["upper_bound"],
            name=f"service_upper_bound_{service_id}",
        )

    for (service_id, node_num, user_num), y_var in y.items():
        model.addConstr(
            y_var <= x[(service_id, node_num)],
            name=f"serve_only_if_deployed_{service_id}_{node_num}_{user_num}",
        )

    makespan = {}
    for user in users:
        dependency = user.mserv_dependency
        if not dependency:
            makespan[user.num] = LinExpr(0.0)
            continue

        first_service = dependency[0]
        first_stage = quicksum(
            y[(first_service, node_num, user.num)]
            * (
                user.request_datasize
                * inverse_channel_rate(channel, user.serv_node, node_num)
                + mserv_by_num[first_service].request_resource
                / node_by_num[node_num].computing_power
            )
            for node_num in candidate_nodes[first_service]
        )

        intermediate_stages = quicksum(
            y[(current_service, current_node, user.num)]
            * (
                mserv_by_num[previous_service].send_datasize
                * quicksum(
                    y[(previous_service, previous_node, user.num)]
                    * inverse_channel_rate(channel, previous_node, current_node)
                    for previous_node in candidate_nodes[previous_service]
                )
                + mserv_by_num[current_service].request_resource
                / node_by_num[current_node].computing_power
            )
            for previous_service, current_service in zip(
                dependency,
                dependency[1:],
            )
            for current_node in candidate_nodes[current_service]
        )

        last_service = dependency[-1]
        output_stage = quicksum(
            y[(last_service, node_num, user.num)]
            * mserv_by_num[last_service].send_datasize
            * inverse_channel_rate(channel, node_num, user.serv_node)
            for node_num in candidate_nodes[last_service]
        )

        makespan[user.num] = first_stage + intermediate_stages + output_stage
        model.addConstr(
            makespan[user.num] <= max_makespan,
            name=f"max_makespan_{user.num}",
        )

    deployment_cost = quicksum(
        x_var * mserv_by_num[service_id].place_cost
        for (service_id, _), x_var in x.items()
    )
    model.addConstr(
        deployment_cost <= max_deploy_cost,
        name="max_deployment_cost",
    )

    for node_num, node in node_by_num.items():
        model.addConstr(
            quicksum(
                x[(service_id, node_num)]
                * mserv_by_num[service_id].memory_demand
                for service_id in candidate_nodes
                if (service_id, node_num) in x
            )
            <= node.memory,
            name=f"node_memory_{node_num}",
        )

    total_makespan_expression = quicksum(makespan.values())
    model.setObjective(
        deployment_cost + total_makespan_expression,
        GRB.MINIMIZE,
    )

    if model_output_path:
        output_path = os.path.abspath(model_output_path)
        output_dir = os.path.dirname(output_path)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
        model.write(output_path)

    model.optimize()

    print("\n========== Gurobi RL-group evaluation finished ==========")
    print(f"Status = {status_name(model.Status)} ({model.Status})")
    print(f"Runtime = {model.Runtime:.6f} sec")
    print(f"Branch-and-bound nodes = {model.NodeCount:.0f}")
    print(f"Solutions found = {model.SolCount}")
    print(f"Variables = {model.NumVars}")
    print(f"Linear constraints = {model.NumConstrs}")
    print(f"Quadratic constraints = {model.NumQConstrs}")
    print(f"MAX_DEPLOY_COST = {max_deploy_cost}")
    print(f"MAX_MAKESPAN = {max_makespan}")

    result = {
        "status": model.Status,
        "status_name": status_name(model.Status),
        "runtime": model.Runtime,
        "node_count": model.NodeCount,
        "solution_count": model.SolCount,
        "max_deploy_cost": max_deploy_cost,
        "max_makespan": max_makespan,
    }

    if model.SolCount == 0:
        print("No feasible incumbent solution was found.")
        return result

    x_result = sorted(
        (service_id, node_num)
        for (service_id, node_num), variable in x.items()
        if variable.X > 0.9
    )
    y_result = sorted(
        (service_id, node_num, user_num)
        for (service_id, node_num, user_num), variable in y.items()
        if variable.X > 0.9
    )
    deployment_cost_value = deployment_cost.getValue()
    makespan_values = {
        user_num: expression.getValue()
        for user_num, expression in makespan.items()
    }
    total_makespan = sum(makespan_values.values())

    print(f"Objective = {model.ObjVal:.6f}")
    print(f"Best bound = {model.ObjBound:.6f}")
    print(f"MIP gap = {model.MIPGap * 100:.6f}%")
    print(f"Deployment cost = {deployment_cost_value:.6f}")
    print(f"Total makespan = {total_makespan:.6f}")
    print(f"Maximum user makespan = {max(makespan_values.values(), default=0.0):.6f}")
    print(f"x(i, k) = {x_result}")
    print(f"y(i, k, h) = {y_result}")

    print("\nSelected deployment nodes in each RL group:")
    selected_by_service = defaultdict(set)
    for service_id, node_num in x_result:
        selected_by_service[service_id].add(node_num)
    for service_id in sorted(partition_info):
        for group_info in partition_info[service_id]["groups"]:
            selected_nodes = sorted(
                selected_by_service[service_id].intersection(group_info["nodes"])
            )
            print(
                f"  microservice={service_id}, "
                f"global_group={group_info['global_group_index']}, "
                f"selected={selected_nodes}, quota={group_info['quota']}"
            )

    result.update(
        {
            "objective": model.ObjVal,
            "best_bound": model.ObjBound,
            "mip_gap": model.MIPGap,
            "deployment_cost": deployment_cost_value,
            "total_makespan": total_makespan,
            "x_result": x_result,
            "y_result": y_result,
        }
    )
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate a KNN+RL node partition by solving the grouped "
            "microservice-placement problem with Gurobi."
        )
    )
    parser.add_argument(
        "--data",
        default=DEFAULT_DATA_PATH,
        help="Path to the input .xls data file.",
    )
    parser.add_argument(
        "--groups",
        default=DEFAULT_GROUP_PATH,
        help="Path to the KNN+RL result.json file.",
    )
    parser.add_argument(
        "--time-limit",
        type=float,
        default=None,
        help="Optional Gurobi time limit in seconds.",
    )
    parser.add_argument(
        "--model-output",
        default=None,
        help="Optional path for writing the generated Gurobi model.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    arguments = parse_args()
    gurobi_evaluate_rl_groups(
        data_path=arguments.data,
        group_path=arguments.groups,
        time_limit=arguments.time_limit,
        model_output_path=arguments.model_output,
    )
