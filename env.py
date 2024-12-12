import random
from stdout_redirect import RedirectStdoutToFileAndConsole
from values import load_data
from strategy import *
from gurobi_solver import gurobi_solve
from benders_solver import benders_solve
from greedy_combine import greedy_combine


def random_algo(data_path: str) -> None:
    random.seed(77)
    edgenode_list, microservice_list, user_list, channelrate_dict, channel_connectivity = load_data(data_path)
    random_mserv_place(edgenode_list, microservice_list)
    print_mserv_place_state(edgenode_list)
    random_task_routing(edgenode_list, microservice_list, user_list, channelrate_dict)
    print_objective(edgenode_list, user_list)


def baseline_algo(data_path: str) -> None:
    edgenode_list, microservice_list, user_list, channelrate_dict, channel_connectivity = load_data(data_path)
    baseline_mserv_place(edgenode_list, microservice_list, user_list, channelrate_dict)
    print_mserv_place_state(edgenode_list)
    baseline_task_routing(edgenode_list, microservice_list, user_list, channelrate_dict)
    print_objective(edgenode_list, user_list)


def gurobi_algo(data_path: str) -> None:
    redirector = RedirectStdoutToFileAndConsole("output/gurobi.txt")
    redirector.start()
    edgenode_list, microservice_list, user_list, channelrate_dict, channel_connectivity = load_data(data_path)
    gurobi_solve(edgenode_list, microservice_list, user_list, channelrate_dict)
    redirector.stop()


def benders_algo(data_path: str) -> None:
    edgenode_list, microservice_list, user_list, channelrate_dict, channel_connectivity = load_data(data_path)
    benders_solve(edgenode_list, microservice_list, user_list, channelrate_dict)


def combine_algo(data_path: str) -> None:
    redirector = RedirectStdoutToFileAndConsole("output/combine.txt")
    redirector.start()
    edgenode_list, microservice_list, user_list, channelrate_dict, channel_connectivity = load_data(data_path)
    greedy_combine(edgenode_list, microservice_list, user_list, channelrate_dict, channel_connectivity, 0.5)
    redirector.stop()
