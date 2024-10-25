import random
from values import *
from strategy import *


def random_algo(data_path: str) -> None:
    random.seed(77)
    edgenode_list, microservice_list, user_list, channelrate_dict = load_data(data_path)
    random_mserv_place(edgenode_list, microservice_list)
    print_mserv_place_state(edgenode_list)
    random_task_routing(edgenode_list, microservice_list, user_list, channelrate_dict)
    print_objective(edgenode_list, user_list)


def baseline_algo(data_path: str) -> None:
    edgenode_list, microservice_list, user_list, channelrate_dict = load_data(data_path)
    baseline_mserv_place(edgenode_list, microservice_list, user_list, channelrate_dict)
    print_mserv_place_state(edgenode_list)
    baseline_task_routing(edgenode_list, microservice_list, user_list, channelrate_dict)
    print_objective(edgenode_list, user_list)
