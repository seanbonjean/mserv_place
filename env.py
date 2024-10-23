import random
from values import *
from strategy import *
from utils import *

random.seed(77)
edgenode_list, microservice_list, user_list, channelrate_dict = load_data("data/test_input.xls")
# random_mserv_place(edgenode_list, microservice_list)
baseline_mserv_place(edgenode_list, microservice_list, user_list, channelrate_dict)
print_mserv_place_state(edgenode_list)
