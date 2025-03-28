from env import *

DATA_PATH = "data/10e_22m_160.xls"

# random_algo(DATA_PATH)
# baseline_algo(DATA_PATH)
# gurobi_algo(DATA_PATH)
# benders_algo(DATA_PATH)
# combine_algo(DATA_PATH)
combine_algo_with_delta(DATA_PATH)
# combine_algo_without_dependency_filter(DATA_PATH)
