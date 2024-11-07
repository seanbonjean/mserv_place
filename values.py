class CONSTANTS:
    MAX_MAKESPAN = 200  # 各任务的允许的最大完成时间(Th_max)
    MAX_DEPLOY_COST = 1000  # 部署微服务的允许的最大成本(C_max)


import xlrd as rd, xlwt as wt
from objects import *


def load_data(datapath: str) -> tuple:
    """
    从excel读取数据
    """
    data = rd.open_workbook(datapath)

    sheet = data.sheet_by_name("EdgeNode")
    edgenode_list = []
    for i in range(1, sheet.nrows):
        line = sheet.row_values(i)
        edgenode_list.append(EdgeNode(int(line[0]), line[1], line[2]))

    sheet = data.sheet_by_name("MicroService")
    microservice_list = []
    for i in range(1, sheet.nrows):
        line = sheet.row_values(i)
        microservice_list.append(MicroService(int(line[0]), line[1], line[2], line[3], line[4]))

    sheet1 = data.sheet_by_name("User")
    sheet2 = data.sheet_by_name("User.mserv_dependency")
    user_list = []
    for i in range(1, sheet1.nrows):
        line = sheet1.row_values(i)
        mserv_dep = sheet2.row_values(i)
        mserv_dep = mserv_dep[:mserv_dep.index('')]
        user_list.append(User(int(line[0]), line[1], int(line[2]), list(map(int, mserv_dep[1:]))))

    sheet = data.sheet_by_name("channel_rate")
    channelrate_dict = {}
    for i in range(sheet.nrows):
        for j in range(sheet.ncols):
            channelrate_dict[(i, j)] = sheet.cell(i, j).value

    return edgenode_list, microservice_list, user_list, channelrate_dict


if __name__ == '__main__':
    edgenode_list, microservice_list, user_list, channelrate_dict = load_data("data/test_input.xls")
    print(channelrate_dict[(8, 9)])
