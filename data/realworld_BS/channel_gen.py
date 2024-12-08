from tkinter import READABLE

import xlrd as rd, xlwt as wt
import math
from dijkstra import get_shortest_path, calculate_speed

BAND_VALUE = 0.02


def haversine(BS1: tuple, BS2: tuple) -> float:
    """
    根据基站的经纬度计算基站间距离
    """
    # Haversine formula to calculate the distance
    R = 6371.0  # Radius of the Earth in kilometers
    num1, lon1, lat1 = BS1
    num2, lon2, lat2 = BS2
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = math.sin(dlat / 2.0) ** 2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2.0) ** 2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    distance = R * c
    return distance


def channel_gen(distance_threshold: float, read_path: str, write_path: str) -> None:
    """
    根据基站经纬度数据，生成基站间信道速率
    distance_threshold: 距离阈值，基站间距离<阈值 即视为基站间相连通
    """
    # 读取基站经纬度
    read_book = rd.open_workbook(read_path)
    sheet = read_book.sheet_by_index(0)
    BS_list = []
    for i in range(1, sheet.nrows):
        line = sheet.row_values(i)
        BS_list.append((i - 1, line[4], line[5]))

    distances = {}
    # 调用Haversine公式计算基站间距离
    for i, BS1 in enumerate(BS_list):
        for BS2 in BS_list[i + 1:]:
            distances[(BS1[0], BS2[0])] = haversine(BS1, BS2)

    sparse_channel = {}
    connectivity = []
    # 将距离低于某一阈值的两基站连通，计算非全连接的信道速率
    for (i, j), distance in distances.items():
        if distance < distance_threshold:
            tran_rate = BAND_VALUE * math.log(1 + (0.5 * (127 + 30 * math.log(distance, 10))) / (2 * 10 ** (-13)), 2)
            sparse_channel[(i, j)] = tran_rate
            sparse_channel[(j, i)] = tran_rate
            connectivity.append((i, j))

    full_channel = {}
    # 调用dijkstra算法补充未连接节点，生成全连接的信道速率
    for i in range(len(BS_list)):
        for j in range(len(BS_list)):
            if i == j:
                full_channel[(i, j)] = -1
                continue
            path = get_shortest_path(sparse_channel, sheet.nrows, i, j)
            speed = calculate_speed(path, sparse_channel)
            full_channel[(i, j)] = speed

    # 将全连接的信道速率写入文件
    write_book = wt.Workbook()
    sheet = write_book.add_sheet("channel")
    for i in range(len(BS_list)):
        for j in range(len(BS_list)):
            sheet.write(i, j, full_channel[(i, j)])
    sheet = write_book.add_sheet("connect")
    for i in range(len(BS_list)):
        for j in range(len(BS_list)):
            if (i, j) in connectivity or (j, i) in connectivity:
                sheet.write(i, j, 1)
            elif i == j:
                sheet.write(i, j, -1)
            else:
                sheet.write(i, j, 0)
    write_book.save(write_path)


if __name__ == '__main__':
    READ_PATH = "realworld_BS_new1208.xls"
    WRITE_PATH = "channel.xls"
    THRESHOLD = 30  # 连通BS的距离阈值
    channel_gen(THRESHOLD, READ_PATH, WRITE_PATH)
