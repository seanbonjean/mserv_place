import random


def matrix_gen(row: int, col: int, rand_size: tuple, save_path: str) -> None:
    """
    随机生成矩阵
    """
    with open(save_path, 'w') as file:
        for _ in range(row):
            s = ""
            for _ in range(col):
                s += ' ' + str(random.randint(rand_size[0], rand_size[1]))
            file.write(s.strip() + '\n')


def symmetric_gen(row: int, col: int, rand_size: tuple, save_path: str) -> None:
    """
    随机生成对称矩阵
    """
    metrix = {}
    for i in range(row):
        for j in range(col):
            if i == j:
                metrix[(i, j)] = 0
            elif i < j:
                metrix[(i, j)] = random.randint(rand_size[0], rand_size[1])
            elif i > j:
                metrix[(i, j)] = metrix[(j, i)]
    with open(save_path, 'w') as file:
        s = ""
        for j in range(col):
            s = ' '.join([str(metrix[(i, j)]) for i in range(row)])
            file.write(s.strip() + '\n')


if __name__ == "__main__":
    SAVE_PATH = "rand.txt"

    matrix_gen(30, 1, (100, 300), SAVE_PATH)
    # symmetric_gen(100, 100, (1, 9), SAVE_PATH)
