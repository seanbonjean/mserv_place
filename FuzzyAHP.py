from mpmath.libmp import normalize


class FN:
    """
    FuzzyNumber模糊数对象
    """

    def __init__(self, l: float, m: float, u: float):
        """
        :param l: 模糊数下限
        :param m: 模糊数中位
        :param u: 模糊数上限
        """
        self.l = l
        self.m = m
        self.u = u

    def __add__(self, other):
        return FN(self.l + other.l, self.m + other.m, self.u + other.u)

    def __truediv__(self, other):
        if isinstance(other, (int, float)):
            return FN(self.l / other, self.m / other, self.u / other)
        else:
            raise TypeError("除数类型错误")

    def __repr__(self):
        return f"FuzzyNumber(l={self.l}, m={self.m}, u={self.u})"

    def __str__(self):
        return f"({self.l}, {self.m}, {self.u})"

    def defuzzify(self):
        """
        解模糊
        """
        return (self.l + self.m + self.u) / 3


def assert_matrix(matrix):
    """
    检查矩阵是否为模糊矩阵
    :param matrix: 待检查的矩阵
    """
    row_len = len(matrix)
    for row in matrix:
        col_len = 0
        for item in row:
            col_len += 1
            if not isinstance(item, FN):
                raise ValueError("矩阵中包含非模糊数")
        if col_len != row_len:
            raise ValueError("矩阵不是方阵")


def fuzzy_normalize(matrix):
    """
    归一化模糊矩阵
    :param matrix: 需要归一化的模糊比较矩阵
    :return: 归一化后的矩阵
    """
    N = len(matrix)
    col_sums = []  # 列和
    for col_index in range(N):
        col_sum = FN(0, 0, 0)
        for row_index in range(N):
            col_sum += matrix[row_index][col_index]
        col_sums.append(col_sum)

    normalized_matrix = []
    for row in matrix:
        normalized_row = [num / col_sums[col_index].defuzzify() for col_index, num in enumerate(row)]
        normalized_matrix.append(normalized_row)
    return normalized_matrix


def cal_fuzzy_weight(matrix):
    """
    计算模糊权重
    :param matrix: 对应的模糊比较矩阵
    """
    weights = []  # 模糊权重，即每行的模糊数平均值
    for row in matrix:
        weight = FN(0, 0, 0)
        for num in row:
            weight += num
        weights.append(weight / len(row))
    return weights


def defuzzify_weights(weights):
    """
    解模糊
    :param weights: 模糊权重
    """
    return [weight.defuzzify() for weight in weights]


def fuzzyAHP(criteria_matrix):
    """
    封装整个过程
    """
    assert_matrix(criteria_matrix)
    normalized_matrix = fuzzy_normalize(criteria_matrix)
    fuzzy_weights = cal_fuzzy_weight(normalized_matrix)
    weights = defuzzify_weights(fuzzy_weights)

    # 对权重归一化（但由于前面对矩阵的归一化，其实权重已经是归一化过的，这里其实可有可无）
    weight_total = sum(weights)
    normalized_weights = [weight / weight_total for weight in weights]

    return normalized_weights
