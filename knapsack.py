import math


def fptas_knapsack(weights, profits, capacity, epsilon):
    """
    FPTAS 算法求解背包问题
    :param weights: 物品重量列表
    :param profits: 物品利润列表
    :param capacity: 背包容量
    :param epsilon: 容差值 范围0~1
    :return: 最大利润和放置方案
    """
    if epsilon <= 0 or epsilon >= 1:
        raise ValueError("Invalid epsilon value")

    n = len(profits)  # 物品数量
    max_profit = max(profits)  # 所有物品利润中的最大值

    # Step 1: 计算缩放因子 K
    K = (epsilon * max_profit) / n

    # Step 2: 缩放后的利润
    scaled_profits = [math.floor(p / K) for p in profits]

    # Step 3: 初始化动态规划表
    # dp[j] 表示容量为 j 时的最大缩放利润值
    dp = [0] * (capacity + 1)
    place_method_recorder = [{} for _ in range(capacity + 1)]

    # Step 4: 填充动态规划表 (允许多次选择每个物品)
    for i in range(n):
        weight = weights[i]
        scaled_profit = scaled_profits[i]
        for j in range(weight, capacity + 1):
            # dp[j] = max(dp[j], dp[j - weight] + scaled_profit)
            if dp[j] < dp[j - weight] + scaled_profit:
                dp[j] = dp[j - weight] + scaled_profit
                place_method_recorder[j] = place_method_recorder[j - weight].copy()
                place_method_recorder[j].setdefault(i, 0)
                place_method_recorder[j][i] += 1

    # Step 5: 找到符合条件的最大缩放利润
    max_scaled_profit = max(dp)
    max_index = dp.index(max_scaled_profit)

    # Step 6: 转换回原利润值
    actual_profit = 0
    for item, count in place_method_recorder[max_index].items():
        actual_profit += count * profits[item]
    """
    approximate_profit = max_scaled_profit * K
    if abs(actual_profit - approximate_profit) >= 0.1:
        raise ValueError("近似误差较大")
    """
    return actual_profit, place_method_recorder[max_index]


if __name__ == '__main__':
    weights = [1, 2, 3, 4]
    profits = [1.5, 3.4, 5.1, 6.9]
    capacity = 10
    epsilon = 0.1
    max_profit, place_method = fptas_knapsack(weights, profits, capacity, epsilon)
    print("Max Profit:", max_profit)
    print("Place Method:", place_method)
