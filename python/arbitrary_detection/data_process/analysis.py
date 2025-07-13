import numpy as np
import pymysql

# 源数据库配置
database_config = {
    "host": "127.0.0.1",
    "port": 3306,
    "user": "root",
    "password": "root123456",
    "database": "arbitrary",
    "charset": "utf8mb4"
}


def data_analyze(data):
    numbers = np.array(data)

    mean = np.mean(numbers)  # 平均值
    median = np.median(numbers)  # 中位数
    min_value = np.min(numbers)  # 最小值
    max_value = np.max(numbers)  # 最大值
    std_dev = np.std(numbers)  # 标准差
    variance = np.var(numbers)  # 方差
    percentile_10 = np.percentile(numbers, 10)
    percentile_25 = np.percentile(numbers, 25)
    percentile_75 = np.percentile(numbers, 75)
    percentile_90 = np.percentile(numbers, 90)

    print(f"10分位数: {percentile_10}")
    print(f"25分位数: {percentile_25}")
    print(f"75分位数: {percentile_75}")
    print(f"90分位数: {percentile_90}")
    print(f"平均值: {mean}")
    print(f"中位数: {median}")
    print(f"最小值: {min_value}")
    print(f"最大值: {max_value}")
    print(f"标准差: {std_dev}")
    print(f"方差: {variance}")


def quantile_trim(data, lower_percent=0.1, upper_percent=0.9):
    """
    对一维数据进行分位截断，去除偏小和偏大的值。

    参数:
        data (array-like): 输入数据
        lower_percent (float): 下界百分位（例如 0.1 表示 10%）
        upper_percent (float): 上界百分位（例如 0.9 表示 90%）

    返回:
        trimmed_data (np.ndarray): 截断后的数据
    """
    data = np.asarray(data)
    lower = np.percentile(data, lower_percent * 100)
    upper = np.percentile(data, upper_percent * 100)
    return data[(data >= lower) & (data <= upper)]


def select_data(select_sql):
    conn = pymysql.connect(**database_config)
    with conn.cursor() as cursor:
        cursor.execute(select_sql)
        rows = cursor.fetchall()
        data_list = [row[0] for row in rows]
        data_len_list = [len(data) for data in data_list]
        return data_len_list


def operate(table):
    if table == "ethereum":
        select_sql = """
        SELECT data
        FROM ethereum_transactions 
        LIMIT 50000
        """
    elif table == "comparison":
        select_sql = """
        SELECT data
        FROM comparison_transactions
        LIMIT 50000
        """
    print("=====origin=====")
    data_len_list = select_data(select_sql)
    data_analyze(data_len_list)
    print("=====filtered======")
    filter_list = quantile_trim(data_len_list)
    data_analyze(filter_list)


if __name__ == '__main__':
    print("---------ethereum----------")
    operate("ethereum")
    print("---------------------------")

    print("---------comparison--------")
    operate("comparison")
    print("---------------------------")
