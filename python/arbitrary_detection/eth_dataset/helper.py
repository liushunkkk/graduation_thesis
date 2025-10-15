import pandas as pd

if __name__ == '__main__':
    df = pd.read_csv("./datasets/test.csv")

    # 提取 block_number 列
    block_numbers = df["block_number"]

    # 计算最小值和最大值
    min_block = block_numbers.min()
    max_block = block_numbers.max()

    print("test")
    print(f"最小区块号：{min_block}")
    print(f"最大区块号：{max_block}")
    print(f"个数：{len(block_numbers)}")

    df = pd.read_csv("./datasets/train.csv")

    # 提取 block_number 列
    block_numbers = df["block_number"]

    # 计算最小值和最大值
    min_block = block_numbers.min()
    max_block = block_numbers.max()

    print("train")
    print(f"最小区块号：{min_block}")
    print(f"最大区块号：{max_block}")
    print(f"个数：{len(block_numbers)}")

    # test
    # 最小区块号：47560341
    # 最大区块号：47570103
    # 个数：59719
    # train
    # 最小区块号：47462117
    # 最大区块号：47560341
    # 个数：238877
