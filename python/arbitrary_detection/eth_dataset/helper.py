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

    print(len(df[df["label"] == 1]))
    print(len(df[df["label"] == 0]))

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

    # 最小区块号：10866588
    # 最大区块号：11080466
    # 个数：411905
    # 19963
    # 391942
    # train
    # 最小区块号：10000126
    # 最大区块号：10866587
    # 个数：239566
