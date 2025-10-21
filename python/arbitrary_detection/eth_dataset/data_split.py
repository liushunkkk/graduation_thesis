import os

import pandas as pd


def split_origin_data():
    test_ratio = 0.2
    pos_csv = "../files/eth_positive_data.csv"
    neg_csv = "../files/eth_negative_data.csv"

    train_path = "./datasets/train_data.csv"
    test_path = "./datasets/test_data.csv"

    if not os.path.exists(train_path):
        print("重新构建")

        # 读取数据并打上标签
        pos_df = pd.read_csv(pos_csv)
        pos_df['label'] = 1
        neg_df = pd.read_csv(neg_csv)
        neg_df['label'] = 0

        # 合并并按 block_number 排序
        df = pd.concat([pos_df, neg_df]).sort_values("block_number").reset_index(drop=True)

        # 计算目标划分位置
        total_len = len(df)
        target_test_size = int(total_len * test_ratio)
        split_index = total_len - target_test_size  # 80% 位置

        # 找到分界处的 block_number
        boundary_block = df.iloc[split_index]["block_number"]
        print("boundary_block: ", boundary_block) # 10866587

        # 所有 block_number < boundary_block 的样本为训练集
        # 所有 block_number >= boundary_block 的样本为测试集
        # 注意：如果 boundary_block 出现的样本被部分划分到测试集，
        # 我们要把它们全部归入训练集（即不拆 block）
        train_df = df[df["block_number"] <= boundary_block]
        test_df = df[df["block_number"] > boundary_block]

        # 如果测试集太小（例如边界 block 体量大），可在这里打印查看比例
        print(f"训练集: {len(train_df)}, 测试集: {len(test_df)}, 实际比例: {len(test_df) / len(df):.2f}")

        # 保存结果
        os.makedirs("./datasets", exist_ok=True)
        train_df.to_csv(train_path, index=False)
        test_df.to_csv(test_path, index=False)

    else:
        print("已存在")
    # 训练集: 239566, 测试集: 59887, 实际比例: 0.20


def supple_negative_data():
    origin_neg_csv = "./datasets/test_data.csv"
    origin_neg_df = pd.read_csv(origin_neg_csv)
    supple_neg_csv = "../files/test_eth_negative_data.csv"
    supple_neg_df = pd.read_csv(supple_neg_csv)
    supple_neg_df['label'] = 0
    print("supple_neg_df 长度:", len(supple_neg_df)) # 375812
    merged_df = pd.concat([origin_neg_df, supple_neg_df], ignore_index=True)
    merged_df = merged_df.drop_duplicates(subset='tx_hash', keep='first')
    merged_df.to_csv('./datasets/new_test_data.csv', index=False)
    print("已合并到new_test_data.csv")
    print("new_test_data长度: ", len(merged_df)) # 411905


if __name__ == '__main__':
    split_origin_data()
    supple_negative_data()
