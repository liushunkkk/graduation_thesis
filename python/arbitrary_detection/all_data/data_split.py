import os

import pandas as pd

if __name__ == '__main__':
    test_ratio = 0.2
    pos_csv = "./datasets/positive_data.csv"
    neg_csv = "./datasets/negative_data.csv"
    if not os.path.exists(f"./datasets/train.csv"):
        print("重新构建")
        pos_df = pd.read_csv(pos_csv)
        pos_df['label'] = 1
        neg_df = pd.read_csv(neg_csv)
        neg_df['label'] = 0
        df = pd.concat([pos_df, neg_df]).sort_values("block_number").reset_index(drop=True)
        test_size = int(len(df) * test_ratio)
        df.iloc[:-test_size].to_csv(f"./datasets/train.csv", index=False)
        df.iloc[-test_size:].to_csv(f"./datasets/test.csv", index=False)
    else:
        print("已存在")