# 读取 csv
import pandas as pd

df = pd.read_csv("../all_data/datasets/test.csv")

# 只保留其中一列，例如名为 'column_name'
df = df[['tx_hash']]

# 保存到新文件
df.to_csv("./result.csv", index=False)
