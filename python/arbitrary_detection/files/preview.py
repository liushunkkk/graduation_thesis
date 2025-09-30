# 读取前 100 行正负样本
import pandas as pd

df_pos = pd.read_csv("./positive_data.csv", nrows=100)
df_neg = pd.read_csv("./negative_data.csv", nrows=100)

df_pos.to_csv("./preview_positive_data.csv", index=False)
df_neg.to_csv("./preview_negative_data.csv", index=False)