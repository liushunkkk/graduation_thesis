# 读取前 100 行正负样本
import pandas as pd

df_pos = pd.read_csv("./positive_data.csv", nrows=50)
df_neg = pd.read_csv("./negative_data.csv", nrows=50)

df_pos.to_csv("./positive_data_preview.csv", index=False)
df_neg.to_csv("./negative_data_preview.csv", index=False)

df_pos = pd.read_csv("./eth_positive_data.csv", nrows=50)
# df_neg = pd.read_csv("./negative_data.csv", nrows=50)

df_pos.to_csv("./eth_positive_data_preview.csv", index=False)
# df_neg.to_csv("./eth_negative_data_preview.csv", index=False)