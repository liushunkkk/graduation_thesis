# 读取 csv
import pandas as pd

df = pd.read_csv("../all_data/datasets/test.csv")

# 只保留['tx_hash', "label"]两列
df = df[['tx_hash', "label"]]

# 保存到新文件
df.to_csv("./result.csv", index=False)
#
# res = pd.read_csv("./result.csv")
#
# res["label"] = df["label"]
#
# res.to_csv("./result.csv")
