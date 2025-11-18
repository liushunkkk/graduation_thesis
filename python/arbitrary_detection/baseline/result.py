# 读取 csv
import pandas as pd

df = pd.read_csv("../all_data/datasets/test.csv")

# 只保留['tx_hash', "label"]两列，最终会有：mclaughlin_result,arbinet_result,our_result,christof_result
df = df[['tx_hash', "label"]]

# 保存到新文件
df.to_csv("./result.csv", index=False)

# df = pd.read_csv("./result.csv")
#
# result = df[
#     (df["label"] == 0) &
#     (df["mclaughlin_result"] == 1) &
#     (df["arbinet_result"] == 0) &
#     (df["our_result"] == 0) &
#     (df["christof_result"] == 1)
#     ]
# pd.set_option('display.max_colwidth', None)
# pd.set_option('display.max_rows', None)
# print(result)
