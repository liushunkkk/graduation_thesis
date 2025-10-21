import ast

import pandas as pd
import numpy as np
import json
from scipy.stats.mstats import winsorize
from sklearn.preprocessing import StandardScaler


def serialize_single_data(data_hex: str):
    if pd.isna(data_hex) or data_hex == "":
        return []
    s = data_hex.lower().replace("0x", "")
    tokens = []
    if len(s) >= 8:
        tokens.append(s[:8])
    for i in range(8, len(s), 64):
        tokens.append(s[i:i + 64])
    return tokens


def serialize_logs(logs_json_str):
    if pd.isna(logs_json_str) or logs_json_str == "":
        return []
    try:
        logs_list = json.loads(logs_json_str)
    except:
        return []
    return logs_list


def count_transfer(logs_list):
    TRANSFER_SIG = "0xddf252ad1be2c89b69c2b068fc378daa952ba7f163c4a11628f55a4df523b3ef"  # Transfer 事件
    count = 0
    for log in logs_list:
        topics = log.get("topics", [])
        if len(topics) > 0 and topics[0].lower() == TRANSFER_SIG:
            count += 1
    return count


def append_and_process_features(file_path):
    df = pd.read_csv(file_path)

    # 新增三列
    df['data_len'] = df['data'].apply(lambda x: len(x))
    df['logs_len'] = df['logs'].apply(lambda x: len(serialize_logs(x)))
    df['transfer_len'] = df['logs'].apply(lambda x: count_transfer(serialize_logs(x)))

    # log1p 压缩
    for col in ['data_len', 'logs_len', 'transfer_len']:
        df[col] = np.log1p(df[col])

    # 缩尾处理
    for col in ['data_len', 'logs_len', 'transfer_len']:
        df[col] = winsorize(df[col], limits=(0.01, 0.01))

    # 标准化
    scaler = StandardScaler()
    df[['data_len', 'logs_len', 'transfer_len']] = scaler.fit_transform(df[['data_len', 'logs_len', 'transfer_len']])

    # 合并到 num_feature
    def merge_num_feature(row):
        base = ast.literal_eval(row['num_feature'])
        extra = [row['data_len'], row['logs_len'], row['transfer_len']]
        return base + extra

    df['num_feature'] = df.apply(merge_num_feature, axis=1)

    df.to_csv(file_path, index=False)
    print(f"{file_path} 已更新 num_feature 并追加三列")


if __name__ == "__main__":
    files = [
        "./datasets/train.csv",
        "./datasets/test.csv",
    ]

    for f in files:
        append_and_process_features(f)
