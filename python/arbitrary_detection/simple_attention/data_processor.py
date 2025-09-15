import json

import pandas as pd
import numpy as np
from scipy.stats.mstats import winsorize
from sklearn.preprocessing import StandardScaler
import hashlib


class NumFeaturePreprocessor:
    def __init__(self, input_file: str, output_file: str, vocab_size=20000, data_max_len=48, logs_max_len=64):
        self.input_file = input_file
        self.output_file = output_file
        self.df = None
        self.features = None
        self.scaler = StandardScaler()
        self.vocab_size = vocab_size
        self.data_max_len = data_max_len
        self.logs_max_len = logs_max_len

        # 数值特征列
        self.num_cols = [
            "gas_price",
            "gas_tip_cap",
            "gas_fee_cap",
            "gas",
            "value",
            "gas_used",
            "effective_gas_price",
            "transaction_index",
        ]

        self.cols = [
            "gas_price",
            "gas_tip_cap",
            "gas_fee_cap",
            "gas",
            "value",
            "gas_used",
            "effective_gas_price",
            "transaction_index",
            "data",
            "logs"
        ]

    def load_data(self):
        """读取CSV"""
        self.df = pd.read_csv(self.input_file)
        self.features = self.df[self.cols].copy()
        print(f"✅ 数据加载完成，共 {len(self.df)} 条记录")

    def fill_missing(self):
        """缺失值处理"""
        # gas_price 填充
        self.features["gas_price"] = self.features["gas_price"].fillna(
            self.features["gas_fee_cap"]
        )
        # gas_tip_cap 和 gas_fee_cap 填充
        self.features["gas_tip_cap"] = self.features["gas_tip_cap"].fillna(
            self.features["gas_price"]
        )
        self.features["gas_fee_cap"] = self.features["gas_fee_cap"].fillna(
            self.features["gas_price"]
        )

        for col in self.num_cols:
            self.features[col] = pd.to_numeric(self.features[col], errors='coerce')

        # 其他缺失值用中位数补
        self.features[self.num_cols] = self.features[self.num_cols].fillna(self.features[self.num_cols].median())
        print("✅ 缺失值处理完成")

    def log_transform(self):
        """log1p 压缩"""
        for col in self.num_cols:
            self.features[col] = np.log1p(self.features[col])
        print("✅ log1p 压缩完成")

    def winsorize_outliers(self, limits=(0.01, 0.01)):
        """缩尾，处理极端值"""
        for col in self.num_cols:
            self.features[col] = winsorize(self.features[col], limits=limits)
        print("✅ 缩尾处理完成")

    def standardize(self):
        """z-score 标准化"""
        self.features[self.num_cols] = self.scaler.fit_transform(self.features[self.num_cols])
        print("✅ 标准化完成")

    def token_to_id(self, token: str):
        h = hashlib.blake2b(token.encode("utf-8"), digest_size=8).hexdigest()
        return int(h, 16) % self.vocab_size

    @staticmethod
    def serialize_single_data(data_hex: str):
        """Data 序列化 → token list"""
        s = data_hex.lower().replace("0x", "")
        tokens = []

        # 函数选择器
        if len(s) >= 8:
            tokens.append(s[:8])
        # 参数槽位
        for i in range(8, len(s), 64):
            tokens.append(s[i:i + 64])
        return tokens

    def data_to_ids(self, data_hex: str):
        tokens = self.serialize_single_data(data_hex)
        ids = [self.token_to_id(t) for t in tokens][:self.data_max_len]
        # padding
        if len(ids) < self.data_max_len:
            ids += [0] * (self.data_max_len - len(ids))
        return ids

    @staticmethod
    def serialize_single_log(log: dict):
        s = ""
        for t in log.get("topics", []):
            s += t.lower().replace("0x", "")
        s += log.get("data", "").lower().replace("0x", "")
        tokens = [s[i:i + 64] for i in range(0, len(s), 64)]
        return tokens

    def logs_to_ids(self, logs_json_str):
        if pd.isna(logs_json_str) or logs_json_str == "":
            return [0] * self.logs_max_len
        try:
            logs_list = json.loads(logs_json_str)
        except:
            return [0] * self.logs_max_len

        all_tokens = []
        for log_item in logs_list:
            all_tokens += self.serialize_single_log(log_item)

        ids = [self.token_to_id(t) for t in all_tokens][:self.logs_max_len]
        if len(ids) < self.logs_max_len:
            ids += [0] * (self.logs_max_len - len(ids))
        return ids

    def logs_to_tokens(self, logs_json_str):
        if pd.isna(logs_json_str) or logs_json_str == "":
            return [0] * self.logs_max_len
        try:
            logs_list = json.loads(logs_json_str)
        except:
            return [0] * self.logs_max_len

        all_tokens = []
        for log_item in logs_list:
            all_tokens += self.serialize_single_log(log_item)

        return all_tokens

    def save(self):
        """保存处理后的结果"""
        self.features.to_csv(self.output_file, index=False)
        print(f"✅ 特征保存完成，输出文件：{self.output_file}")

    def run(self):
        """完整流程"""
        self.load_data()

        # 处理数值特征
        self.fill_missing()
        self.log_transform()
        self.winsorize_outliers()
        self.standardize()
        # 数值特征拼接
        self.features["num_feature"] = self.features[self.num_cols].values.tolist()
        print("--------数值特征完成---------")

        # 处理data
        self.features["data_token"] = self.features["data"].apply(self.serialize_single_data)
        self.features["data_feature"] = self.features["data"].apply(self.data_to_ids)
        print("--------Data特征完成---------")

        # 处理logs
        self.features["logs_token"] = self.features["logs"].apply(self.logs_to_tokens)
        self.features["logs_feature"] = self.features["logs"].apply(self.logs_to_ids)
        print("--------Logs特征完成---------")

        self.save()


if __name__ == "__main__":
    input_file = "../files/positive_data.csv"
    output_file = "./datasets/positive_data.csv"
    processor = NumFeaturePreprocessor(input_file, output_file)
    processor.run()

    output = pd.read_csv(output_file)
    print(output.columns)
    print(output[:1])

    input_file = "../files/negative_data.csv"
    output_file = "./datasets/negative_data.csv"
    processor = NumFeaturePreprocessor(input_file, output_file)
    processor.run()

    output = pd.read_csv(output_file)
    print(output.columns)
    print(output[:1])
