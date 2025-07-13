import pandas as pd
import numpy as np
import string

from one_hot.data_encoder import DataEncoder


class OneHotEncoder:
    def __init__(self):
        self.negative_data = None
        self.positive_data = None
        self.positive_encoded_data = None
        self.negative_encoded_data = None
        self.DataEncoder = DataEncoder()

    def load_dataset(self):
        self.positive_data = pd.read_csv("../files/positive_data.csv", sep=",")
        self.negative_data = pd.read_csv("../files/negative_data.csv", sep=",")

    def encode_data(self):
        # 构造字符表：大小写字母 + 数字
        vocab = list(string.ascii_letters + string.digits)
        char_to_idx = {char: idx for idx, char in enumerate(vocab)}
        max_len = 1024
        vocab_size = len(vocab)

        def encode_string(s):
            s = str(s)[:max_len].ljust(max_len)  # 截断或补空格补齐
            result = np.zeros((max_len, vocab_size), dtype=np.float32)
            for i, c in enumerate(s):
                if c in char_to_idx:
                    result[i][char_to_idx[c]] = 1.0
            return result

        # 编码 positive data 中的 data 列
        self.positive_encoded_data = np.stack(
            self.positive_data["data"].apply(encode_string)
        )

        # 编码 negative data 中的 data 列
        self.negative_encoded_data = np.stack(
            self.negative_data["data"].apply(encode_string)
        )

    def train_data_classifier(self):
        self.DataEncoder.train_classifier(self.positive_encoded_data, self.negative_encoded_data)

    def visualize_data_features(self):
        self.DataEncoder.visualize_features(self.positive_encoded_data, self.negative_encoded_data)

    def get_positive_data(self):
        return self.positive_data

    def get_negative_data(self):
        return self.negative_data

    def get_positive_encoded(self):
        return self.positive_encoded_data

    def get_negative_encoded(self):
        return self.negative_encoded_data
