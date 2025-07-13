import pandas as pd
import numpy as np
import string
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report, confusion_matrix


class OneHotEncoder:
    def __init__(self):
        self.negative_data = None
        self.positive_data = None
        self.positive_encoded = None
        self.negative_encoded = None

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
        self.positive_encoded = np.stack(
            self.positive_data["data"].apply(encode_string)
        )

        # 编码 negative data 中的 data 列
        self.negative_encoded = np.stack(
            self.negative_data["data"].apply(encode_string)
        )

    def train_classifier(self):
        # 构造标签
        y_pos = np.ones(len(self.positive_encoded))
        y_neg = np.zeros(len(self.negative_encoded))

        # 合并数据和标签
        X = np.concatenate([self.positive_encoded, self.negative_encoded], axis=0)
        y = np.concatenate([y_pos, y_neg], axis=0)

        # 扁平化：每个样本从 (1024, vocab_size) 转为 (1024 * vocab_size)
        X = X.reshape((X.shape[0], -1))

        # 划分训练集和测试集
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

        # 训练逻辑回归
        print("==== Logistic Regression ====")
        lr = LogisticRegression(max_iter=1000)
        lr.fit(X_train, y_train)
        y_pred_lr = lr.predict(X_test)
        print("Classification Report:")
        print(classification_report(y_test, y_pred_lr))
        print("Confusion Matrix:")
        print(confusion_matrix(y_test, y_pred_lr))

        # 训练多层感知机
        print("\n==== MLP Classifier ====")
        mlp = MLPClassifier(hidden_layer_sizes=(256,), max_iter=20, random_state=42)
        mlp.fit(X_train, y_train)
        y_pred_mlp = mlp.predict(X_test)
        print("Classification Report:")
        print(classification_report(y_test, y_pred_mlp))
        print("Confusion Matrix:")
        print(confusion_matrix(y_test, y_pred_mlp))

    def visualize_features(self):
        # 准备标签
        y_pos = np.ones(len(self.positive_encoded))
        y_neg = np.zeros(len(self.negative_encoded))

        # 合并数据和标签
        X = np.concatenate([self.positive_encoded, self.negative_encoded], axis=0)
        y = np.concatenate([y_pos, y_neg], axis=0)

        # Flatten
        X = X.reshape((X.shape[0], -1))

        # PCA降维到2维
        pca = PCA(n_components=2)
        X_pca = pca.fit_transform(X)

        # 绘图
        plt.figure(figsize=(10, 6))
        plt.scatter(X_pca[y == 0, 0], X_pca[y == 0, 1], c='blue', label='Negative', alpha=0.5, s=10)
        plt.scatter(X_pca[y == 1, 0], X_pca[y == 1, 1], c='red', label='Positive', alpha=0.5, s=10)
        plt.title("PCA of Encoded 'data' Field")
        plt.xlabel("PC1")
        plt.ylabel("PC2")
        plt.legend()
        plt.grid(True)
        plt.show()

    def get_positive_data(self):
        return self.positive_data

    def get_negative_data(self):
        return self.negative_data

    def get_positive_encoded(self):
        return self.positive_encoded

    def get_negative_encoded(self):
        return self.negative_encoded
