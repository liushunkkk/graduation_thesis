import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report, confusion_matrix


class DataEncoder:
    def train_classifier(self, positive_encoded, negative_encoded):
        # 构造标签
        y_pos = np.ones(len(positive_encoded))
        y_neg = np.zeros(len(negative_encoded))

        # 合并数据和标签
        X = np.concatenate([positive_encoded, negative_encoded], axis=0)
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

    def visualize_features(self, positive_encoded, negative_encoded):
        # 准备标签
        y_pos = np.ones(len(positive_encoded))
        y_neg = np.zeros(len(negative_encoded))

        # 合并数据和标签
        X = np.concatenate([positive_encoded, negative_encoded], axis=0)
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
