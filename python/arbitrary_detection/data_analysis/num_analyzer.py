import json

import pandas as pd
import numpy as np
from scipy.stats import norm
import matplotlib.pyplot as plt
import seaborn as sns


class NumFeatureAnalyzer:
    def __init__(self, data_file, num_cols, label_col="label"):
        """
        参数：
        - data_file: 包含正负样本的 CSV 文件路径
        - num_cols: 要分析的数值特征列列表
        - label_col: 标签列名，默认 'label'，1=正样本, 0=负样本
        """
        self.df = pd.read_csv(data_file)
        self.num_cols = num_cols
        self.label_col = label_col

        # 分开正负样本
        self.pos_df = self.df[self.df[label_col] == 1]
        self.neg_df = self.df[self.df[label_col] == 0]

        # 确保数值列为 float
        for col in num_cols:
            self.pos_df[col] = pd.to_numeric(self.pos_df[col], errors='coerce')
            self.neg_df[col] = pd.to_numeric(self.neg_df[col], errors='coerce')

    def cohen_d(self, pos, neg):
        """计算 Cohen's d：衡量两个分布均值差的效应量"""
        return (pos.mean() - neg.mean()) / np.sqrt((pos.std() ** 2 + neg.std() ** 2) / 2)

    def overlap_area(self, pos, neg, bins=1000):
        """计算两个正态分布近似的重叠面积"""
        mu1, sigma1 = pos.mean(), pos.std()
        mu2, sigma2 = neg.mean(), neg.std()
        sigma1 = sigma1 if sigma1 > 0 else 1e-6
        sigma2 = sigma2 if sigma2 > 0 else 1e-6
        x_min = min(pos.min(), neg.min())
        x_max = max(pos.max(), neg.max())
        x = np.linspace(x_min, x_max, bins)
        pdf1 = norm.pdf(x, mu1, sigma1)
        pdf2 = norm.pdf(x, mu2, sigma2)
        overlap = np.minimum(pdf1, pdf2).sum() * (x_max - x_min) / bins
        return overlap

    def analyze(self):
        """输出每个数值特征的统计差异指标"""
        results = []
        for col in self.num_cols:
            pos_vals = self.pos_df[col].dropna()
            neg_vals = self.neg_df[col].dropna()
            if len(pos_vals) == 0 or len(neg_vals) == 0:
                continue
            d = self.cohen_d(pos_vals, neg_vals)
            overlap = self.overlap_area(pos_vals, neg_vals)
            results.append({
                'feature': col,
                'pos_mean': pos_vals.mean(),
                'neg_mean': neg_vals.mean(),
                'pos_std': pos_vals.std(),
                'neg_std': neg_vals.std(),
                'cohen_d': d,
                'overlap_area': overlap
            })
        results.sort(key=lambda x: abs(x['cohen_d']), reverse=True)
        return pd.DataFrame(results)

    def plot_distributions(self, features=None, bins=50, right_clip_quantile=0.99):
        """
        可视化正负样本在指定特征上的分布
        features: 要画图的特征列表（默认画所有）
        bins: 直方图箱数
        right_clip_quantile: 右侧截断分位数，比如 0.99 表示只保留 <= 99% 分位数的数据
        """
        if features is None:
            features = self.num_cols

        for col in features:
            pos_vals = self.pos_df[col].dropna()
            neg_vals = self.neg_df[col].dropna()

            if len(pos_vals) == 0 or len(neg_vals) == 0:
                print(f"跳过特征 {col}（缺少有效数据）")
                continue

            # 截断右尾长尾值
            upper = max(pos_vals.quantile(right_clip_quantile),
                        neg_vals.quantile(right_clip_quantile))
            pos_vals = pos_vals[pos_vals <= upper]
            neg_vals = neg_vals[neg_vals <= upper]

            plt.figure(figsize=(8, 5))
            sns.histplot(pos_vals, bins=bins, color="blue", label="Positive",
                         kde=True, stat="density", alpha=0.5)
            sns.histplot(neg_vals, bins=bins, color="red", label="Negative",
                         kde=True, stat="density", alpha=0.5)

            plt.title(f"Distribution of {col} (clipped ≤ {right_clip_quantile:.0%})")
            plt.xlabel(col)
            plt.ylabel("Density")
            plt.legend()
            plt.grid(True, linestyle="--", alpha=0.5)
            plt.show()


class StructFeatureAnalyzer:
    TRANSFER_TOPIC = "0xddf252ad1be2c89b69c2b068fc378daa952ba7f163c4a11628f55a4df523b3ef"

    def __init__(self, merged_file, clip_percentile=0.99):
        """初始化：读入包含 label 列的合并数据"""
        self.merged_df = pd.read_csv(merged_file)
        self.clip_percentile = clip_percentile

        # 按 label 拆分
        self.pos_df = self.merged_df[self.merged_df["label"] == 1].reset_index(drop=True)
        self.neg_df = self.merged_df[self.merged_df["label"] == 0].reset_index(drop=True)

        print(
            f"✅ 数据加载完成：正样本 {len(self.pos_df)} 条，负样本 {len(self.neg_df)} 条，总计 {len(self.merged_df)} 条。")

    def clip_tail(self, series):
        """裁剪长尾分布"""
        if series.dropna().empty:
            return series
        upper = np.nanquantile(series, self.clip_percentile)
        return series.clip(upper=upper)

    def _print_and_plot(self, pos_vals, neg_vals, merged_vals, title, xlabel):
        """打印统计并绘制分布"""
        print(f"=== {title} ===")
        for name, vals in [("正样本", pos_vals), ("负样本", neg_vals), ("合并样本", merged_vals)]:
            print(f"{name}：")
            print(vals.describe())
            print(f"90分位: {vals.quantile(0.9):.2f}, 95分位: {vals.quantile(0.95):.2f}\n")
        print("-------------------------------------------")

        # 绘图
        pos_clip = self.clip_tail(pos_vals)
        neg_clip = self.clip_tail(neg_vals)
        merged_clip = self.clip_tail(merged_vals)

        plt.figure(figsize=(8, 5))
        sns.histplot(pos_clip, bins=50, color="blue", label="Positive", stat="density", alpha=0.4)
        sns.histplot(neg_clip, bins=50, color="red", label="Negative", stat="density", alpha=0.4)
        sns.histplot(merged_clip, bins=50, color="green", label="Merged", stat="density", alpha=0.3)

        for vals, color, label in [
            (pos_vals, "blue", "Pos"),
            (neg_vals, "red", "Neg"),
            (merged_vals, "green", "Merged")
        ]:
            if vals.dropna().empty:
                continue
            p90 = vals.quantile(0.9)
            p95 = vals.quantile(0.95)
            plt.axvline(p90, color=color, linestyle="--", alpha=0.7)
            plt.text(p90, plt.ylim()[1] * 0.9, f"{label} P90={p90:.1f}", color=color, rotation=90, va="top")
            plt.axvline(p95, color=color, linestyle=":", alpha=0.7)
            plt.text(p95, plt.ylim()[1] * 0.8, f"{label} P95={p95:.1f}", color=color, rotation=90, va="top")

        plt.title(title + " (clipped for visualization)")
        plt.xlabel(xlabel)
        plt.ylabel("Density")
        plt.legend()
        plt.grid(True, linestyle="--", alpha=0.5)
        plt.tight_layout()
        plt.show()

    # ====================== data 列长度分析 ======================
    def analyze_data_length(self):
        self.merged_df["data_len"] = self.merged_df["data"].fillna("").apply(len)
        self.pos_df["data_len"] = self.pos_df["data"].fillna("").apply(len)
        self.neg_df["data_len"] = self.neg_df["data"].fillna("").apply(len)
        self._print_and_plot(
            self.pos_df["data_len"],
            self.neg_df["data_len"],
            self.merged_df["data_len"],
            "Distribution of data length",
            "data length"
        )

    # ====================== logs 条数分析 ======================
    def analyze_logs_count(self):
        def safe_count(x):
            try:
                logs = json.loads(x) if isinstance(x, str) else []
                return len(logs) if isinstance(logs, list) else 0
            except Exception:
                return 0

        self.merged_df["logs_count"] = self.merged_df["logs"].apply(safe_count)
        self.pos_df["logs_count"] = self.pos_df["logs"].apply(safe_count)
        self.neg_df["logs_count"] = self.neg_df["logs"].apply(safe_count)
        self._print_and_plot(
            self.pos_df["logs_count"],
            self.neg_df["logs_count"],
            self.merged_df["logs_count"],
            "Distribution of logs count",
            "logs count"
        )

    # ====================== Transfer 事件分析 ======================
    def analyze_transfer_events(self):
        def count_transfer(x):
            try:
                logs = json.loads(x) if isinstance(x, str) else []
                cnt = 0
                for log in logs:
                    if isinstance(log, dict) and "topics" in log and len(log["topics"]) > 0:
                        if log["topics"][0].lower() == self.TRANSFER_TOPIC:
                            cnt += 1
                return cnt
            except Exception:
                return 0

        self.merged_df["transfer_count"] = self.merged_df["logs"].apply(count_transfer)
        self.pos_df["transfer_count"] = self.pos_df["logs"].apply(count_transfer)
        self.neg_df["transfer_count"] = self.neg_df["logs"].apply(count_transfer)
        self._print_and_plot(
            self.pos_df["transfer_count"],
            self.neg_df["transfer_count"],
            self.merged_df["transfer_count"],
            "Distribution of Transfer event count",
            "Transfer event count"
        )


if __name__ == "__main__":
    global TARGET
    TARGET = "all_data"  # all_data or half_data
    num_cols = [
        "gas_price",
        "gas_tip_cap",
        "gas_fee_cap",
        "gas",
        "value",
        "gas_used",
        "effective_gas_price",
        "transaction_index"
    ]
    analyzer = NumFeatureAnalyzer(
        data_file=f"../{TARGET}/datasets/new_test_data.csv",
        num_cols=num_cols
    )
    # 查看差异分析结果
    df_results = analyzer.analyze()
    print(df_results)
    # 可视化部分特征分布
    analyzer.plot_distributions(features=num_cols)

    # analyzer = StructFeatureAnalyzer(f"../{TARGET}/datasets/train_data.csv", clip_percentile=0.95)
    # analyzer.analyze_data_length()
    # analyzer.analyze_logs_count()
    # analyzer.analyze_transfer_events()
