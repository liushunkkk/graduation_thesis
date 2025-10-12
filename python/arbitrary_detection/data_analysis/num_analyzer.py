import json

import pandas as pd
import numpy as np
from scipy.stats import norm
import matplotlib.pyplot as plt
import seaborn as sns


class NumFeatureAnalyzer:
    def __init__(self, pos_file, neg_file, num_cols):
        self.pos_df = pd.read_csv(pos_file)
        self.neg_df = pd.read_csv(neg_file)
        self.num_cols = num_cols

        # 确保数值列为 float
        for col in num_cols:
            self.pos_df[col] = pd.to_numeric(self.pos_df[col], errors='coerce')
            self.neg_df[col] = pd.to_numeric(self.neg_df[col], errors='coerce')

    def cohen_d(self, pos, neg):
        return (pos.mean() - neg.mean()) / np.sqrt((pos.std() ** 2 + neg.std() ** 2) / 2)

    def overlap_area(self, pos, neg, bins=1000):
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
        results = []
        for col in self.num_cols:
            pos_vals = self.pos_df[col].dropna()
            neg_vals = self.neg_df[col].dropna()
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
        return results

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

            # 只截断右边长尾
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

    def __init__(self, pos_file, neg_file, clip_percentile=0.99):
        self.pos_df = pd.read_csv(pos_file)
        self.neg_df = pd.read_csv(neg_file)
        self.clip_percentile = clip_percentile
        self.merged_df = pd.concat([self.pos_df, self.neg_df], ignore_index=True)

    def clip_tail(self, series):
        upper = np.nanquantile(series, self.clip_percentile)
        return series.clip(upper=upper)

    def _print_and_plot(self, pos_vals, neg_vals, merged_vals, title, xlabel):
        """通用输出统计+画图逻辑"""

        # ---- 打印原始统计 ----
        print(f"=== {title} (原始) ===")
        for name, vals in [("正样本", pos_vals), ("负样本", neg_vals), ("合并样本", merged_vals)]:
            print(f"{name}：")
            print(vals.describe())
            print(f"90分位: {vals.quantile(0.9):.2f}, 95分位: {vals.quantile(0.95):.2f}\n")
        print("-------------------------------------------")

        # ---- 绘图（裁剪后可视化） ----
        pos_clip = self.clip_tail(pos_vals)
        neg_clip = self.clip_tail(neg_vals)
        merged_clip = self.clip_tail(merged_vals)

        plt.figure(figsize=(8, 5))
        sns.histplot(pos_clip, bins=50, color="blue", label="Positive", stat="density", alpha=0.4)
        sns.histplot(neg_clip, bins=50, color="red", label="Negative", stat="density", alpha=0.4)
        sns.histplot(merged_clip, bins=50, color="green", label="Merged", stat="density", alpha=0.3)

        # ---- 添加P90 / P95线 ----
        for vals, color, label in [
            (pos_vals, "blue", "Pos"),
            (neg_vals, "red", "Neg"),
            (merged_vals, "green", "Merged")
        ]:
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
        self.pos_df["data_len"] = self.pos_df["data"].fillna("").apply(len)
        self.neg_df["data_len"] = self.neg_df["data"].fillna("").apply(len)
        self.merged_df["data_len"] = self.merged_df["data"].fillna("").apply(len)
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

        self.pos_df["logs_count"] = self.pos_df["logs"].apply(safe_count)
        self.neg_df["logs_count"] = self.neg_df["logs"].apply(safe_count)
        self.merged_df["logs_count"] = self.merged_df["logs"].apply(safe_count)
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

        self.pos_df["transfer_count"] = self.pos_df["logs"].apply(count_transfer)
        self.neg_df["transfer_count"] = self.neg_df["logs"].apply(count_transfer)
        self.merged_df["transfer_count"] = self.merged_df["logs"].apply(count_transfer)
        self._print_and_plot(
            self.pos_df["transfer_count"],
            self.neg_df["transfer_count"],
            self.merged_df["transfer_count"],
            "Distribution of Transfer event count",
            "Transfer event count"
        )


if __name__ == "__main__":
    global TARGET
    TARGET = "all_data" # all_data or half_data
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
    analyzer = NumFeatureAnalyzer(f"../{TARGET}/datasets/positive_data.csv", f"../{TARGET}/datasets/negative_data.csv", num_cols)
    results = analyzer.analyze()
    for r in results:
        print(f"{r['feature']}: cohen_d={r['cohen_d']:.3f}, overlap_area={r['overlap_area']:.3f}, "
              f"pos_mean={r['pos_mean']:.3f}, neg_mean={r['neg_mean']:.3f}")

    analyzer.plot_distributions(features=num_cols, right_clip_quantile=0.95)

    analyzer = StructFeatureAnalyzer(f"../{TARGET}/datasets/positive_data.csv", f"../{TARGET}/datasets/negative_data.csv",
                                     clip_percentile=0.95)
    analyzer.analyze_data_length()
    analyzer.analyze_logs_count()
    analyzer.analyze_transfer_events()
