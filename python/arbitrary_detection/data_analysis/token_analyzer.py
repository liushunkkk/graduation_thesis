import pandas as pd
import json


class TokenLengthAnalyzer:
    def __init__(self, csv_path):
        self.df = pd.read_csv(csv_path)

    def serialize_data(self, data_hex: str):
        """按你的Data序列化规则拆分token"""
        if pd.isna(data_hex) or not data_hex.startswith("0x"):
            return []
        s = data_hex[2:].lower()
        tokens = []
        if len(s) >= 8:
            tokens.append(s[:8])
        for i in range(8, len(s), 64):
            tokens.append(s[i:i + 64])
        return tokens

    def serialize_logs(self, logs_json: str):
        """按你的Logs序列化规则拆分token"""
        if pd.isna(logs_json):
            return []
        try:
            logs = json.loads(logs_json)
        except:
            return []

        tokens = []
        for log in logs:
            long_str = ""
            for topic in log.get("topics", []):
                long_str += topic[2:] if topic.startswith("0x") else topic
            data_str = log.get("data", "")
            if data_str.startswith("0x"):
                data_str = data_str[2:]
            long_str += data_str
            # 按64字符拆分
            for i in range(0, len(long_str), 64):
                tokens.append(long_str[i:i + 64])
        return tokens

    def analyze_length_distribution(self, label=""):
        data_lengths = []
        logs_lengths = []

        for idx, row in self.df.iterrows():
            data_tokens = self.serialize_data(row.get("data", ""))
            logs_tokens = self.serialize_logs(row.get("logs", ""))
            data_lengths.append(len(data_tokens))
            logs_lengths.append(len(logs_tokens))

        data_series = pd.Series(data_lengths)
        logs_series = pd.Series(logs_lengths)

        print(f"------ {label} ------")
        print("Data token长度分布:")
        print(data_series.describe())
        print(f"P90: {data_series.quantile(0.9):.2f}, P95: {data_series.quantile(0.95):.2f}")

        print("\nLogs token长度分布:")
        print(logs_series.describe())
        print(f"P90: {logs_series.quantile(0.9):.2f}, P95: {logs_series.quantile(0.95):.2f}")
        print("---------------------------------------------------\n")

        return data_series, logs_series


if __name__ == "__main__":
    global TARGET
    TARGET = "all_data" # all_data or half_data
    pos_analyzer = TokenLengthAnalyzer(f"../{TARGET}/datasets/positive_data.csv")
    pos_data_len, pos_logs_len = pos_analyzer.analyze_length_distribution("positive result")

    neg_analyzer = TokenLengthAnalyzer(f"../{TARGET}/datasets/negative_data.csv")
    neg_data_len, neg_logs_len = neg_analyzer.analyze_length_distribution("negative result")

    merged_df = pd.concat([pos_analyzer.df, neg_analyzer.df], ignore_index=True)
    merged_analyzer = TokenLengthAnalyzer.__new__(TokenLengthAnalyzer)
    merged_analyzer.df = merged_df
    merged_data_len, merged_logs_len = merged_analyzer.analyze_length_distribution("merged result (positive + negative)")