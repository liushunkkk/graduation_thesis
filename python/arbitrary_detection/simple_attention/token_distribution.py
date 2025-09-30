import pandas as pd
from collections import Counter
import ast
import json
import hashlib
import matplotlib.pyplot as plt


class TokenDistributionAnalyzer:
    def __init__(self, filepath, data_col='data', logs_col='logs'):
        self.filepath = filepath
        self.data_col = data_col
        self.logs_col = logs_col
        self.df = pd.read_csv(filepath)

    # ------------------- tokenization -------------------
    def tokenize_data(self, data_str):
        if pd.isna(data_str) or data_str == '':
            return []
        s = data_str.lower().lstrip('0x')
        tokens = [s[:8]] + [s[i:i + 64] for i in range(8, len(s), 64)]
        return tokens

    def tokenize_logs(self, logs_str):
        if pd.isna(logs_str) or logs_str == '':
            return []
        try:
            logs_list = json.loads(logs_str)
        except:
            try:
                logs_list = ast.literal_eval(logs_str)
            except:
                return []
        all_tokens = []
        for log in logs_list:
            topics = ''.join([t.lower().lstrip('0x') for t in log.get('topics', [])])
            data = log.get('data', '').lower().lstrip('0x')
            s = topics + data
            tokens = [s[i:i + 64] for i in range(0, len(s), 64)]
            all_tokens.extend(tokens)
        return all_tokens

    # ------------------- hashing -------------------
    def token_to_id(self, token, vocab_size):
        """
        将 token 映射到指定 vocab_size 范围内
        """
        h = hashlib.blake2b(token.encode("utf-8"), digest_size=8).hexdigest()
        return int(h, 16) % vocab_size

    # ------------------- 统计 -------------------
    def get_token_counts(self, col, tokenizer):
        all_tokens = []
        for val in self.df[col].dropna():
            all_tokens.extend(tokenizer(val))
        return Counter(all_tokens)

    def map_counts_to_vocab(self, token_counts, vocab_size):
        mapped_counts = Counter()
        for token, count in token_counts.items():
            tid = self.token_to_id(token, vocab_size)
            mapped_counts[tid] += count
        return mapped_counts

    # ------------------- 可视化 -------------------
    def plot_distribution(self, token_counts, title):
        counts = list(token_counts.values())
        plt.figure(figsize=(10, 5))
        plt.hist(counts, bins=100, log=True)
        plt.title(title)
        plt.xlabel('Token Frequency')
        plt.ylabel('Count (log scale)')
        plt.show()

    # ------------------- 分析入口 -------------------
    def analyze(self, vocab_sizes=[20000, 40000, 60000, 80000, 100000]):
        for col, tokenizer in [(self.data_col, self.tokenize_data), (self.logs_col, self.tokenize_logs)]:
            # 原始分布
            raw_counts = self.get_token_counts(col, tokenizer)
            print(f"{col} 原始 unique tokens: {len(raw_counts)}")
            # self.plot_distribution(raw_counts, f"{col} Raw Token Frequency Distribution")

            # 映射到不同 vocab_size
            for vocab in vocab_sizes:
                mapped_counts = self.map_counts_to_vocab(raw_counts, vocab)
                print(f"{col} mapped to vocab_size={vocab}, unique ids: {len(mapped_counts)}")
                # self.plot_distribution(mapped_counts, f"{col} Token Frequency Distribution (vocab={vocab})")


if __name__ == "__main__":
    print("------positive result-------")
    a = TokenDistributionAnalyzer('../files/positive_data.csv')
    a.analyze()

    print("------negative result-------")
    b = TokenDistributionAnalyzer('../files/negative_data.csv')
    b.analyze()
