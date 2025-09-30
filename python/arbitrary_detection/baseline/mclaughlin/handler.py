import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


class CSVHandler:
    """
    处理套利/非套利 CSV 文件交易数据，调用 ArbitrageDetector 分析，并计算指标
    """

    def __init__(self, detector):
        self.detector = detector

    def analyze_files(self, data_csv: str):
        """
        分析正负样本文件

        :return: 分析结果 DataFrame
        """
        df_data = pd.read_csv(data_csv)
        results = []

        for idx, row in df_data.iterrows():
            tx_hash = row.get("tx_hash")
            logs_json = row.get("logs")
            label = row.get("label")
            if not logs_json or pd.isna(logs_json):
                continue
            try:
                analysis = self.detector.analyze_transaction(logs_json)
                is_arb = analysis["is_arbitrage"]
                pivot_token = analysis["cycles"][0]["pivot"] if is_arb else None
                profit_raw = analysis["cycles"][0]["profit_raw"] if is_arb else 0
                results.append({
                    "tx_hash": tx_hash,
                    "is_arbitrage": is_arb,
                    "pivot_token": pivot_token,
                    "profit_raw": profit_raw,
                    "label": label
                })
            except Exception as e:
                results.append({
                    "tx_hash": tx_hash,
                    "is_arbitrage": False,
                    "pivot_token": None,
                    "profit_raw": 0,
                    "label": label,
                    "error": str(e)
                })

        return pd.DataFrame(results)

    @staticmethod
    def compute_metrics(df, pred_col="is_arbitrage", true_col="label"):
        """
        根据 DataFrame 计算详细指标，包括 TP、FP、TN、FN
        :param df: 包含预测列和真实标签列
        :param pred_col: 预测列名
        :param true_col: 真实标签列名
        """
        df_eval = df[df[true_col].notna()]
        y_true = df_eval[true_col].astype(int)
        y_pred = df_eval[pred_col].astype(int)

        TP = ((y_true == 1) & (y_pred == 1)).sum()
        FP = ((y_true == 0) & (y_pred == 1)).sum()
        TN = ((y_true == 0) & (y_pred == 0)).sum()
        FN = ((y_true == 1) & (y_pred == 0)).sum()

        total = len(df_eval)
        correct = (y_true == y_pred).sum()
        wrong = total - correct

        accuracy = correct / total
        precision = TP / (TP + FP) if (TP + FP) > 0 else 0
        recall = TP / (TP + FN) if (TP + FN) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

        print(f"Total samples: {total}")
        print(f"Correct predictions: {correct}, Wrong predictions: {wrong}")
        print(f"TP: {TP}, FP: {FP}, TN: {TN}, FN: {FN}")
        print(f"Accuracy: {accuracy:.3f}")
        print(f"Precision: {precision:.3f}")
        print(f"Recall: {recall:.3f}")
        print(f"F1 score: {f1:.3f}")

        return {
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "TP": TP,
            "FP": FP,
            "TN": TN,
            "FN": FN,
            "total": total,
            "correct": correct,
            "wrong": wrong
        }
