import pandas as pd

from baseline.mclaughlin.detector import TransferArbitrageDetector
from baseline.mclaughlin.handler import MclaughlinHandler

if __name__ == '__main__':
    detector = TransferArbitrageDetector()
    handler = MclaughlinHandler(detector)

    # 分析正负样本
    df_results = handler.analyze_files("../../all_data/datasets/test.csv")

    # 计算指标
    metrics, y_pred = handler.compute_metrics(df_results)
    print(metrics)

    res = pd.read_csv("../result.csv")
    res["mclaughlin_result"] = y_pred

    res.to_csv("../result.csv", index=False)

    # 查看前几行结果
    # print(df_results.head(10))
