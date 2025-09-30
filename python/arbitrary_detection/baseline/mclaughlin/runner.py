from baseline.mclaughlin.detector import ArbitrageDetector
from baseline.mclaughlin.handler import CSVHandler

if __name__ == '__main__':
    detector = ArbitrageDetector()
    handler = CSVHandler(detector)

    # 分析正负样本
    df_results = handler.analyze_files("../../simple_attention/datasets/test.csv")

    # 计算指标
    metrics = handler.compute_metrics(df_results)
    print(metrics)

    # 查看前几行结果
    # print(df_results.head(10))
