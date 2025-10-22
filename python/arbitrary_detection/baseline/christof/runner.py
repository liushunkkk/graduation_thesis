from collections import Counter

import pandas as pd
from web3 import Web3

from baseline.christof.detector import SwapArbitrageDetector
from baseline.christof.handler import ChristofHandler

if __name__ == '__main__':
    bsc_rpc = "https://bsc-mainnet.core.chainstack.com/04b893f36a6e7162205b532a6efbee07"  # 或你的节点
    w3 = Web3(Web3.HTTPProvider(bsc_rpc))
    assert w3.is_connected(), "连接 BSC 节点失败"

    detector = SwapArbitrageDetector(w3)

    handler = ChristofHandler(detector)

    # 分析正负样本
    df_results = handler.analyze_files("../../all_data/datasets/test.csv")

    count = Counter(detector.swap_sig)
    print(count)

    # 计算指标
    metrics, y_pred = handler.compute_metrics(df_results)
    print(metrics)

    res = pd.read_csv("../result.csv")
    res["christof_result"] = y_pred

    res.to_csv("../result.csv", index=False)

    # 查看前几行结果
    # print(df_results.head(10))
