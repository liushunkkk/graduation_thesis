import json
from collections import defaultdict, Counter
from datetime import datetime

import numpy as np
from matplotlib import pyplot as plt

if __name__ == '__main__':
    lab = "multi_lab_log_files"
    log_file = f"../{lab}/bsc-rpc.log"

    # 数据存储
    send_txs = defaultdict(list)

    with open(log_file, "r", encoding="utf-8") as f:
        for line in f:
            try:
                log = json.loads(line)
            except json.JSONDecodeError:
                continue

            msg = log.get("msg", "")
            ts = log.get("timestamp")
            if ts:
                t = datetime.strptime(ts, "%Y-%m-%dT%H:%M:%S.%f%z")
                t_sec = int(t.timestamp())  # 秒时间戳

            # === xxx-send ===
            if "-send" in msg:
                # 从 msg 中提取 builder 名字
                builder = msg.split("-send")[0].strip("[]")
                cost = log.get("cost")
                txs = log.get("txs")
                userId = log.get("userId")
                if txs is not None:
                    send_txs[builder].append(txs)

    lab = "base_lab_log_files"
    log_file = f"../{lab}/bsc-rpc.log"

    # 数据存储
    send_txs_base = defaultdict(list)

    with open(log_file, "r", encoding="utf-8") as f:
        for line in f:
            try:
                log = json.loads(line)
            except json.JSONDecodeError:
                continue

            msg = log.get("msg", "")
            ts = log.get("timestamp")
            if ts:
                t = datetime.strptime(ts, "%Y-%m-%dT%H:%M:%S.%f%z")
                t_sec = int(t.timestamp())  # 秒时间戳

            # === xxx-send ===
            if "-send" in msg:
                # 从 msg 中提取 builder 名字
                builder = msg.split("-send")[0].strip("[]")
                cost = log.get("cost")
                txs = log.get("txs")
                userId = log.get("userId")
                if txs is not None:
                    send_txs_base[builder].append(txs)

    for builder, txs_list in send_txs.items():
        c = Counter(txs_list)
        c1 = Counter(send_txs_base[builder])

        # 统一 X 轴
        all_keys = sorted(set(c.keys()) | set(c1.keys()))
        x = np.arange(len(all_keys))  # X轴位置

        width = 0.2  # 柱宽

        plt.figure(figsize=(6, 4))

        plt.bar(x - width / 2, [c1.get(k, 0) for k in all_keys], width=width, label="mutil-layer lab")
        plt.bar(x + width / 2, [c.get(k, 0) for k in all_keys], width=width, label="comparison lab")
        plt.xlabel("Txs")
        plt.ylabel("Send count")
        plt.title(f"Builder txs distribution")
        plt.xticks(x, all_keys)  # 把x轴标记换成txs的值
        plt.legend(
            loc="upper left",
            bbox_to_anchor=(1.02, 1),  # 调整位置：x>1表示右移到图外
            borderaxespad=0,
            frameon=False
        )
        plt.tight_layout(rect=[0, 0, 0.95, 0.9])  # 为图例留出右边空间
        plt.show()

        break
