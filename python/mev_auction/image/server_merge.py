import json
from collections import defaultdict
from datetime import datetime

from matplotlib import pyplot as plt

if __name__ == '__main__':
    log_file = "../base_lab_log_files/share-node.log"

    # 数据存储
    send_counts_per_sec_share_node = defaultdict(lambda: defaultdict(int))  # builder -> {秒时间 -> count}

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
                bucket = (t_sec // 3) * 3  # 3秒一个桶
                send_counts_per_sec_share_node[builder][bucket] += 1

    log_file = "../base_lab_log_files/bsc-rpc.log"

    # 数据存储
    send_counts_per_sec_bsc_rpc = defaultdict(lambda: defaultdict(int))  # builder -> {秒时间 -> count}

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
                bucket = (t_sec // 3) * 3  # 3秒一个桶
                send_counts_per_sec_bsc_rpc[builder][bucket] += 1

    builder = "blockrazor"

    share_counts = send_counts_per_sec_share_node.get("blockrazor", {})
    bsc_counts = send_counts_per_sec_bsc_rpc.get("blockRazor", {})

    if not share_counts and not bsc_counts:
        print(f"builder '{builder}' not found in data.")
    else:
        plt.figure(figsize=(12, 5))

        # === share-node ===
        if share_counts:
            times_share = sorted(share_counts.keys())
            t0_share = times_share[0]
            times_offset_share = [t - t0_share for t in times_share]
            share_y = [share_counts[t] for t in times_share]
            plt.plot(times_offset_share, share_y, label="MEV Share Node", marker="o")

        # === bsc-rpc ===
        if bsc_counts:
            times_bsc = sorted(bsc_counts.keys())
            t0_bsc = times_bsc[0]
            times_offset_bsc = [t - t0_bsc for t in times_bsc]
            bsc_y = [bsc_counts[t] for t in times_bsc]
            plt.plot(times_offset_bsc, bsc_y, label="MP-RPC Node", marker="o")

        plt.xlabel("Relative Time (s, 3-second bucket offset)")
        plt.ylabel("Avg txs per 3 seconds")
        plt.title(f"builder sent count")
        plt.legend(
            loc="upper left",
            bbox_to_anchor=(1.02, 1),  # 调整位置：x>1表示右移到图外
            borderaxespad=0,
            frameon=False
        )
        plt.tight_layout(rect=[0, 0, 0.85, 1])  # 为图例留出右边空间
        plt.show()
