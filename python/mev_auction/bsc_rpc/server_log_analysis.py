import json
from datetime import datetime
from collections import defaultdict, Counter
import numpy as np
import matplotlib.pyplot as plt

if __name__ == '__main__':

    log_file = "../log_files/bsc-rpc.log"

    # 数据存储
    pool_records = []
    send_costs = defaultdict(list)
    send_costs_user = defaultdict(list)
    send_txs = defaultdict(list)
    send_counts_per_sec = defaultdict(lambda: defaultdict(int))  # builder -> {秒时间 -> count}

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

            # === pool-state ===
            if "[pool-state]" in msg:
                # 假设 group 和 bundles 在 log 里直接是 key
                # 有些 zap 日志可能嵌套在 fields 里
                group = log.get("group")
                bundles = log.get("bundles")
                if group is not None and bundles is not None:
                    pool_records.append({
                        "time": t,
                        "group": group,
                        "bundles": bundles
                    })

            # === xxx-send ===
            if "-send" in msg:
                # 从 msg 中提取 builder 名字
                builder = msg.split("-send")[0].strip("[]")
                cost = log.get("cost")
                txs = log.get("txs")
                userId = log.get("userId")
                if cost is not None:
                    send_costs_user[userId].append(cost)
                    send_costs[builder].append(cost)
                if txs is not None:
                    send_txs[builder].append(txs)
                # 统计每秒流量
                bucket = (t_sec // 3) * 3  # 3秒一个桶
                send_counts_per_sec[builder][bucket] += 1

    # ==================== 可视化 pool-state ====================
    if pool_records:
        pool_records_sorted = sorted(pool_records, key=lambda r: r["time"])

        times = [r["time"] for r in pool_records]
        groups = [r["group"] for r in pool_records]
        bundles = [r["bundles"] for r in pool_records]
        relative_seconds = [(t - times[0]).total_seconds() for t in times]
        plt.figure(figsize=(12, 5))
        plt.plot(relative_seconds, groups, label="Group Size")
        plt.plot(relative_seconds, bundles, label="Bundle Size")
        plt.xlabel("Time(s)")
        plt.xlim(0, 220)
        plt.ylabel("Size")
        plt.title("Pool-State Over Time")
        plt.legend(
            loc="upper left",
            bbox_to_anchor=(1.02, 1),  # 调整位置：x>1表示右移到图外
            borderaxespad=0,
            frameon=False
        )
        plt.tight_layout(rect=[0, 0, 0.85, 1])  # 为图例留出右边空间
        plt.show()


    # ==================== cost 统计信息 ====================
    def stats(arr):
        arr = np.array(arr)
        return {
            "count": len(arr),
            "min": int(arr.min()),
            "max": int(arr.max()),
            "mean": float(arr.mean()),
            "median": float(np.median(arr)),
            "p90": float(np.percentile(arr, 90)),
            "p95": float(np.percentile(arr, 95)),
            "p99": float(np.percentile(arr, 99))
        }


    for builder, costs in send_costs.items():
        print(f"=== builder {builder} send cost ====")
        print(stats(costs))

    for userId, costs in send_costs_user.items():
        print(f"=== user {userId} send cost ====")
        print(stats(costs))

    # ==================== txs 分布统计 ====================
    for builder, txs_list in send_txs.items():
        c = Counter(txs_list)
        print(f"=== {builder} txs distribution ===")
        print(dict(c))
        plt.figure(figsize=(6, 4))
        plt.bar(c.keys(), c.values())
        plt.xlabel("txs")
        plt.ylabel("Count")
        plt.title(f"{builder} txs distribution")
        plt.show()
        break

    # ==================== 全局统计（合并所有 builder） ====================
    all_costs = []
    all_txs = []

    for costs in send_costs.values():
        all_costs.extend(costs)

    for txs_list in send_txs.values():
        all_txs.extend(txs_list)

    print("=== Global send cost ===")
    print(stats(all_costs))

    c = Counter(all_txs)
    print("=== Global txs distribution ===")
    print(dict(c))

    plt.figure(figsize=(6, 4))
    plt.bar(c.keys(), c.values())
    plt.xlabel("txs")
    plt.ylabel("Count")
    plt.title("Global txs distribution")
    plt.show()

    # ================= 每秒流量可视化 =================
    for builder, sec_counts in send_counts_per_sec.items():
        times_sorted = sorted(sec_counts.keys())
        counts_sorted = [sec_counts[t] for t in times_sorted]

        # X轴从0开始秒数
        times_offset = [t - times_sorted[0] for t in times_sorted]

        plt.figure(figsize=(12, 5))
        plt.plot(times_offset, counts_sorted, marker='o')
        plt.xlabel("Time (s)")
        plt.ylabel("Txs per second")
        plt.title(f"{builder} send per second")
        plt.tight_layout()
        plt.show()
        break
