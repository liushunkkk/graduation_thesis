import json
import numpy as np
from datetime import datetime, timedelta
from collections import defaultdict

from matplotlib import pyplot as plt

lab = "base_lab_log_files"


def parse_log_line(line):
    try:
        return json.loads(line)
    except:
        return None


def summarize(values):
    if not values:
        return {}
    arr = np.array(values)
    return {
        "count": len(arr),
        "min": int(arr.min()),
        "max": int(arr.max()),
        "mean": int(arr.mean()),
        "median": int(np.median(arr)),
        "p90": int(np.percentile(arr, 90)),
        "p95": int(np.percentile(arr, 95)),
        "p99": int(np.percentile(arr, 99)),
    }


def analyze_log(filename):
    mev_costs = []
    raw_costs = []
    user1_tx_map = {}
    user2_tx_map = {}
    user3_tx_map = {}
    user_tx_used = set()
    latencies1 = []
    latencies2 = []
    latencies3 = []

    # 新增按秒统计 receive one level bundle cost
    bundle_costs_per_sec = defaultdict(list)
    bundle_costs_per_sec_compare = defaultdict(list)

    with open(filename, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            data = parse_log_line(line)
            if not data:
                continue

            msg = data.get("msg", "")
            cost = data.get("cost")
            ts = data.get("timestamp")
            t_sec = None
            if ts:
                try:
                    t = datetime.strptime(ts, "%Y-%m-%dT%H:%M:%S.%f%z")
                    t_sec = t.replace(microsecond=0)  # 按秒聚合
                except:
                    pass

            # 1. 统计 sendMevBundle
            if "[call-eth_sendMevBundle]" in msg and cost is not None:
                mev_costs.append(int(cost))
                continue

            # 2. 统计 sendRawTransaction
            if "[call-eth_sendRawTransaction]" in msg and cost is not None:
                raw_costs.append(int(cost))
                continue

            # 3. 用户发送成功
            if "[User-1]" in msg and "开始发送交易" in msg:
                tx_hash = data.get("txHash")
                send_time = data.get("sendTime")
                if tx_hash and send_time:
                    user1_tx_map[tx_hash] = int(send_time)
                continue
            if "[User-2]" in msg and "开始发送交易" in msg:
                tx_hash = data.get("txHash")
                send_time = data.get("sendTime")
                if tx_hash and send_time:
                    user2_tx_map[tx_hash] = int(send_time)
                continue
            if "[User-3]" in msg and "开始发送交易" in msg:
                tx_hash = data.get("txHash")
                send_time = data.get("sendTime")
                if tx_hash and send_time:
                    user3_tx_map[tx_hash] = int(send_time)
                continue

            # 4. 搜索者接收到 bundle
            if "receive one level bundle" in msg:
                tx_hashes = data.get("txHashes", [])
                recv_time = data.get("receiveTime")
                if not tx_hashes or not recv_time:
                    continue
                for h in tx_hashes:
                    if h in user1_tx_map and h not in user_tx_used:
                        user_tx_used.add(h)
                        delay = int(recv_time) - user1_tx_map[h]
                        if delay > 0:
                            # 统计 [Searcher][01] receive one level bundle cost
                            if "[Searcher][01]" in msg and t_sec:
                                bundle_costs_per_sec[t_sec].append(int(delay))
                            elif "[Searcher][06]" in msg and t_sec:
                                bundle_costs_per_sec_compare[t_sec].append(int(delay))
                            latencies1.append(delay)
                        break  # 找到就可以了
                    elif h in user2_tx_map and h not in user_tx_used:
                        user_tx_used.add(h)
                        delay = int(recv_time) - user2_tx_map[h]
                        if delay > 0:
                            # 统计 [Searcher][01] receive one level bundle cost
                            if "[Searcher][01]" in msg and t_sec:
                                bundle_costs_per_sec[t_sec].append(int(delay))
                            elif "[Searcher][06]" in msg and t_sec:
                                bundle_costs_per_sec_compare[t_sec].append(int(delay))
                            latencies2.append(delay)
                        break  # 找到就可以了
                    elif h in user3_tx_map and h not in user_tx_used:
                        user_tx_used.add(h)
                        delay = int(recv_time) - user3_tx_map[h]
                        if delay > 0:
                            # 统计 [Searcher][01] receive one level bundle cost
                            if "[Searcher][01]" in msg and t_sec:
                                bundle_costs_per_sec[t_sec].append(int(delay))
                            elif "[Searcher][06]" in msg and t_sec:
                                bundle_costs_per_sec_compare[t_sec].append(int(delay))
                            latencies3.append(delay)
                        break  # 找到就可以了

    print("时间单位：微妙")
    print("==== sendMevBundle cost ====")
    print(summarize(mev_costs))
    print("\n==== sendRawTransaction cost ====")
    print(summarize(raw_costs))
    print("\n==== 用户1交易推流延迟统计（receiveTime - sendTime） ====")
    print(summarize(latencies1))
    print("\n==== 用户2交易推流延迟统计（receiveTime - sendTime） ====")
    print(summarize(latencies2))
    print("\n==== 用户3交易推流延迟统计（receiveTime - sendTime） ====")
    print(summarize(latencies2))

    # ==================== bundle cost 按秒平均 ====================
    if bundle_costs_per_sec:
        times_sorted = sorted(bundle_costs_per_sec.keys())
        avg_costs = [np.mean(bundle_costs_per_sec[t]) // 1000 for t in times_sorted]
        # times_sorted = sorted(bundle_costs_per_sec_compare.keys())
        avg_costs_compare = [np.mean(bundle_costs_per_sec_compare[t]) // 1000 for t in times_sorted]

        # 生成从 0 开始的相对秒数
        relative_seconds = [(t - times_sorted[0]).total_seconds() for t in times_sorted]
        plt.figure(figsize=(12, 5))
        plt.plot(relative_seconds, avg_costs, label='searcher-01', marker='o')
        plt.plot(relative_seconds, avg_costs_compare, label='searcher-06', marker='o')
        plt.xlabel("Time (s)")
        plt.ylabel("Average Receive Time (ms)")
        plt.ylim(0, 100)
        plt.title("[Searcher] Receive User Transaction Stream Wait Over Time")
        plt.legend()
        plt.tight_layout()
        plt.show()

    p95 = int(np.percentile(latencies1, 95))
    if bundle_costs_per_sec:
        times_sorted = sorted(bundle_costs_per_sec.keys())
        avg_costs_filtered = []
        for t in times_sorted:
            costs = np.array(bundle_costs_per_sec[t])
            # 剔除超过 p95 的值
            costs = costs[costs <= p95]
            if len(costs) == 0:
                avg_costs_filtered.append(np.nan)
            else:
                avg_costs_filtered.append(np.mean(costs) // 1000)
        relative_seconds = [(t - times_sorted[0]).total_seconds() for t in times_sorted]
        plt.figure(figsize=(12, 5))
        plt.plot(relative_seconds, avg_costs_filtered, marker='o')
        plt.xlabel("Time (s)")
        plt.ylabel("Average Receive Time (ms)")
        plt.ylim(0, 100)
        plt.title("[Searcher][01] Receive User Transaction Stream Wait Over Time (filter > p95)")
        plt.tight_layout()
        plt.show()


if __name__ == "__main__":
    analyze_log(f"../{lab}/bsc-rpc-client.log")
