import json
from collections import defaultdict
import networkx as nx

TRANSFER_SIG = "0xddf252ad1be2c89b69c2b068fc378daa952ba7f163c4a11628f55a4df523b3ef"


class TransferArbitrageDetector:
    """
    基于 A Large-Scale Study of the Ethereum Arbitrage Ecosystem 的原子套利识别方法
    只处理 ERC-20 Transfer 事件
    """

    def parse_transfers(self, logs_json):
        """解析交易 receipt 中的 Transfer 事件"""
        if isinstance(logs_json, str):
            logs = json.loads(logs_json)
        else:
            logs = logs_json

        transfers = []
        for log in logs:
            topics = log.get("topics", [])
            if not topics or topics[0].lower() != TRANSFER_SIG:
                continue
            from_addr = "0x" + topics[1][-40:]
            to_addr = "0x" + topics[2][-40:]
            token_addr = log.get("address", "").lower()
            try:
                value = int(log.get("data", "0x0"), 16)
            except:
                value = 0
            transfers.append((token_addr, from_addr.lower(), to_addr.lower(), value))
        return transfers

    def infer_exchanges(self, transfers):
        """
        推断可能的兑换地址，并生成 token -> token 边
        """
        # 统计每个地址各 token 流入 / 流出
        addr_token_flow = defaultdict(lambda: defaultdict(int))
        addresses = set()
        for token, frm, to, value in transfers:
            addr_token_flow[frm][token] -= value  # 发出
            addr_token_flow[to][token] += value  # 收入
            addresses.update([frm, to])

        exchanges = []
        for addr in addresses:
            flows = addr_token_flow[addr]
            received = [t for t, v in flows.items() if v > 0]
            sent = [t for t, v in flows.items() if v < 0]

            # 只要有流入流出即可
            if len(received) > 0 and len(sent) > 0:
                for in_token in received:
                    for out_token in sent:
                        if in_token != out_token:  # 并且流入!=流出
                            exchanges.append({
                                "executor": addr,
                                "in_token": in_token,
                                "out_token": out_token,
                                "in_amount": flows[in_token],
                                "out_amount": -flows[out_token]
                            })
        return exchanges

    def build_graph(self, exchanges):
        """根据 exchanges 构建 MultiDiGraph"""
        G = nx.MultiDiGraph()
        for ex in exchanges:
            a = ex["in_token"]
            b = ex["out_token"]
            G.add_node(a)
            G.add_node(b)
            G.add_edge(a, b, executor=ex["executor"],
                       in_amount=ex["in_amount"],
                       out_amount=ex["out_amount"])
        return G

    def detect_cycles_and_pivot(self, exchanges):
        """检测图中的循环，并计算 pivot token 和净利润"""
        Di = nx.DiGraph()
        for ex in exchanges:
            Di.add_edge(ex["in_token"], ex["out_token"])

        cycles = list(nx.simple_cycles(Di))
        results = []

        if not cycles:
            return results

        for cycle in cycles:
            # 统计这个cycle内是否有某个token的某个地址增长了
            pivot = None
            best_profit = 0
            address = None
            for token in cycle:
                addr_profit = defaultdict(int)
                for ex in exchanges:
                    if ex["in_token"] == token:
                        addr_profit[ex["executor"]] += ex["in_amount"]
                    elif ex["out_token"] == token:
                        addr_profit[ex["executor"]] -= ex["out_amount"]
                for addr, profit in addr_profit.items():
                    if profit > best_profit:
                        best_profit = profit
                        pivot = token
                        address = addr
            if best_profit > 0:
                results.append({
                    "cycle": cycle,
                    "pivot": pivot,
                    "address": address,
                    "profit_raw": best_profit,
                })
        return results

    def analyze_transaction(self, logs_json):
        """端到端分析一笔交易"""
        transfers = self.parse_transfers(logs_json)
        exchanges = self.infer_exchanges(transfers)
        G = self.build_graph(exchanges)
        cycles_info = self.detect_cycles_and_pivot(exchanges)
        return {
            "transfers_count": len(transfers),
            "exchanges": exchanges,
            "graph": G,
            "cycles": cycles_info,
            "is_arbitrage": len(cycles_info) > 0
        }
