import time

import torch
from torch_geometric.data import Data
from collections import defaultdict
import json

from web3 import Web3

import utils.builders


class ArbiNetTransactionBuilder:
    def __init__(self, df):
        """
        df: pandas DataFrame，包含交易字段和 label
        """
        self.df = df
        self.w3 = Web3(Web3.HTTPProvider("https://bsc-mainnet.core.chainstack.com/0424da0312cca554dccbaed288ee0c2d"))

        self.graphs = []

    def build_graphs(self):
        for i, row in self.df.iterrows():
            try:
                logs = json.loads(row['logs'])
                print(f"load [{i}] [{row['tx_hash']}]")
                tx_label = row['label']

                ca_addresses = set()

                # 收集交易中所有地址
                addresses = set()
                for log in logs:
                    topics = log.get('topics', [])
                    ca_addresses.add(log['address'])  # 只有合约才能发日志
                    if len(topics) > 0 and topics[
                        0].lower() == "0xddf252ad1be2c89b69c2b068fc378daa952ba7f163c4a11628f55a4df523b3ef":
                        from_addr = '0x' + topics[1][-40:]
                        to_addr = '0x' + topics[2][-40:]
                        addresses.add(from_addr)
                        addresses.add(to_addr)
                if len(addresses) == 0:
                    # 构不出图，直接当作非套利交易
                    label = torch.tensor([0], dtype=torch.long)
                    dummy = Data(x=torch.zeros((1, 14)), edge_index=torch.zeros((2, 0), dtype=torch.long), y=label)
                    self.graphs.append(dummy)
                    continue

                node2id = {addr: idx for idx, addr in enumerate(addresses)}
                num_nodes = len(node2id)
                x = torch.zeros((num_nodes, 14), dtype=torch.float)
                edge_list = []

                # 初始化 token 出入统计
                in_token_value = defaultdict(list)
                out_token_value = defaultdict(list)
                in_tokens = defaultdict(set)
                out_tokens = defaultdict(set)
                in_count = defaultdict(int)
                out_count = defaultdict(int)
                all_tokens = set()
                all_transfers = 0

                # 遍历日志，构建边和节点特征统计
                for log in logs:
                    topics = log.get('topics', [])
                    if len(topics) == 0 or topics[
                        0].lower() != "0xddf252ad1be2c89b69c2b068fc378daa952ba7f163c4a11628f55a4df523b3ef":
                        continue
                    from_addr = '0x' + topics[1][-40:]
                    to_addr = '0x' + topics[2][-40:]
                    token_addr = log['address']
                    val = float(int(log['data'], 16) if len(log['data']) > 2 else 0)

                    f_id = node2id[from_addr]
                    t_id = node2id[to_addr]

                    # 边
                    edge_list.append([f_id, t_id])

                    # 出入统计
                    out_count[f_id] += 1
                    out_tokens[f_id].add(token_addr)
                    out_token_value[f_id].append((token_addr, val))

                    in_count[t_id] += 1
                    in_tokens[t_id].add(token_addr)
                    in_token_value[t_id].append((token_addr, val))

                    all_tokens.add(token_addr)
                    all_transfers += 1

                # 计算每个节点特征
                for addr, idx in node2id.items():
                    # 计算 token_net
                    token_in_sum = defaultdict(float)
                    token_out_sum = defaultdict(float)
                    for token, val in in_token_value.get(idx, []):
                        token_in_sum[token] += val
                    for token, val in out_token_value.get(idx, []):
                        token_out_sum[token] += val

                    all_tokens = set(token_in_sum.keys()).union(token_out_sum.keys())
                    token_net = {}
                    pos_count = neg_count = zero_count = 0
                    for token in all_tokens:
                        net = token_in_sum.get(token, 0) - token_out_sum.get(token, 0)
                        token_net[token] = net
                        if net > 0:
                            pos_count += 1
                        elif net < 0:
                            neg_count += 1
                        else:
                            zero_count += 1

                    if addr in ca_addresses:
                        is_ca_address = 1
                    else:
                        # is_ca_address = 0
                        addr_checksum = Web3.to_checksum_address(addr)
                        code = self.w3.eth.get_code(addr_checksum)
                        is_ca_address = 1 if code == b'' or code == b'0x' else 0
                        if is_ca_address:
                            ca_addresses.add(addr)
                    # 节点特征 14 维（保留原注释）
                    x[idx, 0] = neg_count  # count of tokens whose profit is smaller than 0
                    x[idx, 1] = pos_count  # count of tokens whose profit is greater than 0
                    x[idx, 2] = zero_count  # count of tokens whose profit is 0
                    x[idx, 3] = len(out_tokens[idx])  # count of tokens sent at least once at the address
                    x[idx, 4] = len(in_tokens[idx])  # count of tokens received at least once at the address
                    x[idx, 5] = len(all_tokens)  # count of tokens transferred at least once in this transaction
                    x[idx, 6] = 1 if len(addr) > 0 and addr != "0x" else 0  # If the address is null address or not
                    x[
                        idx, 7] = 1 if addr in utils.builders.BSC_BUILDER else 0  # If the address is builder address or not
                    x[idx, 8] = is_ca_address  # If the address is CA or EOA
                    x[idx, 9] = 1 if addr == row[
                        'from'] else 0  # If the address is from address of the transaction or not
                    x[idx, 10] = 1 if addr == row['to'] else 0  # If the address is to address of the transaction or not
                    x[idx, 11] = out_count[idx]  # count of transfers sent from the address
                    x[idx, 12] = in_count[idx]  # count of transfers received at the address
                    x[idx, 13] = all_transfers  # count of transfers in this transaction

                edge_index = torch.tensor(edge_list, dtype=torch.long).t().contiguous()
                y = torch.tensor([tx_label], dtype=torch.long)

                data = Data(x=x, edge_index=edge_index, y=y)
                self.graphs.append(data)
                # if len(self.graphs) == 10000:
                #     torch.save(self.graphs, f"train_graphs_{time.time()}.pt")
                #     self.graphs = []


            except:
                label = torch.tensor([0], dtype=torch.long)
                dummy = Data(x=torch.zeros((1, 14)), edge_index=torch.zeros((2, 0), dtype=torch.long), y=label)
                self.graphs.append(dummy)
                continue

        return self.graphs
