import json
import random
import pandas as pd
from hexbytes import HexBytes
from web3 import Web3
from web3.datastructures import AttributeDict
from tqdm import tqdm

# ===== 连接以太坊节点 =====
w3 = Web3(Web3.HTTPProvider("https://ethereum-mainnet.core.chainstack.com/6f10b9165406508ccd2e4caca9b2285b"))

# ===== 读取正样本数据 =====
df_pos = pd.read_csv("../files/eth_positive_data.csv", dtype=str)

# 分组统计：block_number -> 套利交易hash列表
arbi_by_block = (
    df_pos.groupby("block_number")["tx_hash"]
    .apply(lambda x: [h.lower() for h in x])
    .to_dict()
)

print(f"共发现 {len(arbi_by_block)} 个区块包含套利交易")


# ===== 辅助函数 =====
def to_json(obj) -> str:
    """递归转json-friendly对象"""

    def convert(o):
        if isinstance(o, HexBytes):
            return "0x" + o.hex()
        if isinstance(o, AttributeDict):
            return {k: convert(v) for k, v in dict(o).items()}
        if isinstance(o, list):
            return [convert(i) for i in o]
        return o

    return json.dumps(convert(obj), ensure_ascii=False)


# ===== 收集负样本 =====
neg_data = []

for block_number, arbi_txs in tqdm(arbi_by_block.items(), desc="Collecting negatives"):
    try:
        block_number_int = int(block_number)
        print("handling block number", block_number_int)
        block = w3.eth.get_block(block_number_int, full_transactions=True)
        all_txs = block.transactions

        # 所有交易哈希
        all_hashes = [("0x" + tx.hash.hex().lower()) for tx in all_txs]

        # 剩余非套利交易
        non_arbi_hashes = [h for h in all_hashes if h not in arbi_txs]

        # 过滤非空 input 的交易
        valid_txs = [tx for tx in all_txs if (
                "0x" + tx.hash.hex().lower()) in non_arbi_hashes and tx.input.hex() != "" and tx.input.hex() != "0x"]

        n = len(arbi_txs)
        if n == 0 or len(valid_txs) == 0:
            continue

        sample_size = min(2 * n, len(valid_txs))
        selected_txs = random.sample(valid_txs, sample_size)

        for tx in selected_txs:
            try:
                receipt = w3.eth.get_transaction_receipt(tx.hash)
                row = {
                    "tx_hash": "0x" + tx.hash.hex(),
                    "block_number": tx.blockNumber,
                    "gas_price": tx.gasPrice,
                    "gas": tx.gas,
                    "to": tx.to,
                    "value": tx.value,
                    "data": "0x" + tx.input.hex(),
                    "from": tx["from"],
                    "logs": to_json(receipt.logs),
                    "gas_used": receipt.gasUsed,
                    "effective_gas_price": getattr(receipt, "effectiveGasPrice", None),
                    "transaction_index": receipt.transactionIndex,
                }
                neg_data.append(row)
            except Exception as e:
                print(f"获取交易 {tx.hash.hex()} 失败: {e}")

    except Exception as e:
        print(f"区块 {block_number} 处理失败: {e}")

# ===== 保存负样本 =====
df_neg = pd.DataFrame(neg_data)
df_neg.to_csv("../files/eth_negative_data.csv", index=False, encoding="utf-8")
print(f"已保存负样本至 ../files/eth_negative_data.csv，共 {len(df_neg)} 条记录。")
