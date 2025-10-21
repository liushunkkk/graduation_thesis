import json
import pandas as pd
from hexbytes import HexBytes
from web3 import Web3
from web3.datastructures import AttributeDict
from tqdm import tqdm

# ===== 连接以太坊节点 =====
w3 = Web3(Web3.HTTPProvider("https://ethereum-mainnet.core.chainstack.com/6f10b9165406508ccd2e4caca9b2285b"))

# ===== 读取原始数据 =====
# 请替换为你的CSV文件路径
input_csv = "../files/latest_eth_data.csv"
df = pd.read_csv(input_csv, dtype=str)
print(f"读取到 {len(df)} 条交易数据")


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


# ===== 补充交易信息 =====
enriched_data = []

for idx, row in tqdm(df.iterrows(), total=len(df), desc="补充交易信息"):
    tx_hash = row['tx_hash'].lower()
    try:
        # 获取交易详情
        tx = w3.eth.get_transaction(tx_hash)
        if not tx:
            raise ValueError("未找到交易记录")

        # 获取交易收据
        receipt = w3.eth.get_transaction_receipt(tx_hash)

        # 构建补充后的记录
        enriched_row = {
            # 保留原始列
            "tx_hash": tx_hash,
            "block_number": row['block_number'],
            "label": row['label'],

            # 补充的交易信息
            "from": tx["from"],
            "to": tx.to,
            "value": tx.value,  # 以wei为单位
            "gas": tx.gas,
            "gas_price": tx.gasPrice,
            "data": "0x" + tx.input.hex(),
            "transaction_index": receipt.transactionIndex,
            "gas_used": receipt.gasUsed,
            "effective_gas_price": getattr(receipt, "effectiveGasPrice", None),
            "logs": to_json(receipt.logs),
        }

        enriched_data.append(enriched_row)

    except Exception as e:
        print(f"\n处理交易 {tx_hash} 失败: {str(e)}")

# ===== 保存结果 =====
output_csv = "../files/enriched_latest_eth_transactions.csv"
df_enriched = pd.DataFrame(enriched_data)
df_enriched.to_csv(output_csv, index=False, encoding="utf-8")

print(f"\n处理完成！成功补充 {len(enriched_data)} 条交易")
print(f"结果已保存至 {output_csv}")
