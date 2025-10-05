import json
import pandas as pd
from hexbytes import HexBytes
from web3 import Web3
from web3.datastructures import AttributeDict

# 连接到以太坊节点（可以用 geth, infura, alchemy 等）
w3 = Web3(Web3.HTTPProvider("https://ethereum-mainnet.core.chainstack.com/6f10b9165406508ccd2e4caca9b2285b"))

data_list = []


def to_json(obj) -> str:
    def convert(o):
        if isinstance(o, HexBytes):
            return "0x" + o.hex()
        if isinstance(o, AttributeDict):
            return {k: convert(v) for k, v in dict(o).items()}
        if isinstance(o, list):
            return [convert(i) for i in o]
        return o

    return json.dumps(convert(obj), ensure_ascii=False)


with open('../files/arbitrage_results.json', 'r', encoding='utf-8') as f:
    for line in f:
        line = line.strip()
        if not line:
            continue
        try:
            data = json.loads(line)

            tx_hash = data['transaction']['hash']
            sender = data['transaction']['from']

            # ===== 获取交易信息 =====
            tx = w3.eth.get_transaction(tx_hash)
            # ===== 获取交易收据 =====
            receipt = w3.eth.get_transaction_receipt(tx_hash)

            row = {
                "tx_hash": "0x" + tx.hash.hex(),
                "block_number": tx.blockNumber,
                "gas_price": tx.gasPrice,
                "gas": tx.gas,
                "to": tx.to,
                "value": tx.value,
                "data": "0x" + tx.input.hex(),  # 交易输入数据
                "from": sender,
                "logs": to_json(receipt.logs),  # 转成字符串存储
                "gas_used": receipt.gasUsed,
                "effective_gas_price": getattr(receipt, "effectiveGasPrice", None),
                "transaction_index": receipt.transactionIndex,
            }

            data_list.append(row)

            print(len(data_list), row)
            if len(data_list) == 100000:
                break

        except json.JSONDecodeError as e:
            print(f"解析错误: {e}，跳过该行")
        except Exception as e:
            print(f"获取交易 {tx_hash} 出错: {e}")

# ===== 存到 CSV =====
df = pd.DataFrame(data_list)
df.to_csv("../files/eth_positive_data.csv", index=False, encoding="utf-8")
print("以保存 ../files/eth_positive_data.csv")

