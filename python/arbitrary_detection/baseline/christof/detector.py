import json
import string

import pandas as pd
import requests
from web3 import Web3
from eth_abi import decode


class SwapArbitrageDetector:
    """
    基于 Swap 事件检测原子套利交易（V2 / V3 / DODO V2）
    自动识别 DEX 类型
    """

    # 事件 hash 映射到 DEX 类型
    HASH_TO_DEX = {
        "0xd78ad95fa46c994b6551d0da85fc275fe613ce37657fb8d5e3d130840159d822": "v2",
        # PancakeSwap / Uniswap V2 Swap(address,uint256,uint256,uint256,uint256,address)
        "0xc42079f94a6350d7e6235f29174924f928cc2ac818eb64fed8004e115fbcca67": "v3",
        # Uniswap V3
        # Swap(address,address,int256,int256,uint160,uint128,int24)
        "0x19b47279256b2a23a1665c810c8d55a1758940ee09377d4f8d26497a3577dc83": "v3",
        # Swap(address,address,int256,int256,uint160,uint128,int24,uint128,uint128)
        "0xc2c0245e056d5fb095f04cd6373bc770802ebd1e6c918eb78fdef843cdb37b0f": "dodo_v2",
        # DODO V2 DODOSwap(address,address,uint256,uint256,address,address)
        "0x0e8e403c2d36126272b08c75823e988381d9dc47f2f0a9a080d95f891d95c469": "woo",
        # WooSwap(address,address,uint256,uint256,address,address,address,uint256,uint256)
        "0x2170c741c41531aec20e7c107c24eecfdd15e69c9bb0a8dd37b1840b9e0b207b": "balancer_v2"
        # Swap(bytes32,address,address,uint256,uint256)
    }

    def __init__(self, w3: Web3):
        self.w3 = w3
        self.swap_sig = list()
        df = pd.read_csv("event_signatures.csv")
        # 统一把 hex_signature 转小写
        df["hex_signature"] = df["hex_signature"].str.lower()
        self.hex2text = dict(zip(df["hex_signature"], df["text_signature"]))
        self.pool_cache = {}  # 缓存 pool_address -> (token0, token1)

    def get_text_signature(self, hex_signature: str) -> str:
        """
        根据 hex_signature 查询对应的 text_signature
        :param hex_signature: 事件的16进制签名 (如 0xddf252ad...)
        :return: 对应的 text_signature (如 Transfer(address,address,uint256))，如果没找到返回 None
        """
        return self.hex2text.get(hex_signature, "")

    def get_pool_tokens(self, pool_address):
        # 转为 checksum 地址
        pool_address = Web3.to_checksum_address(pool_address)

        # 先查缓存
        if pool_address in self.pool_cache:
            return self.pool_cache[pool_address]

        try:
            pool_contract = self.w3.eth.contract(
                address=pool_address,
                abi=[
                    {"inputs": [], "name": "token0",
                     "outputs": [{"internalType": "address", "name": "", "type": "address"}],
                     "stateMutability": "view", "type": "function"},
                    {"inputs": [], "name": "token1",
                     "outputs": [{"internalType": "address", "name": "", "type": "address"}],
                     "stateMutability": "view", "type": "function"}
                ]
            )
            token0 = pool_contract.functions.token0().call()
            token1 = pool_contract.functions.token1().call()
            # 缓存结果
            self.pool_cache[pool_address] = (token0, token1)
            return token0, token1
        except Exception as e:
            print(f"Failed to fetch pool tokens for {pool_address}: {e}")
            self.pool_cache[pool_address] = ("", "")
            return "", ""

    # ------------------ V2 Swap ------------------
    def decode_v2_swap(self, log):
        data_bytes = bytes.fromhex(log["data"][2:])
        amount0In, amount1In, amount0Out, amount1Out = decode(
            ['uint256', 'uint256', 'uint256', 'uint256'], data_bytes
        )
        token0, token1 = self.get_pool_tokens(log["address"])
        if not token0 or not token1:
            return []  # 获取失败直接跳过

        print(f"V2 contract: [{log['address']}] => [{token0}], [{token1}]")

        swaps = []
        if amount0In > 0 or amount1Out > 0:
            swaps.append({"in_token": token0, "out_token": token1, "amount_in": amount0In,
                          "amount_out": amount1Out, "dex": log["address"]})
        if amount1In > 0 or amount0Out > 0:
            swaps.append({"in_token": token1, "out_token": token0, "amount_in": amount1In,
                          "amount_out": amount0Out, "dex": log["address"]})
        return swaps

    # ------------------ V3 Swap ------------------
    def decode_v3_swap(self, log):
        # 解析 indexed 参数
        sender = "0x" + log["topics"][1][-40:]
        recipient = "0x" + log["topics"][2][-40:]

        data_bytes = bytes.fromhex(log["data"][2:])
        n_items = len(data_bytes) // 32  # 每个元素 32 字节

        # 根据 data 长度选择参数列表
        if n_items == 5:  # V3 Swap 7 参数版本
            amount0, amount1, sqrtPriceX96, liquidity, tick = decode(
                ['int256', 'int256', 'uint160', 'uint128', 'int24'], data_bytes
            )
        elif n_items == 7:  # V3 Swap 9 参数版本
            amount0, amount1, sqrtPriceX96, liquidity, tick, protocolFeesToken0, protocolFeesToken1 = decode(
                ['int256', 'int256', 'uint160', 'uint128', 'int24', 'uint128', 'uint128'], data_bytes
            )
        else:
            raise ValueError(f"Unknown V3 Swap data length: {n_items * 32} bytes")

        token0, token1 = self.get_pool_tokens(log["address"])
        if not token0 or not token1:
            return []  # 获取失败直接跳过

        print(f"V3 contract: [{log['address']}] => [{token0}], [{token1}]")

        swaps = []
        if amount0 > 0:
            swaps.append({"sender": sender, "recipient": recipient, "in_token": token0, "out_token": token1,
                          "amount_in": amount0, "amount_out": amount1, "dex": log["address"]})
        if amount1 > 0:
            swaps.append({"sender": sender, "recipient": recipient, "in_token": token1, "out_token": token0,
                          "amount_in": amount1, "amount_out": amount0, "dex": log["address"]})

        return swaps

    # ------------------ DODO V2 Swap ------------------
    def decode_dodo_swap(self, log):
        data_bytes = bytes.fromhex(log["data"][2:])
        sender, fromToken, toToken, fromAmount, toAmount, trader = decode(
            ['address', 'address', 'address', 'uint256', 'uint256', 'address'], data_bytes
        )
        print(f"dodo v2 contract: [{log['address']}] => [{fromToken}], [{toToken}]")
        return [{"in_token": fromToken, "out_token": toToken, "amount_in": fromAmount,
                 "amount_out": toAmount, "dex": log["address"]}]

    def decode_woo_swap(self, log):
        # 从 topics 拿 indexed 参数
        fromToken = "0x" + log["topics"][1][-40:]
        toToken = "0x" + log["topics"][2][-40:]
        to_addr = "0x" + log["topics"][3][-40:]

        # 解析 data 中非 indexed 参数
        data_bytes = bytes.fromhex(log["data"][2:])
        fromAmount, toAmount, from_addr, rebateTo, swapVol, swapFee = decode(
            ['uint256', 'uint256', 'address', 'address', 'uint256', 'uint256'],
            data_bytes
        )

        print(f"woo contract: [{log['address']}] => [{fromToken}], [{toToken}]")

        return [{
            "in_token": fromToken,
            "out_token": toToken,
            "amount_in": fromAmount,
            "amount_out": toAmount,
            "sender": from_addr,
            "recipient": to_addr,
            "dex": log["address"],
        }]

    def decode_balancer_v2_swap(self, log):
        pool_id = log["topics"][1]  # bytes32
        token_in = "0x" + log["topics"][2][-40:]
        token_out = "0x" + log["topics"][3][-40:]

        data_bytes = bytes.fromhex(log["data"][2:])
        amount_in, amount_out = decode(['uint256', 'uint256'], data_bytes)

        print(f"balancer v2 contract: [{log['address']}] => [{token_in}], [{token_out}]")

        return [{
            "in_token": token_in,
            "out_token": token_out,
            "amount_in": amount_in,
            "amount_out": amount_out,
            "dex": log["address"]
        }]

    # ------------------ 自动解析 ------------------
    def parse_log(self, log):
        """根据 log 的 hash 自动识别 DEX 并解析"""
        log_hash = log["topics"][0]
        dex_type = self.HASH_TO_DEX.get(log_hash.lower())
        sig = self.get_text_signature(log_hash.lower())
        if "Swap(" in sig:
            self.swap_sig.append(sig)
        if dex_type == "v2":
            return self.decode_v2_swap(log)
        elif dex_type == "v3":
            return self.decode_v3_swap(log)
        elif dex_type == "dodo_v2":
            return self.decode_dodo_swap(log)
        elif dex_type == "woo":
            return self.decode_woo_swap(log)
        elif dex_type == "balancer_v2":
            return self.decode_balancer_v2_swap(log)
        else:
            return []

    # ------------------ 构建套利环 ------------------
    def detect_arbitrage(self, logs_json):
        """
        输入 logs 列表，返回发现的套利环
        """
        if isinstance(logs_json, str):
            logs = json.loads(logs_json)
        else:
            logs = logs_json

        swaps_sequence = []
        for log in logs:
            swaps_sequence.extend(self.parse_log(log))

        arbitrages = []
        used = [False] * len(swaps_sequence)

        for i, swap in enumerate(swaps_sequence):
            if used[i]:
                continue

            chain = [swap]
            used[i] = True
            balances = {swap["in_token"]: -swap["amount_in"], swap["out_token"]: swap["amount_out"]}
            last_out_token = swap["out_token"]
            last_dex = swap["dex"]

            while True:
                extended = False
                for j, next_swap in enumerate(swaps_sequence):
                    if used[j]:
                        continue
                    if (next_swap["in_token"] == last_out_token and
                            next_swap["dex"] != last_dex):
                        # 扩展链
                        chain.append(next_swap)
                        balances[next_swap["in_token"]] = balances.get(next_swap["in_token"], 0) - next_swap[
                            "amount_in"]
                        balances[next_swap["out_token"]] = balances.get(next_swap["out_token"], 0) + next_swap[
                            "amount_out"]
                        last_out_token = next_swap["out_token"]
                        last_dex = next_swap["dex"]
                        used[j] = True
                        extended = True
                        break
                if not extended:
                    break

            # 检查是否形成环
            if chain[0]["in_token"] == last_out_token:
                arbitrages.append({"swaps": chain, "token_balances": balances})

        return {
            "is_arbitrage": len(arbitrages) > 0,
            "arbitrages": arbitrages
        }
