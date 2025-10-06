import json

import numpy as np
import pandas as pd
import torch
from flask import Flask, request, jsonify
from pydantic import BaseModel, Field
from typing import List

from sklearn.preprocessing import StandardScaler

import char_level.model_trainer

app = Flask(__name__)


def load_model(load_path):
    """加载完整模型"""
    # 加载完整模型
    model = torch.load(load_path)
    model.eval()  # 设置为评估模式
    print(f"模型已从 {load_path} 加载")
    return model


save_path = "arbi_model.pth"
loaded_model: char_level.model_trainer.TxClassifier = load_model(save_path)


def serialize_single_data(data_hex: str):
    """Data 序列化 → token list"""
    s = data_hex.lower().replace("0x", "")
    tokens = []

    # 函数选择器
    if len(s) >= 8:
        tokens.append(s[:8])
    # 参数槽位
    for i in range(8, len(s), 64):
        tokens.append(s[i:i + 64])
    return tokens


def serialize_single_log(log: dict):
    s = ""
    for t in log.get("topics", []):
        s += t.lower().replace("0x", "")
    s += log.get("data", "").lower().replace("0x", "")
    tokens = [s[i:i + 64] for i in range(0, len(s), 64)]
    return tokens


def logs_to_tokens(logs_list):
    all_tokens = []
    for log_item in logs_list:
        all_tokens += serialize_single_log(log_item)

    return all_tokens


# 解析JSON字符串的工具函数
def parse_json_safely(json_str: str) -> dict:
    """安全解析JSON字符串，返回字典或空字典"""
    try:
        return json.loads(json_str)
    except json.JSONDecodeError as e:
        print(f"JSON解析错误: {e}")
        return {}


# 定义单个检测的输入结构体
class ArbiRequest(BaseModel):
    txHash: str = Field(..., description="交易Hash")
    txJson: str = Field(..., description="交易json序列化数据")
    receiptJson: str = Field(..., description="收据json序列化数据")


# 定义单个检测的输出结构体
class ArbiResult(BaseModel):
    success: bool
    txHash: str
    arbitragePercent: float


# 定义批量检测的输出结构体
class BatchArbiResult(BaseModel):
    successItems: int
    results: List[ArbiResult]


# 处理单个套利检测
def process_single_arbi(request_data: ArbiRequest) -> ArbiResult:
    # 实际业务逻辑
    arbitrage_percent = 0.5

    # 解析txJson和receiptJson
    tx_data = parse_json_safely(request_data.txJson)
    receipt_data = parse_json_safely(request_data.receiptJson)

    data = {
        'tx_hash': request_data.txHash,
        'gas': tx_data['gas'],
        'data': tx_data['input'],
        'logs': receipt_data['logs'],
        'gas_used': receipt_data['gasUsed']}
    df = pd.DataFrame([data])  # 用列表包裹字典，确保是一行数据

    # 示例：日志数量作为特征
    df['logs_token'] = df['logs'].apply(logs_to_tokens)
    df["data_token"] = df["data"].apply(serialize_single_data)

    df['data_len'] = df['data_token'].apply(lambda x: len(x))
    df['logs_len'] = df['logs'].apply(lambda x: len(x))

    df['gas'] = pd.to_numeric(df['gas'], errors='coerce').fillna(0)
    df['gas_used'] = pd.to_numeric(df['gas_used'], errors='coerce').fillna(0)
    df['gas'] = np.log1p(df['gas'])
    df['gas_used'] = np.log1p(df['gas_used'])
    df['data_len'] = np.log1p(df['data_len'])
    df['logs_len'] = np.log1p(df['logs_len'])
    df[['gas', 'gas_used', 'data_len', 'logs_len']] = StandardScaler().fit_transform(
        df[['gas', 'gas_used', 'data_len', 'logs_len']])

    # 如果是PyTorch模型，转换为张量
    input_tensor = torch.tensor(df.values, dtype=torch.float32)

    with torch.no_grad():
        output = loaded_model(input_tensor)
        print("模型输出:", output)

    return ArbiResult(
        success=True,
        txHash=request_data.txHash,
        arbitragePercent=output[0].item
    )


# 单个套利检测接口
@app.route('/detectArbi', methods=['POST'])
def detect_arbi():
    try:
        # 解析并验证请求数据
        request_data = ArbiRequest(**request.get_json())

        # 处理请求
        result = process_single_arbi(request_data)

        # 返回结果
        return jsonify(result.dict())
    except Exception as e:
        return jsonify({"error": str(e)}), 400


# 批量套利检测接口
@app.route('/batchDetectArbi', methods=['POST'])
def batch_detect_arbi():
    try:
        # 解析并验证批量请求数据
        batch_data = [ArbiRequest(**item) for item in request.get_json()]

        # 处理批量请求
        results = [process_single_arbi(item) for item in batch_data]
        success_count = sum(1 for res in results if res.success)

        # 构建批量结果
        batch_result = BatchArbiResult(
            successItems=success_count,
            results=results
        )

        return jsonify(batch_result.dict())
    except Exception as e:
        return jsonify({"error": str(e)}), 400


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
