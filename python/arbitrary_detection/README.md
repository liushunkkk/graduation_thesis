# 数据处理与模型训练

data_process: 读取数据库中的数据并保存到csv文件中
- 会读取交易表和收据表然后进行数据的合并
- 需要设置在数据收集阶段保存的mysql数据源
- 最终的结果输出到`files`包下
- 运行`runner.py`文件即可

files：数据基础文件保存包
- 运行`preview.py`文件，即可提取部分数据用于预览
- `positive_data_preview.csv`和`negative_data_preview.csv`即位数据集的预览文件，可以在这里看到每条数据的所有内容
- 每条数据包含如下字段：
  - ['tx_hash', 'tx_type', 'block_number', 'nonce', 'gas_price', 'gas_tip_cap', 'gas_fee_cap', 'gas', 'to', 'value', 'data', 'access_list', 'v', 'r', 's', 'transaction_json', 'from', 'post_state', 'status', 'cumulative_gas_used', 'bloom', 'logs', 'contract_address', 'gas_used', 'effective_gas_price', 'blob_gas_used', 'blob_gas_price', 'block_hash', 'transaction_index', 'receipt_json']


