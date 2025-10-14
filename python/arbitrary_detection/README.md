# 原子套利交易识别方法

方法依赖见：`requirements.txt`文件
此外，从数据库导出文件，还需要用到`pymysql`库，直接通过pip安装即可。

## 数据处理

1、首先进行数据集文件导出：

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

2、然后进行数据集预处理：

依次运行`all_data`包下的`data_processor.py`和`data_expand.py`即可
他会读取files文件夹下的数据文件，并进行数据预处理，并会在all_data包下生成`datasets`文件夹
里面会生成`negative_data.csv`和`positive_data.csv`文件，这就是处理后的数据文件

3、数据集拆分：

运行`all_data`包下的`data_split.py`文件,会在步骤2中的datasets包下生成`train.csv`和`test.csv`文件，
就是各个方法需要使用到的最终数据集

