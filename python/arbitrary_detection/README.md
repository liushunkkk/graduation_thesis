# 原子套利交易识别方法

方法依赖见：`requirements.txt`文件。

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

2、数据集拆分：

运行`all_data`包下的`data_split.py`文件,会在该包下生成`datasets`包，并在该包下生成`train_data.csv`和`test_data.csv`文件，并且会针对测试数据集进行扩充，得到`new_test_data.csv`。
就是各个方法需要使用到的最终数据集。

3、然后进行数据集预处理：

依次运行`all_data`包下的`data_processor.py`和`data_expand.py`即可
他会读取`all_data/datasets`文件夹下的`train_data.csv`和`new_test_data.csv`数据文件，并进行数据预处理，
里面会生成`train.csv`和`test.csv`文件，这就是处理后的数据文件，也是最终用于训练和测试的文件。

## 模型训练

### 说明：

针对数值特征以及data和logs的序列特征的预处理已在上一步骤中完成，可直接用于模型训练。

本文针对序列特征生成token嵌入有两种方法：

1. 使用序列模型学习token内的序列关系，得到token的嵌入
2. 使用hash映射的方法，直接映射到固定词表空间

第1种方法模型的学习能力更强大，是本文的默认方法，模型定义在`char_level`包下；

第2种方法模型更简单，训练参数更少，训练起来更快速，但是学习能力会有折扣，模型定义在`hash_embedding`包下。

### 训练运行：

直接运行两个包下的`model_trainer.py`文件即可，可以给模型设置下列参数：
- use_features：使用的特征，可以自由组合num，data和logs，如`["data", "logs"]`
- char_mode：token内信息的学习聚合方式，支持lstm，cnn和mean，lstm是本文最佳
- seq_mode：token间信息的学习聚合方法，支持lstm，cnn，mean和attn_pos，attn_pos带位置参数的注意力机制是本文最佳
- epoch：训练轮数，一般15或20轮即可
- batch_size：批次大小，默认128
- lr：学习率，默认1e-3
- 其他参数可以自行修改源文件，进行自定义

本文的实验通过调整上述参数即可实现

### ETH链

ETH链的数据收集和处理以及模型训练均在`eth_dataset`包下：
- `handler`文件用于从开源数据集的json文件中加载数据到csv文件中
- 收集方式和BSC一致，见`collector`相关文件
- 数据划分和处理也保持一致，和`all_data`包下的文件基本相同，见`data`开头相关文件，
- 模型定义`model_trainer.py`与`char_level`包下保持一致

### 其他：

运行结果将会直接打印在控制台，模型结果会保存在运行目录的`models`文件夹下;

如果需要进行不同链的交叉验证，需要开启`model_trainer.py`下的main函数的`only_test`注释，然后修改模型路径和数据集路径即可