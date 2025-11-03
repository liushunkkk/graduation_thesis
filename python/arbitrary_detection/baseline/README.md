# baseline说明

mclaughlin包：

McLaughlin, R., Kruegel, C., & Vigna, G. (2023). A Large Scale Study of the Ethereum Arbitrage Ecosystem. *32nd USENIX Security Symposium (USENIX Security 23)*, 3295–3312. https://www.usenix.org/conference/usenixsecurity23/presentation/mclaughlin

christof包：

Christof Ferreira Torres, Albin Mamuti, Ben Weintraub, Cristina Nita-Rotaru, and Shweta Shinde. 2024. Rolling in the Shadows: Analyzing the Extraction of MEV Across Layer-2 Rollups. In Proceedings of the 2024 on ACM SIGSAC Conference on Computer and Communications Security (CCS '24). Association for Computing Machinery, New York, NY, USA, 2591–2605. https://doi.org/10.1145/3658644.3690259

arbinet包：

Park S, Jeong W, Lee Y, et al. Unraveling the MEV enigma: ABI-free detection model using graph neural networks[J]. Future Generation Computer Systems, 2024, 153: 70-83.

## baseline方法原理：

McLaughlin首先通过分析每笔交易中的 ERC-20 Transfer 事件，推断代币交换操作，然后构建有向多重图，并利用 Johnson 算法在图中搜索闭环路径判断是否形成代币买入与卖出的循环，以此作为判定标准；

Christof和McLaughlin不同的是，他分析的是去中心化交易所的swap事件，并基于此构建代币流动的图形结构，以图中存在流动环来决策是否是套利交易。根据论文并结合BSC的[实际市场情况](https://defillama.com/chain/bsc)，识别swap事件在表xx中给出：

| 去中心化交易所事件       | 事件签名                                                     | 事件哈希标识                                                 |
| ------------------------ | ------------------------------------------------------------ | ------------------------------------------------------------ |
| PancakeSwap & Uniswap V2 | Swap(address,uint256,uint256,uint256,uint256,address)        | 0xd78ad95fa46c994b6551d0da85fc275fe613ce37657fb8d5e3d130840159d822 |
| Uniswap V3               | Swap(address,address,int256,int256,uint160,uint128,int24)    | 0xc42079f94a6350d7e6235f29174924f928cc2ac818eb64fed8004e115fbcca67 |
| Uniswap V3               | Swap(address,address,int256,int256,uint160,uint128,int24,uint128,uint128) | 0x19b47279256b2a23a1665c810c8d55a1758940ee09377d4f8d26497a3577dc83 |
| DODO V2                  | DODOSwap(address,address,uint256,uint256,address,address)    | 0xc2c0245e056d5fb095f04cd6373bc770802ebd1e6c918eb78fdef843cdb37b0f |
| Woo                      | WooSwap(address,address,uint256,uint256,address,address,address,uint256,uint256) | 0x0e8e403c2d36126272b08c75823e988381d9dc47f2f0a9a080d95f891d95c469 |
| balancer v2              | Swap(bytes32,address,address,uint256,uint256)                | 0x2170c741c41531aec20e7c107c24eecfdd15e69c9bb0a8dd37b1840b9e0b207b |

ArbiNet提出了一种 ABI-Free 的 GNN 检测方法。该方法以 ERC-20 代币转账事件为唯一输入，将每笔交易中的地址构建为节点，节点之间存在转账关系，并为每个节点的构建14 维特征，最后通过图神经网络执行分类任务。

## 输入格式说明

所有的baseline都是使用的统一的数据集，也就是根据arbitrary_detection根目录下的readme文档中的操作生成的`train.csv`和`test.csv`，具体可以见相关readme文档。
这里面存的数据可以见`files`包下的预览csv文件。
所有baseline的数据的输入格式是一致的，可以通过预览csv文件查看。



## 包说明

mclaughlin包：
- detector：用于分析某个交易，并构建交易图
- handler：用于分析整个数据集，并调用detector进行检测
- runner：执行文件

christof包:
- sig_getter：用于从开源api获取方法的签名，**在执行runner之前，需要执行该文件**
- 其他三个文件的作用和mclaughlin包一致

arbinet包:
- builder: 用于分析单条交易，构建图结构
- model：图神经网络模型定义
- trainer：训练过程定义
- runner：执行文件

## 如何运行

所有的baseline都提供统一的运行入口，也就是各个包下的`runner.py`文件，直接运行该文件即可。

其中前两种方法不需要有训练集，只需要测试集；
后一种方法需要有训练集和测试集；
训练集使用的是正负样本1:2的规模；测试则对测试集进行扩充后，正负样本比例约为1:20，更符合实际数据情况。
最终，测试集中正负样本比例约为1:20（20,021和396,556个）。


## 实验结果

训练集和测试集数据集地址：https://zenodo.org/records/17513600

- 真阳性TP：模型检测为套利交易，实际也为套利交易的样本数量
- 假阳性FP：模型检测为套利交易，实际为非套利交易的样本数量
- 真阴性TN：模型检测为非套利交易，实际也为非套利交易的样本数量
- 假阴性FN：模型检测为非套利交易，实际为套利交易的样本数量
- 准确率Acc：模型检测正确的样本数量占测试数据总样本数量的比率
- 精确率Prec：模型检测为套利交易的样本中，真实套利交易样本所占的比例
- 召回率Recall：测试数据的所有套利交易样本中，被模型预测为套利交易的样本所占的比例
- F1-Score：能够综合评估模型检测的精确率和召回率的指标

测试集中正样本：20,021，负样本：396,556个.
```
McLaughlin:
TP=19378, FP=85100, TN=311456, FN=643 Acc=0.794 Prec=0.185 Recall=0.968, F1=0.311
Christof:
TP=17541, FP=9364, TN=387192, FN=2480 Acc=0.972 Prec=0.652 Recall=0.876, F1=0.748
ArbiNet:
TP=19399, TN=385448, FP=11108, FN=622 Acc=0.972, Prec=0.636, Recall=0.969, F1=0.768
```