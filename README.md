# 融合公平调度与套利识别的区块链MEV拍卖平台研究

### 文件目录

套利交易识别方法的相关文件在：[go/arbitrage_detection](https://github.com/liushunkkk/graduation_thesis/tree/master/go/arbitrage_detection)和[python/arbitrary_detection](https://github.com/liushunkkk/graduation_thesis/tree/master/python/arbitrary_detection)包下

MEV拍卖平台的相关文件在[go/mev_auction](https://github.com/liushunkkk/graduation_thesis/tree/master/go/mev_auction)和[python/mev_auction](https://github.com/liushunkkk/graduation_thesis/tree/master/python/mev_auction)包下



### 实验环境

- 套利交易识别方法的所有实验均在Ubuntu 20.04操作系统下完成，硬件环境为NVIDIA GeForce RTX 3090 GPU（显存 24GB），驱动版本为515.48.07，CUDA版本为11.7。模型的构建与训练基于PyTorch深度学习框架实现。
- MEV拍卖平台的所有实验均在MacBook Air上进行，设备配备Apple M2处理器（8核，4性能核+ 4能效核）和16 GB内存。



### 套利识别对比结果分析

下表展示了三种现有的基线原子套利交易识别对比方法和本文提出的方法的性能结果。

| 模型方法   | TP    | FP    | TN     | FN   |
| ---------- | ----- | ----- | ------ | ---- |
| McLaughlin | 19378 | 85100 | 311456 | 643  |
| Ferreira   | 17541 | 9364  | 387192 | 2480 |
| ArbiNet    | 19399 | 11108 | 385448 | 622  |
| 本文方法   | 19748 | 3437  | 393119 | 273  |

在 BSC 链上，Swap操作的底层通常由多次Transfer事件构成，因此无论交易发生于哪种去中心化交易所，Transfer日志几乎都会被记录。基于Transfer的识别方法（如McLaughlin和ArbiNet）能够覆盖更广泛的交易行为，从而TP数量要更多。但由于Transfer事件也大量出现在普通转账、质押等非套利操作中，这类方法更容易受到噪声影响，导致导致FP数量也相对较高。相较之下，Ferreira方法依赖Swap事件进行识别，尽管不同DEX的Swap事件定义不统一，使其无法覆盖所有交换操作、TP数量较低，但由于事件语义更明确，其对非套利交易的过滤更有效，因此FP数量也较少。

引入深度学习的ArbiNet方法比基于图论的McLaughlin和Ferreira方法在综合表现上更突出，体现了深度学习模型在捕捉复杂交易模式与隐含关系特征方面的优势。然而，ArbiNet的图构建仍依赖于Transfer日志，所能学习到的交易结构信息相对有限；同时，低纬度的节点表征特征，难以刻画更复杂的交互关系，限制了图神经网络的表达能力，因此提升幅度并不显著。相比之下，本文方法在特征工程上进行了更全面的设计：首先，在日志处理中不再局限于Transfer事件，而是对所有链上日志进行统一编码，使模型能够捕捉到图论方法与ArbiNet无法覆盖的更丰富行为信号；其次，将交易输入数据纳入特征体系，使模型能够从函数调用结构和参数组合中识别潜在套利模式；同时，数值特征也在一定程度上反映了交易执行的成本与复杂度。得益于多维特征的协同作用，本文方法在特征表达能力与分类性能上均表现更优，展现出相较现有方法更突出的潜力与优势。

补充表格：

| 模型方法   | Accuracy | Precision | Recall | F1-score |
| ---------- | -------- | --------- | ------ | -------- |
| McLaughlin | 0.794    | 0.185     | 0.968  | 0.311    |
| Ferreira   | 0.972    | 0.652     | 0.877  | 0.748    |
| ArbiNet    | 0.972    | 0.636     | 0.969  | 0.768    |
| 本文方法   | 0.991    | 0.852     | 0.986  | 0.914    |
