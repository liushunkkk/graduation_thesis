# 融合公平调度与套利识别的区块链MEV拍卖平台研究

数据集的完整收集流程在：[go/arbitrage_detection](https://github.com/liushunkkk/graduation_thesis/tree/master/go/arbitrage_detection)

数据集的预处理以及转换为csv文件的过程在：[python/arbitrary_detection](https://github.com/liushunkkk/graduation_thesis/tree/master/python/arbitrary_detection)

此外，baseline的复现以及本文模型的训练等都会在python/arbitrary_detection下进行



实验环境：

- 套利交易识别方法的所有实验均在Ubuntu 20.04操作系统下完成，硬件环境为NVIDIA GeForce RTX 3090 GPU（显存 24GB），驱动版本为515.48.07，CUDA版本为11.7。模型的构建与训练基于PyTorch深度学习框架实现。
- MEV拍卖平台的所有实验均在MacBook Air上进行，设备配备Apple M2处理器（8核，4性能核+ 4能效核）和16 GB内存。
