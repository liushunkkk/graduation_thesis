# baseline说明

mclaughlin包：

McLaughlin, R., Kruegel, C., & Vigna, G. (2023). A Large Scale Study of the Ethereum Arbitrage Ecosystem. *32nd USENIX Security Symposium (USENIX Security 23)*, 3295–3312. https://www.usenix.org/conference/usenixsecurity23/presentation/mclaughlin

christof包：

Christof Ferreira Torres, Albin Mamuti, Ben Weintraub, Cristina Nita-Rotaru, and Shweta Shinde. 2024. Rolling in the Shadows: Analyzing the Extraction of MEV Across Layer-2 Rollups. In Proceedings of the 2024 on ACM SIGSAC Conference on Computer and Communications Security (CCS '24). Association for Computing Machinery, New York, NY, USA, 2591–2605. https://doi.org/10.1145/3658644.3690259

arbinet包：

Park S, Jeong W, Lee Y, et al. Unraveling the MEV enigma: ABI-free detection model using graph neural networks[J]. Future Generation Computer Systems, 2024, 153: 70-83.

## 如何运行？

1、数据集准备

所有的baseline都是使用的统一的数据集，也就是根据总readme文档生成的`train.csv`和`test.csv`，具体可以见总readme文档。
这里面存的数据可以见`files`包下的预览csv文件。

2、运行

所有的baseline都提供统一的运行入口，也就是各个包下的`runner.py`文件，直接运行该文件即可