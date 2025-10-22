import os.path

import pandas as pd
import torch

from baseline.arbinet.builder import ArbiNetTransactionBuilder
from baseline.arbinet.model import ArbiNetGNN
from baseline.arbinet.trainer import ArbiNetTrainer

if __name__ == '__main__':
    TARGET = "all_data"
    device = 'cuda:4' if torch.cuda.is_available() else 'cpu'
    print('Using device:', device)

    # ================= 1. 构建训练图 =================
    print("load train...")
    if os.path.exists("train_graphs1.pt"):
        train_graphs = torch.load("train_graphs1.pt", weights_only=False)  # 加载回来
        print("load from train_graphs.pt")
    else:
        train_df = pd.read_csv(f"../../{TARGET}/datasets/train.csv")
        builder_train = ArbiNetTransactionBuilder(train_df)
        train_graphs = builder_train.build_graphs()
        torch.save(train_graphs, "train_graphs1.pt")  # 保存整个列表
        print("load from train.csv and build")

    # ================= 2. 构建测试图 =================
    print("load test...")
    if os.path.exists("test_graphs2.pt"):
        test_graphs = torch.load("test_graphs2.pt", weights_only=False)  # 加载回来
        print("load from test_graphs.pt")
    else:
        test_df = pd.read_csv(f"../../{TARGET}/datasets/test.csv")
        builder_test = ArbiNetTransactionBuilder(test_df)
        test_graphs = builder_test.build_graphs()
        torch.save(test_graphs, "test_graphs2.pt")  # 保存整个列表
        print("load from test.csv and build")
    print("total train graph: ", len(train_graphs))
    print("total test graph: ", len(test_graphs))

    # ================= 3. 初始化模型 =================
    print("init model")
    model = ArbiNetGNN(in_channels=14, hidden_channels=32).to(device)

    # ================= 4. 初始化 Trainer =================
    print("init trainer")
    trainer = ArbiNetTrainer(model, train_graphs, test_graphs, lr=0.001, device=device)

    # ================= 5. 训练 =================
    print("training...")
    trainer.train(epochs=1, log_interval=1)

    # ================= 6. 测试 =================
    print("testing...")
    metrics, y_pred = trainer.test()

    res = pd.read_csv("../result.csv")
    res["arbinet_result"] = y_pred

    res.to_csv("../result.csv", index=False)

    # ================= 7. 单笔交易预测示例 =================
    # model.eval()
    # with torch.no_grad():
    #     data = train_graph.to(device)
    #     # batch 全置 0 表示单图
    #     batch_vector = torch.zeros(data.x.size(0), dtype=torch.long, device=device)
    #     out = model(data.x, data.edge_index, batch_vector)
    #     pred = (torch.sigmoid(out) > 0.5).long()
    #     print("示例交易预测:", "套利交易" if pred.item() == 1 else "非套利交易")
