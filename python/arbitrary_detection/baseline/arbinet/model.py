from torch_geometric.nn import SAGEConv, global_mean_pool
import torch.nn as nn
import torch.nn.functional as F


class ArbiNetGNN(nn.Module):
    def __init__(self, in_channels=14, hidden_channels=32):
        super().__init__()
        self.conv1 = SAGEConv(in_channels, hidden_channels)
        self.conv2 = SAGEConv(hidden_channels, hidden_channels)
        self.lin = nn.Linear(hidden_channels, 1)  # 单输出

    def forward(self, x, edge_index, batch):
        # 节点卷积
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = self.conv2(x, edge_index)
        x = F.relu(x)
        # 节点聚合 -> 图级表示
        x = global_mean_pool(x, batch)  # [num_graphs, hidden_channels]
        # 图分类输出
        x = self.lin(x)  # [num_graphs, 1]
        return x  # 训练时使用 BCEWithLogitsLoss
