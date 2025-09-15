import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import pandas as pd
from sklearn.metrics import roc_curve, auc, precision_recall_curve, average_precision_score, confusion_matrix, classification_report
import matplotlib.pyplot as plt

# ====== Dataset ======
class TxDataset(Dataset):
    def __init__(self, csv_file):
        df = pd.read_csv(csv_file)
        self.num_features = df['num_feature'].apply(eval).tolist()
        self.data_features = df['data_feature'].apply(eval).tolist()
        self.logs_features = df['logs_feature'].apply(eval).tolist()
        self.labels = df['label'].tolist()

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        num_feat = torch.tensor(self.num_features[idx], dtype=torch.float32)
        data_feat = torch.tensor(self.data_features[idx], dtype=torch.long)
        logs_feat = torch.tensor(self.logs_features[idx], dtype=torch.long)
        label = torch.tensor(self.labels[idx], dtype=torch.float32)
        return num_feat, data_feat, logs_feat, label

# ====== Encoder ======
class TxEncoder(nn.Module):
    def __init__(self, num_dim=8, vocab_size=20000, emb_dim=128, data_len=48, logs_len=64):
        super().__init__()
        self.num_proj = nn.Linear(num_dim, emb_dim)
        self.data_emb = nn.Embedding(vocab_size, emb_dim, padding_idx=0)
        self.logs_emb = nn.Embedding(vocab_size, emb_dim, padding_idx=0)
        self.data_query = nn.Parameter(torch.randn(emb_dim))
        self.logs_query = nn.Parameter(torch.randn(emb_dim))

    def forward(self, num_feat, data_ids, logs_ids):
        # 数值特征投影
        num_emb = self.num_proj(num_feat)  # (B,D)

        # Embedding
        data_emb = self.data_emb(data_ids)  # (B,Ld,D)
        logs_emb = self.logs_emb(logs_ids)  # (B,Ll,D)

        # 单向注意力池化
        def attention_pool(x, query):
            q = query.unsqueeze(0).unsqueeze(1)  # (1,1,D)
            scores = torch.matmul(x, q.transpose(-1, -2)).squeeze(-1)  # (B,L)
            attn = torch.softmax(scores, dim=1).unsqueeze(-1)  # (B,L,1)
            out = (x * attn).sum(1)  # (B,D)
            return out

        data_feat = attention_pool(data_emb, self.data_query)
        logs_feat = attention_pool(logs_emb, self.logs_query)

        # 拼接
        out = torch.cat([num_emb, data_feat, logs_feat], dim=1)
        return out

# ====== 分类器 ======
class TxClassifier(nn.Module):
    def __init__(self, encoder, emb_dim=128, hidden_dim=256):
        super().__init__()
        self.encoder = encoder
        self.mlp = nn.Sequential(
            nn.Linear(3*emb_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, num_feat, data_ids, logs_ids):
        enc = self.encoder(num_feat, data_ids, logs_ids)
        return self.mlp(enc).squeeze(-1)

# ====== 训练并评估 ======
def train_and_evaluate(pos_csv, neg_csv, batch_size=32, epochs=5, lr=1e-3):
    # 合并数据集
    pos_df = pd.read_csv(pos_csv)
    pos_df['label'] = 1
    neg_df = pd.read_csv(neg_csv)
    neg_df['label'] = 0
    df = pd.concat([pos_df, neg_df]).reset_index(drop=True)
    df.to_csv("merged.csv", index=False)

    print("数据合并完成")

    dataset = TxDataset("merged.csv")
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    print('数据加载完成')

    encoder = TxEncoder(num_dim=len(dataset.num_features[0]))
    model = TxClassifier(encoder)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.BCEWithLogitsLoss()

    print('开始训练')

    # ===== 训练 =====
    for epoch in range(epochs):
        total_loss = 0
        for num_feat, data_ids, logs_ids, label in loader:
            optimizer.zero_grad()
            out = model(num_feat, data_ids, logs_ids)
            loss = criterion(out, label)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"Epoch {epoch+1} loss={total_loss/len(loader):.4f}")

    # ===== 评估 =====
    model.eval()
    y_true, y_pred_prob = [], []
    with torch.no_grad():
        for num_feat, data_ids, logs_ids, label in loader:
            out = model(num_feat, data_ids, logs_ids)
            y_pred_prob.extend(torch.sigmoid(out).tolist())
            y_true.extend(label.tolist())

    # 混淆矩阵
    y_pred_label = [1 if p>=0.5 else 0 for p in y_pred_prob]
    cm = confusion_matrix(y_true, y_pred_label)
    tn, fp, fn, tp = cm.ravel()
    print(f"TP: {tp}, FP: {fp}, TN: {tn}, FN: {fn}")
    print(classification_report(y_true, y_pred_label))

    # ROC
    fpr, tpr, _ = roc_curve(y_true, y_pred_prob)
    roc_auc = auc(fpr, tpr)
    plt.figure()
    plt.plot(fpr, tpr, label=f'ROC AUC={roc_auc:.3f}')
    plt.plot([0,1],[0,1],'--',color='gray')
    plt.xlabel('FPR')
    plt.ylabel('TPR')
    plt.title('ROC Curve')
    plt.legend()
    plt.show()

    # PR
    precision, recall, _ = precision_recall_curve(y_true, y_pred_prob)
    ap = average_precision_score(y_true, y_pred_prob)
    plt.figure()
    plt.plot(recall, precision, label=f'AP={ap:.3f}')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve')
    plt.legend()
    plt.show()

    return model

if __name__ == "__main__":
    pos_csv = "./datasets/positive_data.csv"
    neg_csv = "./datasets/negative_data.csv"
    model = train_and_evaluate(pos_csv, neg_csv)