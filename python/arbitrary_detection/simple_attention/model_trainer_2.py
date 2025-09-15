import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.metrics import roc_curve, auc, precision_recall_curve, average_precision_score
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
        B = num_feat.size(0)
        num_emb = self.num_proj(num_feat)
        data_emb = self.data_emb(data_ids)
        logs_emb = self.logs_emb(logs_ids)

        def attention_pool(x, query):
            q = query.unsqueeze(0).unsqueeze(1)
            scores = torch.matmul(x, q.transpose(-1, -2)).squeeze(-1)
            attn = torch.softmax(scores, dim=1).unsqueeze(-1)
            return (x * attn).sum(1)

        data_feat = attention_pool(data_emb, self.data_query)
        logs_feat = attention_pool(logs_emb, self.logs_query)

        out = torch.cat([num_emb, data_feat, logs_feat], dim=1)
        return out

# ====== 分类器 ======
class TxClassifier(nn.Module):
    def __init__(self, encoder, emb_dim=128, hidden_dim=256):
        super().__init__()
        self.encoder = encoder
        self.mlp = nn.Sequential(
            nn.Linear(3 * emb_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, num_feat, data_ids, logs_ids):
        enc = self.encoder(num_feat, data_ids, logs_ids)
        out = self.mlp(enc)
        return out.squeeze(-1)

# ====== 评估函数 ======
def evaluate(loader, model, thresholds=None, device=torch.device('cpu')):
    if thresholds is None:
        thresholds = [0.1 * i for i in range(1, 10)]
    model.eval()
    y_true, y_pred_prob = [], []
    with torch.no_grad():
        for num_feat, data_ids, logs_ids, label in loader:
            num_feat = num_feat.to(device)
            data_ids = data_ids.to(device)
            logs_ids = logs_ids.to(device)
            label = label.to(device)

            out = model(num_feat, data_ids, logs_ids)
            y_pred_prob.extend(torch.sigmoid(out).tolist())
            y_true.extend(label.tolist())

    results = []
    for th in thresholds:
        y_pred_label = [1 if p >= th else 0 for p in y_pred_prob]
        cm = confusion_matrix(y_true, y_pred_label)
        tn, fp, fn, tp = cm.ravel()
        acc = accuracy_score(y_true, y_pred_label)
        prec = precision_score(y_true, y_pred_label)
        recall = recall_score(y_true, y_pred_label)
        f1 = f1_score(y_true, y_pred_label)
        results.append((th, tp, fp, tn, fn, acc, prec, recall, f1))
    return results, y_true, y_pred_prob

# ====== 绘制训练曲线 ======
def plot_metrics(train_metrics, val_metrics, metric_name):
    plt.figure()
    plt.plot(train_metrics, label='train')
    plt.plot(val_metrics, label='val')
    plt.xlabel('Epoch')
    plt.ylabel(metric_name)
    plt.title(f'{metric_name} over Epochs')
    plt.legend()
    plt.show()

# ====== 训练函数 ======
def train_model(pos_csv, neg_csv, batch_size=512, epochs=5, lr=1e-3, val_ratio=0.2):
    pos_df = pd.read_csv(pos_csv)
    pos_df['label'] = 1
    neg_df = pd.read_csv(neg_csv)
    neg_df['label'] = 0
    df = pd.concat([pos_df, neg_df]).reset_index(drop=True)
    df.to_csv("merged.csv", index=False)

    dataset = TxDataset("merged.csv")
    val_size = int(len(dataset) * val_ratio)
    train_size = len(dataset) - val_size
    train_set, val_set = random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False)

    encoder = TxEncoder(num_dim=len(dataset.num_features[0]))
    model = TxClassifier(encoder)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.BCEWithLogitsLoss()

    # 保存各指标
    train_acc_list, val_acc_list = [], []
    train_f1_list, val_f1_list = [], []

    # 设置在gpu上工作
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)
    model.to(device)

    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for num_feat, data_ids, logs_ids, label in train_loader:
            num_feat = num_feat.to(device)
            data_ids = data_ids.to(device)
            logs_ids = logs_ids.to(device)
            label = label.to(device)

            optimizer.zero_grad()
            out = model(num_feat, data_ids, logs_ids)
            loss = criterion(out, label)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        # ====== 训练集指标 ======
        train_results, _, _ = evaluate(train_loader, model, device=device)
        val_results, y_true_val, y_pred_prob_val = evaluate(val_loader, model, device=device)

        train_acc_list.append(train_results[4][5])  # 阈值0.5对应的Accuracy
        train_f1_list.append(train_results[4][8])
        val_acc_list.append(val_results[4][5])
        val_f1_list.append(val_results[4][8])

        print(f"Epoch {epoch+1} loss={total_loss/len(train_loader):.4f} "
              f"TrainAcc={train_acc_list[-1]:.3f} TrainF1={train_f1_list[-1]:.3f} "
              f"ValAcc={val_acc_list[-1]:.3f} ValF1={val_f1_list[-1]:.3f}")

    # 绘制训练曲线
    plot_metrics(train_acc_list, val_acc_list, 'Accuracy')
    plot_metrics(train_f1_list, val_f1_list, 'F1-score')

    # 最终测试集 ROC / PR 曲线
    fpr, tpr, _ = roc_curve(y_true_val, y_pred_prob_val)
    roc_auc = auc(fpr, tpr)
    plt.figure()
    plt.plot(fpr, tpr, label=f'ROC AUC={roc_auc:.3f}')
    plt.plot([0,1],[0,1],'--',color='gray')
    plt.xlabel('FPR'); plt.ylabel('TPR'); plt.title('ROC Curve'); plt.legend(); plt.show()

    precision, recall, _ = precision_recall_curve(y_true_val, y_pred_prob_val)
    ap = average_precision_score(y_true_val, y_pred_prob_val)
    plt.figure()
    plt.plot(recall, precision, label=f'AP={ap:.3f}')
    plt.xlabel('Recall'); plt.ylabel('Precision'); plt.title('Precision-Recall Curve'); plt.legend(); plt.show()

    return model

# ====== 使用 ======
if __name__ == "__main__":
    pos_csv = "./datasets/positive_data.csv"
    neg_csv = "./datasets/negative_data.csv"
    model = train_model(pos_csv, neg_csv, epochs=10)