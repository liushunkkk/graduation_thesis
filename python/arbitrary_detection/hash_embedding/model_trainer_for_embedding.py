import os

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
        # [gas_price,gas_tip_cap,gas_fee_cap,gas,value,gas_used,effective_gas_price,transaction_index,data_len,logs_len,transfer_len]
        selected_idx = [3, 5, 8, 9, 10]
        # 原始 num_feature 是字符串形式，需要先 eval，再筛选
        self.num_features = df['num_feature'].apply(lambda x: [eval(x)[i] for i in selected_idx]).tolist()
        # self.num_features = df['num_feature'].apply(eval).tolist()
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
    def __init__(self, num_dim=8, data_vocab_size=60000, logs_vocab_size=80000,
                 emb_dim=128, data_len=48, logs_len=64, mode='attention', n_heads=4, use_features=None):
        super().__init__()
        if use_features is None:
            use_features = ["num", "data", "logs"]
        self.use_features = use_features
        self.emb_dim = emb_dim
        self.mode = mode
        self.n_heads = n_heads

        print(self.use_features)

        # 数值特征线性投影
        if "num" in use_features:
            self.num_proj = nn.Linear(num_dim, emb_dim // 4)

        # 数据 embedding
        if "data" in use_features:
            self.data_emb = nn.Embedding(data_vocab_size, emb_dim, padding_idx=0)

        # 日志 embedding
        if "logs" in use_features:
            self.logs_emb = nn.Embedding(logs_vocab_size, emb_dim, padding_idx=0)

        # Attention 单向
        if mode == 'attention':
            if "data" in use_features:
                self.data_query = nn.Parameter(torch.randn(emb_dim))
            if "logs" in use_features:
                self.logs_query = nn.Parameter(torch.randn(emb_dim))

        # LSTM
        elif mode == 'lstm':
            if "data" in use_features:
                self.data_lstm = nn.LSTM(input_size=emb_dim, hidden_size=emb_dim, batch_first=True)
            if "logs" in use_features:
                self.logs_lstm = nn.LSTM(input_size=emb_dim, hidden_size=emb_dim, batch_first=True)

        # CNN
        elif mode == 'cnn':
            if "data" in use_features:
                self.data_cnn = nn.Sequential(
                    nn.Conv1d(emb_dim, emb_dim, kernel_size=3, padding=1),
                    nn.ReLU(),
                    nn.AdaptiveMaxPool1d(1)
                )
            if "logs" in use_features:
                self.logs_cnn = nn.Sequential(
                    nn.Conv1d(emb_dim, emb_dim, kernel_size=3, padding=1),
                    nn.ReLU(),
                    nn.AdaptiveMaxPool1d(1)
                )

        # Multihead attention
        elif mode in ['multihead', 'multihead_pos']:
            if "data" in use_features:
                self.data_attn = nn.MultiheadAttention(embed_dim=emb_dim, num_heads=n_heads, batch_first=True)
                self.norm_data = nn.LayerNorm(emb_dim)
            if "logs" in use_features:
                self.logs_attn = nn.MultiheadAttention(embed_dim=emb_dim, num_heads=n_heads, batch_first=True)
                self.norm_logs = nn.LayerNorm(emb_dim)
            # 位置编码
            if mode == 'multihead_pos':
                if "data" in use_features:
                    self.data_pos = nn.Parameter(torch.randn(data_len, emb_dim))
                if "logs" in use_features:
                    self.logs_pos = nn.Parameter(torch.randn(logs_len, emb_dim))

    def forward(self, num_feat, data_ids, logs_ids):
        outs = []
        B = num_feat.size(0) if num_feat is not None else (
            data_ids.size(0) if data_ids is not None else logs_ids.size(0))

        if "num" in self.use_features:
            num_emb = self.num_proj(num_feat)
            outs.append(num_emb)

        if "data" in self.use_features:
            data_emb = self.data_emb(data_ids)  # (B, L, D)
            if self.mode == 'attention':
                q = self.data_query.unsqueeze(0).unsqueeze(1)
                scores = torch.matmul(data_emb, q.transpose(-1, -2)).squeeze(-1)
                attn = torch.softmax(scores, dim=1).unsqueeze(-1)
                data_feat = (data_emb * attn).sum(1)
            elif self.mode == 'lstm':
                out, (h_n, _) = self.data_lstm(data_emb)
                data_feat = h_n.squeeze(0)
            elif self.mode == 'cnn':
                data_feat = self.data_cnn(data_emb.transpose(1, 2)).squeeze(-1)
            elif self.mode in ['multihead', 'multihead_pos']:
                if self.mode == 'multihead_pos':
                    data_emb = data_emb + self.data_pos.unsqueeze(0)
                out, _ = self.data_attn(data_emb, data_emb, data_emb)
                data_feat = self.norm_data(out.mean(dim=1))
            outs.append(data_feat)

        if "logs" in self.use_features:
            logs_emb = self.logs_emb(logs_ids)
            if self.mode == 'attention':
                q = self.logs_query.unsqueeze(0).unsqueeze(1)
                scores = torch.matmul(logs_emb, q.transpose(-1, -2)).squeeze(-1)
                attn = torch.softmax(scores, dim=1).unsqueeze(-1)
                logs_feat = (logs_emb * attn).sum(1)
            elif self.mode == 'lstm':
                out, (h_n, _) = self.logs_lstm(logs_emb)
                logs_feat = h_n.squeeze(0)
            elif self.mode == 'cnn':
                logs_feat = self.logs_cnn(logs_emb.transpose(1, 2)).squeeze(-1)
            elif self.mode in ['multihead', 'multihead_pos']:
                if self.mode == 'multihead_pos':
                    logs_emb = logs_emb + self.logs_pos.unsqueeze(0)
                out, _ = self.logs_attn(logs_emb, logs_emb, logs_emb)
                logs_feat = self.norm_logs(out.mean(dim=1))
            outs.append(logs_feat)

        return torch.cat(outs, dim=1)


# ====== 分类器 ======
class TxClassifier(nn.Module):
    def __init__(self, encoder, emb_dim=128, hidden_dim=256):
        super().__init__()
        self.encoder = encoder

        # 输入维度取决于 use_features 数量
        in_dim = len(encoder.use_features) * emb_dim
        if 'num' in encoder.use_features:
            in_dim -= emb_dim
            in_dim += emb_dim // 4

        self.mlp = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
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
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred_label).ravel()
        acc = accuracy_score(y_true, y_pred_label)
        prec = precision_score(y_true, y_pred_label, zero_division=0)
        recall = recall_score(y_true, y_pred_label, zero_division=0)
        f1 = f1_score(y_true, y_pred_label, zero_division=0)
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
def train_model(pos_csv, neg_csv, use_features=None, mode='cnn', data_vocab_size=60000, logs_vocab_size=80000,
                batch_size=512, epochs=5, lr=1e-3, test_ratio=0.2):
    if use_features is None:
        use_features = ["num", "data", "logs"]
    if not os.path.exists(f"../{TARGET}/datasets/train.csv"):
        print("Loading data...")
        pos_df = pd.read_csv(pos_csv)
        pos_df['label'] = 1
        neg_df = pd.read_csv(neg_csv)
        neg_df['label'] = 0
        print("Loading data 完成...")

        print("按照block_number排序后，拆分数据集...")
        # 合并并排序
        df = pd.concat([pos_df, neg_df]).sort_values(by="block_number").reset_index(drop=True)

        # 划分
        test_size = int(len(df) * test_ratio)
        train_df = df.iloc[:-test_size]
        test_df = df.iloc[-test_size:]

        # 保存
        train_df.to_csv(f"../{TARGET}/datasets/train.csv", index=False)
        test_df.to_csv(f"../{TARGET}/datasets/test.csv", index=False)
        print(f"保存完成: ../{TARGET}/datasets/train.csv {len(train_df)} 条, ../{TARGET}/datasets/test.csv {len(test_df)} 条")

    print("load train and test dataset...")
    train_set = TxDataset(f"../{TARGET}/datasets/train.csv")
    test_set = TxDataset(f"../{TARGET}/datasets/test.csv")

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False)

    encoder = TxEncoder(data_vocab_size=data_vocab_size, logs_vocab_size=logs_vocab_size,
                        num_dim=len(train_set.num_features[0]), use_features=use_features, mode=mode)
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

        # ====== 训练集和测试集指标 ======
        train_results, _, _ = evaluate(train_loader, model, device=device)
        test_results, y_true_test, y_pred_prob_test = evaluate(test_loader, model, device=device)

        # 取阈值 0.5 的指标（results[4]）
        th, tp, fp, tn, fn, acc, prec, recall, f1 = train_results[4]
        th_t, tp_t, fp_t, tn_t, fn_t, acc_t, prec_t, recall_t, f1_t = test_results[4]

        train_acc_list.append(acc)
        train_f1_list.append(f1)
        val_acc_list.append(acc_t)
        val_f1_list.append(f1_t)

        print(f"Epoch {epoch + 1} loss={total_loss / len(train_loader):.4f}")
        print(f"  Train: TP={tp}, TN={tn}, FP={fp}, FN={fn}, "
              f"Acc={acc:.3f}, Prec={prec:.3f}, Recall={recall:.3f}, F1={f1:.3f}")
        print(f"  Test : TP={tp_t}, TN={tn_t}, FP={fp_t}, FN={fn_t}, "
              f"Acc={acc_t:.3f}, Prec={prec_t:.3f}, Recall={recall_t:.3f}, F1={f1_t:.3f}")

    # 绘制训练曲线
    plot_metrics(train_acc_list, val_acc_list, 'Accuracy')
    plot_metrics(train_f1_list, val_f1_list, 'F1-score')

    # 最终测试集 ROC / PR 曲线
    fpr, tpr, _ = roc_curve(y_true_test, y_pred_prob_test)
    roc_auc = auc(fpr, tpr)
    plt.figure()
    plt.plot(fpr, tpr, label=f'ROC AUC={roc_auc:.3f}')
    plt.plot([0, 1], [0, 1], '--', color='gray')
    plt.xlabel('FPR')
    plt.ylabel('TPR')
    plt.title('ROC Curve')
    plt.legend()
    plt.show()

    precision, recall, _ = precision_recall_curve(y_true_test, y_pred_prob_test)
    ap = average_precision_score(y_true_test, y_pred_prob_test)
    plt.figure()
    plt.plot(recall, precision, label=f'AP={ap:.3f}')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve')
    plt.legend()
    plt.show()

    return model


# ====== 使用 ======
if __name__ == "__main__":
    global TARGET
    TARGET = "all_data"  # half_data or all_data
    pos_csv = f"../{TARGET}/datasets/positive_data.csv"
    neg_csv = f"../{TARGET}/datasets/negative_data.csv"
    # attention, cnn, lstm, multihead, multihead_pos
    model = train_model(pos_csv, neg_csv, batch_size=128, mode='multihead', use_features=["num", "data", "logs"],
                        epochs=30)
