import os
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import matplotlib.pyplot as plt

# ====== 工具函数 ======
HEX_MAP = {c: i for i, c in enumerate("0123456789abcdef")}


def hex_to_tensor_list(hex_list, token_len=64):
    """将token序列转为字符索引矩阵，每个token由若干字符组成"""
    tokens_idx = []
    for token in hex_list:
        s = token.lower()
        s = s[:token_len].ljust(token_len, "0")  # 截断或填充到固定长度
        tokens_idx.append([HEX_MAP.get(ch, 0) for ch in s])
    return tokens_idx


def pad_tokens(tokens, max_len, token_len):
    """对 token 列表进行 pad，确保长度为 max_len，每个 token 长度为 token_len"""
    pad_token = [0] * token_len
    if len(tokens) < max_len:
        tokens = tokens + [pad_token] * (max_len - len(tokens))
    else:
        tokens = tokens[:max_len]
    return tokens


class TxDataset(Dataset):
    def __init__(self, csv_file, token_len=64, max_data_tokens=48, max_logs_tokens=64):
        df = pd.read_csv(csv_file)
        selected_idx = [3, 5, 8, 9]
        # 原始 num_feature 是字符串形式，需要先 eval，再筛选
        self.num_features = df['num_feature'].apply(lambda x: [eval(x)[i] for i in selected_idx]).tolist()
        self.labels = df['label'].tolist()

        # 直接读取 data_token / logs_token（无 0x 前缀）
        self.data_tokens = df['data_token'].apply(eval).tolist()
        self.logs_tokens = df['logs_token'].apply(eval).tolist()

        # pad 到固定长度
        self.data_tokens = [pad_tokens(hex_to_tensor_list(t, token_len), max_data_tokens, token_len) for t in
                            self.data_tokens]
        self.logs_tokens = [pad_tokens(hex_to_tensor_list(t, token_len), max_logs_tokens, token_len) for t in
                            self.logs_tokens]

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        num_feat = torch.tensor(self.num_features[idx], dtype=torch.float32)
        data_tensor = torch.tensor(self.data_tokens[idx], dtype=torch.long)
        logs_tensor = torch.tensor(self.logs_tokens[idx], dtype=torch.long)
        label = torch.tensor(self.labels[idx], dtype=torch.float32)
        return num_feat, data_tensor, logs_tensor, label


# ====== CharEncoder ======
class CharEncoder(nn.Module):
    """对单个token的字符序列编码为一个向量"""

    def __init__(self, char_emb_dim=16, out_dim=128, mode='mean'):
        super().__init__()
        self.mode = mode
        self.char_emb = nn.Embedding(16, char_emb_dim)
        if mode == 'cnn':
            self.encoder = nn.Sequential(
                nn.Conv1d(char_emb_dim, out_dim, kernel_size=3, padding=1),
                nn.ReLU(),
                nn.AdaptiveMaxPool1d(1)
            )
        elif mode == 'lstm':
            self.lstm = nn.LSTM(char_emb_dim, out_dim, batch_first=True)
        else:  # mean pooling
            self.out_proj = nn.Linear(char_emb_dim, out_dim)

    def forward(self, x):
        emb = self.char_emb(x)  # (B, L, C)
        if self.mode == 'cnn':
            out = self.encoder(emb.transpose(1, 2)).squeeze(-1)
        elif self.mode == 'lstm':
            _, (h, _) = self.lstm(emb)
            out = h.squeeze(0)
        else:
            out = self.out_proj(emb.mean(dim=1))
        return out


# ====== Encoder ======
class TxEncoder(nn.Module):
    def __init__(self, num_dim=8, emb_dim=128, use_features=None,
                 token_len=64, data_max=48, logs_max=64):
        super().__init__()
        if use_features is None:
            use_features = ["num", "data", "logs"]
        self.use_features = use_features

        if "num" in use_features:
            self.num_proj = nn.Linear(num_dim, emb_dim // 4)

        if "data" in use_features:
            self.data_token_encoder = CharEncoder(out_dim=emb_dim)
            self.data_attn = nn.MultiheadAttention(emb_dim, num_heads=4, batch_first=True)
            self.norm_data = nn.LayerNorm(emb_dim)

        if "logs" in use_features:
            self.logs_token_encoder = CharEncoder(out_dim=emb_dim)
            self.logs_attn = nn.MultiheadAttention(emb_dim, num_heads=4, batch_first=True)
            self.norm_logs = nn.LayerNorm(emb_dim)

    def forward(self, num_feat, data_tokens, logs_tokens):
        outs = []
        if "num" in self.use_features:
            outs.append(self.num_proj(num_feat))

        if "data" in self.use_features:
            B, T, L = data_tokens.shape
            data_flat = data_tokens.view(B * T, L)
            data_token_vecs = self.data_token_encoder(data_flat).view(B, T, -1)
            out, _ = self.data_attn(data_token_vecs, data_token_vecs, data_token_vecs)
            outs.append(self.norm_data(out.mean(dim=1)))

        if "logs" in self.use_features:
            B, T, L = logs_tokens.shape
            logs_flat = logs_tokens.view(B * T, L)
            logs_token_vecs = self.logs_token_encoder(logs_flat).view(B, T, -1)
            out, _ = self.logs_attn(logs_token_vecs, logs_token_vecs, logs_token_vecs)
            outs.append(self.norm_logs(out.mean(dim=1)))

        return torch.cat(outs, dim=1)


# ====== 分类器 ======
class TxClassifier(nn.Module):
    def __init__(self, encoder, emb_dim=128, hidden_dim=256):
        super().__init__()
        self.encoder = encoder
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
        return self.mlp(enc).squeeze(-1)


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


# ====== 训练函数 ======
def train_model(pos_csv, neg_csv, use_features=None,
                batch_size=128, epochs=10, lr=1e-3, test_ratio=0.2):
    if use_features is None:
        use_features = ["num", "data", "logs"]

    if not os.path.exists(f"../{TARGET}/datasets/train.csv"):
        pos_df = pd.read_csv(pos_csv)
        pos_df['label'] = 1
        neg_df = pd.read_csv(neg_csv)
        neg_df['label'] = 0
        df = pd.concat([pos_df, neg_df]).sort_values("block_number").reset_index(drop=True)
        test_size = int(len(df) * test_ratio)
        df.iloc[:-test_size].to_csv(f"../{TARGET}/datasets/train.csv", index=False)
        df.iloc[-test_size:].to_csv(f"../{TARGET}/datasets/test.csv", index=False)

    train_set = TxDataset(f"../{TARGET}/datasets/train.csv")
    test_set = TxDataset(f"../{TARGET}/datasets/test.csv")

    print("data loaded...", use_features)
    print("training set: ", len(train_set))
    print("testing set: ", len(test_set))

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_set, batch_size=batch_size)

    encoder = TxEncoder(num_dim=len(train_set.num_features[0]), use_features=use_features)
    model = TxClassifier(encoder)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.BCEWithLogitsLoss()

    # 保存各指标
    train_acc_list, val_acc_list = [], []
    train_f1_list, val_f1_list = [], []

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for num_feat, data_ids, logs_ids, label in train_loader:
            num_feat, data_ids, logs_ids, label = num_feat.to(device), data_ids.to(device), logs_ids.to(
                device), label.to(device)
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

    print("Training finished.")
    return model


if __name__ == "__main__":
    global TARGET
    TARGET = "all_data"  # half_data or all_data
    pos_csv = f"../{TARGET}/datasets/positive_data.csv"
    neg_csv = f"../{TARGET}/datasets/negative_data.csv"
    model = train_model(pos_csv, neg_csv, use_features=["logs", "data", "num"], epochs=20)
