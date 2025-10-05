import torch
import torch.nn as nn

class ArbiNetTrainer:
    def __init__(self, model, train_loader, test_loader=None, lr=0.01, device='cuda'):
        """
        model: 图分类 GNN 模型
        train_loader: PyG DataLoader 对象（训练集）
        test_loader: PyG DataLoader 对象（测试集，可选）
        device: 'cuda' 或 'cpu'
        """
        self.device = device if torch.cuda.is_available() else 'cpu'
        self.model = model.to(self.device)
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        self.criterion = nn.BCEWithLogitsLoss()  # 二分类

    def train(self, epochs=50, log_interval=1):
        self.model.train()
        for epoch in range(epochs):
            total_loss = 0
            all_pred = []
            all_true = []
            for batch in self.train_loader:
                batch = batch.to(self.device)
                self.optimizer.zero_grad()
                out = self.model(batch.x, batch.edge_index, batch.batch)  # [batch_size, 1]
                loss = self.criterion(out.view(-1), batch.y.float())
                loss.backward()
                self.optimizer.step()
                total_loss += loss.item()

                pred = (torch.sigmoid(out) > 0.5).long()
                all_pred.append(pred.cpu())
                all_true.append(batch.y.cpu())

            all_pred = torch.cat(all_pred)
            all_true = torch.cat(all_true)
            metrics = self.evaluate_metrics(all_pred, all_true)

            if epoch % log_interval == 0:
                print(f"Epoch {epoch} | Loss: {total_loss:.4f} | "
                      f"Acc: {metrics['accuracy']:.4f} | "
                      f"Precision: {metrics['precision']:.4f} | "
                      f"Recall: {metrics['recall']:.4f} | "
                      f"F1: {metrics['f1']:.4f} | "
                      f"TP: {metrics['tp']} | TN: {metrics['tn']} | "
                      f"FP: {metrics['fp']} | FN: {metrics['fn']}")

    def test(self):
        if self.test_loader is None:
            print("No test loader provided.")
            return None
        self.model.eval()
        all_pred = []
        all_true = []
        with torch.no_grad():
            for batch in self.test_loader:
                batch = batch.to(self.device)
                out = self.model(batch.x, batch.edge_index, batch.batch)
                pred = (torch.sigmoid(out) > 0.5).long()
                all_pred.append(pred.cpu())
                all_true.append(batch.y.cpu())

        all_pred = torch.cat(all_pred)
        all_true = torch.cat(all_true)
        metrics = self.evaluate_metrics(all_pred, all_true)
        print(f"Test Metrics | Acc: {metrics['accuracy']:.4f} | "
              f"Precision: {metrics['precision']:.4f} | Recall: {metrics['recall']:.4f} | "
              f"F1: {metrics['f1']:.4f} | TP: {metrics['tp']} | TN: {metrics['tn']} | "
              f"FP: {metrics['fp']} | FN: {metrics['fn']}")
        return metrics

    @staticmethod
    def evaluate_metrics(pred, true):
        tp = ((pred == 1) & (true == 1)).sum().item()
        tn = ((pred == 0) & (true == 0)).sum().item()
        fp = ((pred == 1) & (true == 0)).sum().item()
        fn = ((pred == 0) & (true == 1)).sum().item()

        accuracy = (tp + tn) / (tp + tn + fp + fn + 1e-9)
        precision = tp / (tp + fp + 1e-9)
        recall = tp / (tp + fn + 1e-9)
        f1 = 2 * precision * recall / (precision + recall + 1e-9)

        return {
            'tp': tp,
            'tn': tn,
            'fp': fp,
            'fn': fn,
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1
        }