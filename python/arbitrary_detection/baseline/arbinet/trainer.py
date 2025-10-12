import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm  # 用于显示进度条，需安装：pip install tqdm


class ArbiNetTrainer:
    def __init__(self, model, train_loader, test_loader=None, lr=0.001, weight_decay=1e-5, device='cuda'):
        """
        改进点：
        1. 增加权重衰减(weight_decay)抑制过拟合
        2. 优化设备选择逻辑
        """
        self.device = device if (device == 'cuda' and torch.cuda.is_available()) else 'cpu'
        self.model = model.to(self.device)
        self.train_loader = train_loader
        self.test_loader = test_loader

        # 优化器增加权重衰减，学习率默认值调小
        self.optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=lr,
            weight_decay=weight_decay
        )

        # 二分类损失函数，支持标签为长整型
        self.criterion = nn.BCEWithLogitsLoss()

    def train(self, epochs=50, log_interval=1, gradient_accumulation_steps=1):
        """
        改进点：
        1. 增加梯度累积，支持大batch效果
        2. 使用tqdm显示进度条，直观查看训练速度
        3. 修复指标计算逻辑，确保维度匹配
        """
        self.model.train()
        for epoch in range(epochs):
            total_loss = 0.0
            all_pred = []
            all_true = []

            # 使用tqdm显示批量处理进度
            loop = tqdm(self.train_loader, total=len(self.train_loader), leave=False)
            loop.set_description(f"Epoch [{epoch}/{epochs}]")

            for step, batch in enumerate(loop):
                batch = batch.to(self.device)

                # 模型输出：[batch_size, 1]
                out = self.model(batch.x, batch.edge_index, batch.batch)
                # 确保标签形状与输出一致：[batch_size]
                loss = self.criterion(out.view(-1), batch.y.float().view(-1))

                # 梯度累积（适合显存小的场景，等效增大batch_size）
                loss = loss / gradient_accumulation_steps
                loss.backward()

                # 每gradient_accumulation_steps步更新一次参数
                if (step + 1) % gradient_accumulation_steps == 0:
                    self.optimizer.step()
                    self.optimizer.zero_grad()

                total_loss += loss.item() * gradient_accumulation_steps  # 恢复真实损失值

                # 计算预测结果（确保维度正确）
                pred = (torch.sigmoid(out) > 0.5).long().view(-1)
                all_pred.append(pred.detach().cpu())  # detach()减少显存占用
                all_true.append(batch.y.cpu().view(-1))

            # 合并所有批次的结果
            all_pred = torch.cat(all_pred)
            all_true = torch.cat(all_true)
            metrics = self.evaluate_metrics(all_pred, all_true)

            # 按间隔打印日志
            if epoch % log_interval == 0:
                print(f"\nEpoch {epoch} | Loss: {total_loss:.4f} | "
                      f"Acc: {metrics['accuracy']:.4f} | "
                      f"Precision: {metrics['precision']:.4f} | "
                      f"Recall: {metrics['recall']:.4f} | "
                      f"F1: {metrics['f1']:.4f} | "
                      f"TP: {metrics['tp']} | TN: {metrics['tn']} | "
                      f"FP: {metrics['fp']} | FN: {metrics['fn']}")
            if self.test_loader is not None:
                print(f"Epoch {epoch} | 测试指标 ", end="")
                self.test()  # 调用测试方法

    def test(self):
        """改进点：统一预测逻辑，确保与训练时的处理一致"""
        if self.test_loader is None:
            print("No test loader provided.")
            return None

        self.model.eval()
        all_pred = []
        all_true = []

        with torch.no_grad():  # 关闭梯度计算，节省显存和时间
            loop = tqdm(self.test_loader, total=len(self.test_loader), leave=False)
            loop.set_description("Testing")

            for batch in loop:
                batch = batch.to(self.device)
                out = self.model(batch.x, batch.edge_index, batch.batch)
                pred = (torch.sigmoid(out) > 0.5).long().view(-1)

                all_pred.append(pred.cpu())
                all_true.append(batch.y.cpu().view(-1))

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
        """改进点：增加输入校验，确保计算正确"""
        # 确保输入是一维张量且长度一致
        assert pred.dim() == 1 and true.dim() == 1, "预测和标签必须是一维张量"
        assert len(pred) == len(true), "预测和标签长度必须一致"

        # 计算混淆矩阵（确保标签是0/1二值）
        tp = ((pred == 1) & (true == 1)).sum().item()
        tn = ((pred == 0) & (true == 0)).sum().item()
        fp = ((pred == 1) & (true == 0)).sum().item()
        fn = ((pred == 0) & (true == 1)).sum().item()

        # 计算指标（加1e-9避免除零错误）
        total = tp + tn + fp + fn + 1e-9
        accuracy = (tp + tn) / total
        precision = tp / (tp + fp + 1e-9)
        recall = tp / (tp + fn + 1e-9)
        f1 = 2 * precision * recall / (precision + recall + 1e-9)

        return {
            'tp': tp, 'tn': tn, 'fp': fp, 'fn': fn,
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1
        }
