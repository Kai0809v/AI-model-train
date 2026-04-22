import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# 导入前置模块
from data_loader import create_dataloaders
from model_architecture import TCN_Informer_Model, True_TCN_Informer


# ==========================================
# 1. 评估指标计算函数
# ==========================================
def calculate_metrics(y_true, y_pred):
    """计算常用的时间序列预测指标"""
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)

    return {'MSE': mse, 'RMSE': rmse, 'MAE': mae, 'R2': r2}


# ==========================================
# 2. 早停机制 (Early Stopping)
# ==========================================
class EarlyStopping:
    def __init__(self, patience=7, verbose=False, delta=0):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.inf
        self.delta = delta

    def __call__(self, val_loss, model, path='checkpoint.pth'):
        score = -val_loss
        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model, path)
        elif score < self.best_score + self.delta:
            self.counter += 1
            if self.verbose:
                print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model, path)
            self.counter = 0

    def save_checkpoint(self, val_loss, model, path):
        if self.verbose:
            print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        torch.save(model.state_dict(), path)
        self.val_loss_min = val_loss


# ==========================================
# 3. 核心训练与测试循环
# ==========================================
def train_and_evaluate(pkl_path, epochs=50, learning_rate=0.001, device='cuda' if torch.cuda.is_available() else 'cpu'):
    print(f"--- 启动训练引擎 (使用设备: {device}) ---")

    # 1. 获取数据
     # 🏆 基于序列长度实验结果，采用1天历史窗口（最优配置）
    # 关于序列长度的实验结果，见 experiment.md 实验#4.3
    seq_len, label_len, pred_len = 96, 48, 24
    train_loader, val_loader, test_loader, bundle = create_dataloaders(
        pkl_path, seq_len=seq_len, label_len=label_len, pred_len=pred_len, batch_size=32
    )

    # 提取 PCA 后的特征维度 (即 TCN 的输入维度)
    input_dim = bundle['train'][0].shape[1]
    scaler_y = bundle['scaler_y']  # 用于反归一化

    # 2. 初始化模型 (使用稳定的基线配置)
    model = True_TCN_Informer(
        tcn_input_dim=input_dim,
        # ❌ tcn_channels=[32, 64, 128],
        # ✅ 大幅削减 TCN 通道，只提取最核心的时序突变
        tcn_channels=[16, 32],
        seq_len=seq_len,
        label_len=label_len,
        pred_len=pred_len,
        d_model=64,
        n_heads=4,
        e_layers=2,
        dropout=0.15      # 加大随机失活率
    ).to(device)

    #  ~~抛弃容易被极端天气带偏的 MSE~~ 既然解决了过拟合问题，现在再使用 MSE
    criterion = nn.MSELoss()

    #  启用 Huber Loss
    # delta 参数控制了从 MSE 转变为 MAE 的临界点，可以后续用 NRBO 调优这个值
    # criterion = nn.HuberLoss(delta=0.1)

    # 拉长退火周期，避免陷入局部最优
    # optimizer = optim.Adam(model.parameters(), lr=0.0005, weight_decay=1e-4)
    # ✅ 将 weight_decay 放大 10 倍到 1e-3，严厉惩罚异常膨胀的权重
    # ❌ optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-3)
    # 1e-3好像过于严格了，让模型欠拟合了，改回1e-4
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=epochs, eta_min=1e-6
    )
    early_stopping = EarlyStopping(patience=10, verbose=True)
    train_losses, val_losses = [], []

    # 3. 训练循环
    for epoch in range(epochs):
        model.train()
        epoch_train_loss = []

        # 接收 DataLoader 吐出的时间标记 (seq_x_mark, dec_x_mark)
        for i, (seq_x, seq_x_mark, dec_x, dec_x_mark, target_y) in enumerate(train_loader):
            seq_x = seq_x.to(device)
            seq_x_mark = seq_x_mark.to(device)  # 新增
            dec_x = dec_x.to(device)
            dec_x_mark = dec_x_mark.to(device)  # 新增
            target_y = target_y.to(device)

            optimizer.zero_grad()

            # 前向传播 (将四种输入完整喂给真正的 Informer)
            pred = model(seq_x, seq_x_mark, dec_x, dec_x_mark)

            # 计算损失 (只计算预测长度 pred_len 部分的 MSE)
            loss = criterion(pred.squeeze(-1), target_y)
            epoch_train_loss.append(loss.item())

            # 反向传播
            loss.backward()
            optimizer.step()

        # 4. 验证循环
        model.eval()
        epoch_val_loss = []
        with torch.no_grad():
            for seq_x, seq_x_mark, dec_x, dec_x_mark, target_y in val_loader:
                seq_x = seq_x.to(device)
                seq_x_mark = seq_x_mark.to(device)
                dec_x = dec_x.to(device)
                dec_x_mark = dec_x_mark.to(device)
                target_y = target_y.to(device)

                pred = model(seq_x, seq_x_mark, dec_x, dec_x_mark)
                loss = criterion(pred.squeeze(-1), target_y)
                epoch_val_loss.append(loss.item())

        # 记录并打印 Log
        t_loss = np.average(epoch_train_loss)
        v_loss = np.average(epoch_val_loss)
        train_losses.append(t_loss)
        val_losses.append(v_loss)
        print(f"Epoch: {epoch + 1:02d} | Train Loss: {t_loss:.5f} | Val Loss: {v_loss:.5f}")

        # 更新学习率调度器 (CosineAnnealing 平滑下降)
        scheduler.step()
        current_lr = optimizer.param_groups[0]['lr']
        print(f"           Learning Rate: {current_lr:.6f}")

        # 触发早停机制
        early_stopping(v_loss, model, path='best_tcn_informer.pth')
        if early_stopping.early_stop:
            print("🚀 触发早停机制，训练提前结束。")
            break

    # ==========================================
    # 7. 模型测试与指标评估
    # ==========================================
    print("\n--- 开始测试集评估 ---")
    # 加载表现最好的模型权重
    model.load_state_dict(torch.load('best_tcn_informer.pth'))
    model.eval()

    preds_list = []
    trues_list = []

    with torch.no_grad():
        for seq_x, seq_x_mark, dec_x, dec_x_mark, target_y in test_loader:
            seq_x = seq_x.to(device)
            seq_x_mark = seq_x_mark.to(device)
            dec_x = dec_x.to(device)
            dec_x_mark = dec_x_mark.to(device)

            # 输出预测结果
            pred = model(seq_x, seq_x_mark, dec_x, dec_x_mark).detach().cpu().numpy()
            true = target_y.detach().cpu().numpy()

            preds_list.append(pred.squeeze(-1))
            trues_list.append(true)

    # 拼接所有批次的数据: Shape 变为 [样本总数, pred_len]
    preds = np.concatenate(preds_list, axis=0)
    trues = np.concatenate(trues_list, axis=0)

    # ⭐️ 核心步骤：反归一化 ⭐️
    preds_inverse = scaler_y.inverse_transform(preds.reshape(-1, 1)).reshape(preds.shape)
    trues_inverse = scaler_y.inverse_transform(trues.reshape(-1, 1)).reshape(trues.shape)

    # 💡 终极物理约束：夜间强制归零 + 功率上限限制
    # 逻辑：如果这个时间点的真实功率微乎其微(比如小于设备装机容量的 0.5%)，
    # 我们在物理上认定这是夜晚或极度恶劣无法发电的时刻，直接将预测值覆写为 0。
    night_mask = (trues_inverse < 0.05)  # 假设 0.05 MW 以下算作无光照
    preds_inverse[night_mask] = 0.0
    
    # 💡 物理约束1：光伏功率不可能为负数
    preds_inverse = np.maximum(0, preds_inverse)
    
    # 💡 物理约束2：光伏功率不应超过装机容量（假设130MW）
    MAX_CAPACITY = 130.0  # MW TODO： 后需更换数据集时，需要根据实际情况调整
    preds_inverse = np.minimum(preds_inverse, MAX_CAPACITY)

    # 计算整体指标 (针对未来所有预测步长的平均表现)
    metrics = calculate_metrics(trues_inverse.flatten(), preds_inverse.flatten())
    print("\n📊 最终测试集评估指标:")
    for k, v in metrics.items():
        print(f"   {k}: {v:.4f}")

    # 在计算 metrics 之后，加入残差分析绘图
    residuals = trues_inverse.flatten() - preds_inverse.flatten()

    plt.figure(figsize=(8, 6))
    plt.hist(residuals, bins=50, color='teal', alpha=0.7, edgecolor='black')
    plt.axvline(0, color='red', linestyle='dashed', linewidth=2)
    plt.title('Prediction Residual Distribution')
    plt.xlabel('Error (MW)')
    plt.ylabel('Frequency')
    plt.grid(axis='y', alpha=0.75)
    plt.savefig('residual_histogram.png', dpi=300, bbox_inches='tight')
    # ==========================================
    # 8. 可视化预测结果 (抽取第一个样本的连续 24 小时预测)
    # ==========================================
    plt.figure(figsize=(12, 5))
    plt.plot(trues_inverse[0, :], label='Ground Truth (Real Power)', color='blue', marker='o')
    plt.plot(preds_inverse[0, :], label='TCN-Informer Prediction', color='red', linestyle='--', marker='x')
    plt.title('PV Power Forecasting (24 Steps into the Future)')
    plt.xlabel('Future Time Steps')
    plt.ylabel('Power (MW)')
    plt.legend()
    plt.grid(True)
    plt.savefig('prediction_result.png')
    print("\n🖼️ 预测对比曲线已保存为 prediction_result.png")

    return metrics


if __name__ == "__main__":
    # 执行流水线
    train_and_evaluate("processed_data/model_ready_data.pkl")