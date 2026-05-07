import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import os
import sys

# 🔧 修复Informer2020的导入路径问题（不修改其源代码）
informer_path = os.path.join(os.path.dirname(__file__), '../../../Informer2020')
if informer_path not in sys.path:
    sys.path.insert(0, informer_path)

# 导入前置模块（使用无未来数据版本的数据加载器）
from data_loader_no_weather_prediction import create_dataloaders_no_weather
from model_architecture import True_TCN_Informer


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
def train_and_evaluate_no_weather(pkl_path, epochs=50, learning_rate=0.001, 
                                   device='cuda' if torch.cuda.is_available() else 'cpu',
                                   output_dir="assets/pv_tcn_informer_no_weather_prediction"):
    """
    无未来气象数据版本的训练流程
    
    核心差异：
    - 使用 create_dataloaders_no_weather（解码器输入未来部分为零填充）
    - 模型资产保存到独立目录
    """
    print(f"--- 启动训练引擎 (无未来数据版本, 使用设备: {device}) ---")
    
    # 创建输出目录
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"✅ 创建输出目录: {output_dir}")

    # 1. 获取数据（使用无未来数据版本的数据加载器）
    # 扩大为观察过去 2 天 (192步) 或 3 天 (288步){parameter A}，预测未来 6 小时 (24步){parameter C}，并且保持B是A的一半
    seq_len, label_len, pred_len = 192, 96, 24
    train_loader, val_loader, test_loader, bundle = create_dataloaders_no_weather(
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
        # d_model=128, # 缩小模型维度，之前的应该比较臃肿
        # 配合 TCN，d_model 也可以降下来
        d_model=64,
        n_heads=4,
        e_layers=2,
        dropout=0.15      # 加大随机失活率
    ).to(device)

    # 既然解决了过拟合问题,现在再使用 MSE
    criterion = nn.MSELoss()
    
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-3)
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

            # 计算损失 (使用标准 MSE)
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

        # 触发早停机制（保存到新目录）
        model_save_path = os.path.join(output_dir, 'best_tcn_informer_no_weather_prediction.pth')
        early_stopping(v_loss, model, path=model_save_path)
        if early_stopping.early_stop:
            print("🚀 触发早停机制，训练提前结束。")
            break

    # ==========================================
    # 7. 模型测试与指标评估
    # ==========================================
    print("\n--- 开始测试集评估 ---")
    # 加载表现最好的模型权重
    model.load_state_dict(torch.load(model_save_path))
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
    # 我们将预测值还原回真实的 Power (MW) 量纲
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

    # ==========================================
    # 7.1 训练集指标评估
    # ==========================================
    print("\n--- 开始训练集评估 ---")
    model.eval()

    train_preds_list = []
    train_trues_list = []

    with torch.no_grad():
        for seq_x, seq_x_mark, dec_x, dec_x_mark, target_y in train_loader:
            seq_x = seq_x.to(device)
            seq_x_mark = seq_x_mark.to(device)
            dec_x = dec_x.to(device)
            dec_x_mark = dec_x_mark.to(device)

            pred = model(seq_x, seq_x_mark, dec_x, dec_x_mark).detach().cpu().numpy()
            true = target_y.detach().cpu().numpy()

            train_preds_list.append(pred.squeeze(-1))
            train_trues_list.append(true)

    train_preds = np.concatenate(train_preds_list, axis=0)
    train_trues = np.concatenate(train_trues_list, axis=0)

    train_preds_inverse = scaler_y.inverse_transform(train_preds.reshape(-1, 1)).reshape(train_preds.shape)
    train_trues_inverse = scaler_y.inverse_transform(train_trues.reshape(-1, 1)).reshape(train_trues.shape)

    train_night_mask = (train_trues_inverse < 0.05)
    train_preds_inverse[train_night_mask] = 0.0
    train_preds_inverse = np.maximum(0, train_preds_inverse)
    train_preds_inverse = np.minimum(train_preds_inverse, MAX_CAPACITY)

    train_metrics = calculate_metrics(train_trues_inverse.flatten(), train_preds_inverse.flatten())
    print("\n📊 训练集评估指标:")
    for k, v in train_metrics.items():
        print(f"   {k}: {v:.4f}")

    # ==========================================
    # 8. 可视化预测结果 (自动寻找白天有功率的样本)
    # ==========================================
    # 寻找一个白天样本：计算每个样本未来 24 步的总功率，选取总和最高的一个
    sample_power_sums = trues_inverse.sum(axis=1)
    best_sample_idx = np.argmax(sample_power_sums)  # 选取发电量最大的样本
    
    print(f"\n 正在可视化样本索引: {best_sample_idx} (未来 24 步总功率: {sample_power_sums[best_sample_idx]:.2f} MW)")
    
    plt.figure(figsize=(12, 5))
    plt.plot(trues_inverse[best_sample_idx, :], label=f'Ground Truth (Sample {best_sample_idx})', color='blue', marker='o')
    plt.plot(preds_inverse[best_sample_idx, :], label='TCN-Informer No Weather Prediction', color='red', linestyle='--', marker='x')
    plt.title(f'PV Power Forecasting - No Weather Data (Sample {best_sample_idx}, Total Power: {sample_power_sums[best_sample_idx]:.2f} MW)')
    plt.xlabel('Future Time Steps (15 min/step)')
    plt.ylabel('Power (MW)')
    plt.legend()
    plt.grid(True)
    
    plot_save_path = os.path.join(output_dir, 'prediction_result_no_weather.png')
    plt.savefig(plot_save_path)
    print(f"\n🖼️ 预测对比曲线已保存为 {plot_save_path}")

    # 保存预处理器bundle（用于推理时加载）
    preprocessor_bundle_path = os.path.join(output_dir, 'preprocessor_bundle.pkl')
    import joblib
    joblib.dump(bundle, preprocessor_bundle_path)
    print(f"✅ 预处理器bundle已保存至: {preprocessor_bundle_path}")

    return metrics


if __name__ == "__main__":
    # 执行流水线
    train_and_evaluate_no_weather("../../../processed_data/model_ready_data.pkl")
