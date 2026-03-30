# ============================================
# 这一版没有使用集成学习策略
# [+] fix 已修复验证集逻辑
# [-] SmoothL1Loss不可行，因为part1的CEE...把目标变量 Y 里的高频噪音全部提纯洗掉了
# ============================================

import torch
import torch.nn as nn
import numpy as np
import lightgbm as lgb
import matplotlib.pyplot as plt
import math

from joblib import load, dump
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# ============================================
# 1 加载数据 (包含干净标签和原始标签)
# ============================================

print("正在加载数据...")
train_set = load("train_set")
train_label = load("train_label")  # 训练用：干净的平滑目标

test_set = load("test_set")
test_label_clean = load("test_label_clean")  # 评估用：对比趋势
test_label_raw = load("test_label_raw")  # 评估用：计算最终真实误差！

scaler_y = load("scaler_y")

print("模型输入 X 形状:", train_set.shape)
print("模型目标 Y 形状:", train_label.shape)

# ============================================
# 2 LightGBM 特征选择（基于干净目标拟合）
# ============================================

print("\n开始 LightGBM 空间特征降维...")

train_mean = np.mean(train_set.numpy(), axis=1)
train_std = np.std(train_set.numpy(), axis=1)
train_max = np.max(train_set.numpy(), axis=1)
train_min = np.min(train_set.numpy(), axis=1)
train_skew = np.mean(((train_set.numpy() - np.mean(train_set.numpy(), axis=1, keepdims=True)) /
                      (np.std(train_set.numpy(), axis=1, keepdims=True) + 1e-8)) ** 3, axis=1)

train_features_stat = np.concatenate([train_mean, train_std, train_max, train_min, train_skew], axis=1)
y_lgb = train_label.numpy().ravel()

lgb_model = lgb.LGBMRegressor(
    n_estimators=500,
    learning_rate=0.03,
    max_depth=8,
    min_child_samples=20,
    random_state=42
)
lgb_model.fit(train_features_stat, y_lgb)

importance = lgb_model.feature_importances_
feature_dim = train_set.shape[2]

importance_per_var = []
for i in range(feature_dim):
    score = importance[i] + importance[i + feature_dim] + importance[i + 2 * feature_dim] + \
            importance[i + 3 * feature_dim] + importance[i + 4 * feature_dim]
    importance_per_var.append(score)

importance_per_var = np.array(importance_per_var)

# 【优化：基于阈值而非比例的筛选】
# 策略：保留重要性 > 平均重要性 10% 的所有特征
threshold = np.mean(importance_per_var) * 0.1
selected_features = np.where(importance_per_var > threshold)[0]

# 确保至少保留 90% 的特征
if len(selected_features) < int(feature_dim * 0.9):
    top_k = int(feature_dim * 0.9)
    selected_features = np.argsort(importance_per_var)[-top_k:]

print(f"原始特征维度：{feature_dim} -> 筛选后维度：{len(selected_features)} (基于阈值筛选)")

# ============================================
# 3 💡核心修复：分离验证集与测试集 Loader
# ============================================

train_set_selected = train_set[:, :, selected_features]
test_set_selected = test_set[:, :, selected_features]

batch_size = 64

# 1. 训练集 (使用 Clean Label 训练)
train_dataset = TensorDataset(train_set_selected, train_label)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

# 2. 验证集 (🚨 修复点：使用 Clean Label 指导 Scheduler 和早停，防止被高频噪声误导！)
val_dataset = TensorDataset(test_set_selected, test_label_clean)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

# 3. 测试集 (仅用于最终步骤的真实世界性能评估，装载 Raw Label)
test_dataset = TensorDataset(test_set_selected, test_label_raw)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)


# ============================================
# 4 定义带位置编码的 Transformer
# ============================================

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:, :x.size(1), :]
        return x


class TransformerModel(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.embedding = nn.Linear(input_dim, 256)
        self.pos_encoder = PositionalEncoding(d_model=256)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=256, nhead=8, dim_feedforward=512,
            batch_first=True, dropout=0.1
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=4)

        self.attention_pooling = nn.Sequential(
            nn.Linear(256, 128),
            nn.Tanh(),
            nn.Linear(128, 1)
        )

        self.fc = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 1)
        )

    def forward(self, x, return_attention=False):
        x = self.embedding(x)
        x = self.pos_encoder(x)
        x = self.transformer(x)

        attention_weights = self.attention_pooling(x)
        attention_weights = torch.softmax(attention_weights, dim=1)
        context = torch.sum(x * attention_weights, dim=1)

        out = self.fc(context)

        if return_attention:
            return out, attention_weights
        return out


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 设置随机种子保证单体模型可复现
torch.manual_seed(42)
np.random.seed(42)

model = TransformerModel(input_dim=len(selected_features)).to(device)

# ============================================
# 5 模型训练
# ============================================

criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.0005, weight_decay=1e-5)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)

epochs = 50

train_losses = []
val_losses = []

print(f"\n开始在 {device} 上训练 Transformer...")
best_val_loss = float('inf')
patience_counter = 0
early_stop_patience = 15

for epoch in range(epochs):
    model.train()
    epoch_loss = 0

    for x, y in train_loader:
        x, y = x.to(device), y.to(device)
        pred = model(x)
        loss = criterion(pred, y)

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        epoch_loss += loss.item()

    epoch_loss /= len(train_loader)
    train_losses.append(epoch_loss)

    # 🚨 修复点：验证集评估（使用 val_loader 中的 Clean Label 指导早停）
    model.eval()
    val_loss = 0
    with torch.no_grad():
        for x, y in val_loader:
            x = x.to(device)
            pred = model(x)
            val_loss += criterion(pred, y.to(device)).item()
    val_loss /= len(val_loader)
    val_losses.append(val_loss)

    scheduler.step(val_loss)

    if (epoch + 1) % 5 == 0 or epoch == 0:
        print(f"Epoch [{epoch + 1}/{epochs}] - Train Loss: {epoch_loss:.6f} - Val Loss: {val_loss:.6f}")

    # 早停机制
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        patience_counter = 0
        # 保存最佳模型
        torch.save(model.state_dict(), "transformer_weights_best.pth")
    else:
        patience_counter += 1
        if patience_counter >= early_stop_patience:
            print(f"早停触发于 Epoch {epoch + 1}")
            break

# 加载最佳模型
model.load_state_dict(torch.load("transformer_weights_best.pth"))

# ============================================
# 6 测试与反归一化 (直接对抗真实环境噪声)
# ============================================

model.eval()
pred_list = []

with torch.no_grad():
    for x, _ in test_loader:  # 最终测试使用 Raw Label
        x = x.to(device)
        pred = model(x)
        pred_list.append(pred.cpu().numpy())

pred_scaled = np.vstack(pred_list)

# 反归一化
pred_real = scaler_y.inverse_transform(pred_scaled)
true_raw_real = scaler_y.inverse_transform(test_label_raw.numpy())
true_clean_real = scaler_y.inverse_transform(test_label_clean.numpy())

# ============================================
# 7 评估指标 (核心：计算真实世界的误差)
# ============================================

rmse = np.sqrt(mean_squared_error(true_raw_real, pred_real))
mae = mean_absolute_error(true_raw_real, pred_real)
r2 = r2_score(true_raw_real, pred_real)

print("\n--- 最终工业级评估结果 (对抗真实高频噪声) ---")
print(f"RMSE: {rmse:.4f} MW")
print(f"MAE : {mae:.4f} MW")
print(f"R2  : {r2:.4f}")

# ============================================
# 8 可视化：损失曲线与预测拟合度
# ============================================

# 诊断代码，重要性排名
# print(f"\n=== 特征选择详情 ===")
# print(f"原始维度：{feature_dim}")
# print(f"筛选后维度：{len(selected_features)}")
# print(f"保留比例：{len(selected_features)/feature_dim:.2%}")
#
# # 查看重要性排名
# sorted_idx = np.argsort(importance_per_var)[::-1]
# print("\nTop 10 重要特征:")
# for rank, idx in enumerate(sorted_idx[:10], 1):
#     print(f"  {rank}. 特征 {idx}: {importance_per_var[idx]:.4f}")
#
# # 检查最后 2 个特征（风切变）的排名。

# 不不不，后来part1移除了风切变，最后两个特征不再是风切变

# wind_shear_70 = feature_dim - 2
# wind_shear_50 = feature_dim - 1
# rank_70 = np.where(sorted_idx == wind_shear_70)[0][0] + 1
# rank_50 = np.where(sorted_idx == wind_shear_50)[0][0] + 1
# print(f"\n风切变 70/10 排名：{rank_70}/{feature_dim}")
# print(f"风切变 50/10 排名：{rank_50}/{feature_dim}")
# print(f"是否被保留：{'✅ 是' if wind_shear_70 in selected_features else '❌ 否'}, "
#       f"{'✅ 是' if wind_shear_50 in selected_features else '❌ 否'}")




plt.figure(figsize=(18, 6))

# 图 1：训练&验证损失曲线
plt.subplot(1, 3, 1)
plt.plot(range(1, len(train_losses) + 1), train_losses, marker='o', linestyle='-', color='b', label='Train')
plt.plot(range(1, len(val_losses) + 1), val_losses, marker='s', linestyle='--', color='orange', label='Val (Clean)')
plt.title("Training & Validation Loss Curve")
plt.xlabel("Epoch")
plt.ylabel("MSE Loss")
plt.legend()
plt.grid(True, linestyle='--', alpha=0.6)

# 图 2：预测曲线对比
plt.subplot(1, 3, 2)
plot_len = 200
plt.plot(true_raw_real[:plot_len], label="True Raw Power (Noisy)", color='gray', alpha=0.5, linewidth=1)
plt.plot(true_clean_real[:plot_len], label="True Clean Power (Target)", color='green', linestyle='--', linewidth=2)
plt.plot(pred_real[:plot_len], label="Model Prediction", color='red', linewidth=2)
plt.title("Wind Power Prediction (First 200 Hours)")
plt.xlabel("Time Step")
plt.ylabel("Power (MW)")
plt.legend()
plt.grid(True, linestyle='--', alpha=0.6)

# 图 3：预测 vs 真实散点图
plt.subplot(1, 3, 3)
plt.scatter(true_raw_real[::10], pred_real[::10], alpha=0.5, s=20, c='blue', label='Samples')
plt.plot([true_raw_real.min(), true_raw_real.max()],
         [true_raw_real.min(), true_raw_real.max()],
         'r--', linewidth=2, label='Perfect Fit')
plt.title("Prediction vs True (Scatter)")
plt.xlabel("True Power (MW)")
plt.ylabel("Predicted Power (MW)")
plt.legend()
plt.grid(True, linestyle='--', alpha=0.6)

plt.tight_layout()
plt.show()

# ============================================
# 9 保存全套模型资产 (为 PySide6 GUI 部署做准备)
# ============================================

print("\n开始保存模型部署资产...")

# 🚨 修复点：统一保存命名，这里可以直接重命名最佳权重，方便后续调用
import os

os.rename("transformer_weights_best.pth", "transformer_weights_single_minmax.pth")
print("[+] 已保存: transformer_weights_single_minmax.pth")

dump(lgb_model, "lgb_feature_selector.joblib")
print("[+] 已保存: lgb_feature_selector.joblib")

np.save("selected_features_indices.npy", selected_features)
print("[+] 已保存: selected_features_indices.npy")

print("\n🎉 所有训练与资产保存工作圆满完成！")

"""
RMSE: 10.2789 MW
MAE : 6.6826 MW
R2  : 0.9709
"""