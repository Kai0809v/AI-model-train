# ============================================
# 多步长预测模型 - 单模型版
# 核心策略：训练单个Transformer模型，评估去噪后趋势预测能力
# ============================================

import torch
import torch.nn as nn
import numpy as np
import lightgbm as lgb
import matplotlib.pyplot as plt
import math
import random

from joblib import load, dump
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# ============================================
# 1 加载多步数据集
# ============================================

print("正在加载多步数据集...")
multi_step_datasets = load("./processed_data/multi_step_datasets.pkl")
scaler_y = load("./scaler_y")  # scaler_y 保存在根目录

for h in [1, 4, 8]:  # 🔧 移除16步
    data = multi_step_datasets[h]
    print(f"\n{h}步预测数据:")
    print(f"  训练集: {data['train_x'].shape}, 标签: {data['train_y'].shape}")
    if 'val_x' in data:
        print(f"  验证集: {data['val_x'].shape}, 标签: {data['val_y'].shape}")
    print(f"  测试集: {data['test_x'].shape}, 标签: {data['test_y'].shape}")

# ============================================
# 2 LightGBM 特征选择（使用训练集）
# ============================================

print("\n开始 LightGBM 空间特征降维...")

train_set_1step = multi_step_datasets[1]['train_x']
train_mean = np.mean(train_set_1step.numpy(), axis=1)
train_std = np.std(train_set_1step.numpy(), axis=1)
train_max = np.max(train_set_1step.numpy(), axis=1)
train_min = np.min(train_set_1step.numpy(), axis=1)
train_skew = np.mean(((train_set_1step.numpy() - np.mean(train_set_1step.numpy(), axis=1, keepdims=True)) /
                      (np.std(train_set_1step.numpy(), axis=1, keepdims=True) + 1e-8)) ** 3, axis=1)

train_features_stat = np.concatenate([train_mean, train_std, train_max, train_min, train_skew], axis=1)
y_lgb = multi_step_datasets[1]['train_y'].numpy()[:, 0]  # 使用干净标签训练LGBM

lgb_model = lgb.LGBMRegressor(
    n_estimators=500,
    learning_rate=0.03,
    max_depth=8,
    min_child_samples=20,
    random_state=42
)
lgb_model.fit(train_features_stat, y_lgb)

importance = lgb_model.feature_importances_
feature_dim = train_set_1step.shape[2]

importance_per_var = []
for i in range(feature_dim):
    score = importance[i] + importance[i + feature_dim] + importance[i + 2 * feature_dim] + \
            importance[i + 3 * feature_dim] + importance[i + 4 * feature_dim]
    importance_per_var.append(score)

importance_per_var = np.array(importance_per_var)

threshold = np.mean(importance_per_var) * 0.1
selected_features = np.where(importance_per_var > threshold)[0]

if len(selected_features) < int(feature_dim * 0.9):
    top_k = int(feature_dim * 0.9)
    selected_features = np.argsort(importance_per_var)[-top_k:]

print(f"原始特征维度：{feature_dim} -> 筛选后维度：{len(selected_features)}")

#  对所有步长和所有数据集（训练/验证/测试）应用特征选择
for h in [1, 4, 8]:
    multi_step_datasets[h]['train_x'] = multi_step_datasets[h]['train_x'][:, :, selected_features]
    multi_step_datasets[h]['test_x'] = multi_step_datasets[h]['test_x'][:, :, selected_features]

    # 🔧 如果有验证集，也要筛选
    if 'val_x' in multi_step_datasets[h]:
        multi_step_datasets[h]['val_x'] = multi_step_datasets[h]['val_x'][:, :, selected_features]
        print(f"  {h}步: 已筛选 train_x, val_x, test_x")
    else:
        print(f"  {h}步: 已筛选 train_x, test_x (无验证集)")


# ============================================
# 3 Transformer模型（V4稳定架构，架构从V4版本后就没有大改过）
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


class SimpleMultiStepTransformer(nn.Module):
    """稳定架构"""

    def __init__(self, input_dim, horizon=1):
        super().__init__()

        self.embedding = nn.Linear(input_dim, 256)
        self.pos_encoder = PositionalEncoding(d_model=256)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=256, nhead=8, dim_feedforward=512,
            batch_first=True, dropout=0.2  # 🚨 增加 Dropout 以应对原始数据噪声
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
            nn.Dropout(0.3),  # 🚨 增加第一层 Dropout
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.3),  # 🚨 增加第二层 Dropout
            nn.Linear(64, horizon)
        )

    def forward(self, x):
        x = self.embedding(x)
        x = self.pos_encoder(x)
        x = self.transformer(x)

        attention_weights = self.attention_pooling(x)
        attention_weights = torch.softmax(attention_weights, dim=1)
        context = torch.sum(x * attention_weights, dim=1)

        output = self.fc(context)
        return output


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ============================================
# 4 集成学习：训练多个模型
# 暂时不使用集成学习策略了，花的时间太多了，还能减少一点部署的体积
# ============================================

print(f"\n开始在 {device} 上训练集成模型（V7）...")

# 🔧 使用单一随机种子
seed = 42
torch.manual_seed(seed)
np.random.seed(seed)
random.seed(seed)

print(f"\n开始在 {device} 上训练模型...")

results = {}

for h in [1, 4, 8]:
    print(f"\n{'=' * 60}")
    print(f"训练 {h}步预测模型")
    print(f"{'=' * 60}")

    # 创建模型
    model = SimpleMultiStepTransformer(
        input_dim=len(selected_features),
        horizon=h
    ).to(device)

    # 数据加载器
    batch_size = 64
    train_dataset = TensorDataset(
        multi_step_datasets[h]['train_x'],
        multi_step_datasets[h]['train_y']
    )

    # 🔧 如果有验证集，使用验证集进行早停；否则使用测试集
    has_validation = 'val_x' in multi_step_datasets[h]
    if has_validation:
        val_dataset = TensorDataset(
            multi_step_datasets[h]['val_x'],
            multi_step_datasets[h]['val_y']  # 🚨 统一切换为原始功率标签
        )
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
        print(f"  使用验证集进行早停监控（原始标签）")
    else:
        val_dataset = TensorDataset(
            multi_step_datasets[h]['test_x'],
            multi_step_datasets[h]['test_y']  # 🚨 统一切换为原始功率标签
        )
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
        print(f"  无验证集，使用测试集进行早停监控（原始标签）")

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    # V4混合策略配置
    if h == 1:
        optimizer = torch.optim.Adam(model.parameters(), lr=0.0005, weight_decay=1e-5)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=5
        )
        epochs = 50

    elif h == 4:
        optimizer = torch.optim.Adam(model.parameters(), lr=0.0004, weight_decay=2e-5)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=6
        )
        epochs = 55

    else:  # h == 8
        optimizer = torch.optim.AdamW(model.parameters(), lr=0.0003, weight_decay=5e-4)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=8
        )
        epochs = 60

    # 训练循环
    best_val_loss = float('inf')
    patience_counter = 0
    early_stop_patience = 15

    for epoch in range(epochs):
        model.train()
        epoch_loss = 0

        for x, y in train_loader:
            x, y = x.to(device), y.to(device)

            pred = model(x)
            loss = nn.MSELoss()(pred, y)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            epoch_loss += loss.item()

        epoch_loss /= len(train_loader)

        # 验证（使用验证集或测试集）
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for x, y in val_loader:
                x = x.to(device)
                pred = model(x)
                val_loss += nn.MSELoss()(pred, y.to(device)).item()

        val_loss /= len(val_loader)
        scheduler.step(val_loss)

        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(f"  Epoch [{epoch + 1}/{epochs}] - Train Loss: {epoch_loss:.6f} - Val Loss: {val_loss:.6f}")

        # 早停
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            torch.save(model.state_dict(), f"transformer_h{h}_best.pth")
        else:
            patience_counter += 1
            if patience_counter >= early_stop_patience:
                print(f"  早停触发于 Epoch {epoch + 1}")
                break

    # 加载最佳模型
    model.load_state_dict(torch.load(f"transformer_h{h}_best.pth"))

    print(f"\n--- {h}步预测评估 ---")

    # 预测
    model.eval()
    pred_list = []

    with torch.no_grad():
        for x, _ in DataLoader(
                TensorDataset(multi_step_datasets[h]['test_x'],
                              multi_step_datasets[h]['test_y']),  # 🚨 切换为原始标签
                batch_size=64, shuffle=False
        ):
            x = x.to(device)
            pred = model(x)
            pred_list.append(pred.cpu().numpy())

    pred_scaled = np.vstack(pred_list)

    # 获取真实值：直接评估真实环境性能
    true_raw_scaled = multi_step_datasets[h]['test_y'].numpy()

    # 反归一化
    pred_real = np.zeros_like(pred_scaled)
    true_real_raw = np.zeros_like(true_raw_scaled)

    for col in range(h):
        # 模型预测的是原始功率，反归一化后得到 MW
        pred_real[:, col] = scaler_y.inverse_transform(pred_scaled[:, col:col + 1]).flatten()
        true_real_raw[:, col] = scaler_y.inverse_transform(true_raw_scaled[:, col:col + 1]).flatten()

    # 📊 评估：真实环境性能（基于原始功率标签）
    rmse = np.sqrt(mean_squared_error(true_real_raw.flatten(), pred_real.flatten()))
    mae = mean_absolute_error(true_real_raw.flatten(), pred_real.flatten())
    r2 = r2_score(true_real_raw.flatten(), pred_real.flatten())

    print(f"\n  【真实环境评估】（vs 原始含噪信号）")
    print(f"    RMSE: {rmse:.4f} MW")
    print(f"    MAE : {mae:.4f} MW")
    print(f"    R²  : {r2:.4f}")

    results[h] = {
        'pred_real': pred_real,
        'true_real': true_real_raw,
        'rmse': rmse,
        'mae': mae,
        'r2': r2
    }

# ============================================
# 6 保存模型
# ============================================

print("\n开始保存模型资产...")

# 保存LightGBM特征选择器
dump(lgb_model, "./processed_data/lgb_feature_selector.joblib")
print("[+] 已保存: lgb_feature_selector.joblib")

# 保存特征索引
np.save("./processed_data/selected_features_indices.npy", selected_features)
print("[+] 已保存: selected_features_indices.npy")

# 保存Transformer模型
for h in [1, 4, 8]:
    model_package = {
        'horizon': h,
        'model_state_dict': torch.load(f"transformer_h{h}_best.pth"),
        'feature_dim': len(selected_features),
        'performance': {
            'rmse': results[h]['rmse'],
            'mae': results[h]['mae'],
            'r2': results[h]['r2']
        }
    }

    torch.save(model_package, f"transformer_model_h{h}.pth")
    print(f"[+] 已保存: transformer_model_h{h}.pth")

print("\n🎉 模型训练完成！")
print("\n=== 最终性能汇总（基于原始实际功率） ===")
for h in [1, 4, 8]:
    print(f"{h}步预测: RMSE={results[h]['rmse']:.4f} MW, MAE={results[h]['mae']:.4f} MW, R²={results[h]['r2']:.4f}")
"""
修改前（没加验证集时）
1步: RMSE=9.7282 MW, MAE=6.0694 MW, R²=0.9742
4步: RMSE=19.6523 MW, MAE=12.3841 MW, R²=0.8946
8步: RMSE=27.3576 MW, MAE=17.9583 MW, R²=0.7954
修改后（预测降噪目标）：
1步: RMSE=6.9626 MW, MAE=4.5269 MW, R²=0.9833
4步: RMSE=13.7455 MW, MAE=8.4194 MW, R²=0.9350
8步: RMSE=21.4384 MW, MAE=14.3759 MW, R²=0.8421
修改后（不对目标降噪）：
[+] 已保存: lgb_feature_selector.joblib
[+] 已保存: selected_features_indices.npy
[+] 已保存: transformer_model_h1.pth
[+] 已保存: transformer_model_h4.pth
[+] 已保存: transformer_model_h8.pth

🎉 模型训练完成！

=== 最终性能汇总（基于原始实际功率） ===
1步预测: RMSE=6.2366 MW, MAE=3.9534 MW, R²=0.9867
4步预测: RMSE=14.8643 MW, MAE=9.2731 MW, R²=0.9241
8步预测: RMSE=21.7561 MW, MAE=13.5699 MW, R²=0.8376
"""