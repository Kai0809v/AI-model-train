# ============================================
# 多步长预测模型训练脚本
# 支持 1/4/8/16 步灵活预测
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
multi_step_datasets = load("multi_step_datasets.pkl")
scaler_y = load("scaler_y")

# 验证数据加载
for h in [1, 4, 8, 16]:
    data = multi_step_datasets[h]
    print(f"\n{h}步预测数据:")
    print(f"  训练集: {data['train_x'].shape}, 标签: {data['train_y'].shape}")
    print(f"  测试集: {data['test_x'].shape}, 标签: {data['test_y_raw'].shape}")

# ============================================
# 2 LightGBM 特征选择（复用part1的结果）
# ============================================

print("\n开始 LightGBM 空间特征降维...")

# 使用1步数据的训练集进行特征选择
train_set_1step = multi_step_datasets[1]['train_x']
train_mean = np.mean(train_set_1step.numpy(), axis=1)
train_std = np.std(train_set_1step.numpy(), axis=1)
train_max = np.max(train_set_1step.numpy(), axis=1)
train_min = np.min(train_set_1step.numpy(), axis=1)
train_skew = np.mean(((train_set_1step.numpy() - np.mean(train_set_1step.numpy(), axis=1, keepdims=True)) /
                      (np.std(train_set_1step.numpy(), axis=1, keepdims=True) + 1e-8)) ** 3, axis=1)

train_features_stat = np.concatenate([train_mean, train_std, train_max, train_min, train_skew], axis=1)
y_lgb = multi_step_datasets[1]['train_y'].numpy()[:, 0]  # 取第1步作为目标

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

# 基于阈值筛选
threshold = np.mean(importance_per_var) * 0.1
selected_features = np.where(importance_per_var > threshold)[0]

if len(selected_features) < int(feature_dim * 0.9):
    top_k = int(feature_dim * 0.9)
    selected_features = np.argsort(importance_per_var)[-top_k:]

print(f"原始特征维度：{feature_dim} -> 筛选后维度：{len(selected_features)}")

# 对所有步长的数据进行特征选择
for h in [1, 4, 8, 16]:
    multi_step_datasets[h]['train_x'] = multi_step_datasets[h]['train_x'][:, :, selected_features]
    multi_step_datasets[h]['test_x'] = multi_step_datasets[h]['test_x'][:, :, selected_features]

# ============================================
# 3 定义多步长Transformer模型
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


class MultiHorizonTransformer(nn.Module):
    """
    多步长预测模型：
    - 共享Transformer编码器
    - 独立的预测头用于不同步长
    """
    def __init__(self, input_dim, horizons=[1, 4, 8, 16]):
        super().__init__()
        
        # 共享编码器
        self.embedding = nn.Linear(input_dim, 256)
        self.pos_encoder = PositionalEncoding(d_model=256)
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=256, nhead=8, dim_feedforward=512,
            batch_first=True, dropout=0.1
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=4)
        
        # 注意力池化
        self.attention_pooling = nn.Sequential(
            nn.Linear(256, 128),
            nn.Tanh(),
            nn.Linear(128, 1)
        )
        
        # 多步长预测头
        self.heads = nn.ModuleDict({
            f'h_{h}': nn.Sequential(
                nn.Linear(256, 128),
                nn.ReLU(),
                nn.Dropout(0.2),
                nn.Linear(128, 64),
                nn.ReLU(),
                nn.Dropout(0.2),
                nn.Linear(64, h)  # 输出h步
            ) for h in horizons
        })
    
    def forward(self, x, horizon=1):
        # 编码
        x = self.embedding(x)
        x = self.pos_encoder(x)
        x = self.transformer(x)
        
        # 注意力池化得到上下文向量
        attention_weights = self.attention_pooling(x)
        attention_weights = torch.softmax(attention_weights, dim=1)
        context = torch.sum(x * attention_weights, dim=1)
        
        # 选择对应的预测头
        head_key = f'h_{horizon}'
        output = self.heads[head_key](context)
        
        return output


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 设置随机种子
torch.manual_seed(42)
np.random.seed(42)
random.seed(42)

model = MultiHorizonTransformer(
    input_dim=len(selected_features),
    horizons=[1, 4, 8, 16]
).to(device)

print(f"\n模型参数量: {sum(p.numel() for p in model.parameters()):,}")

# ============================================
# 4 加权损失函数
# ============================================

def weighted_mse_loss(predictions, targets, horizon):
    """
    加权MSE：近期权重更高，远期权重递减
    
    优化：降低衰减系数，让远期预测也能得到充分优化
    """
    # 🔧 优化：降低衰减系数从0.15到0.08，让远期权重更高
    weights = torch.exp(-torch.arange(horizon, dtype=torch.float32) * 0.08)
    weights = weights.to(predictions.device)
    weights = weights.unsqueeze(0)  # (1, horizon)
    
    # 计算加权MSE
    mse_per_step = (predictions - targets) ** 2
    weighted_mse = (mse_per_step * weights).mean()
    
    return weighted_mse


# ============================================
# 5 训练循环
# ============================================

# 为每个步长创建DataLoader
batch_size = 64
dataloaders = {}

for h in [1, 4, 8, 16]:
    data = multi_step_datasets[h]
    train_dataset = TensorDataset(data['train_x'], data['train_y'])
    test_dataset = TensorDataset(data['test_x'], data['test_y_raw'])
    
    dataloaders[h] = {
        'train': DataLoader(train_dataset, batch_size=batch_size, shuffle=True),
        'test': DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    }

optimizer = torch.optim.Adam(model.parameters(), lr=0.0005, weight_decay=1e-5)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)

epochs = 50
train_losses = {h: [] for h in [1, 4, 8, 16]}
val_losses = {h: [] for h in [1, 4, 8, 16]}

print(f"\n开始在 {device} 上训练多步长Transformer...")
best_val_loss = {h: float('inf') for h in [1, 4, 8, 16]}
patience_counter = {h: 0 for h in [1, 4, 8, 16]}
early_stop_patience = 15

for epoch in range(epochs):
    model.train()
    epoch_losses = {h: 0 for h in [1, 4, 8, 16]}
    
    # 训练阶段：随机采样不同步长
    all_horizons = [1, 4, 8, 16]
    random.shuffle(all_horizons)
    
    for h in all_horizons:
        loader = dataloaders[h]['train']
        
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            
            pred = model(x, horizon=h)
            loss = weighted_mse_loss(pred, y, h)
            
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            epoch_losses[h] += loss.item()
        
        epoch_losses[h] /= len(loader)
        train_losses[h].append(epoch_losses[h])
    
    # 验证阶段：评估所有步长
    model.eval()
    val_epoch_losses = {h: 0 for h in [1, 4, 8, 16]}
    
    with torch.no_grad():
        for h in [1, 4, 8, 16]:
            loader = dataloaders[h]['test']
            for x, y in loader:
                x = x.to(device)
                pred = model(x, horizon=h)
                val_epoch_losses[h] += weighted_mse_loss(pred, y.to(device), h).item()
            
            val_epoch_losses[h] /= len(loader)
            val_losses[h].append(val_epoch_losses[h])
    
    # 学习率调整（使用1步的验证损失）
    scheduler.step(val_epoch_losses[1])
    
    if (epoch + 1) % 5 == 0 or epoch == 0:
        print(f"\nEpoch [{epoch + 1}/{epochs}]")
        for h in [1, 4, 8, 16]:
            print(f"  {h}步 - Train Loss: {epoch_losses[h]:.6f} | Val Loss: {val_epoch_losses[h]:.6f}")
    
    # 早停机制（基于1步验证损失）
    if val_epoch_losses[1] < best_val_loss[1]:
        best_val_loss[1] = val_epoch_losses[1]
        patience_counter[1] = 0
        torch.save(model.state_dict(), "transformer_multi_horizon_best.pth")
    else:
        patience_counter[1] += 1
        if patience_counter[1] >= early_stop_patience:
            print(f"\n早停触发于 Epoch {epoch + 1}")
            break

# 加载最佳模型
model.load_state_dict(torch.load("transformer_multi_horizon_best.pth"))
print("\n[+] 已加载最佳模型")

# ============================================
# 6 评估各步长性能
# ============================================

print("\n=== 评估各步长预测性能 ===")

results = {}

for h in [1, 4, 8, 16]:
    print(f"\n--- {h}步预测评估 ---")
    
    model.eval()
    pred_list = []
    true_list = []
    
    with torch.no_grad():
        for x, y in dataloaders[h]['test']:
            x = x.to(device)
            pred = model(x, horizon=h)
            pred_list.append(pred.cpu().numpy())
            true_list.append(y.numpy())
    
    pred_scaled = np.vstack(pred_list)
    true_scaled = np.vstack(true_list)
    
    # 反归一化（逐列反归一化）
    pred_real = np.zeros_like(pred_scaled)
    true_real = np.zeros_like(true_scaled)
    
    for col in range(h):
        pred_real[:, col] = scaler_y.inverse_transform(pred_scaled[:, col:col+1]).flatten()
        true_real[:, col] = scaler_y.inverse_transform(true_scaled[:, col:col+1]).flatten()
    
    # 计算指标
    rmse = np.sqrt(mean_squared_error(true_real.flatten(), pred_real.flatten()))
    mae = mean_absolute_error(true_real.flatten(), pred_real.flatten())
    r2 = r2_score(true_real.flatten(), pred_real.flatten())
    
    print(f"RMSE: {rmse:.4f} MW")
    print(f"MAE : {mae:.4f} MW")
    print(f"R2  : {r2:.4f}")
    
    results[h] = {
        'pred_real': pred_real,
        'true_real': true_real,
        'rmse': rmse,
        'mae': mae,
        'r2': r2
    }

# ============================================
# 7 可视化
# ============================================

plt.figure(figsize=(20, 12))

# 图1-4：各步长预测曲线对比
for idx, h in enumerate([1, 4, 8, 16], 1):
    plt.subplot(2, 2, idx)
    
    # 取前100个样本的平均趋势
    n_samples = min(100, len(results[h]['pred_real']))
    pred_mean = results[h]['pred_real'][:n_samples].mean(axis=0)
    true_mean = results[h]['true_real'][:n_samples].mean(axis=0)
    
    timesteps = np.arange(h)
    plt.plot(timesteps, true_mean, label="True Power", color='green', linewidth=2, marker='o')
    plt.plot(timesteps, pred_mean, label="Predicted Power", color='red', linewidth=2, marker='s')
    
    plt.title(f"{h}-Step Forecast (Avg of {n_samples} samples)\nRMSE: {results[h]['rmse']:.2f} MW")
    plt.xlabel("Forecast Horizon (15min steps)")
    plt.ylabel("Power (MW)")
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.6)

plt.tight_layout()
plt.savefig("multi_horizon_results.png", dpi=150, bbox_inches='tight')
plt.show()

# 图5：各步长RMSE对比
plt.figure(figsize=(10, 6))
horizons = [1, 4, 8, 16]
rmses = [results[h]['rmse'] for h in horizons]
maes = [results[h]['mae'] for h in horizons]

x_pos = np.arange(len(horizons))
width = 0.35

plt.bar(x_pos - width/2, rmses, width, label='RMSE', color='steelblue')
plt.bar(x_pos + width/2, maes, width, label='MAE', color='coral')

plt.xlabel('Forecast Horizon (steps)')
plt.ylabel('Error (MW)')
plt.title('Multi-Horizon Prediction Performance')
plt.xticks(x_pos, horizons)
plt.legend()
plt.grid(True, linestyle='--', alpha=0.6, axis='y')

plt.tight_layout()
plt.savefig("multi_horizon_comparison.png", dpi=150, bbox_inches='tight')
plt.show()

# ============================================
# 8 保存模型资产
# ============================================

print("\n开始保存模型部署资产...")

import os
os.rename("transformer_multi_horizon_best.pth", "transformer_multi_horizon.pth")
print("[+] 已保存: transformer_multi_horizon.pth")

dump(lgb_model, "lgb_feature_selector.joblib")
print("[+] 已保存: lgb_feature_selector.joblib")

np.save("selected_features_indices.npy", selected_features)
print("[+] 已保存: selected_features_indices.npy")

print("\n🎉 多步长模型训练完成！")
print("\n=== 最终性能汇总 ===")
for h in [1, 4, 8, 16]:
    print(f"{h}步: RMSE={results[h]['rmse']:.4f} MW, MAE={results[h]['mae']:.4f} MW, R²={results[h]['r2']:.4f}")
