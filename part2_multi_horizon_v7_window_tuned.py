# ============================================
# 多步长预测模型 V7 - 差异化窗口版
# 核心改进：为不同步长使用定制化的窗口大小
# 1步: 36步(9h), 4步: 48步(12h), 8步: 72步(18h)
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
# 1 加载差异化窗口数据集
# ============================================

print("正在加载差异化窗口数据集...")
multi_step_datasets = load("multi_step_datasets_window_tuned.pkl")
scaler_y = load("scaler_y")

for h in [1, 4, 8]:
    data = multi_step_datasets[h]
    print(f"\n{h}步预测数据:")
    print(f"  训练集: {data['train_x'].shape}, 标签: {data['train_y'].shape}")
    print(f"  测试集: {data['test_x'].shape}, 标签: {data['test_y_raw'].shape}")

# ============================================
# 2 LightGBM 特征选择
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
y_lgb = multi_step_datasets[1]['train_y'].numpy()[:, 0]

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

for h in [1, 4, 8]:
    multi_step_datasets[h]['train_x'] = multi_step_datasets[h]['train_x'][:, :, selected_features]
    multi_step_datasets[h]['test_x'] = multi_step_datasets[h]['test_x'][:, :, selected_features]

# ============================================
# 3 Transformer模型（V4稳定架构）
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
    """V4稳定架构，支持任意序列长度"""
    def __init__(self, input_dim, horizon=1):
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
# 4 训练循环（V4混合策略）
# ============================================

print(f"\n开始在 {device} 上训练多步长模型（V7差异化窗口）...")

models = {}
results = {}

for h in [1, 4, 8]:
    print(f"\n{'='*60}")
    print(f"训练 {h}步预测模型（窗口={multi_step_datasets[h]['train_x'].shape[1]}步）")
    print(f"{'='*60}")
    
    # 设置随机种子
    torch.manual_seed(42)
    np.random.seed(42)
    random.seed(42)
    
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
    test_dataset = TensorDataset(
        multi_step_datasets[h]['test_x'],
        multi_step_datasets[h]['test_y_raw']
    )
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    # V4混合策略配置
    if h == 1:
        optimizer = torch.optim.Adam(model.parameters(), lr=0.0005, weight_decay=1e-5)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=5
        )
        epochs = 50
        print(f"  [Strategy] Short-term: Adam + lr=0.0005")
        
    elif h == 4:
        optimizer = torch.optim.Adam(model.parameters(), lr=0.0004, weight_decay=2e-5)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=6
        )
        epochs = 55
        print(f"  [Strategy] Mid-term: Adam + lr=0.0004")
        
    else:  # h == 8
        optimizer = torch.optim.AdamW(model.parameters(), lr=0.0003, weight_decay=5e-4)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=8
        )
        epochs = 60
        print(f"  [Strategy] Long-term: AdamW + Strong Reg + lr=0.0003")
    
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
        
        # 验证
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for x, y in test_loader:
                x = x.to(device)
                pred = model(x)
                val_loss += nn.MSELoss()(pred, y.to(device)).item()
        
        val_loss /= len(test_loader)
        scheduler.step(val_loss)
        
        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(f"  Epoch [{epoch + 1}/{epochs}] - Train Loss: {epoch_loss:.6f} - Val Loss: {val_loss:.6f}")
        
        # 早停
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            torch.save(model.state_dict(), f"transformer_h{h}_v7_best.pth")
        else:
            patience_counter += 1
            if patience_counter >= early_stop_patience:
                print(f"  早停触发于 Epoch {epoch + 1}")
                break
    
    # 加载最佳模型
    model.load_state_dict(torch.load(f"transformer_h{h}_v7_best.pth"))
    models[h] = model
    
    # 评估
    print(f"\n  评估 {h}步模型...")
    model.eval()
    pred_list = []
    true_list = []
    
    with torch.no_grad():
        for x, y in test_loader:
            x = x.to(device)
            pred = model(x)
            pred_list.append(pred.cpu().numpy())
            true_list.append(y.numpy())
    
    pred_scaled = np.vstack(pred_list)
    true_scaled = np.vstack(true_list)
    
    # 反归一化
    pred_real = np.zeros_like(pred_scaled)
    true_real = np.zeros_like(true_scaled)
    
    for col in range(h):
        pred_real[:, col] = scaler_y.inverse_transform(pred_scaled[:, col:col+1]).flatten()
        true_real[:, col] = scaler_y.inverse_transform(true_scaled[:, col:col+1]).flatten()
    
    # 计算指标
    rmse = np.sqrt(mean_squared_error(true_real.flatten(), pred_real.flatten()))
    mae = mean_absolute_error(true_real.flatten(), pred_real.flatten())
    r2 = r2_score(true_real.flatten(), pred_real.flatten())
    
    print(f"  RMSE: {rmse:.4f} MW")
    print(f"  MAE : {mae:.4f} MW")
    print(f"  R2  : {r2:.4f}")
    
    results[h] = {
        'pred_real': pred_real,
        'true_real': true_real,
        'rmse': rmse,
        'mae': mae,
        'r2': r2
    }

# ============================================
# 5 可视化
# ============================================

plt.figure(figsize=(18, 6))

for idx, h in enumerate([1, 4, 8], 1):
    plt.subplot(1, 3, idx)
    
    n_samples = min(100, len(results[h]['pred_real']))
    pred_mean = results[h]['pred_real'][:n_samples].mean(axis=0)
    true_mean = results[h]['true_real'][:n_samples].mean(axis=0)
    
    timesteps = np.arange(h)
    plt.plot(timesteps, true_mean, label="True Power", color='green', linewidth=2, marker='o')
    plt.plot(timesteps, pred_mean, label="Predicted Power", color='red', linewidth=2, marker='s')
    
    plt.title(f"{h}-Step Forecast (V7 Window-Tuned)\nRMSE: {results[h]['rmse']:.2f} MW, R²: {results[h]['r2']:.3f}")
    plt.xlabel("Forecast Horizon (15min steps)")
    plt.ylabel("Power (MW)")
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.6)

plt.tight_layout()
plt.savefig("multi_horizon_v7_window_tuned_results.png", dpi=150, bbox_inches='tight')
plt.show()

# 性能对比图
plt.figure(figsize=(10, 6))

horizons = [1, 4, 8]
rmses = [results[h]['rmse'] for h in horizons]
r2s = [results[h]['r2'] for h in horizons]

x_pos = np.arange(len(horizons))
width = 0.35

plt.bar(x_pos - width/2, rmses, width, label='RMSE (MW)', color='steelblue')
plt.bar(x_pos + width/2, r2s, width, label='R²', color='coral')

plt.xlabel('Forecast Horizon (steps)')
plt.ylabel('Metric Value')
plt.title('Multi-Horizon Prediction Performance (V7 - Window Tuned)')
plt.xticks(x_pos, horizons)
plt.legend()
plt.grid(True, linestyle='--', alpha=0.6, axis='y')

plt.tight_layout()
plt.savefig("multi_horizon_v7_comparison.png", dpi=150, bbox_inches='tight')
plt.show()

# ============================================
# 6 保存模型
# ============================================

print("\n开始保存模型资产...")

dump(lgb_model, "lgb_feature_selector.joblib")
print("[+] 已保存: lgb_feature_selector.joblib")

np.save("selected_features_indices.npy", selected_features)
print("[+] 已保存: selected_features_indices.npy")

print("\n🎉 差异化窗口模型V7训练完成！")
print("\n=== 最终性能汇总 ===")
for h in [1, 4, 8]:
    window_size = multi_step_datasets[h]['train_x'].shape[1]
    print(f"{h}步 (窗口={window_size}): RMSE={results[h]['rmse']:.4f} MW, MAE={results[h]['mae']:.4f} MW, R²={results[h]['r2']:.4f}")
"""
1步 (窗口=36): RMSE=10.0517 MW, MAE=6.9142 MW, R²=0.9724 
4步 (窗口=48): RMSE=20.1461 MW, MAE=13.1948 MW, R²=0.8893 
8步 (窗口=72): RMSE=28.8546 MW, MAE=19.1758 MW, R²=0.7686
"""