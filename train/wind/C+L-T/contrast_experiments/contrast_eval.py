import os
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import math
from joblib import load
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# ============================================
# 1. 结构与设置
# ============================================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
os.makedirs('../../../../contrast', exist_ok=True)


# 1.1 基线 GRU 结构
class BaselineGRU(nn.Module):
    def __init__(self, input_dim=4, hidden_dim=64):
        super().__init__()
        self.gru = nn.GRU(input_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        _, h_n = self.gru(x)
        return self.fc(h_n.squeeze(0))


# 1.2 你的方案 V4 Transformer 结构 (必须原样保留以加载权重)
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
    def __init__(self, input_dim, horizon=1):
        super().__init__()
        self.embedding = nn.Linear(input_dim, 256)
        self.pos_encoder = PositionalEncoding(d_model=256)
        encoder_layer = nn.TransformerEncoderLayer(d_model=256, nhead=8, dim_feedforward=512, batch_first=True,
                                                   dropout=0.1)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=4)
        self.attention_pooling = nn.Sequential(nn.Linear(256, 128), nn.Tanh(), nn.Linear(128, 1))
        self.fc = nn.Sequential(nn.Linear(256, 128), nn.ReLU(), nn.Dropout(0.2), nn.Linear(128, 64), nn.ReLU(),
                                nn.Dropout(0.2), nn.Linear(64, horizon))

    def forward(self, x):
        x = self.embedding(x)
        x = self.pos_encoder(x)
        x = self.transformer(x)
        attention_weights = torch.softmax(self.attention_pooling(x), dim=1)
        context = torch.sum(x * attention_weights, dim=1)
        return self.fc(context)


# ============================================
# 2. 加载数据 (提取前600个采样点)
# ============================================
print("加载数据与特征...")
multi_step_datasets = load("../../../../multi_step_datasets.pkl")
scaler_y = load("../../../../scaler_y")
data_1step = multi_step_datasets[1]

num_samples = 600

# 准备真实值
y_true_raw_scaled = data_1step['test_y_raw'][:num_samples].numpy()
y_true = scaler_y.inverse_transform(y_true_raw_scaled).flatten()

# 准备基线模型特征 (轮毂高度风速、轮毂高度风向、温度、实际发电功率)
test_x_full = data_1step['test_x'][:num_samples]
test_x_raw = test_x_full[:, :, [5, 10, 11, 14]]
test_x_raw_flat = test_x_raw.numpy().reshape(test_x_raw.shape[0], -1)

# 准备训练集数据用于评估
train_x_full = data_1step['train_x']
train_x_raw = train_x_full[:, :, [5, 10, 11, 14]]
train_x_raw_flat = train_x_raw.numpy().reshape(train_x_raw.shape[0], -1)
train_y_raw_scaled = data_1step['train_y'].numpy()
train_y = scaler_y.inverse_transform(train_y_raw_scaled).flatten()

# 准备训练集数据用于评估
train_x_full = data_1step['train_x']
train_x_raw = train_x_full[:, :, [5, 10, 11, 14]]
train_x_raw_flat = train_x_raw.numpy().reshape(train_x_raw.shape[0], -1)
train_y_raw_scaled = data_1step['train_y'].numpy()
train_y = scaler_y.inverse_transform(train_y_raw_scaled).flatten()

# 准备训练集数据用于评估
train_x_full = data_1step['train_x']
train_x_raw = train_x_full[:, :, [5, 10, 11, 14]]
train_x_raw_flat = train_x_raw.numpy().reshape(train_x_raw.shape[0], -1)
train_y_raw_scaled = data_1step['train_y'].numpy()
train_y = scaler_y.inverse_transform(train_y_raw_scaled).flatten()

# 准备提议模型特征 (加载选定的特征索引)
selected_features = np.load("../../../../selected_features_indices.npy")
test_x_enhanced = test_x_full[:, :, selected_features]

predictions = {}

# ============================================
# 3. 加载模型并推理
# ============================================
print("开始推理预测...")

# 推理 GRU
gru = BaselineGRU(4).to(device)
gru.load_state_dict(torch.load("../../../../contrast/Baseline_GRU.pth"))
gru.eval()
with torch.no_grad():
    predictions['GRU'] = scaler_y.inverse_transform(gru(test_x_raw.to(device)).cpu().numpy()).flatten()

# 推理 RF
rf = load("../../../../contrast/Baseline_RF.joblib")
predictions['随机森林'] = scaler_y.inverse_transform(rf.predict(test_x_raw_flat).reshape(-1, 1)).flatten()

# 推理 XGBoost
xgb_model = load("../../../../contrast/Baseline_XGBoost.joblib")
predictions['XGBoost'] = scaler_y.inverse_transform(
    xgb_model.predict(test_x_raw_flat).reshape(-1, 1)).flatten()

# XGBoost 训练集与测试集指标评估
print("\n=== XGBoost 模型评估 ===")
xgb_train_pred_scaled = xgb_model.predict(train_x_raw_flat).reshape(-1, 1)
xgb_train_pred = scaler_y.inverse_transform(xgb_train_pred_scaled).flatten()
xgb_test_pred = predictions['XGBoost']

for set_name, y_pred, y_true_set in [('训练集', xgb_train_pred, train_y), ('测试集', predictions['XGBoost'], y_true)]:
    rmse = np.sqrt(mean_squared_error(y_true_set, y_pred))
    mae = mean_absolute_error(y_true_set, y_pred)
    r2 = r2_score(y_true_set, y_pred)
    print(f"[{set_name}] RMSE: {rmse:.4f} MW | MAE: {mae:.4f} MW | R2: {r2:.4f}")



for set_name, y_pred, y_true_set in [('训练集', xgb_train_pred, train_y), ('测试集', predictions['XGBoost'], y_true)]:
    rmse = np.sqrt(mean_squared_error(y_true_set, y_pred))
    mae = mean_absolute_error(y_true_set, y_pred)
    r2 = r2_score(y_true_set, y_pred)
    print(f"[{set_name}] RMSE: {rmse:.4f} MW | MAE: {mae:.4f} MW | R2: {r2:.4f}")

# 推理 Proposed: V6 Ensemble (加载资产)
print("加载 Proposed 模型 V6 集成资产...")
ensemble_package = torch.load("../../../../ensemble_models_h1_v6.pth", map_location=device, weights_only=False)
feature_dim = ensemble_package['feature_dim']
num_models = ensemble_package['num_models']

proposed_preds_list = []
for i in range(num_models):
    model = SimpleMultiStepTransformer(input_dim=feature_dim, horizon=1).to(device)
    model.load_state_dict(ensemble_package['model_state_dicts'][i])
    model.eval()
    with torch.no_grad():
        pred = model(test_x_enhanced.to(device)).cpu().numpy()
        proposed_preds_list.append(pred)

# 取 5 个模型的平均结果
pred_ensemble_scaled = np.mean(proposed_preds_list, axis=0)
predictions['CEEMDAN+LightGBM-Transformer'] = scaler_y.inverse_transform(pred_ensemble_scaled).flatten()

# ============================================
# 4. 评估结果输出
# ============================================
print("\n=== 对比评估结果 (单步预测, 600样本点) ===")
for name, pred in predictions.items():
    rmse = np.sqrt(mean_squared_error(y_true, pred))
    mae = mean_absolute_error(y_true, pred)
    r2 = r2_score(y_true, pred)
    print(f"[{name}]")
    print(f"  RMSE: {rmse:.4f} MW | MAE: {mae:.4f} MW | R²: {r2:.4f}")

# ============================================
# 5. 绘图 1: 综合折线预测图
# ============================================
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

plt.figure(figsize=(18, 6))
plt.plot(y_true, label='真实功率', color='black', linewidth=2.5, zorder=5)

colors = {'GRU': '#1f77b4', '随机森林': '#2ca02c',
          'XGBoost': '#ff7f0e', 'CEEMDAN+LightGBM-Transformer': '#d62728'}

for name, pred in predictions.items():
    linewidth = 2 if name == 'CEEMDAN+LightGBM-Transformer' else 1.2
    alpha = 1.0 if name == 'CEEMDAN+LightGBM-Transformer' else 0.7
    plt.plot(pred, label=f'{name}', color=colors[name], linewidth=linewidth, alpha=alpha)

plt.title('风电功率预测-预测曲线对比', fontsize=14, pad=15)
plt.xlabel('时间步 (15分钟)', fontsize=12)
plt.ylabel('功率 (MW)', fontsize=12)
plt.legend(loc='upper right', framealpha=0.9)
plt.grid(True, linestyle='--', alpha=0.5)
plt.tight_layout()
plt.savefig('./contrast_experiments/prediction_curves_comparison.png', dpi=200)
print("\n[+] 已保存: ./contrast_experiments/prediction_curves_comparison.png")
plt.close()

# ============================================
# 6. 绘图 2: 各模型拟合散点图 (分开绘制并保存)
# ============================================
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

for name, pred in predictions.items():
    plt.figure(figsize=(8, 6))
    
    max_val = max(np.max(y_true), np.max(pred))
    min_val = min(np.min(y_true), np.min(pred))
    
    plt.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label='理想拟合')
    plt.scatter(y_true, pred, alpha=0.6, color=colors[name], edgecolor='k', s=30)
    
    plt.title(f'{name} 拟合效果', fontsize=12)
    plt.xlabel('真实功率 (MW)')
    plt.ylabel('预测功率 (MW)')
    plt.legend(loc='upper left')
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    
    filename = f"./contrast_experiments/scatter_{name}.png".replace('-', '_')
    plt.savefig(filename, dpi=150)
    print(f"[+] 已保存: {filename}")
    plt.close()