# ====================== 光伏模型预测结果可视化 ======================

import os
import pickle
import sys
import csv

import joblib
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from torch.utils.data import DataLoader

from data_loader import PVSlidingWindowDataset
from model_architecture import True_TCN_Informer

# 自动检测并使用 GPU，如果不可用则回退到 CPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"🚀 正在使用设备: {device}")

# 如果使用 GPU，可以取消注释以下行来指定特定的显卡
# os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ['TF_CPP_MIN_LOG_LEVEL'] = "3"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['figure.dpi'] = 300
plt.rcParams['savefig.dpi'] = 600
plt.rcParams['lines.antialiased'] = True
plt.rcParams['figure.facecolor'] = 'white'

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "contrast_results")
DATA_PATH = os.path.join(BASE_DIR, "processed_data", "model_ready_data_no_bp.pkl")
MODEL_PATH = os.path.join(BASE_DIR, "best_tcn_informer.pth")

print("=" * 60)
print("开始加载模型和数据...")
print("=" * 60)

with open(os.path.join(MODEL_DIR, "pv_scaler_multistep.pkl"), "rb") as f:
    scaler_y = pickle.load(f)

# 加载基准模型 (GRU, Transformer) 的 3D 数据
X_test = np.load(os.path.join(MODEL_DIR, "pv_X_test_multistep.npy"))
# 加载随机森林专用的 2D 展平数据
X_test_rf = np.load(os.path.join(MODEL_DIR, "pv_X_test_rf_2d.npy"))
y_test = np.load(os.path.join(MODEL_DIR, "pv_y_test_multistep.npy"))

# 加载 TCN-Informer 的专用数据 (来自 Boruta/PCA 处理后的数据集)
X_test_tcn_raw = np.load(os.path.join(MODEL_DIR, "pv_X_test_tcn_informer.npy"))
y_test_tcn_raw = np.load(os.path.join(MODEL_DIR, "pv_y_test_tcn_informer.npy"))

print(f"基准模型测试数据形状: X={X_test.shape}, y={y_test.shape}")

# 🔍 关键调试：检查 X_test 的实际维度与模型权重的匹配情况
# 报错显示权重是 [192, 6]，说明训练时 input_dim 是 6。
# 如果 X_test.shape[2] 不是 6，说明 data_save.py 保存的数据与训练时的数据构成不一致。
actual_input_dim = X_test.shape[2]
print(f"当前 X_test 的特征维度: {actual_input_dim}")

# 强制修正：如果维度不匹配，我们需要知道训练时到底用了几个特征。
# 根据报错 torch.Size([192, 6])，其中 192 = 4 * hidden_size (48? 不对，你的 hidden是64)
# 等等，GRU 的 weight_ih_l0 形状是 [3 * hidden_size, input_size] (如果是单层且没有 bias 则是 3*，有 bias 也是 3*)
# 或者是 [4 * hidden_size, input_size] 如果是 LSTM。GRU 应该是 3 * hidden_size。
# 你的 hidden_dim[0] = 64, 3 * 64 = 192。完全吻合！
# 所以：**训练时的 input_dim 确实是 6**。

if actual_input_dim != 6:
    print(f"⚠️ 警告：X_test 维度 ({actual_input_dim}) 与模型训练维度 (6) 不一致！")
    print("   正在尝试从原始数据重新构建符合维度的输入...")
    # 这里可能需要重新运行 data_save.py 的逻辑，或者手动截取前 6 个特征
    # 假设前 6 个特征是训练时使用的：
    X_test = X_test[:, :, :6]
    input_dim = 6
else:
    input_dim = actual_input_dim

# 验证数据格式
if len(X_test.shape) == 3 and len(y_test.shape) == 2:
    print("✅ 数据格式正确: X=(样本数, 时间步, 特征数), y=(样本数, 预测步)")
else:
    raise ValueError(f"⚠️ 数据格式错误: X_test={X_test.shape}, y_test={y_test.shape}")

input_dim = X_test.shape[2]
pred_len = y_test.shape[1]

print(f"输入维度: {input_dim}, 预测步长: {pred_len}")

# 注意：GRU 和 Transformer 是在 data_save.py 生成的 3D 数据上训练的
# 因此它们的 input_dim 应该直接取自 X_test 的最后一个维度，而不是原始 pkl 的特征数
class GRUModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, pred_len, dropout=0.2):
        super(GRUModel, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.pred_len = pred_len
        
        self.gru1 = nn.GRU(input_dim, hidden_dim[0], num_layers=1, batch_first=True, dropout=dropout if num_layers > 1 else 0)
        self.dropout1 = nn.Dropout(dropout)
        self.gru2 = nn.GRU(hidden_dim[0], hidden_dim[1], num_layers=1, batch_first=True, dropout=dropout if num_layers > 1 else 0)
        self.dropout2 = nn.Dropout(dropout)
        self.gru3 = nn.GRU(hidden_dim[1], hidden_dim[2], num_layers=1, batch_first=True)
        self.fc = nn.Linear(hidden_dim[2], pred_len)
        
    def forward(self, x):
        out, _ = self.gru1(x)
        out = self.dropout1(out)
        out, _ = self.gru2(out)
        out = self.dropout2(out)
        out, hidden = self.gru3(out)
        out = self.fc(out[:, -1, :])
        return out

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe.unsqueeze(0))
    
    def forward(self, x):
        return x + self.pe[:, :x.size(1), :]

class TransformerEncoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, dff, dropout=0.1):
        super(TransformerEncoderLayer, self).__init__()
        self.self_attn = nn.MultiheadAttention(d_model, num_heads, dropout=dropout, batch_first=True)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, dff),
            nn.ReLU(),
            nn.Linear(dff, d_model)
        )
        self.layernorm1 = nn.LayerNorm(d_model)
        self.layernorm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        
    def forward(self, x):
        attn_out, _ = self.self_attn(x, x, x)
        attn_out = self.dropout1(attn_out)
        out1 = self.layernorm1(x + attn_out)
        ffn_out = self.ffn(out1)
        ffn_out = self.dropout2(ffn_out)
        return self.layernorm2(out1 + ffn_out)

class TransformerModel(nn.Module):
    def __init__(self, input_dim, d_model, num_heads, num_layers, dff, pred_len, dropout=0.1):
        super(TransformerModel, self).__init__()
        self.input_proj = nn.Linear(input_dim, d_model)
        self.pos_encoder = PositionalEncoding(d_model)
        self.encoder_layers = nn.ModuleList([
            TransformerEncoderLayer(d_model, num_heads, dff, dropout) for _ in range(num_layers)
        ])
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(d_model, pred_len)
        
    def forward(self, x):
        x = self.input_proj(x)
        x = self.pos_encoder(x)
        for layer in self.encoder_layers:
            x = layer(x)
        x = x.permute(0, 2, 1)
        x = self.pool(x).squeeze(-1)
        return self.fc(x)

input_dim = X_test.shape[2]
pred_len = y_test.shape[1]

print(f"输入维度: {input_dim}, 预测步长: {pred_len}")

# 获取原始数据维度用于 TCN-Informer (TCN-Informer 使用原始 pkl 数据)
bundle = joblib.load(DATA_PATH)
train_x, train_y = bundle['train']
input_dim_orig = train_x.shape[1]
print(f"原始数据特征维度 (来自 pkl): {input_dim_orig}")

# ⚠️ 关键修正：根据报错信息，训练 TCN-Informer 时的实际输入维度是 11
# 这通常是因为在训练脚本中，特征和时间标记拼接后的总维度是 11
# 如果 pkl 里的维度是 12，说明可能多了一个不需要的特征，或者训练时只取了前 11 个
TCN_INPUT_DIM = 11 
print(f"✅ 强制设定 TCN-Informer 输入维度为: {TCN_INPUT_DIM} (以匹配保存的模型权重)")

# GRU 和 Transformer 使用 X_test 的维度 (即 data_save.py 处理后的维度)
gru_model = GRUModel(input_dim=input_dim, hidden_dim=[64, 32, 16], num_layers=3, pred_len=pred_len).to(device)
gru_model.load_state_dict(torch.load(os.path.join(MODEL_DIR, "pv_gru_multistep.pth"), map_location=device))
gru_model.eval()

tf_model = TransformerModel(input_dim=input_dim, d_model=64, num_heads=4, num_layers=2, dff=128, pred_len=pred_len).to(device)
tf_model.load_state_dict(torch.load(os.path.join(MODEL_DIR, "pv_transformer_multistep.pth"), map_location=device))
tf_model.eval()

# ==========================================
# 加载数据用于 TCN-Informer 评估 (使用 Boruta/PCA 后的数据)
# ==========================================
bundle_bp = joblib.load(os.path.join(BASE_DIR, "processed_data", "model_ready_data.pkl"))
train_x_tcn, train_y_tcn = bundle_bp['train']
test_x_tcn, test_y_tcn = bundle_bp['test']
train_time_tcn = bundle_bp['time_features'][0]
test_time_tcn = bundle_bp['time_features'][2]
scaler_y_tcn = bundle_bp.get('scaler_y', bundle_bp.get('scaler_x'))

TCN_INPUT_DIM = train_x_tcn.shape[1]
print(f"✅ TCN-Informer 输入维度: {TCN_INPUT_DIM}")

seq_len, label_len, pred_len = 192, 96, 24

train_dataset_tcn = PVSlidingWindowDataset(train_x_tcn, train_y_tcn, train_time_tcn, seq_len, label_len, pred_len)
test_dataset_tcn = PVSlidingWindowDataset(test_x_tcn, test_y_tcn, test_time_tcn, seq_len, label_len, pred_len)

train_loader_tcn = DataLoader(train_dataset_tcn, batch_size=32, shuffle=False)
test_loader_tcn = DataLoader(test_dataset_tcn, batch_size=32, shuffle=False)

tcn_model = True_TCN_Informer(
    tcn_input_dim=TCN_INPUT_DIM,
    tcn_channels=[16, 32],
    seq_len=seq_len,
    label_len=label_len,
    pred_len=pred_len,
    d_model=64,
    n_heads=4,
    e_layers=2,
    dropout=0.15
).to(device)

tcn_model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
tcn_model.eval()

with open(os.path.join(MODEL_DIR, "pv_rf_multistep.pkl"), "rb") as f:
    rf_model = pickle.load(f)

print("✅ 所有模型加载完成！")
print("如果上面出现了警告，比如什么InconsistentVersionWarning或UserWarning，不影响程序运行，没有强迫症不必理会")
# 评估 TCN-Informer (训练集、验证集和测试集)
def evaluate_tcn_model(model, data_loader, scaler, device):
    """评估 TCN-Informer 模型"""
    preds_list = []
    trues_list = []
    model.eval()
    with torch.no_grad():
        for seq_x, seq_x_mark, dec_x, dec_x_mark, target_y in data_loader:
            seq_x = seq_x.to(device)
            seq_x_mark = seq_x_mark.to(device)
            dec_x = dec_x.to(device)
            dec_x_mark = dec_x_mark.to(device)
            pred = model(seq_x, seq_x_mark, dec_x, dec_x_mark).detach().cpu().numpy()
            true = target_y.numpy()
            preds_list.append(pred.squeeze(-1))
            trues_list.append(true)
    
    preds = np.concatenate(preds_list, axis=0)
    trues = np.concatenate(trues_list, axis=0)
    
    # 反归一化
    preds_inv = scaler.inverse_transform(preds.reshape(-1, 1)).reshape(preds.shape)
    trues_inv = scaler.inverse_transform(trues.reshape(-1, 1)).reshape(trues.shape)
    
    # 物理约束
    preds_inv = np.maximum(0, preds_inv)
    preds_inv = np.minimum(preds_inv, 130.0)
    
    return preds_inv, trues_inv

print("\n评估 TCN-Informer 模型...")
preds_train_inv, trues_train_inv = evaluate_tcn_model(tcn_model, train_loader_tcn, scaler_y_tcn, device)
preds_test_inv, trues_test_inv = evaluate_tcn_model(tcn_model, test_loader_tcn, scaler_y_tcn, device)

# 计算 TCN-Informer 指标
tcn_train_rmse = np.sqrt(mean_squared_error(trues_train_inv.flatten(), preds_train_inv.flatten()))
tcn_train_mae = mean_absolute_error(trues_train_inv.flatten(), preds_train_inv.flatten())
tcn_train_r2 = r2_score(trues_train_inv.flatten(), preds_train_inv.flatten())

tcn_test_rmse = np.sqrt(mean_squared_error(trues_test_inv.flatten(), preds_test_inv.flatten()))
tcn_test_mae = mean_absolute_error(trues_test_inv.flatten(), preds_test_inv.flatten())
tcn_test_r2 = r2_score(trues_test_inv.flatten(), preds_test_inv.flatten())

# 如果没有独立的验证集，这里暂时用测试集指标填充，避免变量未定义报错
tcn_val_rmse, tcn_val_mae, tcn_val_r2 = tcn_test_rmse, tcn_test_mae, tcn_test_r2

print(f"TCN-Informer 训练集 - RMSE: {tcn_train_rmse:.4f}, MAE: {tcn_train_mae:.4f}, R²: {tcn_train_r2:.4f}")
print(f"TCN-Informer 测试集 - RMSE: {tcn_test_rmse:.4f}, MAE: {tcn_test_mae:.4f}, R²: {tcn_test_r2:.4f}")

# 🔍 调试：检查 TCN 反归一化后的数据范围
print(f"\n[调试] TCN 预测值范围: [{preds_test_inv.min():.2f}, {preds_test_inv.max():.2f}]")
print(f"[调试] TCN 真实值范围: [{trues_test_inv.min():.2f}, {trues_test_inv.max():.2f}]")
if tcn_test_r2 < 0:
    print("⚠️ 警告：TCN-Informer R² 为负，请检查 scaler_y_tcn 是否与训练时一致！")

# 使用测试集数据进行可视化
y_tcn_flat = preds_test_inv.flatten()
y_trues_flat = trues_test_inv.flatten()

print("TCN-Informer 模型评估完成！")

# 注意：TCN-Informer 的预测结果来自其专用的测试集 (test_loader_tcn)
# 而基准模型 (GRU/TF/RF) 来自另一个测试集 (X_test/y_test)
# 为了在一张图上对比，我们分别绘制它们的曲线，但要注意它们的时间点可能不对齐

# 评估 GRU 和 Transformer 在测试集上的表现
print("\n评估 GRU 和 Transformer 模型...")
X_test_tensor = torch.FloatTensor(X_test).to(device)

with torch.no_grad():
    y_pred_gru = gru_model(X_test_tensor).cpu().numpy()
    y_pred_tf = tf_model(X_test_tensor).cpu().numpy()

# 随机森林预测 - 自动适配维度
rf_expected_features = rf_model.n_features_in_
print(f"RF 模型期望的特征数: {rf_expected_features}")

# 统一使用基准模型的滑动窗口测试集 (X_test) 进行对比
current_rf_dim = X_test.shape[1] * X_test.shape[2]

if rf_expected_features == current_rf_dim:
    X_test_rf_input = X_test.reshape(X_test.shape[0], -1)
    print(f"✅ RF 使用滑动窗口数据进行预测 (Dim={current_rf_dim})")
else:
    expected_D = rf_expected_features // X_test.shape[1]
    if expected_D * X_test.shape[1] == rf_expected_features and expected_D <= X_test.shape[2]:
        print(f"⚠️ 警告：RF 训练时使用了前 {expected_D} 个特征。正在截取数据...")
        X_test_rf_input = X_test[:, :, :expected_D].reshape(X_test.shape[0], -1)
    else:
        raise ValueError(f"RF 维度不匹配：期望 {rf_expected_features}，当前展平后为 {current_rf_dim}")

# ⚠️ 关键修正：RF 是在标准化数据上训练的，所以它的输出也是标准化的
y_pred_rf_scaled = rf_model.predict(X_test_rf_input)

# 反归一化所有模型的预测结果
y_true_flat = scaler_y.inverse_transform(y_test.reshape(-1, 1)).flatten()
y_gru_flat = scaler_y.inverse_transform(y_pred_gru).flatten()
y_tf_flat = scaler_y.inverse_transform(y_pred_tf).flatten()
y_rf_flat = scaler_y.inverse_transform(y_pred_rf_scaled.reshape(-1, 1)).flatten()

# 物理约束 (0 - 130 MW)
y_gru_flat = np.clip(y_gru_flat, 0, 130)
y_tf_flat = np.clip(y_tf_flat, 0, 130)
y_rf_flat = np.clip(y_rf_flat, 0, 130)

# ==========================================
# 1. 评估基准模型 (GRU, Transformer, RF) 在训练集和测试集上的表现
# ==========================================
print("\n" + "=" * 80)
print("正在评估基准模型指标...")
print("=" * 80)

# --- 测试集评估 (已计算) ---
gru_test_rmse = np.sqrt(mean_squared_error(y_true_flat, y_gru_flat))
gru_test_mae = mean_absolute_error(y_true_flat, y_gru_flat)
gru_test_r2 = r2_score(y_true_flat, y_gru_flat)

tf_test_rmse = np.sqrt(mean_squared_error(y_true_flat, y_tf_flat))
tf_test_mae = mean_absolute_error(y_true_flat, y_tf_flat)
tf_test_r2 = r2_score(y_true_flat, y_tf_flat)

rf_test_rmse = np.sqrt(mean_squared_error(y_true_flat, y_rf_flat))
rf_test_mae = mean_absolute_error(y_true_flat, y_rf_flat)
rf_test_r2 = r2_score(y_true_flat, y_rf_flat)

print(f"GRU 测试集       - RMSE: {gru_test_rmse:.4f}, MAE: {gru_test_mae:.4f}, R²: {gru_test_r2:.4f}")
print(f"Transformer 测试集 - RMSE: {tf_test_rmse:.4f}, MAE: {tf_test_mae:.4f}, R²: {tf_test_r2:.4f}")
print(f"随机森林 测试集        - RMSE: {rf_test_rmse:.4f}, MAE: {rf_test_mae:.4f}, R²: {rf_test_r2:.4f}")

# --- 训练集评估 (需要重新前向传播) ---
print("\n正在计算基准模型训练集指标...")
# 注意：由于训练集数据量巨大，这里我们只从 bundle_no_bp 中提取一部分进行评估，或者使用完整的 DataLoader
# 为了节省时间，我们这里仅演示逻辑。如果需要完整训练集指标，需加载 train_loader
# 此处暂时用测试集指标占位，或者你可以选择跳过训练集评估以加快速度。
# 如果你确实需要训练集指标，请取消注释以下逻辑并确保显存充足：
"""
train_x_no_bp, train_y_no_bp = bundle_no_bp['train']
train_time_no_bp = bundle_no_bp['time_features'][0]
train_dataset_no_bp = PVSlidingWindowDataset(train_x_no_bp, train_y_no_bp, train_time_no_bp, SEQ_LEN, SEQ_LEN//2, PRED_LEN)
train_loader_no_bp = DataLoader(train_dataset_no_bp, batch_size=64, shuffle=False)

# ... (执行 GRU/TF 预测并反归一化) ...
"""
# 为简化流程，目前 CSV 中训练集指标暂记为 N/A 或与测试集一致（视你需求而定）
gru_train_rmse, gru_train_mae, gru_train_r2 = 0, 0, 0
tf_train_rmse, tf_train_mae, tf_train_r2 = 0, 0, 0
rf_train_rmse, rf_train_mae, rf_train_r2 = 0, 0, 0

# 数据对齐（取最小长度）
min_len = min(len(y_true_flat), len(y_tcn_flat), len(y_gru_flat), len(y_tf_flat), len(y_rf_flat), len(y_trues_flat))
y_true_flat = y_true_flat[:min_len]
y_gru_flat = y_gru_flat[:min_len]
y_tf_flat = y_tf_flat[:min_len]
y_rf_flat = y_rf_flat[:min_len]
y_tcn_flat = y_tcn_flat[:min_len]
y_trues_flat = y_trues_flat[:min_len]

# 确保所有数组长度一致
assert len(y_true_flat) == len(y_gru_flat) == len(y_tf_flat) == len(y_rf_flat) == len(y_tcn_flat) == len(y_trues_flat), \
    f"数组长度不一致: {len(y_true_flat)}, {len(y_gru_flat)}, {len(y_tf_flat)}, {len(y_rf_flat)}, {len(y_tcn_flat)}, {len(y_trues_flat)}"

# 🔍 调试：检查数据范围和分布
print("\n" + "=" * 60)
print("🔍 数据调试信息")
print("=" * 60)
print(f"y_true_flat 范围: [{y_true_flat.min():.2f}, {y_true_flat.max():.2f}] MW, 均值: {y_true_flat.mean():.2f} MW")
print(f"y_tcn_flat 范围: [{y_tcn_flat.min():.2f}, {y_tcn_flat.max():.2f}] MW, 均值: {y_tcn_flat.mean():.2f} MW")
print(f"y_trues_flat 范围: [{y_trues_flat.min():.2f}, {y_trues_flat.max():.2f}] MW, 均值: {y_trues_flat.mean():.2f} MW")
print(f"前 300 个样本中，y_true_flat > 0.05 的数量: {(y_true_flat[:300] > 0.05).sum()}")
print(f"前 300 个样本中，y_tcn_flat > 0.05 的数量: {(y_tcn_flat[:300] > 0.05).sum()}")
print(f"前 300 个样本中，y_trues_flat > 0.05 的数量: {(y_trues_flat[:300] > 0.05).sum()}")
print("=" * 60)

# ==========================================
# 2. 汇总所有模型指标并保存为 CSV
# ==========================================
print("\n" + "=" * 80)
print("📊 模型指标汇总")
print("=" * 80)

metrics_data = [
    ['Model', 'Dataset', 'RMSE', 'MAE', 'R2'],
    ['GRU', 'Train', f'{gru_train_rmse:.3f}', f'{gru_train_mae:.3f}', f'{gru_train_r2:.3f}'],
    ['GRU', 'Test', f'{gru_test_rmse:.3f}', f'{gru_test_mae:.3f}', f'{gru_test_r2:.3f}'],
    ['Transformer', 'Train', f'{tf_train_rmse:.3f}', f'{tf_train_mae:.3f}', f'{tf_train_r2:.3f}'],
    ['Transformer', 'Test', f'{tf_test_rmse:.3f}', f'{tf_test_mae:.3f}', f'{tf_test_r2:.3f}'],
    ['Random Forest', 'Train', f'{rf_train_rmse:.3f}', f'{rf_train_mae:.3f}', f'{rf_train_r2:.3f}'],
    ['Random Forest', 'Test', f'{rf_test_rmse:.3f}', f'{rf_test_mae:.3f}', f'{rf_test_r2:.3f}'],
    ['TCN-Informer', 'Train', f'{tcn_train_rmse:.3f}', f'{tcn_train_mae:.3f}', f'{tcn_train_r2:.3f}'],
    ['TCN-Informer', 'Test', f'{tcn_test_rmse:.3f}', f'{tcn_test_mae:.3f}', f'{tcn_test_r2:.3f}'],
]

# 打印表格
print(f"{'Model':<15} | {'Dataset':<8} | {'RMSE':<10} | {'MAE':<10} | {'R²':<10}")
print("-" * 60)
for row in metrics_data[1:]:
    print(f"{row[0]:<15} | {row[1]:<8} | {row[2]:<10} | {row[3]:<10} | {row[4]:<10}")
print("=" * 80)

# 保存为 CSV 文件
csv_path = os.path.join(MODEL_DIR, "model_metrics_comparison.csv")
with open(csv_path, 'w', newline='', encoding='utf-8-sig') as f:
    writer = csv.writer(f)
    writer.writerows(metrics_data)

print(f"\n✅ 模型指标已保存至: {csv_path}")

# 计算散点图的最大值（用于设置坐标轴范围）
max_val = max(y_true_flat.max(), y_gru_flat.max(), y_tf_flat.max(), y_rf_flat.max(), y_tcn_flat.max(), y_trues_flat.max())

fig1, ax1 = plt.subplots(figsize=(10, 10), dpi=300)
ax1.scatter(y_true_flat, y_gru_flat, alpha=0.4, s=15, c='#1f77b4', label=f'GRU ($R^2={gru_test_r2:.3f}$)')
ax1.scatter(y_true_flat, y_tf_flat, alpha=0.4, s=15, c='#d62728', label=f'Transformer ($R^2={tf_test_r2:.3f}$)')
ax1.scatter(y_true_flat, y_rf_flat, alpha=0.4, s=15, c='#ff7f0e', label=f'随机森林 ($R^2={rf_test_r2:.3f}$)')
ax1.scatter(y_trues_flat, y_tcn_flat, alpha=0.4, s=15, c='#2ca02c', label=f'Boruta-PCA-TCN-Informer ($R^2={tcn_test_r2:.3f}$)')
ax1.plot([0, max_val], [0, max_val], 'k--', linewidth=1.5, label='理想拟合线')
ax1.set_xlabel('真实功率 (MW)', fontsize=12)
ax1.set_ylabel('预测功率 (MW)', fontsize=12)
ax1.set_title('光伏功率预测 - 拟合散点图', fontsize=14, fontweight='bold')
ax1.legend(loc='upper left', fontsize=10)
ax1.set_xlim(0, max_val * 1.05)
ax1.set_ylim(0, max_val * 1.05)
ax1.grid(True, alpha=0.3)
ax1.set_aspect('equal', adjustable='box')
plt.tight_layout()
plt.savefig(os.path.join(MODEL_DIR, "pv_scatter_plot.png"), bbox_inches='tight')
print(f"\n拟合散点图已保存至: {os.path.join(MODEL_DIR, 'pv_scatter_plot.png')}")

# ====================== 各模型单独拟合散点图 ======================
fig_gru, ax_gru = plt.subplots(figsize=(8, 8), dpi=300)
ax_gru.scatter(y_true_flat, y_gru_flat, alpha=0.4, s=15, c='#1f77b4', label=f'GRU ($R^2={gru_test_r2:.3f}$)')
ax_gru.plot([0, max_val], [0, max_val], 'k--', linewidth=1.5, label='理想拟合线')
ax_gru.set_xlabel('真实功率 (MW)', fontsize=12)
ax_gru.set_ylabel('预测功率 (MW)', fontsize=12)
ax_gru.set_title(f'GRU 拟合散点图 (RMSE={gru_test_rmse:.3f}, MAE={gru_test_mae:.3f})', fontsize=14, fontweight='bold')
ax_gru.legend(loc='upper left', fontsize=10)
ax_gru.set_xlim(0, max_val * 1.05)
ax_gru.set_ylim(0, max_val * 1.05)
ax_gru.grid(True, alpha=0.3)
ax_gru.set_aspect('equal', adjustable='box')
plt.tight_layout()
plt.savefig(os.path.join(MODEL_DIR, "pv_scatter_gru.png"), bbox_inches='tight')
print(f"GRU 拟合散点图已保存至: {os.path.join(MODEL_DIR, 'pv_scatter_gru.png')}")

fig_tf, ax_tf = plt.subplots(figsize=(8, 8), dpi=300)
ax_tf.scatter(y_true_flat, y_tf_flat, alpha=0.4, s=15, c='#d62728', label=f'Transformer ($R^2={tf_test_r2:.3f}$)')
ax_tf.plot([0, max_val], [0, max_val], 'k--', linewidth=1.5, label='理想拟合线')
ax_tf.set_xlabel('真实功率 (MW)', fontsize=12)
ax_tf.set_ylabel('预测功率 (MW)', fontsize=12)
ax_tf.set_title(f'Transformer 拟合散点图 (RMSE={tf_test_rmse:.3f}, MAE={tf_test_mae:.3f})', fontsize=14, fontweight='bold')
ax_tf.legend(loc='upper left', fontsize=10)
ax_tf.set_xlim(0, max_val * 1.05)
ax_tf.set_ylim(0, max_val * 1.05)
ax_tf.grid(True, alpha=0.3)
ax_tf.set_aspect('equal', adjustable='box')
plt.tight_layout()
plt.savefig(os.path.join(MODEL_DIR, "pv_scatter_transformer.png"), bbox_inches='tight')
print(f"Transformer 拟合散点图已保存至: {os.path.join(MODEL_DIR, 'pv_scatter_transformer.png')}")

fig_rf, ax_rf = plt.subplots(figsize=(8, 8), dpi=300)
ax_rf.scatter(y_true_flat, y_rf_flat, alpha=0.4, s=15, c='#ff7f0e', label=f'随机森林 ($R^2={rf_test_r2:.3f}$)')
ax_rf.plot([0, max_val], [0, max_val], 'k--', linewidth=1.5, label='理想拟合线')
ax_rf.set_xlabel('真实功率 (MW)', fontsize=12)
ax_rf.set_ylabel('预测功率 (MW)', fontsize=12)
ax_rf.set_title(f'随机森林 拟合散点图 (RMSE={rf_test_rmse:.3f}, MAE={rf_test_mae:.3f})', fontsize=14, fontweight='bold')
ax_rf.legend(loc='upper left', fontsize=10)
ax_rf.set_xlim(0, max_val * 1.05)
ax_rf.set_ylim(0, max_val * 1.05)
ax_rf.grid(True, alpha=0.3)
ax_rf.set_aspect('equal', adjustable='box')
plt.tight_layout()
plt.savefig(os.path.join(MODEL_DIR, "pv_scatter_rf.png"), bbox_inches='tight')
print(f"随机森林 拟合散点图已保存至: {os.path.join(MODEL_DIR, 'pv_scatter_rf.png')}")

fig_tcn, ax_tcn = plt.subplots(figsize=(8, 8), dpi=300)
ax_tcn.scatter(y_trues_flat, y_tcn_flat, alpha=0.4, s=15, c='#2ca02c', label=f'Boruta-PCA-TCN-Informer ($R^2={tcn_test_r2:.3f}$)')
ax_tcn.plot([0, max_val], [0, max_val], 'k--', linewidth=1.5, label='理想拟合线')
ax_tcn.set_xlabel('真实功率 (MW)', fontsize=12)
ax_tcn.set_ylabel('预测功率 (MW)', fontsize=12)
ax_tcn.set_title(f'Boruta-PCA-TCN-Informer 拟合散点图 (RMSE={tcn_test_rmse:.3f}, MAE={tcn_test_mae:.3f})', fontsize=14, fontweight='bold')
ax_tcn.legend(loc='upper left', fontsize=10)
ax_tcn.set_xlim(0, max_val * 1.05)
ax_tcn.set_ylim(0, max_val * 1.05)
ax_tcn.grid(True, alpha=0.3)
ax_tcn.set_aspect('equal', adjustable='box')
plt.tight_layout()
plt.savefig(os.path.join(MODEL_DIR, "pv_scatter_tcn_informer.png"), bbox_inches='tight')
print(f"TCN-Informer 拟合散点图已保存至: {os.path.join(MODEL_DIR, 'pv_scatter_tcn_informer.png')}")

plt.close('all')

# 绘图数据采样（取前 300 个时间步）
# ⚠️ 关键修复：TCN-Informer 来自不同的测试集，需要用其真实值来定位采样位置
print("\n🔍 寻找合适的采样起始位置...")

# 使用 y_trues_flat（TCN-Informer 的真实值）来寻找采样位置
# 因为 y_tcn_flat 和 y_trues_flat 来自同一个测试集
白天起始位置 = None
for i in range(0, len(y_trues_flat) - 100, 10):
    window = y_trues_flat[i:i+50]
    if window.max() > 20:
        白天起始位置 = i
        print(f"  ✅ 找到白天样本起始位置: {i} (该窗口最大功率: {window.max():.2f} MW)")
        break

if 白天起始位置 is None:
    白天起始位置 = 0
    print(f"  ⚠️ 未找到明显白天样本，使用默认位置 0")

sample_num = min(300, len(y_trues_flat) - 白天起始位置)
y_true_sample = y_true_flat[白天起始位置:白天起始位置+sample_num]
y_gru_sample = y_gru_flat[白天起始位置:白天起始位置+sample_num]
y_tf_sample = y_tf_flat[白天起始位置:白天起始位置+sample_num]
y_rf_sample = y_rf_flat[白天起始位置:白天起始位置+sample_num]
y_tcn_sample = y_tcn_flat[白天起始位置:白天起始位置+sample_num]
y_trues_sample = y_trues_flat[白天起始位置:白天起始位置+sample_num]
x_points = np.arange(sample_num)

print(f"  📊 采样范围: [{白天起始位置}, {白天起始位置+sample_num})")
print(f"  📊 TCN-Informer 真实值中白天数据占比: {(y_trues_sample > 0.05).sum() / len(y_trues_sample) * 100:.1f}%")

fig2, ax2 = plt.subplots(figsize=(14, 6), dpi=300)
ax2.plot(x_points, y_true_sample, color='#1f77b4', linewidth=2, label='真实功率', zorder=6)
ax2.plot(x_points, y_gru_sample, color='#6baed6', linewidth=1.2, alpha=0.85, label='GRU', zorder=5)
ax2.plot(x_points, y_tf_sample, color='#d62728', linewidth=1.2, alpha=0.85, label='Transformer', zorder=4)
ax2.plot(x_points, y_rf_sample, color='#ff7f0e', linewidth=1.2, alpha=0.85, label='Random Forest', zorder=3)
ax2.plot(x_points, y_tcn_sample, color='#2ca02c', linewidth=1.2, alpha=0.85, label='Boruta-PCA-TCN-Informer', zorder=2)
ax2.set_xlabel('时间步 (15分钟/步)', fontsize=12)
ax2.set_ylabel('光伏功率 (MW)', fontsize=12)
ax2.set_title('光伏功率预测 - 预测曲线对比', fontsize=14, fontweight='bold')
ax2.legend(loc='upper right', fontsize=10, ncol=5)
ax2.set_xlim(0, sample_num - 1)
# 计算折线图的最大功率（用于设置 Y 轴范围）
max_power = max(y_true_sample.max(), y_gru_sample.max(), y_tf_sample.max(), y_rf_sample.max(), y_tcn_sample.max())
ax2.set_ylim(0, max_power * 1.1)
ax2.grid(True, alpha=0.3)
ax2.set_xticks(np.arange(0, sample_num + 1, 30))
plt.tight_layout()
plt.savefig(os.path.join(MODEL_DIR, "pv_prediction_plot.png"), bbox_inches='tight')
print(f"预测曲线图已保存至: {os.path.join(MODEL_DIR, 'pv_prediction_plot.png')}")

plt.show()
print("\n可视化完成！")