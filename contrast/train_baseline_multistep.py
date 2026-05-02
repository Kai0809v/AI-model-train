"""
训练基准模型（GRU/Transformer/RF）用于与 TCN-Informer 公平对比
- 预测任务：192 步历史 → 24 步未来（多步预测）
- 数据源：model_ready_data_no_bp.pkl（不带 Boruta 特征选择）
- 输出：保存到 contrast/ 目录
- 框架：PyTorch（支持 GPU 加速）
"""
import os
import sys
import pickle
import joblib
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

from data_loader import PVSlidingWindowDataset

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"使用设备: {device}")

# ==========================================
# 1. 全局配置
# ==========================================

# 超参数
SEQ_LEN = 192      # 输入序列长度（2天）
PRED_LEN = 24      # 预测步长（6小时）
BATCH_SIZE = 32
EPOCHS = 50
LEARNING_RATE = 1e-3
PATIENCE = 10      # 早停耐心值

# 路径配置（使用相对路径）
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# 基准模型使用不带 Boruta/PCA 的数据
DATA_PATH = os.path.join(BASE_DIR, "processed_data", "model_ready_data_no_bp.pkl")
MODEL_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "contrast_results")
os.makedirs(MODEL_DIR, exist_ok=True)

print("=" * 80)
print("开始训练基准模型（多步预测 192→24） - PyTorch 版本")
print("=" * 80)

# ==========================================
# 2. 加载数据
# ==========================================
print("\n1. 加载数据包...")
bundle = joblib.load(DATA_PATH)
train_x, train_y = bundle['train']
val_x, val_y = bundle['val']
test_x, test_y = bundle['test']
train_time = bundle['time_features'][0]
val_time = bundle['time_features'][1]
test_time = bundle['time_features'][2]
scaler_y = bundle['scaler_y']
scaler_x = bundle['scaler_x']

print(f"   训练集形状: {train_x.shape}, {train_y.shape}")
print(f"   验证集形状: {val_x.shape}, {val_y.shape}")
print(f"   测试集形状: {test_x.shape}, {test_y.shape}")

# 构建 PyTorch DataLoader
train_dataset = PVSlidingWindowDataset(train_x, train_y, train_time, SEQ_LEN, SEQ_LEN//2, PRED_LEN)
val_dataset = PVSlidingWindowDataset(val_x, val_y, val_time, SEQ_LEN, SEQ_LEN//2, PRED_LEN)
test_dataset = PVSlidingWindowDataset(test_x, test_y, test_time, SEQ_LEN, SEQ_LEN//2, PRED_LEN)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

print(f"   ✅ DataLoader 构建完成")

input_dim = train_x.shape[1]
print(f"   输入特征维度: {input_dim}")

# ==========================================
# 3. 定义 GRU 模型（PyTorch）
# ==========================================
print("\n" + "=" * 80)
print("3. 训练 GRU 模型（多步预测）")
print("=" * 80)

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

gru_model = GRUModel(
    input_dim=input_dim,
    hidden_dim=[64, 32, 16],
    num_layers=3,
    pred_len=PRED_LEN,
    dropout=0.2
).to(device)

criterion = nn.MSELoss()
optimizer = optim.Adam(gru_model.parameters(), lr=LEARNING_RATE)
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS, eta_min=1e-6)

print(gru_model)

best_val_loss = float('inf')
patience_counter = 0

for epoch in range(EPOCHS):
    gru_model.train()
    train_loss = 0
    for batch in train_loader:
        seq_x, _, _, _, target_y = batch
        seq_x = seq_x.to(device)
        target_y = target_y.to(device)
        
        optimizer.zero_grad()
        output = gru_model(seq_x)
        loss = criterion(output, target_y)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(gru_model.parameters(), max_norm=1.0)
        optimizer.step()
        train_loss += loss.item()
    
    train_loss /= len(train_loader)
    
    gru_model.eval()
    val_loss = 0
    with torch.no_grad():
        for batch in val_loader:
            seq_x, _, _, _, target_y = batch
            seq_x = seq_x.to(device)
            target_y = target_y.to(device)
            output = gru_model(seq_x)
            loss = criterion(output, target_y)
            val_loss += loss.item()
    
    val_loss /= len(val_loader)
    scheduler.step()
    
    print(f"Epoch {epoch+1:02d} | Train Loss: {train_loss:.5f} | Val Loss: {val_loss:.5f} | LR: {optimizer.param_groups[0]['lr']:.6f}")
    
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        torch.save(gru_model.state_dict(), os.path.join(MODEL_DIR, "pv_gru_multistep.pth"))
        patience_counter = 0
    else:
        patience_counter += 1
        if patience_counter >= PATIENCE:
            print(f"   ✅ 早停触发，训练结束")
            break

gru_model.load_state_dict(torch.load(os.path.join(MODEL_DIR, "pv_gru_multistep.pth")))

gru_model.eval()
y_pred_gru_list = []
y_test_list = []
with torch.no_grad():
    for batch in test_loader:
        seq_x, _, _, _, target_y = batch
        seq_x = seq_x.to(device)
        output = gru_model(seq_x)
        y_pred_gru_list.append(output.cpu().numpy())
        y_test_list.append(target_y.numpy())

y_pred_gru = np.concatenate(y_pred_gru_list, axis=0)
y_test_gru = np.concatenate(y_test_list, axis=0)

gru_rmse = np.sqrt(mean_squared_error(y_test_gru.flatten(), y_pred_gru.flatten()))
gru_mae = mean_absolute_error(y_test_gru.flatten(), y_pred_gru.flatten())
gru_r2 = r2_score(y_test_gru.flatten(), y_pred_gru.flatten())

print(f"\n✅ GRU 测试集指标:")
print(f"   RMSE: {gru_rmse:.4f}, MAE: {gru_mae:.4f}, R²: {gru_r2:.4f}")
print(f"   模型已保存至: {os.path.join(MODEL_DIR, 'pv_gru_multistep.pth')}")

# ==========================================
# 4. 定义 Transformer 模型（PyTorch）
# ==========================================
print("\n" + "=" * 80)
print("4. 训练 Transformer 模型（多步预测）")
print("=" * 80)
def xyd():
    print("希望你这次不要搞什么离谱操作了")
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

tf_model = TransformerModel(
    input_dim=input_dim,
    d_model=64,
    num_heads=4,
    num_layers=2,
    dff=128,
    pred_len=PRED_LEN,
    dropout=0.1
).to(device)

criterion = nn.MSELoss()
tf_optimizer = optim.Adam(tf_model.parameters(), lr=1e-4)
tf_scheduler = optim.lr_scheduler.CosineAnnealingLR(tf_optimizer, T_max=EPOCHS, eta_min=1e-6)

print(tf_model)

best_val_loss = float('inf')
patience_counter = 0

for epoch in range(EPOCHS):
    tf_model.train()
    train_loss = 0
    for batch in train_loader:
        seq_x, _, _, _, target_y = batch
        seq_x = seq_x.to(device)
        target_y = target_y.to(device)
        
        tf_optimizer.zero_grad()
        output = tf_model(seq_x)
        loss = criterion(output, target_y)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(tf_model.parameters(), max_norm=1.0)
        tf_optimizer.step()
        train_loss += loss.item()
    
    train_loss /= len(train_loader)
    
    tf_model.eval()
    val_loss = 0
    with torch.no_grad():
        for batch in val_loader:
            seq_x, _, _, _, target_y = batch
            seq_x = seq_x.to(device)
            target_y = target_y.to(device)
            output = tf_model(seq_x)
            loss = criterion(output, target_y)
            val_loss += loss.item()
    
    val_loss /= len(val_loader)
    tf_scheduler.step()
    
    print(f"Epoch {epoch+1:02d} | Train Loss: {train_loss:.5f} | Val Loss: {val_loss:.5f} | LR: {tf_optimizer.param_groups[0]['lr']:.6f}")
    
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        torch.save(tf_model.state_dict(), os.path.join(MODEL_DIR, "pv_transformer_multistep.pth"))
        patience_counter = 0
    else:
        patience_counter += 1
        if patience_counter >= PATIENCE:
            print(f"   ✅ 早停触发，训练结束")
            break

tf_model.load_state_dict(torch.load(os.path.join(MODEL_DIR, "pv_transformer_multistep.pth")))

tf_model.eval()
y_pred_tf_list = []
with torch.no_grad():
    for batch in test_loader:
        seq_x, _, _, _, target_y = batch
        seq_x = seq_x.to(device)
        output = tf_model(seq_x)
        y_pred_tf_list.append(output.cpu().numpy())

y_pred_tf = np.concatenate(y_pred_tf_list, axis=0)

tf_rmse = np.sqrt(mean_squared_error(y_test_gru.flatten(), y_pred_tf.flatten()))
tf_mae = mean_absolute_error(y_test_gru.flatten(), y_pred_tf.flatten())
tf_r2 = r2_score(y_test_gru.flatten(), y_pred_tf.flatten())

print(f"\n✅ Transformer 测试集指标:")
print(f"   RMSE: {tf_rmse:.4f}, MAE: {tf_mae:.4f}, R²: {tf_r2:.4f}")
print(f"   模型已保存至: {os.path.join(MODEL_DIR, 'pv_transformer_multistep.pth')}")

# ==========================================
# 5. 训练随机森林模型
# ==========================================
print("\n" + "=" * 80)
print("5. 训练随机森林模型（多步预测）")
print("=" * 80)

X_train_rf = train_x.reshape(train_x.shape[0], -1)
X_val_rf = val_x.reshape(val_x.shape[0], -1)
X_test_rf = test_x.reshape(test_x.shape[0], -1)

rf_model = RandomForestRegressor(
    n_estimators=100,
    max_depth=20,
    min_samples_split=10,
    min_samples_leaf=5,
    random_state=42,
    n_jobs=-1
)

rf_model.fit(X_train_rf, train_y)
print(f"   ✅ 随机森林训练完成")

y_pred_rf = rf_model.predict(X_test_rf)
rf_rmse = np.sqrt(mean_squared_error(y_test_gru.flatten(), y_pred_rf.flatten()))
rf_mae = mean_absolute_error(y_test_gru.flatten(), y_pred_rf.flatten())
rf_r2 = r2_score(y_test_gru.flatten(), y_pred_rf.flatten())

print(f"\n✅ 随机森林测试集指标:")
print(f"   RMSE: {rf_rmse:.4f}, MAE: {rf_mae:.4f}, R²: {rf_r2:.4f}")

with open(os.path.join(MODEL_DIR, "pv_rf_multistep.pkl"), "wb") as f:
    pickle.dump(rf_model, f)
print(f"   模型已保存至: {os.path.join(MODEL_DIR, 'pv_rf_multistep.pkl')}")

# ==========================================
# 6. 保存测试集和 Scaler
# ==========================================
print("\n" + "=" * 80)
print("6. 保存测试集和 Scaler...")
print("=" * 80)

# 保存滑动窗口后的数据（3D格式）
X_test_list, y_test_list = [], []
for batch in test_loader:
    seq_x, _, _, _, target_y = batch
    X_test_list.append(seq_x.numpy())
    y_test_list.append(target_y.numpy())

X_test_3d = np.concatenate(X_test_list, axis=0)
y_test_3d = np.concatenate(y_test_list, axis=0)

np.save(os.path.join(MODEL_DIR, "pv_X_test_multistep.npy"), X_test_3d)
np.save(os.path.join(MODEL_DIR, "pv_y_test_multistep.npy"), y_test_3d)

print(f"   ✅ 测试集已保存 (3D: {X_test_3d.shape}, {y_test_3d.shape})")

with open(os.path.join(MODEL_DIR, "pv_scaler_multistep.pkl"), "wb") as f:
    pickle.dump(scaler_y, f)

print(f"   ✅ Scaler 已保存")

# ==========================================
# 7. 汇总对比
# ==========================================
print("\n" + "=" * 80)
print("📊 基准模型对比汇总（多步预测 192→24）")
print("=" * 80)
print(f"{'模型':<15} | {'RMSE':<10} | {'MAE':<10} | {'R²':<10}")
print("-" * 55)
print(f"{'GRU':<15} | {gru_rmse:<10.4f} | {gru_mae:<10.4f} | {gru_r2:<10.4f}")
print(f"{'Transformer':<15} | {tf_rmse:<10.4f} | {tf_mae:<10.4f} | {tf_r2:<10.4f}")
print(f"{'Random Forest':<15} | {rf_rmse:<10.4f} | {rf_mae:<10.4f} | {rf_r2:<10.4f}")
print("=" * 80)
print("\n✅ 所有基准模型训练完成！")
print(f"   模型文件已保存至: {MODEL_DIR}")
print("   下一步：修改 pv_plot_models.py 以加载多步预测模型")
xyd()