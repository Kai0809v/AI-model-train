"""
单独训练随机森林基准模型（用于公平对比）
- 输入：model_ready_data_no_bp.pkl (无 Boruta/PCA)
- 处理：192步滑动窗口 -> 展平为 2D 向量
- 输出：pv_rf_multistep.pkl
"""
import os
import pickle
import joblib
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from torch.utils.data import DataLoader
from data_loader import PVSlidingWindowDataset

# ==========================================
# 1. 配置路径与参数
# ==========================================
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(BASE_DIR, "processed_data", "model_ready_data_no_bp.pkl")
MODEL_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "contrast_results")
os.makedirs(MODEL_DIR, exist_ok=True)

SEQ_LEN = 192
PRED_LEN = 24
BATCH_SIZE = 64 # RF 不需要太大的 batch，但 DataLoader 加载快一点好

print("=" * 80)
print("开始训练随机森林基准模型（多步预测 192→24）")
print("=" * 80)

# ==========================================
# 2. 加载数据并构建滑动窗口
# ==========================================
print("\n1. 加载数据并构建滑动窗口...")
bundle = joblib.load(DATA_PATH)
train_x, train_y = bundle['train']
val_x, val_y = bundle['val']
test_x, test_y = bundle['test']
train_time = bundle['time_features'][0]
val_time = bundle['time_features'][1]
test_time = bundle['time_features'][2]

# 使用统一的 Dataset 类构建窗口
train_dataset = PVSlidingWindowDataset(train_x, train_y, train_time, SEQ_LEN, SEQ_LEN//2, PRED_LEN)
test_dataset = PVSlidingWindowDataset(test_x, test_y, test_time, SEQ_LEN, SEQ_LEN//2, PRED_LEN)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

# 从 DataLoader 中提取出完整的 3D 数组
def extract_from_loader(loader):
    X_list, y_list = [], []
    for seq_x, _, _, _, target_y in loader:
        X_list.append(seq_x.numpy())
        y_list.append(target_y.numpy())
    return np.concatenate(X_list, axis=0), np.concatenate(y_list, axis=0)

X_train_3d, y_train = extract_from_loader(train_loader)
X_test_3d, y_test = extract_from_loader(test_loader)

print(f"   训练集形状 (3D): {X_train_3d.shape}")
print(f"   测试集形状 (3D): {X_test_3d.shape}")

# ==========================================
# 3. 数据展平 (关键步骤：实现公平对比)
# ==========================================
print("\n2. 展平数据以适配随机森林...")
# 将 (N, 192, D) 展平为 (N, 192*D)
N_train, seq_len, dim = X_train_3d.shape
N_test, _, _ = X_test_3d.shape

X_train_rf = X_train_3d.reshape(N_train, -1)
X_test_rf = X_test_3d.reshape(N_test, -1)

print(f"   训练集形状 (2D): {X_train_rf.shape}")
print(f"   测试集形状 (2D): {X_test_rf.shape}")
print(f"   ✅ 每个样本现在包含 {seq_len * dim} 个特征（利用了完整历史信息）")

# ==========================================
# 4. 训练随机森林
# ==========================================
print("\n3. 训练随机森林模型...")
rf_model = RandomForestRegressor(
    n_estimators=100,
    max_depth=20,
    min_samples_split=10,
    min_samples_leaf=5,
    random_state=42,
    n_jobs=-1, # 使用所有 CPU 核心
    verbose=1
)

rf_model.fit(X_train_rf, y_train)
print("   ✅ 训练完成")

# ==========================================
# 5. 评估与保存
# ==========================================
print("\n4. 评估模型性能...")
y_pred = rf_model.predict(X_test_rf)

rmse = np.sqrt(mean_squared_error(y_test.flatten(), y_pred.flatten()))
mae = mean_absolute_error(y_test.flatten(), y_pred.flatten())
r2 = r2_score(y_test.flatten(), y_pred.flatten())

print(f"   测试集指标:")
print(f"   RMSE: {rmse:.4f}")
print(f"   MAE:  {mae:.4f}")
print(f"   R²:   {r2:.4f}")

save_path = os.path.join(MODEL_DIR, "pv_rf_multistep.pkl")
with open(save_path, "wb") as f:
    pickle.dump(rf_model, f)

print(f"\n✅ 模型已保存至: {save_path}")
print("   现在可以运行 pv_plot_models.py 进行公平对比了！")
