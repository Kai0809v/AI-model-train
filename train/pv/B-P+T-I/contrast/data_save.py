import os
import pickle
import joblib
import numpy as np
from torch.utils.data import DataLoader

from data_loader import PVSlidingWindowDataset

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# 基准模型数据源 (无 Boruta/PCA)
DATA_PATH_NO_BP = os.path.join(BASE_DIR, "processed_data", "model_ready_data_no_bp.pkl")
# TCN-Informer 数据源 (有 Boruta/PCA)
DATA_PATH_BP = os.path.join(BASE_DIR, "processed_data", "model_ready_data.pkl")
MODEL_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "contrast_results")

SEQ_LEN = 192
PRED_LEN = 24
BATCH_SIZE = 32

print("加载基准模型数据 (No BP)...")
bundle_no_bp = joblib.load(DATA_PATH_NO_BP)
test_x_no_bp, test_y_no_bp = bundle_no_bp['test']
test_time_no_bp = bundle_no_bp['time_features'][2]
scaler_y_no_bp = bundle_no_bp.get('scaler_y', bundle_no_bp.get('scaler_x'))

print("加载 TCN-Informer 数据 (With BP)...")
bundle_bp = joblib.load(DATA_PATH_BP)
test_x_bp, test_y_bp = bundle_bp['test']
test_time_bp = bundle_bp['time_features'][2]
scaler_y_bp = bundle_bp.get('scaler_y', bundle_bp.get('scaler_x'))

# ==================== 1. 为基准模型 (GRU, Transformer, RF) 保存数据 ====================
print("正在构建基准模型滑动窗口数据集...")
test_dataset_no_bp = PVSlidingWindowDataset(test_x_no_bp, test_y_no_bp, test_time_no_bp, SEQ_LEN, SEQ_LEN//2, PRED_LEN)
test_loader_no_bp = DataLoader(test_dataset_no_bp, batch_size=BATCH_SIZE, shuffle=False)

X_test_list, y_test_list = [], []
for batch in test_loader_no_bp:
    seq_x, _, _, _, target_y = batch
    X_test_list.append(seq_x.numpy())
    y_test_list.append(target_y.numpy())

X_test_3d = np.concatenate(X_test_list, axis=0)
y_test_3d = np.concatenate(y_test_list, axis=0)

np.save(os.path.join(MODEL_DIR, "pv_X_test_multistep.npy"), X_test_3d)
np.save(os.path.join(MODEL_DIR, "pv_y_test_multistep.npy"), y_test_3d)

# 为 RF 准备展平数据
X_test_rf_2d = X_test_3d.reshape(X_test_3d.shape[0], -1)
np.save(os.path.join(MODEL_DIR, "pv_X_test_rf_2d.npy"), X_test_rf_2d)

with open(os.path.join(MODEL_DIR, "pv_scaler_multistep.pkl"), "wb") as f:
    pickle.dump(scaler_y_no_bp, f)

print(f"✅ 基准模型数据已保存: X={X_test_3d.shape}, RF_Flat={X_test_rf_2d.shape}")

# ==================== 2. 为 TCN-Informer 保存数据 ====================
print("正在构建 TCN-Informer 滑动窗口数据集...")
test_dataset_bp = PVSlidingWindowDataset(test_x_bp, test_y_bp, test_time_bp, SEQ_LEN, SEQ_LEN//2, PRED_LEN)
test_loader_bp = DataLoader(test_dataset_bp, batch_size=BATCH_SIZE, shuffle=False)

X_tcn_list, y_tcn_list = [], []
for batch in test_loader_bp:
    seq_x, _, _, _, target_y = batch
    X_tcn_list.append(seq_x.numpy())
    y_tcn_list.append(target_y.numpy())

X_test_tcn = np.concatenate(X_tcn_list, axis=0)
y_test_tcn = np.concatenate(y_tcn_list, axis=0)

np.save(os.path.join(MODEL_DIR, "pv_X_test_tcn_informer.npy"), X_test_tcn)
np.save(os.path.join(MODEL_DIR, "pv_y_test_tcn_informer.npy"), y_test_tcn)

print(f"✅ TCN-Informer 数据已保存: X={X_test_tcn.shape}")
