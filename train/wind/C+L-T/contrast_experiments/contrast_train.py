import os
import torch
import torch.nn as nn
import numpy as np
from joblib import load, dump
from torch.utils.data import DataLoader, TensorDataset
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor

# 创建对比实验输出目录
os.makedirs('../../../../contrast', exist_ok=True)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ============================================
# 1. 加载数据与特征分离
# ============================================
print("正在加载数据集...")
multi_step_datasets = load("../../../../multi_step_datasets.pkl")
data_1step = multi_step_datasets[1]

train_x_full = data_1step['train_x']
train_y_clean = data_1step['train_y']  # 干净标签

print("提取原始特征(基线使用)...")
train_x_raw = train_x_full[:, :, [5, 10, 11, 14]]


# ============================================
# 2. 定义基线 GRU 模型结构
# ============================================
class BaselineGRU(nn.Module):
    def __init__(self, input_dim=4, hidden_dim=64):
        super().__init__()
        self.gru = nn.GRU(input_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        _, h_n = self.gru(x)
        return self.fc(h_n.squeeze(0))


def train_pytorch_model(model, train_x, train_y, model_name, epochs=30):
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.MSELoss()
    dataset = TensorDataset(train_x, train_y)
    loader = DataLoader(dataset, batch_size=64, shuffle=True)

    print(f"\n开始训练 {model_name}...")
    model.train()
    for epoch in range(epochs):
        total_loss = 0
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            pred = model(x)
            loss = criterion(pred, y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch + 1}/{epochs}, Loss: {total_loss / len(loader):.6f}")

    torch.save(model.state_dict(), f"./contrast/{model_name}.pth")
    print(f"[+] 保存模型: {model_name}.pth")


# ============================================
# 3. 训练对比模型
# ============================================

# 3.1 训练 Baseline: GRU (原始特征)
gru_model = BaselineGRU(input_dim=4)
train_pytorch_model(gru_model, train_x_raw, train_y_clean, "Baseline_GRU")

# 准备机器学习模型需要的 2D 展平数据
train_x_raw_flat = train_x_raw.numpy().reshape(train_x_raw.shape[0], -1)
y_np = train_y_clean.numpy().ravel()

# 3.2 训练 Baseline: 随机森林 (原始特征)
print("\n开始训练 Baseline_RandomForest...")
rf_model = RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42, n_jobs=-1)
rf_model.fit(train_x_raw_flat, y_np)
dump(rf_model, "../../../../contrast/Baseline_RF.joblib")
print("[+] 保存模型: Baseline_RF.joblib")

# 3.3 训练 Baseline: XGBoost (原始特征)
print("\n开始训练 Baseline_XGBoost...")
xgb_model = XGBRegressor(
    n_estimators=100,
    learning_rate=0.03,
    max_depth=4,
    min_child_weight=10,
    subsample=0.7,
    colsample_bytree=0.7,
    reg_alpha=1.0,
    reg_lambda=5.0,
    gamma=1.0,
    random_state=42,
    n_jobs=-1
)
xgb_model.fit(train_x_raw_flat, y_np)
dump(xgb_model, "../../../../contrast/Baseline_XGBoost.joblib")
print("[+] 保存模型: Baseline_XGBoost.joblib")

print("\n🎉 所有基线对比模型训练完成并已保存至 ./contrast_experiments 目录！")