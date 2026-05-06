# ============================================
# CEEMDAN + LightGBM + Transformer 消融实验
# 这一版比上次多了一个w/o CEEMDAN
# w/o ，意即without，我看别的论文这么写的
# 搭配part1d2.py使用
# ============================================

import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import lightgbm as lgb
import matplotlib.pyplot as plt
import seaborn as sns
import math
import os

from joblib import load
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# ============================================
# 0 全局设置
# ============================================
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['figure.dpi'] = 300
os.makedirs("ablation_results/plots", exist_ok=True)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ============================================
# 1 加载两套数据
# ============================================
print("加载数据...")
# 包含 CEEMDAN 的数据
train_set_full = load("full_train_set")
test_set_full = load("full_test_set")
train_label = load("full_train_label")  # 无论有无CEEMDAN，label都是一样的
test_label = load("full_test_label")

# 不含 CEEMDAN 的数据
train_set_noceem = load("noceem_train_set")
test_set_noceem = load("noceem_test_set")

scaler_y = load("../../../../scaler_y")


# ============================================
# 2 定义模型
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
        return x + self.pe[:, :x.size(1), :]


class FlexibleTransformerModel(nn.Module):
    def __init__(self, input_dim, use_pe=True):
        super().__init__()
        self.use_pe = use_pe
        self.embedding = nn.Linear(input_dim, 128)
        if self.use_pe:
            self.pos_encoder = PositionalEncoding(d_model=128)
        encoder_layer = nn.TransformerEncoderLayer(d_model=128, nhead=8, dim_feedforward=256, batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=3)
        self.fc = nn.Sequential(nn.Linear(128, 64), nn.ReLU(), nn.Linear(64, 1))

    def forward(self, x):
        x = self.embedding(x)
        if self.use_pe:
            x = self.pos_encoder(x)
        x = self.transformer(x)
        x = x[:, -1, :]
        return self.fc(x)


# ============================================
# 3 训练与评估封装函数 (动态加载特征与 LGB 筛选)
# ============================================
def train_and_evaluate(model_name, use_ceemdan, use_lgb, use_pe, epochs=30, batch_size=64):
    print(f"\n{'=' * 50}")
    print(f"当前模型：{model_name} | CEEMDAN:{use_ceemdan} | LGB:{use_lgb} | PE:{use_pe}")

    # 1. 根据 CEEMDAN 开关选择基础数据源
    base_train_x = train_set_full if use_ceemdan else train_set_noceem
    base_test_x = test_set_full if use_ceemdan else test_set_noceem
    feature_dim = base_train_x.shape[2]

    # 2. 根据 LightGBM 开关动态筛选特征
    if use_lgb:
        print(f"执行 LightGBM 特征筛选 (原始维度：{feature_dim})...")
        train_mean = np.mean(base_train_x.numpy(), axis=1)
        train_std = np.std(base_train_x.numpy(), axis=1)
        train_max = np.max(base_train_x.numpy(), axis=1)
        train_features_lgb = np.concatenate([train_mean, train_std, train_max], axis=1)

        lgb_model = lgb.LGBMRegressor(n_estimators=300, learning_rate=0.05, max_depth=6)
        lgb_model.fit(train_features_lgb, train_label.numpy())

        importance = lgb_model.feature_importances_
        importance_per_var = [
            importance[i] + importance[i + feature_dim] + importance[i + 2 * feature_dim]
            for i in range(feature_dim)
        ]

        top_k = int(feature_dim * 0.7)
        selected_features = np.argsort(importance_per_var)[-top_k:]

        final_train_x = base_train_x[:, :, selected_features]
        final_test_x = base_test_x[:, :, selected_features]
        print(f"保留特征维度：{len(selected_features)}")
    else:
        final_train_x = base_train_x
        final_test_x = base_test_x

    input_dim = final_train_x.shape[2]

    # 3. 数据加载与模型初始化
    train_loader = DataLoader(TensorDataset(final_train_x, train_label), batch_size=batch_size, shuffle=True)
    eval_train_loader = DataLoader(TensorDataset(final_train_x, train_label), batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(TensorDataset(final_test_x, test_label), batch_size=batch_size, shuffle=False)

    model = FlexibleTransformerModel(input_dim=input_dim, use_pe=use_pe).to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    # 4. 训练
    for epoch in range(epochs):
        model.train()
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            loss = criterion(model(x), y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    # 5. 推理与反归一化
    def get_predictions(loader):
        model.eval()
        preds, trues = [], []
        with torch.no_grad():
            for x, y in loader:
                preds.append(model(x.to(device)).cpu().numpy())
                trues.append(y.numpy())
        return scaler_y.inverse_transform(np.vstack(trues)), scaler_y.inverse_transform(np.vstack(preds))

    true_train, pred_train = get_predictions(eval_train_loader)
    true_test, pred_test = get_predictions(test_loader)

    metrics = {
        "Model": model_name,
        "Train_RMSE": np.sqrt(mean_squared_error(true_train, pred_train)),
        "Train_MAE": mean_absolute_error(true_train, pred_train),
        "Train_R2": r2_score(true_train, pred_train),
        "Test_RMSE": np.sqrt(mean_squared_error(true_test, pred_test)),
        "Test_MAE": mean_absolute_error(true_test, pred_test),
        "Test_R2": r2_score(true_test, pred_test)
    }
    return metrics, true_train, pred_train, true_test, pred_test


# ============================================
# 4 循环执行消融实验
# ============================================
# 添加了 w/o CEEMDAN 实验配置
ablation_configs = {
    "CEEMDAN-LightGBM-Transformer (Full)": {"use_ceemdan": True, "use_lgb": True, "use_pe": True},
    "LightGBM-Transformer (w/o CEEMDAN)": {"use_ceemdan": False, "use_lgb": True, "use_pe": True},
    "CEEMDAN-Transformer (w/o LightGBM)": {"use_ceemdan": True, "use_lgb": False, "use_pe": True},
    "CEEMDAN-LightGBM (w/o PE)": {"use_ceemdan": True, "use_lgb": True, "use_pe": False},
    "Baseline (w/o All)": {"use_ceemdan": False, "use_lgb": False, "use_pe": False}
}

all_metrics = []
predictions_dict = {}
train_predictions_dict = {}

for name, config in ablation_configs.items():
    metrics, true_train, pred_train, true_test, pred_test = train_and_evaluate(
        model_name=name,
        use_ceemdan=config["use_ceemdan"],
        use_lgb=config["use_lgb"],
        use_pe=config["use_pe"],
        epochs=30
    )
    all_metrics.append(metrics)
    predictions_dict[name] = pred_test
    train_predictions_dict[name] = pred_train

predictions_dict["True_Value"] = true_test
train_predictions_dict["True_Value"] = true_train

# 保存指标
df_metrics = pd.DataFrame(all_metrics)
df_metrics.to_csv("ablation_results/ablation_metrics_summary.csv", index=False)
print("\n[√] 评估指标已保存")
print(df_metrics.to_string(index=False))

# ============================================
# 5 绘图代码 - 新增训练集和测试集拟合曲线
# ============================================
plot_points = 200
colors = ['#d73027', '#fc8d59', '#fee090', '#91bfdb', '#4575b4']

# 5.1 原有预测曲线对比图（测试集）
plt.figure(figsize=(14, 6))
plt.plot(predictions_dict["True_Value"][:plot_points], label="True", color='black', linewidth=2, linestyle='--')

for idx, name in enumerate(ablation_configs.keys()):
    plt.plot(predictions_dict[name][:plot_points], label=name, color=colors[idx], alpha=0.8)

plt.title("不同消融模型预测结果对比\nAblation Models Prediction Comparison", fontsize=16)
plt.legend(loc="upper right")
plt.tight_layout()
plt.savefig("ablation_results/plots/Prediction_Curves.pdf")
plt.close()

# 5.2 新增：训练集拟合曲线（每个模型单独绘制）
print("\n[7] 正在生成并保存各模型训练集拟合曲线...")
for idx, name in enumerate(ablation_configs.keys()):
    plt.figure(figsize=(12, 6))
    plt.plot(train_predictions_dict["True_Value"][:plot_points], label="True Value", color='black', linewidth=2, linestyle='--')
    plt.plot(train_predictions_dict[name][:plot_points], label=f"{name} Prediction", color=colors[idx], alpha=0.8)
    
    plt.title(f"{name} 训练集拟合曲线\nTraining Set Fitting Curve", fontsize=14)
    plt.xlabel("Sample Index", fontsize=12)
    plt.ylabel("Value", fontsize=12)
    plt.legend(loc="upper right")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    safe_name = name.replace("/", "_").replace(" ", "_").replace("(", "").replace(")", "")
    plt.savefig(f"ablation_results/plots/Train_Fitting_{safe_name}.pdf")
    plt.savefig(f"ablation_results/plots/Train_Fitting_{safe_name}.png", dpi=300)
    plt.close()

# 5.3 新增：测试集拟合曲线（每个模型单独绘制）
print("\n[8] 正在生成并保存各模型测试集拟合曲线...")
for idx, name in enumerate(ablation_configs.keys()):
    plt.figure(figsize=(12, 6))
    plt.plot(predictions_dict["True_Value"][:plot_points], label="True Value", color='black', linewidth=2, linestyle='--')
    plt.plot(predictions_dict[name][:plot_points], label=f"{name} Prediction", color=colors[idx], alpha=0.8)
    
    plt.title(f"{name} 测试集拟合曲线\nTest Set Fitting Curve", fontsize=14)
    plt.xlabel("Sample Index", fontsize=12)
    plt.ylabel("Value", fontsize=12)
    plt.legend(loc="upper right")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    safe_name = name.replace("/", "_").replace(" ", "_").replace("(", "").replace(")", "")
    plt.savefig(f"ablation_results/plots/Test_Fitting_{safe_name}.pdf")
    plt.savefig(f"ablation_results/plots/Test_Fitting_{safe_name}.png", dpi=300)
    plt.close()

# 5.4 新增：综合训练集拟合对比图（所有模型在一张图上）
print("\n[9] 正在生成综合训练集拟合对比图...")
plt.figure(figsize=(14, 7))
for idx, name in enumerate(ablation_configs.keys()):
    plt.plot(train_predictions_dict[name][:plot_points], label=name, color=colors[idx], alpha=0.7)

plt.plot(train_predictions_dict["True_Value"][:plot_points], label="True Value", color='black', linewidth=2, linestyle='--')
plt.title("各消融模型训练集拟合对比\nComprehensive Training Set Fitting Comparison", fontsize=14)
plt.xlabel("Sample Index", fontsize=12)
plt.ylabel("Value", fontsize=12)
plt.legend(loc="upper right")
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("ablation_results/plots/Train_Test_Fitting_Comprehensive.pdf")
plt.savefig("ablation_results/plots/Train_Test_Fitting_Comprehensive.png", dpi=300)
plt.close()

# 5.5 新增：综合测试集拟合对比图（所有模型在一张图上）
print("\n[10] 正在生成综合测试集拟合对比图...")
plt.figure(figsize=(14, 7))
for idx, name in enumerate(ablation_configs.keys()):
    plt.plot(predictions_dict[name][:plot_points], label=name, color=colors[idx], alpha=0.7)

plt.plot(predictions_dict["True_Value"][:plot_points], label="True Value", color='black', linewidth=2, linestyle='--')
plt.title("各消融模型测试集拟合对比\nComprehensive Test Set Fitting Comparison", fontsize=14)
plt.xlabel("Sample Index", fontsize=12)
plt.ylabel("Value", fontsize=12)
plt.legend(loc="upper right")
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("ablation_results/plots/Test_Fitting_Comprehensive.pdf")
plt.savefig("ablation_results/plots/Test_Fitting_Comprehensive.png", dpi=300)
plt.close()

# 5.6 新增：预测值 vs 实际值散点图（每个模型单独绘制）
print("\n[11] 正在生成并保存各模型预测值 vs 实际值散点图...")
for idx, name in enumerate(ablation_configs.keys()):
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    # 训练集散点图
    ax1 = axes[0]
    true_train_vals = train_predictions_dict["True_Value"]
    pred_train_vals = train_predictions_dict[name]
    ax1.scatter(true_train_vals, pred_train_vals, alpha=0.5, s=20, color=colors[idx])
    
    # 绘制 y=x 参考线
    min_val = min(true_train_vals.min(), pred_train_vals.min())
    max_val = max(true_train_vals.max(), pred_train_vals.max())
    ax1.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label='Perfect Fit')
    
    ax1.set_xlabel('Actual Value', fontsize=12)
    ax1.set_ylabel('Predicted Value', fontsize=12)
    ax1.set_title(f'{name} - 训练集散点图\nTraining Set Scatter Plot', fontsize=14)
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 测试集散点图
    ax2 = axes[1]
    true_test_vals = predictions_dict["True_Value"]
    pred_test_vals = predictions_dict[name]
    ax2.scatter(true_test_vals, pred_test_vals, alpha=0.5, s=20, color=colors[idx])
    
    min_val = min(true_test_vals.min(), pred_test_vals.min())
    max_val = max(true_test_vals.max(), pred_test_vals.max())
    ax2.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label='Perfect Fit')
    
    ax2.set_xlabel('Actual Value', fontsize=12)
    ax2.set_ylabel('Predicted Value', fontsize=12)
    ax2.set_title(f'{name} - 测试集散点图\nTest Set Scatter Plot', fontsize=14)
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    safe_name = name.replace("/", "_").replace(" ", "_").replace("(", "").replace(")", "")
    plt.savefig(f"ablation_results/plots/Scatter_Plot_{safe_name}.pdf")
    plt.savefig(f"ablation_results/plots/Scatter_Plot_{safe_name}.png", dpi=300)
    plt.close()

# 5.7 新增：综合预测值 vs 实际值散点对比图（所有模型在一张图上）
print("\n[12] 正在生成综合预测值 vs 实际值散点对比图...")
fig, axes = plt.subplots(1, 2, figsize=(18, 7))

# 训练集综合散点图
ax1 = axes[0]
true_train_vals = train_predictions_dict["True_Value"]
for idx, name in enumerate(ablation_configs.keys()):
    pred_train_vals = train_predictions_dict[name]
    ax1.scatter(true_train_vals, pred_train_vals, alpha=0.4, s=15, 
                color=colors[idx], label=name, edgecolors='w', linewidth=0.5)

min_val = min(true_train_vals.min(), pred_train_vals.min())
max_val = max(true_train_vals.max(), pred_train_vals.max())
ax1.plot([min_val, max_val], [min_val, max_val], 'k--', linewidth=2, label='Perfect Fit')

ax1.set_xlabel('Actual Value', fontsize=12)
ax1.set_ylabel('Predicted Value', fontsize=12)
ax1.set_title('各消融模型训练集散点对比\nComprehensive Training Set Scatter Plot', fontsize=14)
ax1.legend(loc='upper left', fontsize=9)
ax1.grid(True, alpha=0.3)

# 测试集综合散点图
ax2 = axes[1]
true_test_vals = predictions_dict["True_Value"]
for idx, name in enumerate(ablation_configs.keys()):
    pred_test_vals = predictions_dict[name]
    ax2.scatter(true_test_vals, pred_test_vals, alpha=0.4, s=15, 
                color=colors[idx], label=name, edgecolors='w', linewidth=0.5)

min_val = min(true_test_vals.min(), pred_test_vals.min())
max_val = max(true_test_vals.max(), pred_test_vals.max())
ax2.plot([min_val, max_val], [min_val, max_val], 'k--', linewidth=2, label='Perfect Fit')

ax2.set_xlabel('Actual Value', fontsize=12)
ax2.set_ylabel('Predicted Value', fontsize=12)
ax2.set_title('各消融模型测试集散点对比\nComprehensive Test Set Scatter Plot', fontsize=14)
ax2.legend(loc='upper left', fontsize=9)
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig("ablation_results/plots/Scatter_Plot_Comprehensive.pdf")
plt.savefig("ablation_results/plots/Scatter_Plot_Comprehensive.png", dpi=300)
plt.close()

# 绘制残差图（动态网格大小）
fig, axes = plt.subplots(3, 2, figsize=(15, 12))
axes = axes.flatten()
print("\n[13] 正在生成并独立保存残差分布直方图...")

# 1. 预先计算全局残差范围，用于统一 X 轴
all_residuals = {}
for name in ablation_configs.keys():
    res = predictions_dict["True_Value"] - predictions_dict[name]
    all_residuals[name] = res.flatten()

global_residuals = np.concatenate(list(all_residuals.values()))
# X轴：找到绝对值最大的误差，留出 10% 的边缘空白
max_abs_error = np.max(np.abs(global_residuals)) * 1.1
x_lim_min, x_lim_max = -max_abs_error, max_abs_error

# 2. 预先计算统一的 Bins 和 Y 轴最大值
# 强制划分 50 个等宽的区间，确保所有直方图的柱子宽度一模一样
bins_edges = np.linspace(x_lim_min, x_lim_max, 51)
max_y_freq = 0

for name in ablation_configs.keys():
    res = all_residuals[name]
    # 利用 numpy 提前统计在这些区间内的最高频数
    counts, _ = np.histogram(res, bins=bins_edges)
    if np.max(counts) > max_y_freq:
        max_y_freq = np.max(counts)

# Y轴：给顶部留出 15% 的空间，防止 KDE 平滑曲线或最高柱子贴边
y_lim_max = max_y_freq * 1.15

# 3. 循环绘制并独立保存
for idx, name in enumerate(ablation_configs.keys()):
    plt.figure(figsize=(8, 6))

    res = all_residuals[name]

    # 绘制直方图和 KDE 曲线 (传入预设的 bins_edges 保证绝对对齐)
    sns.histplot(res, bins=bins_edges, kde=True, color=colors[idx], edgecolor="w", legend=False)

    # 强制统一 X 轴和 Y 轴范围
    plt.xlim(x_lim_min, x_lim_max)
    plt.ylim(0, y_lim_max)

    plt.title(f"{name} 模型残差分布 (Residuals)", fontsize=14, fontweight='bold', pad=15)
    plt.xlabel("预测误差 (Error)", fontsize=12)
    plt.ylabel("频数 (Frequency)", fontsize=12)

    # 添加辅助零线
    plt.axvline(x=0, color='black', linestyle='--', linewidth=1.5)

    plt.tight_layout()

    safe_name = name.replace("/", "_").replace(" ", "_").replace("(", "").replace(")", "")
    file_prefix = f"ablation_results/plots/Residual_{idx + 1}_{safe_name}"

    plt.savefig(f"{file_prefix}.pdf")
    plt.savefig(f"{file_prefix}.png", dpi=300)
    plt.close()

print("[√] 所有独立残差图表已保存至 ablation_results/plots/ 文件夹下。")