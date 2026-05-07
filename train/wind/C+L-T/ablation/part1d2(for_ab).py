# ============================================
# 这一版修改相比上一次，对是否进行CEEMDAN分解分为两批数据，
# 跟之前一样进行了分解的数据也用了不一样的变量名
# 作为ablation_experiment_d2.py的数据处理脚本
# ============================================
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from PyEMD import CEEMDAN
from joblib import dump


# ============================================
# 1 读取数据
# ============================================
def load_data(file_path, data_size=None):
    data = pd.read_csv(file_path)
    if data_size is not None:
        data = data.iloc[:data_size]
    print("数据尺寸:", data.shape)
    return data


# ============================================
# 2 CEEMDAN分解
# ============================================
def ceemdan_decompose(power_series):
    print("开始CEEMDAN分解...")
    ceemdan = CEEMDAN(trials=100, epsilon=0.005)
    imfs = ceemdan(power_series)
    imfs_T = imfs.T
    print("IMF数量:", imfs_T.shape[1])
    imf_df = pd.DataFrame(imfs_T, columns=[f'imf{i}' for i in range(imfs_T.shape[1])])
    return imf_df, imfs


# ============================================
# 3 构建特征矩阵 (新增控制变量 use_ceemdan)
# ============================================
def build_feature_dataframe(original_data, imf_df, use_ceemdan=True):
    feature_cols = [
        '测风塔10m风速(m/s)', '测风塔30m风速(m/s)', '测风塔50m风速(m/s)', '测风塔70m风速(m/s)', '轮毂高度风速(m/s)',
        '测风塔10m风向(°)', '测风塔30m风向(°)', '测风塔50m风向(°)', '测风塔70m风向(°)', '轮毂高度风向(°)',
        '温度(°)', '气压(hPa)', '湿度(%)', '实际发电功率（mw）'
    ]

    if use_ceemdan and imf_df is not None:
        data = pd.concat([original_data, imf_df], axis=1)
        imf_cols = [col for col in data.columns if "imf" in col]
        feature_cols = feature_cols + imf_cols
    else:
        data = original_data.copy()

    X = data[feature_cols]
    y = data[['实际发电功率（mw）']]
    return X, y


# ============================================
# 4/5/6 时间序列划分、标准化、构造滑动窗口
# ============================================
def process_pipeline(X, y, window_size, split_rate=0.9, prefix=""):
    # 划分
    n = len(X)
    split_index = int(n * split_rate)
    X_train, X_test = X.iloc[:split_index], X.iloc[split_index:]
    y_train, y_test = y.iloc[:split_index], y.iloc[split_index:]

    # 标准化
    scaler_x = StandardScaler()
    scaler_y = StandardScaler()
    X_train_scaled = scaler_x.fit_transform(X_train)
    X_test_scaled = scaler_x.transform(X_test)
    y_train_scaled = scaler_y.fit_transform(y_train)
    y_test_scaled = scaler_y.transform(y_test)

    # 仅保存一次 y 的 scaler 即可（有无 CEEMDAN 的 y 是一样的）
    if prefix == "full_":
        dump(scaler_y, "scaler_y")

    # 滑动窗口
    X_train_win, y_train_win, X_test_win, y_test_win = [], [], [], []
    X_all = np.vstack((X_train_scaled, X_test_scaled))
    y_all = np.vstack((y_train_scaled, y_test_scaled))

    for i in range(window_size, len(X_all)):
        X_window = X_all[i - window_size:i]
        y_target = y_all[i]
        if i < split_index:
            X_train_win.append(X_window)
            y_train_win.append(y_target)
        else:
            X_test_win.append(X_window)
            y_test_win.append(y_target)

    train_set = torch.tensor(np.array(X_train_win)).float()
    train_label = torch.tensor(np.array(y_train_win)).float()
    test_set = torch.tensor(np.array(X_test_win)).float()
    test_label = torch.tensor(np.array(y_test_win)).float()

    # 保存数据
    dump(train_set, f"{prefix}train_set")
    dump(train_label, f"{prefix}train_label")
    dump(test_set, f"{prefix}test_set")
    dump(test_label, f"{prefix}test_label")

    print(f"[{prefix[:-1]}] 训练集: {train_set.shape}, 测试集: {test_set.shape}")


def plot_ceemdan(power_series, imfs):
    num_imfs = imfs.shape[0]
    plt.figure(figsize=(12, 2 * (num_imfs + 1)))
    plt.subplot(num_imfs + 1, 1, 1)
    plt.plot(power_series, 'r')
    plt.title("Original Wind Power Series")
    for i in range(num_imfs):
        plt.subplot(num_imfs + 1, 1, i + 2)
        plt.plot(imfs[i], 'g')
        plt.title(f"IMF {i + 1}")
    plt.tight_layout()
    plt.savefig('IMFsFenjie.png', bbox_inches='tight', dpi=300)
    plt.show()


# ============================================
# 主程序
# ============================================
if __name__ == "__main__":
    DATA_PATH = "wind_data.csv"
    DATA_SIZE = 6000
    WINDOW_SIZE = 24
    SPLIT_RATE = 0.9

    data = load_data(DATA_PATH, DATA_SIZE)
    power_series = data['实际发电功率（mw）'].values

    # 获取分解分量
    imf_df, imfs = ceemdan_decompose(power_series)
    plot_ceemdan(power_series, imfs)

    # ================= 分别处理并保存两套数据 =================
    print("\n--- 生成包含 CEEMDAN 的完整数据 ---")
    X_full, y_full = build_feature_dataframe(data, imf_df, use_ceemdan=True)
    process_pipeline(X_full, y_full, WINDOW_SIZE, SPLIT_RATE, prefix="full_")

    print("\n--- 生成不含 CEEMDAN 的对照组数据 ---")
    X_no_ceemdan, y_no_ceemdan = build_feature_dataframe(data, imf_df, use_ceemdan=False)
    process_pipeline(X_no_ceemdan, y_no_ceemdan, WINDOW_SIZE, SPLIT_RATE, prefix="noceem_")
    """
    数据尺寸: (6000, 15)
开始CEEMDAN分解...
IMF数量: 11

--- 生成包含 CEEMDAN 的完整数据 ---
[full] 训练集: torch.Size([5376, 24, 25]), 测试集: torch.Size([600, 24, 25])

--- 生成不含 CEEMDAN 的对照组数据 ---
[noceem] 训练集: torch.Size([5376, 24, 14]), 测试集: torch.Size([600, 24, 14])
    """