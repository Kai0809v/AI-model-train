import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch

from sklearn.preprocessing import StandardScaler, MinMaxScaler
from PyEMD import CEEMDAN
from joblib import dump, load


# ============================================
# 1 读取数据（支持选择数据量）
# ============================================

def load_data(file_path, data_size=None, start_index=0):
    data = pd.read_csv(file_path)

    if start_index > 0:
        data = data.iloc[start_index:]

    if data_size is not None:
        if len(data) > data_size:
            data = data.iloc[:data_size]

    print("数据尺寸:", data.shape, "\n数据起点:", start_index)

    # 检查并删除 NaN 值
    if data.isnull().values.any():
        print(f"警告：发现{data.isnull().sum().sum()}个 NaN 值，正在删除...")
        data = data.dropna()
        print("删除 NaN 后的数据尺寸:", data.shape)

    return data


# ============================================
# 2 时间序列划分（严格防泄露）- 支持三集划分
# ============================================

def time_series_split(data, split_rate=0.9, use_validation=False, val_ratio=0.1):
    """
    时间序列数据划分
    
    参数:
        data: 原始数据
        split_rate: 训练集比例（当 use_validation=False 时）
        use_validation: 是否使用验证集
        val_ratio: 验证集占总数据的比例（当 use_validation=True 时，默认10%）
    
    返回:
        如果使用验证集: (train_data, val_data, test_data, train_end_idx, val_end_idx)
        否则: (train_data, test_data, split_index)
    """
    n = len(data)
    
    if use_validation:
        # 三集划分: 训练集 : 验证集 : 测试集 = 8:1:1
        train_end_idx = int(n * (1 - val_ratio * 2))  # 80%
        val_end_idx = int(n * (1 - val_ratio))         # 90%
        
        train_data = data.iloc[:train_end_idx].copy()
        val_data = data.iloc[train_end_idx:val_end_idx].copy()
        test_data = data.iloc[val_end_idx:].copy()
        
        print("训练集尺寸:", train_data.shape)
        print("验证集尺寸:", val_data.shape)
        print("测试集尺寸:", test_data.shape)
        
        return train_data, val_data, test_data, train_end_idx, val_end_idx
    else:
        # 原有的两集划分
        split_index = int(n * split_rate)
        train_data = data.iloc[:split_index].copy()
        test_data = data.iloc[split_index:].copy()
        
        print("训练集尺寸:", train_data.shape)
        print("测试集尺寸:", test_data.shape)
        
        return train_data, test_data, split_index


# ============================================
# 3 核心创新：目标变量 Y 去噪提纯
# ============================================

def ceemdan_denoise_target(power_series, drop_k=1):
    """
    使用 CEEMDAN 对功率目标进行离线去噪。
    drop_k: 剔除的高频噪声 IMF 数量（默认剔除 IMF0 ）和 IMF1
    """
    print("开始对目标功率进行 CEEMDAN 分解提纯...")

    ceemdan = CEEMDAN(trials=100, epsilon=0.005)
    imfs = ceemdan(power_series)

    # 打印分解出的总层数
    total_imfs = imfs.shape[0]
    print(f"共分解出 {total_imfs} 个 IMF 分量（含残差）。")

    if total_imfs <= drop_k:
        print("警告：分解层数过少，跳过去噪。")
        return power_series

    # 重构信号：舍弃前 drop_k 个高频噪声，将剩余的中低频 IMF 和残差求和
    clean_series = np.sum(imfs[drop_k:], axis=0)
    return clean_series


# ============================================
# 附加分析模块：可视化不同 drop_k 的去噪效果
# ============================================
def visualize_drop_k_experiments(power_series, display_length=500):
    """
    对比测试 drop_k = 1, 2, 3 时的信号重构效果
    """
    print(f"\n正在进行 CEEMDAN 分解用于可视化测试 (取前 {display_length} 个点)...")

    # 为了快速绘图，我们只取一段连续的数据进行分解分析
    sample_series = power_series[:display_length]
    ceemdan = CEEMDAN(trials=100, epsilon=0.005)
    imfs = ceemdan(sample_series)

    total_imfs = imfs.shape[0]
    print(f"该片段共分解出 {total_imfs} 个 IMF 分量。")

    plt.figure(figsize=(15, 10))

    # 1. 绘制原始数据的基准线
    plt.subplot(4, 1, 1)
    plt.plot(sample_series, label="Raw Power (Original)", color='black', alpha=0.7)
    plt.title("Wind Power Target Denoising Analysis (Ablation Study on drop_k)")
    plt.legend(loc="upper right")
    plt.grid(True, linestyle='--', alpha=0.6)

    # 2. 循环绘制 drop_k = 1, 2, 3 的效果
    colors = ['#1f77b4', '#2ca02c', '#d62728']  # 蓝，绿，红

    for k in range(1, 4):
        if total_imfs > k:
            # 舍弃前 k 个高频分量，重构剩余分量
            clean_series = np.sum(imfs[k:], axis=0)

            plt.subplot(4, 1, k + 1)
            plt.plot(sample_series, label="Raw Power", color='gray', alpha=0.4)
            plt.plot(clean_series, label=f"Clean Power (drop_k={k})", color=colors[k - 1], linewidth=1.5)

            # 计算平滑后的方差保留率 (用于量化去噪程度)
            var_raw = np.var(sample_series)
            var_clean = np.var(clean_series)
            retained_variance = (var_clean / var_raw) * 100

            plt.title(f"Drop Top {k} IMFs (Retained Variance: {retained_variance:.1f}%)")
            plt.legend(loc="upper right")
            plt.grid(True, linestyle='--', alpha=0.6)

    plt.tight_layout()
    plt.show()


# ============================================
# 4 构建特征 (X 保持原始，Y 替换为纯净版) - 支持验证集
# ============================================

def extract_features_and_targets(train_data, test_data, val_data=None):
    weather_cols = [
        '测风塔10m风速(m/s)', '测风塔30m风速(m/s)', '测风塔50m风速(m/s)',
        '测风塔70m风速(m/s)', '轮毂高度风速(m/s)', '测风塔10m风向(°)',
        '测风塔30m风向(°)', '测风塔50m风向(°)', '测风塔70m风向(°)',
        '轮毂高度风向(°)', '温度(°)', '气压(hPa)', '湿度(%)'
    ]

    # X：原始气象特征
    X_train_raw = train_data[weather_cols].values
    X_test_raw = test_data[weather_cols].values
    
    if val_data is not None:
        X_val_raw = val_data[weather_cols].values

    # Y_raw：保存原始功率
    y_train_raw = train_data['实际发电功率（mw）'].values
    y_test_raw = test_data['实际发电功率（mw）'].values
    
    if val_data is not None:
        y_val_raw = val_data['实际发电功率（mw）'].values

    # 🚨 关键修复：使用原始训练数据拟合 Scaler，确保尺度覆盖真实波动
    from sklearn.preprocessing import MinMaxScaler
    global_scaler_y = MinMaxScaler(feature_range=(0, 1))
    global_scaler_y.fit(y_train_raw.reshape(-1, 1))
    dump(global_scaler_y, "./scaler_y")  # 提前保存，供 part2 使用
    print("[+] 已基于原始训练数据拟合 scaler_y 并保存")

    # 🚨 修复：直接将 1 维的原始功率拼接到气象特征末尾
    # 不平铺成 24 列，因为后续的滑动窗口会自动帮我们把这 1 维展开成 96 步的历史轨迹
    X_train = np.hstack([X_train_raw, y_train_raw.reshape(-1, 1)])
    X_test = np.hstack([X_test_raw, y_test_raw.reshape(-1, 1)])
    
    if val_data is not None:
        X_val = np.hstack([X_val_raw, y_val_raw.reshape(-1, 1)])

    def add_physics_features(data, data_index=None):
        feats = []

        # 1. 平均风速（5 个高度的平均）
        avg_wind = np.mean(data[:, :5], axis=1, keepdims=True)
        feats.append(avg_wind)

        # 2. 风速立方（捕捉 P ∝ v³关系）
        wind_cubed = data[:, :5] ** 3
        feats.append(wind_cubed)

        # 3. 风速标准差（表征风切变）
        wind_std = np.std(data[:, :5], axis=1, keepdims=True)
        feats.append(wind_std)

        # 4. 最大风速
        wind_max = np.max(data[:, :5], axis=1, keepdims=True)
        feats.append(wind_max)

        # 5. 风向一致性（计算风向的余弦相似度）
        direction_cols = data[:, 5:10]
        direction_cos = np.cos(np.deg2rad(direction_cols))
        direction_mean = np.mean(direction_cos, axis=1, keepdims=True)
        feats.append(direction_mean)

        # 6. 空气密度修正因子（温度、气压、湿度的综合影响）
        temp = data[:, 10]
        pressure = data[:, 11]
        humidity = data[:, 12]
        air_density_factor = (pressure / 1013.25) * (288.15 / (temp + 273.15))
        feats.append(air_density_factor.reshape(-1, 1))

        # 7. 风能密度近似值
        wind_power_density = 0.5 * air_density_factor * (avg_wind ** 3).flatten()
        feats.append(wind_power_density.reshape(-1, 1))

        # 8. 时间周期特征（15 分钟数据的日内/周内周期）
        if data_index is not None:
            # 假设数据从某天的 00:00 开始
            hour_of_day = (data_index % 96) / 96 * 24  # 一天中的时刻
            day_of_week = (data_index // 96) % 7  # 一周中的第几天

            # 正弦编码时间周期
            time_sin = np.sin(2 * np.pi * hour_of_day / 24).reshape(-1, 1)
            time_cos = np.cos(2 * np.pi * hour_of_day / 24).reshape(-1, 1)
            dow_sin = np.sin(2 * np.pi * day_of_week / 7).reshape(-1, 1)
            dow_cos = np.cos(2 * np.pi * day_of_week / 7).reshape(-1, 1)

            feats.extend([time_sin, time_cos, dow_sin, dow_cos])

        # 🔧 新增：功率滞后特征（显式添加关键时间点的历史功率）
        # data的最后一列是功率
        power_col = data[:, -1]
        lags = [1, 2, 4, 8, 12, 24, 48]  # 滞后步数：15min, 30min, 1h, 2h, 3h, 6h, 12h
        for lag in lags:
            lagged = np.roll(power_col, lag)
            # 边界处理：用第一个有效值填充
            lagged[:lag] = power_col[lag-1] if lag > 0 else power_col[0]
            feats.append(lagged.reshape(-1, 1))
        
        # 🔧 新增：功率变化率（一阶差分）- 仅使用历史信息
        # power_diff[t] = power[t] - power[t-1]，只依赖当前和过去
        power_diff = np.zeros_like(power_col)
        power_diff[1:] = np.diff(power_col)  # 从第2个点开始计算差分
        power_diff[0] = 0  # 第一个点设为0
        feats.append(power_diff.reshape(-1, 1))
        
        # 🔧 新增：功率移动平均（平滑趋势）- 仅使用历史数据
        # 使用 pandas 进行向量化因果滚动，确保不泄露未来信息且效率更高
        import pandas as pd
        power_series = pd.Series(power_col)
        for window in [4, 12, 24]:  # 1h, 3h, 6h 移动平均
            # min_periods=1 确保在序列开头也能计算（使用已有历史数据）
            ma = power_series.rolling(window=window, min_periods=1).mean().values
            feats.append(ma.reshape(-1, 1))

        return np.hstack([data] + feats)

    X_train = add_physics_features(X_train, data_index=np.arange(len(X_train)))
    X_test = add_physics_features(X_test, data_index=np.arange(len(X_test)))
    
    if val_data is not None:
        X_val = add_physics_features(X_val, data_index=np.arange(len(X_val)))

    print(f"衍生特征构建完毕，最终特征维度: {X_train.shape[1]}")

    # 🚨 修复：统一使用原始功率作为训练和评估目标，避免“指标虚高”问题
    y_train_raw = y_train_raw.reshape(-1, 1)
    y_test_raw = y_test_raw.reshape(-1, 1)
    
    if val_data is not None:
        y_val_raw = y_val_raw.reshape(-1, 1)

    if val_data is not None:
        return X_train, X_val, X_test, y_train_raw, y_val_raw, y_test_raw
    else:
        return X_train, X_test, y_train_raw, y_test_raw


# ============================================
# 5 独立标准化
# 不用 StandardScaler 处理目标变量 Y，改用 MinMaxScaler。
# ============================================

from sklearn.preprocessing import StandardScaler, MinMaxScaler  # 引入 MinMaxScaler


def normalize_data(X_train, X_test, y_train, y_test, X_val=None, y_val=None):
    scaler_x = StandardScaler()

    # 🚨 核心修复：Y 的 Scaler 已在 extract_features 中基于原始数据拟合并保存
    # 这里直接加载，确保所有数据都在同一物理尺度下归一化
    scaler_y = load("./scaler_y")

    X_train_scaled = scaler_x.fit_transform(X_train)
    X_test_scaled = scaler_x.transform(X_test)
    
    if X_val is not None:
        X_val_scaled = scaler_x.transform(X_val)

    # 训练/验证/测试集：统一对原始功率标签进行归一化
    y_train_scaled = scaler_y.transform(y_train)
    y_test_scaled = scaler_y.transform(y_test)
    
    if y_val is not None:
        y_val_scaled = scaler_y.transform(y_val)

    dump(scaler_x, "./scaler_x")

    if X_val is not None:
        return X_train_scaled, X_val_scaled, X_test_scaled, y_train_scaled, y_val_scaled, y_test_scaled
    else:
        return X_train_scaled, X_test_scaled, y_train_scaled, y_test_scaled

# ============================================
# 6 构造滑动窗口 (完美防泄露)
# ============================================

def build_multi_step_windows(X_all, y_raw_all, window_size, train_end_idx, val_end_idx, horizons=[1, 4, 8, 16]):
    """
    为不同预测步长生成分组数据（支持验证集）
    
    参数:
        X_all: 全部特征数据 (N, features)
        y_raw_all: 原始功率标签 (N, 1)
        window_size: 窗口大小
        train_end_idx: 训练集结束索引
        val_end_idx: 验证集结束索引（如果无验证集则等于train_end_idx）
        horizons: 预测步长列表
    
    返回:
        datasets: dict, key为horizon, value包含train/val/test数据
    """
    datasets = {}
    
    for h in horizons:
        print(f"\n构建 {h}步预测数据...")
        train_x, train_y = [], []
        val_x, val_y = [], []
        test_x, test_y = [], []
        
        # 训练集：预测未来h步
        for i in range(window_size, train_end_idx - h + 1):
            X_window = X_all[i - window_size:i]
            y_future = y_raw_all[i:i+h].flatten()  # (h,)
            train_x.append(X_window)
            train_y.append(y_future)
        
        # 验证集（如果有）
        has_validation = val_end_idx > train_end_idx
        if has_validation:
            for i in range(train_end_idx + window_size, val_end_idx - h + 1):
                X_window = X_all[i - window_size:i]
                y_future = y_raw_all[i:i+h].flatten()
                
                val_x.append(X_window)
                val_y.append(y_future)
        
        # 测试集
        start_test_idx = val_end_idx if has_validation else train_end_idx
        for i in range(start_test_idx + window_size, len(X_all) - h + 1):
            X_window = X_all[i - window_size:i]
            y_future = y_raw_all[i:i+h].flatten()
            
            test_x.append(X_window)
            test_y.append(y_future)
        
        datasets[h] = {
            'train_x': torch.tensor(np.array(train_x)).float(),
            'train_y': torch.tensor(np.array(train_y)).float(),
            'test_x': torch.tensor(np.array(test_x)).float(),
            'test_y': torch.tensor(np.array(test_y)).float(),
        }
        
        # 如果有验证集，添加到字典中
        if has_validation and val_x:
            datasets[h]['val_x'] = torch.tensor(np.array(val_x)).float()
            datasets[h]['val_y'] = torch.tensor(np.array(val_y)).float()
        
        print(f"  训练集: {len(train_x)} 样本")
        if has_validation and val_x:
            print(f"  验证集: {len(val_x)} 样本")
        print(f"  测试集: {len(test_x)} 样本")
        print(f"  输入形状: {train_x[0].shape}")
        print(f"  输出形状: {train_y[0].shape}")
    
    return datasets


def build_windows(X_all, y_raw_all, _, window_size, train_end_idx, val_end_idx):
    """
    原有的单步预测数据构建（支持验证集）
    
    参数:
        X_all: 全部特征数据
        y_raw_all: 原始功率标签
        _: 占位符，保持接口兼容
        window_size: 窗口大小
        train_end_idx: 训练集结束索引
        val_end_idx: 验证集结束索引（如果无验证集则等于train_end_idx）
    """
    train_x, train_y = [], []
    val_x, val_y = [], []
    test_x, test_y = [], []

    # 训练集
    for i in range(window_size, train_end_idx):
        X_window = X_all[i - window_size:i]
        train_x.append(X_window)
        train_y.append(y_raw_all[i])

    # 验证集（如果有）
    has_validation = val_end_idx > train_end_idx
    if has_validation:
        for i in range(train_end_idx + window_size, val_end_idx):
            X_window = X_all[i - window_size:i]
            val_x.append(X_window)
            val_y.append(y_raw_all[i])

    # 测试集
    start_test_idx = val_end_idx if has_validation else train_end_idx
    for i in range(start_test_idx + window_size, len(X_all)):
        X_window = X_all[i - window_size:i]
        test_x.append(X_window)
        test_y.append(y_raw_all[i])

    result = [
        torch.tensor(np.array(train_x)).float(),
        torch.tensor(np.array(train_y)).float(),
    ]
    
    # 如果有验证集，插入到结果中
    if has_validation and val_x:
        result.extend([
            torch.tensor(np.array(val_x)).float(),
            torch.tensor(np.array(val_y)).float(),
        ])
    
    result.extend([
        torch.tensor(np.array(test_x)).float(),
        torch.tensor(np.array(test_y)).float()
    ])
    
    return tuple(result)


# ============================================
# 主程序
# ============================================

if __name__ == "__main__":
    DATA_PATH = r"D:\APredict\data\wind_data.csv"  # 每 15 分钟一个数据点
    START_INDEX = 7500  #  从数据中间开始截取，确保季节性覆盖
    DATA_SIZE = 20000   #  使用 20,000 条数据（约 208 天）
    WINDOW_SIZE = 48
    USE_VALIDATION = True  #  启用验证集
    VAL_RATIO = 0.1        #  验证集和测试集各占 10%（8:1:1 划分）

    # Step 1 & 2: 加载与划分
    data = load_data(DATA_PATH, DATA_SIZE, START_INDEX)
    
    if USE_VALIDATION:
        train_data, val_data, test_data, train_end_idx, val_end_idx = time_series_split(
            data, use_validation=True, val_ratio=VAL_RATIO
        )
    else:
        train_data, test_data, split_index = time_series_split(data, split_rate=0.9)
        val_data = None

    # 可视化不同 drop_k 的去噪效果，这里drop_k取1较佳，下次需要时再取消注释
    train_power_for_plot = train_data['实际发电功率（mw）'].values
    visualize_drop_k_experiments(train_power_for_plot, display_length=800)

    # Step 3 & 4: 提取特征与目标去噪
    if USE_VALIDATION:
        X_train, X_val, X_test, y_train_raw, y_val_raw, y_test_raw = \
            extract_features_and_targets(train_data, test_data, val_data)
    else:
        X_train, X_test, y_train_raw, y_test_raw = \
            extract_features_and_targets(train_data, test_data)

    # Step 5: 标准化
    if USE_VALIDATION:
        X_train_s, X_val_s, X_test_s, y_train_s, y_val_s, y_test_s = normalize_data(
            X_train, X_test, y_train_raw, y_test_raw,
            X_val=X_val, y_val=y_val_raw
        )
    else:
        X_train_s, X_test_s, y_train_s, y_test_s = normalize_data(
            X_train, X_test, y_train_raw, y_test_raw
        )

    # Step 6: 滑动窗口 (修复泄露问题)
    if USE_VALIDATION:
        X_all = np.vstack((X_train_s, X_val_s, X_test_s))
        y_raw_all = np.vstack((y_train_s, y_val_s, y_test_s))
    else:
        X_all = np.vstack((X_train_s, X_test_s))
        y_raw_all = np.vstack((y_train_s, y_test_s))

    # 🔧 新增：构建多步预测数据
    print("\n=== 构建多步预测数据集 ===")
    if USE_VALIDATION:
        multi_step_datasets = build_multi_step_windows(
            X_all, y_raw_all, WINDOW_SIZE, 
            train_end_idx, val_end_idx,
            horizons=[1, 4, 8]
        )
    else:
        multi_step_datasets = build_multi_step_windows(
            X_all, y_raw_all, WINDOW_SIZE, 
            split_index, split_index,
            horizons=[1, 4, 8]
        )
    
    # 保存多步数据集
    import os
    os.makedirs("./processed_data", exist_ok=True)
    dump(multi_step_datasets, "./processed_data/multi_step_datasets.pkl")
    print("\n[+] 已保存: multi_step_datasets.pkl")
    
    # 保留原有的单步数据（兼容性）
    if USE_VALIDATION:
        train_set, train_label_raw, val_set, val_label_raw, \
        test_set, test_label_raw = build_windows(
            X_all, y_raw_all, None, WINDOW_SIZE, 
            train_end_idx, val_end_idx
        )
    else:
        train_set, train_label_raw, test_set, test_label_raw = build_windows(
            X_all, y_raw_all, None, WINDOW_SIZE, split_index, split_index
        )

    # Step 7: 保存数据
    dump(train_set, "./processed_data/train_set")
    dump(train_label_raw, "./processed_data/train_label")  # 模型训练的目标（原始功率）
    
    if USE_VALIDATION:
        dump(val_set, "./processed_data/val_set")
        dump(val_label_raw, "./processed_data/val_label_raw")
        print("[+] 已保存验证集数据")

    dump(test_set, "./processed_data/test_set")
    dump(test_label_raw, "./processed_data/test_label_raw")  # 用于评估模型在真实环境中的最终误差！

    print("\n🎉 数据处理与保存完成！")
    print("模型输入 X 形状:", train_set.shape)
    print("原始目标 Y 形状:", train_label_raw.shape)
    if USE_VALIDATION:
        print("验证集 X 形状:", val_set.shape)
        print("验证集 Y 形状:", val_label_raw.shape)
        print("测试集 X 形状:", test_set.shape)
        print("测试集 Y 形状:", test_label_raw.shape)
    """
数据尺寸: (20000, 15) 
数据起点: 7500
训练集尺寸: (16000, 15)
验证集尺寸: (2000, 15)
测试集尺寸: (2000, 15)
[+] 已基于原始训练数据拟合 scaler_y 并保存
衍生特征构建完毕，最终特征维度: 40

--- 训练集目标去噪 ---
开始对目标功率进行 CEEMDAN 分解提纯...
共分解出 14 个 IMF 分量（含残差）。

--- 测试集目标去噪 ---
开始对目标功率进行 CEEMDAN 分解提纯...
共分解出 9 个 IMF 分量（含残差）。

--- 验证集目标去噪 ---
开始对目标功率进行 CEEMDAN 分解提纯...
共分解出 8 个 IMF 分量（含残差）。

=== 构建多步预测数据集 ===

构建 1步预测数据...
  训练集: 15952 样本
  验证集: 1952 样本
  测试集: 1952 样本
  输入形状: (48, 40)
  输出形状: (1,)

构建 4步预测数据...
  训练集: 15949 样本
  验证集: 1949 样本
  测试集: 1949 样本
  输入形状: (48, 40)
  输出形状: (4,)

构建 8步预测数据...
  训练集: 15945 样本
  验证集: 1945 样本
  测试集: 1945 样本
  输入形状: (48, 40)
  输出形状: (8,)

[+] 已保存: multi_step_datasets.pkl
[+] 已保存验证集数据

🎉 数据处理与保存完成！
模型输入 X 形状: torch.Size([15952, 48, 40])
干净目标 Y 形状: torch.Size([15952, 1])
验证集 X 形状: torch.Size([1952, 48, 40])
验证集 Y 形状: torch.Size([1952, 1])
测试集 X 形状: torch.Size([1952, 48, 40])
测试集 Y 形状: torch.Size([1952, 1])
    """