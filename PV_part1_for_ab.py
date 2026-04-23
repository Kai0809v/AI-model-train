# ==========================================================
# 消融实验专属数据处理脚本
# ==========================================================
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestRegressor
from boruta import BorutaPy
import joblib
import os

# ====== 猴子补丁：修复 Boruta 在新版 Numpy 下的报错 ======
# TODO： 这里到时候提醒一下
np.int = int
np.float = float
np.bool = bool
# ==========================================================

def run_feature_optimization_pipeline(input_path, output_dir="processed_data"):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    print("1. 正在读取并清洗原始数据...")
    df = pd.read_excel(input_path)

    # 映射列名
    col_mapping = {
        'Time(year-month-day h:m:s)': 'Time',
        'Total solar irradiance (W/m2)': 'TSI',
        'Direct normal irradiance (W/m2)': 'DNI',
        'Global horicontal irradiance (W/m2)': 'GHI',
        'Air temperature  (°C) ': 'Temp',
        'Atmosphere (hpa)': 'Atmosphere',
        'Relative humidity (%)': 'Humidity',
        'Power (MW)': 'Power'
    }
    df.rename(columns=col_mapping, inplace=True)

    print("2. 提取时间周期特征 (并执行 Informer 专属的 [-0.5, 0.5] 归一化)...")
    df['Time'] = pd.to_datetime(df['Time'])

    # ✅ 严格映射到 [-0.5, 0.5] 区间
    df['Month'] = (df['Time'].dt.month - 1) / 11.0 - 0.5
    df['Day'] = (df['Time'].dt.day - 1) / 30.0 - 0.5
    df['DayOfWeek'] = df['Time'].dt.dayofweek / 6.0 - 0.5
    df['Hour'] = df['Time'].dt.hour / 23.0 - 0.5

    # 15分钟数据，minute // 15 会得到 0, 1, 2, 3
    # 除以 3.0 映射到 [0, 1]，再减去 0.5 得到 [-0.5, 0.5]
    df['Minute'] = (df['Time'].dt.minute // 15) / 3.0 - 0.5

    print("2.5. 🆕 构建增强特征（仅非线性交互特征，移除滞后/滚动特征以避免数据泄露）...")
    
    # === 特征工程优化: 非线性交互特征 ===
    # 辐照度与温度的交互（影响光伏板效率）
    df['TSI_Temp_interaction'] = df['TSI'] * df['Temp']
    df['GHI_Temp_interaction'] = df['GHI'] * df['Temp']
    
    # 辐照度与湿度的比值（湿度影响大气透射率）
    df['TSI_Humidity_ratio'] = df['TSI'] / (df['Humidity'] + 1e-6)  # 避免除零
    df['GHI_Humidity_ratio'] = df['GHI'] / (df['Humidity'] + 1e-6)
    
    # DNI与GHI的比值（反映云层覆盖程度）
    df['DNI_GHI_ratio'] = df['DNI'] / (df['GHI'] + 1e-6)
    
    # 温度平方项（非线性效应）
    df['Temp_squared'] = df['Temp'] ** 2
    
    print(f"   -> 新增特征数量: 6个（仅交互特征，已移除滞后和滚动统计特征以避免数据泄露）")

    # 准备特征矩阵 X 和目标向量 y
    # 严格将时间特征踢出降维管线，让 Boruta 和 PCA 专注于处理天气数据
    # 🆕 仅保留气象特征和交互特征，移除所有滞后/滚动特征以避免数据泄露
    feature_cols = [
        'TSI', 'DNI', 'GHI', 'Temp', 'Atmosphere', 'Humidity',  # 原始气象特征
        'TSI_Temp_interaction', 'GHI_Temp_interaction',  # 交互特征
        'TSI_Humidity_ratio', 'GHI_Humidity_ratio',  # 比值特征
        'DNI_GHI_ratio',  # 云层指标
        'Temp_squared'  # 非线性项
    ]
    target_col = 'Power'

    X = df[feature_cols].values
    y = df[target_col].values

    print("3. 执行 8:1:1 时序划分...")
    n = len(df)
    train_end = int(n * 0.8)
    val_end = int(n * 0.9)

    X_train, y_train = X[:train_end], y[:train_end]
    X_val, y_val = X[train_end:val_end], y[train_end:val_end]
    X_test, y_test = X[val_end:], y[val_end:]

    print("4. 标准化处理 (基于训练集拟合)...")
    scaler_x = StandardScaler()
    scaler_y = StandardScaler()

    X_train_scaled = scaler_x.fit_transform(X_train)
    X_val_scaled = scaler_x.transform(X_val)
    X_test_scaled = scaler_x.transform(X_test)

    y_train_scaled = scaler_y.fit_transform(y_train.reshape(-1, 1)).flatten()
    y_val_scaled = scaler_y.transform(y_val.reshape(-1, 1)).flatten()
    y_test_scaled = scaler_y.transform(y_test.reshape(-1, 1)).flatten()

    print("5. 执行 Boruta 特征筛选 (仅在训练集上运行)...")
    rf = RandomForestRegressor(n_jobs=-1, max_depth=5, random_state=42)
    # perc=90 表示比 90% 的影子特征强才保留
    feat_selector = BorutaPy(rf, n_estimators='auto', verbose=0, random_state=42, perc=90)
    feat_selector.fit(X_train_scaled, y_train_scaled)

    selected_indices = feat_selector.support_
    selected_features = np.array(feature_cols)[selected_indices]
    print(f"   -> Boruta 选出的关键特征: {selected_features.tolist()}")

    X_train_boruta = X_train_scaled[:, selected_indices]
    X_val_boruta = X_val_scaled[:, selected_indices]
    X_test_boruta = X_test_scaled[:, selected_indices]

    # 后面试试0.98。  5、6步如果用了气象因子更多的数据集或许作用更显著些？
    # 4.2:尝试0.98->0.95(回退)
    PCA_components = 0.95
    print(f"6. 执行 PCA 降维 (消除共线性，保留 {PCA_components*100}% 信息)...")
    pca = PCA(n_components=PCA_components)
    X_train_pca = pca.fit_transform(X_train_boruta)
    X_val_pca = pca.transform(X_val_boruta)
    X_test_pca = pca.transform(X_test_boruta)

    print(f"   -> PCA 完成：维度从 {X_train_boruta.shape[1]} 降至 {pca.n_components_}")
    # === 这一块用在写文章中，展示 PCA 降维后的主成分方差贡献率，需要的话再绘制图表 ===
    print(f"   -> 各主成分方差贡献率:")
    for i in range(pca.n_components_):
        print(f"      PC{i+1}: {pca.explained_variance_ratio_[i]:.4f} "
              f"(累积: {sum(pca.explained_variance_ratio_[:i+1]):.4f})")
    # ====================================================================

    print("7. 保存优化后的数据包与模型...")
    # 打包所有组件，方便后续 TCN-Informer 直接加载
    # 注意：时间特征用于 Informer 的时间编码器
    bundle = {
        'train': (X_train_pca, y_train_scaled),
        'val': (X_val_pca, y_val_scaled),
        'test': (X_test_pca, y_test_scaled),
        'raw_test_y': y_test,  # 用于最后计算真实的 RMSE/MAE
        'scaler_x': scaler_x,
        'scaler_y': scaler_y,
        'pca': pca,
        'selected_features': selected_features,
        # 新增：时间特征 (Month, Day, Weekday, Hour, Minute) 用于 Informer 时间编码
        'time_features': (df[['Month', 'Day', 'DayOfWeek', 'Hour', 'Minute']].values[:train_end],
                          df[['Month', 'Day', 'DayOfWeek', 'Hour', 'Minute']].values[train_end:val_end],
                          df[['Month', 'Day', 'DayOfWeek', 'Hour', 'Minute']].values[val_end:])
    }

    joblib.dump(bundle, os.path.join(output_dir, "model_ready_data.pkl"))

    # 导出 CSV 预览版
    pca_cols = [f'PC{i + 1}' for i in range(pca.n_components_)]
    pd.DataFrame(X_train_pca, columns=pca_cols).to_csv(os.path.join(output_dir, "train_features.csv"), index=False)

    print(f"✅ 处理完成！模型就绪数据已保存至: {output_dir}/model_ready_data.pkl")

def message():
    print("居然没报错吗？\n 👍 恭喜你啊 x 被我恭喜到了，接下来打开./ablation文件夹📂运行里面的文件吧")


def run_without_bp_pipeline(input_path, output_dir="processed_data"):
    """
    无BP处理的数据生成管道：
    - 使用增强特征（仅交互特征，移除滞后+滚动以避免数据泄露）
    - 跳过 Boruta 特征选择
    - 跳过 PCA 降维
    仅做标准化
    """
    import joblib
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    print("===== [无BP处理] 数据管道 =====")
    print("1. 正在读取并清洗原始数据...")
    df = pd.read_excel(input_path)

    col_mapping = {
        'Time(year-month-day h:m:s)': 'Time',
        'Total solar irradiance (W/m2)': 'TSI',
        'Direct normal irradiance (W/m2)': 'DNI',
        'Global horicontal irradiance (W/m2)': 'GHI',
        'Air temperature  (°C) ': 'Temp',
        'Atmosphere (hpa)': 'Atmosphere',
        'Relative humidity (%)': 'Humidity',
        'Power (MW)': 'Power'
    }
    df.rename(columns=col_mapping, inplace=True)

    print("2. 提取时间周期特征...")
    df['Time'] = pd.to_datetime(df['Time'])
    df['Month'] = (df['Time'].dt.month - 1) / 11.0 - 0.5
    df['Day'] = (df['Time'].dt.day - 1) / 30.0 - 0.5
    df['DayOfWeek'] = df['Time'].dt.dayofweek / 6.0 - 0.5
    df['Hour'] = df['Time'].dt.hour / 23.0 - 0.5
    df['Minute'] = (df['Time'].dt.minute // 15) / 3.0 - 0.5

    print("2.5. 构建增强特征（仅交互特征，移除滞后/滚动以避免数据泄露）...")
    
    # === 非线性交互特征 ===
    df['TSI_Temp_interaction'] = df['TSI'] * df['Temp']
    df['GHI_Temp_interaction'] = df['GHI'] * df['Temp']
    df['TSI_Humidity_ratio'] = df['TSI'] / (df['Humidity'] + 1e-6)
    df['GHI_Humidity_ratio'] = df['GHI'] / (df['Humidity'] + 1e-6)
    df['DNI_GHI_ratio'] = df['DNI'] / (df['GHI'] + 1e-6)
    df['Temp_squared'] = df['Temp'] ** 2

    feature_cols = [
        'TSI', 'DNI', 'GHI', 'Temp', 'Atmosphere', 'Humidity',  # 原始气象特征
        'TSI_Temp_interaction', 'GHI_Temp_interaction',  # 交互特征
        'TSI_Humidity_ratio', 'GHI_Humidity_ratio',  # 比值特征
        'DNI_GHI_ratio',  # 云层指标
        'Temp_squared'  # 非线性项
    ]
    target_col = 'Power'

    X = df[feature_cols].values
    y = df[target_col].values

    print("3. 执行 8:1:1 时序划分...")
    n = len(df)
    train_end = int(n * 0.8)
    val_end = int(n * 0.9)

    X_train, y_train = X[:train_end], y[:train_end]
    X_val, y_val = X[train_end:val_end], y[train_end:val_end]
    X_test, y_test = X[val_end:], y[val_end:]

    print("4. 标准化处理...")
    scaler_x = StandardScaler()
    scaler_y = StandardScaler()

    X_train_scaled = scaler_x.fit_transform(X_train)
    X_val_scaled = scaler_x.transform(X_val)
    X_test_scaled = scaler_x.transform(X_test)

    y_train_scaled = scaler_y.fit_transform(y_train.reshape(-1, 1)).flatten()
    y_val_scaled = scaler_y.transform(y_val.reshape(-1, 1)).flatten()
    y_test_scaled = scaler_y.transform(y_test.reshape(-1, 1)).flatten()

    print(f"   -> 特征维度: {X_train_scaled.shape[1]} (无PCA降维)")

    bundle = {
        'train': (X_train_scaled, y_train_scaled),
        'val': (X_val_scaled, y_val_scaled),
        'test': (X_test_scaled, y_test_scaled),
        'raw_test_y': y_test,
        'scaler_x': scaler_x,
        'scaler_y': scaler_y,
        'selected_features': feature_cols,
        'time_features': (df[['Month', 'Day', 'DayOfWeek', 'Hour', 'Minute']].values[:train_end],
                          df[['Month', 'Day', 'DayOfWeek', 'Hour', 'Minute']].values[train_end:val_end],
                          df[['Month', 'Day', 'DayOfWeek', 'Hour', 'Minute']].values[val_end:])
    }

    joblib.dump(bundle, os.path.join(output_dir, "model_ready_data_no_bp.pkl"))
    print(f"✅ [无BP处理] 数据已保存至: {output_dir}/model_ready_data_no_bp.pkl")


if __name__ == "__main__":
    PV_DATA = "./data/PV130MW.xlsx"
    run_feature_optimization_pipeline(PV_DATA)
    run_without_bp_pipeline(PV_DATA)
    message()

