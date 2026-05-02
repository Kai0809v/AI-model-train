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
np.int = int
np.float = float
np.bool = bool
# ==========================================================


def add_lag_and_rolling_features(df, power_lags=[4, 12, 24], weather_lags=[4], rolling_windows=[12, 48, 96]):
    """
    为单个数据集添加滞后和滚动特征（避免数据泄露）
    """
    df_feat = df.copy()
    
    for lag in power_lags:
        df_feat[f'Power_lag_{lag}'] = df_feat['Power'].shift(lag)
    
    for lag in weather_lags:
        df_feat[f'TSI_lag_{lag}'] = df_feat['TSI'].shift(lag)
        df_feat[f'DNI_lag_{lag}'] = df_feat['DNI'].shift(lag)
        df_feat[f'GHI_lag_{lag}'] = df_feat['GHI'].shift(lag)
    
    for window in rolling_windows:
        df_feat[f'Power_rolling_mean_{window}'] = df_feat['Power'].rolling(window=window).mean()
        df_feat[f'Power_rolling_std_{window}'] = df_feat['Power'].rolling(window=window).std()
        df_feat[f'TSI_rolling_mean_{window}'] = df_feat['TSI'].rolling(window=window).mean()
        df_feat[f'GHI_rolling_mean_{window}'] = df_feat['GHI'].rolling(window=window).mean()
    
    df_feat.ffill(inplace=True)
    df_feat.bfill(inplace=True)
    
    return df_feat


def run_feature_optimization_pipeline(input_path, output_dir="processed_data"):
    """
    BP处理管道：完整增强特征 + Boruta + PCA
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    print("===== [BP处理] 数据管道 =====")
    print("1. 读取并清洗原始数据...")
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

    print("2.5. 构建增强特征（交互+温度修正）...")
    df['TSI_Temp_interaction'] = df['TSI'] * df['Temp']
    df['GHI_Temp_interaction'] = df['GHI'] * df['Temp']
    df['TSI_Humidity_ratio'] = df['TSI'] / (df['Humidity'] + 1e-6)
    df['GHI_Humidity_ratio'] = df['GHI'] / (df['Humidity'] + 1e-6)
    df['DNI_GHI_ratio'] = df['DNI'] / (df['GHI'] + 1e-6)
    df['Temp_squared'] = df['Temp'] ** 2
    df['TSI_Corrected'] = df['TSI'] * (1 - 0.004 * (df['Temp'] - 25))
    df['GHI_Corrected'] = df['GHI'] * (1 - 0.004 * (df['Temp'] - 25))

    print("3. 执行 8:1:1 时序划分...")
    n = len(df)
    train_end = int(n * 0.8)
    val_end = int(n * 0.9)

    df_train = df[:train_end].copy()
    df_val = df[train_end:val_end].copy()
    df_test = df[val_end:].copy()

    print("3.5. 添加滞后和滚动特征...")
    df_train = add_lag_and_rolling_features(df_train)
    df_val = add_lag_and_rolling_features(df_val)
    df_test = add_lag_and_rolling_features(df_test)

    feature_cols = [
        'TSI', 'DNI', 'GHI', 'Temp', 'Atmosphere', 'Humidity',
        'TSI_Temp_interaction', 'GHI_Temp_interaction',
        'TSI_Humidity_ratio', 'GHI_Humidity_ratio', 'DNI_GHI_ratio',
        'Temp_squared', 'TSI_Corrected', 'GHI_Corrected',
        'Power_lag_4', 'Power_lag_12', 'Power_lag_24',
        'TSI_lag_4', 'DNI_lag_4', 'GHI_lag_4',
        'Power_rolling_mean_12', 'Power_rolling_std_12',
        'Power_rolling_mean_48', 'Power_rolling_std_48',
        'Power_rolling_mean_96', 'Power_rolling_std_96',
        'TSI_rolling_mean_12', 'GHI_rolling_mean_12',
        'TSI_rolling_mean_48', 'GHI_rolling_mean_48',
        'TSI_rolling_mean_96', 'GHI_rolling_mean_96',
    ]
    target_col = 'Power'

    X_train = df_train[feature_cols].values
    y_train = df_train[target_col].values
    X_val = df_val[feature_cols].values
    y_val = df_val[target_col].values
    X_test = df_test[feature_cols].values
    y_test = df_test[target_col].values

    print("4. 标准化处理...")
    scaler_x = StandardScaler()
    scaler_y = StandardScaler()
    X_train_scaled = scaler_x.fit_transform(X_train)
    X_val_scaled = scaler_x.transform(X_val)
    X_test_scaled = scaler_x.transform(X_test)
    y_train_scaled = scaler_y.fit_transform(y_train.reshape(-1, 1)).flatten()
    y_val_scaled = scaler_y.transform(y_val.reshape(-1, 1)).flatten()
    y_test_scaled = scaler_y.transform(y_test.reshape(-1, 1)).flatten()

    print("5. 执行 Boruta 特征筛选...")
    rf = RandomForestRegressor(n_jobs=-1, max_depth=5, random_state=42)
    feat_selector = BorutaPy(rf, n_estimators='auto', verbose=0, random_state=42, perc=90)
    feat_selector.fit(X_train_scaled, y_train_scaled)
    selected_indices = feat_selector.support_
    selected_features = np.array(feature_cols)[selected_indices]
    print(f"   -> Boruta 选出: {selected_features.tolist()}")

    X_train_boruta = X_train_scaled[:, selected_indices]
    X_val_boruta = X_val_scaled[:, selected_indices]
    X_test_boruta = X_test_scaled[:, selected_indices]

    print("6. 执行 PCA 降维 (保留95%方差)...")
    pca = PCA(n_components=0.95)
    X_train_pca = pca.fit_transform(X_train_boruta)
    X_val_pca = pca.transform(X_val_boruta)
    X_test_pca = pca.transform(X_test_boruta)
    print(f"   -> PCA: {X_train_boruta.shape[1]} → {pca.n_components_} 维")

    bundle = {
        'train': (X_train_pca, y_train_scaled),
        'val': (X_val_pca, y_val_scaled),
        'test': (X_test_pca, y_test_scaled),
        'raw_test_y': y_test,
        'scaler_x': scaler_x,
        'scaler_y': scaler_y,
        'pca': pca,
        'selected_features': selected_features,
        'time_features': (df_train[['Month', 'Day', 'DayOfWeek', 'Hour', 'Minute']].values,
                          df_val[['Month', 'Day', 'DayOfWeek', 'Hour', 'Minute']].values,
                          df_test[['Month', 'Day', 'DayOfWeek', 'Hour', 'Minute']].values)
    }

    joblib.dump(bundle, os.path.join(output_dir, "model_ready_data.pkl"))
    print(f"✅ [BP-FULL处理] 数据已保存\n")


def run_feature_optimization_pipeline_base(input_path, output_dir="processed_data"):
    """
    BP-BASE处理管道：交互特征 + Boruta + PCA
    排除：滞后特征 + 滚动统计
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    print("===== [BP-BASE处理] 数据管道 =====")
    print("1. 读取并清洗原始数据...")
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

    print("2.5. 构建交互特征（不含滞后和滚动）...")
    df['TSI_Temp_interaction'] = df['TSI'] * df['Temp']
    df['GHI_Temp_interaction'] = df['GHI'] * df['Temp']
    df['TSI_Humidity_ratio'] = df['TSI'] / (df['Humidity'] + 1e-6)
    df['GHI_Humidity_ratio'] = df['GHI'] / (df['Humidity'] + 1e-6)
    df['DNI_GHI_ratio'] = df['DNI'] / (df['GHI'] + 1e-6)
    df['Temp_squared'] = df['Temp'] ** 2
    df['TSI_Corrected'] = df['TSI'] * (1 - 0.004 * (df['Temp'] - 25))
    df['GHI_Corrected'] = df['GHI'] * (1 - 0.004 * (df['Temp'] - 25))

    print("3. 执行 8:1:1 时序划分...")
    n = len(df)
    train_end = int(n * 0.8)
    val_end = int(n * 0.9)

    feature_cols = [
        'TSI', 'DNI', 'GHI', 'Temp', 'Atmosphere', 'Humidity',
        'TSI_Temp_interaction', 'GHI_Temp_interaction',
        'TSI_Humidity_ratio', 'GHI_Humidity_ratio', 'DNI_GHI_ratio',
        'Temp_squared', 'TSI_Corrected', 'GHI_Corrected'
    ]
    target_col = 'Power'

    X_train = df[feature_cols].values[:train_end]
    y_train = df[target_col].values[:train_end]
    X_val = df[feature_cols].values[train_end:val_end]
    y_val = df[target_col].values[train_end:val_end]
    X_test = df[feature_cols].values[val_end:]
    y_test = df[target_col].values[val_end:]

    print("4. 标准化处理...")
    scaler_x = StandardScaler()
    scaler_y = StandardScaler()
    X_train_scaled = scaler_x.fit_transform(X_train)
    X_val_scaled = scaler_x.transform(X_val)
    X_test_scaled = scaler_x.transform(X_test)
    y_train_scaled = scaler_y.fit_transform(y_train.reshape(-1, 1)).flatten()
    y_val_scaled = scaler_y.transform(y_val.reshape(-1, 1)).flatten()
    y_test_scaled = scaler_y.transform(y_test.reshape(-1, 1)).flatten()

    print("5. 执行 Boruta 特征筛选...")
    rf = RandomForestRegressor(n_jobs=-1, max_depth=5, random_state=42)
    feat_selector = BorutaPy(rf, n_estimators='auto', verbose=0, random_state=42, perc=90)
    feat_selector.fit(X_train_scaled, y_train_scaled)
    selected_indices = feat_selector.support_
    selected_features = np.array(feature_cols)[selected_indices]
    print(f"   -> Boruta 选出: {selected_features.tolist()}")

    X_train_boruta = X_train_scaled[:, selected_indices]
    X_val_boruta = X_val_scaled[:, selected_indices]
    X_test_boruta = X_test_scaled[:, selected_indices]

    print("6. 执行 PCA 降维 (保留95%方差)...")
    pca = PCA(n_components=0.95)
    X_train_pca = pca.fit_transform(X_train_boruta)
    X_val_pca = pca.transform(X_val_boruta)
    X_test_pca = pca.transform(X_test_boruta)
    print(f"   -> PCA: {X_train_boruta.shape[1]} → {pca.n_components_} 维")

    bundle = {
        'train': (X_train_pca, y_train_scaled),
        'val': (X_val_pca, y_val_scaled),
        'test': (X_test_pca, y_test_scaled),
        'raw_test_y': y_test,
        'scaler_x': scaler_x,
        'scaler_y': scaler_y,
        'pca': pca,
        'selected_features': selected_features,
        'time_features': (df[['Month', 'Day', 'DayOfWeek', 'Hour', 'Minute']].values[:train_end],
                          df[['Month', 'Day', 'DayOfWeek', 'Hour', 'Minute']].values[train_end:val_end],
                          df[['Month', 'Day', 'DayOfWeek', 'Hour', 'Minute']].values[val_end:])
    }

    joblib.dump(bundle, os.path.join(output_dir, "model_ready_data_base.pkl"))
    print(f"✅ [BP-BASE处理] 数据已保存\n")


def run_without_bp_pipeline(input_path, output_dir="processed_data"):
    """
    NoBP处理管道：仅基础气象特征 + 时间特征
    - 无增强特征（无交互、无滞后、无滚动）
    - 无 Boruta 特征选择
    - 无 PCA 降维
    仅做标准化
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    print("===== [NoBP处理] 数据管道 =====")
    print("1. 读取并清洗原始数据...")
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

    # ⚠️ NoBP处理：仅使用基础气象特征，无任何增强特征
    feature_cols = [
        'TSI', 'DNI', 'GHI', 'Temp', 'Atmosphere', 'Humidity'  # 仅基础气象特征
    ]
    target_col = 'Power'

    print("3. 执行 8:1:1 时序划分...")
    n = len(df)
    train_end = int(n * 0.8)
    val_end = int(n * 0.9)

    X_train = df[feature_cols].values[:train_end]
    y_train = df[target_col].values[:train_end]
    X_val = df[feature_cols].values[train_end:val_end]
    y_val = df[target_col].values[train_end:val_end]
    X_test = df[feature_cols].values[val_end:]
    y_test = df[target_col].values[val_end:]

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
    print(f"✅ [NoBP处理] 数据已保存\n")


if __name__ == "__main__":
    PV_DATA = "./data/PV130MW.xlsx"
    run_feature_optimization_pipeline(PV_DATA)
    run_without_bp_pipeline(PV_DATA)
    print("🎉 数据处理完成！请运行 ablation 文件夹中的消融实验脚本")