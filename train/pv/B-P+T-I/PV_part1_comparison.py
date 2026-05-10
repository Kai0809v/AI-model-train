import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA, KernelPCA, SparsePCA
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
    
    # === 1. 功率滞后特征 ===
    for lag in power_lags:
        df_feat[f'Power_lag_{lag}'] = df_feat['Power'].shift(lag)
    
    # === 2. 气象特征滞后 ===
    for lag in weather_lags:
        df_feat[f'TSI_lag_{lag}'] = df_feat['TSI'].shift(lag)
        df_feat[f'DNI_lag_{lag}'] = df_feat['DNI'].shift(lag)
        df_feat[f'GHI_lag_{lag}'] = df_feat['GHI'].shift(lag)
    
    # === 3. 滚动统计特征 ===
    for window in rolling_windows:
        df_feat[f'Power_rolling_mean_{window}'] = df_feat['Power'].rolling(window=window).mean()
        df_feat[f'Power_rolling_std_{window}'] = df_feat['Power'].rolling(window=window).std()
        df_feat[f'TSI_rolling_mean_{window}'] = df_feat['TSI'].rolling(window=window).mean()
        df_feat[f'GHI_rolling_mean_{window}'] = df_feat['GHI'].rolling(window=window).mean()
    
    # 填充 NaN
    df_feat.ffill(inplace=True)
    df_feat.bfill(inplace=True)
    
    return df_feat


def run_dimensionality_reduction(X_train, X_val, X_test, method='pca', **kwargs):
    """
    执行降维处理
    
    参数:
        X_train, X_val, X_test: 训练/验证/测试集特征
        method: 降维方法 ('pca', 'kpca', 'spca')
        **kwargs: 传递给降维算法的额外参数
    
    返回:
        (X_train_reduced, X_val_reduced, X_test_reduced, reducer_model)
    """
    print(f"\n   -> 使用 {method.upper()} 进行降维...")
    
    if method == 'pca':
        # 标准 PCA
        n_components = kwargs.get('n_components', 0.95)
        reducer = PCA(n_components=n_components)
        X_train_reduced = reducer.fit_transform(X_train)
        X_val_reduced = reducer.transform(X_val)
        X_test_reduced = reducer.transform(X_test)
        
        print(f"   -> PCA 完成：维度从 {X_train.shape[1]} 降至 {reducer.n_components_}")
        print(f"   -> 累积方差解释率: {sum(reducer.explained_variance_ratio_):.4f}")
        
    elif method == 'kpca':
        # 核 PCA
        n_components = kwargs.get('n_components', None)
        kernel = kwargs.get('kernel', 'rbf')
        gamma = kwargs.get('gamma', None)
        
        if n_components is None:
            n_components = min(10, X_train.shape[1])  # KPCA 需要指定具体维度
        
        reducer = KernelPCA(n_components=n_components, kernel=kernel, gamma=gamma, random_state=42)
        X_train_reduced = reducer.fit_transform(X_train)
        X_val_reduced = reducer.transform(X_val)
        X_test_reduced = reducer.transform(X_test)
        
        print(f"   -> KPCA 完成：维度从 {X_train.shape[1]} 降至 {n_components}")
        print(f"   -> 核函数: {kernel}, Gamma: {gamma}")
        
    elif method == 'spca':
        # 稀疏 PCA
        n_components = kwargs.get('n_components', None)
        alpha = kwargs.get('alpha', 1.0)
        
        if n_components is None:
            n_components = min(10, X_train.shape[1])
        
        reducer = SparsePCA(n_components=n_components, alpha=alpha, random_state=42, max_iter=1000)
        X_train_reduced = reducer.fit_transform(X_train)
        X_val_reduced = reducer.transform(X_val)
        X_test_reduced = reducer.transform(X_test)
        
        print(f"   -> SPCA 完成：维度从 {X_train.shape[1]} 降至 {n_components}")
        print(f"   -> 稀疏系数 alpha: {alpha}")
        
    else:
        raise ValueError(f"不支持的降维方法: {method}")
    
    return X_train_reduced, X_val_reduced, X_test_reduced, reducer


def run_feature_optimization_with_method(input_path, output_dir="processed_data", 
                                         reduction_method='pca', **reduction_kwargs):
    """
    光伏预测特征工程流程（Boruta + 可配置的降维方法）
    
    参数:
        input_path: 原始数据路径
        output_dir: 输出目录
        reduction_method: 降维方法 ('pca', 'kpca', 'spca')
        **reduction_kwargs: 降维方法的额外参数
    """
    # 根据降维方法生成输出文件名
    method_suffix = reduction_method.upper()
    output_filename = f"model_ready_data_{method_suffix.lower()}.pkl"
    output_path = os.path.join(output_dir, output_filename)
    
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    print("="*60)
    print(f"开始特征工程流程 - 降维方法: {reduction_method.upper()}")
    print("="*60)

    print("\n1. 正在读取并清洗原始数据...")
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

    df['Month'] = (df['Time'].dt.month - 1) / 11.0 - 0.5
    df['Day'] = (df['Time'].dt.day - 1) / 30.0 - 0.5
    df['DayOfWeek'] = df['Time'].dt.dayofweek / 6.0 - 0.5
    df['Hour'] = df['Time'].dt.hour / 23.0 - 0.5
    df['Minute'] = (df['Time'].dt.minute // 15) / 3.0 - 0.5

    print("2.5. 构建增强特征（仅非线性交互 + 温度修正）...")
    
    # 非线性交互特征
    df['TSI_Temp_interaction'] = df['TSI'] * df['Temp']
    df['GHI_Temp_interaction'] = df['GHI'] * df['Temp']
    df['TSI_Humidity_ratio'] = df['TSI'] / (df['Humidity'] + 1e-6)
    df['GHI_Humidity_ratio'] = df['GHI'] / (df['Humidity'] + 1e-6)
    df['DNI_GHI_ratio'] = df['DNI'] / (df['GHI'] + 1e-6)
    df['Temp_squared'] = df['Temp'] ** 2

    # 温度修正特征
    df['TSI_Corrected'] = df['TSI'] * (1 - 0.004 * (df['Temp'] - 25))
    df['GHI_Corrected'] = df['GHI'] * (1 - 0.004 * (df['Temp'] - 25))

    print(f"   -> 新增特征: 6个交互特征 + 2个温度修正特征")

    print("\n3. 执行 8:1:1 时序划分...")
    n = len(df)
    train_end = int(n * 0.8)
    val_end = int(n * 0.9)

    df_train = df[:train_end].copy()
    df_val = df[train_end:val_end].copy()
    df_test = df[val_end:].copy()
    
    print(f"   -> 训练集: {len(df_train)} 样本")
    print(f"   -> 验证集: {len(df_val)} 样本")
    print(f"   -> 测试集: {len(df_test)} 样本")
    
    print("\n3.5. 在各个集合上独立计算滞后和滚动特征...")
    df_train = add_lag_and_rolling_features(df_train)
    df_val = add_lag_and_rolling_features(df_val)
    df_test = add_lag_and_rolling_features(df_test)

    print("\n4. 准备特征矩阵 X 和目标向量 y...")
    feature_cols = [
        'TSI', 'DNI', 'GHI', 'Temp', 'Atmosphere', 'Humidity',
        'TSI_Temp_interaction', 'GHI_Temp_interaction',
        'TSI_Humidity_ratio', 'GHI_Humidity_ratio',
        'DNI_GHI_ratio', 'Temp_squared',
        'TSI_Corrected', 'GHI_Corrected',
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
    
    print(f"   -> 总特征数: {len(feature_cols)} 个")

    print("\n5. 标准化处理 (基于训练集拟合)...")
    scaler_x = StandardScaler()
    scaler_y = StandardScaler()

    X_train_scaled = scaler_x.fit_transform(X_train)
    X_val_scaled = scaler_x.transform(X_val)
    X_test_scaled = scaler_x.transform(X_test)

    y_train_scaled = scaler_y.fit_transform(y_train.reshape(-1, 1)).flatten()
    y_val_scaled = scaler_y.transform(y_val.reshape(-1, 1)).flatten()
    y_test_scaled = scaler_y.transform(y_test.reshape(-1, 1)).flatten()

    print("\n6. 执行 Boruta 特征筛选 (仅在训练集上运行)...")
    rf = RandomForestRegressor(n_jobs=-1, max_depth=5, random_state=42)
    feat_selector = BorutaPy(rf, n_estimators='auto', verbose=0, random_state=42, perc=90)
    feat_selector.fit(X_train_scaled, y_train_scaled)

    selected_indices = feat_selector.support_
    selected_features = np.array(feature_cols)[selected_indices]
    print(f"   -> Boruta 选出的关键特征数量: {sum(selected_indices)} / {len(feature_cols)}")
    print(f"   -> 关键特征: {selected_features.tolist()}")

    X_train_boruta = X_train_scaled[:, selected_indices]
    X_val_boruta = X_val_scaled[:, selected_indices]
    X_test_boruta = X_test_scaled[:, selected_indices]

    print("\n7. 执行降维处理...")
    X_train_reduced, X_val_reduced, X_test_reduced, reducer = run_dimensionality_reduction(
        X_train_boruta, X_val_boruta, X_test_boruta,
        method=reduction_method,
        **reduction_kwargs
    )

    print("\n8. 保存优化后的数据包与模型...")
    bundle = {
        'train': (X_train_reduced, y_train_scaled),
        'val': (X_val_reduced, y_val_scaled),
        'test': (X_test_reduced, y_test_scaled),
        'raw_test_y': y_test,
        'scaler_x': scaler_x,
        'scaler_y': scaler_y,
        'reducer': reducer,  # 保存降维模型
        'reduction_method': reduction_method,  # 记录使用的降维方法
        'selected_features': selected_features,
        'time_features': (
            df_train[['Month', 'Day', 'DayOfWeek', 'Hour', 'Minute']].values,
            df_val[['Month', 'Day', 'DayOfWeek', 'Hour', 'Minute']].values,
            df_test[['Month', 'Day', 'DayOfWeek', 'Hour', 'Minute']].values
        )
    }

    joblib.dump(bundle, output_path)

    # 导出 CSV 预览版
    reduced_dim = X_train_reduced.shape[1]
    component_cols = [f'PC{i + 1}' for i in range(reduced_dim)]
    pd.DataFrame(X_train_reduced, columns=component_cols).to_csv(
        os.path.join(output_dir, f"train_features_{method_suffix.lower()}.csv"), 
        index=False
    )

    print(f"\n✅ {reduction_method.upper()} 处理完成！")
    print(f"   模型就绪数据已保存至: {output_path}")
    print(f"   降维后维度: {reduced_dim}")
    print("="*60)
    
    return output_path


if __name__ == "__main__":
    PV_DATA = r"D:\APredict\data\PV130MW.xlsx"
    
    # 测试三种降维方法
    methods_to_test = [
        ('pca', {'n_components': 0.95}),
        ('kpca', {'n_components': 10, 'kernel': 'rbf', 'gamma': 0.1}),
        ('spca', {'n_components': 10, 'alpha': 1.0}),
    ]
    
    for method, kwargs in methods_to_test:
        try:
            print(f"\n\n{'#'*60}")
            print(f"# 开始测试: {method.upper()}")
            print(f"{'#'*60}\n")
            
            output_path = run_feature_optimization_with_method(
                PV_DATA, 
                output_dir="processed_data",
                reduction_method=method,
                **kwargs
            )
            
            print(f"\n✓ {method.upper()} 成功完成！\n")
            
        except Exception as e:
            print(f"\n✗ {method.upper()} 失败: {str(e)}\n")
            import traceback
            traceback.print_exc()
