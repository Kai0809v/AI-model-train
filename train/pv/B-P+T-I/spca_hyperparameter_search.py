"""
SPCA 超参数网格搜索实验

目标：找到最优的 SPCA 配置（n_components 和 alpha）

分析当前结果：
- PCA: RMSE=3.5552, R²=0.9822 (基线)
- SPCA (n=10, alpha=1.0): RMSE=3.3699, R²=0.9840 ✓ 已超越 PCA

优化方向：
1. n_components: 尝试更多维度配置 (5, 8, 10, 12, 15, 20)
2. alpha: 调整稀疏程度 (0.01, 0.1, 0.5, 1.0, 2.0, 5.0)
   - alpha 越小 → 越密集（接近普通 PCA）
   - alpha 越大 → 越稀疏（特征选择更强）
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import SparsePCA
from sklearn.ensemble import RandomForestRegressor
from boruta import BorutaPy
import joblib
import os
import itertools

# ====== 猴子补丁 ======
np.int = int
np.float = float
np.bool = bool


def add_lag_and_rolling_features(df, power_lags=[4, 12, 24], weather_lags=[4], rolling_windows=[12, 48, 96]):
    """添加滞后和滚动特征"""
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


def run_spca_with_params(input_path, n_components, alpha, output_dir="processed_data"):
    """
    使用指定参数运行 SPCA 特征工程
    
    返回:
        output_path: 生成的 pkl 文件路径
        reducer_info: 降维模型信息字典
    """
    print(f"\n{'='*70}")
    print(f"SPCA 参数配置: n_components={n_components}, alpha={alpha}")
    print(f"{'='*70}")

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    print("\n1. 读取并清洗数据...")
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

    print("2. 提取时间特征...")
    df['Time'] = pd.to_datetime(df['Time'])
    df['Month'] = (df['Time'].dt.month - 1) / 11.0 - 0.5
    df['Day'] = (df['Time'].dt.day - 1) / 30.0 - 0.5
    df['DayOfWeek'] = df['Time'].dt.dayofweek / 6.0 - 0.5
    df['Hour'] = df['Time'].dt.hour / 23.0 - 0.5
    df['Minute'] = (df['Time'].dt.minute // 15) / 3.0 - 0.5

    print("3. 构建增强特征...")
    df['TSI_Temp_interaction'] = df['TSI'] * df['Temp']
    df['GHI_Temp_interaction'] = df['GHI'] * df['Temp']
    df['TSI_Humidity_ratio'] = df['TSI'] / (df['Humidity'] + 1e-6)
    df['GHI_Humidity_ratio'] = df['GHI'] / (df['Humidity'] + 1e-6)
    df['DNI_GHI_ratio'] = df['DNI'] / (df['GHI'] + 1e-6)
    df['Temp_squared'] = df['Temp'] ** 2
    df['TSI_Corrected'] = df['TSI'] * (1 - 0.004 * (df['Temp'] - 25))
    df['GHI_Corrected'] = df['GHI'] * (1 - 0.004 * (df['Temp'] - 25))

    print("4. 时序划分 (8:1:1)...")
    n = len(df)
    train_end = int(n * 0.8)
    val_end = int(n * 0.9)

    df_train = df[:train_end].copy()
    df_val = df[train_end:val_end].copy()
    df_test = df[val_end:].copy()
    
    df_train = add_lag_and_rolling_features(df_train)
    df_val = add_lag_and_rolling_features(df_val)
    df_test = add_lag_and_rolling_features(df_test)

    print("5. 准备特征矩阵...")
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

    X_train = df_train[feature_cols].values
    y_train = df_train['Power'].values
    X_val = df_val[feature_cols].values
    y_val = df_val['Power'].values
    X_test = df_test[feature_cols].values
    y_test = df_test['Power'].values

    print("6. 标准化处理...")
    scaler_x = StandardScaler()
    scaler_y = StandardScaler()

    X_train_scaled = scaler_x.fit_transform(X_train)
    X_val_scaled = scaler_x.transform(X_val)
    X_test_scaled = scaler_x.transform(X_test)

    y_train_scaled = scaler_y.fit_transform(y_train.reshape(-1, 1)).flatten()
    y_val_scaled = scaler_y.transform(y_val.reshape(-1, 1)).flatten()
    y_test_scaled = scaler_y.transform(y_test.reshape(-1, 1)).flatten()

    print("7. Boruta 特征筛选...")
    rf = RandomForestRegressor(n_jobs=-1, max_depth=5, random_state=42)
    feat_selector = BorutaPy(rf, n_estimators='auto', verbose=0, random_state=42, perc=90)
    feat_selector.fit(X_train_scaled, y_train_scaled)

    selected_indices = feat_selector.support_
    selected_features = np.array(feature_cols)[selected_indices]
    print(f"   -> 选出 {sum(selected_indices)} 个关键特征")

    X_train_boruta = X_train_scaled[:, selected_indices]
    X_val_boruta = X_val_scaled[:, selected_indices]
    X_test_boruta = X_test_scaled[:, selected_indices]

    print(f"8. 执行 SPCA 降维 (n_components={n_components}, alpha={alpha})...")
    reducer = SparsePCA(
        n_components=n_components, 
        alpha=alpha, 
        random_state=42, 
        max_iter=1000
    )
    X_train_reduced = reducer.fit_transform(X_train_boruta)
    X_val_reduced = reducer.transform(X_val_boruta)
    X_test_reduced = reducer.transform(X_test_boruta)

    # 计算稀疏度（每列非零元素的比例）
    sparsity = np.mean(np.abs(reducer.components_) > 1e-10)
    print(f"   -> 降维完成: {X_train_boruta.shape[1]} → {n_components}")
    print(f"   -> 稀疏度: {sparsity:.2%} (越低表示越稀疏)")

    print("9. 保存数据包...")
    output_filename = f"model_ready_data_spca_n{n_components}_a{alpha}.pkl"
    output_path = os.path.join(output_dir, output_filename)
    
    bundle = {
        'train': (X_train_reduced, y_train_scaled),
        'val': (X_val_reduced, y_val_scaled),
        'test': (X_test_reduced, y_test_scaled),
        'raw_test_y': y_test,
        'scaler_x': scaler_x,
        'scaler_y': scaler_y,
        'reducer': reducer,
        'reduction_method': 'spca',
        'spca_params': {'n_components': n_components, 'alpha': alpha},
        'selected_features': selected_features,
        'time_features': (
            df_train[['Month', 'Day', 'DayOfWeek', 'Hour', 'Minute']].values,
            df_val[['Month', 'Day', 'DayOfWeek', 'Hour', 'Minute']].values,
            df_test[['Month', 'Day', 'DayOfWeek', 'Hour', 'Minute']].values
        )
    }

    joblib.dump(bundle, output_path)
    print(f"   -> 已保存: {output_path}")

    reducer_info = {
        'n_components': n_components,
        'alpha': alpha,
        'sparsity': sparsity,
        'input_dim': X_train_boruta.shape[1],
        'output_path': output_path
    }

    return output_path, reducer_info


def grid_search_spca(input_path, output_dir="processed_data"):
    """
    SPCA 网格搜索
    
    测试不同的 n_components 和 alpha 组合
    """
    print("="*80)
    print("SPCA 超参数网格搜索")
    print("="*80)
    
    # 定义搜索空间
    n_components_list = [5, 8, 10, 12, 15, 20]
    alpha_list = [0.01, 0.1, 0.5, 1.0, 2.0, 5.0]
    
    total_configs = len(n_components_list) * len(alpha_list)
    print(f"\n搜索空间:")
    print(f"  n_components: {n_components_list}")
    print(f"  alpha: {alpha_list}")
    print(f"  总配置数: {total_configs}\n")
    
    results = []
    
    for idx, (n_comp, alpha) in enumerate(itertools.product(n_components_list, alpha_list), 1):
        print(f"\n[{idx}/{total_configs}] 测试配置: n_components={n_comp}, alpha={alpha}")
        
        try:
            output_path, reducer_info = run_spca_with_params(
                input_path, 
                n_components=n_comp, 
                alpha=alpha,
                output_dir=output_dir
            )
            
            results.append({
                'n_components': n_comp,
                'alpha': alpha,
                'sparsity': reducer_info['sparsity'],
                'output_path': output_path,
                'status': 'success'
            })
            
            print(f"✓ 配置 {idx} 完成\n")
            
        except Exception as e:
            print(f"✗ 配置 {idx} 失败: {str(e)}\n")
            results.append({
                'n_components': n_comp,
                'alpha': alpha,
                'sparsity': None,
                'output_path': None,
                'status': f'failed: {str(e)}'
            })
    
    # 保存搜索结果
    results_df = pd.DataFrame(results)
    results_csv = os.path.join(output_dir, "spca_grid_search_results.csv")
    results_df.to_csv(results_csv, index=False)
    
    print("\n" + "="*80)
    print("网格搜索完成！")
    print("="*80)
    print(f"\n成功生成 {len([r for r in results if r['status'] == 'success'])} 个数据文件")
    print(f"结果已保存至: {results_csv}")
    print("\n下一步：运行 PV_part2_comparison.py 训练这些配置对应的模型")
    
    return results_df


if __name__ == "__main__":
    PV_DATA = r"D:\APredict\data\PV130MW.xlsx"
    
    # 运行网格搜索
    results = grid_search_spca(PV_DATA, output_dir="processed_data")
    
    # 打印推荐配置（基于经验规则）
    print("\n" + "="*80)
    print("推荐配置（基于理论分析）：")
    print("="*80)
    print("""
根据 SPCA 的特性，以下配置可能表现较好：

1. 中等维度 + 适度稀疏 (平衡性能和可解释性)
   - n_components=10, alpha=0.5
   - n_components=12, alpha=1.0

2. 较高维度 + 低稀疏 (接近 PCA，但保留稀疏优势)
   - n_components=15, alpha=0.1
   - n_components=20, alpha=0.01

3. 低维度 + 高稀疏 (强特征选择)
   - n_components=8, alpha=2.0
   - n_components=5, alpha=5.0

建议优先训练以上 6 个配置，观察效果后再决定是否训练全部 36 个配置。
""")
