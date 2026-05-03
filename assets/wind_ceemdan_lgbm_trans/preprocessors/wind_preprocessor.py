"""
风电CEEMDAN-LGBM-Transformer预处理器
复刻api_v6.py中CEEMDAN_LGBM_Transformer_Predictor._preprocess()的逻辑
14个基础特征 → 49个衍生特征 → LightGBM特征选择 → 标准化
"""
import numpy as np
import pandas as pd
import torch
from joblib import load
import os


class Wind_Preprocessor:
    """
    风电CEEMDAN-LGBM-Transformer预处理器
    负责物理特征衍生、标准化和LightGBM特征选择
    """
    
    def __init__(self, asset_dir):
        """
        :param asset_dir: 资产目录路径 (assets/wind_ceemdan_lgbm_trans/assets/)
        """
        self.asset_dir = asset_dir
        self.window_size = 96  # 24小时历史窗口
        self._load_assets()
        
    def _load_assets(self):
        """加载预处理所需的资产"""
        # 加载scaler_x, scaler_y
        scaler_x_path = os.path.join(self.asset_dir, "scaler_x")
        scaler_y_path = os.path.join(self.asset_dir, "scaler_y")
        self.scaler_x = load(scaler_x_path)
        self.scaler_y = load(scaler_y_path)
        
        # 加载LightGBM选中的特征索引
        selected_indices_path = os.path.join(self.asset_dir, "selected_features_indices.npy")
        self.selected_features = np.load(selected_indices_path)
        
        # 预期的基础列名顺序 (需与GUI传入的DataFrame保持一致)
        self.expected_cols = [
            '测风塔10m风速(m/s)', '测风塔30m风速(m/s)', '测风塔50m风速(m/s)',
            '测风塔70m风速(m/s)', '轮毂高度风速(m/s)', '测风塔10m风向(°)',
            '测风塔30m风向(°)', '测风塔50m风向(°)', '测风塔70m风向(°)',
            '轮毂高度风向(°)', '温度(°)', '气压(hPa)', '湿度(%)',
            '实际发电功率（mw）'  # 历史功率必须紧跟在最后
        ]
        
    def transform(self, df: pd.DataFrame) -> torch.Tensor:
        """
        对历史数据进行完整的特征工程+标准化+LightGBM特征选择
        :param df: 历史数据DataFrame(至少96行,必须包含14个基础特征列)
        :return: tensor_x [1, 96, selected_dim]
        """
        if len(df) < self.window_size:
            raise ValueError(f"输入数据长度不足!需要至少{self.window_size}行历史数据,当前仅有{len(df)}行。")
        
        # 🔧 智能处理:如果DataFrame没有DatetimeIndex,尝试自动转换
        if not isinstance(df.index, pd.DatetimeIndex):
            time_col_candidates = ['时间', 'timestamp', 'datetime', '日期', 'time']
            found_time_col = None
            
            for col_name in time_col_candidates:
                if col_name in df.columns:
                    found_time_col = col_name
                    break
            
            if found_time_col:
                df = df.copy()
                df[found_time_col] = pd.to_datetime(df[found_time_col])
                df.set_index(found_time_col, inplace=True)
            else:
                print(f"⚠️ 警告:未找到时间列,将使用索引推算时间特征")
        
        # 🔧 NaN值检测与修复
        if df.isnull().any().any():
            print(f"⚠️ 检测到NaN值,正在进行填充修复...")
            df = df.copy()
            
            critical_cols = [col for col in self.expected_cols if col in df.columns]
            df[critical_cols] = df[critical_cols].ffill().bfill()
            
            for col in critical_cols:
                if df[col].isnull().any():
                    df[col].fillna(df[col].mean(), inplace=True)
            
            print(f"✓ NaN值修复完成,剩余NaN数量:{df.isnull().sum().sum()}")
        
        # 截取最近的96步数据
        df_recent = df.iloc[-self.window_size:].copy()
        
        # 提取基础数据(14维)
        base_data = df_recent[self.expected_cols].values
        
        feats = []
        # 1-4. 风速相关统计
        avg_wind = np.mean(base_data[:, :5], axis=1, keepdims=True)
        feats.append(avg_wind)
        feats.append(base_data[:, :5] ** 3)  # wind_cubed
        feats.append(np.std(base_data[:, :5], axis=1, keepdims=True))  # wind_std
        feats.append(np.max(base_data[:, :5], axis=1, keepdims=True))  # wind_max
        
        # 5. 风向余弦均值
        direction_cols = base_data[:, 5:10]
        direction_cos = np.cos(np.deg2rad(direction_cols))
        feats.append(np.mean(direction_cos, axis=1, keepdims=True))
        
        # 6-7. 空气密度与风能密度
        temp = base_data[:, 10]
        pressure = base_data[:, 11]
        air_density_factor = (pressure / 1013.25) * (288.15 / (temp + 273.15))
        feats.append(air_density_factor.reshape(-1, 1))
        
        wind_power_density = 0.5 * air_density_factor * (avg_wind ** 3).flatten()
        feats.append(wind_power_density.reshape(-1, 1))
        
        # 8. 时间周期特征(智能降级策略)
        data_index = np.arange(len(df) - self.window_size, len(df))
        
        if isinstance(df_recent.index, pd.DatetimeIndex):
            # 精准模式:从时间戳提取
            hour_of_day = df_recent.index.hour + df_recent.index.minute / 60.0
            day_of_week = df_recent.index.dayofweek
            
            time_sin = np.sin(2 * np.pi * hour_of_day / 24).values.reshape(-1, 1)
            time_cos = np.cos(2 * np.pi * hour_of_day / 24).values.reshape(-1, 1)
            dow_sin = np.sin(2 * np.pi * day_of_week / 7).values.reshape(-1, 1)
            dow_cos = np.cos(2 * np.pi * day_of_week / 7).values.reshape(-1, 1)
        else:
            # 降级模式:假设数据从某天00:00开始,每15分钟一个数据点
            hour_of_day = (data_index % 96) / 96 * 24
            day_of_week = (data_index // 96) % 7
            
            time_sin = np.sin(2 * np.pi * hour_of_day / 24).reshape(-1, 1)
            time_cos = np.cos(2 * np.pi * hour_of_day / 24).reshape(-1, 1)
            dow_sin = np.sin(2 * np.pi * day_of_week / 7).reshape(-1, 1)
            dow_cos = np.cos(2 * np.pi * day_of_week / 7).reshape(-1, 1)
        
        feats.extend([time_sin, time_cos, dow_sin, dow_cos])
        
        # 功率滞后特征
        power_col = base_data[:, -1]
        lags = [1, 2, 4, 8, 12, 24, 48]
        for lag in lags:
            lagged = np.roll(power_col, lag)
            lagged[:lag] = power_col[lag-1] if lag > 0 else power_col[0]
            feats.append(lagged.reshape(-1, 1))
        
        # 功率变化率(一阶差分)
        power_diff = np.zeros_like(power_col)
        power_diff[1:] = np.diff(power_col)
        power_diff[0] = 0
        feats.append(power_diff.reshape(-1, 1))
        
        # 功率移动平均
        for window in [4, 12, 24]:
            ma = np.zeros_like(power_col)
            for i in range(len(power_col)):
                start_idx = max(0, i - window + 1)
                ma[i] = np.mean(power_col[start_idx:i+1])
            feats.append(ma.reshape(-1, 1))
        
        # 拼接全部特征(14 + 35 = 49维)
        full_features = np.hstack([base_data] + feats)
        
        # 归一化
        full_features_scaled = self.scaler_x.transform(full_features)
        
        # LightGBM特征选择
        selected_features_scaled = full_features_scaled[:, self.selected_features]
        
        # 转换为PyTorch张量:形状(Batch=1, Seq=96, Features)
        tensor_x = torch.tensor(selected_features_scaled).unsqueeze(0).float()
        return tensor_x
    
    def check_wind_speed_threshold(self, df: pd.DataFrame) -> tuple:
        """
        检查切入风速阈值(3.0 m/s)
        :param df: 输入DataFrame
        :return: (should_block, current_wind_speed)
                 - should_block: True表示风速过低,应该锁零
                 - current_wind_speed: 当前风速值
        """
        current_wind_speed = df['轮毂高度风速(m/s)'].iloc[-1]
        should_block = current_wind_speed < 3.0
        return should_block, current_wind_speed
