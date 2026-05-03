"""
光伏TCN-Informer预处理器
复刻PV_part1.py的完整特征工程流程: 32原始特征 → StandardScaler → Boruta筛选28个 → PCA降维11个
"""
import numpy as np
import pandas as pd
from joblib import load
import os


class PV_Preprocessor:
    """
    光伏TCN-Informer预处理器
    负责特征工程、标准化、Boruta筛选和PCA降维
    """
    
    def __init__(self, asset_dir):
        """
        :param asset_dir: 资产目录路径 (assets/pv_tcn_informer/assets/)
        """
        self.asset_dir = asset_dir
        self._load_assets()
        
    def _load_assets(self):
        """加载预处理所需的资产"""
        bundle_path = os.path.join(self.asset_dir, "model_ready_data.pkl")
        self.bundle = load(bundle_path)
        
        self.scaler_x = self.bundle['scaler_x']      # 期望32个输入
        self.scaler_y = self.bundle['scaler_y']
        self.pca = self.bundle['pca']                 # 期望28个输入
        
        # 完整的32个原始特征列(与PV_part1.py的feature_cols一致)
        self.all_feature_cols = [
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
        
        # Boruta选中的28个特征
        self.selected_features = self.bundle['selected_features']
        
        # 计算选中特征在all_feature_cols中的索引
        self.selected_indices = [
            self.all_feature_cols.index(feat) 
            for feat in self.selected_features
        ]
        
    def transform(self, df: pd.DataFrame) -> tuple:
        """
        对历史数据进行完整的特征工程+标准化+Boruta筛选+PCA降维
        :param df: 历史数据DataFrame(至少192行,必须包含32个特征列)
        :return: (pca_features, time_features)
                 - pca_features: [192, 11] PCA降维后的特征
                 - time_features: [192, 5] 时间标记(Month, Day, DayOfWeek, Hour, Minute)
        """
        if len(df) < 192:
            raise ValueError(f"输入数据不足!需要至少192行,当前{len(df)}行")
        
        # 1. 提取最近的192行
        df_recent = df.iloc[-192:].copy()
        
        # 2. NaN值修复
        if df_recent.isnull().any().any():
            critical_cols = [col for col in self.all_feature_cols if col in df_recent.columns]
            df_recent[critical_cols] = df_recent[critical_cols].ffill().bfill()
            for col in critical_cols:
                if df_recent[col].isnull().any():
                    df_recent[col].fillna(df_recent[col].mean(), inplace=True)
        
        # 3. 提取32个原始特征
        base_features_full = df_recent[self.all_feature_cols].values  # [192, 32]
        
        # 4. StandardScaler标准化
        base_features_scaled = self.scaler_x.transform(base_features_full)  # [192, 32]
        
        # 5. Boruta特征筛选: 从32个中选出28个
        base_features_selected = base_features_scaled[:, self.selected_indices]  # [192, 28]
        
        # 6. PCA降维到11维
        pca_features = self.pca.transform(base_features_selected)  # [192, 11]
        
        # 7. 提取时间特征
        time_features = self._extract_time_features(df_recent)  # [192, 5]
        
        return pca_features, time_features
    
    def transform_future_with_weather(self, future_df: pd.DataFrame) -> np.ndarray:
        """
        处理未来气象数据(有未来数据模式)
        :param future_df: 未来24步的气象数据DataFrame(必须包含32个特征列)
        :return: PCA特征数组 [24, 11]
        """
        # 1. 验证列名
        if not all(col in future_df.columns for col in self.all_feature_cols):
            missing = [col for col in self.all_feature_cols if col not in future_df.columns]
            raise ValueError(f"future_df缺少列: {missing}")
        
        # 2. 提取32个特征并标准化
        future_features_full = future_df[self.all_feature_cols].values  # [24, 32]
        future_features_scaled = self.scaler_x.transform(future_features_full)  # [24, 32]
        
        # 3. Boruta筛选
        future_features_selected = future_features_scaled[:, self.selected_indices]  # [24, 28]
        
        # 4. PCA降维
        future_pca = self.pca.transform(future_features_selected)  # [24, 11]
        
        return future_pca
    
    def approximate_future_without_weather(self, df_recent: pd.DataFrame) -> np.ndarray:
        """
        用历史数据近似未来24步特征(无未来数据模式)
        :param df_recent: 最近192步的历史数据(已包含32个特征)
        :return: 近似的PCA特征数组 [24, 11]
        """
        # 取最后4步的平均值作为未来24步的近似
        last_n_steps = 4
        approx_features_full = df_recent[self.all_feature_cols].iloc[-last_n_steps:].mean().values.reshape(1, -1)
        
        # 复制24次
        approx_features_repeated = np.repeat(approx_features_full, 24, axis=0)  # [24, 32]
        
        # 标准化
        approx_scaled = self.scaler_x.transform(approx_features_repeated)  # [24, 32]
        
        # Boruta筛选
        approx_selected = approx_scaled[:, self.selected_indices]  # [24, 28]
        
        # PCA降维
        approx_pca = self.pca.transform(approx_selected)  # [24, 11]
        
        return approx_pca
    
    def generate_future_time_features(self, last_timestamp: pd.Timestamp) -> np.ndarray:
        """
        生成未来24步的时间特征
        :param last_timestamp: 最后一个历史时间点
        :return: 时间特征数组 [24, 5]
        """
        future_times = pd.date_range(
            start=last_timestamp + pd.Timedelta(minutes=15),
            periods=24,
            freq='15min'
        )
        
        temp_df = pd.DataFrame(index=future_times)
        
        month = (temp_df.index.month - 1) / 11.0 - 0.5
        day = (temp_df.index.day - 1) / 30.0 - 0.5
        dow = temp_df.index.dayofweek / 6.0 - 0.5
        hour = temp_df.index.hour / 23.0 - 0.5
        minute = (temp_df.index.minute // 15) / 3.0 - 0.5
        
        time_feats = np.column_stack([month.values, day.values, dow.values, hour.values, minute.values])
        return time_feats
    
    def _extract_time_features(self, df: pd.DataFrame) -> np.ndarray:
        """
        提取时间特征(Month, Day, DayOfWeek, Hour, Minute)
        严格映射到[-0.5, 0.5]区间(与PV_part1.py一致)
        """
        if not isinstance(df.index, pd.DatetimeIndex):
            time_col_candidates = ['Time', '时间', 'timestamp', 'datetime']
            for col_name in time_col_candidates:
                if col_name in df.columns:
                    df = df.copy()
                    df[col_name] = pd.to_datetime(df[col_name])
                    df.set_index(col_name, inplace=True)
                    break
            else:
                raise ValueError("未找到时间列,无法提取时间特征")
        
        month = (df.index.month - 1) / 11.0 - 0.5
        day = (df.index.day - 1) / 30.0 - 0.5
        dow = df.index.dayofweek / 6.0 - 0.5
        hour = df.index.hour / 23.0 - 0.5
        minute = (df.index.minute // 15) / 3.0 - 0.5
        
        time_feats = np.column_stack([month.values, day.values, dow.values, hour.values, minute.values])
        return time_feats
