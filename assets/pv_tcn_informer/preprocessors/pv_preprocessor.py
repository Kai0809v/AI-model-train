"""
光伏TCN-Informer预处理器
复刻PV_part1.py的完整特征工程流程: 原始数据 → 列名映射 → 交互特征 → 滞后/滚动特征 → 32原始特征 → StandardScaler → Boruta筛选28个 → PCA降维11个

支持两种输入格式:
1. 原始数据(7+1列): 包含原始列名如 'Total solar irradiance (W/m2)', 'Power (MW)' 等
2. 已处理数据(32列): 包含短名如 'TSI', 'Power_lag_4' 等
"""
import numpy as np
import pandas as pd
from joblib import load
import os


class PV_Preprocessor:
    """
    光伏TCN-Informer预处理器
    负责列名映射、特征工程、标准化、Boruta筛选和PCA降维
    """
    
    # 原始列名 → 短名 映射 (与PV_part1.py一致)
    COL_MAPPING = {
        'Time(year-month-day h:m:s)': 'Time',
        'Total solar irradiance (W/m2)': 'TSI',
        'Direct normal irradiance (W/m2)': 'DNI',
        'Global horicontal irradiance (W/m2)': 'GHI',
        'Air temperature  (°C) ': 'Temp',
        'Air temperature  (°C)': 'Temp',
        'Atmosphere (hpa)': 'Atmosphere',
        'Relative humidity (%)': 'Humidity',
        'Power (MW)': 'Power',
    }
    
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
        
    def _is_raw_data(self, df: pd.DataFrame) -> bool:
        """判断DataFrame是否为原始数据(需要列名映射和特征工程)"""
        # 如果已经有32个特征列中的大部分，说明已经处理过了
        existing_feature_cols = [c for c in self.all_feature_cols if c in df.columns]
        if len(existing_feature_cols) >= 20:
            return False  # 已经处理过
        # 检查是否有原始列名
        raw_cols = set(self.COL_MAPPING.keys())
        df_cols = set(df.columns.astype(str))
        return len(raw_cols & df_cols) >= 3  # 至少有3个原始列名
    
    def _normalize_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        将原始数据转换为包含32个特征列的标准格式
        1. 列名映射 (原始长名 → 短名)
        2. 时间列解析
        3. 构建交互特征 (6个)
        4. 构建温度修正特征 (2个)
        5. 构建滞后特征 (6个)
        6. 构建滚动统计特征 (12个)
        7. 填充NaN值
        """
        df_out = df.copy()
        
        # 1. 列名映射
        # 匹配原始列名(注意可能有额外空格)
        import re
        rename_map = {}
        for raw_col in df_out.columns:
            raw_col_stripped = raw_col.strip()
            
            # 🔧 精确匹配
            if raw_col_stripped in self.COL_MAPPING:
                rename_map[raw_col] = self.COL_MAPPING[raw_col_stripped]
                continue
            
            # 🔧 模糊匹配：将多个连续空格替换为单个空格后再匹配
            normalized_col = re.sub(r'\s+', ' ', raw_col_stripped)
            if normalized_col in self.COL_MAPPING:
                rename_map[raw_col] = self.COL_MAPPING[normalized_col]
                continue
            
            # 🔧 更宽松的匹配：忽略所有空格后匹配核心关键词
            # 例如: "Air temperature  (°C)" -> "Airtemperature(°C)"
            no_space_col = re.sub(r'\s+', '', raw_col_stripped)
            for mapping_key, short_name in self.COL_MAPPING.items():
                no_space_key = re.sub(r'\s+', '', mapping_key)
                if no_space_col == no_space_key:
                    rename_map[raw_col] = short_name
                    break
        
        if rename_map:
            df_out.rename(columns=rename_map, inplace=True)
        
        # 2. 解析时间列并设为索引
        time_col = None
        for col_name in ['Time', 'time', 'timestamp', 'datetime']:
            if col_name in df_out.columns:
                time_col = col_name
                break
        
        if time_col is not None:
            df_out[time_col] = pd.to_datetime(df_out[time_col])
            if not isinstance(df_out.index, pd.DatetimeIndex):
                df_out.set_index(time_col, inplace=True)
        
        # 3. 构建交互特征 (与PV_part1.py一致)
        if 'TSI_Temp_interaction' not in df_out.columns:
            df_out['TSI_Temp_interaction'] = df_out['TSI'] * df_out['Temp']
        if 'GHI_Temp_interaction' not in df_out.columns:
            df_out['GHI_Temp_interaction'] = df_out['GHI'] * df_out['Temp']
        if 'TSI_Humidity_ratio' not in df_out.columns:
            df_out['TSI_Humidity_ratio'] = df_out['TSI'] / (df_out['Humidity'] + 1e-6)
        if 'GHI_Humidity_ratio' not in df_out.columns:
            df_out['GHI_Humidity_ratio'] = df_out['GHI'] / (df_out['Humidity'] + 1e-6)
        if 'DNI_GHI_ratio' not in df_out.columns:
            df_out['DNI_GHI_ratio'] = df_out['DNI'] / (df_out['GHI'] + 1e-6)
        if 'Temp_squared' not in df_out.columns:
            df_out['Temp_squared'] = df_out['Temp'] ** 2
        
        # 4. 温度修正特征 (与PV_part1.py一致)
        if 'TSI_Corrected' not in df_out.columns:
            df_out['TSI_Corrected'] = df_out['TSI'] * (1 - 0.004 * (df_out['Temp'] - 25))
        if 'GHI_Corrected' not in df_out.columns:
            df_out['GHI_Corrected'] = df_out['GHI'] * (1 - 0.004 * (df_out['Temp'] - 25))
        
        # 5. 滞后特征 (与PV_part1.py一致)
        for lag in [4, 12, 24]:
            col_name = f'Power_lag_{lag}'
            if col_name not in df_out.columns:
                df_out[col_name] = df_out['Power'].shift(lag)
        for lag in [4]:
            for feat in ['TSI', 'DNI', 'GHI']:
                col_name = f'{feat}_lag_{lag}'
                if col_name not in df_out.columns:
                    df_out[col_name] = df_out[feat].shift(lag)
        
        # 6. 滚动统计特征 (与PV_part1.py一致)
        for window in [12, 48, 96]:
            col_name = f'Power_rolling_mean_{window}'
            if col_name not in df_out.columns:
                df_out[col_name] = df_out['Power'].rolling(window=window).mean()
            col_name = f'Power_rolling_std_{window}'
            if col_name not in df_out.columns:
                df_out[col_name] = df_out['Power'].rolling(window=window).std()
            for feat in ['TSI', 'GHI']:
                col_name = f'{feat}_rolling_mean_{window}'
                if col_name not in df_out.columns:
                    df_out[col_name] = df_out[feat].rolling(window=window).mean()
        
        # 7. 填充NaN值 (滞后和滚动特征会在开头产生NaN)
        df_out.ffill(inplace=True)
        df_out.bfill(inplace=True)
        
        # 最终检查: 确保所有32个特征列都存在
        missing_cols = [c for c in self.all_feature_cols if c not in df_out.columns]
        if missing_cols:
            raise ValueError(f"特征工程后仍缺少列: {missing_cols}")
        
        return df_out
    
    def transform(self, df: pd.DataFrame) -> tuple:
        """
        对历史数据进行完整的特征工程+标准化+Boruta筛选+PCA降维
        
        支持两种输入格式:
        - 原始数据(7+1列): 包含原始列名如 'Total solar irradiance (W/m2)' 等
        - 已处理数据(32列): 包含短名如 'TSI', 'Power_lag_4' 等
        
        :param df: 历史数据DataFrame(至少192行原始数据)
        :return: (pca_features, time_features, last_timestamp)
                 - pca_features: [192, 11] PCA降维后的特征
                 - time_features: [192, 5] 时间标记(Month, Day, DayOfWeek, Hour, Minute)
                 - last_timestamp: 最后一个时间点(pd.Timestamp), 用于生成未来时间特征
        """
        # 0. 如果是原始数据，先进行列名映射和特征工程
        if self._is_raw_data(df):
            df = self._normalize_dataframe(df)
        
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
        
        # 8. 获取最后的时间戳
        last_timestamp = self._get_last_timestamp(df_recent)
        
        return pca_features, time_features, last_timestamp
    
    def transform_future_with_weather(self, future_df: pd.DataFrame) -> np.ndarray:
        """
        处理未来气象数据(有未来数据模式)
        
        支持两种输入格式:
        - 原始数据(7+1列): 包含原始列名
        - 已处理数据(32列): 包含短名
        
        :param future_df: 未来24步的气象数据DataFrame(至少24行)
        :return: PCA特征数组 [24, 11]
        """
        # 0. 如果是原始数据，先进行列名映射和特征工程
        if self._is_raw_data(future_df):
            future_df = self._normalize_dataframe(future_df)
        
        # 🔧 0.5. 自动调整行数到24行（避免序列长度不匹配）
        if len(future_df) != 24:
            print(f"⚠️  警告: future_df有{len(future_df)}行，自动调整为24行")
            if len(future_df) > 24:
                # 如果超过24行，取最后24行（最新的预测）
                future_df = future_df.iloc[-24:].copy()
                print(f"   → 截取最后24行")
            else:
                # 如果不足24行，用最后一行重复填充
                last_row = future_df.iloc[-1:].copy()
                repeat_count = 24 - len(future_df)
                future_df = pd.concat([future_df] + [last_row] * repeat_count, ignore_index=True)
                print(f"   → 用最后一行重复填充{repeat_count}次")
        
        # 1. 验证列名
        if not all(col in future_df.columns for col in self.all_feature_cols):
            missing = [col for col in self.all_feature_cols if col not in future_df.columns]
            raise ValueError(f"future_df缺少列: {missing}")
        
        # 🔧 2. NaN值修复（关键：必须在标准化前处理）
        critical_cols = [col for col in self.all_feature_cols if col in future_df.columns]
        
        # 第一步：前向填充 + 后向填充
        future_df[critical_cols] = future_df[critical_cols].ffill().bfill()
        
        # 第二步：对仍有NaN的列用均值填充（避免inplace=True的Copy-on-Write问题）
        for col in critical_cols:
            if future_df[col].isnull().any():
                col_mean = future_df[col].mean()
                # 如果均值也是NaN（整列都是NaN），用0填充
                if pd.isna(col_mean):
                    col_mean = 0.0
                future_df[col] = future_df[col].fillna(col_mean)
        
        # 🔧 第三步：最终验证，确保没有NaN
        if future_df[critical_cols].isnull().any().any():
            raise ValueError(f"NaN值修复失败，仍包含NaN的列: {[col for col in critical_cols if future_df[col].isnull().any()]}")
        
        # 3. 提取32个特征并标准化
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
        
        支持两种输入格式:
        - 原始数据: 会先进行列名映射和特征工程
        - 已处理数据: 直接使用
        
        :param df_recent: 最近192步的历史数据
        :return: 近似的PCA特征数组 [24, 11]
        """
        # 0. 如果是原始数据，先进行列名映射和特征工程
        if self._is_raw_data(df_recent):
            df_recent = self._normalize_dataframe(df_recent)
        
        # 🔧 NaN值修复（确保计算平均值时没有NaN）
        critical_cols = [col for col in self.all_feature_cols if col in df_recent.columns]
        
        # 第一步：前向填充 + 后向填充
        df_recent[critical_cols] = df_recent[critical_cols].ffill().bfill()
        
        # 第二步：对仍有NaN的列用均值填充（避免inplace=True的Copy-on-Write问题）
        for col in critical_cols:
            if df_recent[col].isnull().any():
                col_mean = df_recent[col].mean()
                # 如果均值也是NaN（整列都是NaN），用0填充
                if pd.isna(col_mean):
                    col_mean = 0.0
                df_recent[col] = df_recent[col].fillna(col_mean)
        
        # 🔧 第三步：最终验证，确保没有NaN
        if df_recent[critical_cols].isnull().any().any():
            raise ValueError(f"NaN值修复失败，仍包含NaN的列: {[col for col in critical_cols if df_recent[col].isnull().any()]}")
        
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
    
    def _get_last_timestamp(self, df: pd.DataFrame) -> pd.Timestamp:
        """从DataFrame中获取最后一个时间戳"""
        if isinstance(df.index, pd.DatetimeIndex):
            return df.index[-1]
        # 尝试从列中查找时间列
        for col_name in ['Time', '时间', 'timestamp', 'datetime']:
            if col_name in df.columns:
                return pd.to_datetime(df[col_name].iloc[-1])
        raise ValueError("未找到时间列,无法获取最后时间戳")
    
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
