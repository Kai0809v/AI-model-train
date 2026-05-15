"""
数据加载器模块
负责文件读取、格式识别、数据清洗和验证
完全独立于GUI，可单独测试和复用
"""

import os
import pandas as pd
from typing import Optional, Tuple, List
from gui_config import (
    SUPPORTED_FILE_FORMATS,
    PV_POWER_COLUMNS,
    WIND_POWER_COLUMNS,
    TIME_COLUMNS
)


class DataLoader:
    """
    统一数据加载器
    支持CSV和Excel格式，提供数据清洗和验证功能
    """
    
    def __init__(self):
        self.supported_extensions = list(SUPPORTED_FILE_FORMATS.keys())
    
    def load_file(self, file_path: str) -> pd.DataFrame:
        """
        智能加载数据文件（支持CSV和Excel）
        
        Args:
            file_path: 文件路径
            
        Returns:
            DataFrame对象
            
        Raises:
            FileNotFoundError: 文件不存在
            ValueError: 不支持的文件格式
            Exception: 文件读取失败
        """
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"文件不存在: {file_path}")
        
        # 识别文件格式
        file_ext = os.path.splitext(file_path)[1].lower()
        
        if file_ext not in self.supported_extensions:
            raise ValueError(
                f"不支持的文件格式: {file_ext}。"
                f"仅支持: {', '.join(self.supported_extensions)}"
            )
        
        try:
            # 根据格式选择读取方法
            if file_ext == '.csv':
                df = pd.read_csv(file_path)
            elif file_ext in ['.xlsx', '.xls']:
                engine = 'openpyxl' if file_ext == '.xlsx' else None
                df = pd.read_excel(file_path, engine=engine)
            else:
                raise ValueError(f"未处理的文件格式: {file_ext}")
            
            return df
            
        except Exception as e:
            raise Exception(f"文件读取失败: {str(e)}")
    
    def clean_data(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, int]:
        """
        数据清洗：删除空白行、处理缺失值
        
        Args:
            df: 原始DataFrame
            
        Returns:
            (清洗后的DataFrame, 删除的行数)
        """
        original_rows = len(df)
        
        # 1. 删除全是NaN的空白行
        df_cleaned = df.dropna(how='all').reset_index(drop=True)
        cleaned_rows = len(df_cleaned)
        removed_count = original_rows - cleaned_rows
        
        if removed_count > 0:
            print(f"⚠️ 检测到 {removed_count} 行空白数据，已自动清除")
        
        # 2. 检查并修复部分缺失值的列
        nan_summary = df_cleaned.isnull().sum()
        cols_with_nan = nan_summary[nan_summary > 0]
        
        if len(cols_with_nan) > 0:
            print(f"⚠️ 以下列存在缺失值，将进行修复:\n{cols_with_nan.to_string()}")
            
            # 关键特征列列表
            critical_cols = [
                '测风塔 10m 风速 (m/s)', '测风塔 30m 风速 (m/s)', 
                '测风塔 50m 风速 (m/s)', '测风塔 70m 风速 (m/s)',
                '轮毂高度风速 (m/s)', '测风塔 10m 风向 (°)',
                '测风塔 30m 风向 (°)', '测风塔 50m 风向 (°)', 
                '测风塔 70m 风向 (°)', '轮毂高度风向 (°)',
                '温度 (°)', '气压 (hPa)', '湿度 (%)',
                '实际发电功率（mw）',
                # 光伏特征列
                'TSI', 'DNI', 'GHI', 'Temp', 'Atmosphere', 'Humidity',
            ]
            
            available_cols = [col for col in critical_cols if col in df_cleaned.columns]
            
            if available_cols:
                # 前向填充 + 后向填充
                df_cleaned[available_cols] = df_cleaned[available_cols].ffill().bfill()
                
                # 如果还有NaN，用均值填补
                for col in available_cols:
                    if df_cleaned[col].isnull().any():
                        mean_val = df_cleaned[col].mean()
                        df_cleaned[col].fillna(mean_val, inplace=True)
        
        return df_cleaned, removed_count
    
    def validate_data(self, df: pd.DataFrame, model_name: str, 
                     min_rows: int = 96) -> Tuple[bool, str]:
        """
        验证数据是否满足模型要求
        
        Args:
            df: DataFrame对象
            model_name: 模型名称
            min_rows: 最小行数要求
            
        Returns:
            (是否有效, 错误消息)
        """
        if df is None or len(df) == 0:
            return False, "数据为空"
        
        if len(df) < min_rows:
            return False, (
                f"❌ 数据量不足！当前有 {len(df)} 行有效数据，"
                f"{model_name} 模型需要至少 {min_rows} 行。"
            )
        
        return True, "数据验证通过"
    
    def find_power_column(self, df: pd.DataFrame, scenario: str) -> Optional[str]:
        """
        智能查找功率列名
        
        Args:
            df: DataFrame对象
            scenario: 场景类型（"光伏"或"风电"）
            
        Returns:
            功率列名，未找到返回None
        """
        if "光伏" in scenario:
            possible_cols = PV_POWER_COLUMNS
        else:
            possible_cols = WIND_POWER_COLUMNS
        
        for col in possible_cols:
            if col in df.columns:
                return col
        
        return None
    
    def find_time_column(self, df: pd.DataFrame) -> Optional[str]:
        """
        智能查找时间列名
        
        Args:
            df: DataFrame对象
            
        Returns:
            时间列名，未找到返回None
        """
        # 精确匹配
        for col in TIME_COLUMNS:
            if col in df.columns:
                return col
        
        # 模糊匹配
        for col in df.columns:
            if '时间' in col or 'time' in col.lower() or 'date' in col.lower():
                return col
        
        return None
    
    def get_file_extension(self, file_path: str) -> str:
        """获取文件扩展名（小写）"""
        return os.path.splitext(file_path)[1].lower()
    
    def get_file_format_description(self, file_path: str) -> str:
        """获取文件格式描述（用于日志显示）"""
        ext = self.get_file_extension(file_path).upper()
        format_map = {
            '.CSV': 'CSV',
            '.XLSX': 'Excel',
            '.XLS': 'Excel'
        }
        return format_map.get(ext, ext)
    
    def load_future_weather_data(self, file_path: str) -> pd.DataFrame:
        """
        加载未来气象数据文件（光伏预测专用）
        
        🔧 关键改进：
        - 自动移除不需要的Power列
        - 提升鲁棒性，避免因为包含功率列而报错
        
        Args:
            file_path: 未来气象数据文件路径
            
        Returns:
            DataFrame对象（已移除Power列）
            
        Raises:
            FileNotFoundError: 文件不存在
            ValueError: 不支持的文件格式或缺少必要的气象特征列
        """
        # 1. 加载文件
        df = self.load_file(file_path)
        
        # 2. 自动移除Power列（如果存在）
        power_columns_to_remove = []
        for col in ['Power (MW)', 'Power']:
            if col in df.columns:
                power_columns_to_remove.append(col)
        
        if power_columns_to_remove:
            # 静默移除，不输出警告信息（这是预期行为）
            df = df.drop(columns=power_columns_to_remove)
        
        # 3. 验证是否包含必要的气象特征列（至少需要部分核心列）
        required_weather_cols = ['TSI', 'DNI', 'GHI', 'Temp', 'Atmosphere', 'Humidity']
        # 也检查原始列名格式
        raw_weather_cols = [
            'Total solar irradiance (W/m2)',
            'Direct normal irradiance (W/m2)', 
            'Global horicontal irradiance (W/m2)',
            'Air temperature  (°C)',
            'Atmosphere (hpa)',
            'Relative humidity (%)'
        ]
        
        # 检查是否有任意一种格式的列
        has_short_names = any(col in df.columns for col in required_weather_cols)
        has_raw_names = any(col in df.columns for col in raw_weather_cols)
        
        if not has_short_names and not has_raw_names:
            raise ValueError(
                f"未来气象数据缺少必要的气象特征列！\n"
                f"可用列: {list(df.columns)}\n"
                f"需要包含以下任一格式的列:\n"
                f"  - 短名格式: TSI, DNI, GHI, Temp, Atmosphere, Humidity\n"
                f"  - 原始格式: Total solar irradiance, Direct normal irradiance, etc."
            )
        
        return df
