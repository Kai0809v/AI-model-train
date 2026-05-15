"""
预测控制器模块
负责业务逻辑编排、数据验证和错误处理
作为GUI和API之间的中间层，实现解耦
"""

import re
from typing import Dict, Any, Optional
import pandas as pd
from api_v8 import ForecastService
from data_loader_module import DataLoader
from gui_config import PREDICTION_SCENARIOS


class PredictionController:
    """
    预测业务控制器
    封装完整的预测流程：加载 → 清洗 → 验证 → 预测
    """
    
    def __init__(self):
        self.forecast_service = ForecastService()
        self.data_loader = DataLoader()
    
    def get_available_scenarios(self) -> list:
        """获取所有可用的预测场景"""
        return list(PREDICTION_SCENARIOS.keys())
    
    def get_models_for_scenario(self, scenario: str) -> Dict[str, str]:
        """
        获取指定场景的可用模型列表
        
        Args:
            scenario: 场景名称
            
        Returns:
            {显示名称: 后端名称} 字典
        """
        if scenario not in PREDICTION_SCENARIOS:
            return {}
        return PREDICTION_SCENARIOS[scenario]["models"]
    
    def get_steps_options_for_scenario(self, scenario: str) -> list:
        """
        获取指定场景的步长选项
        
        Args:
            scenario: 场景名称
            
        Returns:
            步长选项列表
        """
        if scenario not in PREDICTION_SCENARIOS:
            return []
        return PREDICTION_SCENARIOS[scenario]["steps_options"]
    
    def parse_steps_from_text(self, steps_text: str) -> int:
        """
        从步长文本中解析步数
        
        Args:
            steps_text: 步长描述文本，如 "一小时（4 步）"
            
        Returns:
            步数值
        """
        # 尝试从文本中提取数字，格式如 "（4 步）"
        match = re.search(r'（(\d+)\s*步）', steps_text)
        if match:
            return int(match.group(1))
        
        # 兼容其他格式
        if "单步" in steps_text:
            return 1
        
        # 默认返回1
        return 1
    
    def requires_future_weather(self, model_display_name: str) -> bool:
        """
        判断模型是否需要未来气象数据
        
        Args:
            model_display_name: 模型显示名称
            
        Returns:
            是否需要未来气象数据
        """
        return "有未来气象数据" in model_display_name
    
    def execute_prediction(self, 
                          file_path: str,
                          model_backend_name: str,
                          steps: int,
                          mode: str = "auto",
                          future_weather_path: Optional[str] = None) -> Dict[str, Any]:
        """
        执行完整的预测流程
        
        Args:
            file_path: 历史数据文件路径
            model_backend_name: 模型后端名称
            steps: 预测步长
            mode: 预测模式（with_future/without_future/auto）
            future_weather_path: 未来气象数据文件路径（可选）
            
        Returns:
            预测结果字典，包含success标志和数据
        """
        try:
            # 第1步：加载数据
            df = self.data_loader.load_file(file_path)
            
            # 第2步：清洗数据
            df_cleaned, removed_count = self.data_loader.clean_data(df)
            
            # 第3步：验证数据
            min_rows = self._get_min_rows_for_model(model_backend_name)
            is_valid, error_msg = self.data_loader.validate_data(
                df_cleaned, model_backend_name, min_rows
            )
            
            if not is_valid:
                return {
                    "success": False,
                    "error": error_msg
                }
            
            # 第4步：构建额外参数
            kwargs = {}
            
            # 光伏模型需要mode参数
            if model_backend_name == "PV_TCN_Informer":
                kwargs["mode"] = mode
                
                # 如果需要未来气象数据，加载并传入
                if mode == "with_future" and future_weather_path:
                    # 🔧 使用专用的未来气象数据加载方法（自动处理Power列）
                    future_df = self.data_loader.load_future_weather_data(future_weather_path)
                    kwargs["future_weather_df"] = future_df
            
            # 第5步：执行预测
            result = self.forecast_service.run(
                model_backend_name, 
                df_cleaned, 
                steps, 
                **kwargs
            )
            
            # 第6步：添加元信息
            if result.get("success"):
                result["model_name"] = model_backend_name
                result["steps"] = steps
                result["data_rows_used"] = len(df_cleaned)
            
            return result
            
        except FileNotFoundError as e:
            return {"success": False, "error": str(e)}
        except ValueError as e:
            return {"success": False, "error": str(e)}
        except Exception as e:
            import traceback
            traceback.print_exc()
            return {"success": False, "error": f"预测过程中发生严重系统异常：{str(e)}"}
    
    def _get_min_rows_for_model(self, model_name: str) -> int:
        """
        获取模型所需的最小数据行数
        
        Args:
            model_name: 模型名称
            
        Returns:
            最小行数
        """
        # 光伏模型需要192行，风电需要96行
        if model_name in ["PV_TCN_Informer", "PV_TCN_Informer_NoWeather"]:
            return 192
        else:
            return 96
    
    def validate_future_weather_file(self, file_path: str) -> tuple:
        """
        验证未来气象数据文件
        
        Args:
            file_path: 文件路径
            
        Returns:
            (是否有效, 错误消息)
        """
        if not file_path:
            return False, "未提供未来气象数据文件"
        
        ext = self.data_loader.get_file_extension(file_path)
        if ext not in ['.csv', '.xlsx', '.xls']:
            return False, f"未来气象数据文件格式不支持: {ext}。仅支持 .csv 和 .xlsx 格式。"
        
        return True, "验证通过"
