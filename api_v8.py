# ============================================
# 统一预测接口（可直接对接 GUI）
# TODO：在GUI选择有缺失的数据预测时会有报错
# ============================================

import math
import os
import sys

import pandas as pd
import torch
import torch.nn as nn

# 修复Informer2020的导入路径问题
# 将Informer2020目录添加到Python路径中
informer_path = os.path.join(os.path.dirname(__file__), 'Informer2020')
if informer_path not in sys.path:
    sys.path.insert(0, informer_path)

# 导入光伏预测相关模块

# ============================================
# 资源路径处理工具（解决打包后路径失效问题）
# ============================================
def resource_path(relative_path):
    """获取资源的绝对路径，兼容开发环境和 PyInstaller 打包环境"""
    if hasattr(sys, '_MEIPASS'):
        return os.path.join(sys._MEIPASS, relative_path)
    return os.path.join(os.path.abspath("."), relative_path)

# 🔧 修复 PyTorch 2.6 的 weights_only 安全限制
# 注意：np._core.multiarray.scalar 不能直接传递给 add_safe_globals
# 如果需要加载模型，会在模型加载时动态处理
try:
    import numpy as np
    if hasattr(np._core, 'multiarray') and hasattr(np._core.multiarray, 'scalar'):
        torch.serialization.add_safe_globals([np._core.multiarray.scalar])
except Exception:
    pass


# ============================================
# 1 模型网络结构 (必须与训练时完全一致)
# ============================================
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:, :x.size(1), :]
        return x


class TransformerModel(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.embedding = nn.Linear(input_dim, 256)
        self.pos_encoder = PositionalEncoding(d_model=256)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=256, nhead=8, dim_feedforward=512,
            batch_first=True, dropout=0.1
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=4)

        self.attention_pooling = nn.Sequential(
            nn.Linear(256, 128),
            nn.Tanh(),
            nn.Linear(128, 1)
        )

        self.fc = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 1)
        )

    def forward(self, x, return_attention=False):
        x = self.embedding(x)
        x = self.pos_encoder(x)
        x = self.transformer(x)

        attention_weights = self.attention_pooling(x)
        attention_weights = torch.softmax(attention_weights, dim=1)
        context = torch.sum(x * attention_weights, dim=1)

        out = self.fc(context)
        return out


class SimpleMultiStepTransformer(nn.Module):
    """V6集成学习版 - 支持多步预测"""
    def __init__(self, input_dim, horizon=1):
        super().__init__()
        
        self.embedding = nn.Linear(input_dim, 256)
        self.pos_encoder = PositionalEncoding(d_model=256)
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=256, nhead=8, dim_feedforward=512,
            batch_first=True, dropout=0.1
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=4)
        
        self.attention_pooling = nn.Sequential(
            nn.Linear(256, 128),
            nn.Tanh(),
            nn.Linear(128, 1)
        )
        
        self.fc = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, horizon)
        )
    
    def forward(self, x):
        x = self.embedding(x)
        x = self.pos_encoder(x)
        x = self.transformer(x)
        
        attention_weights = self.attention_pooling(x)
        attention_weights = torch.softmax(attention_weights, dim=1)
        context = torch.sum(x * attention_weights, dim=1)
        
        output = self.fc(context)
        return output


# ============================================
# 2 专属预测器 API 封装
# ============================================
class CEEMDAN_LGBM_Transformer_Predictor:
    """
    风电CEEMDAN-LGBM-Transformer预测器(重构版)
    委托给独立的预处理器和模型包装器
    """
    def __init__(self, model_dir):
        """
        初始化风电预测器
        :param model_dir: 模型资产目录路径
        """
        from assets.wind_ceemdan_lgbm_trans.preprocessors import Wind_Preprocessor
        from assets.wind_ceemdan_lgbm_trans.models import Wind_ModelWrapper
        
        self.model_dir = model_dir
        asset_dir = os.path.join(model_dir, "assets")
        
        # 初始化预处理器
        self.preprocessor = Wind_Preprocessor(asset_dir)
        
        # 初始化模型包装器
        self.model_wrapper = Wind_ModelWrapper(asset_dir)
        
        print(f"[✓] 风电CEEMDAN-LGBM-Transformer预测器初始化成功")
    
    def predict(self, df: pd.DataFrame, steps: int = 1) -> dict:
        """
        统一预测接口:支持单步和多步预测
        :param df: 历史数据DataFrame(至少96行)
        :param steps: 预测步长(1/4/8)
        :return: 预测结果字典
        """
        try:
            # 1. 检查切入风速阈值
            should_block, current_wind_speed = self.preprocessor.check_wind_speed_threshold(df)
            if should_block:
                predictions = [0.0] * steps
                return {
                    "success": True,
                    "predictions": predictions,
                    "steps": steps,
                    "note": f"Blocked by Cut-in Wind Speed threshold ({current_wind_speed:.2f} m/s < 3.0 m/s)"
                }
            
            # 2. 预处理数据
            tensor_x = self.preprocessor.transform(df)
            
            # 3. 模型推理
            result = self.model_wrapper.predict(
                tensor_x, 
                steps=steps, 
                scaler_y=self.preprocessor.scaler_y
            )
            
            return result
            
        except Exception as e:
            import traceback
            traceback.print_exc()
            return {"success": False, "error": str(e)}


# --------------------------------------------
# TODO: 为后续算法预留的骨架，写完后在下方调度器里解开注释即可
# --------------------------------------------


class PV_TCN_Informer_Predictor:
    """
    光伏TCN-Informer预测器(重构版)
    委托给独立的预处理器和模型包装器
    """
    def __init__(self, model_dir):
        """
        初始化光伏TCN-Informer预测器
        :param model_dir: 模型资产目录路径
        """
        from assets.pv_tcn_informer.preprocessors import PV_Preprocessor
        from assets.pv_tcn_informer.models import PV_ModelWrapper
        
        self.model_dir = model_dir
        asset_dir = os.path.join(model_dir, "assets")
        
        # 初始化预处理器
        self.preprocessor = PV_Preprocessor(asset_dir)
        
        # 初始化模型包装器
        self.model_wrapper = PV_ModelWrapper(asset_dir, input_dim=11)
        
        # 加载scaler_y用于反归一化
        self.scaler_y = self.preprocessor.scaler_y
        
        print(f"[✓] 光伏TCN-Informer预测器初始化成功")
    
    def predict(self, df: pd.DataFrame, steps: int = 1, future_weather_df: pd.DataFrame = None, mode: str = "auto") -> dict:
        """
        统一预测接口
        :param df: 历史数据DataFrame(至少192行,每行15分钟,必须包含32个特征列)
        :param steps: 预测步长(1-24),每个步长代表15分钟
        :param future_weather_df: 可选,未来气象数据DataFrame(24行,32个特征列)
        :param mode: 预测模式
                   - "with_future": 必须提供future_weather_df
                   - "without_future": 不使用未来气象,用历史数据近似
                   - "auto": 如果提供了future_weather_df则使用,否则自动降级
        :return: 预测结果字典
        """
        try:
            # 验证步长范围
            if steps < 1 or steps > self.model_wrapper.max_pred_len:
                return {"success": False, "error": f"预测步长必须在1-{self.model_wrapper.max_pred_len}之间"}
            
            # 1. 预处理历史数据
            pca_history, time_history, last_timestamp = self.preprocessor.transform(df)  # [192, 11], [192, 5], Timestamp
            
            # 2. 确定解码器输入的未来部分
            if mode == "with_future":
                if future_weather_df is None:
                    raise ValueError("mode='with_future'时必须提供future_weather_df参数")
                pca_future = self.preprocessor.transform_future_with_weather(future_weather_df)
            elif mode == "without_future":
                pca_future = self.preprocessor.approximate_future_without_weather(df)
            else:  # mode == "auto"
                if future_weather_df is not None:
                    pca_future = self.preprocessor.transform_future_with_weather(future_weather_df)
                else:
                    pca_future = self.preprocessor.approximate_future_without_weather(df)
            
            # 3. 构建Informer所需的四种输入
            seq_x = torch.FloatTensor(pca_history).unsqueeze(0)  # [1, 192, 11]
            seq_x_mark = torch.FloatTensor(time_history).unsqueeze(0)  # [1, 192, 5]
            
            # 生成未来时间标记
            time_future = self.preprocessor.generate_future_time_features(last_timestamp)  # [24, 5]
            time_full = np.vstack([time_history, time_future])  # [216, 5]
            
            # 拼接PCA特征
            pca_full = np.vstack([pca_history, pca_future])  # [216, 11]
            
            # 解码器输入: 后96步历史 + 前24步未来
            dec_start_idx = 192 - 96  # 96
            dec_x = torch.FloatTensor(pca_full[dec_start_idx:]).unsqueeze(0)  # [1, 120, 11]
            dec_x_mark = torch.FloatTensor(time_full[dec_start_idx:]).unsqueeze(0)  # [1, 120, 5]
            
            # 4. 模型推理
            pred_scaled = self.model_wrapper.predict(seq_x, seq_x_mark, dec_x, dec_x_mark, steps)
            
            # 5. 反归一化
            preds_inverse = np.zeros_like(pred_scaled)
            for col in range(steps):
                # pred_scaled[:, col:col+1] 形状为 [1, 1],需要reshape为 [1, 1]
                col_data = pred_scaled[:, col].reshape(-1, 1)  # [1, 1]
                preds_inverse[:, col] = self.scaler_y.inverse_transform(col_data).flatten()
            
            # 6. 应用物理约束
            MAX_CAPACITY = 130.0  # MW
            night_mask = (preds_inverse < 0.05)
            preds_inverse[night_mask] = 0.0
            preds_inverse = np.maximum(0, preds_inverse)
            preds_inverse = np.minimum(preds_inverse, MAX_CAPACITY)
            
            predictions = preds_inverse[0].tolist()
            
            return {
                "success": True,
                "predictions": predictions,
                "steps": steps,
                "model_name": "PV_TCN_Informer"
            }
            
        except Exception as e:
            import traceback
            traceback.print_exc()
            return {"success": False, "error": str(e)}


class PV_TCN_Informer_NoWeather_Predictor:
    """
    光伏TCN-Informer预测器 - 无未来气象数据版本
    专用于没有天气预报数据的场景，使用零填充策略
    
    资产结构：
    - 预处理器: assets/pv_tcn_informer/preprocessors/pv_preprocessor_no_weather_prediction.py
    - 模型文件: assets/pv_tcn_informer/assets_no_weather/best_tcn_informer_no_weather_prediction.pth
    - Bundle文件: assets/pv_tcn_informer/assets_no_weather/preprocessor_bundle.pkl
    """
    def __init__(self, model_dir=None):
        """
        初始化无未来数据版本的光伏预测器
        :param model_dir: 模型资产目录路径 (默认: assets/pv_tcn_informer/)
        """
        # 导入无未来气象数据版本的预处理器
        from assets.pv_tcn_informer.preprocessors.pv_preprocessor_no_weather_prediction import PV_Preprocessor_NoWeather
        from assets.pv_tcn_informer.models import PV_ModelWrapper
        
        # 默认使用统一的pv_tcn_informer目录
        if model_dir is None:
            model_dir = os.path.join(os.path.dirname(__file__), "assets", "pv_tcn_informer")
        
        self.model_dir = model_dir
        
        # 无未来数据版本的资产子目录
        asset_dir = os.path.join(model_dir, "assets_no_weather")
        
        # 初始化专用预处理器（零填充策略）
        self.preprocessor = PV_Preprocessor_NoWeather(asset_dir)
        
        # 初始化模型包装器（加载无未来数据版本的模型）
        self.model_wrapper = PV_ModelWrapper(
            asset_dir, 
            input_dim=11, 
            model_filename="best_tcn_informer_no_weather_prediction.pth"
        )
        
        # 加载scaler_y用于反归一化
        self.scaler_y = self.preprocessor.scaler_y
        
        print(f"[✓] 光伏TCN-Informer无未来数据预测器初始化成功")
    
    def predict(self, df: pd.DataFrame, steps: int = 1) -> dict:
        """
        无未来数据模式的预测接口
        :param df: 历史数据DataFrame(至少192行,每行15分钟,必须包含32个特征列)
        :param steps: 预测步长(1-24),每个步长代表15分钟
        :return: 预测结果字典
        """
        try:
            # 验证步长范围
            if steps < 1 or steps > self.model_wrapper.max_pred_len:
                return {"success": False, "error": f"预测步长必须在1-{self.model_wrapper.max_pred_len}之间"}
            
            # 1. 预处理历史数据
            pca_history, time_history, last_timestamp = self.preprocessor.transform(df)  # [192, 11], [192, 5], Timestamp
            
            # 2. 构建解码器输入的未来部分（零填充）
            pca_future = self.preprocessor.approximate_future_without_weather(df, pred_len=24)  # [24, 11] 零矩阵
            
            # 3. 构建Informer所需的四种输入
            seq_x = torch.FloatTensor(pca_history).unsqueeze(0)  # [1, 192, 11]
            seq_x_mark = torch.FloatTensor(time_history).unsqueeze(0)  # [1, 192, 5]
            
            # 生成未来时间标记
            time_future = self.preprocessor.generate_future_time_features(last_timestamp, pred_len=24)  # [24, 5]
            time_full = np.vstack([time_history, time_future])  # [216, 5]
            
            # 拼接PCA特征（历史 + 零填充未来）
            pca_full = np.vstack([pca_history, pca_future])  # [216, 11]
            
            # 解码器输入: 后96步历史 + 前24步零填充
            dec_start_idx = 192 - 96  # 96
            dec_x = torch.FloatTensor(pca_full[dec_start_idx:]).unsqueeze(0)  # [1, 120, 11]
            dec_x_mark = torch.FloatTensor(time_full[dec_start_idx:]).unsqueeze(0)  # [1, 120, 5]
            
            # 4. 模型推理
            pred_scaled = self.model_wrapper.predict(seq_x, seq_x_mark, dec_x, dec_x_mark, steps)
            
            # 5. 反归一化
            preds_inverse = np.zeros_like(pred_scaled)
            for col in range(steps):
                col_data = pred_scaled[:, col].reshape(-1, 1)  # [1, 1]
                preds_inverse[:, col] = self.scaler_y.inverse_transform(col_data).flatten()
            
            # 6. 应用物理约束
            MAX_CAPACITY = 130.0  # MW
            night_mask = (preds_inverse < 0.05)
            preds_inverse[night_mask] = 0.0
            preds_inverse = np.maximum(0, preds_inverse)
            preds_inverse = np.minimum(preds_inverse, MAX_CAPACITY)
            
            predictions = preds_inverse[0].tolist()
            
            return {
                "success": True,
                "predictions": predictions,
                "steps": steps,
                "model_name": "PV_TCN_Informer_NoWeather",
                "mode": "without_future_zero_padding"
            }
            
        except Exception as e:
            import traceback
            traceback.print_exc()
            return {"success": False, "error": str(e)}


class CNN_LSTM_Attention_Predictor:
    def __init__(self, model_dir):
        # self.scaler_x = load(os.path.join(model_dir, "scaler_x"))
        pass


class Transformer_BiLSTM_Predictor:
    def __init__(self, model_dir):
        pass


class LSTM_Predictor:
    def __init__(self, model_dir):
        pass


# ============================================
# 3. 统一调度器（支持多模型动态扩展）
# ============================================
class ForecastService:
    def __init__(self, base_models_dir="assets"):
        # ★ 动态获取当前脚本所在的绝对路径，并拼接出总的仓库文件夹
        # 使用 resource_path 确保打包后能找到 assets 目录
        BASE_DIR = os.path.dirname(resource_path(__file__)) if not hasattr(sys, '_MEIPASS') else resource_path('.')
        self.base_models_dir = os.path.join(BASE_DIR, base_models_dir)

        # ★ 注册表：将 "GUI传入的名字" 映射到 -> (类名, "该模型专属的子文件夹名")
        self._model_registry = {
            "CEEMDAN_LGBM_Transformer": (CEEMDAN_LGBM_Transformer_Predictor, "wind_ceemdan_lgbm_trans"),
            "PV_TCN_Informer": (PV_TCN_Informer_Predictor, "pv_tcn_informer"),  # 有未来气象数据版本
            "PV_TCN_Informer_NoWeather": (PV_TCN_Informer_NoWeather_Predictor, "pv_tcn_informer"),  # 无未来气象数据版本（共用同一目录）

            # TODO：后续加上这几种算法
            # "CNN-LSTM-Attention": (CNN_LSTM_Attention_Predictor, "wind_cnn_lstm_att"),
            # "Transformer-BiLSTM": (Transformer_BiLSTM_Predictor, "wind_trans_bilstm"),
            # "LSTM": (LSTM_Predictor, "wind_lstm"),
        }

        # 实例缓存池：存储已经加载过的模型实例，避免重复加载浪费时间
        self._loaded_models = {}

    def _get_model(self, model_name):
        """获取模型实例，如果未加载则动态加载"""
        if model_name not in self._model_registry:
            raise ValueError(f"未知模型: {model_name}。可选模型包括: {list(self._model_registry.keys())}")

        if model_name not in self._loaded_models:
            ModelClass, sub_folder_name = self._model_registry[model_name]

            # 拼接出该算法专属的绝对路径 (例如: E:/project/pretrained_models/wind_ceemdan_lgbm_trans)
            specific_model_dir = os.path.join(self.base_models_dir, sub_folder_name)

            print(f"首次调用，正在从 {specific_model_dir} 加载模型 [{model_name}]...")

            # 将拼接好的专属路径传入预测器的 __init__ 函数
            self._loaded_models[model_name] = ModelClass(model_dir=specific_model_dir)

        return self._loaded_models[model_name]

    def run(self, model_name, df, steps=1, **kwargs):
        """
        统一调度接口
        :param model_name: 模型名称
        :param df: 输入数据DataFrame
        :param steps: 预测步长 (1-24)
        :param kwargs: 其他参数(如future_weather_df, mode等),会透传给模型的predict方法
        :return: 预测结果字典
        """
        try:
            # 自动获取已初始化的专属模型实例
            model = self._get_model(model_name)

            # 🔧 统一调用predict方法，传入steps参数和其他可选参数
            return model.predict(df, steps=steps, **kwargs)

        except Exception as e:
            return {"success": False, "error": str(e)}