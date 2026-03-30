# ============================================
# 统一预测接口（可直接对接 GUI）
# v2改了算法，但是api中与训练时的特征构建逻辑不一致
# ============================================

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from joblib import load
from PyEMD import CEEMDAN
import math
import os

import os
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import math
from joblib import load


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


# ============================================
# 2 专属预测器 API 封装
# ============================================
class CEEMDAN_LGBM_Transformer_Predictor:
    def __init__(self, model_dir):
        """
        初始化预测器，加载该算法专属的特征索引、标准化器和模型权重
        """
        self.model_dir = model_dir
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.window_size = 96  # 训练时设定的窗口大小 (24小时)

        # 1. 加载资产文件
        try:
            self.scaler_x = load(os.path.join(model_dir, "scaler_x"))
            self.scaler_y = load(os.path.join(model_dir, "scaler_y"))
            self.selected_features = np.load(os.path.join(model_dir, "selected_features_indices.npy"))
            weight_path = os.path.join(model_dir, "transformer_weights_single_minmax.pth") # transformer_weights_singlelearn.pth
        except FileNotFoundError as e:
            raise FileNotFoundError(f"加载模型资产失败，请检查 {model_dir} 目录下是否缺少文件: {str(e)}")

        # 2. 实例化模型并加载权重
        self.model = TransformerModel(input_dim=len(self.selected_features)).to(self.device)
        self.model.load_state_dict(torch.load(weight_path, map_location=self.device))
        self.model.eval()  # 务必切换到推断模式

        # 预期的基础列名顺序 (需与 GUI 传入的 DataFrame 保持一致)
        self.expected_cols = [
            '测风塔10m风速(m/s)', '测风塔30m风速(m/s)', '测风塔50m风速(m/s)',
            '测风塔70m风速(m/s)', '轮毂高度风速(m/s)', '测风塔10m风向(°)',
            '测风塔30m风向(°)', '测风塔50m风向(°)', '测风塔70m风向(°)',
            '轮毂高度风向(°)', '温度(°)', '气压(hPa)', '湿度(%)',
            '实际发电功率（mw）'  # 历史功率必须紧跟在最后
        ]

    def _preprocess(self, df: pd.DataFrame) -> torch.Tensor:
        """
        复刻 part1v3.py 中的物理特征衍生与切片逻辑
        """
        if len(df) < self.window_size:
            raise ValueError(f"输入数据长度不足！需要至少 {self.window_size} 行历史数据，当前仅有 {len(df)} 行。")

        # 🔧 智能处理：如果 DataFrame 没有 DatetimeIndex，尝试自动转换
        if not isinstance(df.index, pd.DatetimeIndex):
            # 尝试查找常见的时间列名
            time_col_candidates = ['时间', 'timestamp', 'datetime', '日期', 'time']
            found_time_col = None

            for col_name in time_col_candidates:
                if col_name in df.columns:
                    found_time_col = col_name
                    break

            if found_time_col:
                # 转换为 DatetimeIndex
                df = df.copy()
                df[found_time_col] = pd.to_datetime(df[found_time_col])
                df.set_index(found_time_col, inplace=True)
            else:
                # 如果找不到时间列，记录警告并使用降级策略
                print(f"⚠️ 警告：未找到时间列，将使用索引推算时间特征")

        # 截取最近的 96 步数据
        df_recent = df.iloc[-self.window_size:].copy()

        # 提取基础数据 (14维)
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

        # 8. 时间周期特征 (智能降级策略)
        # 计算相对于窗口起点的索引位置
        data_index = np.arange(len(df) - self.window_size, len(df))

        # 优先使用 DatetimeIndex，如果不存在则回退到索引推算
        if isinstance(df_recent.index, pd.DatetimeIndex):
            # print("[调试] 精准模式")
            # 精准模式：从时间戳提取
            hour_of_day = df_recent.index.hour + df_recent.index.minute / 60.0
            day_of_week = df_recent.index.dayofweek

            time_sin = np.sin(2 * np.pi * hour_of_day / 24).values.reshape(-1, 1)
            time_cos = np.cos(2 * np.pi * hour_of_day / 24).values.reshape(-1, 1)
            dow_sin = np.sin(2 * np.pi * day_of_week / 7).values.reshape(-1, 1)
            dow_cos = np.cos(2 * np.pi * day_of_week / 7).values.reshape(-1, 1)
        else:
            # print("[调试] 降级模式")
            # 降级模式：假设数据从某天 00:00 开始，每 15 分钟一个数据点
            # 这与 part1v3.py 训练时的逻辑完全一致
            hour_of_day = (data_index % 96) / 96 * 24  # 一天中的时刻 (0-24)
            day_of_week = (data_index // 96) % 7  # 一周中的第几天 (0-6)

            time_sin = np.sin(2 * np.pi * hour_of_day / 24).reshape(-1, 1)
            time_cos = np.cos(2 * np.pi * hour_of_day / 24).reshape(-1, 1)
            dow_sin = np.sin(2 * np.pi * day_of_week / 7).reshape(-1, 1)
            dow_cos = np.cos(2 * np.pi * day_of_week / 7).reshape(-1, 1)

        feats.extend([time_sin, time_cos, dow_sin, dow_cos])

        # 拼接全部特征
        full_features = np.hstack([base_data] + feats)

        # 归一化 (使用训练时拟合的 scaler_x)
        full_features_scaled = self.scaler_x.transform(full_features)

        # LightGBM 空间降维切片 (极度重要)
        selected_features_scaled = full_features_scaled[:, self.selected_features]

        # 转换为 PyTorch 张量: 形状 (Batch=1, Seq=96, Features)
        tensor_x = torch.tensor(selected_features_scaled).unsqueeze(0).float()
        return tensor_x

    def predict(self, df: pd.DataFrame) -> dict:
        """
        单步预测：预测未来第 1 步 (未来 15 分钟) 的功率
        """
        try:
            tensor_x = self._preprocess(df).to(self.device)

            with torch.no_grad():
                pred_scaled = self.model(tensor_x)

            pred_real = self.scaler_y.inverse_transform(pred_scaled.cpu().numpy())

            # 返回标准的 JSON/字典 格式，供 GUI 解析渲染
            return {
                "success": True,
                "prediction": float(pred_real[0][0]),
                "model_name": "CEEMDAN_LGBM_Transformer"
            }
        except Exception as e:
            return {"success": False, "error": str(e)}

    def predict_multi(self, df: pd.DataFrame, steps: int) -> dict:
        """
        多步自回归预测：滚动预测未来多步
        注意：实际业务中多步预测最好使用未来的气象预报(NWP)数据。
        如果这里 GUI 只传入了历史数据，则采用最简单的自回归(特征平移)进行滚动。
        """
        try:
            predictions = []
            # 复制一份用于滚动更新的内部 df
            current_df = df.copy()

            for step in range(steps):
                # 1. 预测当前步
                tensor_x = self._preprocess(current_df).to(self.device)
                with torch.no_grad():
                    pred_scaled = self.model(tensor_x)
                pred_real = self.scaler_y.inverse_transform(pred_scaled.cpu().numpy())
                step_pred_value = float(pred_real[0][0])
                predictions.append(step_pred_value)

                # 2. 构造虚拟的下一步数据 (自回归填补)
                last_row = current_df.iloc[-1:].copy()
                
                # 🔧 修复：显式转换索引类型并计算新时间戳
                if isinstance(current_df.index, pd.DatetimeIndex):
                    # 如果是 DatetimeIndex，计算下一个时间点
                    last_timestamp = current_df.index[-1]
                    new_timestamp = last_timestamp + pd.Timedelta(minutes=15)
                    
                    # 创建新的单行 DataFrame
                    new_row_dict = last_row.to_dict('records')[0]
                    new_row_dict['实际发电功率（mw）'] = step_pred_value
                    
                    # 使用新索引创建 Series 然后转为 DataFrame
                    new_row = pd.DataFrame([new_row_dict], index=[new_timestamp])
                else:
                    # 如果没有 DatetimeIndex，使用整数索引
                    new_index = current_df.index[-1] + 1
                    new_row_dict = last_row.to_dict('records')[0]
                    new_row_dict['实际发电功率（mw）'] = step_pred_value
                    new_row = pd.DataFrame([new_row_dict], index=[new_index])
                
                # 将新行拼接到数据框
                current_df = pd.concat([current_df, new_row], axis=0)

            return {
                "success": True,
                "predictions": predictions,
                "steps": steps
            }
        except Exception as e:
            return {"success": False, "error": str(e)}


# --------------------------------------------
# TODO: 为后续算法预留的骨架，写完后在下方调度器里解开注释即可
# --------------------------------------------
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
    def __init__(self, base_models_dir="pretrained"):
        # ★ 动态获取当前脚本所在的绝对路径，并拼接出总的仓库文件夹
        BASE_DIR = os.path.dirname(os.path.abspath(__file__))
        self.base_models_dir = os.path.join(BASE_DIR, base_models_dir)

        # ★ 注册表：将 "GUI传入的名字" 映射到 -> (类名, "该模型专属的子文件夹名")
        self._model_registry = {
            "CEEMDAN_LGBM_Transformer": (CEEMDAN_LGBM_Transformer_Predictor, "wind_ceemdan_lgbm_trans"),

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

    def run(self, model_name, df, steps=1):
        try:
            # 自动获取已初始化的专属模型实例
            model = self._get_model(model_name)

            if steps == 1:
                return model.predict(df)
            else:
                return model.predict_multi(df, steps)

        except Exception as e:
            return {"success": False, "error": str(e)}