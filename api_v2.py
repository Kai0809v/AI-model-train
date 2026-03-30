# ============================================
# 统一预测接口（可直接对接 GUI）
# 相比v1，除了路径文件名以外，因为对模型修改（Transformer 模型结构不一致、位置编码的前向传播不一致、特征聚合方式不同），进行了修改
# ============================================

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from joblib import load
from PyEMD import CEEMDAN
import math
import os


# ============================================
# 1. 深度学习模型结构定义（底层算子）
# ============================================
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        div_term = torch.exp(torch.arange(0, d_model, 2).float() *
                             (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe.unsqueeze(0))

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

    def forward(self, x):
        x = self.embedding(x)
        x = self.pos_encoder(x)
        x = self.transformer(x)

        attention_weights = self.attention_pooling(x)
        attention_weights = torch.softmax(attention_weights, dim=1)
        context = torch.sum(x * attention_weights, dim=1)

        out = self.fc(context)
        return out


# ============================================
# 2. 核心预测器类（具体算法实现）
# ============================================

class CEEMDAN_LGBM_Transformer_Predictor:
    def __init__(self, model_dir):
        """
        :param model_dir: 该算法专属的模型资产绝对路径 (由调度器传入)
        """
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.window_size = 24

        # ===== 基础特征列 =====
        self.base_feature_cols = [
            '测风塔10m风速(m/s)', '测风塔30m风速(m/s)', '测风塔50m风速(m/s)',
            '测风塔70m风速(m/s)', '轮毂高度风速(m/s)',
            '测风塔10m风向(°)', '测风塔30m风向(°)', '测风塔50m风向(°)',
            '测风塔70m风向(°)', '轮毂高度风向(°)',
            '温度(°)', '气压(hPa)', '湿度(%)',
            '实际发电功率（mw）'
        ]

        try:
            # ★ 改变：全部从自己专属的 model_dir 下加载资产，避免与其他算法冲突
            self.scaler_x = load(os.path.join(model_dir, "scaler_x"))
            self.scaler_y = load(os.path.join(model_dir, "scaler_y"))

            self.selected_features = np.load(os.path.join(model_dir, "selected_features_indices.npy"))

            self.model = TransformerModel(len(self.selected_features)).to(self.device)
            # TODO：注意模型名称，在几种不同的模型调整下我的模型名称也略有不同，如果要换一种调整情况的模型使用也要注意这里
            self.model.load_state_dict(
                torch.load(os.path.join(model_dir, "transformer_weights_singlelearn.pth"), map_location=self.device)
            )
            self.model.eval()

            self.ceemdan = CEEMDAN(trials=100, epsilon=0.005)

        except Exception as e:
            raise RuntimeError(f"加载 CEEMDAN-LGBM-Transformer 模型文件失败，请检查 [{model_dir}] 下的文件是否齐全: {e}")

    def _validate_input(self, df):
        if not isinstance(df, pd.DataFrame):
            raise ValueError("输入必须是 pandas DataFrame")
        if len(df) < self.window_size:
            raise ValueError(f"数据至少需要 {self.window_size} 行")
        missing_cols = [c for c in self.base_feature_cols if c not in df.columns]
        if missing_cols:
            raise ValueError(f"缺少必要列: {missing_cols}")

    def _predict_one(self, df):
        # 强制转换为 float，防止 object 类型报错
        power_series = np.nan_to_num(df['实际发电功率（mw）'].values.astype(float))
        
        # 获取用于构建特征的索引位置（用于计算时间周期特征）
        # 假设最后一行是当前时刻，根据其在 DataFrame 中的位置推算
        data_index = np.arange(len(df) - self.window_size, len(df))
        
        # ===== 1. 构建基础特征（与训练时保持一致） =====
        weather_cols = [
            '测风塔 10m 风速 (m/s)', '测风塔 30m 风速 (m/s)', '测风塔 50m 风速 (m/s)',
            '测风塔 70m 风速 (m/s)', '轮毂高度风速 (m/s)',
            '测风塔 10m 风向 (°)', '测风塔 30m 风向 (°)', '测风塔 50m 风向 (°)',
            '测风塔 70m 风向 (°)', '轮毂高度风向 (°)',
            '温度 (°)', '气压 (hPa)', '湿度 (%)'
        ]
        
        base_features = df[weather_cols].values.astype(float)
        
        # 拼接原始功率（1 列），与训练时逻辑一致
        full_features = np.hstack([base_features, power_series.reshape(-1, 1)])
        
        # ===== 2. 添加物理衍生特征（必须与训练时完全一致） =====
        feats = []
        
        # 1. 平均风速（5 个高度的平均）
        avg_wind = np.mean(full_features[:, :5], axis=1, keepdims=True)
        feats.append(avg_wind)
        
        # 2. 风速立方（捕捉 P ∝ v³关系）
        wind_cubed = full_features[:, :5] ** 3
        feats.append(wind_cubed)
        
        # 3. 风速标准差（表征风切变）
        wind_std = np.std(full_features[:, :5], axis=1, keepdims=True)
        feats.append(wind_std)
        
        # 4. 最大风速
        wind_max = np.max(full_features[:, :5], axis=1, keepdims=True)
        feats.append(wind_max)
        
        # 5. 风向一致性（计算风向的余弦相似度）
        direction_cols = full_features[:, 5:10]
        direction_cos = np.cos(np.deg2rad(direction_cols))
        direction_mean = np.mean(direction_cos, axis=1, keepdims=True)
        feats.append(direction_mean)
        
        # 6. 空气密度修正因子（温度、气压、湿度的综合影响）
        temp = full_features[:, 10]
        pressure = full_features[:, 11]
        humidity = full_features[:, 12]
        air_density_factor = (pressure / 1013.25) * (288.15 / (temp + 273.15))
        feats.append(air_density_factor.reshape(-1, 1))
        
        # 7. 风能密度近似值
        wind_power_density = 0.5 * air_density_factor * (avg_wind ** 3).flatten()
        feats.append(wind_power_density.reshape(-1, 1))
        
        # 8. 时间周期特征（15 分钟数据的日内/周内周期）
        if data_index is not None:
            hour_of_day = (data_index % 96) / 96 * 24
            day_of_week = (data_index // 96) % 7
            
            time_sin = np.sin(2 * np.pi * hour_of_day / 24).reshape(-1, 1)
            time_cos = np.cos(2 * np.pi * hour_of_day / 24).reshape(-1, 1)
            dow_sin = np.sin(2 * np.pi * day_of_week / 7).reshape(-1, 1)
            dow_cos = np.cos(2 * np.pi * day_of_week / 7).reshape(-1, 1)
            
            feats.extend([time_sin, time_cos, dow_sin, dow_cos])
        
        # 合并所有特征
        full_features_with_physics = np.hstack([full_features] + feats)
        
        # ===== 3. 对功率序列进行 CEEMDAN 分解（提取 IMF 特征） =====
        ceemdan = CEEMDAN(trials=100, epsilon=0.005)
        imfs = ceemdan(power_series)
        
        expected_total_features = self.scaler_x.mean_.shape[0]
        expected_imf_count = expected_total_features - full_features_with_physics.shape[1]
        
        # IMF 对齐（与训练时保持一致）
        if imfs.shape[0] > expected_imf_count:
            imfs = imfs[:expected_imf_count]
        elif imfs.shape[0] < expected_imf_count:
            pad = np.zeros((expected_imf_count - imfs.shape[0], imfs.shape[1]))
            imfs = np.vstack((imfs, pad))
        
        # 将 IMF 特征拼接到每个时间步
        # imfs 形状：(n_imfs, n_samples) -> 需要转置并广播到每个时间步
        imfs_T = imfs.T  # (n_samples, n_imfs)
        
        # 取最近 window_size 个时间步的特征
        full_features_final = np.hstack([
            full_features_with_physics[-self.window_size:],
            imfs_T[-self.window_size:]
        ])
        
        # ===== 4. 标准化与预测 =====
        window_scaled = self.scaler_x.transform(full_features_final)
        window_filtered = window_scaled[:, self.selected_features]
        
        x = torch.tensor(window_filtered, dtype=torch.float32).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            pred_scaled = self.model(x).cpu().numpy()
        
        pred = self.scaler_y.inverse_transform(pred_scaled)
        return float(pred[0][0])

    def predict(self, df):
        try:
            self._validate_input(df)
            result = self._predict_one(df)
            return {"success": True, "prediction": result}
        except Exception as e:
            return {"success": False, "error": str(e)}

    def predict_multi(self, df, steps=24):
        try:
            self._validate_input(df)
            df_copy = df.copy()
            preds = []
            for _ in range(steps):
                pred = self._predict_one(df_copy)
                preds.append(pred)

                # iloc[-1:] 返回的是 DataFrame 而不是 Series，不会丢失数据类型
                new_row = df_copy.iloc[-1:].copy()
                new_row['实际发电功率（mw）'] = pred

                # 由于 new_row 已经是 DataFrame，直接 concat 即可
                df_copy = pd.concat([df_copy, new_row], ignore_index=True)

            return {"success": True, "predictions": preds}
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

            # 拼接出该算法专属的路径
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