# ============================================
# 统一预测接口（可直接对接 GUI）
# TODO：在GUI选择有缺失的数据预测时会有报错
# ============================================

import os
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import math
from joblib import load

# 🔧 修复 PyTorch 2.6 的 weights_only 安全限制
# 允许加载包含 numpy 标量的模型文件
torch.serialization.add_safe_globals([np.core.multiarray.scalar])


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
            
            # 🔧 V6集成学习版：检查是否存在集成模型文件
            ensemble_h1_path = os.path.join(model_dir, "ensemble_models_h1_v6.pth")
            if os.path.exists(ensemble_h1_path):
                self.use_ensemble = True
                self.ensemble_models = {}
                for h in [1, 4, 8]:
                    ensemble_path = os.path.join(model_dir, f"ensemble_models_h{h}_v6.pth")
                    if os.path.exists(ensemble_path):
                        ensemble_package = torch.load(ensemble_path, map_location=self.device, weights_only=False)
                        models = []
                        for state_dict in ensemble_package['model_state_dicts']:
                            model = SimpleMultiStepTransformer(
                                input_dim=len(self.selected_features),
                                horizon=h
                            ).to(self.device)
                            model.load_state_dict(state_dict)
                            model.eval()
                            models.append(model)
                        self.ensemble_models[h] = models
                        print(f"[+] 已加载 {h}步集成模型（{len(models)}个子模型）")
                
                print("[✓] 使用V6集成学习模式")
            else:
                # 回退到单模型模式
                self.use_ensemble = False
                weight_path = os.path.join(model_dir, "transformer_weights_single_minmax.pth")
                self.model = TransformerModel(input_dim=len(self.selected_features)).to(self.device)
                self.model.load_state_dict(torch.load(weight_path, map_location=self.device, weights_only=False))
                self.model.eval()
                print("[✓] 使用传统单模型模式")
                
        except FileNotFoundError as e:
            raise FileNotFoundError(f"加载模型资产失败，请检查 {model_dir} 目录下是否缺少文件: {str(e)}")

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

        # 🔧 NaN 值检测与修复
        if df.isnull().any().any():
            print(f"⚠️ 检测到 NaN 值，正在进行填充修复...")
            # 创建副本避免修改原始数据
            df = df.copy()
            
            # 对于风速、风向等关键特征，使用前向填充 (FFill)
            # 因为这类型数据有连续性，用前一个时刻的值填补是合理的
            critical_cols = [col for col in self.expected_cols if col in df.columns]
            df[critical_cols] = df[critical_cols].fillna(method='ffill')
            
            # 如果开头也有 NaN (前向填充无法覆盖)，则用后向填充补充
            df[critical_cols] = df[critical_cols].fillna(method='bfill')
            
            # 如果还有剩余 NaN(极端情况),用该列的均值填补
            df[critical_cols] = df[critical_cols].fillna(df[critical_cols].mean())
            
            print(f"✓ NaN 值修复完成，剩余 NaN 数量：{df.isnull().sum().sum()}")

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

        # 🔧 补全：功率滞后特征（与训练时 part1.py 保持一致）
        # 基础数据的最后一列是功率
        power_col = base_data[:, -1]
        lags = [1, 2, 4, 8, 12, 24, 48]  # 滞后步数：15min, 30min, 1h, 2h, 3h, 6h, 12h
        for lag in lags:
            lagged = np.roll(power_col, lag)
            # 边界处理：用第一个有效值填充
            lagged[:lag] = power_col[lag-1] if lag > 0 else power_col[0]
            feats.append(lagged.reshape(-1, 1))
        
        # 🔧 补全：功率变化率（一阶差分）- 仅使用历史信息
        power_diff = np.zeros_like(power_col)
        power_diff[1:] = np.diff(power_col)  # 从第2个点开始计算差分
        power_diff[0] = 0  # 第一个点设为0
        feats.append(power_diff.reshape(-1, 1))
        
        # 🔧 补全：功率移动平均（平滑趋势）- 仅使用历史数据
        for window in [4, 12, 24]:  # 1h, 3h, 6h 移动平均
            ma = np.zeros_like(power_col)
            for i in range(len(power_col)):
                # 只使用 t-window+1 到 t 的历史数据
                start_idx = max(0, i - window + 1)
                ma[i] = np.mean(power_col[start_idx:i+1])
            feats.append(ma.reshape(-1, 1))

        # 拼接全部特征
        full_features = np.hstack([base_data] + feats)

        # 归一化 (使用训练时拟合的 scaler_x)
        full_features_scaled = self.scaler_x.transform(full_features)

        # LightGBM 空间降维切片 (极度重要)
        selected_features_scaled = full_features_scaled[:, self.selected_features]

        # 转换为 PyTorch 张量: 形状 (Batch=1, Seq=96, Features)
        tensor_x = torch.tensor(selected_features_scaled).unsqueeze(0).float()
        return tensor_x

    def predict(self, df: pd.DataFrame, steps: int = 1) -> dict:
        """
        统一预测接口：支持单步和多步预测
        自动根据steps参数选择对应的集成模型
        """
        try:
            # 🔧 验证步长是否支持
            if steps not in [1, 4, 8]:
                return {"success": False, "error": f"不支持的预测步长：{steps}。仅支持 [1, 4, 8]"}
            
            # 🔧 第一道防线：检查并处理 NaN 值
            if df.isnull().any().any():
                print(f"⚠️ predict() 检测到 NaN 值，将尝试修复...")
                df = df.copy()
                
                # 对关键列进行 NaN 修复
                critical_cols = [col for col in self.expected_cols if col in df.columns]
                df[critical_cols] = df[critical_cols].fillna(method='ffill').fillna(method='bfill')
                
                # 如果还有 NaN，使用该列窗口内的均值
                for col in critical_cols:
                    if df[col].isnull().any():
                        df[col].fillna(df[col].mean(), inplace=True)

            # 1. 提取当前这一步的真实原始风速 (未归一化的原始 df)
            current_wind_speed = df['轮毂高度风速(m/s)'].iloc[-1]

            # 💡 物理规则拦截：如果风速小于 3.0 m/s，直接锁死为 0
            if current_wind_speed < 3.0:
                predictions = [0.0] * steps
                if steps == 1:
                    return {
                        "success": True,
                        "prediction": 0.0,
                        "model_name": "CEEMDAN_LGBM_Transformer",
                        "note": "Blocked by Cut-in Wind Speed threshold"
                    }
                else:
                    return {
                        "success": True,
                        "predictions": predictions,
                        "steps": steps,
                        "note": "Blocked by Cut-in Wind Speed threshold"
                    }

            # 2. 正常经过深度学习网络预测
            tensor_x = self._preprocess(df).to(self.device)
            
            if self.use_ensemble:
                # 🔧 V6集成学习模式：使用对应步长的集成模型
                models = self.ensemble_models[steps]
                all_preds = []
                with torch.no_grad():
                    for model in models:
                        pred = model(tensor_x)
                        all_preds.append(pred.cpu().numpy())
                
                # 取平均：形状 (1, steps)
                pred_scaled = np.mean(all_preds, axis=0)
            else:
                # 传统单模型模式：仅支持1步
                if steps != 1:
                    return {"success": False, "error": "传统单模型模式仅支持1步预测"}
                with torch.no_grad():
                    pred_scaled = self.model(tensor_x).cpu().numpy()

            # 反归一化每一列
            if steps == 1:
                pred_real = self.scaler_y.inverse_transform(pred_scaled)
                final_pred = float(pred_real[0][0])
                
                # 💡 第二道防线：如果模型预测出负数，强行置 0
                if final_pred < 0:
                    final_pred = 0.0
                
                return {
                    "success": True,
                    "prediction": final_pred,
                    "model_name": "CEEMDAN_LGBM_Transformer"
                }
            else:
                # 多步预测：逐列反归一化
                pred_real = np.zeros_like(pred_scaled)
                for col in range(steps):
                    pred_real[:, col] = self.scaler_y.inverse_transform(
                        pred_scaled[:, col:col+1]
                    ).flatten()
                
                # 负功率修正
                predictions = pred_real[0].tolist()
                predictions = [max(0.0, p) for p in predictions]
                
                return {
                    "success": True,
                    "predictions": predictions,
                    "steps": steps
                }
        except Exception as e:
            return {"success": False, "error": str(e)}

    def predict_multi(self, df: pd.DataFrame, steps: int) -> dict:
        """
        多步预测：直接使用对应的集成模型（非自回归）
        支持 1步、4步、8步
        """
        try:
            # 🔧 验证步长是否支持
            if steps not in [1, 4, 8]:
                return {"success": False, "error": f"不支持的预测步长：{steps}。仅支持 [1, 4, 8]"}
            
            # 预处理数据
            if df.isnull().any().any():
                print(f"⚠️ predict_multi() 检测到 NaN 值，正在预处理...")
                df = df.copy()
                critical_cols = [col for col in self.expected_cols if col in df.columns]
                df[critical_cols] = df[critical_cols].fillna(method='ffill').fillna(method='bfill')
                for col in critical_cols:
                    if df[col].isnull().any():
                        df[col].fillna(df[col].mean(), inplace=True)

            # 检查风速条件
            current_wind_speed = df['轮毂高度风速(m/s)'].iloc[-1]
            if current_wind_speed < 3.0:
                # 风速过低，所有步长都返回0
                predictions = [0.0] * steps
                return {
                    "success": True,
                    "predictions": predictions,
                    "steps": steps,
                    "note": "Blocked by Cut-in Wind Speed threshold"
                }

            # 准备输入
            tensor_x = self._preprocess(df).to(self.device)
            
            if self.use_ensemble:
                # 🔧 V6集成学习模式：使用对应步长的集成模型
                models = self.ensemble_models[steps]
                all_preds = []
                with torch.no_grad():
                    for model in models:
                        pred = model(tensor_x)
                        all_preds.append(pred.cpu().numpy())
                
                # 取平均：形状 (1, steps)
                pred_scaled = np.mean(all_preds, axis=0)
            else:
                # 传统模式：不支持多步（理论上不会走到这里）
                return {"success": False, "error": "传统单模型模式不支持多步预测"}

            # 反归一化每一列
            pred_real = np.zeros_like(pred_scaled)
            for col in range(steps):
                pred_real[:, col] = self.scaler_y.inverse_transform(
                    pred_scaled[:, col:col+1]
                ).flatten()

            # 负功率修正
            predictions = pred_real[0].tolist()
            predictions = [max(0.0, p) for p in predictions]

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
    def __init__(self, base_models_dir="assets"):
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
        """
        统一调度接口
        :param model_name: 模型名称
        :param df: 输入数据DataFrame
        :param steps: 预测步长 (1, 4, 8)
        :return: 预测结果字典
        """
        try:
            # 自动获取已初始化的专属模型实例
            model = self._get_model(model_name)

            # 🔧 统一调用predict方法，传入steps参数
            return model.predict(df, steps=steps)

        except Exception as e:
            return {"success": False, "error": str(e)}