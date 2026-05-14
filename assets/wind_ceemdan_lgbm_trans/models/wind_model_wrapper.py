"""
风电CEEMDAN-LGBM-Transformer模型包装器
负责加载集成学习模型(h1/h4/h8)并执行推理
"""
import torch
import torch.nn as nn
import numpy as np
import os
import math


class PositionalEncoding(nn.Module):
    """位置编码(与训练代码保持一致)"""
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


class SimpleMultiStepTransformer(nn.Module):
    """与训练代码part2_multi_horizon_v7.py保持一致"""
    def __init__(self, input_dim, horizon=1):
        super().__init__()
        
        self.embedding = nn.Linear(input_dim, 256)
        self.pos_encoder = PositionalEncoding(d_model=256)
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=256, nhead=8, dim_feedforward=512,
            batch_first=True, dropout=0.2
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
            nn.Dropout(0.3),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
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


class Wind_ModelWrapper:
    """
    风电CEEMDAN-LGBM-Transformer模型包装器（单模型版）
    负责加载 h1/h4/h8 三个步长的独立模型并执行推理
    """
    
    def __init__(self, asset_dir):
        """
        :param asset_dir: 资产目录路径
        """
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.asset_dir = asset_dir
        
        # 加载集成学习模型或单模型
        self._load_models(asset_dir)
    
    def _load_models(self, asset_dir):
        """
        加载h1/h4/h8三个步长的单模型（适配 part2_multi_horizon_v7.py 格式）
        是的，又双叒叕更新了
        """
        self.single_models = {}
        
        for h in [1, 4, 8]:
            model_path = os.path.join(asset_dir, f"transformer_model_h{h}.pth")
            if os.path.exists(model_path):
                try:
                    # 修复 PyTorch 2.6+ 的 weights_only 限制
                    model_package = torch.load(model_path, map_location=self.device, weights_only=False)
                    
                    # 适配新保存的键名 'model_state_dict'
                    state_dict = model_package.get('model_state_dict') or model_package.get('state_dict')
                    
                    # 获取特征维度，优先从包中读取，否则从权重推断
                    input_dim = model_package.get('feature_dim', None)
                    if input_dim is None and state_dict:
                        input_dim = state_dict['embedding.weight'].shape[1]
                    else:
                        input_dim = 37 # 默认 fallback
                    
                    model = SimpleMultiStepTransformer(
                        input_dim=input_dim,
                        horizon=h
                    ).to(self.device)
                    
                    model.load_state_dict(state_dict)
                    model.eval()
                    self.single_models[h] = model
                    print(f"[+] 已加载{h}步单模型 (Feature Dim: {input_dim})")
                except Exception as e:
                    print(f"[!] 加载{h}步模型失败: {e}")
            else:
                print(f"[!] 未找到模型文件: {model_path}")
        
        if not self.single_models:
            print("[!] 警告: 未加载到任何可用模型")
        else:
            print("[✓] 单模型加载完成")
    
    def predict(self, tensor_x, steps=1, scaler_y=None):
        """
        执行单模型推理
        :param tensor_x: 输入张量 [1, 96, selected_dim]
        :param steps: 预测步长(1/4/8)
        :param scaler_y: y的标准化器(用于反归一化)
        :return: 预测字典{"success": True, "predictions": [...]}
        """
        try:
            # 验证步长
            if steps not in [1, 4, 8]:
                return {"success": False, "error": f"不支持的预测步长:{steps}。仅支持[1, 4, 8]"}
            
            if not hasattr(self, 'single_models') or steps not in self.single_models:
                return {"success": False, "error": f"未加载{steps}步模型或模型不可用"}
            
            model = self.single_models[steps]
            tensor_x = tensor_x.to(self.device)
            
            with torch.no_grad():
                pred_scaled = model(tensor_x).cpu().numpy()  # [1, steps]
            
            # 反归一化
            if scaler_y is not None:
                pred_real = np.zeros_like(pred_scaled)
                for col in range(steps):
                    pred_real[:, col] = scaler_y.inverse_transform(
                        pred_scaled[:, col:col+1]
                    ).flatten()
                
                # 负功率修正
                predictions = pred_real[0].tolist()
                predictions = [max(0.0, p) for p in predictions]
            else:
                predictions = pred_scaled[0].tolist()
            
            return {
                "success": True,
                "predictions": predictions,
                "steps": steps
            }
            
        except Exception as e:
            import traceback
            traceback.print_exc()
            return {"success": False, "error": str(e)}
