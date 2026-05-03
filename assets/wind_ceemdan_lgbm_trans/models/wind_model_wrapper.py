"""
风电CEEMDAN-LGBM-Transformer模型包装器
负责加载集成学习模型(h1/h4/h8)并执行推理
"""
import torch
import torch.nn as nn
import numpy as np
import os


class SimpleMultiStepTransformer(nn.Module):
    """简化版多步Transformer模型(与训练时保持一致)"""
    def __init__(self, input_dim, horizon=1, d_model=64, nhead=4, num_layers=2, dim_feedforward=128, dropout=0.1):
        super().__init__()
        self.input_proj = nn.Linear(input_dim, d_model)
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward, dropout=dropout)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.horizon = horizon
        self.fc = nn.Linear(d_model, horizon)
        
    def forward(self, x):
        # x: [batch, seq_len, features]
        x = self.input_proj(x)  # [batch, seq_len, d_model]
        x = x.permute(1, 0, 2)  # [seq_len, batch, d_model]
        x = self.transformer_encoder(x)  # [seq_len, batch, d_model]
        x = x[-1, :, :]  # [batch, d_model] - 取最后一个时间步
        output = self.fc(x)  # [batch, horizon]
        return output


class Wind_ModelWrapper:
    """
    风电CEEMDAN-LGBM-Transformer模型包装器
    负责加载集成学习模型并执行推理
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
        """加载h1/h4/h8三个步长的集成模型"""
        ensemble_h1_path = os.path.join(asset_dir, "ensemble_models_h1_v6.pth")
        
        if os.path.exists(ensemble_h1_path):
            # V6集成学习模式
            self.use_ensemble = True
            self.ensemble_models = {}
            
            for h in [1, 4, 8]:
                ensemble_path = os.path.join(asset_dir, f"ensemble_models_h{h}_v6.pth")
                if os.path.exists(ensemble_path):
                    ensemble_package = torch.load(ensemble_path, map_location=self.device, weights_only=False)
                    models = []
                    
                    # 从第一个模型获取input_dim
                    first_state_dict = ensemble_package['model_state_dicts'][0]
                    input_dim = first_state_dict['input_proj.weight'].shape[1]
                    
                    for state_dict in ensemble_package['model_state_dicts']:
                        model = SimpleMultiStepTransformer(
                            input_dim=input_dim,
                            horizon=h
                        ).to(self.device)
                        model.load_state_dict(state_dict)
                        model.eval()
                        models.append(model)
                    
                    self.ensemble_models[h] = models
                    print(f"[+] 已加载{h}步集成模型({len(models)}个子模型)")
            
            print("[✓] 使用V6集成学习模式")
        else:
            # 回退到传统单模型模式(如果需要可以扩展)
            self.use_ensemble = False
            print("[!] 警告:未找到集成模型文件,仅支持占位")
    
    def predict(self, tensor_x, steps=1, scaler_y=None):
        """
        执行模型推理
        :param tensor_x: 输入张量 [1, 96, selected_dim]
        :param steps: 预测步长(1/4/8)
        :param scaler_y: y的标准化器(用于反归一化)
        :return: 预测字典{"success": True, "predictions": [...]}
        """
        try:
            # 验证步长
            if steps not in [1, 4, 8]:
                return {"success": False, "error": f"不支持的预测步长:{steps}。仅支持[1, 4, 8]"}
            
            if not self.use_ensemble:
                return {"success": False, "error": "未加载集成模型"}
            
            # 获取对应步长的模型集合
            models = self.ensemble_models.get(steps)
            if models is None:
                return {"success": False, "error": f"未找到{steps}步模型"}
            
            # 集成学习推理:多个模型投票
            tensor_x = tensor_x.to(self.device)
            all_preds = []
            
            with torch.no_grad():
                for model in models:
                    pred = model(tensor_x)  # [1, steps]
                    all_preds.append(pred.cpu().numpy())
            
            # 取平均:[1, steps]
            pred_scaled = np.mean(all_preds, axis=0)
            
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
