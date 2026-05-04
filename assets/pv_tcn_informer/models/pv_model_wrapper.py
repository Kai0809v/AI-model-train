"""
光伏TCN-Informer模型包装器
负责加载True_TCN_Informer模型并执行推理
"""
import torch
import torch.nn as nn
import numpy as np
import os
import sys

# 添加Informer2020路径
informer_path = os.path.join(os.path.dirname(__file__), '../../../../Informer2020')
if informer_path not in sys.path:
    sys.path.insert(0, informer_path)

from model_architecture import True_TCN_Informer


class PV_ModelWrapper:
    """
    光伏TCN-Informer模型包装器
    负责加载模型和执行推理
    """
    
    def __init__(self, asset_dir, input_dim=11, model_filename="best_tcn_informer.pth"):
        """
        :param asset_dir: 资产目录路径
        :param input_dim: PCA降维后的特征维度(默认11)
        :param model_filename: 模型文件名(默认best_tcn_informer.pth)
        """
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.input_dim = input_dim
        self.model_filename = model_filename
        
        # 固定参数(与训练时保持一致)
        self.seq_len = 192
        self.label_len = 96
        self.max_pred_len = 24
        
        # 加载模型
        self._load_model(asset_dir)
    
    def _load_model(self, asset_dir):
        """加载训练好的模型权重"""
        model_path = os.path.join(asset_dir, self.model_filename)
        
        self.model = True_TCN_Informer(
            tcn_input_dim=self.input_dim,
            tcn_channels=[16, 32],
            seq_len=self.seq_len,
            label_len=self.label_len,
            pred_len=self.max_pred_len,
            d_model=64,
            n_heads=4,
            e_layers=2,
            dropout=0.15
        ).to(self.device)
        
        self.model.load_state_dict(
            torch.load(model_path, map_location=self.device, weights_only=False)
        )
        self.model.eval()
        
        print(f"[✓] TCN-Informer模型加载成功 (输入维度: {self.input_dim})")
    
    def predict(self, seq_x, seq_x_mark, dec_x, dec_x_mark, steps=24):
        """
        执行模型推理
        :param seq_x: 编码器输入 [1, 192, 11]
        :param seq_x_mark: 编码器时间标记 [1, 192, 5]
        :param dec_x: 解码器输入 [1, 120, 11] (96历史+24未来)
        :param dec_x_mark: 解码器时间标记 [1, 120, 5]
        :param steps: 预测步长(1-24)
        :return: 预测结果 [1, steps] (标准化后的值)
        """
        # 移动到设备
        seq_x = seq_x.to(self.device)
        seq_x_mark = seq_x_mark.to(self.device)
        dec_x = dec_x.to(self.device)
        dec_x_mark = dec_x_mark.to(self.device)
        
        # 模型推理
        with torch.no_grad():
            pred_scaled = self.model(seq_x, seq_x_mark, dec_x, dec_x_mark)
            pred_scaled = pred_scaled.cpu().numpy()  # [1, 24]
        
        # 截取所需步长
        pred_scaled_truncated = pred_scaled[:, :steps]  # [1, steps]
        
        return pred_scaled_truncated
