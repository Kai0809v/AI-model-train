"""
光伏TCN-Informer模型包装器
负责加载True_TCN_Informer模型并执行推理
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import sys

# 添加Informer2020路径
informer_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../Informer2020'))
if informer_path not in sys.path:
    sys.path.insert(0, informer_path)

from Informer2020.models.model import Informer


# ==========================================
# TCN 核心组件
# ==========================================
class Chomp1d(nn.Module):
    """用于裁剪卷积后的多余填充，保证严格的因果性"""

    def __init__(self, chomp_size):
        super(Chomp1d, self).__init__()
        self.chomp_size = chomp_size

    def forward(self, x):
        return x[:, :, :-self.chomp_size].contiguous()


class TemporalBlock(nn.Module):
    """TCN 的基本残差块"""

    def __init__(self, n_inputs, n_outputs, kernel_size, stride, dilation, padding, dropout=0.2):
        super(TemporalBlock, self).__init__()
        self.conv1 = nn.Conv1d(n_inputs, n_outputs, kernel_size, stride=stride, padding=padding, dilation=dilation)
        self.chomp1 = Chomp1d(padding)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)

        self.conv2 = nn.Conv1d(n_outputs, n_outputs, kernel_size, stride=stride, padding=padding, dilation=dilation)
        self.chomp2 = Chomp1d(padding)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout)

        self.net = nn.Sequential(self.conv1, self.chomp1, self.relu1, self.dropout1,
                                 self.conv2, self.chomp2, self.relu2, self.dropout2)

        self.downsample = nn.Conv1d(n_inputs, n_outputs, 1) if n_inputs != n_outputs else None
        self.relu = nn.ReLU()

    def forward(self, x):
        out = self.net(x)
        res = x if self.downsample is None else self.downsample(x)
        return self.relu(out + res)


class TemporalConvNet(nn.Module):
    """TCN 主网络"""

    def __init__(self, num_inputs, num_channels, kernel_size=3, dropout=0.2):
        super(TemporalConvNet, self).__init__()
        layers = []
        num_levels = len(num_channels)
        for i in range(num_levels):
            dilation_size = 2 ** i
            in_channels = num_inputs if i == 0 else num_channels[i - 1]
            out_channels = num_channels[i]
            padding = (kernel_size - 1) * dilation_size
            layers += [
                TemporalBlock(in_channels, out_channels, kernel_size, stride=1, dilation=dilation_size, padding=padding,
                              dropout=dropout)]
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        x = x.transpose(1, 2)
        y = self.network(x)
        return y.transpose(1, 2)


# ==========================================
# TCN-Informer 模型
# ==========================================
class True_TCN_Informer(nn.Module):
    def __init__(self, tcn_input_dim, tcn_channels, seq_len, label_len, pred_len,
                 d_model=512, n_heads=8, e_layers=3, d_layers=1, dropout=0.05):
        super(True_TCN_Informer, self).__init__()

        self.tcn = TemporalConvNet(num_inputs=tcn_input_dim, num_channels=tcn_channels)
        tcn_out_dim = tcn_channels[-1]

        self.informer = Informer(
            enc_in=tcn_out_dim,
            dec_in=tcn_out_dim,
            c_out=1,
            seq_len=seq_len,
            label_len=label_len,
            out_len=pred_len,
            factor=5,
            d_model=d_model,
            n_heads=n_heads,
            e_layers=e_layers,
            d_layers=d_layers,
            d_ff=d_model * 4,
            dropout=dropout,
            attn='prob',
            embed='timeF',
            freq='t',
            activation='gelu',
            output_attention=False
        )

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec):
        tcn_encoded_x = self.tcn(x_enc)
        tcn_decoded_x = self.tcn(x_dec)
        dec_out = self.informer(tcn_encoded_x, x_mark_enc, tcn_decoded_x, x_mark_dec)
        return dec_out


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