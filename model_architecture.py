import torch
import torch.nn as nn
import torch.nn.functional as F


# ==========================================
# 1. 核心组件：因果卷积 (Causal Convolution)
# ==========================================
class Chomp1d(nn.Module):
    """
    用于裁剪卷积后的多余填充，保证严格的“因果性” (Causal)。
    即：时刻 t 的输出绝对不能看到 t+1 的数据（防止数据泄露）。
    """

    def __init__(self, chomp_size):
        super(Chomp1d, self).__init__()
        self.chomp_size = chomp_size

    def forward(self, x):
        return x[:, :, :-self.chomp_size].contiguous()


class TemporalBlock(nn.Module):
    """TCN 的基本残差块"""

    def __init__(self, n_inputs, n_outputs, kernel_size, stride, dilation, padding, dropout=0.2):
        super(TemporalBlock, self).__init__()
        # 空洞因果卷积 1
        self.conv1 = nn.Conv1d(n_inputs, n_outputs, kernel_size, stride=stride, padding=padding, dilation=dilation)
        self.chomp1 = Chomp1d(padding)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)

        # 空洞因果卷积 2
        self.conv2 = nn.Conv1d(n_outputs, n_outputs, kernel_size, stride=stride, padding=padding, dilation=dilation)
        self.chomp2 = Chomp1d(padding)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout)

        self.net = nn.Sequential(self.conv1, self.chomp1, self.relu1, self.dropout1,
                                 self.conv2, self.chomp2, self.relu2, self.dropout2)

        # 残差连接 (如果输入输出通道数不同，使用 1x1 卷积对齐)
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
            # 计算因果卷积的 padding 大小
            padding = (kernel_size - 1) * dilation_size
            layers += [
                TemporalBlock(in_channels, out_channels, kernel_size, stride=1, dilation=dilation_size, padding=padding,
                              dropout=dropout)]
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        # TCN 要求输入形状为 [Batch, Channels, Seq_Len]，因此需要转置
        x = x.transpose(1, 2)
        y = self.network(x)
        # 输出后再转置回 [Batch, Seq_Len, Channels] 供 Informer 使用
        return y.transpose(1, 2)


# ==========================================
# 2. 终极组装：TCN-Informer 架构
# - TCN_Informer_Model这一个是前期测试用的，True_TCN_Informer是正式使用的模型
# ==========================================
class TCN_Informer_Model(nn.Module):
    def __init__(self,
                 input_dim,  # 输入特征维度 (PCA维度 + 时间维度)
                 tcn_channels,  # TCN 各层通道数，如 [32, 64, 128]
                 d_model,  # Informer 的隐藏层维度，如 512
                 n_heads,  # Informer 多头注意力头数
                 e_layers,  # Informer 编码器层数
                 pred_len,  # 预测长度，如 24
                 dropout=0.1):
        super(TCN_Informer_Model, self).__init__()

        # 1. TCN 特征提取模块
        self.tcn = TemporalConvNet(num_inputs=input_dim, num_channels=tcn_channels, dropout=dropout)

        # 2. 维度对齐桥梁 (将 TCN 提取的特征维度对齐到 d_model)
        # 注意：tcn_channels[-1] 是 TCN 最后一层的输出通道数
        self.projection = nn.Linear(tcn_channels[-1], d_model)

        # 3. Informer 核心模块
        # 声明：这里假设您已经导入了标准的 Informer Encoder 和 Decoder
        # 从 github.com/zhouhaoyi/Informer2020 引入
        # self.informer_encoder = Encoder(...)
        # self.informer_decoder = Decoder(...)

        # 为了演示代码的连贯性，这里用占位符代表 Informer 的线性输出层
        # 实际上，Informer 会在这里进行复杂的 ProbSparse Attention 计算
        self.informer_out_layer = nn.Linear(d_model, 1)  # 最终预测单一目标：Power

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec):
        """
        x_enc: 历史观测窗口数据 [Batch, Seq_Len, Features]
        x_mark_enc: 历史观测窗口时间戳特征
        x_dec: 解码器输入 (通常是前面一部分真实值 + 后面待预测部分的占位符)
        x_mark_dec: 预测窗口时间戳特征
        """
        # 第一阶段：TCN 处理序列，提取深层时序特征
        tcn_out = self.tcn(x_enc)  # Shape: [Batch, Seq_Len, tcn_channels[-1]]

        # 第二阶段：维度对齐
        enc_input = self.projection(tcn_out)  # Shape: [Batch, Seq_Len, d_model]

        # 第三阶段：送入 Informer (伪代码逻辑，具体视调用的库而定)
        # 1. 位置编码和时间嵌入融入 enc_input
        # 2. enc_output = self.informer_encoder(enc_input)
        # 3. dec_output = self.informer_decoder(x_dec_embedded, enc_output)
        # 4. final_pred = self.informer_out_layer(dec_output)

        # 演示用直接映射 (代表 Informer 内部处理后输出)
        final_pred = self.informer_out_layer(enc_input[:, -24:, :])  # 假设输出未来 24 步

        return final_pred



# TODO：导入官方的 Informer 模型，其实不加“Informer2020.”也能被识别到
from Informer2020.models.model import Informer


class True_TCN_Informer(nn.Module):
    def __init__(self, tcn_input_dim, tcn_channels, seq_len, label_len, pred_len,
                 d_model=512, n_heads=8, e_layers=3, d_layers=1, dropout=0.05):
        super(True_TCN_Informer, self).__init__()

        # 1. 依然保留 TCN 作为前置局部特征提取器
        self.tcn = TemporalConvNet(num_inputs=tcn_input_dim, num_channels=tcn_channels)
        tcn_out_dim = tcn_channels[-1]

        # 2. 核心：调用官方的真正 Informer！
        # 将 TCN 的输出维度作为 Informer 的输入维度 (enc_in, dec_in)
        self.informer = Informer(
            enc_in=tcn_out_dim,  # 编码器输入特征数
            dec_in=tcn_out_dim,  # 解码器输入特征数
            c_out=1,  # 最终预测目标数 (Power = 1)
            seq_len=seq_len,
            label_len=label_len,
            out_len=pred_len,  # 预测长度（官方参数名为out_len），Github的Informer2020的官方定义。嗯，真厉害
            factor=5,  # ProbSparse Attention 的采样因子
            d_model=d_model,
            n_heads=n_heads,
            e_layers=e_layers,
            d_layers=d_layers,
            d_ff=d_model * 4,
            dropout=dropout,
            attn='prob',  # 真正的概率稀疏注意力！
            embed='timeF',  # 激活时间特征嵌入！
            freq='t',  # TODO： 频率(h代表小时级别,t代表分钟级别，换数据的时候需要对齐)
            activation='gelu',
            output_attention=False
        )

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec):
        # 1. TCN 提取气象/历史功率的时序突变特征
        # x_enc shape: [Batch, seq_len, features]
        tcn_encoded_x = self.tcn(x_enc)

        # 解码器输入也需要经过同样的处理 (或者简单起见直接用一个线性层映射)
        # 这里为了严谨，我们假设解码器输入也通过 TCN 提取特征
        tcn_decoded_x = self.tcn(x_dec)

        # 2. 将 TCN 提取的高级特征连同时间标记 (x_mark) 一起送入真正的 Informer
        # Informer 内部会自动处理 Positional Encoding 和 Time Encoding
        dec_out = self.informer(tcn_encoded_x, x_mark_enc, tcn_decoded_x, x_mark_dec)

        # 官方 Informer 的输出形状通常是 [Batch, pred_len, c_out]
        return dec_out

# 测试代码是否能跑通
if __name__ == "__main__":
    # 模拟 PCA 降维后保留了 3 个主成分，加上 4 个时间特征，共 7 个特征
    dummy_input = torch.randn(32, 96, 7)  # [Batch, Seq_Len, Features]

    model = TCN_Informer_Model(
        input_dim=7,
        tcn_channels=[32, 64, 128],  # TCN 有三层
        d_model=512,
        n_heads=8,
        e_layers=2,
        pred_len=24
    )

    # 前向传播测试
    dummy_out = model(dummy_input, None, None, None)
    print(f"模型构建成功！输入维度: {dummy_input.shape}, 预测输出维度: {dummy_out.shape}")