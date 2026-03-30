
# 🌬️ CEEMDAN-LightGBM-Transformer 风电功率智能预测系统

[🇺🇸 English Version](./readme-en.md) | 🇨🇳 中文版

<div align="center">

![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-1.9+-ee4c2c.svg)
![License](https://img.shields.io/badge/License-MIT-green.svg)

**基于混合深度学习架构的风电功率高精度预测系统**

</div>

---

## 📖 目录

- [项目简介](#-项目简介)
- [核心特性](#-核心特性)
- [技术架构](#-技术架构)
- [安装指南](#-安装指南)
- [快速开始](#-快速开始)
- [项目结构](#-项目结构)
- [使用说明](#-使用说明)
- [模型训练](#-模型训练)
- [性能指标](#-性能指标)
- [二次开发](#-二次开发)
- [常见问题](#-常见问题)
- [许可证](#-许可证)

---

## 🎯 项目简介

本项目是一个**工业级风电功率预测系统**，采用先进的 **CEEMDAN-LightGBM-Transformer** 混合深度学习架构。系统通过 CEEMDAN 信号分解技术去除噪声干扰，结合 Transformer 注意力机制和 LightGBM 特征选择，实现了对风电功率的高精度预测。

系统提供**友好的图形界面（GUI）**，支持单步预测和多步滚动预测，并内置物理规则校验（如切入风速限制），确保预测结果符合实际发电规律。

### 应用场景
- ⚡ 风电场发电计划制定
- ⚡ 电网调度与储能配置
- ⚡ 电力市场交易决策
- ⚡ 风机运维策略优化

---

## ✨ 核心特性

### 🔥 技术创新
- **CEEMDAN 自适应噪声分解**：有效剔除原始功率数据中的高频噪声，提取纯净趋势分量
- **LightGBM 空间特征降维**：智能筛选关键气象特征，提升模型泛化能力
- **Transformer 注意力机制**：捕捉长序列时间依赖关系，动态关注关键时间步
- **位置编码增强**：引入时间周期特征（日内/周内），提升时序建模能力

### 🛡️ 工程优化
- **双重物理规则校验**：
  - 切入风速拦截（< 3.0 m/s 直接判定为 0）
  - 负功率修正（强制置 0，符合物理规律）
- **智能数据清洗**：自动检测并修复 NaN 值，支持前向/后向填充和均值填补
- **早停机制 + 学习率调度**：防止过拟合，加速收敛
- **MinMaxScaler 目标归一化**：将功率严格压缩至 [0, 1] 区间，提升训练稳定性

### 💻 用户体验
- **现代化 GUI 界面**：新能源风格主题，支持登录/注册功能
- **可视化图表**：Matplotlib 嵌入，实时展示预测曲线和误差分布
- **多步预测**：支持 1 步（十五分钟）、4 步（1 小时）、8 步（2 小时）、16 步（4 小时）滚动预测
- **日志系统**：实时显示模型加载、数据校验、推理进度等信息

---

## 🏗️ 技术架构

### 整体流程
```
mermaid
graph LR
    A[原始气象数据] --> B[CEEMDAN 去噪]
    B --> C[物理特征衍生]
    C --> D[LightGBM 特征选择]
    D --> E[Transformer 编码]
    E --> F[注意力池化]
    F --> G[全连接回归]
    G --> H[反归一化]
    H --> I[物理规则校验]
    I --> J[最终预测]
```
### 数据流
```
mermaid
graph TD
    subgraph 训练阶段
        T1[历史功率 + 气象数据] --> T2[CEEMDAN 分解]
        T2 --> T3[剔除高频 IMF 分量]
        T3 --> T4[干净标签 Y_clean]
        T4 --> T5[训练 Transformer]
    end
    
    subgraph 预测阶段
        P1[实时气象输入] --> P2[特征衍生]
        P2 --> P3[标准化 + 降维]
        P3 --> P4[模型推理]
        P4 --> P5[风速拦截]
        P5 --> P6[负值修正]
        P6 --> P7[输出功率]
    end
```
---

# 📦 安装指南
### 系统要求
- 操作系统：Windows 10/11、Linux、macOS
- Python 版本：3.8 及以上
- GPU 加速（可选）：NVIDIA CUDA 11.0+

---
💡 **提示**
普通用户：只需安装 `torch, numpy, pandas, matplotlib, scikit-learn, joblib, PySide6` 即可运行 GUI 进行预测
研究人员：如需重新训练模型或修改算法，请额外安装 `PyEMD` 和 `lightgbm` 等

---

### 按需安装依赖（无需安装全部库）
#### 场景1：仅使用 GUI/API 预测（普通用户首选）
仅安装**运行必备依赖**，无需训练相关库：
```bash
# 第一步：创建并激活虚拟环境（推荐）
conda create -n wind_forecast python=3.9
conda activate wind_forecast

# 第二步：安装核心运行依赖
pip install torch numpy pandas matplotlib scikit-learn joblib PySide6
```

#### 场景2：模型训练 / 二次开发（研究人员）
在场景1基础上，**补充安装训练/开发依赖**：
```bash
# 额外安装训练专用库
pip install PyEMD lightgbm
```

### 验证安装
```bash
# 验证基础依赖（普通用户）
python -c "import torch, pandas, PySide6; print('基础依赖安装成功')"

# 验证全量依赖（开发/训练用户）
python -c "import PyEMD, lightgbm; print('全量依赖安装成功')"
```

## 🚀 快速开始

### 方式 1：运行 GUI 界面（推荐普通用户）
```
bash
python GUI.py
```
启动后将看到登录界面，默认无需密码即可进入主界面。在主界面中：
1. 选择 CSV 历史数据文件
2. 选择预测场景（当前支持风电功率预测）
3. 选择算法模型（CEEMDAN-LGBM-Transformer）
4. 选择预测步长（单步/1 小时/2 小时/4 小时）
5. 点击"🚀 开始智能预测"

### 方式 2：调用 API 接口（面向开发者）
```
python
from api_v5 import ForecastService
import pandas as pd

# 初始化服务
service = ForecastService(base_models_dir="pretrained")

# 读取历史数据
df = pd.read_csv("your_data.csv")

# 单步预测
result = service.run("CEEMDAN_LGBM_Transformer", df, steps=1)
print(f"预测功率：{result['prediction']:.2f} MW")

# 多步预测（未来 4 步）
result = service.run("CEEMDAN_LGBM_Transformer", df, steps=4)
print(f"未来 4 步预测：{result['predictions']}")
```
---

## 📁 项目结构

```

CEEMDAN/
├── GUI.py                          # 🖼️ 图形用户界面（面向用户）
├── api_v5.py                       # 🔌 预测服务 API 接口
│
├── train/                          # 🎓 训练模块目录
│   ├── part1_v6_stable.py          # 数据预处理 + CEEMDAN 去噪
│   └── part2_v6_stable.py          # Transformer 模型训练
│
├── pretrained/                     # 📦 预训练模型仓库
│   └── wind_ceemdan_lgbm_trans/    # CEEMDAN-LGBM-Transformer 模型
│       ├── scaler_x                # 特征标准化器
│       ├── scaler_y                # 目标归一化器
│       ├── selected_features_indices.npy  # 特征选择掩码
│       └── transformer_weights_single_minmax.pth  # 模型权重
│
├── res/                            # 🎨 资源文件
│   ├── background.png              # GUI 背景图
│   └── icon.png                    # 应用图标
│
└── README.md                       # 📖 本说明文档
```
---

## 📖 使用说明

### 数据格式要求
CSV 文件需包含以下列（列名必须完全匹配）：

| 列名 | 说明 | 单位 |
|------|------|------|
| `测风塔 10m 风速 (m/s)` | 10 米高度风速 | m/s |
| `测风塔 30m 风速 (m/s)` | 30 米高度风速 | m/s |
| `测风塔 50m 风速 (m/s)` | 50 米高度风速 | m/s |
| `测风塔 70m 风速 (m/s)` | 70 米高度风速 | m/s |
| `轮毂高度风速 (m/s)` | 风机轮毂高度风速 | m/s |
| `测风塔 10m 风向 (°)` | 10 米高度风向 | ° |
| `测风塔 30m 风向 (°)` | 30 米高度风向 | ° |
| `测风塔 50m 风向 (°)` | 50 米高度风向 | ° |
| `测风塔 70m 风向 (°)` | 70 米高度风向 | ° |
| `轮毂高度风向 (°)` | 风机轮毂高度风向 | ° |
| `温度 (°)` | 环境温度 | °C |
| `气压 (hPa)` | 大气压强 | hPa |
| `湿度 (%)` | 相对湿度 | % |
| `实际发电功率（mw）` | 历史有功功率 | MW |

### 示例数据格式
```
csv
测风塔 10m 风速 (m/s),测风塔 30m 风速 (m/s),...,实际发电功率（mw）
6.2,6.8,...,12.5
6.3,6.9,...,12.8
...
```
### GUI 操作流程
1. **登录系统**：输入任意用户名和密码（无需注册，直接点击登录）
2. **选择数据**：点击"浏览"按钮选择 CSV 文件
3. **配置参数**：
   - 预测场景：风电功率预测
   - 算法模型：CEEMDAN-LGBM-Transformer
   - 预测步长：单步/多步
4. **开始预测**：点击按钮后等待进度条完成
5. **查看结果**：
   - 左侧显示预测功率值（MW）
   - 中间显示预测曲线图
   - 底部显示系统日志

---

## 🎓 模型训练

### 前置准备
确保已准备好完整的历史数据文件 `wind_data.csv`，格式符合上述要求。

### 步骤 1：数据预处理
```
bash
cd train
python part1_v6_stable.py
```
**执行内容**：
- 读取 CSV 数据并划分训练集/测试集（9:1）
- CEEMDAN 分解目标功率，剔除高频噪声（drop_k=1）
- 构建物理衍生特征（风速立方、空气密度、时间周期等）
- 标准化处理（StandardScaler for X, MinMaxScaler for Y）
- 滑动窗口切片（window_size=96，即 24 小时历史数据）
- 保存处理后的数据到根目录

**输出文件**：
- `train_set`, `train_label`：训练集输入和干净标签
- `test_set`, `test_label_clean`, `test_label_raw`：测试集数据

### 步骤 2：模型训练
```
bash
python part2_stable.py
```
**执行内容**：
- LightGBM 特征重要性评估与降维
- 构建 Transformer 编码器（d_model=256, 4 层 encoder）
- 训练模型（Adam 优化器，初始 lr=0.0005）
- 早停机制（patience=15）+ ReduceLROnPlateau 调度
- 在测试集上评估 RMSE、MAE、R²
- 保存最佳模型权重到 `pretrained/wind_ceemdan_lgbm_trans/`

**训练耗时**：
- CPU: 约 30-60 分钟
- GPU: 约 5-10 分钟

### 调整超参数
如需修改训练配置，编辑 `train/part1_v6_stable.py` 和 `part2_stable.py`：

```
python
# part1_v6_stable.py
DATA_SIZE = 11520      # 训练数据量（行数）
WINDOW_SIZE = 96       # 历史窗口长度（24 小时）
SPLIT_RATE = 0.9       # 训练集比例

# part2_stable.py
batch_size = 64        # 批次大小
epochs = 50            # 最大训练轮数
early_stop_patience = 15  # 早停耐心值
```
---

## 📊 性能指标

在独立测试集上的表现（对抗真实高频噪声）：

| 指标 | 数值 | 说明 |
|------|------|------|
| **RMSE** | 10.28 MW | 均方根误差 |
| **MAE** | 6.68 MW | 平均绝对误差 |
| **R²** | 0.9709 | 决定系数（越接近 1 越好） |

### 对比实验
- **未去噪 baseline**：R² ≈ 0.92
- **CEEMDAN 去噪后**：R² ≈ 0.97（↑ 5.4% 提升）
- **无物理规则拦截**：低风速时出现负功率异常
- **启用双重校验**：100% 符合物理规律

---

## 🔧 二次开发

### 扩展新算法
1. 在 `api_v5.py` 中定义新的 Predictor 类（参考 `CNN_LSTM_Attention_Predictor` 骨架）
2. 实现 `predict()` 和 `predict_multi()` 方法
3. 在 `ForecastService._model_registry` 中注册新模型
4. 训练新模型并保存到 `pretrained/` 目录

### 修改网络结构
编辑 `train/part2_v6_stable.py` 中的 `TransformerModel` 类：
```
python
class TransformerModel(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.embedding = nn.Linear(input_dim, 256)  # 修改隐藏层维度
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=256, nhead=8,  # 调整注意力头数
            dim_feedforward=512,   # 修改前馈网络维度
            dropout=0.1
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=4)  # 修改层数
```
### 自定义特征工程
编辑 `train/part1_v6_stable.py` 中的 `add_physics_features()` 函数，添加领域特定特征。

---

## ❓ 常见问题

### Q1: 提示 "No module named 'PyEMD'"
**A**: 安装 PyEMD 包：
```
bash
pip install PyEMD
# 如果失败，尝试：
pip install emd-signal
```
### Q2: GUI 启动后显示黑屏/闪退
**A**: 
1. 检查是否安装了 PySide6：`pip show PySide6`
2. 确保 `res/background.png` 和 `res/icon.png` 存在
3. 临时注释掉 GUI.py 末尾的背景图设置代码

### Q3: 预测结果出现负值
**A**: 这是正常现象，API 已内置负值修正逻辑。如需调整，修改 `api_v5.py` 第 272-274 行：
```
python
if final_pred < 0:
    final_pred = 0.0  # 可改为警告或其他处理逻辑
```
### Q4: 如何使用自己的数据集？
**A**: 
1. 确保 CSV 列名与要求完全一致
2. 修改 `part1_v6_stable.py` 中的 `DATA_PATH`
3. 重新运行训练流程


---

## 📮 联系方式

如有问题或建议，欢迎通过以下方式联系：

- 📧 Email: zhouyukai.kevin@qq.com

---

<div align="center">

**⭐ 如果这个项目对你有帮助，请我们支持！⭐**

西昌学院机器学习风力发电预测小组

</div>
