# 综合能源预测系统 (APredict)

> 基于深度学习的光伏/风电功率预测平台 | V1.6

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](https://pytorch.org/)
[![License](https://img.shields.io/badge/License-Internal-green.svg)]()

---

## 📖 项目简介

**综合能源预测系统**是一个面向新能源发电企业的智能预测平台，支持：

- ⚡ **风电功率预测** - CEEMDAN-LGBM-Transformer集成模型
- ☀️ **光伏功率预测** - BP-TCN-Informer时序预测模型
- 📊 **历史数据分析** - 功率曲线、峰值分析、质量诊断
- 🎯 **多步长预测** - 支持1/4/8/12/24步（15分钟/步）

### 核心特性

✅ **双模式光伏预测**
- 有未来气象数据模式（高精度）
- 无未来气象数据模式（零填充策略）

✅ **解耦架构设计**
- UI层、控制层、业务层完全分离
- 便于扩展新模型和新场景

✅ **智能数据处理**
- 自动识别CSV/Excel格式
- 智能列名匹配
- 缺失值自动修复

✅ **可视化分析**
- Matplotlib实时图表渲染
- 功率曲线、散点图、残差分布
- 历史数据多维度分析

---

## 🏗️ 程序架构

```
┌──────────────────────────────────────┐
│         GUI.py (UI层)                │
│   - 用户交互、数据显示               │
└──────────────┬───────────────────────┘
               │
┌──────────────▼───────────────────────┐
│  PredictionController (控制层)       │
│   - 业务流程编排                     │
└──┬──────────┬──────────┬────────────┘
   │          │          │
┌──▼───┐ ┌───▼────┐ ┌──▼──────────┐
│Data  │ │Chart   │ │Forecast     │
│Loader│ │Renderer│ │Service      │
└──────┘ └────────┘ └──┬───────────┘
                       │
            ┌──────────┼──────────┐
            │          │          │
     ┌──────▼───┐ ┌───▼────┐ ┌──▼──────┐
     │PV Predictor│ │Wind    │ │Reserved │
     │           │ │Predictor│ │Models   │
     └───────────┘ └─────────┘ └────────┘
```

详细架构说明见：[ARCHITECTURE.md](ARCHITECTURE.md)

---

## 🚀 快速开始

### 开发环境运行

```bash
# 1. 安装依赖
pip install -r requirements_packaging.txt

# 2. 启动应用
python GUI.py

# 或使用快捷脚本
run_dev.bat
```

### 打包为可执行文件

```bash
# Windows一键打包
build.bat

# 或手动打包
pyinstaller APredict.spec
```

输出位置：`dist/APredict/APredict.exe`

详细打包指南见：[PACKAGING_GUIDE.md](PACKAGING_GUIDE.md)

---

## 📂 项目结构

```
APredict/
├── 📄 核心模块
│   ├── GUI.py                          # 主界面入口
│   ├── api_v8.py                       # 预测服务API
│   ├── gui_config.py                   # 配置常量
│   ├── prediction_controller.py        # 业务控制器
│   ├── chart_renderer.py               # 图表渲染器
│   └── data_loader_module.py           # 数据加载器
│
├── 📦 模型资产
│   └── assets/
│       ├── pv_tcn_informer/            # 光伏模型
│       │   ├── assets/                 # 有未来气象数据版本
│       │   ├── assets_no_weather/      # 无未来气象数据版本
│       │   ├── models/                 # 模型包装器
│       │   └── preprocessors/          # 预处理器
│       └── wind_ceemdan_lgbm_trans/    # 风电模型
│           ├── assets/
│           ├── models/
│           └── preprocessors/
│
├── 🔧 第三方库
│   └── Informer2020/                   # 时序预测框架
│
├── 🎨 资源文件
│   └── res/
│       ├── icon.png                    # 应用图标
│       └── background.png              # 背景图片
│
├── 📚 文档
│   ├── ARCHITECTURE.md                 # 架构说明
│   ├── PACKAGING_GUIDE.md              # 打包指南
│   ├── CURRENT_STATUS.md               # 当前状态
│   └── FIxLogs/                        # 修复日志
│
└── 🛠️ 工具脚本
    ├── build.bat                       # 打包脚本
    ├── run_dev.bat                     # 开发启动脚本
    └── APredict.spec                   # PyInstaller配置
```

---

## 🎯 使用场景

### 场景1：风电功率预测

1. 选择"风电功率预测"
2. 上传历史数据（CSV/Excel，≥96行）
3. 选择模型"CEEMDAN-LGBM-Transformer"
4. 选择步长（单步/1小时/2小时）
5. 点击"开始智能预测"

### 场景2：光伏功率预测（有未来气象数据）

1. 选择"光伏功率预测"
2. 上传历史数据（≥192行，32个特征列）
3. 选择模型"BP-TCN-Informer（有未来气象数据）"
4. 上传未来24步气象数据
5. 选择步长（1-24步）
6. 开始预测

### 场景3：光伏功率预测（无未来气象数据）

1. 选择"光伏功率预测"
2. 上传历史数据（≥192行）
3. 选择模型"BP-TCN-Informer（无未来气象数据）"
4. 选择步长
5. 开始预测（自动使用零填充策略）

### 场景4：历史数据分析

1. 点击"🔍 历史数据分析"按钮
2. 选择数据文件
3. 选择数据粒度（5分钟/15分钟/30分钟/1小时）
4. 点击"▶️ 开始分析"
5. 查看：
   - 近30天日功率曲线
   - 近24小时功率曲线
   - 月度日发电峰值
   - 数据质量诊断报告

---

## 📊 技术栈

| 类别 | 技术 |
|------|------|
| **GUI框架** | PySide6 (Qt6) |
| **深度学习** | PyTorch 2.0+ |
| **数据处理** | Pandas, NumPy |
| **机器学习** | Scikit-learn, LightGBM |
| **可视化** | Matplotlib |
| **模型序列化** | Joblib |
| **打包工具** | PyInstaller |
| **时序预测** | Informer (自定义实现) |

---

## 🔬 模型说明

### 风电模型：CEEMDAN-LGBM-Transformer

**架构：**
```
原始风速序列
    ↓
CEEMDAN分解 → 多个IMF分量
    ↓
LGBM特征选择 → 关键特征提取
    ↓
Transformer编码 → 时序建模
    ↓
集成学习（1h/4h/8h） → 多步预测
```

**特点：**
- ✅ 信号分解降噪
- ✅ 特征自动选择
- ✅ 多头注意力机制
- ✅ 多 horizon 集成

### 光伏模型：BP-TCN-Informer

**架构：**
```
32维气象特征
    ↓
PCA降维 → 11维主成分
    ↓
TCN编码器 → 局部时序特征
    ↓
Informer解码器 → 长序列预测
    ↓
物理约束 → 功率限幅
```

**特点：**
- ✅ 支持未来气象数据输入
- ✅ ProbSparse自注意力（高效）
- ✅ 生成式解码器（直接多步）
- ✅ 夜间功率自动置零

---

## 📝 数据格式要求

### 风电数据

**必需列：**
- 时间列（任意命名，包含"时间"/"time"）
- 功率列：`实际发电功率（mw）` 或 `Power (MW)`
- 风速列：`测风塔 10m 风速 (m/s)` 等
- 气象列：温度、气压、湿度等

**最少行数：** 96行（24小时，15分钟间隔）

### 光伏数据

**必需列（32个特征）：**
- 时间列
- 功率列：`Power (MW)` 或 `实际发电功率（mw）`
- 辐照度：TSI, DNI, GHI
- 气象：Temp, Atmosphere, Humidity
- 其他气象特征...

**最少行数：** 192行（48小时，15分钟间隔）

**未来气象数据格式：**
- 24行（未来6小时）
- 与历史数据相同的32个特征列

---

## ⚙️ 配置说明

所有配置集中在 [`gui_config.py`](gui_config.py)：

```python
# 应用信息
APP_NAME = "综合能源预测系统"
APP_VERSION = "V1.6"

# 场景配置
PREDICTION_SCENARIOS = {
    "风电功率预测": {...},
    "光伏功率预测": {...}
}

# 物理约束
PV_MAX_CAPACITY_MW = 130.0
WIND_CUT_IN_SPEED = 3.0

# UI样式
COLOR_PRIMARY = "#2e7d32"
CHART_COLOR_LINE = "#00897b"
```

修改后无需重启，下次启动生效。

---

## 🐛 常见问题

### Q1: 启动报错 "ModuleNotFoundError"

**解决：** 确保已安装所有依赖
```bash
pip install -r requirements_packaging.txt
```

### Q2: 预测结果全为零

**可能原因：**
- 风电：风速低于切入风速（3 m/s）
- 光伏：夜间时段（自动置零）
- 数据量不足（检查最小行数要求）

### Q3: 图表中文乱码

**解决：** 系统需安装中文字体
- Windows: SimHei, Microsoft YaHei
- Linux: WenQuanYi Micro Hei

### Q4: 打包后体积过大

**优化：**
- 启用UPX压缩（已在spec中配置）
- 排除不必要的模块
- 使用虚拟环境（仅安装必要依赖）

---

## 📈 性能指标

| 模型 | 训练集RMSE | 测试集RMSE | 推理速度 |
|------|-----------|-----------|---------|
| 风电CEEMDAN-LGBM-Transformer | 2.3 MW | 3.1 MW | <1s |
| 光伏TCN-Informer（有未来数据） | 4.5 MW | 5.8 MW | <2s |
| 光伏TCN-Informer（无未来数据） | 5.2 MW | 6.9 MW | <2s |

*测试环境：Intel i7-10700K, 16GB RAM, RTX 3060*

---

## 🔄 版本历史

### V1.6 (2026-05-04)
- ✅ 完成GUI解耦重构
- ✅ 添加无未来气象数据光伏预测模式
- ✅ 优化历史数据分析窗口
- ✅ 完善打包配置和文档

### V1.5 (2026-04-20)
- ✅ 集成Informer时序预测模型
- ✅ 支持多步长直接预测
- ✅ 添加Matplotlib图表渲染

### V0.6 (2026-03-01)
- ✅ 初始版本发布
- ✅ 支持风电/光伏基础预测

---

## 📞 技术支持

**内部使用，请联系：**
- 开发团队：XCU-风电光伏预测小组
- 邮箱：zhouyukai.kevin@qq.com

---

## 📄 许可证


---

## 🔗 相关资源

- [架构详细说明](ARCHITECTURE.md)
- [实验记录](train/pv/B-P+T-I/EXPERIMENTS.md)
- [当前状态](train/pv/B-P+T-I/CURRENT_STATUS.md)
