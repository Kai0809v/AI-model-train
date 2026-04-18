# 光伏功率预测模型改进方案

### 目前的指标见 [CURRENT_STATUS.md](CURRENT_STATUS.md) 中的最近的输出

---

## ✅ 已尝试的改进
1~3点已踩坑，详见 [OPTIMIZATION_PITFALLS.md](OPTIMIZATION_PITFALLS.md)
### 1. **损失函数优化** 
**改动位置**: `PV_part2.py` 第96-98行

```python
# 从 MSE 改为 Huber Loss (delta=0.5)
criterion = nn.HuberLoss(delta=0.5)
```

**原理**: 
- Huber Loss 结合了 MSE 和 MAE 的优点
- 对小误差使用平方损失（保持梯度敏感）
- 对大误差使用线性损失（降低异常值影响）
- 特别适合光伏数据中的极端天气情况

**预期提升**: R² +0.01~0.02

---

### 2. **模型容量增强**
**改动位置**: `PV_part2.py` 第79-94行

| 参数 | 修改前 | 修改后 | 理由 |
|------|--------|--------|------|
| TCN通道 | [16, 32] | [32, 64, 128] | 增强局部特征提取能力 |
| d_model | 64 | 128 | 提升注意力机制表达能力 |
| n_heads | 4 | 8 | 更多注意力视角 |
| e_layers | 2 | 3 | 更深层的时序建模 |
| dropout | 0.15 | 0.1 | 配合其他正则化手段适度降低 |

**预期提升**: R² +0.02~0.03

---

### 3. **优化器升级**
**改动位置**: `PV_part2.py` 第103-108行

```python
# 从 Adam 升级为 AdamW
optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-4)

# 从普通余弦退火升级为带热重启的版本
scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
    optimizer, T_0=10, T_mult=2, eta_min=1e-6
)
```

**优势**:
- **AdamW**: 解耦权重衰减，更好的泛化性能
- **CosineAnnealingWarmRestarts**: 周期性学习率重启，帮助跳出局部最优

**预期提升**: R² +0.01~0.015

---

### 4. **物理约束强化**
**改动位置**: `PV_part2.py` 第203-218行

```python
# 约束1: 夜间强制归零
night_mask = (trues_inverse < 0.05)
preds_inverse[night_mask] = 0.0

# 约束2: 非负约束
preds_inverse = np.maximum(0, preds_inverse)

# 约束3: 装机容量上限（新增）
MAX_CAPACITY = 130.0  # MW
preds_inverse = np.minimum(preds_inverse, MAX_CAPACITY)
```

**意义**: 确保预测结果符合物理规律，消除不合理的极端值

**预期提升**: R² +0.005~0.01

---

## 🚀 高级改进：NRBO自动超参数优化

### 使用方法

#### 第一步：安装依赖
```bash
pip install optuna
```

#### 第二步：运行NRBO优化器
```bash
python nrbo_tuner.py
```

这将执行50次试验，自动搜索最优超参数组合。

#### 第三步：查看结果
优化完成后会生成：
- `best_nrbo_params.json`: 最优超参数配置
- `nrbo_optimization_history.png`: 优化历史曲线
- `nrbo_param_importance.png`: 参数重要性分析
- `best_tcn_informer_nrbo.pth`: 最优模型权重

### NRBO搜索空间

| 超参数 | 搜索范围 | 类型 |
|--------|----------|------|
| tcn_num_layers | 2-4 | 整数 |
| tcn_base_channels | {16, 32, 64} | 分类 |
| d_model | {64, 128, 256} | 分类 |
| n_heads | {4, 8} | 分类 |
| e_layers | 2-4 | 整数 |
| learning_rate | 1e-4 ~ 1e-2 | 对数均匀 |
| weight_decay | 1e-5 ~ 1e-3 | 对数均匀 |
| dropout | 0.05-0.2 | 浮点 |
| huber_delta | 0.1-1.0 | 浮点 |
| seq_len_option | {96, 192, 288} | 分类 |

### 核心算法
- **采样器**: TPE (Tree-structured Parzen Estimator)
- **剪枝策略**: Median Pruner（中值剪枝，提前终止差劲试验）
- **目标函数**: 验证集 R²

**预期提升**: R² +0.03~0.05（相比基线可达 **0.92~0.94**）

---

## Next Steps

### 1. **数据层面**
- **增加气象预报数据**: 引入未来时刻的天气预报作为额外特征
- **多站点融合**: 如果有多个光伏电站数据，可构建多任务学习
- **数据增强**: 对历史数据进行时间平移、噪声注入等增强

### 2. **特征工程**
- **非线性特征交互**: 使用多项式特征或核方法捕捉辐照度与温度的非线性关系
- **滞后特征**: 添加过去1h、3h、6h的功率滞后项
- **滚动统计量**: 过去24小时的均值、标准差、最大值

### 3. **模型架构**
- **注意力机制增强**: 
  - 尝试 AutoCorrelation (Autoformer)
  - 或使用 Flow Forecasting 的 Cross Attention
- **多尺度TCN**: 并行不同膨胀率的TCN分支
- **残差连接优化**: 在TCN和Informer之间添加跳跃连接

### 4. **训练策略**
- **课程学习**: 先从简单样本（晴天）开始训练，逐步加入复杂样本
- **对抗训练**: 添加微小扰动提高鲁棒性
- **集成学习**: 训练多个模型取平均

### 5. **后处理**
- **卡尔曼滤波**: 对预测结果进行时序平滑
- **分位数回归**: 提供预测区间而非点估计
- **误差校正模型**: 训练一个小型模型专门修正系统性偏差

---


## ⚠️ 注意事项

1. **过拟合监控**: 增大模型容量后需密切关注验证集表现
2. **训练时间**: NRBO优化可能需要数小时（取决于GPU性能）
3. **随机种子**: 设置固定种子以保证实验可复现性
4. **硬件要求**: 建议使用至少8GB显存的GPU

---

## 📝 快速开始

```bash
# 1. 先运行基础改进版本
python PV_part1.py          # 特征工程
python PV_part2.py          # 训练+评估（已包含改进）

# 2. 如果效果满意，可跳过；否则继续
pip install optuna
python nrbo_tuner.py        # NRBO自动调优
```

---


**最后更新**: 2026-04-18  
**维护者**: Kai0809v
