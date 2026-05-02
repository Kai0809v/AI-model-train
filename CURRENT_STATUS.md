# 当前状态与下一步行动

## 📊 当前配置

**模型配置**（暂时不写特征工程部分）:
```python
seq_len=192, label_len=96, pred_len=24
tcn_channels=[16, 32]
d_model=64
n_heads=4
e_layers=2
dropout=0.15
criterion=MSELoss()
optimizer=Adam(lr=0.001, weight_decay=1e-3)
scheduler=CosineAnnealingLR(T_max=epochs, eta_min=1e-6)
patience=10
batch_size=32
```
### 时间序列参数

| 参数 | 值 | 含义 |
|------|-----|------|
| `seq_len` | 192 | **输入序列长度**：观察过去 2 天（192步 × 15分钟 = 48小时） |
| `label_len` | 96 | **解码器输入长度**：1 天（96步 × 15分钟 = 24小时） |
| `pred_len` | 24 | **预测长度**：未来 6 小时（24步 × 15分钟） |

> 在 Informer 中，`label_len` 是解码器从真实值开始的位置（用于引导），`pred_len` 是纯预测的长度

---

### TCN (Temporal Convolutional Network) 参数

| 参数 | 值 | 含义 |
|------|-----|------|
| `tcn_channels` | `[16, 32]` | TCN 两层的通道数，逐层提取时序特征 |

---

### Informer 核心架构参数

| 参数 | 值 | 含义 |
|------|-----|------|
| `d_model` | 64 | 模型隐藏层维度 |
| `n_heads` | 4 | 多头注意力机制的头数 |
| `e_layers` | 2 | 编码器层数（Transformer 层数） |
| `dropout` | 0.15 | 随机失活率，防止过拟合 |

---

### 训练参数

| 参数 | 值 | 含义 |
|------|-----|------|
| `batch_size` | 32 | 每批次训练的样本数 |
| `learning_rate` | 0.001 | 初始学习率 |
| `weight_decay` | 1e-3 | L2 正则化权重衰减 |
| `patience` | 10 | 早停机制：验证loss 连续 10 次不下降则停止训练 |
| `scheduler` | CosineAnnealingLR | 学习率调度：余弦退火，最低降至 1e-6 |
| `criterion` | MSELoss() | 损失函数：均方误差 |

---

### 物理约束（后处理）

| 约束 | 值 |
|------|-----|
| 夜间阈值 | < 0.05 MW 强制归零 |
| 非负约束 | 预测值 ≥ 0 |
| 装机容量 | ≤ 130 MW |

这些参数共同构成了一个针对光伏功率预测优化的 TCN-Informer 混合模型架构。

---

## ✅ 已完成的改进

1. **物理约束强化**
   - 夜间强制归零（< 0.05 MW）
   - 非负约束
   - 装机容量上限（130 MW）

2. **文档完善**
   - [OPTIMIZATION_PITFALLS.md](OPTIMIZATION_PITFALLS.md) - 避坑指南
   - [IMPROVEMENTS.md](IMPROVEMENTS.md) - 改进方案总览
   - [readme.md](readme.md) - 项目说明更新

3. **已采取的行动**
   - [x] 优先级1，特征工程优化（R²从0.892提升至0.972）
   - [x] 序列长度实验（找到最优seq_len=96，R²进一步提升至0.979）
   - [x] weight_decay优化（R²从0.979提升至0.981）


---

## 最近的输出
使用了`序列长度实验结果对比`得出的最优序列长度,并且调整了optimizer weight_decay=1e-4
```text
Epoch: 50 | Train Loss: 0.01048 | Val Loss: 0.01342
           Learning Rate: 0.000001
EarlyStopping counter: 10 out of 10
🚀 触发早停机制，训练提前结束。

--- 开始测试集评估 ---

📊 最终测试集评估指标:
   MSE: 13.4915
   RMSE: 3.6731
   MAE: 1.7523
   R2: 0.9810
```

**序列长度实验结果对比**:
```text
配置                   R²         RMSE       MAE        MSE       
----------------------------------------------------------------------
1天历史 (96)          0.9788 ✅   3.8847     1.7845     15.0913   ← 最优
1.5天历史 (144)       0.9750      4.2212     1.9537     17.8184
2天历史 (192)         0.9698      4.6339     2.1197     21.4731
3天历史 (288)         0.9738      4.3159     2.0001     18.6271
```
---
## ❌ 已排除的错误方向

详见 [OPTIMIZATION_PITFALLS.md](OPTIMIZATION_PITFALLS.md)

## 🎯 推荐的下一步行动

详见 [IMPROVEMENTS.md](IMPROVEMENTS.md) 中的“下一步优化方案”章节

### 优先级 1：集成学习（简单有效）

**目标**: R² 提升至 0.982~0.984，提升稳定性

**具体任务**:
- [ ] 训练5个不同随机种子的模型
- [ ] 取平均预测结果
- [ ] 评估集成效果

**预计工作量**: 2-3小时

---

### 优先级 2：NRBO自动调优（精细优化）

**目标**: R² 提升至 0.982~0.985

**前提条件**: 
- 当前配置已经非常稳定（R² = 0.9810）
- 已找到最优序列长度

**具体任务**:
```bash
pip install optuna
python nrbo_tuner.py
```

**预计工作量**: 2-4小时（自动运行）

---

### 优先级 3：微调超参数

**可尝试的调整**:
- [ ] 调整 learning_rate: 0.0005 ~ 0.002
- [ ] 调整 weight_decay: 5e-4 ~ 2e-3
- [ ] 调整 dropout: 0.1 ~ 0.2
- [ ] 调整 batch_size: 16, 32, 64


## 📝 快速开始命令

```bash
# 1. 验证基线配置
python PV_part1.py
python PV_part2.py
python quick_test.py

# 2. 查看避坑指南
cat OPTIMIZATION_PITFALLS.md

# 3. 根据优先级选择下一步行动
```
---

## 📝 实验记录模板

每次实验记录以下信息：

### 实验 #X - [简短描述]
**日期**: YYYY-MM-DD  
**改动内容**: 

**结果**:
```text
```

**分析**:

**结论**:
✅ 保留 / ❌ 放弃 / 🔄 需要进一步调整

其他记录见 [EXPERIMENTS.md](EXPERIMENTS.md)

---

**最后更新**: 2026-04-19  
**维护者**: Kai0809v