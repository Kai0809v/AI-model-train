# 光伏功率预测模型 - 最新改进方案

> **当前指标**: MSE=64.51, RMSE=8.03, MAE=3.61, **R²=0.9093**  
> **目标指标**: R² ≥ 0.92 (短期), R² ≥ 0.95 (长期)  
> **最后更新**: 2026-04-23

---

## 📋 已实施的改进（按优先级）

### ✅ 改进 #1: Masked MSE Loss（夜间屏蔽损失）

**实施状态**: ✅ 已完成  
**修改文件**: `PV_part2.py`

#### 核心思想
在训练时忽略夜间样本的误差，让模型专注于学习白天的复杂模式。夜晚不发电是物理规律，不应浪费模型容量拟合传感器噪声。

#### 实现细节
```python
def masked_mse_loss(preds, trues, threshold=0.1):
    """
    掩码MSE损失：只在白天（功率>threshold）计算误差
    """
    mask = (trues > threshold).float()
    squared_error = (preds.squeeze(-1) - trues) ** 2
    masked_error = squared_error * mask
    
    # 防止除零：如果整个batch都是夜晚，退化为普通MSE
    mask_sum = mask.sum()
    if mask_sum < 1:
        return squared_error.mean()
    
    return masked_error.sum() / mask_sum
```

#### 优势分析
- ✅ **零数据泄露风险**：不涉及历史功率值
- ✅ **物理意义明确**：符合"夜晚不发电"的硬约束
- ✅ **聚焦学习资源**：减少模型对噪声的过拟合
- ✅ **实施成本低**：仅修改损失函数

#### 预期提升
- R²: 0.9093 → **0.915~0.925** (+0.6%~1.7%)
- RMSE: 8.03 → **7.5~7.8** (-3%~6%)

#### 阈值选择建议
- 当前阈值: 0.1 MW (约装机容量的0.08%)
- 可调范围: 0.05~0.2 MW
- 调优方法: 观察验证集Loss曲线，选择使验证集R²最高的阈值

---

### ✅ 改进 #2: VIP特征通道（非线性物理特征）

**实施状态**: ✅ 已完成  
**修改文件**: `PV_part1.py`

#### 核心思想
绕过PCA的线性降维，直接将具有强物理意义的非线性特征拼接到主成分上。

#### 新增特征
```python
# 1. 大气透射率（反映云层和大气质量）
Clearness_Index = GHI / TSI

# 2. 温度效率损失（高温降低光伏板转换效率）
Temperature_Efficiency_Loss = Temp × GHI
```

#### 为什么选择这两个特征？
| 特征 | 物理意义 | 与Power的相关性 |
|------|---------|----------------|
| Clearness_Index | 大气透射率，反映云层遮挡程度 | 高（直接影响实际到达地面的辐照度） |
| Temperature_Efficiency_Loss | 温度导致的效率损失 | 中高（光伏板在高温下效率下降） |

#### 为什么不使用原方案的TSI²和Temp³？
- ❌ **TSI²**: 光伏效应接近线性关系，平方项缺乏物理依据
- ❌ **Temp³**: 温度立方项在物理上无明确解释
- ✅ **Clearness_Index**: 气象学标准指标，直接反映大气条件
- ✅ **Temperature_Efficiency_Loss**: 符合光伏板温度系数特性

#### 技术实现
```python
# 步骤1: 构造VIP特征
df['Clearness_Index'] = df['GHI'] / (df['TSI'] + 1e-6)
df['Temperature_Efficiency_Loss'] = df['Temp'] * df['GHI']

# 步骤2: 独立标准化（避免被PCA破坏）
scaler_vip = StandardScaler()
vip_scaled = scaler_vip.fit_transform(vip_features)

# 步骤3: 与PCA输出拼接
X_train_final = np.hstack((X_train_pca, vip_train))
```

#### 优势分析
- ✅ **无数据泄露风险**：纯气象特征变换
- ✅ **保留非线性信息**：绕过PCA的线性限制
- ✅ **物理可解释性强**：基于光伏领域知识

#### 潜在风险
- ⚠️ **可能与现有特征冗余**：需要通过实验验证
- ⚠️ **增加输入维度**：从N维变为N+2维（影响较小）

#### 预期提升
- R²: 0.9093 → **0.912~0.918** (+0.3%~0.9%)
- 若与Masked MSE Loss结合: **0.918~0.928**

---

### 🔧 改进 #3: NRBO自动超参数优化（系统性搜索）

**实施状态**: ✅ 工具已创建，待运行  
**脚本文件**: `nrbo_tuner.py`

#### 核心思想
使用贝叶斯优化（TPE采样器）系统性地搜索超参数空间，避免人工猜测。

#### 搜索空间设计
| 超参数 | 搜索范围 | 类型 | 说明 |
|--------|----------|------|------|
| tcn_channels | {small:[16,32], medium:[32,64], large:[64,128]} | 分类 | TCN通道数 |
| d_model | {64, 96, 128} | 分类 | Informer隐藏层维度 |
| n_heads | {4, 8} | 分类 | 注意力头数 |
| e_layers | [2, 4] | 整数 | 编码器层数 |
| learning_rate | [5e-4, 2e-3] | 对数均匀 | 学习率 |
| weight_decay | [5e-5, 5e-4] | 对数均匀 | L2正则化强度 |
| dropout | [0.1, 0.2] | 浮点 | 随机失活率 |

#### 优化策略
- **采样器**: TPE (Tree-structured Parzen Estimator)
- **剪枝器**: Median Pruner（提前终止表现差的试验）
- **快速评估**: 每个trial只训练5个epoch，使用前10个batch
- **总试验数**: 20次（可根据时间调整）

#### 使用方法
```bash
# 第一步：安装依赖
pip install optuna

# 第二步：运行优化
python nrbo_tuner.py

# 第三步：查看结果
cat best_nrbo_params.json
```

#### 输出文件
- `best_nrbo_params.json`: 最优超参数配置
- `nrbo_optimization_history.png`: 优化历史曲线
- （可选）`best_tcn_informer_nrbo.pth`: 最优模型权重

#### 预期提升
- R²: 0.9093 → **0.915~0.925** (+0.6%~1.7%)
- 若与前两个改进结合: **0.925~0.935**

#### 注意事项
- ⚠️ **运行时间长**: 20次试验约需1-2小时（取决于GPU性能）
- ⚠️ **快速评估可能不准确**: 完整训练后可能有偏差
- ✅ **建议作为最终优化步骤**: 先验证前两个改进的效果

---

## 🎯 推荐的实施路径

### 阶段一：立即验证（今天完成）
```bash
# 1. 重新运行数据处理（应用VIP特征）
python PV_part1.py

# 2. 训练模型（应用Masked MSE Loss）
python PV_part2.py

# 3. 记录新指标
```

**预期结果**:
- 基线: R² = 0.9093
- 改进后: R² ≈ **0.915~0.925**
- 提升幅度: **+0.6%~1.7%**

### 阶段二：深度优化（明天进行）
```bash
# 如果阶段一成功，继续运行NRBO优化
pip install optuna
python nrbo_tuner.py

# 使用最优参数重新训练
# （需要手动将best_nrbo_params.json中的参数填入PV_part2.py）
```

**预期结果**:
- 改进后: R² ≈ **0.925~0.935**
- 累计提升: **+1.6%~2.8%**

### 阶段三：集成学习（可选，追求极致性能）
```python
# 训练5个不同随机种子的模型
seeds = [42, 123, 456, 789, 1024]
predictions = []

for seed in seeds:
    torch.manual_seed(seed)
    np.random.seed(seed)
    train_and_evaluate(...)  # 保存预测结果
    predictions.append(preds)

# 取平均
final_preds = np.mean(predictions, axis=0)
```

**预期结果**:
- 最终: R² ≈ **0.930~0.940**
- 累计提升: **+2.1%~3.4%**

---

## ⚠️ 风险评估与避坑指南

### 高风险操作（已避免）
❌ **不要添加功率滞后特征**  
- 原因：导致严重数据泄露（见OPTIMIZATION_PITFALLS.md案例#4）
- 症状：R²虚高至0.97+，但实际部署时失效

❌ **不要大幅增大模型容量**  
- 原因：参数量暴增导致过拟合（见OPTIMIZATION_PITFALLS.md案例#1）
- 症状：Train Loss低但Val Loss高

❌ **不要盲目降低factor参数**  
- 原因：seq_len=96较短，ProbSparse优势不明显
- 收益不确定，可能适得其反

### 中风险操作（需谨慎）
⚠️ **VIP特征可能冗余**  
- 缓解措施：通过Boruta筛选，如果VIP特征未被选中则移除
- 监控指标：对比有无VIP特征的验证集R²

⚠️ **Masked MSE阈值敏感**  
- 缓解措施：尝试多个阈值（0.05, 0.1, 0.15, 0.2），选择最优
- 监控指标：观察训练/验证Loss曲线的差距

### 低风险操作（推荐优先）
✅ **Masked MSE Loss**  
- 物理意义明确，几乎必然提升

✅ **NRBO优化**  
- 系统性搜索，避免人工偏见

---

## 📊 改进效果追踪表

| 改进项 | 实施状态 | 预期R²提升 | 实际R²提升 | 备注 |
|--------|---------|-----------|-----------|------|
| 基线配置 | ✅ | 0.8920 | 0.8920 | 稳定参考点 |
| 物理约束强化 | ✅ | +0.005~0.01 | +0.0093 | 已达成 |
| **Masked MSE Loss** | ✅ | +0.005~0.015 | 待测试 | **今日重点** |
| **VIP特征通道** | ✅ | +0.003~0.009 | 待测试 | **今日重点** |
| NRBO超参数优化 | 🔧 | +0.006~0.016 | 待测试 | 明日进行 |
| 集成学习 | 📅 | +0.005~0.010 | 待测试 | 可选 |
| **组合改进** | - | **+0.02~0.04** | - | **目标: 0.93+** |

---

## 🔬 实验记录模板

每次运行后请填写此表格：

```markdown
### 实验 #X: [实验名称]
- **日期**: YYYY-MM-DD
- **改动内容**: 
  - [列出具体修改]
- **超参数配置**:
  - tcn_channels: [...]
  - d_model: ...
  - learning_rate: ...
  - ...
- **测试结果**:
  - MSE: ...
  - RMSE: ...
  - MAE: ...
  - R²: ...
- **观察与分析**:
  - [记录训练曲线、过拟合情况等]
- **结论**: [成功/失败/需进一步调优]
```

---

## 💡 下一步行动清单

### 立即执行（今天）
- [x] 实施Masked MSE Loss
- [x] 实施VIP特征通道
- [ ] 运行`python PV_part1.py`重新处理数据
- [ ] 运行`python PV_part2.py`训练并评估
- [ ] 记录新指标并与基线对比

### 短期计划（明天）
- [ ] 如果改进有效，运行NRBO优化
- [ ] 分析最优超参数组合
- [ ] 使用最优参数重新训练完整模型

### 中期计划（本周内）
- [ ] 尝试集成学习（如果需要进一步提升）
- [ ] 撰写实验报告
- [ ] 准备论文/项目文档

### 长期探索（未来方向）
- [ ] 引入天气预报数据作为额外特征
- [ ] 尝试多站点融合（如果有数据）
- [ ] 研究分位数回归（提供预测区间）

---

## 📚 参考资料

1. **OPTIMIZATION_PITFALLS.md**: 已验证的失败方向和避坑指南
2. **IMPROVEMENTS.md**: 详细的改进方案和理论分析
3. **Informer论文**: Zhou et al., "Informer: Beyond Efficient Transformer for Long Sequence Time-Series Forecasting", AAAI 2021
4. **TCN论文**: Bai et al., "An Empirical Evaluation of Generic Convolutional and Recurrent Networks for Sequence Modeling", arXiv 2018

---

**维护者**: Kai0809v  
**版本**: v1.0 (2026-04-23)
