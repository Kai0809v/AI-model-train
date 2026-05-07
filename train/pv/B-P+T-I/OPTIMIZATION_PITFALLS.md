# 光伏功率预测模型优化 - 避坑指南

> **核心理念**：排除错误方向本身就是进步。本文档记录了在优化过程中验证失败的配置组合，避免后续重复踩坑。

---

## 📊 基线性能（参考标准）

| 配置 | MSE | RMSE | MAE | R² |
|------|-----|------|-----|-----|
| **基线** | 76.86 | 8.77 | 4.00 | **0.8920** |

**基线配置详情**：
```python
tcn_channels=[16, 32]
d_model=64
n_heads=4
e_layers=2
dropout=0.15
criterion=MSELoss()
optimizer=Adam(lr=0.001, weight_decay=1e-4)
scheduler=CosineAnnealingLR
```

---

## ❌ 已验证的失败方向

### 失败案例 #4：功率滞后特征导致数据泄露 ⚠️ 严重  因为先计算了滞后特征，再划分的。见git V0.7

**尝试配置**：
```python
# 在 PV_part1.py 中添加
for lag_steps in [4, 12, 24]:
    # === 特征工程优化 1: 滞后特征 ===
    # 过去 1h (4步), 3h (12步), 6h (24步) 的功率值
    for lag_steps in [4, 12, 24]:
        lag_hours = lag_steps // 4
        df[f'Power_lag_{lag_hours}h'] = df['Power'].shift(lag_steps)
    
    # === 特征工程优化 2: 滚动统计量 ===
    # 过去 24小时 (96步) 的均值和标准差
    rolling_window = 96  # 24小时 * 4个15分钟
    df['Power_rolling_mean_24h'] = df['Power'].rolling(window=rolling_window, min_periods=1).mean()
    df['Power_rolling_std_24h'] = df['Power'].rolling(window=rolling_window, min_periods=1).std()
    
    # 过去 6小时 (24步) 的均值
    rolling_window_6h = 24
    df['Power_rolling_mean_6h'] = df['Power'].rolling(window=rolling_window_6h, min_periods=1).mean()
    
df['Power_rolling_mean_24h'] = df['Power'].rolling(96).mean()
df['Power_rolling_std_24h'] = df['Power'].rolling(96).std()

# 中间省略部分代码，详情见git V0.7的代码
    print("3. 执行 8:1:1 时序划分...")
    n = len(df)
    train_end = int(n * 0.8)
    val_end = int(n * 0.9)
# ……
    X_train, y_train = X[:train_end], y[:train_end]
    X_val, y_val = X[train_end:val_end], y[train_end:val_end]
    X_test, y_test = X[val_end:], y[val_end:]

    print("4. 标准化处理 (基于训练集拟合)...")
# ……
```

**结果**：
- R²: **虚高至 0.9788**
- RMSE: **虚假降低至 3.88**
- **存在严重的数据泄露问题，指标大幅虚高**

**数据泄露原因分析**：

1. **滞后特征的本质问题**：
   ```
   时间点 t: 预测 Power(t+1), Power(t+2), ..., Power(t+24)
   滞后特征: Power_lag_1h = Power(t-3)  ← 这是过去的值
   
   但在滑动窗口中：
   - seq_x = features[t:t+96]     # 包含 t 到 t+95 的特征
   - target_y = targets[t+96:t+120] # 预测 t+96 到 t+119
   
   问题：这些滞后值是在整个数据集上预先计算的
   测试集的 Power_lag_1h 依赖于测试集内部的过去功率值
   在实际部署时，您无法获得这些"过去值"（因为要预测的就是未来功率）
   ```

2. **滚动统计量更严重**：
   ```python
   df['Power_rolling_mean_24h'] = df['Power'].rolling(window=96).mean()
   ```
   - 滚动均值在时间 `t` = mean(Power[t-95:t])
   - **这个计算是在整个数据集上一次性完成的**
   - 包括训练集、验证集、测试集的所有数据都参与了滚动计算
   - **测试集的滚动统计量"看到"了测试集窗口内的所有数据！**

3. **时序划分顺序错误**：
   ```python
   # 错误的顺序：
   df['Power_lag_1h'] = df['Power'].shift(4)  # 先计算特征
   df['Power_rolling_mean'] = df['Power'].rolling(96).mean()
   
   X_train = X[:train_end]  # 后划分数据集
   X_test = X[val_end:]
   
   # 正确的顺序应该是：
   X_train, X_test = train_test_split(...)  # 先划分
   # 然后在训练集上单独计算滞后和滚动特征
   ```

**为什么指标虚高**：
- 模型"偷看"了过去的功率值来预测未来
- 光伏功率具有强自相关性，知道过去值就能很好地预测未来
- 但这在实际应用中是不可能的（您要预测的就是未来功率）
- **相当于考试前偷偷看了答案**

**正确做法**：
```python
# ✅ 只使用气象特征及其交互项（无数据泄露风险）
feature_cols = [
    'TSI', 'DNI', 'GHI', 'Temp', 'Atmosphere', 'Humidity',
    'TSI_Temp_interaction', 'GHI_Temp_interaction',
    'TSI_Humidity_ratio', 'GHI_Humidity_ratio',
    'DNI_GHI_ratio', 'Temp_squared'
]

# ❌ 移除所有功率相关的滞后和滚动特征
# 'Power_lag_1h', 'Power_lag_3h', 'Power_lag_6h',
# 'Power_rolling_mean_24h', 'Power_rolling_std_24h'
```

**教训**：
> ⚠️ **任何基于目标变量（Power）的历史值构建的特征都存在数据泄露风险**
> 
> 除非：
> 1. 在时序划分**之后**单独为每个集合计算
> 2. 使用严格的滚动窗口，确保不"偷看"未来
> 3. 在实际部署时能够获得这些历史值
>
> **对于光伏功率预测，最安全的做法是只使用气象特征**

---

### 失败案例 #1：过度增强模型容量

**尝试配置**：
```python
tcn_channels=[32, 64, 128]  # 从2层增至3层，通道数翻倍
d_model=128                  # 从64增至128
n_heads=8                    # 从4增至8
e_layers=3                   # 从2增至3
dropout=0.1                  # 从0.15降至0.1
```

**结果**：
- R²: 0.8920 → **0.8675** (下降 2.75%)
- RMSE: 8.77 → **9.71** (上升 10.7%)

**失败原因分析**：
1. **参数量暴增**：TCN通道数增加3倍 + d_model翻倍 = 参数量增加约5-6倍
2. **训练数据不足**：光伏数据集规模有限，大模型容易记住噪声而非学习规律
3. **正则化不足**：dropout从0.15降至0.1，进一步加剧过拟合
4. **同时改动过多变量**：无法定位哪个参数导致性能下降

**教训**：
> ⚠️ **不要同时大幅调整多个超参数**。每次只改1-2个参数，隔离变量影响。

---

### 失败案例 #2：保守缩减模型容量

**尝试配置**：
```python
tcn_channels=[32, 64]       # 中等容量
d_model=96                   # 适中维度
e_layers=2                   # 保持2层
dropout=0.12                 # 略微增强正则化
criterion=HuberLoss(delta=0.3)
optimizer=AdamW(weight_decay=5e-5)
scheduler=CosineAnnealingLR
```

**结果**：
- R²: 0.8920 → **0.8584** (下降 3.78%)
- RMSE: 8.77 → **10.04** (上升 14.5%)
- **训练仅16轮就早停**，说明严重欠拟合

**失败原因分析**：
1. **模型容量不足**：[32,64] 的TCN无法捕捉复杂的气象-功率非线性关系
2. **Huber delta过小**：delta=0.3 过于偏向MAE，削弱了对大误差的惩罚
3. **学习率策略不当**：CosineAnnealingLR在欠拟合情况下无法充分收敛
4. **weight_decay过低**：5e-5 的正则化力度不够

**教训**：
> ⚠️ **模型容量不能盲目削减**。光伏预测需要足够的表达能力来捕捉气象因素的复杂交互。

---

### 失败案例 #3：激进的大模型 + Huber Loss

**尝试配置**：
```python
tcn_channels=[32, 64, 128]
d_model=128
e_layers=3
criterion=HuberLoss(delta=0.5)
optimizer=AdamW(weight_decay=1e-4)
learning_rate=0.001
```

**结果**：
- R²: 0.8920 → **0.8255** (下降 7.45%)
- RMSE: 8.77 → **11.14** (上升 27.0%)
- **训练22轮早停**，Train-Val Loss差距巨大

**失败原因分析**：
1. **严重的过拟合**：Train Loss=0.008 vs Val Loss=0.043（5.4倍差距）
2. **Huber Loss与大容量模型不匹配**：Huber对异常值鲁棒，但大模型本身就容易过拟合
3. **学习率过高**：0.001的学习率对于大模型来说太快，导致震荡
4. **早停机制过早触发**：patience=10，但模型可能需要在更低学习率下继续训练

**教训**：
> ⚠️ **大模型需要更精细的调优**：更低的学习率、更强的正则化、更长的训练时间。

---

## 🔍 关键发现总结

### 1️⃣ 模型容量的"甜蜜点"

```
小模型 [16,32] + d_model=64  → R²=0.892 ✅ 稳定
中模型 [32,64] + d_model=96  → R²=0.858 ❌ 欠拟合
大模型 [32,64,128] + d_model=128 → R²=0.825~0.867 ❌ 过拟合/不稳定
```

**结论**：当前数据集规模下，**基线的小模型配置已经接近最优**。

---

### 2️⃣ 损失函数的选择

| Loss函数 | 适用场景 | 本项目的表现 |
|----------|---------|------------|
| MSELoss | 通用场景，对大误差敏感 | ✅ R²=0.892（基线） |
| HuberLoss(delta=0.3) | 极端异常值多 | ❌ R²=0.858（欠拟合） |
| HuberLoss(delta=0.5) | 平衡鲁棒性与精度 | ❌ R²=0.825~0.867（不稳定） |

**结论**：光伏数据的异常值并不极端到需要Huber Loss，**MSE已经足够**。

---

### 3️⃣ 优化器的选择

| 优化器 | 优势 | 本项目的表现 |
|--------|------|------------|
| Adam | 自适应学习率，稳定 | ✅ 基线成功 |
| AdamW | 解耦权重衰减，理论上更好 | ❌ 未观察到明显优势 |

**结论**：对于当前任务复杂度，**Adam和AdamW差异不大**。

---

### 4️⃣ 学习率调度策略

| 调度器 | 特点 | 本项目的表现 |
|--------|------|------------|
| CosineAnnealingLR | 平滑下降 | ✅ 基线成功 |
| CosineAnnealingWarmRestarts | 周期性重启 | ❌ 导致不稳定 |

**结论**：**标准余弦退火更适合本项目**，重启机制反而引入不必要的波动。

---

## 💡 正确的优化方向建议

基于以上失败经验，我将接下来值得尝试的方向放在了[IMPROVEMENTS.md](IMPROVEMENTS.md)中

---

## 🔑 核心原则总结

1. **单一变量原则**：每次只改1-2个参数
2. **渐进式改进**：小步快跑，避免大幅跳跃
3. **监控过拟合**：Train-Val Loss差距 > 0.01 需警惕
4. **早停耐心**：patience至少设为10-15
5. **记录一切**：每个实验都要有完整记录
6. **回归基线**：如果改进失败，记录并回退到稳定配置

---

**最后更新**: 2026-04-23  
**维护者**: Kai0809v  
**版本**: v0.4
