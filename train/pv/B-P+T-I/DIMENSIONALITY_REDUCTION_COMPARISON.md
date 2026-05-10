# 降维方法对比实验指南

## 📋 实验概述

本实验对比三种降维方法在光伏功率预测任务中的表现：

1. **PCA (Principal Component Analysis)** - 主成分分析
   - 线性降维方法
   - 保留最大方差方向
   - 计算效率高，解释性强

2. **KPCA (Kernel PCA)** - 核主成分分析
   - 非线性降维方法
   - 通过核函数映射到高维空间后再降维
   - 能捕捉数据的非线性结构

3. **SPCA (Sparse PCA)** - 稀疏主成分分析
   - 带稀疏约束的 PCA
   - 每个主成分只使用少量原始特征
   - 提高模型可解释性

---

## 🚀 快速开始

### 步骤 1: 生成三种降维方法的数据文件

```bash
python PV_part1_comparison.py
```

这个脚本会：
- 读取原始数据 `D:\APredict\data\PV130MW.xlsx`
- 执行 Boruta 特征选择
- 分别用 PCA、KPCA、SPCA 进行降维
- 生成三个数据文件：
  - `processed_data/model_ready_data_pca.pkl`
  - `processed_data/model_ready_data_kpca.pkl`
  - `processed_data/model_ready_data_spca.pkl`

### 步骤 2: 运行对比训练实验

```bash
python PV_part2_comparison.py
```

这个脚本会：
- 依次训练三个模型（使用相同的超参数）
- 评估测试集性能
- 生成可视化结果
- 输出对比表格

---

## 📊 预期输出

### 1. 模型文件
- `best_tcn_informer_pca.pth` - PCA 方法的最佳模型
- `best_tcn_informer_kpca.pth` - KPCA 方法的最佳模型
- `best_tcn_informer_spca.pth` - SPCA 方法的最佳模型

### 2. 可视化图片
- `prediction_result_pca.png` - PCA 方法的预测曲线
- `prediction_result_kpca.png` - KPCA 方法的预测曲线
- `prediction_result_spca.png` - SPCA 方法的预测曲线
- `training_comparison.png` - 三种方法的训练损失对比图

### 3. 结果文件
- `dimensionality_reduction_comparison.csv` - 详细的指标对比表格

### 4. 控制台输出
如下参数的结果
```python
    # 测试三种降维方法
    methods_to_test = [
        ('pca', {'n_components': 0.95}),
        ('kpca', {'n_components': 10, 'kernel': 'rbf', 'gamma': 0.1}),
        ('spca', {'n_components': 10, 'alpha': 1.0}),
    ]
```
```
================================================================================
实验结果汇总对比
================================================================================

方法         MSE          RMSE         MAE          R²           训练时间(s)     
--------------------------------------------------------------------------------
PCA        12.6391      3.5552       1.7088       0.9822       811.47      
KPCA       34.2865      5.8555       2.6974       0.9518       2137.86     
SPCA       11.3564      3.3699       1.6258       0.9840       1250.52     

🏆 最佳方法: SPCA (RMSE: 3.3699)
```

---

## ⚙️ 参数调整

### 修改降维参数

编辑 `PV_part1_comparison.py` 文件的最后部分：

```python
methods_to_test = [
    ('pca', {'n_components': 0.95}),  # 保留 95% 方差
    ('kpca', {'n_components': 10, 'kernel': 'rbf', 'gamma': 0.1}),
    ('spca', {'n_components': 10, 'alpha': 1.0}),
]
```

**KPCA 可选核函数：**
- `'linear'` - 线性核（等同于 PCA）
- `'poly'` - 多项式核
- `'rbf'` - 高斯核（推荐）
- `'sigmoid'` - Sigmoid 核

**SPCA 参数说明：**
- `n_components`: 降维后的维度数
- `alpha`: 稀疏系数，越大越稀疏（推荐范围 0.1-10）

### 修改训练参数

编辑 `PV_part2_comparison.py` 的 `compare_all_methods` 调用：

```python
result = train_and_evaluate(
    pkl_path=pkl_path,
    reduction_method=method_name,
    epochs=50,              # 训练轮数
    learning_rate=0.001     # 学习率
)
```

---

## 🔍 结果分析要点

### 1. 预测精度对比
- **RMSE**（均方根误差）：越小越好，主要评价指标
- **R²**（决定系数）：越接近 1 越好
- **MAE**（平均绝对误差）：越小越好

### 2. 训练效率对比
- **训练时间**：PCA 通常最快，KPCA 最慢
- **收敛速度**：观察训练损失曲线的下降速度

### 3. 过拟合风险
- 比较训练损失和验证损失的差距
- 差距过大可能表示过拟合

### 4. 方法特性
- **PCA**: 适合线性关系明显的数据，计算快
- **KPCA**: 适合非线性结构，但需要调参（kernel, gamma）
- **SPCA**: 适合特征冗余多的场景，可解释性好

---

## 💡 常见问题

### Q1: KPCA 效果不好怎么办？
尝试不同的核函数和 gamma 值：
```python
('kpca', {'n_components': 10, 'kernel': 'poly', 'gamma': 0.01})
('kpca', {'n_components': 15, 'kernel': 'rbf', 'gamma': 0.05})
```

### Q2: SPCA 太稀疏导致信息丢失？
降低 alpha 值：
```python
('spca', {'n_components': 10, 'alpha': 0.1})  # 更少的稀疏约束
```

### Q3: 想测试更多维度配置？
修改 n_components 参数进行多组实验：
```python
methods_to_test = [
    ('pca_95', {'n_components': 0.95}),
    ('pca_90', {'n_components': 0.90}),
    ('kpca_10', {'n_components': 10, 'kernel': 'rbf', 'gamma': 0.1}),
    ('kpca_15', {'n_components': 15, 'kernel': 'rbf', 'gamma': 0.1}),
]
```

### Q4: 只想测试某一种方法？
注释掉其他方法：
```python
methods_to_test = [
    ('kpca', {'n_components': 10, 'kernel': 'rbf', 'gamma': 0.1}),
    # ('pca', {'n_components': 0.95}),
    # ('spca', {'n_components': 10, 'alpha': 1.0}),
]
```

---

## 📝 技术细节

### 为什么先做 Boruta 再做降维？
1. **Boruta** 剔除无关特征，减少噪声
2. **降维** 消除剩余特征间的共线性
3. 两步结合能获得更高质量的特征表示

### KPCA 的注意事项
- 需要指定具体的 `n_components`（不能用比例）
- `transform` 新数据时依赖训练集的核矩阵
- 对参数 `gamma` 敏感，建议网格搜索

### SPCA 的优势
- 每个主成分是原始特征的稀疏组合
- 更容易解释哪些原始特征重要
- 适合高维稀疏数据

---

## 🎯 下一步优化方向

1. **超参数搜索**：对 KPCA 的 gamma 和 SPCA 的 alpha 进行网格搜索
2. **交叉验证**：使用时间序列交叉验证评估稳定性
3. **集成方法**：结合多种降维方法的结果
4. **自动化调参**：集成 NRBO 自动优化降维参数

---

## 📚 参考资料

- PCA: Jolliffe, I. T. (2002). Principal Component Analysis
- KPCA: Schölkopf, B., et al. (1998). Nonlinear component analysis as a kernel eigenvalue problem
- SPCA: Zou, H., et al. (2006). Sparse principal component analysis
