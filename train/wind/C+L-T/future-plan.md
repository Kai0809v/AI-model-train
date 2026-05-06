## 📋 多步长预测系统优化路线图

### 阶段一：基础架构重构（1-2周）⭐ 优先级最高

#### 1.1 模型架构升级
**目标**：支持灵活的多步长输出，避免误差累积

```
当前架构：单步输出 → 滚动预测48步 ❌
目标架构：多步输出头 → 直接预测N步 ✅

输入: (batch, 48, features)
     ↓
Transformer Encoder (共享)
     ↓
    ┌─────────────┬──────────────┬──────────────┐
    │  Head-1步   │  Head-4步    │  Head-16步   │
    │  (1 output) │  (4 outputs) │  (16 outputs)│
    └─────────────┴──────────────┴──────────────┘
```


**优势**：
- ✅ 不同步长独立训练，互不干扰
- ✅ 避免滚动预测的误差累积
- ✅ API可灵活选择步长（1/4/8/16）

#### 1.2 数据准备优化
修改 [part1.py](file:///D:/AAwindnb/CEEMDAN/part1.py) 生成多步标签：

```python
# 新增：为不同步长生成分组标签
def build_multi_step_windows(X_all, y_clean_all, y_raw_test, window_size, 
                              split_index, horizons=[1, 4, 8, 16]):
    """
    为每个horizon生成对应的训练/测试数据
    """
    datasets = {}
    for h in horizons:
        train_x, train_y = [], []
        test_x, test_y_clean, test_y_raw = [], [], []
        
        # 训练集：预测未来h步
        for i in range(window_size, split_index - h + 1):
            X_window = X_all[i - window_size:i]
            y_future = y_clean_all[i:i+h]  # 未来h步的干净标签
            train_x.append(X_window)
            train_y.append(y_future)
        
        # 测试集同理...
        datasets[h] = {
            'train_x': torch.tensor(np.array(train_x)).float(),
            'train_y': torch.tensor(np.array(train_y)).float(),
            # ...
        }
    
    return datasets
```


---

### 阶段二：模型训练策略（1周）

#### 2.1 多任务学习框架
```python
class MultiHorizonTransformer(nn.Module):
    def __init__(self, input_dim, horizons=[1, 4, 8, 16]):
        super().__init__()
        
        # 共享编码器
        self.embedding = nn.Linear(input_dim, 256)
        self.pos_encoder = PositionalEncoding(256)
        self.transformer = TransformerEncoder(...)
        
        # 多步长预测头
        self.heads = nn.ModuleDict({
            f'h_{h}': nn.Sequential(
                nn.Linear(256, 128),
                nn.ReLU(),
                nn.Linear(128, h)  # 直接输出h步
            ) for h in horizons
        })
    
    def forward(self, x, horizon=1):
        x = self.embedding(x)
        x = self.pos_encoder(x)
        x = self.transformer(x)
        
        # 注意力池化得到上下文向量
        context = self.attention_pooling(x)  # (batch, 256)
        
        # 根据请求的步长选择对应预测头
        head_key = f'h_{horizon}'
        output = self.heads[head_key](context)  # (batch, horizon)
        
        return output
```


#### 2.2 损失函数设计
```python
def multi_horizon_loss(predictions, targets, horizon):
    """
    加权MSE：近期权重更高
    """
    weights = torch.exp(-torch.arange(horizon) * 0.1)  # 指数衰减
    weights = weights.to(predictions.device)
    
    mse_per_step = (predictions - targets) ** 2
    weighted_mse = (mse_per_step * weights).mean()
    
    return weighted_mse
```


#### 2.3 训练流程
```python
# 训练时随机采样不同步长
for epoch in range(epochs):
    for batch_idx, (x, y_dict) in enumerate(train_loader):
        # 随机选择一个步长进行训练
        horizon = random.choice([1, 4, 8, 16])
        y_target = y_dict[horizon]
        
        pred = model(x, horizon=horizon)
        loss = multi_horizon_loss(pred, y_target, horizon)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
```


---



### 阶段四：高级优化（可选，2-3周）

#### 4.1 不确定性量化
```python
# 使用MC Dropout或Ensemble提供置信区间
class BayesianTransformer(nn.Module):
    def forward_with_uncertainty(self, x, horizon, n_samples=50):
        predictions = []
        for _ in range(n_samples):
            self.train()  # 启用Dropout
            pred = self.forward(x, horizon)
            predictions.append(pred)
        
        predictions = torch.stack(predictions)
        mean = predictions.mean(dim=0)
        std = predictions.std(dim=0)
        
        return mean, std  # 返回预测值和不确定性
```




#### 4.3 模型集成（已完成）
```python
# 集成3个不同随机种子的模型
class EnsemblePredictor:
    def __init__(self, model_paths):
        self.models = [load_model(p) for p in model_paths]
    
    def predict(self, x, horizon):
        predictions = [m(x, horizon) for m in self.models]
        return np.mean(predictions, axis=0)  # 平均预测
```


---

## 🎯 具体实施计划

### 第1周：核心重构
- [ ] 修改 `part1.py` 生成多步标签数据
- [ ] 实现 `MultiHorizonTransformer` 模型
- [ ] 编写多任务训练脚本

### 第2周：训练与验证
- [ ] 训练多步长模型
- [ ] 对比各步长指标（1/4/8/16步）
- [ ] 消融实验验证改进效果


---

## 📊 预期效果对比

| 指标 | 当前方案 | 优化后目标 |
|------|---------|-----------|
| 1步 RMSE | 11.08 MW | ≤10.5 MW |
| 4步 RMSE | ~15 MW* | ≤12 MW |
| 8步 RMSE | ~20 MW* | ≤14 MW |
| 16步 RMSE | ~25 MW* | ≤16 MW |
| API响应时间 | - | <100ms |
| 支持步长 | 仅48步滚动 | 1/4/8/16灵活选择 |

*\*估算值，当前未测量*

---

