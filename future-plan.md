非常好的实际应用场景！让我为您规划一个面向API部署的优化路线。

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

### 阶段三：API接口设计（3-5天）

#### 3.1 FastAPI接口规范
```python
from fastapi import FastAPI
from pydantic import BaseModel
import torch

app = FastAPI(title="风电功率预测API")

class PredictionRequest(BaseModel):
    historical_data: list  # 过去48步的特征数据
    horizon: int = 1       # 预测步长: 1/4/8/16
    confidence_interval: bool = False  # 是否返回置信区间

class PredictionResponse(BaseModel):
    predictions: list      # 预测值列表
    timestamps: list       # 对应的时间戳
    rmse_estimate: float   # 预估误差
    model_version: str

@app.post("/predict")
async def predict_power(request: PredictionRequest):
    # 1. 数据预处理（标准化、特征工程）
    processed_data = preprocess(request.historical_data)
    
    # 2. 模型推理
    with torch.no_grad():
        predictions = model(
            processed_data, 
            horizon=request.horizon
        )
    
    # 3. 反归一化
    predictions_real = scaler_y.inverse_transform(predictions)
    
    # 4. 后处理（物理约束）
    predictions_real = np.clip(predictions_real, 0, max_capacity)
    
    return PredictionResponse(
        predictions=predictions_real.tolist(),
        timestamps=generate_timestamps(request.horizon),
        rmse_estimate=get_rmse_by_horizon(request.horizon),
        model_version="v2.0"
    )
```


#### 3.2 性能监控与日志
```python
# 记录每次预测的实际误差（用于持续优化）
def log_prediction(actual, predicted, horizon):
    metrics = {
        'timestamp': datetime.now(),
        'horizon': horizon,
        'rmse': np.sqrt(mean_squared_error(actual, predicted)),
        'mae': mean_absolute_error(actual, predicted)
    }
    # 写入数据库或日志文件
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


#### 4.2 在线学习机制
```python
# 定期用新数据微调模型
def online_fine_tune(new_data, learning_rate=1e-5):
    """
    每周用最新数据微调模型，适应数据分布漂移
    """
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    
    for epoch in range(5):  # 少量epoch避免过拟合
        loss = compute_loss(new_data)
        loss.backward()
        optimizer.step()
```


#### 4.3 模型集成
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

### 第3周：API开发
- [ ] 搭建FastAPI服务
- [ ] 实现数据预处理管道
- [ ] 添加监控和日志

### 第4周：部署与测试
- [ ] Docker容器化
- [ ] 压力测试（并发请求）
- [ ] A/B测试对比旧方案

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

## ❓ 下一步行动

我建议按以下顺序执行：

1. **首先确认需求**：您希望优先实现哪个步长（1/4/8/16）？还是全部同时实现？
2. **数据准备**：我先修改 `part1.py` 生成多步标签数据，您看可以吗？
3. **模型改造**：然后实现多步输出架构

**您希望我现在开始修改数据处理脚本，先生成多步标签数据吗？** 还是您有其他优先级考虑？