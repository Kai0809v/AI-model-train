#  目前问题

# 下一步
NRBO 外挂脚本编写



---
```
graph TD
    subgraph Data Processing [阶段一：数据预处理与特征工程]
        A[原始光伏/气象数据] --> B(Boruta 特征选择)
        B -->|剔除无关变量| C(PCA 降维)
        A -->|提取时间特征| D[Month, Day, Hour, Minute]
        C -->|气象主成分| E[处理后数据流]
        D -->|归一化时间戳| E
    end

    subgraph Optimization [阶段二：NRBO 自动寻优]
        F((NRBO 优化器)) -->|下发最优超参数| G
        G -->|返回验证集 Loss| F
    end

    subgraph Neural Network [阶段三：核心预测网络]
        E --> G[TCN-Informer 混合模型]
        G -->|气象通道| H(TCN 局部特征提取)
        G -->|时间通道| I(Time Feature Embedding)
        H --> J[Informer Encoder]
        I --> J
        I --> K[Informer Decoder]
        J --> K
        K --> L(最终输出: 功率预测)
    end
    
    L --> M[残差与指标分析]
```
---
```
   MSE: 76.8587
   RMSE: 8.7669
   MAE: 4.0039
   R2: 0.8920
```


~~完成的时候在TODO里面埋几个坑~~