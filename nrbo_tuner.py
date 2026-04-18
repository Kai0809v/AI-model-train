"""
NRBO (Neural Randomized Bayesian Optimization) 超参数自动调优器
用于 TCN-Informer 光伏功率预测模型的超参数搜索
"""

import torch
import numpy as np
from sklearn.metrics import mean_squared_error, r2_score
import optuna
import warnings
warnings.filterwarnings('ignore')

# 导入项目模块
from data_loader import create_dataloaders
from model_architecture import True_TCN_Informer


class NRBOOptimizer:
    """NRBO 超参数优化器"""
    
    def __init__(self, pkl_path, n_trials=50, device='cuda' if torch.cuda.is_available() else 'cpu'):
        self.pkl_path = pkl_path
        self.n_trials = n_trials
        self.device = device
        self.best_params = None
        self.best_score = -np.inf
        
    def objective(self, trial):
        """
        Optuna 目标函数：定义超参数搜索空间并返回验证集 R²
        """
        # ===== 1. 定义超参数搜索空间 =====
        
        # TCN 结构参数
        tcn_num_layers = trial.suggest_int('tcn_num_layers', 2, 4)
        tcn_base_channels = trial.suggest_categorical('tcn_base_channels', [16, 32, 64])
        
        # 动态生成 TCN 通道配置
        tcn_channels = [tcn_base_channels * (2 ** i) for i in range(tcn_num_layers)]
        
        # Informer 结构参数
        d_model = trial.suggest_categorical('d_model', [64, 128, 256])
        n_heads = trial.suggest_categorical('n_heads', [4, 8])
        e_layers = trial.suggest_int('e_layers', 2, 4)
        
        # 训练超参数
        learning_rate = trial.suggest_float('learning_rate', 1e-4, 1e-2, log=True)
        weight_decay = trial.suggest_float('weight_decay', 1e-5, 1e-3, log=True)
        dropout = trial.suggest_float('dropout', 0.05, 0.2)
        
        # Huber Loss delta 参数
        huber_delta = trial.suggest_float('huber_delta', 0.1, 1.0)
        
        # 序列长度配置（保持 B = A/2 的约束）
        seq_len_option = trial.suggest_categorical('seq_len_option', [96, 192, 288])
        seq_len = seq_len_option
        label_len = seq_len // 2
        pred_len = 24  # 固定预测未来6小时
        
        # ===== 2. 构建数据加载器 =====
        try:
            train_loader, val_loader, test_loader, bundle = create_dataloaders(
                self.pkl_path, 
                seq_len=seq_len, 
                label_len=label_len, 
                pred_len=pred_len, 
                batch_size=32
            )
        except Exception as e:
            print(f"数据加载失败: {e}")
            return -np.inf
        
        input_dim = bundle['train'][0].shape[1]
        
        # ===== 3. 构建模型 =====
        try:
            model = True_TCN_Informer(
                tcn_input_dim=input_dim,
                tcn_channels=tcn_channels,
                seq_len=seq_len,
                label_len=label_len,
                pred_len=pred_len,
                d_model=d_model,
                n_heads=n_heads,
                e_layers=e_layers,
                dropout=dropout
            ).to(self.device)
        except Exception as e:
            print(f"模型构建失败: {e}")
            return -np.inf
        
        # ===== 4. 快速训练（简化版，仅用于超参数评估）=====
        criterion = torch.nn.HuberLoss(delta=huber_delta)
        optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
        
        # 简化的训练循环（只训练 5 个 epoch 用于快速评估）
        num_epochs = 5
        for epoch in range(num_epochs):
            model.train()
            for batch in train_loader:
                seq_x, seq_x_mark, dec_x, dec_x_mark, target_y = batch
                seq_x = seq_x.to(self.device)
                seq_x_mark = seq_x_mark.to(self.device)
                dec_x = dec_x.to(self.device)
                dec_x_mark = dec_x_mark.to(self.device)
                target_y = target_y.to(self.device)
                
                optimizer.zero_grad()
                pred = model(seq_x, seq_x_mark, dec_x, dec_x_mark)
                loss = criterion(pred.squeeze(-1), target_y)
                loss.backward()
                optimizer.step()
        
        # ===== 5. 验证集评估 =====
        model.eval()
        preds_list = []
        trues_list = []
        
        with torch.no_grad():
            for batch in val_loader:
                seq_x, seq_x_mark, dec_x, dec_x_mark, target_y = batch
                seq_x = seq_x.to(self.device)
                seq_x_mark = seq_x_mark.to(self.device)
                dec_x = dec_x.to(self.device)
                dec_x_mark = dec_x_mark.to(self.device)
                
                pred = model(seq_x, seq_x_mark, dec_x, dec_x_mark).detach().cpu().numpy()
                true = target_y.detach().cpu().numpy()
                
                preds_list.append(pred.squeeze(-1))
                trues_list.append(true)
        
        preds = np.concatenate(preds_list, axis=0)
        trues = np.concatenate(trues_list, axis=0)
        
        # 反归一化
        scaler_y = bundle['scaler_y']
        preds_inverse = scaler_y.inverse_transform(preds.reshape(-1, 1)).reshape(preds.shape)
        trues_inverse = scaler_y.inverse_transform(trues.reshape(-1, 1)).reshape(trues.shape)
        
        # 物理约束
        night_mask = (trues_inverse < 0.05)
        preds_inverse[night_mask] = 0.0
        preds_inverse = np.maximum(0, preds_inverse)
        preds_inverse = np.minimum(preds_inverse, 130.0)
        
        # 计算 R² 作为优化目标
        r2 = r2_score(trues_inverse.flatten(), preds_inverse.flatten())
        
        # 打印当前试验结果
        print(f"Trial {trial.number} | R²: {r2:.4f} | "
              f"TCN: {tcn_channels} | d_model: {d_model} | "
              f"lr: {learning_rate:.6f} | dropout: {dropout:.2f}")
        
        return r2
    
    def optimize(self):
        """执行 NRBO 优化"""
        print("=" * 60)
        print("🚀 启动 NRBO 超参数优化")
        print(f"   设备: {self.device}")
        print(f"   试验次数: {self.n_trials}")
        print("=" * 60)
        
        # 创建 Optuna 研究
        study = optuna.create_study(
            direction='maximize',
            sampler=optuna.samplers.TPESampler(seed=42),  # TPE 采样器
            pruner=optuna.pruners.MedianPruner()  # 中值剪枝
        )
        
        # 执行优化
        study.optimize(self.objective, n_trials=self.n_trials)
        
        # 输出最优结果
        print("\n" + "=" * 60)
        print("✅ NRBO 优化完成！")
        print("=" * 60)
        print(f"最佳验证集 R²: {study.best_value:.4f}")
        print(f"\n最优超参数配置:")
        for key, value in study.best_params.items():
            print(f"  {key}: {value}")
        
        self.best_params = study.best_params
        self.best_score = study.best_value
        
        # 保存最优参数
        import json
        with open('best_nrbo_params.json', 'w') as f:
            json.dump(study.best_params, f, indent=2)
        print(f"\n💾 最优参数已保存至: best_nrbo_params.json")
        
        # 可视化优化过程
        try:
            optuna.visualization.plot_optimization_history(study).write_image('nrbo_optimization_history.png')
            optuna.visualization.plot_param_importances(study).write_image('nrbo_param_importance.png')
            print("📊 优化历史图已保存: nrbo_optimization_history.png")
            print("📊 参数重要性图已保存: nrbo_param_importance.png")
        except Exception as e:
            print(f"可视化保存失败: {e}")
        
        return study.best_params


def train_with_best_params(pkl_path, best_params, epochs=50, device='cuda' if torch.cuda.is_available() else 'cpu'):
    """
    使用 NRBO 找到的最优参数进行完整训练
    """
    print("\n" + "=" * 60)
    print("🎯 使用最优参数进行完整训练")
    print("=" * 60)
    
    # 解析最优参数
    tcn_num_layers = best_params['tcn_num_layers']
    tcn_base_channels = best_params['tcn_base_channels']
    tcn_channels = [tcn_base_channels * (2 ** i) for i in range(tcn_num_layers)]
    
    seq_len = best_params['seq_len_option']
    label_len = seq_len // 2
    pred_len = 24
    
    # 加载数据
    train_loader, val_loader, test_loader, bundle = create_dataloaders(
        pkl_path, seq_len=seq_len, label_len=label_len, pred_len=pred_len, batch_size=32
    )
    
    input_dim = bundle['train'][0].shape[1]
    scaler_y = bundle['scaler_y']
    
    # 构建模型
    model = True_TCN_Informer(
        tcn_input_dim=input_dim,
        tcn_channels=tcn_channels,
        seq_len=seq_len,
        label_len=label_len,
        pred_len=pred_len,
        d_model=best_params['d_model'],
        n_heads=best_params['n_heads'],
        e_layers=best_params['e_layers'],
        dropout=best_params['dropout']
    ).to(device)
    
    # 训练配置
    criterion = torch.nn.HuberLoss(delta=best_params['huber_delta'])
    optimizer = torch.optim.AdamW(
        model.parameters(), 
        lr=best_params['learning_rate'], 
        weight_decay=best_params['weight_decay']
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=10, T_mult=2, eta_min=1e-6
    )
    
    # 早停机制
    from PV_part2 import EarlyStopping
    early_stopping = EarlyStopping(patience=10, verbose=True)
    
    # 完整训练循环
    for epoch in range(epochs):
        model.train()
        epoch_train_loss = []
        
        for seq_x, seq_x_mark, dec_x, dec_x_mark, target_y in train_loader:
            seq_x = seq_x.to(device)
            seq_x_mark = seq_x_mark.to(device)
            dec_x = dec_x.to(device)
            dec_x_mark = dec_x_mark.to(device)
            target_y = target_y.to(device)
            
            optimizer.zero_grad()
            pred = model(seq_x, seq_x_mark, dec_x, dec_x_mark)
            loss = criterion(pred.squeeze(-1), target_y)
            epoch_train_loss.append(loss.item())
            loss.backward()
            optimizer.step()
        
        # 验证
        model.eval()
        epoch_val_loss = []
        with torch.no_grad():
            for seq_x, seq_x_mark, dec_x, dec_x_mark, target_y in val_loader:
                seq_x = seq_x.to(device)
                seq_x_mark = seq_x_mark.to(device)
                dec_x = dec_x.to(device)
                dec_x_mark = dec_x_mark.to(device)
                target_y = target_y.to(device)
                
                pred = model(seq_x, seq_x_mark, dec_x, dec_x_mark)
                loss = criterion(pred.squeeze(-1), target_y)
                epoch_val_loss.append(loss.item())
        
        t_loss = np.average(epoch_train_loss)
        v_loss = np.average(epoch_val_loss)
        print(f"Epoch: {epoch + 1:02d} | Train Loss: {t_loss:.5f} | Val Loss: {v_loss:.5f}")
        
        scheduler.step()
        early_stopping(v_loss, model, path='best_tcn_informer_nrbo.pth')
        
        if early_stopping.early_stop:
            print("🚀 触发早停机制，训练提前结束。")
            break
    
    # 测试集评估
    print("\n--- 开始测试集评估 ---")
    model.load_state_dict(torch.load('best_tcn_informer_nrbo.pth'))
    model.eval()
    
    preds_list = []
    trues_list = []
    
    with torch.no_grad():
        for seq_x, seq_x_mark, dec_x, dec_x_mark, target_y in test_loader:
            seq_x = seq_x.to(device)
            seq_x_mark = seq_x_mark.to(device)
            dec_x = dec_x.to(device)
            dec_x_mark = dec_x_mark.to(device)
            
            pred = model(seq_x, seq_x_mark, dec_x, dec_x_mark).detach().cpu().numpy()
            true = target_y.detach().cpu().numpy()
            
            preds_list.append(pred.squeeze(-1))
            trues_list.append(true)
    
    preds = np.concatenate(preds_list, axis=0)
    trues = np.concatenate(trues_list, axis=0)
    
    # 反归一化
    preds_inverse = scaler_y.inverse_transform(preds.reshape(-1, 1)).reshape(preds.shape)
    trues_inverse = scaler_y.inverse_transform(trues.reshape(-1, 1)).reshape(trues.shape)
    
    # 物理约束
    night_mask = (trues_inverse < 0.05)
    preds_inverse[night_mask] = 0.0
    preds_inverse = np.maximum(0, preds_inverse)
    preds_inverse = np.minimum(preds_inverse, 130.0)
    
    # 计算指标
    mse = mean_squared_error(trues_inverse.flatten(), preds_inverse.flatten())
    rmse = np.sqrt(mse)
    mae = np.mean(np.abs(trues_inverse.flatten() - preds_inverse.flatten()))
    r2 = r2_score(trues_inverse.flatten(), preds_inverse.flatten())
    
    print("\n📊 最终测试集评估指标 (NRBO 优化后):")
    print(f"   MSE:  {mse:.4f}")
    print(f"   RMSE: {rmse:.4f}")
    print(f"   MAE:  {mae:.4f}")
    print(f"   R²:   {r2:.4f}")
    
    return {'MSE': mse, 'RMSE': rmse, 'MAE': mae, 'R2': r2}


if __name__ == "__main__":
    # 第一步：执行 NRBO 优化
    optimizer = NRBOOptimizer(
        pkl_path="processed_data/model_ready_data.pkl",
        n_trials=50,  # 可根据需要调整试验次数
        device='cuda' if torch.cuda.is_available() else 'cpu'
    )
    
    best_params = optimizer.optimize()
    
    # 第二步：使用最优参数进行完整训练
    final_metrics = train_with_best_params(
        pkl_path="processed_data/model_ready_data.pkl",
        best_params=best_params,
        epochs=50
    )
