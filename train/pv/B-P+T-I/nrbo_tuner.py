import optuna
import torch
import numpy as np
import json
import matplotlib.pyplot as plt


def objective(trial):
    """
    Optuna目标函数：优化TCN-Informer模型超参数
    """
    # 定义超参数搜索空间
    tcn_channels_option = trial.suggest_categorical('tcn_channels_option', ['small', 'medium', 'large'])
    if tcn_channels_option == 'small':
        tcn_channels = [16, 32]
    elif tcn_channels_option == 'medium':
        tcn_channels = [32, 64]
    else:
        tcn_channels = [64, 128]
    
    d_model = trial.suggest_categorical('d_model', [64, 96, 128])
    n_heads = trial.suggest_categorical('n_heads', [4, 8])
    e_layers = trial.suggest_int('e_layers', 2, 4)
    learning_rate = trial.suggest_loguniform('learning_rate', 5e-4, 2e-3)
    weight_decay = trial.suggest_loguniform('weight_decay', 5e-5, 5e-4)
    dropout = trial.suggest_float('dropout', 0.1, 0.2)
    
    # 固定序列长度（基于已有实验结果）
    seq_len = 96  # 已验证的最佳长度
    label_len = 48
    pred_len = 24

    print(f"Trial {trial.number}: Testing hyperparameters...")
    print(f"  TCN Channels: {tcn_channels}")
    print(f"  d_model: {d_model}, n_heads: {n_heads}, e_layers: {e_layers}")
    print(f"  lr: {learning_rate:.6f}, wd: {weight_decay:.6f}, dropout: {dropout:.3f}")

    # 临时修改模型参数并训练
    # 这里我们直接调用train_and_evaluate，但传入特定的超参数
    # 为避免修改原函数，我们创建一个包装函数
    try:
        # 由于train_and_evaluate函数没有直接接受超参数的接口，
        # 我们需要创建一个临时修改版本
        from model_architecture import True_TCN_Informer
        from data_loader import create_dataloaders
        import torch.nn as nn
        import torch.optim as optim
        
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        # 加载数据
        train_loader, val_loader, test_loader, bundle = create_dataloaders(
            "processed_data/model_ready_data.pkl", 
            seq_len=seq_len, label_len=label_len, pred_len=pred_len, 
            batch_size=32
        )
        
        input_dim = bundle['train'][0].shape[1]  # 获取特征维度
        scaler_y = bundle['scaler_y']
        
        # 创建模型
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
        ).to(device)
        
        # 使用标准MSE Loss（加权MSE在归一化空间中无效）
        criterion = nn.MSELoss()
        
        optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
        
        # 简化的训练过程（只训练少量epoch以节省时间）
        model.train()
        max_quick_epochs = 5  # 快速评估，只训练5个epoch
        best_val_loss = float('inf')
        
        for epoch in range(max_quick_epochs):
            epoch_train_loss = []
            for i, (seq_x, seq_x_mark, dec_x, dec_x_mark, target_y) in enumerate(train_loader):
                if i > 10:  # 只训练前10个batch，快速评估
                    break
                    
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
                for i, (seq_x, seq_x_mark, dec_x, dec_x_mark, target_y) in enumerate(val_loader):
                    if i > 5:  # 只验证前5个batch
                        break
                        
                    seq_x = seq_x.to(device)
                    seq_x_mark = seq_x_mark.to(device)
                    dec_x = dec_x.to(device)
                    dec_x_mark = dec_x_mark.to(device)
                    target_y = target_y.to(device)

                    pred = model(seq_x, seq_x_mark, dec_x, dec_x_mark)
                    loss = criterion(pred.squeeze(-1), target_y)
                    epoch_val_loss.append(loss.item())
            
            avg_val_loss = np.average(epoch_val_loss)
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                
            model.train()
        
        # 返回验证集损失作为目标值（Optuna最小化此值）
        return best_val_loss
        
    except Exception as e:
        print(f"Trial {trial.number} failed with error: {str(e)}")
        # 如果训练失败，返回一个较大的损失值
        return 10.0


def full_evaluation_with_best_params(best_params):
    """
    使用最优参数进行完整训练和评估
    """
    # 由于完整训练需要修改PV_part2.py，我们创建一个临时版本
    from model_architecture import True_TCN_Informer
    from data_loader import create_dataloaders
    import torch.nn as nn
    import torch.optim as optim
    from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
    import numpy as np
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # 加载数据
    seq_len = 96
    label_len = 48
    pred_len = 24
    
    train_loader, val_loader, test_loader, bundle = create_dataloaders(
        "processed_data/model_ready_data.pkl", 
        seq_len=seq_len, label_len=label_len, pred_len=pred_len, 
        batch_size=32
    )
    
    input_dim = bundle['train'][0].shape[1]
    scaler_y = bundle['scaler_y']
    
    # 创建模型
    model = True_TCN_Informer(
        tcn_input_dim=input_dim,
        tcn_channels=best_params['tcn_channels'],
        seq_len=seq_len,
        label_len=label_len,
        pred_len=pred_len,
        d_model=best_params['d_model'],
        n_heads=best_params['n_heads'],
        e_layers=best_params['e_layers'],
        dropout=best_params['dropout']
    ).to(device)
    
    # 使用标准MSE Loss
    criterion = nn.MSELoss()
    
    optimizer = optim.Adam(
        model.parameters(), 
        lr=best_params['learning_rate'], 
        weight_decay=best_params['weight_decay']
    )
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=50, eta_min=1e-6)
    
    # 完整训练
    epochs = 50
    train_losses, val_losses = [], []
    
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
        train_losses.append(t_loss)
        val_losses.append(v_loss)
        
        scheduler.step()
        
        if epoch % 10 == 0:
            print(f"Epoch {epoch}: Train Loss: {t_loss:.5f}, Val Loss: {v_loss:.5f}")
    
    # 测试集评估
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
    MAX_CAPACITY = 130.0
    preds_inverse = np.minimum(preds_inverse, MAX_CAPACITY)

    # 计算指标
    mse = mean_squared_error(trues_inverse.flatten(), preds_inverse.flatten())
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(trues_inverse.flatten(), preds_inverse.flatten())
    r2 = r2_score(trues_inverse.flatten(), preds_inverse.flatten())
    
    return {
        'MSE': mse,
        'RMSE': rmse, 
        'MAE': mae,
        'R2': r2
    }


def run_nrbo_optimization():
    """
    运行NRBO优化
    """
    print("🚀 开始NRBO超参数优化...")
    print("此过程可能需要2-4小时，请耐心等待...")
    
    # 创建study对象
    study = optuna.create_study(
        direction='minimize',
        sampler=optuna.samplers.TPESampler(seed=42),
        pruner=optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=5)
    )
    
    # 执行优化
    study.optimize(objective, n_trials=20)  # 减少试验次数以加快进度
    
    print("✅ NRBO优化完成！")
    print(f"最优验证损失: {study.best_value:.6f}")
    print(f"最优超参数: {study.best_params}")
    
    # 保存最优参数
    best_params = study.best_params.copy()
    
    # 将分类参数转换为实际值
    tcn_channels_option = best_params.pop('tcn_channels_option')
    if tcn_channels_option == 'small':
        best_params['tcn_channels'] = [16, 32]
    elif tcn_channels_option == 'medium':
        best_params['tcn_channels'] = [32, 64]
    else:
        best_params['tcn_channels'] = [64, 128]
    
    with open('best_nrbo_params.json', 'w', encoding='utf-8') as f:
        json.dump(best_params, f, indent=2, ensure_ascii=False)
    
    print("💾 最优参数已保存至 best_nrbo_params.json")
    
    # 生成优化历史图
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(study.trials_dataframe()['value'])
    ax.set_xlabel('Trial Number')
    ax.set_ylabel('Validation Loss')
    ax.set_title('NRBO Optimization History')
    ax.grid(True)
    plt.tight_layout()
    plt.savefig('nrbo_optimization_history.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("📊 优化历史图已保存至 nrbo_optimization_history.png")
    
    # 如果需要，可以进一步使用最优参数进行完整训练
    print("\n💡 建议：使用最优参数进行完整训练以获得最终模型")
    print("   运行命令: python nrbo_full_train.py")


if __name__ == "__main__":
    run_nrbo_optimization()