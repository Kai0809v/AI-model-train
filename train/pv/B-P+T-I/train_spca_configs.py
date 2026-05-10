"""
SPCA 多配置对比训练脚本

从 spca_hyperparameter_search.py 生成的多个数据文件中，
选择最优的几个配置进行训练和对比。
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import os
import time
import glob
import re

# 导入前置模块
from data_loader import create_dataloaders
from model_architecture import True_TCN_Informer


def calculate_metrics(y_true, y_pred):
    """计算评估指标"""
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    return {'MSE': mse, 'RMSE': rmse, 'MAE': mae, 'R2': r2}


class EarlyStopping:
    def __init__(self, patience=7, verbose=False, delta=0):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.inf
        self.delta = delta

    def __call__(self, val_loss, model, path='checkpoint.pth'):
        score = -val_loss
        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model, path)
        elif score < self.best_score + self.delta:
            self.counter += 1
            if self.verbose:
                print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model, path)
            self.counter = 0

    def save_checkpoint(self, val_loss, model, path):
        if self.verbose:
            print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).')
        torch.save(model.state_dict(), path)
        self.val_loss_min = val_loss


def train_single_config(pkl_path, config_name, epochs=50, learning_rate=0.001, 
                        device='cuda' if torch.cuda.is_available() else 'cpu'):
    """训练单个 SPCA 配置"""
    
    print(f"\n{'='*70}")
    print(f"训练配置: {config_name}")
    print(f"数据文件: {pkl_path}")
    print(f"{'='*70}\n")

    # 1. 加载数据
    seq_len, label_len, pred_len = 192, 96, 24
    train_loader, val_loader, test_loader, bundle = create_dataloaders(
        pkl_path, seq_len=seq_len, label_len=label_len, pred_len=pred_len, batch_size=32
    )

    input_dim = bundle['train'][0].shape[1]
    scaler_y = bundle['scaler_y']
    
    print(f"输入特征维度: {input_dim}")

    # 2. 初始化模型
    model = True_TCN_Informer(
        tcn_input_dim=input_dim,
        tcn_channels=[16, 32],
        seq_len=seq_len,
        label_len=label_len,
        pred_len=pred_len,
        d_model=64,
        n_heads=4,
        e_layers=2,
        dropout=0.15
    ).to(device)

    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-3)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)
    
    # 生成模型保存路径
    model_save_path = f'./best_tcn_informer_{config_name}.pth'
    early_stopping = EarlyStopping(patience=10, verbose=True)

    train_losses, val_losses = [], []

    # 3. 训练循环
    start_time = time.time()
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
        
        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(f"Epoch: {epoch + 1:02d} | Train Loss: {t_loss:.5f} | Val Loss: {v_loss:.5f}")

        scheduler.step()
        early_stopping(v_loss, model, path=model_save_path)
        
        if early_stopping.early_stop:
            print("🚀 触发早停机制")
            break

    training_time = time.time() - start_time
    print(f"训练耗时: {training_time:.2f} 秒")

    # 4. 测试集评估
    print("\n--- 测试集评估 ---")
    model.load_state_dict(torch.load(model_save_path))
    model.eval()

    preds_list, trues_list = [], []
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

    metrics = calculate_metrics(trues_inverse.flatten(), preds_inverse.flatten())
    
    print(f"\n📊 [{config_name}] 测试结果:")
    print(f"   MSE:  {metrics['MSE']:.4f}")
    print(f"   RMSE: {metrics['RMSE']:.4f}")
    print(f"   MAE:  {metrics['MAE']:.4f}")
    print(f"   R²:   {metrics['R2']:.4f}")
    print(f"   训练时间: {training_time:.2f} 秒")

    return {
        'config_name': config_name,
        'metrics': metrics,
        'train_losses': train_losses,
        'val_losses': val_losses,
        'training_time': training_time,
        'model_path': model_save_path
    }


def find_spca_data_files(output_dir="processed_data"):
    """查找所有 SPCA 生成的数据文件"""
    pattern = os.path.join(output_dir, "model_ready_data_spca_n*_a*.pkl")
    files = glob.glob(pattern)
    
    configs = []
    for file in files:
        # 从文件名提取参数
        filename = os.path.basename(file)
        match = re.search(r'spca_n(\d+)_a([0-9.]+)\.pkl', filename)
        if match:
            n_comp = int(match.group(1))
            alpha = float(match.group(2))
            config_name = f"SPCA_n{n_comp}_a{alpha}"
            configs.append({
                'config_name': config_name,
                'file_path': file,
                'n_components': n_comp,
                'alpha': alpha
            })
    
    # 按 n_components 和 alpha 排序
    configs.sort(key=lambda x: (x['n_components'], x['alpha']))
    
    return configs


def train_selected_configs(selected_configs, epochs=50, learning_rate=0.001):
    """训练选定的配置"""
    
    print("\n" + "="*80)
    print("开始训练选定的 SPCA 配置")
    print("="*80)
    
    results = {}
    
    for idx, config in enumerate(selected_configs, 1):
        print(f"\n[{idx}/{len(selected_configs)}] 准备训练: {config['config_name']}")
        
        try:
            result = train_single_config(
                pkl_path=config['file_path'],
                config_name=config['config_name'],
                epochs=epochs,
                learning_rate=learning_rate
            )
            
            results[config['config_name']] = result
            
            # 保存中间结果
            save_intermediate_results(results)
            
        except Exception as e:
            print(f"\n✗ {config['config_name']} 训练失败: {str(e)}")
            import traceback
            traceback.print_exc()
    
    return results


def save_intermediate_results(results):
    """保存中间结果到 CSV"""
    if not results:
        return
    
    csv_path = 'spca_training_results.csv'
    
    with open(csv_path, 'w', encoding='utf-8') as f:
        f.write("Config,MSE,RMSE,MAE,R2,Training_Time(s)\n")
        for config_name, result in results.items():
            metrics = result['metrics']
            f.write(f"{config_name},{metrics['MSE']:.4f},{metrics['RMSE']:.4f},"
                   f"{metrics['MAE']:.4f},{metrics['R2']:.4f},{result['training_time']:.2f}\n")
    
    print(f"\n📄 中间结果已保存至: {csv_path}")


def plot_comparison(results):
    """绘制对比图"""
    
    if not results:
        return
    
    # 1. 指标对比柱状图
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    configs = list(results.keys())
    rmse_values = [results[c]['metrics']['RMSE'] for c in configs]
    r2_values = [results[c]['metrics']['R2'] for c in configs]
    mae_values = [results[c]['metrics']['MAE'] for c in configs]
    times = [results[c]['training_time'] for c in configs]
    
    # RMSE 对比
    axes[0, 0].bar(range(len(configs)), rmse_values, color='steelblue', alpha=0.7)
    axes[0, 0].set_xticks(range(len(configs)))
    axes[0, 0].set_xticklabels([c.replace('SPCA_', '') for c in configs], rotation=45, ha='right')
    axes[0, 0].set_ylabel('RMSE', fontsize=11)
    axes[0, 0].set_title('RMSE Comparison', fontsize=12)
    axes[0, 0].grid(True, alpha=0.3)
    
    # R² 对比
    axes[0, 1].bar(range(len(configs)), r2_values, color='forestgreen', alpha=0.7)
    axes[0, 1].set_xticks(range(len(configs)))
    axes[0, 1].set_xticklabels([c.replace('SPCA_', '') for c in configs], rotation=45, ha='right')
    axes[0, 1].set_ylabel('R² Score', fontsize=11)
    axes[0, 1].set_title('R² Score Comparison', fontsize=12)
    axes[0, 1].grid(True, alpha=0.3)
    
    # MAE 对比
    axes[1, 0].bar(range(len(configs)), mae_values, color='coral', alpha=0.7)
    axes[1, 0].set_xticks(range(len(configs)))
    axes[1, 0].set_xticklabels([c.replace('SPCA_', '') for c in configs], rotation=45, ha='right')
    axes[1, 0].set_ylabel('MAE', fontsize=11)
    axes[1, 0].set_title('MAE Comparison', fontsize=12)
    axes[1, 0].grid(True, alpha=0.3)
    
    # 训练时间对比
    axes[1, 1].bar(range(len(configs)), times, color='mediumpurple', alpha=0.7)
    axes[1, 1].set_xticks(range(len(configs)))
    axes[1, 1].set_xticklabels([c.replace('SPCA_', '') for c in configs], rotation=45, ha='right')
    axes[1, 1].set_ylabel('Training Time (s)', fontsize=11)
    axes[1, 1].set_title('Training Time Comparison', fontsize=12)
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('spca_config_comparison.png', dpi=150, bbox_inches='tight')
    print(f"\n📊 配置对比图已保存为: spca_config_comparison.png")
    plt.close()
    
    # 2. 训练损失曲线对比
    plt.figure(figsize=(12, 6))
    for config_name, result in results.items():
        plt.plot(result['val_losses'], label=config_name.replace('SPCA_', ''), linewidth=2)
    
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Validation Loss', fontsize=12)
    plt.title('SPCA Configurations - Validation Loss Comparison', fontsize=14)
    plt.legend(fontsize=10, loc='upper right')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('spca_training_curves.png', dpi=150, bbox_inches='tight')
    print(f"📊 训练曲线对比图已保存为: spca_training_curves.png")
    plt.close()


def main():
    """主函数"""
    
    print("="*80)
    print("SPCA 多配置对比训练")
    print("="*80)
    
    # 1. 查找所有 SPCA 数据文件
    print("\n1. 查找 SPCA 数据文件...")
    configs = find_spca_data_files("processed_data")
    
    if not configs:
        print("✗ 未找到任何 SPCA 数据文件！")
        print("请先运行: python spca_hyperparameter_search.py")
        return
    
    print(f"✓ 找到 {len(configs)} 个 SPCA 配置文件\n")
    for config in configs:
        print(f"   - {config['config_name']}: n={config['n_components']}, alpha={config['alpha']}")
    
    # 2. 让用户选择要训练的配置
    print("\n" + "="*80)
    print("请选择要训练的配置（输入编号，用逗号分隔，或输入 'all' 训练全部）:")
    print("="*80)
    
    for idx, config in enumerate(configs, 1):
        print(f"  {idx}. {config['config_name']}")
    
    choice = input("\n请输入选择: ").strip()
    
    if choice.lower() == 'all':
        selected_configs = configs
    else:
        try:
            indices = [int(x.strip()) - 1 for x in choice.split(',')]
            selected_configs = [configs[i] for i in indices if 0 <= i < len(configs)]
        except:
            print("无效输入，默认训练前 6 个配置")
            selected_configs = configs[:6]
    
    print(f"\n将训练 {len(selected_configs)} 个配置:")
    for config in selected_configs:
        print(f"   - {config['config_name']}")
    
    # 3. 开始训练
    results = train_selected_configs(selected_configs, epochs=50, learning_rate=0.001)
    
    # 4. 汇总结果
    if results:
        print("\n" + "="*80)
        print("最终结果汇总")
        print("="*80)
        
        print(f"\n{'配置':<25} {'MSE':<10} {'RMSE':<10} {'MAE':<10} {'R²':<10} {'时间(s)':<10}")
        print("-" * 80)
        
        for config_name, result in results.items():
            m = result['metrics']
            print(f"{config_name:<25} {m['MSE']:<10.4f} {m['RMSE']:<10.4f} "
                  f"{m['MAE']:<10.4f} {m['R2']:<10.4f} {result['training_time']:<10.2f}")
        
        # 找出最佳配置
        best_config = min(results.keys(), key=lambda x: results[x]['metrics']['RMSE'])
        print(f"\n🏆 最佳配置: {best_config}")
        print(f"   RMSE: {results[best_config]['metrics']['RMSE']:.4f}")
        print(f"   R²:   {results[best_config]['metrics']['R2']:.4f}")
        
        # 绘制对比图
        plot_comparison(results)
        
        print("\n✅ 所有训练完成！")


if __name__ == "__main__":
    main()
