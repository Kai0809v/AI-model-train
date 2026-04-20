"""
序列长度对比实验脚本
自动测试不同的 seq_len 和 weight_decay 配置组合
TODO：探索序列长度与正则化强度的交互效应
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib.pyplot as plt
import os

# 导入项目模块
from data_loader import create_dataloaders
from model_architecture import True_TCN_Informer


class EarlyStopping:
    def __init__(self, patience=10, verbose=False):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.inf

    def __call__(self, val_loss, model, path='temp_checkpoint.pth'):
        score = -val_loss
        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model, path)
        elif score < self.best_score:
            self.counter += 1
            if self.verbose:
                print(f'  EarlyStopping counter: {self.counter}/{self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model, path)
            self.counter = 0

    def save_checkpoint(self, val_loss, model, path):
        if self.verbose:
            print(f'  Val loss decreased ({self.val_loss_min:.5f} --> {val_loss:.5f}). Saving...')
        torch.save(model.state_dict(), path)
        self.val_loss_min = val_loss


def train_single_config(pkl_path, seq_len, label_len, pred_len=24, epochs=50, 
                       learning_rate=0.001, weight_decay=1e-4,
                       device='cuda' if torch.cuda.is_available() else 'cpu'):
    """
    训练单个配置并返回测试结果
    """
    print(f"\n{'='*70}")
    print(f"🧪 测试配置: seq_len={seq_len}, label_len={label_len}, pred_len={pred_len}")
    print(f"{'='*70}")
    
    # 1. 加载数据
    try:
        train_loader, val_loader, test_loader, bundle = create_dataloaders(
            pkl_path, seq_len=seq_len, label_len=label_len, pred_len=pred_len, batch_size=32
        )
    except Exception as e:
        print(f"❌ 数据加载失败: {e}")
        return None
    
    input_dim = bundle['train'][0].shape[1]
    scaler_y = bundle['scaler_y']
    
    # 2. 构建模型（使用稳定的基线配置）
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
    
    # 3. 训练配置
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)
    early_stopping = EarlyStopping(patience=10, verbose=True)
    
    # 4. 训练循环
    print(f"\n开始训练 (最多{epochs}轮)...")
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
        
        if (epoch + 1) % 5 == 0 or epoch == 0:
            print(f"  Epoch {epoch+1:2d}/{epochs} | Train: {t_loss:.5f} | Val: {v_loss:.5f}")
        
        scheduler.step()
        early_stopping(v_loss, model, path='temp_seq_exp.pth')
        
        if early_stopping.early_stop:
            print(f"  🚀 早停于第 {epoch+1} 轮")
            break
    
    # 5. 测试集评估
    print("\n评估测试集...")
    model.load_state_dict(torch.load('temp_seq_exp.pth'))
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
    mae = mean_absolute_error(trues_inverse.flatten(), preds_inverse.flatten())
    r2 = r2_score(trues_inverse.flatten(), preds_inverse.flatten())
    
    metrics = {'MSE': mse, 'RMSE': rmse, 'MAE': mae, 'R2': r2, 
               'weight_decay': weight_decay, 'stopped_epoch': epoch+1}
    
    print(f"\n📊 测试结果:")
    print(f"   MSE:  {mse:.4f}")
    print(f"   RMSE: {rmse:.4f}")
    print(f"   MAE:  {mae:.4f}")
    print(f"   R²:   {r2:.4f}")
    print(f"   weight_decay: {weight_decay}")
    
    # 清理临时文件
    if os.path.exists('temp_seq_exp.pth'):
        os.remove('temp_seq_exp.pth')
    
    return metrics


def run_seq_length_experiment(pkl_path="processed_data/model_ready_data.pkl",
                             test_weight_decay=True):
    """
    运行完整的序列长度对比实验
    
    :param test_weight_decay: 是否同时测试不同的 weight_decay
    """
    print("=" * 70)
    print("🔬 序列长度与正则化强度对比实验")
    print("=" * 70)
    
    if test_weight_decay:
        print("\n📋 实验设计：测试 seq_len × weight_decay 的组合")
        print("  序列长度: 96, 144, 192, 288")
        print("  weight_decay: 1e-4, 1e-3")
        print("  总配置数: 4 × 2 = 8 个")
    else:
        print("\n将测试以下配置:")
        print("  1. seq_len=96  (1天历史)")
        print("  2. seq_len=144 (1.5天历史)")
        print("  3. seq_len=192 (2天历史)")
        print("  4. seq_len=288 (3天历史)")
    
    print("\n预计总耗时: 约60-120分钟（取决于GPU性能）\n")
    
    input("按回车键开始实验...")
    
    # 定义测试配置
    seq_configs = [
        {"seq_len": 96, "label_len": 48, "name": "1天历史(96)"},
        {"seq_len": 144, "label_len": 72, "name": "1.5天历史(144)"},
        {"seq_len": 192, "label_len": 96, "name": "2天历史(192)"},
        {"seq_len": 288, "label_len": 144, "name": "3天历史(288)"},
    ]
    
    weight_decays = [1e-4, 1e-3] if test_weight_decay else [1e-4]
    
    results = []
    total_tests = len(seq_configs) * len(weight_decays)
    test_count = 0
    
    # 逐个测试
    for wd in weight_decays:
        for config in seq_configs:
            test_count += 1
            print(f"\n{'#'*70}")
            print(f"# 实验 {test_count}/{total_tests}: {config['name']} + wd={wd}")
            print(f"{'#'*70}")
            
            metrics = train_single_config(
                pkl_path=pkl_path,
                seq_len=config['seq_len'],
                label_len=config['label_len'],
                pred_len=24,
                epochs=50,
                learning_rate=0.001,
                weight_decay=wd
            )
            
            if metrics:
                results.append({
                    'config': config,
                    'weight_decay': wd,
                    'metrics': metrics
                })
            
            print(f"\n✅ 实验 {test_count} 完成！")
            
            # 如果不是最后一个，询问是否继续
            if test_count < total_tests:
                print("\n提示: 可以休息片刻，或继续下一个实验")
                input("按回车键继续下一个配置...")
    
    # 汇总结果
    print("\n" + "=" * 70)
    print("📊 实验结果汇总")
    print("=" * 70)
    
    if test_weight_decay:
        print(f"\n{'配置':<25} {'wd':<10} {'R²':<10} {'RMSE':<10} {'MAE':<10} {'早停轮数':<10}")
        print("-" * 85)
    else:
        print(f"\n{'配置':<20} {'R²':<10} {'RMSE':<10} {'MAE':<10} {'MSE':<10}")
        print("-" * 70)
    
    best_r2 = -1
    best_config = None
    
    for result in results:
        config_name = result['config']['name']
        m = result['metrics']
        wd = result['weight_decay']
        
        if test_weight_decay:
            stopped = m.get('stopped_epoch', 'N/A')
            print(f"{config_name:<25} {wd:<10.0e} {m['R2']:<10.4f} {m['RMSE']:<10.4f} {m['MAE']:<10.4f} {stopped:<10}")
        else:
            print(f"{config_name:<20} {m['R2']:<10.4f} {m['RMSE']:<10.4f} {m['MAE']:<10.4f} {m['MSE']:<10.4f}")
        
        if m['R2'] > best_r2:
            best_r2 = m['R2']
            best_config = result
    
    print("-" * 85 if test_weight_decay else "-" * 70)
    print(f"\n🏆 最佳配置:")
    print(f"   序列长度: {best_config['config']['name']}")
    print(f"   weight_decay: {best_config['weight_decay']:.0e}")
    print(f"   R² = {best_config['metrics']['R2']:.4f}")
    print(f"   RMSE = {best_config['metrics']['RMSE']:.4f}")
    print(f"   MAE = {best_config['metrics']['MAE']:.4f}")
    
    # 可视化对比
    plot_results(results)
    
    # 保存结果
    save_results(results, best_config)
    
    return results, best_config


def plot_results(results):
    """
    可视化实验结果
    """
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    configs = [r['config']['name'] for r in results]
    r2_scores = [r['metrics']['R2'] for r in results]
    rmses = [r['metrics']['RMSE'] for r in results]
    maes = [r['metrics']['MAE'] for r in results]
    mses = [r['metrics']['MSE'] for r in results]
    
    # R² 对比
    axes[0, 0].bar(configs, r2_scores, color='#2E86AB', alpha=0.7, edgecolor='black')
    axes[0, 0].set_ylabel('R² Score', fontsize=11)
    axes[0, 0].set_title('R² Comparison', fontsize=12, fontweight='bold')
    axes[0, 0].grid(axis='y', alpha=0.3)
    axes[0, 0].set_ylim(min(r2_scores) - 0.01, 1.0)
    
    # RMSE 对比
    axes[0, 1].bar(configs, rmses, color='#A23B72', alpha=0.7, edgecolor='black')
    axes[0, 1].set_ylabel('RMSE (MW)', fontsize=11)
    axes[0, 1].set_title('RMSE Comparison', fontsize=12, fontweight='bold')
    axes[0, 1].grid(axis='y', alpha=0.3)
    
    # MAE 对比
    axes[1, 0].bar(configs, maes, color='#F18F01', alpha=0.7, edgecolor='black')
    axes[1, 0].set_ylabel('MAE (MW)', fontsize=11)
    axes[1, 0].set_title('MAE Comparison', fontsize=12, fontweight='bold')
    axes[1, 0].grid(axis='y', alpha=0.3)
    
    # MSE 对比
    axes[1, 1].bar(configs, mses, color='#C73E1D', alpha=0.7, edgecolor='black')
    axes[1, 1].set_ylabel('MSE', fontsize=11)
    axes[1, 1].set_title('MSE Comparison', fontsize=12, fontweight='bold')
    axes[1, 1].grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('seq_length_comparison.png', dpi=300, bbox_inches='tight')
    print("\n🖼️  对比图已保存: seq_length_comparison.png")
    plt.show()


def save_results(results, best_config, test_weight_decay=True):
    """
    保存实验结果到文本文件
    """
    with open('seq_length_results.txt', 'w', encoding='utf-8') as f:
        f.write("=" * 70 + "\n")
        f.write("序列长度与正则化强度对比实验结果\n")
        f.write("=" * 70 + "\n\n")
        
        for result in results:
            config = result['config']
            m = result['metrics']
            wd = result['weight_decay']
            
            f.write(f"配置: {config['name']}\n")
            f.write(f"  seq_len: {config['seq_len']}\n")
            f.write(f"  label_len: {config['label_len']}\n")
            f.write(f"  weight_decay: {wd:.0e}\n")
            f.write(f"  MSE:  {m['MSE']:.4f}\n")
            f.write(f"  RMSE: {m['RMSE']:.4f}\n")
            f.write(f"  MAE:  {m['MAE']:.4f}\n")
            f.write(f"  R²:   {m['R2']:.4f}\n")
            if 'stopped_epoch' in m:
                f.write(f"  早停轮数: {m['stopped_epoch']}\n")
            f.write("\n")
        
        f.write("-" * 70 + "\n")
        f.write(f"🏆 最佳配置:\n")
        f.write(f"   序列长度: {best_config['config']['name']}\n")
        f.write(f"   seq_len: {best_config['config']['seq_len']}\n")
        f.write(f"   weight_decay: {best_config['weight_decay']:.0e}\n")
        f.write(f"   R²: {best_config['metrics']['R2']:.4f}\n")
        f.write(f"   RMSE: {best_config['metrics']['RMSE']:.4f}\n")
        f.write(f"   MAE: {best_config['metrics']['MAE']:.4f}\n")
    
    print("📝 详细结果已保存: seq_length_results.txt")


if __name__ == "__main__":
    # 默认启用 weight_decay 对比实验
    results, best = run_seq_length_experiment(test_weight_decay=True)
    
    print("\n" + "=" * 70)
    print("✅ 实验全部完成！")
    print("=" * 70)
    print(f"\n💡 建议下一步:")
    print(f"  修改 PV_part2.py 第70行为:")
    print(f"  seq_len, label_len, pred_len = {best['config']['seq_len']}, {best['config']['label_len']}, 24")
    print(f"\n  修改 optimizer 的 weight_decay 为: {best['weight_decay']:.0e}")
    print(f"\n📊 最终性能指标:")
    print(f"  R² = {best['metrics']['R2']:.4f}")
    print(f"  RMSE = {best['metrics']['RMSE']:.4f}")
    print(f"  MAE = {best['metrics']['MAE']:.4f}")
