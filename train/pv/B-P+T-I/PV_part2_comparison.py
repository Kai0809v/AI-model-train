import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import os
import time

# 导入前置模块
from data_loader import create_dataloaders
from model_architecture import True_TCN_Informer


# ==========================================
# 1. 评估指标计算函数
# ==========================================
def calculate_metrics(y_true, y_pred):
    """计算常用的时间序列预测指标"""
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)

    return {'MSE': mse, 'RMSE': rmse, 'MAE': mae, 'R2': r2}


# ==========================================
# 2. 早停机制 (Early Stopping)
# ==========================================
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
            print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        torch.save(model.state_dict(), path)
        self.val_loss_min = val_loss


# ==========================================
# 3. 核心训练与测试循环
# ==========================================
def train_and_evaluate(pkl_path, reduction_method='PCA', epochs=50, learning_rate=0.001, 
                       device='cuda' if torch.cuda.is_available() else 'cpu'):
    print(f"\n{'='*60}")
    print(f"启动训练引擎 - 降维方法: {reduction_method}")
    print(f"使用设备: {device}")
    print(f"{'='*60}\n")

    # 1. 获取数据
    seq_len, label_len, pred_len = 192, 96, 24
    train_loader, val_loader, test_loader, bundle = create_dataloaders(
        pkl_path, seq_len=seq_len, label_len=label_len, pred_len=pred_len, batch_size=32
    )

    # 提取 PCA 后的特征维度 (即 TCN 的输入维度)
    input_dim = bundle['train'][0].shape[1]
    scaler_y = bundle['scaler_y']  # 用于反归一化
    
    print(f"输入特征维度: {input_dim}")

    # 2. 初始化模型 (使用稳定的基线配置)
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
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=epochs, eta_min=1e-6
    )
    
    # 根据降维方法生成不同的模型保存路径
    method_suffix = reduction_method.lower()
    model_save_path = f'./best_tcn_informer_{method_suffix}.pth'
    
    early_stopping = EarlyStopping(patience=10, verbose=True)

    train_losses, val_losses = [], []

    # 3. 训练循环
    start_time = time.time()
    for epoch in range(epochs):
        model.train()
        epoch_train_loss = []

        for i, (seq_x, seq_x_mark, dec_x, dec_x_mark, target_y) in enumerate(train_loader):
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

        # 4. 验证循环
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

        # 记录并打印 Log
        t_loss = np.average(epoch_train_loss)
        v_loss = np.average(epoch_val_loss)
        train_losses.append(t_loss)
        val_losses.append(v_loss)
        
        if (epoch + 1) % 5 == 0 or epoch == 0:  # 每5个epoch打印一次
            print(f"Epoch: {epoch + 1:02d} | Train Loss: {t_loss:.5f} | Val Loss: {v_loss:.5f}")

        scheduler.step()
        current_lr = optimizer.param_groups[0]['lr']

        # 触发早停机制
        early_stopping(v_loss, model, path=model_save_path)
        if early_stopping.early_stop:
            print("🚀 触发早停机制，训练提前结束。")
            break

    training_time = time.time() - start_time
    print(f"\n训练耗时: {training_time:.2f} 秒")

    # ==========================================
    # 5. 模型测试与指标评估
    # ==========================================
    print("\n--- 开始测试集评估 ---")
    model.load_state_dict(torch.load(model_save_path))
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

    # ⭐️ 核心步骤：反归一化 ⭐️
    preds_inverse = scaler_y.inverse_transform(preds.reshape(-1, 1)).reshape(preds.shape)
    trues_inverse = scaler_y.inverse_transform(trues.reshape(-1, 1)).reshape(trues.shape)

    # 💡 物理约束：夜间强制归零 + 功率上限限制
    night_mask = (trues_inverse < 0.05)
    preds_inverse[night_mask] = 0.0
    preds_inverse = np.maximum(0, preds_inverse)
    
    MAX_CAPACITY = 130.0  # MW
    preds_inverse = np.minimum(preds_inverse, MAX_CAPACITY)

    # 计算整体指标
    metrics = calculate_metrics(trues_inverse.flatten(), preds_inverse.flatten())
    print(f"\n📊 [{reduction_method}] 最终测试集评估指标:")
    for k, v in metrics.items():
        print(f"   {k}: {v:.4f}")
    print(f"   训练耗时: {training_time:.2f} 秒")

    # ==========================================
    # 6. 可视化预测结果
    # ==========================================
    sample_power_sums = trues_inverse.sum(axis=1)
    best_sample_idx = np.argmax(sample_power_sums)
    
    print(f"\n正在可视化样本索引: {best_sample_idx} (未来 24 步总功率: {sample_power_sums[best_sample_idx]:.2f} MW)")
    
    plt.figure(figsize=(12, 5))
    plt.plot(trues_inverse[best_sample_idx, :], label=f'Ground Truth', color='blue', marker='o', markersize=4)
    plt.plot(preds_inverse[best_sample_idx, :], label=f'{reduction_method} Prediction', 
             color='red', linestyle='--', marker='x', markersize=4)
    plt.title(f'PV Power Forecasting - {reduction_method} Method\n(Sample {best_sample_idx}, Total Power: {sample_power_sums[best_sample_idx]:.2f} MW)')
    plt.xlabel('Future Time Steps (15 min/step)')
    plt.ylabel('Power (MW)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # 保存图片
    plot_filename = f'prediction_result_{method_suffix}.png'
    plt.savefig(plot_filename, dpi=150, bbox_inches='tight')
    print(f"🖼️ 预测对比曲线已保存为 {plot_filename}")
    plt.close()

    # 返回训练历史
    return {
        'method': reduction_method,
        'metrics': metrics,
        'train_losses': train_losses,
        'val_losses': val_losses,
        'training_time': training_time,
        'model_path': model_save_path,
        'plot_path': plot_filename
    }


def compare_all_methods(data_paths):
    """
    对比所有降维方法的结果
    
    参数:
        data_paths: 字典，键为方法名，值为数据文件路径
                   {'PCA': 'path/to/pca.pkl', 'KPCA': 'path/to/kpca.pkl', ...}
    """
    print("\n" + "="*80)
    print("开始对比实验：PCA vs KPCA vs SPCA")
    print("="*80 + "\n")
    
    results = {}
    
    for method_name, pkl_path in data_paths.items():
        if not os.path.exists(pkl_path):
            print(f"⚠️  跳过 {method_name}: 文件不存在 - {pkl_path}")
            continue
            
        try:
            result = train_and_evaluate(
                pkl_path=pkl_path,
                reduction_method=method_name,
                epochs=50,
                learning_rate=0.001
            )
            results[method_name] = result
            print(f"\n✓ {method_name} 完成\n")
            
        except Exception as e:
            print(f"\n✗ {method_name} 失败: {str(e)}\n")
            import traceback
            traceback.print_exc()
    
    # ==========================================
    # 汇总对比结果
    # ==========================================
    print("\n" + "="*80)
    print("实验结果汇总对比")
    print("="*80)
    
    if results:
        # 打印表格
        print(f"\n{'方法':<10} {'MSE':<12} {'RMSE':<12} {'MAE':<12} {'R²':<12} {'训练时间(s)':<12}")
        print("-" * 80)
        
        for method_name, result in results.items():
            metrics = result['metrics']
            train_time = result['training_time']
            print(f"{method_name:<10} {metrics['MSE']:<12.4f} {metrics['RMSE']:<12.4f} "
                  f"{metrics['MAE']:<12.4f} {metrics['R2']:<12.4f} {train_time:<12.2f}")
        
        # 找出最佳方法
        best_method = min(results.keys(), key=lambda x: results[x]['metrics']['RMSE'])
        print(f"\n🏆 最佳方法: {best_method} (RMSE: {results[best_method]['metrics']['RMSE']:.4f})")
        
        # 绘制训练损失对比图
        plt.figure(figsize=(12, 6))
        for method_name, result in results.items():
            plt.plot(result['val_losses'], label=f'{method_name}', linewidth=2)
        
        plt.xlabel('Epoch', fontsize=12)
        plt.ylabel('Validation Loss', fontsize=12)
        plt.title('Training Loss Comparison: PCA vs KPCA vs SPCA', fontsize=14)
        plt.legend(fontsize=11)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig('training_comparison.png', dpi=150, bbox_inches='tight')
        print(f"\n📊 训练损失对比图已保存为 training_comparison.png")
        plt.close()
        
        # 保存对比结果到 CSV
        import csv
        csv_path = 'dimensionality_reduction_comparison.csv'
        with open(csv_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(['Method', 'MSE', 'RMSE', 'MAE', 'R2', 'Training_Time(s)'])
            for method_name, result in results.items():
                metrics = result['metrics']
                writer.writerow([
                    method_name,
                    f"{metrics['MSE']:.4f}",
                    f"{metrics['RMSE']:.4f}",
                    f"{metrics['MAE']:.4f}",
                    f"{metrics['R2']:.4f}",
                    f"{result['training_time']:.2f}"
                ])
        print(f"📄 对比结果已保存为 {csv_path}")
    
    return results


if __name__ == "__main__":
    # 定义三种降维方法的数据文件路径
    data_paths = {
        'PCA': './processed_data/model_ready_data_pca.pkl',
        'KPCA': './processed_data/model_ready_data_kpca.pkl',
        'SPCA': './processed_data/model_ready_data_spca.pkl',
    }
    
    # 运行对比实验
    results = compare_all_methods(data_paths)
    
    print("\n" + "="*80)
    print("对比实验全部完成！")
    print("="*80)
