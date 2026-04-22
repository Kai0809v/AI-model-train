import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import joblib
import sys

sys.path.append('..')

from data_loader import create_dataloaders
from model_architecture import (
    True_TCN_Informer, BaselineFCModel, 
    InformerOnlyModel, TCN_LinearModel
)
from PV_part1_for_ab import run_feature_optimization_pipeline, run_without_bp_pipeline


PKL_WITH_BP = "../processed_data/model_ready_data.pkl"
PKL_NO_BP = "../processed_data/model_ready_data_no_bp.pkl"
DATA_PATH = "../data/PV130MW.xlsx"
SEQ_LEN, LABEL_LEN, PRED_LEN = 96, 48, 24
EPOCHS = 50
BATCH_SIZE = 32
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'


def calculate_metrics(y_true, y_pred):
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    return {'MSE': mse, 'RMSE': rmse, 'MAE': mae, 'R2': r2}


def get_model(exp_id, input_dim):
    """根据实验ID返回对应模型"""
    if exp_id == 0:
        return BaselineFCModel(input_dim, pred_len=PRED_LEN, hidden_dim=128, dropout=0.2)
    elif exp_id == 1:
        return TCN_LinearModel(input_dim, tcn_channels=[16, 32], pred_len=PRED_LEN, dropout=0.15)
    elif exp_id == 2:
        return InformerOnlyModel(input_dim, SEQ_LEN, LABEL_LEN, PRED_LEN, d_model=64, n_heads=4, e_layers=2, dropout=0.15)
    elif exp_id == 3:
        return True_TCN_Informer(tcn_input_dim=input_dim, tcn_channels=[16, 32], seq_len=SEQ_LEN, label_len=LABEL_LEN, pred_len=PRED_LEN, d_model=64, n_heads=4, e_layers=2, dropout=0.15)
    elif exp_id == 4:
        return True_TCN_Informer(tcn_input_dim=input_dim, tcn_channels=[16, 32], seq_len=SEQ_LEN, label_len=LABEL_LEN, pred_len=PRED_LEN, d_model=64, n_heads=4, e_layers=2, dropout=0.15)


def train_model(model, train_loader, val_loader, epochs, device, exp_name, patience=10):
    """
    训练模型（带早停机制）
    :param patience: 验证集损失不再改善时的容忍轮数
    """
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)
    
    best_val_loss = float('inf')
    best_state = None
    train_losses, val_losses = [], []
    patience_counter = 0  # 早停计数器
    
    for epoch in range(epochs):
        model.train()
        train_loss = []
        for seq_x, seq_x_mark, dec_x, dec_x_mark, target_y in train_loader:
            seq_x, seq_x_mark = seq_x.to(device), seq_x_mark.to(device)
            dec_x, dec_x_mark = dec_x.to(device), dec_x_mark.to(device)
            target_y = target_y.to(device)
            
            optimizer.zero_grad()
            if exp_name == "Baseline_FC":
                pred = model(seq_x, seq_x_mark, dec_x, dec_x_mark)
            elif exp_name in ["BP_TCN", "NoBP_TCN"]:
                pred = model(seq_x, seq_x_mark, dec_x, dec_x_mark, pred_len=PRED_LEN)
            else:
                pred = model(seq_x, seq_x_mark, dec_x, dec_x_mark)
            
            loss = criterion(pred.squeeze(-1), target_y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            train_loss.append(loss.item())
        
        model.eval()
        val_loss = []
        with torch.no_grad():
            for seq_x, seq_x_mark, dec_x, dec_x_mark, target_y in val_loader:
                seq_x, seq_x_mark = seq_x.to(device), seq_x_mark.to(device)
                dec_x, dec_x_mark = dec_x.to(device), dec_x_mark.to(device)
                target_y = target_y.to(device)
                
                if exp_name == "Baseline_FC":
                    pred = model(seq_x, seq_x_mark, dec_x, dec_x_mark)
                elif exp_name in ["BP_TCN", "NoBP_TCN"]:
                    pred = model(seq_x, seq_x_mark, dec_x, dec_x_mark, pred_len=PRED_LEN)
                else:
                    pred = model(seq_x, seq_x_mark, dec_x, dec_x_mark)
                
                loss = criterion(pred.squeeze(-1), target_y)
                val_loss.append(loss.item())
        
        t_loss, v_loss = np.average(train_loss), np.average(val_loss)
        train_losses.append(t_loss)
        val_losses.append(v_loss)
        scheduler.step()
        
        # 早停逻辑：检查验证集损失是否改善
        if v_loss < best_val_loss:
            best_val_loss = v_loss
            best_state = model.state_dict().copy()
            patience_counter = 0  # 重置计数器
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"   ⏹️  早停触发！在第 {epoch+1} 轮停止（最佳验证损失: {best_val_loss:.5f}）")
                break
        
        if (epoch + 1) % 10 == 0:
            print(f"   Epoch {epoch+1:02d}: Train={t_loss:.5f}, Val={v_loss:.5f}")
    
    model.load_state_dict(best_state)
    return model, train_losses, val_losses


def evaluate_model(model, test_loader, scaler_y, device, max_capacity=130.0, exp_name=""):
    model.eval()
    preds_list, trues_list = [], []
    
    with torch.no_grad():
        for seq_x, seq_x_mark, dec_x, dec_x_mark, target_y in test_loader:
            seq_x = seq_x.to(device)
            seq_x_mark = seq_x_mark.to(device)
            dec_x = dec_x.to(device)
            dec_x_mark = dec_x_mark.to(device)
            
            if exp_name == "Baseline_FC":
                pred = model(seq_x, seq_x_mark, dec_x, dec_x_mark)
            elif exp_name in ["BP_TCN", "NoBP_TCN"]:
                pred = model(seq_x, seq_x_mark, dec_x, dec_x_mark, pred_len=PRED_LEN)
            else:
                pred = model(seq_x, seq_x_mark, dec_x, dec_x_mark)
            
            pred = pred.detach().cpu().numpy()
            true = target_y.detach().cpu().numpy()
            
            preds_list.append(pred.squeeze(-1))
            trues_list.append(true)
    
    preds = np.concatenate(preds_list, axis=0)
    trues = np.concatenate(trues_list, axis=0)
    
    preds_inv = scaler_y.inverse_transform(preds.reshape(-1, 1)).reshape(preds.shape)
    trues_inv = scaler_y.inverse_transform(trues.reshape(-1, 1)).reshape(trues.shape)
    
    night_mask = (trues_inv < 0.05)
    preds_inv[night_mask] = 0.0
    preds_inv = np.maximum(0, preds_inv)
    preds_inv = np.minimum(preds_inv, max_capacity)
    
    return preds_inv, trues_inv
def note():
    print("需要的时间有点长，耐心等着吧")

def message2(name):
    if(name == "xyd"):
        print(
            "    ▄▄▄▄     ▄▄▄▄      ▄▄▄▄    ▄▄▄▄▄    \n"
            "  ██▀▀▀▀█   ██▀▀██    ██▀▀██   ██▀▀▀██  \n"
            " ██        ██    ██  ██    ██  ██    ██ \n"
            " ██  ▄▄▄▄  ██    ██  ██    ██  ██    ██ \n"
            " ██  ▀▀██  ██    ██  ██    ██  ██    ██ \n"
            "  ██▄▄▄██   ██▄▄██    ██▄▄██   ██▄▄▄██  \n"
            "    ▀▀▀▀     ▀▀▀▀      ▀▀▀▀    ▀▀▀▀▀    ")
        print(f"居然运行成功了吗，{name}真棒👍👍👍")



def evaluate_on_subsets(model, train_loader, test_loader, scaler_y, device, exp_name):
    """在训练集和测试集上分别评估"""
    model.eval()

    def get_preds(data_loader):
        preds_list, trues_list = [], []
        with torch.no_grad():
            for seq_x, seq_x_mark, dec_x, dec_x_mark, target_y in data_loader:
                seq_x, seq_x_mark = seq_x.to(device), seq_x_mark.to(device)
                dec_x, dec_x_mark = dec_x.to(device), dec_x_mark.to(device)

                if exp_name == "Baseline_FC":
                    pred = model(seq_x, seq_x_mark, dec_x, dec_x_mark)
                elif exp_name in ["BP_TCN", "NoBP_TCN"]:
                    pred = model(seq_x, seq_x_mark, dec_x, dec_x_mark, pred_len=PRED_LEN)
                else:
                    pred = model(seq_x, seq_x_mark, dec_x, dec_x_mark)

                preds_list.append(pred.detach().cpu().numpy().squeeze(-1))
                trues_list.append(target_y.detach().cpu().numpy())

        preds = np.concatenate(preds_list, axis=0)
        trues = np.concatenate(trues_list, axis=0)
        return scaler_y.inverse_transform(preds.reshape(-1,1)).reshape(preds.shape), \
               scaler_y.inverse_transform(trues.reshape(-1,1)).reshape(trues.shape)

    train_preds, train_trues = get_preds(train_loader)
    test_preds, test_trues = get_preds(test_loader)

    return calculate_metrics(train_trues.flatten(), train_preds.flatten()), \
           calculate_metrics(test_trues.flatten(), test_preds.flatten())


def plot_results(exp_id, exp_name, trues, preds, train_losses, val_losses, y_limit=None):
    """
    绘制实验结果图表
    :param y_limit: 统一纵坐标范围，如 (0, 140)
    """
    save_dir = f"ablation_results/exp{exp_id}_{exp_name}"
    os.makedirs(save_dir, exist_ok=True)

    # 1. 预测曲线图 - 选择高功率样本
    # 找到真实值中功率最高的样本
    sample_power_max = np.max(trues, axis=1)  # 每个样本的最大功率
    best_sample_idx = np.argmax(sample_power_max)  # 最大功率样本的索引
    
    plt.figure(figsize=(12, 5))
    plt.plot(trues[best_sample_idx, :], label='Ground Truth', color='blue', marker='o', markersize=4, linewidth=2)
    plt.plot(preds[best_sample_idx, :], label='Prediction', color='red', linestyle='--', marker='x', markersize=4, linewidth=2)
    plt.title(f'Exp{exp_id}: {exp_name} - Prediction Curve (High Power Sample)', fontsize=12, fontweight='bold')
    plt.xlabel('Time Steps', fontsize=11); plt.ylabel('Power (MW)', fontsize=11)
    plt.legend(loc='best', fontsize=10); plt.grid(True, alpha=0.3)
    if y_limit:
        plt.ylim(y_limit)  # 统一纵坐标
    plt.tight_layout()
    plt.savefig(f'{save_dir}/prediction_curve.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"   📈 已绘制高功率样本曲线 (样本索引: {best_sample_idx}, 峰值: {sample_power_max[best_sample_idx]:.2f} MW)")

    # 2. 散点图
    plt.figure(figsize=(8, 8))
    plt.scatter(trues.flatten(), preds.flatten(), alpha=0.5, s=10, c='teal', edgecolors='gray', linewidth=0.5)
    max_val = max(trues.max(), preds.max())
    plt.plot([0, max_val], [0, max_val], 'r--', lw=2, label='Perfect Prediction')
    plt.xlabel('True Power (MW)', fontsize=11); plt.ylabel('Predicted Power (MW)', fontsize=11)
    plt.title(f'Exp{exp_id}: {exp_name} - Scatter Plot', fontsize=12, fontweight='bold')
    plt.legend(loc='best', fontsize=10); plt.grid(True, alpha=0.3)
    plt.xlim(0, max_val * 1.05)
    plt.ylim(0, max_val * 1.05)
    plt.tight_layout()
    plt.savefig(f'{save_dir}/scatter_plot.png', dpi=150, bbox_inches='tight')
    plt.close()

    # 3. 训练曲线
    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label='Train Loss', color='blue', linewidth=2)
    plt.plot(val_losses, label='Val Loss', color='orange', linewidth=2)
    plt.title(f'Exp{exp_id}: {exp_name} - Training Curve', fontsize=12, fontweight='bold')
    plt.xlabel('Epoch', fontsize=11); plt.ylabel('Loss (MSE)', fontsize=11)
    plt.legend(loc='best', fontsize=10); plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f'{save_dir}/training_curve.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"   ✅ 图表已保存至: {save_dir}/")


def run_ablation_experiment():
    print("=" * 60)
    print("🚀 消融实验开始")
    print("=" * 60)

    if not os.path.exists(PKL_NO_BP):
        print("生成无BP处理数据...")
        run_without_bp_pipeline(DATA_PATH)

    if not os.path.exists(PKL_WITH_BP):
        print("生成BP处理数据...")
        run_feature_optimization_pipeline(DATA_PATH)

    results = []

    # 注意这里，你如果已经跑完了前面的实验，但是因为某种原因中断了需要重新跑，那你就可以把已经跑过的实验注释掉再运行这个脚本
    exp_configs = [
        (0, "Baseline_FC", PKL_WITH_BP),  # 基线也应使用BP数据，公平对比模型架构
        (1, "BP_TCN", PKL_WITH_BP),
        (2, "BP_Informer", PKL_WITH_BP),
        (3, "BP_TCN_Informer", PKL_WITH_BP),
        (4, "NoBP_TCN_Informer", PKL_NO_BP),
    ]

    os.makedirs("ablation_results", exist_ok=True)
    # 其实我觉得消融实验的指标以测试集为准，因为测试集是未见过的，所以更代表模型的泛化能力；老师要加就加上吧
    for exp_id, exp_name, pkl_path in exp_configs:
        print(f"\n{'='*60}")
        print(f"🧪 实验 {exp_id}: {exp_name}")
        print(f"{'='*60}")

        train_loader, val_loader, test_loader, bundle = create_dataloaders(
            pkl_path, SEQ_LEN, LABEL_LEN, PRED_LEN, BATCH_SIZE
        )

        input_dim = bundle['train'][0].shape[1]
        scaler_y = bundle['scaler_y']

        print(f"   特征维度: {input_dim}")

        model = get_model(exp_id, input_dim).to(DEVICE)
        params = sum(p.numel() for p in model.parameters())
        print(f"   模型参数量: {params:,}")

        model, train_losses, val_losses = train_model(
            model, train_loader, val_loader, EPOCHS, DEVICE, exp_name
        )
        
        train_metrics, test_metrics = evaluate_on_subsets(
            model, train_loader, test_loader, scaler_y, DEVICE, exp_name
        )
        
        print(f"\n   📊 训练集: RMSE={train_metrics['RMSE']:.4f}, MAE={train_metrics['MAE']:.4f}, R²={train_metrics['R2']:.4f}")
        print(f"   📊 测试集: RMSE={test_metrics['RMSE']:.4f}, MAE={test_metrics['MAE']:.4f}, R²={test_metrics['R2']:.4f}")
        
        test_preds, test_trues = evaluate_model(
            model, test_loader, scaler_y, DEVICE, max_capacity=130.0, exp_name=exp_name
        )
        # 统一纵坐标范围 (0, 140) MW，覆盖光伏功率范围
        plot_results(exp_id, exp_name, test_trues, test_preds, train_losses, val_losses, y_limit=(0, 140))
        
        torch.save(model.state_dict(), f'ablation_results/exp{exp_id}_{exp_name}/model.pth')
        
        results.append({
            'exp_id': exp_id, 'exp_name': exp_name, 'feature_dim': input_dim, 'params': params,
            'train_RMSE': train_metrics['RMSE'], 'train_MAE': train_metrics['MAE'], 'train_R2': train_metrics['R2'],
            'test_RMSE': test_metrics['RMSE'], 'test_MAE': test_metrics['MAE'], 'test_R2': test_metrics['R2']
        })
    
    df_results = pd.DataFrame(results)
    df_results.to_csv('ablation_results/ablation_metrics.csv', index=False)
    
    print("\n" + "="*60)
    print("✅ 消融实验完成！")
    print("="*60)
    print("\n📋 实验结果汇总:")
    print(df_results.to_string(index=False))
    print(f"\n💾 结果已保存至: ablation_results/ablation_metrics.csv")


if __name__ == "__main__":
    note()
    run_ablation_experiment()
    message2("xyd")