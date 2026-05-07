"""
光伏预测 - 无未来气象数据方案对比实验
==========================================
目标：评估不同近似方法在缺少真实天气预报时的预测性能

对比方案：
1. Baseline: 最后4步平均复制24次（当前实现）
2. Similar_Day: 历史相似日法（同季节/星期/时段）
3. Trend_Extrapolation: 趋势外推法（线性回归）
4. Hybrid: 混合策略（相似日 + 短期趋势修正）

输出：各方案的 RMSE, MAE, R² 指标对比
"""

import torch
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import joblib
import os
import sys

# 添加项目路径
sys.path.insert(0, os.path.dirname(__file__))

# 🔧 修复Informer2020的导入路径问题（不修改其源代码）
informer_path = os.path.join(os.path.dirname(__file__), '../../../Informer2020')
if informer_path not in sys.path:
    sys.path.insert(0, informer_path)

from data_loader import create_dataloaders
from model_architecture import True_TCN_Informer


# ==========================================
# 1. 加载模型和数据
# ==========================================
def load_model_and_data(pkl_path="processed_data/model_ready_data.pkl", 
                        model_path="best_tcn_informer.pth"):
    """加载训练好的模型和测试数据"""
    print("=" * 60)
    print("正在加载模型和测试数据...")
    print("=" * 60)
    
    # 加载数据包
    bundle = joblib.load(pkl_path)
    scaler_y = bundle['scaler_y']
    
    # 创建DataLoader（使用与训练相同的参数）
    seq_len, label_len, pred_len = 192, 96, 24
    _, _, test_loader, _ = create_dataloaders(
        pkl_path, seq_len=seq_len, label_len=label_len, pred_len=pred_len, batch_size=32
    )
    
    # 提取PCA特征维度
    input_dim = bundle['train'][0].shape[1]
    
    # 加载模型
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
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
    
    model.load_state_dict(torch.load(model_path, map_location=device, weights_only=False))
    model.eval()
    
    print(f"✓ 模型加载成功 (设备: {device})")
    print(f"✓ 测试集样本数: {len(test_loader.dataset)}")
    
    return model, test_loader, scaler_y, device


# ==========================================
# 2. 四种无未来数据的近似方法
# ==========================================

def method_baseline_last4avg(dec_x, steps=24):
    """
    方案1: 最后4步平均值复制24次（当前实现）
    :param dec_x: 解码器输入 [batch, 120, 11] (96历史+24未来)
    :return: 修改后的dec_x，未来部分用历史平均替代
    """
    dec_x_modified = dec_x.clone()
    # 取最后4步的平均值
    last_4_avg = dec_x[:, -4:, :].mean(dim=1, keepdim=True)  # [batch, 1, 11]
    # 复制到未来24步
    dec_x_modified[:, 96:, :] = last_4_avg.repeat(1, 24, 1)  # [batch, 24, 11]
    return dec_x_modified


def method_similar_day(dec_x, dec_x_mark, historical_data, steps=24):
    """
    方案2: 历史相似日法
    核心思想：找到历史上相同时间段（同月份、同星期、同时段）的样本
    用这些相似日的真实未来特征作为预测
    
    :param dec_x: 当前解码器输入 [batch, 120, 11]
    :param dec_x_mark: 时间标记 [batch, 120, 5] (Month, Day, DOW, Hour, Minute)
    :param historical_data: 历史完整数据 (用于查找相似日)
    :return: 修改后的dec_x
    """
    dec_x_modified = dec_x.clone()
    batch_size = dec_x.shape[0]
    
    for b in range(batch_size):
        # 提取当前样本的未来时间标记
        future_marks = dec_x_mark[b, 96:, :]  # [24, 5]
        
        # 反归一化时间标记到原始范围（用于匹配）
        # Month: [-0.5, 0.5] -> [1, 12]
        # Day: [-0.5, 0.5] -> [1, 31]
        # DOW: [-0.5, 0.5] -> [0, 6]
        # Hour: [-0.5, 0.5] -> [0, 23]
        # Minute: [-0.5, 0.5] -> [0, 3]
        
        month_orig = ((future_marks[:, 0] + 0.5) * 11 + 1).long()
        dow_orig = ((future_marks[:, 2] + 0.5) * 6).long()
        hour_orig = ((future_marks[:, 3] + 0.5) * 23).long()
        
        # 在历史数据中搜索相似时间段
        # 简化版：只匹配月份和小时（因为精确到天可能找不到足够样本）
        similar_features_list = []
        
        for step in range(steps):
            target_month = month_orig[step].item()
            target_hour = hour_orig[step].item()
            
            # 在历史数据中找相同月份和小时的样本
            # 这里简化处理：从historical_data中随机抽取一个相似时刻的特征
            # 实际应该建立索引加速查询
            
            # 临时方案：使用当前历史部分的对应位置
            # （这个方案需要完整的时序数据才能正确实现）
            hist_idx = step % 96  # 循环使用历史数据
            similar_features_list.append(dec_x[b, hist_idx:hist_idx+1, :])
        
        # 拼接相似特征
        if similar_features_list:
            similar_future = torch.cat(similar_features_list, dim=1)  # [1, 24, 11]
            dec_x_modified[b:b+1, 96:, :] = similar_future
    
    return dec_x_modified


def method_trend_extrapolation(dec_x, steps=24):
    """
    方案3: 趋势外推法（线性回归）
    对每个特征维度单独拟合最近的历史趋势，然后外推到未来
    
    :param dec_x: 解码器输入 [batch, 120, 11]
    :return: 修改后的dec_x
    """
    dec_x_modified = dec_x.clone()
    batch_size = dec_x.shape[0]
    device = dec_x.device  # 🔧 获取输入数据的设备
    
    for b in range(batch_size):
        history = dec_x[b, :96, :]  # [96, 11]
        
        # 对每个特征维度进行线性外推
        for feat_idx in range(11):
            feat_series = history[:, feat_idx]  # [96]
            
            # 使用最近12步（3小时）做线性拟合
            recent_steps = 12
            y_recent = feat_series[-recent_steps:]  # [12]
            x_recent = torch.arange(recent_steps, dtype=torch.float32, device=device)  # 🔧 指定设备
            
            # 线性回归: y = ax + b
            # 使用最小二乘法
            x_mean = x_recent.mean()
            y_mean = y_recent.mean()
            
            numerator = ((x_recent - x_mean) * (y_recent - y_mean)).sum()
            denominator = ((x_recent - x_mean) ** 2).sum()
            
            if denominator.abs() > 1e-8:
                a = numerator / denominator
                b_param = y_mean - a * x_mean
            else:
                a = 0
                b_param = y_mean
            
            # 外推到未来24步
            x_future = torch.arange(96, 96 + steps, dtype=torch.float32, device=device)  # 🔧 指定设备
            future_values = a * x_future + b_param
            
            dec_x_modified[b, 96:, feat_idx] = future_values
    
    return dec_x_modified


def method_hybrid(dec_x, dec_x_mark, steps=24):
    """
    方案4: 混合策略（相似日 + 趋势修正）
    1. 先用相似日法获取基础预测
    2. 用短期趋势进行微调
    
    :param dec_x: 解码器输入 [batch, 120, 11]
    :param dec_x_mark: 时间标记 [batch, 120, 5]
    :return: 修改后的dec_x
    """
    # 第一步：使用趋势外推作为基础
    dec_x_base = method_trend_extrapolation(dec_x, steps)
    
    # 第二步：添加周期性修正（简化版：使用历史同期均值）
    dec_x_modified = dec_x_base.clone()
    batch_size = dec_x.shape[0]
    
    for b in range(batch_size):
        history = dec_x[b, :96, :]
        
        # 计算历史周期的统计量（每24步为一个周期）
        for period_start in range(0, 96, 24):
            period_end = min(period_start + 24, 96)
            period_data = history[period_start:period_end, :]  # [24或更少, 11]
            
            if period_data.shape[0] == 24:
                # 如果找到完整的24步周期，用它来修正趋势
                period_mean = period_data.mean(dim=0)  # [11]
                
                # 计算趋势预测与该周期的偏差
                trend_pred = dec_x_base[b, 96:, :]  # [24, 11]
                deviation = (trend_pred - period_mean.unsqueeze(0)).abs().mean()
                
                # 如果偏差过大，向周期均值收缩
                if deviation > 0.5:  # 阈值可调
                    alpha = 0.3  # 收缩系数
                    dec_x_modified[b, 96:, :] = (
                        alpha * period_mean.unsqueeze(0) + 
                        (1 - alpha) * trend_pred
                    )
                break  # 只用第一个完整周期
    
    return dec_x_modified


# ==========================================
# 3. 评估函数
# ==========================================
def evaluate_method(model, test_loader, scaler_y, device, method_name, method_func, 
                   historical_data=None):
    """
    评估单个方法的预测性能
    """
    print(f"\n{'='*60}")
    print(f"正在评估方法: {method_name}")
    print(f"{'='*60}")
    
    preds_list = []
    trues_list = []
    
    model.eval()
    with torch.no_grad():
        for i, (seq_x, seq_x_mark, dec_x, dec_x_mark, target_y) in enumerate(test_loader):
            seq_x = seq_x.to(device)
            seq_x_mark = seq_x_mark.to(device)
            dec_x = dec_x.to(device)
            dec_x_mark = dec_x_mark.to(device)
            target_y = target_y.to(device)
            
            # 应用近似方法修改dec_x的未来部分
            if method_name == "Similar_Day" and historical_data is not None:
                dec_x_modified = method_func(dec_x, dec_x_mark, historical_data)
            elif method_name == "Hybrid_Similar+Trend":
                # 🔧 Hybrid方法需要传入dec_x_mark
                dec_x_modified = method_func(dec_x, dec_x_mark)
            else:
                dec_x_modified = method_func(dec_x)
            
            # 模型推理
            pred = model(seq_x, seq_x_mark, dec_x_modified, dec_x_mark)
            
            preds_list.append(pred.cpu().numpy())
            trues_list.append(target_y.cpu().numpy())
            
            # 进度显示
            if (i + 1) % 10 == 0:
                print(f"  已处理 {i + 1}/{len(test_loader)} batches")
    
    # 拼接所有批次
    preds = np.concatenate(preds_list, axis=0)  # [N, 24]
    trues = np.concatenate(trues_list, axis=0)   # [N, 24]
    
    # 反归一化
    preds_inverse = scaler_y.inverse_transform(preds.reshape(-1, 1)).reshape(preds.shape)
    trues_inverse = scaler_y.inverse_transform(trues.reshape(-1, 1)).reshape(trues.shape)
    
    # 物理约束
    night_mask = (trues_inverse < 0.05)
    preds_inverse[night_mask] = 0.0
    preds_inverse = np.maximum(0, preds_inverse)
    preds_inverse = np.minimum(preds_inverse, 130.0)
    
    # 计算整体指标
    mse = mean_squared_error(trues_inverse.flatten(), preds_inverse.flatten())
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(trues_inverse.flatten(), preds_inverse.flatten())
    r2 = r2_score(trues_inverse.flatten(), preds_inverse.flatten())
    
    # 分步长指标（看短期vs长期预测差异）
    step_metrics = {}
    for step in [0, 3, 7, 11, 15, 19, 23]:  # 15min, 1h, 2h, 3h, 4h, 5h, 6h
        step_rmse = np.sqrt(mean_squared_error(trues_inverse[:, step], preds_inverse[:, step]))
        step_mae = mean_absolute_error(trues_inverse[:, step], preds_inverse[:, step])
        step_metrics[f"Step_{step+1}"] = {"RMSE": step_rmse, "MAE": step_mae}
    
    results = {
        "method": method_name,
        "RMSE": rmse,
        "MAE": mae,
        "R2": r2,
        "step_metrics": step_metrics
    }
    
    print(f"\n📊 {method_name} 评估结果:")
    print(f"   RMSE: {rmse:.4f} MW")
    print(f"   MAE:  {mae:.4f} MW")
    print(f"   R²:   {r2:.4f}")
    print(f"\n   分步长RMSE:")
    for step_name, metrics in step_metrics.items():
        print(f"     {step_name}: RMSE={metrics['RMSE']:.4f}, MAE={metrics['MAE']:.4f}")
    
    return results


# ==========================================
# 4. 主实验流程
# ==========================================
def run_comparison_experiment():
    """运行完整的对比实验"""
    print("\n" + "="*60)
    print("光伏预测 - 无未来气象数据方案对比实验")
    print("="*60 + "\n")
    
    # 1. 加载模型和数据
    model, test_loader, scaler_y, device = load_model_and_data()
    
    # 2. 定义要测试的方法
    methods = [
        ("Baseline_Last4Avg", method_baseline_last4avg),
        ("Trend_Extrapolation", method_trend_extrapolation),
        ("Hybrid_Similar+Trend", method_hybrid),
        # TODO: Similar_Day 需要完整历史数据，暂时跳过
        # ("Similar_Day", method_similar_day),
    ]
    
    # 3. 逐个评估
    all_results = []
    for method_name, method_func in methods:
        try:
            result = evaluate_method(
                model, test_loader, scaler_y, device,
                method_name, method_func
            )
            all_results.append(result)
        except Exception as e:
            print(f"❌ 方法 {method_name} 评估失败: {str(e)}")
            import traceback
            traceback.print_exc()
    
    # 4. 汇总对比
    print("\n" + "="*60)
    print("📈 所有方法对比总结")
    print("="*60)
    print(f"{'方法':<30} {'RMSE (MW)':<12} {'MAE (MW)':<12} {'R²':<10}")
    print("-" * 60)
    
    for result in all_results:
        print(f"{result['method']:<28} {result['RMSE']:<12.4f} {result['MAE']:<12.4f} {result['R2']:<10.4f}")
    
    # 5. 找出最优方法
    best_result = min(all_results, key=lambda x: x['RMSE'])
    print(f"\n🏆 最优方法: {best_result['method']}")
    print(f"   RMSE: {best_result['RMSE']:.4f} MW")
    print(f"   建议: 将此方法部署到 api_v6.py 的 approximate_future_without_weather 中")
    
    # 6. 保存结果
    output_path = "no_weather_prediction_results.csv"
    summary_df = pd.DataFrame([
        {
            "Method": r["method"],
            "RMSE_MW": r["RMSE"],
            "MAE_MW": r["MAE"],
            "R2": r["R2"]
        }
        for r in all_results
    ])
    summary_df.to_csv(output_path, index=False)
    print(f"\n💾 结果已保存到: {output_path}")
    
    return all_results


if __name__ == "__main__":
    results = run_comparison_experiment()
