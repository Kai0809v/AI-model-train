import torch
from torch.utils.data import Dataset, DataLoader
import joblib
import os


class PVSlidingWindowDataset(Dataset):
    def __init__(self, features, targets, time_features, seq_len=96, label_len=48, pred_len=24):
        """
        Informer 专用滑动窗口数据集
        :param features: 经过 Boruta-PCA 处理后的特征矩阵 [N, feature_dim]
        :param targets: 缩放后的功率目标值 [N]
        :param time_features: 时间特征矩阵 [N, 4] (Month, Hour, Day, DayOfWeek)
        :param seq_len: 编码器输入序列长度 (TCN的输入)
        :param label_len: 解码器引导序列长度 (已知的历史真实值)
        :param pred_len: 预测序列长度
        """
        self.features = features
        self.targets = targets
        self.time_features = time_features
        self.seq_len = seq_len
        self.label_len = label_len
        self.pred_len = pred_len

    def __len__(self):
        return len(self.features) - self.seq_len - self.pred_len + 1

    def __getitem__(self, index):
        s_begin = index
        s_end = s_begin + self.seq_len
        r_begin = s_end - self.label_len
        r_end = r_begin + self.label_len + self.pred_len

        # 编码器输入 (PCA 特征)
        seq_x = self.features[s_begin:s_end]
        # 解码器输入 (PCA 特征)
        dec_x = self.features[r_begin:r_end]

        # 时间标记 (用于 Informer 的时间编码)
        seq_x_mark = self.time_features[s_begin:s_end]
        dec_x_mark = self.time_features[r_begin:r_end]

        # 预测目标
        target_y = self.targets[s_end:s_end + self.pred_len]

        return (
            torch.FloatTensor(seq_x),
            torch.FloatTensor(seq_x_mark),
            torch.FloatTensor(dec_x),
            torch.FloatTensor(dec_x_mark),
            torch.FloatTensor(target_y)
        )


def create_dataloaders(pkl_path, seq_len=96, label_len=48, pred_len=24, batch_size=32):
    """
    加载 PKL 数据包并生成 Train, Val, Test 的 DataLoader
    """
    print(f"1. 正在加载优化的数据包: {pkl_path}")
    bundle = joblib.load(pkl_path)

    train_x, train_y = bundle['train']
    val_x, val_y = bundle['val']
    test_x, test_y = bundle['test']

    # 提取时间特征
    train_time = bundle['time_features'][0]
    val_time = bundle['time_features'][1]
    test_time = bundle['time_features'][2]

    print("2. 正在构建 PyTorch Dataset...")
    train_dataset = PVSlidingWindowDataset(train_x, train_y, train_time, seq_len, label_len, pred_len)
    val_dataset = PVSlidingWindowDataset(val_x, val_y, val_time, seq_len, label_len, pred_len)
    test_dataset = PVSlidingWindowDataset(test_x, test_y, test_time, seq_len, label_len, pred_len)

    print("3. 正在封装 DataLoader...")
    # 只有训练集需要 shuffle (打乱)，验证集和测试集必须保持时间序列顺序
    # 添加 num_workers 和 pin_memory 加速数据加载：
    # - num_workers: 使用子进程并行加载数据
    # - pin_memory: 锁页内存，加速 CPU→GPU 传输
    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=True, 
        drop_last=True,
        num_workers=4,
        pin_memory=True
    )
    val_loader = DataLoader(
        val_dataset, 
        batch_size=batch_size, 
        shuffle=False, 
        drop_last=False,
        num_workers=2,
        pin_memory=True
    )
    test_loader = DataLoader(
        test_dataset, 
        batch_size=batch_size, 
        shuffle=False, 
        drop_last=False,
        num_workers=2,
        pin_memory=True
    )

    print(f"✅ DataLoader 构建完成！")
    print(f"   -> Train Batches: {len(train_loader)} (总样本: {len(train_dataset)})")
    print(f"   -> Val Batches:   {len(val_loader)} (总样本: {len(val_dataset)})")
    print(f"   -> Test Batches:  {len(test_loader)} (总样本: {len(test_dataset)})")

    return train_loader, val_loader, test_loader, bundle


# ================= 测试运行 =================
if __name__ == "__main__":
    # 替换为您上一步保存的 pkl 路径
    PKL_FILE = "processed_data/model_ready_data.pkl"

    if os.path.exists(PKL_FILE):
        train_dl, val_dl, test_dl, bundle_info = create_dataloaders(PKL_FILE)

        # 提取第一个 Batch 查看维度是否正确
        seq_x, dec_x, dec_y, target_y = next(iter(train_dl))

        print("\n--- Tensor 维度检查 ---")
        print(f"TCN/编码器输入特征 seq_x:  {seq_x.shape}  -> [Batch, seq_len, PCA_features]")
        print(f"解码器引导特征 dec_x:      {dec_x.shape}  -> [Batch, label_len + pred_len, PCA_features]")
        print(f"解码器目标引导 dec_y:      {dec_y.shape}  -> [Batch, label_len + pred_len]")
        print(f"真实预测计算目标 target_y: {target_y.shape}  -> [Batch, pred_len]")
    else:
        print(f"未找到文件 {PKL_FILE}，请确认上一步特征工程脚本已成功运行。")