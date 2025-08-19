# -*- coding: utf-8 -*-
import os
import glob
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt

def _load_array(path: str, npz_key: str = None):
    """Load npy/npz array."""
    if path.endswith(".npz"):
        with np.load(path) as zf:
            if npz_key is None:
                # 默认取第一个数组
                arr = zf[list(zf.files)[0]]
            else:
                arr = zf[npz_key]
    else:
        arr = np.load(path)
    return arr

class ShallowWaterDataset(Dataset):
    """
    2D SWE 数据集读取器（单通道），支持：
      - 文件形状 (N, T, 128, 128) 或 (N, T, 1, 128, 128)
      - T 应等于 T_in + T_out
    返回:
      input_seq:  (T_in, 1, 128, 128)  float32
      output_seq: (T_out, 1, 128, 128) float32
    """
    def __init__(self,
                 root: str,
                 split: str = "train",
                 t_in: int = 10,
                 t_out: int = 10,
                 file_pattern: str = "*.npy",
                 npz_key: str = None,
                 normalize: str = None,   # None | 'minmax' | 'zscore_sample'
                 transform=None):
        super().__init__()
        assert split in {"train", "val", "test"}
        self.t_in = t_in
        self.t_out = t_out
        self.normalize = normalize
        self.transform = transform
        self.npz_key = npz_key

        # 目录结构建议：root/train, root/val, root/test（若无则直接用 root）
        split_dir = os.path.join(root, split)
        if not os.path.isdir(split_dir):
            split_dir = root

        self.files = sorted(glob.glob(os.path.join(split_dir, file_pattern)))
        if len(self.files) == 0:
            raise FileNotFoundError(f"No files matched: {split_dir}/{file_pattern}")

        # 建立 (file_idx, sample_idx) 索引
        self._index = []
        self._shapes = []
        expected_T = t_in + t_out

        for fi, f in enumerate(self.files):
            arr = _load_array(f, npz_key=self.npz_key)

            # 标准化到 (N, T, 1, 128, 128)
            if arr.ndim == 4:                 # (N, T, H, W)
                N, T, H, W = arr.shape
                arr_shape = (N, T, 1, H, W)
            elif arr.ndim == 5:               # (N, T, C, H, W)
                N, T, C, H, W = arr.shape
                if C != 1:
                    raise ValueError(f"{f}: expect single-channel (C=1), got C={C}")
                arr_shape = (N, T, C, H, W)
            else:
                raise ValueError(f"{f}: expect 4D/5D array, got shape {arr.shape}")

            if T != expected_T:
                raise ValueError(f"{f}: expect T={expected_T}, got T={T}")
            if (H, W) != (128, 128):
                raise ValueError(f"{f}: expect HxW=(128,128), got {(H, W)}")

            self._shapes.append(arr_shape)
            for si in range(N):
                self._index.append((fi, si))

        # 记录均值方差占位（如需可扩展为全局统计）
        self.mean = 0.0
        self.std = 1.0

    def __len__(self):
        return len(self._index)

    def __getitem__(self, idx):
        file_idx, sample_idx = self._index[idx]
        path = self.files[file_idx]
        arr = _load_array(path, npz_key=self.npz_key)

        # 统一到 (N, T, 1, 128, 128)
        if arr.ndim == 4:  # (N, T, H, W) -> (N, T, 1, H, W)
            arr = arr[:, :, None, :, :]

        # 取该样本序列: (T, 1, 128, 128)
        seq = arr[sample_idx].astype(np.float32)

        # 可选归一化
        if self.normalize == "minmax":
            # 逐样本 min-max
            vmin, vmax = np.min(seq), np.max(seq)
            if vmax > vmin:
                seq = (seq - vmin) / (vmax - vmin)
        elif self.normalize == "zscore_sample":
            mu, sigma = seq.mean(), seq.std()
            sigma = sigma if sigma > 1e-6 else 1.0
            seq = (seq - mu) / sigma

        # 额外 transform（若传入）; 期望接收/返回 (T, 1, 128, 128)
        if self.transform is not None:
            seq = self.transform(seq)

        # 切分输入/输出
        T_total = self.t_in + self.t_out
        assert seq.shape[0] == T_total, f"Temporal length mismatch: {seq.shape[0]} vs {T_total}"
        input_seq = seq[:self.t_in]      # (T_in, 1, 128, 128)
        output_seq = seq[self.t_in:]     # (T_out, 1, 128, 128)

        # 转为 torch.Tensor
        input_seq = torch.from_numpy(input_seq).contiguous().float()
        output_seq = torch.from_numpy(output_seq).contiguous().float()
        return input_seq, output_seq

def load_data(batch_size: int,
              val_batch_size: int,
              data_root: str,
              num_workers: int = 4,
              t_in: int = 10,
              t_out: int = 10,
              file_pattern: str = "*.npy",
              npz_key: str = None,
              normalize: str = None,
              aug: bool = False):
    """
    返回 train/val/test 三个 DataLoader 以及 (mean, std) 占位。
    """
    transform = None  # SWE 常为物理量，默认不做几何增广；如需可自定义 transform

    train_set = ShallowWaterDataset(root=data_root, split="train",
                                    t_in=t_in, t_out=t_out,
                                    file_pattern=file_pattern, npz_key=npz_key,
                                    normalize=normalize, transform=transform)
    val_set = ShallowWaterDataset(root=data_root, split="val",
                                  t_in=t_in, t_out=t_out,
                                  file_pattern=file_pattern, npz_key=npz_key,
                                  normalize=normalize, transform=None)
    test_set = ShallowWaterDataset(root=data_root, split="test",
                                   t_in=t_in, t_out=t_out,
                                   file_pattern=file_pattern, npz_key=npz_key,
                                   normalize=normalize, transform=None)

    dl_train = DataLoader(train_set, batch_size=batch_size, shuffle=True,
                          pin_memory=True, num_workers=num_workers, drop_last=True)
    dl_val = DataLoader(val_set, batch_size=val_batch_size, shuffle=False,
                        pin_memory=True, num_workers=num_workers)
    dl_test = DataLoader(test_set, batch_size=val_batch_size, shuffle=False,
                         pin_memory=True, num_workers=num_workers)

    mean, std = 0.0, 1.0
    return dl_train, dl_val, dl_test, mean, std

# -------------------------
# 简单可视化示例
# -------------------------
if __name__ == "__main__":
    # 目录结构建议：
    # SWE/
    #  ├─ train/*.npy (或 .npz)
    #  ├─ val/*.npy
    #  └─ test/*.npy
    root = "./SWE"

    ds = ShallowWaterDataset(root=root, split="train",
                             t_in=10, t_out=10,
                             file_pattern="*.npy", npz_key=None,
                             normalize=None)
    x, y = ds[0]   # x: (10,1,128,128), y: (10,1,128,128)

    # 可视化前 5 帧输入/输出
    import math
    k = 5
    fig, axs = plt.subplots(2, k, figsize=(2*k, 4))
    for i in range(k):
        axs[0, i].imshow(x[i, 0].numpy(), cmap="viridis")
        axs[0, i].axis("off"); axs[0, i].set_title(f"in-{i}")
        axs[1, i].imshow(y[i, 0].numpy(), cmap="viridis")
        axs[1, i].axis("off"); axs[1, i].set_title(f"out-{i}")
    plt.tight_layout(); plt.show()
