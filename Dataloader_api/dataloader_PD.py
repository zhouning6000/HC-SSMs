# -*- coding: utf-8 -*-
import os
import glob
import random
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt

# -------------------------
# 简单的 3D 序列增广（H/V 翻转 + 90°旋转）
# -------------------------
class RandomFlipRotate3D:
    def __init__(self, p_flip=0.5, p_rot=0.5):
        self.p_flip = p_flip
        self.p_rot = p_rot

    def __call__(self, x: np.ndarray) -> np.ndarray:
        # x: (T, C, H, W) 或 (T, H, W)
        if random.random() < self.p_flip:
            # 水平翻转
            x = np.flip(x, axis=-1).copy()
        if random.random() < self.p_flip:
            # 垂直翻转
            x = np.flip(x, axis=-2).copy()
        if random.random() < self.p_rot:
            # 旋转 0/90/180/270 度
            k = random.randint(0, 3)
            # 最后两维 (H, W) 旋转
            x = np.rot90(x, k=k, axes=(-2, -1)).copy()
        return x


# -------------------------
# 读取工具（兼容 npy/npz）
# -------------------------
def _load_array(path: str, npz_key: str = None):
    if path.endswith(".npz"):
        arr = np.load(path)[npz_key] if npz_key is not None else np.load(path)["arr_0"]
    else:
        arr = np.load(path)
    return arr

# -------------------------
# 数据集实现
# -------------------------
class PollutantDiffusionDataset(Dataset):
    """
    期望每个文件包含形状为:
      - (N, T, H, W)  或
      - (N, T, 1, H, W)

    返回:
      input_seq:  (T_in, 1, H, W)
      output_seq: (T_out, 1, H, W)
    """
    def __init__(
        self,
        root: str,
        split: str = "train",
        t_in: int = 10,
        t_out: int = 10,
        file_pattern: str = "*.npy",
        npz_key: str = None,
        transform=None,
        normalize: str = "auto"  # "auto" | None
    ):
        super().__init__()
        assert split in {"train", "val", "test"}
        self.root = root
        self.split = split
        self.t_in = t_in
        self.t_out = t_out
        self.transform = transform
        self.normalize = normalize

        # 目录结构建议：root/train, root/val, root/test
        split_dir = os.path.join(root, split)
        if not os.path.isdir(split_dir):
            # 如果没有按 split 分目录，就用 root 直接匹配
            split_dir = root

        self.files = sorted(glob.glob(os.path.join(split_dir, file_pattern)))
        if len(self.files) == 0:
            raise FileNotFoundError(f"No files matched at: {split_dir}/{file_pattern}")

        self._index = []  # (file_idx, sample_idx)
        self._shapes = [] # 缓存每个文件的 (N, T, H, W) 形状以便索引
        self._npz_key = npz_key

        # 仅做元信息扫描，不把所有数据常驻内存
        for fi, f in enumerate(self.files):
            arr = _load_array(f, npz_key=self._npz_key)
            # 统一到 (N, T, H, W)
            if arr.ndim == 5:  # (N, T, C, H, W)
                if arr.shape[2] != 1:
                    raise ValueError(f"Expect channel=1, got shape {arr.shape} in {f}")
                arr = arr[:, :, 0, :, :]
            elif arr.ndim != 4:
                raise ValueError(f"Expect 4D/5D array per file, got {arr.shape} in {f}")

            N, T, H, W = arr.shape
            if T != (self.t_in + self.t_out):
                raise ValueError(
                    f"{f}: Expect T={self.t_in + self.t_out}, got T={T} (shape={arr.shape})"
                )
            self._shapes.append((N, T, H, W))
            for si in range(N):
                self._index.append((fi, si))

        # 简单记录均值/方差占位（如需全量统计可额外实现）
        self.mean = 0.0
        self.std = 1.0

    def __len__(self):
        return len(self._index)

    def __getitem__(self, idx):
        file_idx, sample_idx = self._index[idx]
        path = self.files[file_idx]
        arr = _load_array(path, npz_key=self._npz_key)  # (N, T, H, W) 或 (N, T, 1, H, W)

        if arr.ndim == 5:
            arr = arr[:, :, 0, :, :]  # 去掉通道维

        # 取出单个样本: (T, H, W)
        seq = arr[sample_idx]  # (T, H, W)
        # 若为 uint8，则归一化到 [0,1]
        if self.normalize == "auto" and seq.dtype == np.uint8:
            seq = seq.astype(np.float32) / 255.0
        else:
            seq = seq.astype(np.float32)

        # 增广/变换（在加通道维之前）
        # 期望 transform 接收 (T, H, W) 或 (T, 1, H, W) 都可，这里先按 (T, H, W) 传入
        if self.transform is not None:
            seq = self.transform(seq)

        # 加通道维 -> (T, 1, H, W)
        if seq.ndim == 3:
            seq = seq[:, None, :, :]

        # 切分输入/输出
        t_total = self.t_in + self.t_out
        assert seq.shape[0] == t_total, f"Temporal length mismatch: got {seq.shape[0]}"
        input_img = seq[: self.t_in]     # (T_in, 1, H, W)
        output_img = seq[self.t_in :]    # (T_out, 1, H, W)

        # 转成 torch tensor
        input_img = torch.from_numpy(input_img).contiguous().float()
        output_img = torch.from_numpy(output_img).contiguous().float()
        return input_img, output_img


# -------------------------
# DataLoader 构建
# -------------------------
def load_data(
    batch_size: int,
    val_batch_size: int,
    data_root: str,
    num_workers: int = 4,
    t_in: int = 10,
    t_out: int = 10,
    file_pattern: str = "*.npy",
    npz_key: str = None,
    aug: bool = False
):
    transform = RandomFlipRotate3D() if aug else None

    train_set = PollutantDiffusionDataset(
        root=data_root, split="train", t_in=t_in, t_out=t_out,
        file_pattern=file_pattern, npz_key=npz_key, transform=transform
    )
    val_set = PollutantDiffusionDataset(
        root=data_root, split="val", t_in=t_in, t_out=t_out,
        file_pattern=file_pattern, npz_key=npz_key, transform=None
    )
    test_set = PollutantDiffusionDataset(
        root=data_root, split="test", t_in=t_in, t_out=t_out,
        file_pattern=file_pattern, npz_key=npz_key, transform=None
    )

    dl_train = DataLoader(train_set, batch_size=batch_size, shuffle=True,
                          pin_memory=True, num_workers=num_workers, drop_last=True)
    dl_val = DataLoader(val_set, batch_size=val_batch_size, shuffle=False,
                        pin_memory=True, num_workers=num_workers)
    dl_test = DataLoader(test_set, batch_size=val_batch_size, shuffle=False,
                         pin_memory=True, num_workers=num_workers)

    # 若需要，可以返回 train_set.mean/std（当前实现为占位）
    mean, std = 0.0, 1.0
    return dl_train, dl_val, dl_test, mean, std


# -------------------------
# 可视化示例
# -------------------------
if __name__ == "__main__":
    # 假设目录结构：
    # data_root/
    #   ├── train/*.npy (或 .npz)
    #   ├── val/*.npy
    #   └── test/*.npy
    data_root = "./Pollutant-Diffusion"

    dataset = PollutantDiffusionDataset(
        root=data_root, split="train", t_in=10, t_out=10,
        file_pattern="*.npy", npz_key=None, transform=None
    )
    input_seq, output_seq = dataset[0]  # (10,1,H,W), (10,1,H,W)

    # 可视化输入序列
    T_in = input_seq.shape[0]
    fig, axes = plt.subplots(1, T_in, figsize=(1.8*T_in, 2))
    for i in range(T_in):
        axes[i].imshow(input_seq[i, 0].cpu().numpy(), cmap="viridis")
        axes[i].axis("off")
        axes[i].set_title(f"in-{i}")
    plt.tight_layout()
    plt.show()

    # 可视化输出序列
    T_out = output_seq.shape[0]
    fig, axes = plt.subplots(1, T_out, figsize=(1.8*T_out, 2))
    for i in range(T_out):
        axes[i].imshow(output_seq[i, 0].cpu().numpy(), cmap="viridis")
        axes[i].axis("off")
        axes[i].set_title(f"out-{i}")
    plt.tight_layout()
    plt.show()
