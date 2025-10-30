import os, glob
from typing import Tuple, List
import numpy as np
import nibabel as nib
import torch
from torch.utils.data import Dataset
from scipy.ndimage import zoom

def to_one_hot(lbl: np.ndarray, K: int) -> np.ndarray:
    oh = np.zeros((K,)+lbl.shape, dtype=np.float32)
    for k in range(K): oh[k][lbl==k] = 1.0
    return oh

def resize3d(arr: np.ndarray, out_shape: Tuple[int,int,int], order: int) -> np.ndarray:
    """Resize 3D volume (D,H,W). order=1 for image, 0 for labels."""
    factors = [o/i for o,i in zip(out_shape, arr.shape)]
    return zoom(arr, factors, order=order)

class HipMRI3D(Dataset):
    """
    root: 目录下包含两个子目录：
      semantic_MRs/*.nii.gz            （原始 MRI 体数据）
      semantic_labels_only/*.nii.gz    （配对的标签体）
    统一输出形状：(1,D,H,W) 图像，(K,D,H,W) one-hot 标签
    """
    def __init__(self, root: str, split: str, num_classes: int=2,
                 target_shape: Tuple[int,int,int]=(64,128,128)):
        # 按病例名分割（简单切 8:1:1，可替换为你的官方 split 列表）
        imgs = sorted(glob.glob(os.path.join(root, "semantic_MRs", "*.nii*")))
        labs = sorted(glob.glob(os.path.join(root, "semantic_labels_only", "*.nii*")))
        assert len(imgs)==len(labs)>0, "No 3D pairs found."

        n = len(imgs)
        if split=="train":     self.pairs = list(zip(imgs[:int(0.8*n)], labs[:int(0.8*n)]))
        elif split=="validate":self.pairs = list(zip(imgs[int(0.8*n):int(0.9*n)], labs[int(0.8*n):int(0.9*n)]))
        else:                  self.pairs = list(zip(imgs[int(0.9*n):], labs[int(0.9*n):]))
        self.num_classes = num_classes
        self.shape = target_shape

    def __len__(self): return len(self.pairs)

    def __getitem__(self, i):
        ip, lp = self.pairs[i]
        img = nib.load(ip).get_fdata(caching='unchanged')     # (H,W,D) or (X,Y,Z)
        lbl = nib.load(lp).get_fdata(caching='unchanged')     # 同 shape，整型标签

        # 统一到 (D,H,W)
        if img.shape[0] != img.shape[-1]:  # 简单启发式
            img = np.moveaxis(img, -1, 0)    # (D,H,W)
            lbl = np.moveaxis(lbl, -1, 0)
        else:
            img = np.transpose(img, (2,0,1)); lbl = np.transpose(lbl, (2,0,1))

        # 标准化 + 尺寸对齐
        img = img.astype(np.float32)
        img = (img - img.mean()) / (img.std() + 1e-6)
        img_r = resize3d(img, self.shape, order=1)
        lbl_r = resize3d(lbl, self.shape, order=0).astype(np.uint8)

        x = torch.from_numpy(img_r[None, ...])                    # (1,D,H,W)
        y = torch.from_numpy(to_one_hot(lbl_r, self.num_classes)) # (K,D,H,W)
        return x, y
