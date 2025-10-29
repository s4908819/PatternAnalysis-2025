# recognition/unet_hipmri_s4908819/dataset_hipmri.py
"""
HipMRI 2D 切片数据集（按“Reading Nifti Files”规范适配）：
- 图像：支持 PNG / NPY / NIfTI(.nii/.nii.gz) 混合读取
- 掩码：支持 PNG / NPY / NIfTI，自动将灰度或整数标签映射为类别索引，并导出 one-hot
- 输出：img -> (1,H,W) float32 (z-score)，mask_onehot -> (K,H,W) float32
- Resize：图像用 BILINEAR，掩码用 NEAREST（避免类别插值）
- 与现有训练脚本兼容：保留 make_loader / make_loader_from_split
"""

import os, glob, random
from typing import Tuple, Optional
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
import nibabel as nib  # NIfTI 支持

# -------------------- utils --------------------

def set_seed(seed: int = 4908819):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def to_one_hot(mask_idx: np.ndarray, num_classes: int, dtype=np.float32) -> np.ndarray:
    """
    mask_idx: HxW 的整数标签(0..K-1)
    return: KxHxW 的 one-hot
    """
    h, w = mask_idx.shape
    out = np.zeros((num_classes, h, w), dtype=dtype)
    for c in range(num_classes):
        out[c] = (mask_idx == c).astype(dtype)
    return out

def _clip_uint8(x: np.ndarray) -> np.ndarray:
    return np.clip(x, 0, 255).astype(np.uint8)

def map_mask_indices_gray(gray: np.ndarray, num_classes: int) -> np.ndarray:
    """
    将灰度 mask 映射为类别索引（适配常见 {0,255} / {0,85,170,255}）。
    若灰度存在轻微噪声，按 85 的倍数做兜底映射。
    """
    vals = set(np.unique(gray).tolist())

    # 2 类（0/255）
    if vals.issubset({0, 255}) or (num_classes == 2 and max(vals, default=0) > 1):
        lut = np.zeros(256, dtype=np.uint8); lut[0] = 0; lut[255] = 1
        return lut[_clip_uint8(gray)]

    # 4 类（0/85/170/255）
    expected4 = {0, 85, 170, 255}
    if vals.issubset(expected4):
        lut = np.zeros(256, dtype=np.uint8); lut[0]=0; lut[85]=1; lut[170]=2; lut[255]=3
        return lut[_clip_uint8(gray)]

    # 兜底：就近到 85 的倍数（四分类常见）
    mapped = np.rint(gray / 85.0).astype(np.int32)
    mapped = np.clip(mapped, 0, max(1, num_classes - 1)).astype(np.uint8)
    return mapped

# -------------------- NIfTI 读取（遵循标准示例） --------------------

def _nifti_get_2d_array(path: str, dtype=np.float32) -> np.ndarray:
    """
    读取 NIfTI 并返回 2D 数组：
    - 若 3D -> 取 [:,:,0]
    - 若 4D -> 取 [:,:,:,0]
    - 若带多余单例维度 -> squeeze
    """
    img = nib.load(path)
    arr = img.get_fdata(caching="unchanged")  # 按标准：read disk only
    arr = np.asarray(arr)
    # 去掉单例维度
    arr = np.squeeze(arr)

    if arr.ndim == 2:
        return arr.astype(dtype)
    if arr.ndim == 3:
        return arr[:, :, 0].astype(dtype)  # sometimes extra dims
    if arr.ndim >= 4:
        return arr[:, :, 0, 0].astype(dtype)  # 兜底

    # 极端兜底
    return np.squeeze(arr).astype(dtype)

# -------------------- Dataset --------------------

class HipMRISliceDataset(Dataset):
    """
    加载 HipMRI/OASIS 风格的 2D 切片数据（图像/掩码成对）。
    - 自动识别 PNG / NPY / NIfTI
    - 图像 z-score 归一化（(x-mean)/std）
    - 掩码转为 one-hot（KxHxW）
    """
    def __init__(
        self,
        img_dir: str,
        mask_dir: str,
        num_classes: int,
        augment: bool = False,
        as_tensor: bool = True,
        debug_once: bool = True,
        resize_to: Optional[Tuple[int, int]] = None,  # (H, W)
    ):
        self.img_paths = sorted(glob.glob(os.path.join(img_dir, "*")))
        self.mask_paths = sorted(glob.glob(os.path.join(mask_dir, "*")))
        assert len(self.img_paths) == len(self.mask_paths) and len(self.img_paths) > 0, \
            f"pairs not found in: {img_dir} | {mask_dir}"

        self.num_classes = num_classes
        self.augment = augment
        self.as_tensor = as_tensor
        self.resize_to = resize_to
        self._printed_debug = not debug_once

    def __len__(self):
        return len(self.img_paths)

    # --- Resize（图像：BILINEAR；掩码：NEAREST） ---
    def _resize_np(self, arr: np.ndarray, is_mask: bool) -> np.ndarray:
        if self.resize_to is None:
            return arr
        H, W = self.resize_to
        if is_mask:
            pil = Image.fromarray(arr.astype(np.uint8))
            pil = pil.resize((W, H), resample=Image.NEAREST)
            return np.array(pil, dtype=np.uint8)
        else:
            pil = Image.fromarray(arr.astype(np.float32))
            pil = pil.resize((W, H), resample=Image.BILINEAR)
            return np.array(pil, dtype=np.float32)

    # --- 读取图像 ---
    def _load_image(self, p: str) -> np.ndarray:
        p_lower = p.lower()
        if p_lower.endswith(".npy"):
            arr = np.load(p).astype(np.float32)
        elif p_lower.endswith(".nii") or p_lower.endswith(".nii.gz"):
            arr = _nifti_get_2d_array(p, dtype=np.float32)  # 按标准规范读取
        else:
            # PNG/JPG -> 单通道 float32
            pil = Image.open(p).convert("F")
            arr = np.array(pil, dtype=np.float32)
        if self.resize_to is not None:
            arr = self._resize_np(arr, is_mask=False)
        return arr

    # --- 读取掩码 ---
    def _load_mask(self, p: str) -> np.ndarray:
        p_lower = p.lower()
        if p_lower.endswith(".npy"):
            raw = np.load(p)
            raw = np.squeeze(raw)
            if np.issubdtype(raw.dtype, np.integer) and raw.max() <= max(1, self.num_classes - 1):
                idx = raw.astype(np.uint8)
            else:
                idx = map_mask_indices_gray(_clip_uint8(raw), self.num_classes)
        elif p_lower.endswith(".nii") or p_lower.endswith(".nii.gz"):
            raw = _nifti_get_2d_array(p, dtype=np.float32)
            # 优先当作整数标签
            raw_round = np.rint(raw).astype(np.int32)
            if raw_round.min() >= 0 and raw_round.max() <= max(1, self.num_classes - 1):
                idx = raw_round.astype(np.uint8)
            else:
                idx = map_mask_indices_gray(_clip_uint8(raw), self.num_classes)
        else:
            pil = Image.open(p).convert("L")
            idx = np.array(pil, dtype=np.uint8)
            idx = map_mask_indices_gray(idx, self.num_classes)

        if self.resize_to is not None:
            idx = self._resize_np(idx, is_mask=True)
        return idx

    def __getitem__(self, i: int):
        ip, mp = self.img_paths[i], self.mask_paths[i]

        # ---- load
        img = self._load_image(ip)       # HxW float32
        idx = self._load_mask(mp)        # HxW uint8 (0..K-1)

        # ---- z-score normalize (按标准注释给出的做法)
        mean = img.mean()
        std = img.std() + 1e-6
        img = (img - mean) / std

        # ---- paired augmentation（可按需扩展）
        if self.augment and random.random() < 0.5:
            img = np.flip(img, axis=1).copy()
            idx = np.flip(idx, axis=1).copy()

        # ---- CHW / one-hot
        img = img[None, ...]                     # 1xHxW
        mask_oh = to_one_hot(idx, self.num_classes, dtype=np.float32)  # KxHxW

        if self.as_tensor:
            img = torch.from_numpy(img).float()
            mask_oh = torch.from_numpy(mask_oh).float()

        if not self._printed_debug:
            self._printed_debug = True
            print(f"[hipmri dataset] sample0 img {tuple(img.shape)} {img.dtype} | "
                  f"mask {tuple(mask_oh.shape)} {mask_oh.dtype}")

        return img, mask_oh

# -------------------- Dataloader 接口（保持不变） --------------------

SPLIT_TO_DIR = {
    "train":    ("keras_png_slices_train",    "keras_png_slices_seg_train"),
    "validate": ("keras_png_slices_validate", "keras_png_slices_seg_validate"),
    "val":      ("keras_png_slices_validate", "keras_png_slices_seg_validate"),
    "test":     ("keras_png_slices_test",     "keras_png_slices_seg_test"),
}

def make_loader(img_dir: str, mask_dir: str, num_classes: int, batch=16, shuffle=True,
                augment=False, workers=2, pin_memory=True,
                resize_to: Optional[Tuple[int, int]] = None) -> DataLoader:
    ds = HipMRISliceDataset(img_dir, mask_dir, num_classes,
                            augment=augment, debug_once=True, resize_to=resize_to)
    return DataLoader(ds, batch_size=batch, shuffle=shuffle,
                      num_workers=workers, pin_memory=pin_memory)

def make_loader_from_split(root: str, split: str, num_classes: int, batch=16, shuffle=True,
                           augment=False, workers=0, pin_memory=True,
                           resize_to: Optional[Tuple[int, int]] = None) -> DataLoader:
    """
    root: 指向包含 keras_png_slices_* 与 keras_png_slices_seg_* 的目录
    """
    img_sub, mask_sub = SPLIT_TO_DIR[split]
    img_dir  = os.path.join(root, img_sub)
    mask_dir = os.path.join(root, mask_sub)
    ds = HipMRISliceDataset(img_dir, mask_dir, num_classes,
                            augment=augment, debug_once=True, resize_to=resize_to)
    return DataLoader(ds, batch_size=batch,
                      shuffle=(shuffle if split == "train" else False),
                      num_workers=workers, pin_memory=pin_memory)
