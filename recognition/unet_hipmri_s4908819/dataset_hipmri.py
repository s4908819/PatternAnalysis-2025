# recognition/unet_hipmri_s4908819/dataset_hipmri.py
"""
HipMRI 2D PNG 切片数据集加载（与 OASIS 版接口保持一致）：
- 输出：img (1,H,W) float32 ；mask_onehot (K,H,W) float32
- 支持灰度标签到类别索引的映射：{0,255} 或 {0,85,170,255} 等
- 目录结构（root 指向包含这些子目录的路径）：
    keras_png_slices_train/
    keras_png_slices_seg_train/
    keras_png_slices_validate/
    keras_png_slices_seg_validate/
    keras_png_slices_test/
    keras_png_slices_seg_test/
"""

import os, glob, random
from typing import List, Tuple, Optional
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader

# ------------- utils -------------
def set_seed(seed: int = 4908819):
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)

def to_one_hot(mask: np.ndarray, num_classes: int) -> np.ndarray:
    """
    mask: HxW integer class indices in [0..K-1]
    return: CxHxW float32 one-hot (0/1)
    """
    h, w = mask.shape
    oh = np.zeros((num_classes, h, w), dtype=np.float32)
    for c in range(num_classes):
        oh[c] = (mask == c).astype(np.float32)
    return oh

def map_mask_indices_gray(m: np.ndarray, num_classes: int) -> np.ndarray:
    """
    将灰度标签映射为类别索引（适配 HipMRI 常见标注）：
      - 二分类：{0,255} -> {0,1}
      - 四分类：{0,85,170,255} -> {0,1,2,3}
    若灰度存在轻微噪声，按 85 的倍数就近映射兜底。
    """
    vals = set(np.unique(m).tolist())

    # 二分类（0/255）
    if vals.issubset({0, 255}) or (num_classes == 2 and max(vals) > 1):
        lut = np.zeros(256, dtype=np.uint8)
        lut[0] = 0
        lut[255] = 1
        return lut[m].astype(np.uint8)

    # 四分类（0/85/170/255）
    expected4 = {0, 85, 170, 255}
    if vals.issubset(expected4):
        lut = np.zeros(256, dtype=np.uint8)
        lut[0] = 0; lut[85] = 1; lut[170] = 2; lut[255] = 3
        return lut[m].astype(np.uint8)

    # 兜底：就近到 85 的倍数
    mapped = np.rint(m / 85.0).astype(np.int32)
    mapped = np.clip(mapped, 0, max(1, num_classes - 1)).astype(np.uint8)
    return mapped

# ------------- dataset -------------

class HipMRISliceDataset(Dataset):
    """
    加载 HipMRI 的 2D (image, mask) 对（PNG 或 NPY）。
    与你现有 SlicePairDataset 的行为保持一致，便于无缝替换。
    """
    def __init__(
        self,
        img_dir: str,
        mask_dir: str,
        num_classes: int,
        augment: bool = False,
        as_tensor: bool = True,
        debug_once: bool = False,
        resize_to: Optional[Tuple[int, int]] = None  # (H,W)，默认为不缩放
    ):
        self.img_paths = sorted(glob.glob(os.path.join(img_dir, "*")))
        self.mask_paths = sorted(glob.glob(os.path.join(mask_dir, "*")))
        assert len(self.img_paths) == len(self.mask_paths) and len(self.img_paths) > 0, \
            f"pairs not found in: {img_dir} | {mask_dir}"

        self.num_classes = num_classes
        self.augment = augment
        self.as_tensor = as_tensor
        self.resize_to = resize_to
        self._printed_debug = not debug_once  # 若 True 则不再打印

    def __len__(self): return len(self.img_paths)

    def _maybe_resize(self, pil_img: Image.Image, is_mask: bool) -> Image.Image:
        if self.resize_to is None:
            return pil_img
        # mask 用 NEAREST，避免插值带来类别混淆；图像用 BILINEAR
        resample = Image.NEAREST if is_mask else Image.BILINEAR
        H, W = self.resize_to
        return pil_img.resize((W, H), resample=resample)

    def _load_img(self, ip: str) -> np.ndarray:
        if ip.lower().endswith(".npy"):
            arr = np.load(ip).astype(np.float32)
        else:
            # 单通道灰度 -> float32
            pil = Image.open(ip).convert("F")
            if self.resize_to is not None:
                pil = self._maybe_resize(pil, is_mask=False)
            arr = np.array(pil, dtype=np.float32)
        return arr

    def _load_mask_gray(self, mp: str) -> np.ndarray:
        """
        加载灰度 mask，并映射到类索引（见 map_mask_indices_gray）。
        输出为 HxW 的 uint8 索引（0..K-1）。
        """
        if mp.lower().endswith(".npy"):
            raw = np.load(mp).astype(np.int64)
            if raw.max() <= max(1, self.num_classes - 1):
                idx = raw.astype(np.uint8)
            else:
                raw = np.clip(raw, 0, 255).astype(np.uint8)
                idx = map_mask_indices_gray(raw, self.num_classes)
        else:
            pil = Image.open(mp).convert("L")
            if self.resize_to is not None:
                pil = self._maybe_resize(pil, is_mask=True)
            gray = np.array(pil, dtype=np.uint8)
            idx = map_mask_indices_gray(gray, self.num_classes)
        return idx

    def __getitem__(self, i: int):
        ip, mp = self.img_paths[i], self.mask_paths[i]

        # ---- load
        img = self._load_img(ip)             # HxW float32
        idx = self._load_mask_gray(mp)       # HxW uint8 in [0..K-1]

        # ---- z-score normalize image
        m, s = img.mean(), img.std() + 1e-6
        img = (img - m) / s  # HxW float32

        # ---- simple paired augmentation（与 OASIS 版一致，必要时可扩展）
        if self.augment and random.random() < 0.5:
            img = np.flip(img, axis=1).copy()
            idx = np.flip(idx, axis=1).copy()

        # ---- to CHW
        img = img[None, ...]  # 1xHxW
        mask_oh = to_one_hot(idx.astype(np.int64), self.num_classes)  # CxHxW float32

        if self.as_tensor:
            img = torch.from_numpy(img).float()
            mask_oh = torch.from_numpy(mask_oh).float()

        # ---- one-time debug print
        if not self._printed_debug:
            self._printed_debug = True
            print(f"[hipmri dataset] sample0 img {tuple(img.shape)} {img.dtype} "
                  f"| mask {tuple(mask_oh.shape)} {mask_oh.dtype}")

        return img, mask_oh

# ------------- loader -------------

# 与 OASIS 相同的 split → 子目录映射
SPLIT_TO_DIR = {
    "train":    ("keras_png_slices_train",    "keras_png_slices_seg_train"),
    "validate": ("keras_png_slices_validate", "keras_png_slices_seg_validate"),
    "val":      ("keras_png_slices_validate", "keras_png_slices_seg_validate"),
    "test":     ("keras_png_slices_test",     "keras_png_slices_seg_test"),
}

def make_loader(img_dir: str, mask_dir: str, num_classes: int, batch=16, shuffle=True,
                augment=False, workers=2, pin_memory=True, resize_to: Optional[Tuple[int,int]] = None) -> DataLoader:
    ds = HipMRISliceDataset(img_dir, mask_dir, num_classes,
                            augment=augment, debug_once=True, resize_to=resize_to)
    return DataLoader(ds, batch_size=batch, shuffle=shuffle,
                      num_workers=workers, pin_memory=pin_memory)

def make_loader_from_split(root, split, num_classes, batch=16, shuffle=True,
                           augment=False, workers=0, resize_to: Optional[Tuple[int,int]] = None) -> DataLoader:
    """
    root: 指向包含 keras_png_slices_* 与 keras_png_slices_seg_* 的目录
    """
    img_sub, mask_sub = SPLIT_TO_DIR[split]
    img_dir  = os.path.join(root, img_sub)
    mask_dir = os.path.join(root, mask_sub)
    ds = HipMRISliceDataset(img_dir, mask_dir, num_classes,
                            augment=augment, debug_once=True, resize_to=resize_to)
    return DataLoader(ds, batch_size=batch, shuffle=shuffle if split == "train" else False,
                      num_workers=workers, pin_memory=True)
