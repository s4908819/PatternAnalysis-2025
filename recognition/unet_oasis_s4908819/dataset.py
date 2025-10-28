# recognition/unet_oasis_s4908819/dataset.py
import os, glob, random
from typing import List, Tuple
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader

# ------------- utils -------------
def set_seed(seed: int = 4908819):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)

def to_one_hot(mask: np.ndarray, num_classes: int) -> np.ndarray:
    """
    mask: HxW integer labels in [0..K-1]
    return: CxHxW float32 one-hot (0/1)
    """
    h, w = mask.shape
    oh = np.zeros((num_classes, h, w), dtype=np.float32)
    # 向量化写法也可，这里保留直观循环
    for c in range(num_classes):
        oh[c] = (mask == c).astype(np.float32)
    return oh

# ------------- dataset -------------

class SlicePairDataset(Dataset):
    """Load 2D (image, mask) pairs saved as PNG (or NPY)."""
    def __init__(self, img_dir: str, mask_dir: str, num_classes: int,
                 augment: bool = False, as_tensor: bool = True, debug_once: bool = False):
        self.img_paths = sorted(glob.glob(os.path.join(img_dir, "*")))
        self.mask_paths = sorted(glob.glob(os.path.join(mask_dir, "*")))
        assert len(self.img_paths) == len(self.mask_paths) and len(self.img_paths) > 0, "pairs not found"
        self.num_classes = num_classes
        self.augment = augment
        self.as_tensor = as_tensor
        self._printed_debug = not debug_once  # 若 True 则不再打印

    def __len__(self): return len(self.img_paths)

    def _load_img(self, ip: str) -> np.ndarray:
        if ip.lower().endswith(".npy"):
            arr = np.load(ip).astype(np.float32)
        else:
            # 单通道灰度
            arr = np.array(Image.open(ip).convert("F"), dtype=np.float32)
        return arr

    def _load_mask(self, mp: str) -> np.ndarray:
        if mp.lower().endswith(".npy"):
            arr = np.load(mp).astype(np.int64)
        else:
            arr = np.array(Image.open(mp).convert("L"), dtype=np.int64)
        return arr

    def __getitem__(self, i: int):
        ip, mp = self.img_paths[i], self.mask_paths[i]

        # ---- load
        img = self._load_img(ip)     # HxW float32
        mask = self._load_mask(mp)   # HxW int64

        # ---- z-score normalize image
        m, s = img.mean(), img.std() + 1e-6
        img = (img - m) / s  # HxW float32

        # ---- simple paired augmentation (example: horizontal flip)
        if self.augment and random.random() < 0.5:
            img = np.flip(img, axis=1).copy()
            mask = np.flip(mask, axis=1).copy()

        # ---- to CHW
        img = img[None, ...]  # 1xHxW
        mask_oh = to_one_hot(mask, self.num_classes)  # CxHxW float32

        if self.as_tensor:
            img = torch.from_numpy(img).float()
            mask_oh = torch.from_numpy(mask_oh).float()

        # ---- one-time debug print
        if not self._printed_debug:
            self._printed_debug = True
            print(f"[dataset] sample0 img {tuple(img.shape)} {img.dtype} | mask {tuple(mask_oh.shape)} {mask_oh.dtype}")

        return img, mask_oh

# ------------- loader -------------

def make_split(all_ids: List[str], ratios=(0.7, 0.15, 0.15), seed=4908819) -> Tuple[List[str], List[str], List[str]]:
    set_seed(seed)
    ids = all_ids[:]
    random.shuffle(ids)
    n = len(ids); n1 = int(n*ratios[0]); n2 = int(n*ratios[1])
    return ids[:n1], ids[n1:n1+n2], ids[n1+n2:]

def make_loader(img_dir: str, mask_dir: str, num_classes: int, batch=16, shuffle=True,
                augment=False, workers=2, pin_memory=True) -> DataLoader:
    ds = SlicePairDataset(img_dir, mask_dir, num_classes, augment=augment, debug_once=True)
    return DataLoader(ds, batch_size=batch, shuffle=shuffle,
                      num_workers=workers, pin_memory=pin_memory)
