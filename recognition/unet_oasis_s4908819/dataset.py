
# recognition/unet_oasis_s4908819/dataset.py
import os, glob, random
from typing import Tuple, List, Optional
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader

def set_seed(seed: int = 4908819):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)

def to_one_hot(mask: np.ndarray, num_classes: int) -> np.ndarray:
    # mask: HxW integer labels in [0..K-1]
    h, w = mask.shape
    oh = np.zeros((num_classes, h, w), dtype=np.float32)
    for c in range(num_classes):
        oh[c] = (mask == c).astype(np.float32)
    return oh

class SlicePairDataset(Dataset):
    """Load 2D (image, mask) pairs saved as PNG (or NPY)."""
    def __init__(self, img_dir: str, mask_dir: str, num_classes: int, 
                 augment: bool = False, as_tensor: bool = True):
        self.img_paths = sorted(glob.glob(os.path.join(img_dir, "*")))
        self.mask_paths = sorted(glob.glob(os.path.join(mask_dir, "*")))
        assert len(self.img_paths) == len(self.mask_paths) and len(self.img_paths) > 0, "pairs not found"
        self.num_classes = num_classes
        self.augment = augment
        self.as_tensor = as_tensor

    def __len__(self): return len(self.img_paths)

    def __getitem__(self, i):
        ip, mp = self.img_paths[i], self.mask_paths[i]
        # ---- load
        if ip.endswith(".npy"):
            img = np.load(ip).astype(np.float32)
        else:
            img = np.array(Image.open(ip).convert("F"), dtype=np.float32)  # single-channel
        if mp.endswith(".npy"):
            mask = np.load(mp).astype(np.int64)
        else:
            mask = np.array(Image.open(mp).convert("L"), dtype=np.int64)

        # ---- normalize image (z-score)
        m, s = img.mean(), img.std() + 1e-6
        img = (img - m) / s  # HxW

        # ---- simple aug
        if self.augment and random.random() < 0.5:
            img = np.flip(img, axis=1).copy()
            mask = np.flip(mask, axis=1).copy()

        # ---- to CHW
        img = img[None, ...]  # 1xHxW
        mask_oh = to_one_hot(mask, self.num_classes)  # CxHxW

        if self.as_tensor:
            img = torch.from_numpy(img)
            mask_oh = torch.from_numpy(mask_oh)
        return img, mask_oh

def make_split(all_ids: List[str], ratios=(0.7, 0.15, 0.15), seed=4908819):
    set_seed(seed)
    ids = all_ids[:]
    random.shuffle(ids)
    n = len(ids); n1 = int(n*ratios[0]); n2 = int(n*ratios[1])
    return ids[:n1], ids[n1:n1+n2], ids[n1+n2:]

def make_loader(img_dir, mask_dir, num_classes, batch=16, shuffle=True, augment=False, workers=2):
    ds = SlicePairDataset(img_dir, mask_dir, num_classes, augment=augment)
    return DataLoader(ds, batch_size=batch, shuffle=shuffle, num_workers=workers, pin_memory=True)

# --- If you prefer reading Nifti directly, adapt this (requires nibabel):
# import nibabel as nib
# def load_nifti_2d(nifti_path: str) -> np.ndarray:
#     arr = nib.load(nifti_path).get_fdata(caching='unchanged')
#     if arr.ndim == 3: arr = arr[:, :, 0]         # drop extra dim if present
#     return arr.astype(np.float32)
