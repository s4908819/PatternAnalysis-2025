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
    Map grayscale mask values to class indices:
      - Common 4-class: {0,85,170,255} -> {0,1,2,3}
      - Common 2-class: {0,255}        -> {0,1}
    If slight grayscale noise exists, fall back to rounding to the nearest multiple of 85.
    """
    vals = set(np.unique(m).tolist())

    # Binary (0/255)
    if vals.issubset({0, 255}) or (num_classes == 2 and max(vals) > 1):
        # Exact LUT
        lut = np.zeros(256, dtype=np.uint8)
        lut[0] = 0
        lut[255] = 1
        mapped = lut[m]
        return mapped.astype(np.uint8)

    # 4-class (0/85/170/255)
    expected4 = {0, 85, 170, 255}
    if vals.issubset(expected4):
        lut = np.zeros(256, dtype=np.uint8)
        lut[0] = 0; lut[85] = 1; lut[170] = 2; lut[255] = 3
        mapped = lut[m]
        return mapped.astype(np.uint8)

    # Fallback: round to nearest multiple of 85 (4-class style)
    mapped = np.rint(m / 85.0).astype(np.int32)
    mapped = np.clip(mapped, 0, max(1, num_classes - 1)).astype(np.uint8)
    return mapped

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
        self._printed_debug = not debug_once  # If True, no further debug prints

    def __len__(self): return len(self.img_paths)

    def _load_img(self, ip: str) -> np.ndarray:
        if ip.lower().endswith(".npy"):
            arr = np.load(ip).astype(np.float32)
        else:
            # single-channel grayscale -> float32
            arr = np.array(Image.open(ip).convert("F"), dtype=np.float32)
        return arr

    def _load_mask_gray(self, mp: str) -> np.ndarray:
        """
        Load a grayscale mask and map to class indices (see map_mask_indices_gray).
        Output is HxW uint8 indices in [0..K-1].
        """
        if mp.lower().endswith(".npy"):
            raw = np.load(mp).astype(np.int64)
            # If already indices (0..K-1), return directly; otherwise attempt mapping
            if raw.max() <= max(1, self.num_classes - 1):
                return raw.astype(np.uint8)
            # Treat raw as grayscale and map
            raw = np.clip(raw, 0, 255).astype(np.uint8)
            idx = map_mask_indices_gray(raw, self.num_classes)
            return idx
        else:
            gray = np.array(Image.open(mp).convert("L"), dtype=np.uint8)
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

        # ---- simple paired augmentation (example: horizontal flip)
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

# Support keras_png_slices_* directory structure
SPLIT_TO_DIR = {
    "train":    ("keras_png_slices_train",    "keras_png_slices_seg_train"),
    "validate": ("keras_png_slices_validate", "keras_png_slices_seg_validate"),
    "val":      ("keras_png_slices_validate", "keras_png_slices_seg_validate"),
    "test":     ("keras_png_slices_test",     "keras_png_slices_seg_test"),
}

def make_loader_from_split(root, split, num_classes, batch=16, shuffle=True, augment=False, workers=0):
    img_sub, mask_sub = SPLIT_TO_DIR[split]
    img_dir  = os.path.join(root, img_sub)
    mask_dir = os.path.join(root, mask_sub)
    ds = SlicePairDataset(img_dir, mask_dir, num_classes, augment=augment, debug_once=True)
    return DataLoader(ds, batch_size=batch, shuffle=shuffle, num_workers=workers, pin_memory=True)
