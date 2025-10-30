# recognition/unet_hipmri_s4908819/dataset_hipmri.py
"""
HipMRI 2D slice dataset (adapted per the “Reading Nifti Files” spec):
- Images: supports PNG / JPG / TIF / NPY / NIfTI (.nii/.nii.gz)
- Masks:  supports PNG / JPG / TIF / NPY / NIfTI; automatically map grayscale or integer labels to class indices
- Output:
  • Binary (binary=True): mask -> (2,H,W) float32 (two-hot, channel order = [background, prostate], prostate label = prostate_label, default 2)
  • Multiclass (binary=False): mask -> (K,H,W) float32 (one-hot)
- Resize: images use BILINEAR, masks use NEAREST (to avoid class interpolation)
- Pairing: pairs by normalized basename (strip case_/seg_ prefixes) to avoid sort mismatches
- Compatible with existing train/eval scripts: keep make_loader / make_loader_from_split
"""

import os, glob, random
from typing import Tuple, Optional, Dict, List
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
import nibabel as nib  # NIfTI support

# -------------------- utils --------------------

def set_seed(seed: int = 4908819):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def to_one_hot(mask_idx: np.ndarray, num_classes: int, dtype=np.float32) -> np.ndarray:
    """
    mask_idx: HxW integer labels (0..K-1)
    return: KxHxW one-hot
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
    Map grayscale mask values to class indices (handles common {0,255} / {0,85,170,255}).
    If there is slight grayscale noise, fall back to rounding to multiples of 85.
    """
    vals = set(np.unique(gray).tolist())

    # Binary (0/255)
    if vals.issubset({0, 255}) or (num_classes == 2 and max(vals, default=0) > 1):
        lut = np.zeros(256, dtype=np.uint8); lut[0] = 0; lut[255] = 1
        return lut[_clip_uint8(gray)]

    # 4-class (0/85/170/255)
    expected4 = {0, 85, 170, 255}
    if vals.issubset(expected4):
        lut = np.zeros(256, dtype=np.uint8); lut[0]=0; lut[85]=1; lut[170]=2; lut[255]=3
        return lut[_clip_uint8(gray)]

    # Fallback: round to nearest multiple of 85 (typical 4-class layout)
    mapped = np.rint(gray / 85.0).astype(np.int32)
    mapped = np.clip(mapped, 0, max(1, num_classes - 1)).astype(np.uint8)
    return mapped

# -------------------- NIfTI reading (following the standard example) --------------------

def _nifti_get_2d_array(path: str, dtype=np.float32) -> np.ndarray:
    """
    Read NIfTI and return a 2D array:
    - If 3D -> take [:,:,0]
    - If 4D -> take [:,:,:,0]
    - If extra singleton dims exist -> squeeze
    """
    img = nib.load(path)
    arr = img.get_fdata(caching="unchanged")  # read directly from disk
    arr = np.asarray(arr)
    arr = np.squeeze(arr)

    if arr.ndim == 2:
        return arr.astype(dtype)
    if arr.ndim == 3:
        return arr[:, :, 0].astype(dtype)
    if arr.ndim >= 4:
        return arr[:, :, 0, 0].astype(dtype)
    return np.squeeze(arr).astype(dtype)

# -------------------- Path resolution: support multiple directory layouts --------------------

PLAIN_SPLIT = {
    "train":    ("keras_png_slices_train",    "keras_png_slices_seg_train"),
    "validate": ("keras_png_slices_validate", "keras_png_slices_seg_validate"),
    "val":      ("keras_png_slices_validate", "keras_png_slices_seg_validate"),
    "test":     ("keras_png_slices_test",     "keras_png_slices_seg_test"),
}
WRAPPED_SPLIT = {
    k: (os.path.join("keras_slices_data", v[0]), os.path.join("keras_slices_data", v[1]))
    for k, v in PLAIN_SPLIT.items()
}
PLAIN_FALLBACK = {
    "train":    ("keras_slices_train",    "keras_slices_seg_train"),
    "validate": ("keras_slices_validate", "keras_slices_seg_validate"),
    "val":      ("keras_slices_validate", "keras_slices_seg_validate"),
    "test":     ("keras_slices_test",     "keras_slices_seg_test"),
}
WRAPPED_FALLBACK = {
    k: (os.path.join("keras_slices_data", v[0]), os.path.join("keras_slices_data", v[1]))
    for k, v in PLAIN_FALLBACK.items()
}

def _resolve_split_dirs(root: str, split: str) -> Tuple[str, str]:
    """
    Try the following layouts in order (favoring the PNG version):
    1) root/keras_slices_data/keras_png_slices_* / *_seg_*
    2) root/keras_png_slices_* / *_seg_*
    3) root/keras_slices_data/keras_slices_* / *_seg_*
    4) root/keras_slices_* / *_seg_*
    """
    tries: List[Tuple[str, str]] = []
    for table in (WRAPPED_SPLIT, PLAIN_SPLIT, WRAPPED_FALLBACK, PLAIN_FALLBACK):
        img_sub, msk_sub = table[split]
        tries.append((os.path.join(root, img_sub), os.path.join(root, msk_sub)))

    for img_dir, msk_dir in tries:
        if os.path.isdir(img_dir) and os.path.isdir(msk_dir):
            return img_dir, msk_dir

    msg = " | ".join([f"{a} / {b}" for a, b in tries])
    raise FileNotFoundError(f"[dataset] Cannot resolve split='{split}' under root={root}. Tried: {msg}")

# -------------------- Dataset --------------------

def _norm_name(bn: str) -> str:
    # Strip case_/seg_ prefixes for consistent pairing
    return os.path.basename(bn).replace("case_", "").replace("seg_", "")

class HipMRISliceDataset(Dataset):
    """
    Load HipMRI/OASIS-style 2D slice pairs (image/mask).
    - Automatically detects PNG/JPG/TIF/NPY/NIfTI
    - Image z-score normalization
    - Mask:
        • binary=True -> (raw == prostate_label) → indices 0/1 → return two-hot (2,H,W), [bg, fg]
        • binary=False -> multiclass indices -> one-hot (K,H,W)
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
        binary: bool = False,
        label_map: Optional[Dict[int, int]] = None,
        prostate_label: int = 2,   # New: prostate label id for binary mode (default 2)
    ):
        # — Pair by normalized names to avoid sort-based mismatches —
        img_all = [p for p in glob.glob(os.path.join(img_dir, "*")) if not os.path.isdir(p)]
        msk_all = [p for p in glob.glob(os.path.join(mask_dir, "*")) if not os.path.isdir(p)]
        img_dict = {_norm_name(p): p for p in img_all}
        msk_dict = {_norm_name(p): p for p in msk_all}
        names = sorted(list(set(img_dict.keys()) & set(msk_dict.keys())))
        assert len(names) > 0, f"no paired files in: {img_dir} | {mask_dir}"
        self.img_paths = [img_dict[n] for n in names]
        self.mask_paths = [msk_dict[n] for n in names]

        self.num_classes = num_classes
        self.augment = augment
        self.as_tensor = as_tensor
        self.resize_to = resize_to
        self.binary = binary
        self.label_map = label_map
        self.prostate_label = int(prostate_label)
        if self.binary and self.num_classes != 2:
            raise ValueError("When binary=True, please set num_classes=2")

        self._printed_debug = not debug_once

    def __len__(self):
        return len(self.img_paths)

    # --- Resize (images: BILINEAR; masks: NEAREST) ---
    def _resize_np(self, arr: np.ndarray, is_mask: bool) -> np.ndarray:
        if self.resize_to is None:
            return arr
        H, W = self.resize_to
        if is_mask:
            pil = Image.fromarray(arr.astype(np.uint8))
            pil = pil.resize((W, H), resample=Image.NEAREST)
            return np.array(pil, dtype=np.uint8)
        else:
            # PIL 'F' mode supports float32
            pil = Image.fromarray(arr.astype(np.float32), mode="F")
            pil = pil.resize((W, H), resample=Image.BILINEAR)
            return np.array(pil, dtype=np.float32)

    # --- Load image ---
    def _load_image(self, p: str) -> np.ndarray:
        p_lower = p.lower()
        if p_lower.endswith(".npy"):
            arr = np.load(p).astype(np.float32)
        elif p_lower.endswith(".nii") or p_lower.endswith(".nii.gz"):
            arr = _nifti_get_2d_array(p, dtype=np.float32)
        else:
            pil = Image.open(p).convert("F")
            arr = np.array(pil, dtype=np.float32)
        if self.resize_to is not None:
            arr = self._resize_np(arr, is_mask=False)
        return arr

    # --- Load mask (raw) ---
    def _load_mask_raw(self, p: str) -> np.ndarray:
        p_lower = p.lower()
        if p_lower.endswith(".npy"):
            raw = np.load(p)
            raw = np.squeeze(raw)
            return raw
        elif p_lower.endswith(".nii") or p_lower.endswith(".nii.gz"):
            raw = _nifti_get_2d_array(p, dtype=np.float32)
            return raw
        else:
            # Read as grayscale (no normalization)
            pil = Image.open(p).convert("L")
            raw = np.array(pil, dtype=np.uint16)
            return raw

    def _mask_to_index(self, raw: np.ndarray) -> np.ndarray:
        """
        Convert any raw mask (grayscale/float/integer) to dense class indices (0..K-1).
        Priority:
        1) binary -> (raw == prostate_label).astype(uint8)
        2) explicit label_map -> map integer-rounded raw via the dict to indices
        3) auto: if integer and within [0..K-1], use directly; otherwise use map_mask_indices_gray
        """
        # Binary: keep “prostate=prostate_label” as foreground
        if self.binary:
            rawi = np.rint(raw).astype(np.int32)
            return (rawi == self.prostate_label).astype(np.uint8)

        if self.label_map is not None:
            rawi = np.rint(raw).astype(np.int32)
            idx = np.zeros_like(rawi, dtype=np.uint8)
            for src, dst in self.label_map.items():
                idx[rawi == int(src)] = int(dst)
            return idx

        # Auto
        if np.issubdtype(raw.dtype, np.integer):
            rawi = raw.astype(np.int32)
        else:
            rawi = np.rint(raw).astype(np.int32)

        if rawi.min() >= 0 and rawi.max() <= max(1, self.num_classes - 1):
            idx = rawi.astype(np.uint8)
        else:
            idx = map_mask_indices_gray(_clip_uint8(rawi), self.num_classes)
        return idx

    def __getitem__(self, i: int):
        ip, mp = self.img_paths[i], self.mask_paths[i]

        # ---- load
        img = self._load_image(ip)            # HxW float32
        raw = self._load_mask_raw(mp)         # HxW (float/int/gray)

        # ---- resize (resize before indexing/normalization)
        if self.resize_to is not None:
            img = self._resize_np(img, is_mask=False)
            raw = self._resize_np(raw, is_mask=True)

        # ---- mask -> indices / one-hot (binary=True returns two-hot)
        idx = self._mask_to_index(raw)        # HxW uint8 (0/1 or 0..K-1)

        # ---- z-score normalize
        mean = img.mean()
        std = img.std() + 1e-6
        img = (img - mean) / std

        # ---- simple augmentation (horizontal flip)
        if self.augment and random.random() < 0.5:
            img = np.flip(img, axis=1).copy()
            idx = np.flip(idx, axis=1).copy()

        # ---- CHW
        img = img[None, ...].astype(np.float32)  # 1xHxW

        if self.binary:
            # Binary output two-hot [bg, fg] (2,H,W) — compatible with CE/Dice multiclass style losses
            # idx values are 0/1: 0=background, 1=foreground (prostate)
            mask = to_one_hot(idx, 2, dtype=np.float32)       # 2xHxW
        else:
            # Multiclass one-hot (KxHxW)
            mask = to_one_hot(idx, self.num_classes, dtype=np.float32)

        if self.as_tensor:
            img = torch.from_numpy(img).float()
            mask = torch.from_numpy(mask).float()

        if not self._printed_debug:
            self._printed_debug = True
            print(f"[hipmri dataset] sample0 img {tuple(img.shape)} {img.dtype} | "
                  f"mask {tuple(mask.shape)} {mask.dtype} | binary={self.binary} plabel={self.prostate_label}")

        return img, mask

# -------------------- Dataloader APIs --------------------

def make_loader(img_dir: str, mask_dir: str, num_classes: int, batch=16, shuffle=True,
                augment=False, workers=2, pin_memory=True,
                resize_to: Optional[Tuple[int, int]] = None,
                binary: bool = False, label_map: Optional[Dict[int, int]] = None,
                prostate_label: int = 2) -> DataLoader:
    ds = HipMRISliceDataset(
        img_dir, mask_dir, num_classes,
        augment=augment, debug_once=True, resize_to=resize_to,
        binary=binary, label_map=label_map, prostate_label=prostate_label
    )
    return DataLoader(ds, batch_size=batch, shuffle=shuffle,
                      num_workers=workers, pin_memory=pin_memory)

def make_loader_from_split(root: str, split: str, num_classes: int, batch=16, shuffle=True,
                           augment=False, workers=0, pin_memory=True,
                           resize_to: Optional[Tuple[int, int]] = None,
                           binary: bool = False, label_map: Optional[Dict[int, int]] = None,
                           prostate_label: int = 2) -> DataLoader:
    """
    root: points to a directory that may contain keras_slices_data/keras_png_slices_* and *_seg_*.
    Automatically searches the following four layouts:
      1) root/keras_slices_data/keras_png_slices_*
      2) root/keras_png_slices_*
      3) root/keras_slices_data/keras_slices_*
      4) root/keras_slices_*
    """
    img_dir, mask_dir = _resolve_split_dirs(root, split)
    ds = HipMRISliceDataset(
        img_dir, mask_dir, num_classes,
        augment=augment, debug_once=True, resize_to=resize_to,
        binary=binary, label_map=label_map, prostate_label=prostate_label
    )
    return DataLoader(ds, batch_size=batch,
                      shuffle=(shuffle if split == "train" else False),
                      num_workers=workers, pin_memory=pin_memory)
