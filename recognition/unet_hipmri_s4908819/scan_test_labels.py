# recognition/unet_hipmri_s4908819/scan_test_labels.py
import os, glob, numpy as np, collections as C
from PIL import Image

# 按你的截图：测试掩码目录
mask_dir = "/content/data/hipmri_slices/keras_slices_data/keras_png_slices_seg_test"

paths = sorted(glob.glob(os.path.join(mask_dir, "*")))
cnt = C.Counter()
bad = 0

def read_mask(p: str) -> np.ndarray:
    pl = p.lower()
    if pl.endswith((".nii", ".nii.gz")):
        import nibabel as nib
        arr = np.squeeze(nib.load(p).get_fdata())
        arr = np.rint(arr).astype(int)
        return arr
    else:
        arr = np.array(Image.open(p), dtype=np.uint16).astype(int)
        return arr

for p in paths[:200]:  # 抽样 200 张够用
    if os.path.isdir(p): 
        continue
    try:
        arr = read_mask(p)
        cnt.update(np.unique(arr).tolist())
    except Exception:
        bad += 1

print("Files scanned (sample):", min(200, len(paths)) - bad, ", skipped/bad:", bad)
print("Unique labels (sampled):", sorted(cnt.keys()))
print("Top-10 freq:", dict(list(cnt.most_common(10))))
