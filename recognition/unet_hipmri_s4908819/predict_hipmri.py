# recognition/unet_hipmri_s4908819/predict_hipmri.py
import os, argparse
import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F

# 使用残差版 U-Net（与训练保持一致）
from modules_resunet import UNetRes  # in_ch/n_classes/base 需与训练一致


def upsample_to_hw(logits: torch.Tensor, hw: tuple[int, int]) -> torch.Tensor:
    """Resize logits[N,C,h,w] to (H,W) via bilinear interpolation (only resize logits)."""
    if logits.shape[-2:] != hw:
        logits = F.interpolate(logits, size=hw, mode="bilinear", align_corners=False)
    return logits


def overlay_mask(img: np.ndarray, pred: np.ndarray, num_classes: int = 4) -> Image.Image:
    """
    img: HxW float (z-scored OK), pred: HxW integer labels [0..K-1]
    Return RGB overlay for quick visual check.
    """
    # 灰度归一化到 [0,1]
    gmin, gmax = float(img.min()), float(img.max())
    img_norm = (img - gmin) / (gmax - gmin + 1e-6)
    rgb = np.stack([img_norm] * 3, axis=-1)

    # 颜色表：背景0不染色，其它类给出区分色；可自行扩展/调整
    palette = [
        (0.0, 0.0, 0.0),   # class 0: background/no tint
        (1.0, 0.0, 0.0),   # class 1: red
        (0.0, 1.0, 0.0),   # class 2: green
        (0.0, 0.0, 1.0),   # class 3: blue
        (1.0, 1.0, 0.0),   # class 4: yellow
        (1.0, 0.0, 1.0),   # class 5: magenta
        (0.0, 1.0, 1.0),   # class 6: cyan
    ]
    while len(palette) < num_classes:
        palette.append((np.random.rand(), np.random.rand(), np.random.rand()))

    # 叠加：保持 60% 原图 + 40% 色彩
    for k in range(1, num_classes):
        tint = np.array(palette[k])
        mask_k = (pred == k)
        if mask_k.any():
            rgb[mask_k] = 0.6 * rgb[mask_k] + 0.4 * tint

    rgb = (np.clip(rgb, 0, 1) * 255).astype(np.uint8)
    return Image.fromarray(rgb)


def load_img(path: str) -> np.ndarray:
    if path.lower().endswith(".npy"):
        arr = np.load(path).astype(np.float32)
    else:
        arr = np.array(Image.open(path).convert("F"), dtype=np.float32)
    return arr


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", required=True, help="path to checkpoint saved by train_hipmri.py")
    ap.add_argument("--img", required=True, help="path to a single PNG/NPY slice")
    ap.add_argument("--num_classes", type=int, default=4)
    ap.add_argument("--out", default="pred_hipmri.png")
    ap.add_argument("--base", type=int, default=32, help="base channels used in training UNetRes")
    ap.add_argument("--device", type=str, default="cuda", help="cuda or cpu")
    args = ap.parse_args()

    # device
    dev = torch.device(args.device if torch.cuda.is_available() else "cpu")

    # load and normalize image
    img = load_img(args.img)                        # HxW float32
    H, W = img.shape
    img_z = (img - img.mean()) / (img.std() + 1e-6)
    x = torch.from_numpy(img_z[None, None, ...]).float().to(dev)  # 1x1xHxW

    # build model (keep same hyper-params as training)
    model = UNetRes(in_ch=1, n_classes=args.num_classes, base=args.base).to(dev)
    ckpt = torch.load(args.weights, map_location=dev)
    # 兼容 {'model': state_dict} 或直接 state_dict
    state_dict = ckpt.get("model", ckpt)
    model.load_state_dict(state_dict, strict=True)
    model.eval()

    with torch.no_grad():
        logits = model(x)                 # [1,C,h',w']
        logits = upsample_to_hw(logits, (H, W))
        probs = torch.sigmoid(logits)[0].cpu().numpy()      # CxHxW
        # 对互斥语义分割：argmax；如做多标签可改成逐类阈值
        pred = probs.argmax(axis=0).astype(np.int64)        # HxW

    overlay = overlay_mask(img, pred, num_classes=args.num_classes)
    overlay.save(args.out)
    print(f"saved -> {args.out}")


if __name__ == "__main__":
    main()
