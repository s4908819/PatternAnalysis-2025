# recognition/unet_oasis_s4908819/predict.py
import os, argparse
import numpy as np
from PIL import Image
import torch
from modules import UNet2D

def overlay_mask(img: np.ndarray, pred: np.ndarray) -> Image.Image:
    """img: HxW float, pred: HxW integer labels (argmax); return RGB overlay"""
    img_norm = (img - img.min()) / (img.max() - img.min() + 1e-6)
    rgb = np.stack([img_norm]*3, axis=-1)
    # simple palette: class 1 red, 2 green, 3 blue (extend as needed)
    colors = {1:(1,0,0), 2:(0,1,0), 3:(0,0,1)}
    for k, c in colors.items():
        rgb[pred==k] = 0.6*rgb[pred==k] + 0.4*np.array(c)
    rgb = (rgb*255).astype(np.uint8)
    return Image.fromarray(rgb)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", required=True)
    ap.add_argument("--img", required=True)        # path to a single PNG/NPY slice
    ap.add_argument("--num_classes", type=int, default=4)
    ap.add_argument("--out", default="pred.png")
    args = ap.parse_args()

    # load image
    if args.img.endswith(".npy"): img = np.load(args.img).astype(np.float32)
    else: img = np.array(Image.open(args.img).convert("F"), dtype=np.float32)
    # z-score
    img = (img - img.mean())/(img.std()+1e-6)
    x = torch.from_numpy(img[None, None, ...]).float()  # 1x1xHxW

    model = UNet2D(in_ch=1, num_classes=args.num_classes)
    state = torch.load(args.weights, map_location="cpu")
    model.load_state_dict(state["model"]); model.eval()

    with torch.no_grad():
        logits = model(x)
        probs = torch.sigmoid(logits)[0].numpy()     # CxHxW
        pred = probs.argmax(axis=0).astype(np.int64) # HxW

    overlay = overlay_mask(img, pred)
    overlay.save(args.out)
    print(f"saved -> {args.out}")

if __name__ == "__main__":
    main()
