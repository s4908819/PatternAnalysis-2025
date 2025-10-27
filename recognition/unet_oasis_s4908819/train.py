# recognition/unet_oasis_s4908819/train.py
import os, argparse, time
import numpy as np
import torch
import torch.nn.functional as F
from torch.optim import AdamW
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

from dataset import make_loader
from modules import UNet2D

def dice_score(pred, target, eps=1e-6):
    # pred: NxCxHxW logits -> probs
    probs = torch.sigmoid(pred)
    probs = (probs > 0.5).float()
    intersect = (probs * target).sum(dim=(0,2,3))
    denom = probs.sum(dim=(0,2,3)) + target.sum(dim=(0,2,3)) + eps
    per_class = (2*intersect/denom)
    return per_class.mean().item(), per_class.detach().cpu().numpy()

def bce_dice_loss(logits, target, eps=1e-6):
    bce = F.binary_cross_entropy_with_logits(logits, target)
    probs = torch.sigmoid(logits)
    intersect = (probs * target).sum(dim=(0,2,3))
    denom = probs.sum(dim=(0,2,3)) + target.sum(dim=(0,2,3)) + eps
    dice = 1 - (2*intersect/denom).mean()
    return bce + dice

def plot_curves(history, outdir):
    os.makedirs(outdir, exist_ok=True)
    x = np.arange(1, len(history["train_loss"])+1)
    plt.figure(); plt.plot(x, history["train_loss"]); plt.plot(x, history["val_loss"])
    plt.xlabel("epoch"); plt.ylabel("loss"); plt.legend(["train","val"]); plt.savefig(os.path.join(outdir,"loss.png")); plt.close()
    plt.figure(); plt.plot(x, history["train_dice"]); plt.plot(x, history["val_dice"])
    plt.xlabel("epoch"); plt.ylabel("mean Dice"); plt.legend(["train","val"]); plt.savefig(os.path.join(outdir,"dice.png")); plt.close()

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_root", type=str, required=True, help="root containing images/ and masks/")
    ap.add_argument("--num_classes", type=int, default=4)
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--out", type=str, default="runs/oasis_unet")
    args = ap.parse_args()

    img_dir = os.path.join(args.data_root, "images")
    mask_dir = os.path.join(args.data_root, "masks")

    train_loader = make_loader(img_dir, mask_dir, args.num_classes, batch=args.batch, shuffle=True, augment=True,  workers=4)
    val_loader   = make_loader(img_dir, mask_dir, args.num_classes, batch=args.batch, shuffle=False, augment=False, workers=2)  # 简化：先同源校验

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = UNet2D(in_ch=1, num_classes=args.num_classes, base=32).to(device)
    opt   = AdamW(model.parameters(), lr=args.lr)

    best = -1.0
    hist = {"train_loss":[], "val_loss":[], "train_dice":[], "val_dice":[]}
    os.makedirs(args.out, exist_ok=True)

    for ep in range(1, args.epochs+1):
        model.train(); tl, td = 0.0, 0.0
        for img, mask in train_loader:
            img, mask = img.to(device), mask.to(device)
            opt.zero_grad()
            logits = model(img)
            loss = bce_dice_loss(logits, mask)
            loss.backward(); opt.step()
            tl += loss.item()*img.size(0)
            md, _ = dice_score(logits.detach(), mask)
            td += md*img.size(0)

        model.eval(); vl, vd = 0.0, 0.0
        with torch.no_grad():
            for img, mask in val_loader:
                img, mask = img.to(device), mask.to(device)
                logits = model(img)
                vl += bce_dice_loss(logits, mask).item()*img.size(0)
                md, _ = dice_score(logits, mask)
                vd += md*img.size(0)

        ntr = len(train_loader.dataset); nva = len(val_loader.dataset)
        tl/=ntr; td/=ntr; vl/=nva; vd/=nva
        hist["train_loss"].append(tl); hist["val_loss"].append(vl)
        hist["train_dice"].append(td); hist["val_dice"].append(vd)

        print(f"[epoch {ep}] loss {tl:.4f}/{vl:.4f}  dice {td:.4f}/{vd:.4f}")

        if vd > best:
            best = vd
            torch.save({"model": model.state_dict()}, os.path.join(args.out, "best.pt"))

        plot_curves(hist, args.out)

    print(f"best val mean Dice = {best:.4f}")

if __name__ == "__main__":
    main()
