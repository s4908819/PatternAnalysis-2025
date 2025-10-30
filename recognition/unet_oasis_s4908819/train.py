# recognition/unet_oasis_s4908819/train.py
import os, argparse, time
import numpy as np
import torch
import torch.nn.functional as F
from torch.optim import AdamW

# Use a headless-friendly backend for matplotlib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from dataset import make_loader_from_split
from modules import UNet  # Must match the class name in modules.py


# ---------------- Utils ----------------
def upsample_to_target(logits: torch.Tensor, target_onehot: torch.Tensor) -> torch.Tensor:
    """Resize logits to the same HxW as target via bilinear interpolation (only resize logits)."""
    if logits.shape[-2:] != target_onehot.shape[-2:]:
        logits = F.interpolate(logits, size=target_onehot.shape[-2:], mode="bilinear", align_corners=False)
    return logits


def ce_dice_loss(logits: torch.Tensor,
                 target_onehot: torch.Tensor,
                 exclude_background: bool = True,
                 eps: float = 1e-6):
    """
    CrossEntropy + (1 - mean Dice) combined loss (Dice used for stability and as a metric).
    - logits: (B, K, H, W) raw scores from model (do NOT apply softmax/sigmoid beforehand)
    - target_onehot: (B, K, H, W) float 0/1 (from dataset)
    Returns: loss, ce_val(detached), dice_val(detached)
    """
    B, K, H, W = logits.shape
    with torch.no_grad():
        target_idx = target_onehot.argmax(dim=1).long()  # (B, H, W)

    # CE: raw logits + integer indices
    ce = F.cross_entropy(logits, target_idx)

    # Dice: use softmax probabilities
    probs = logits.softmax(dim=1)  # (B, K, H, W)

    cls_range = range(1, K) if (exclude_background and K > 1) else range(K)
    dices = []
    for c in cls_range:
        p = probs[:, c]                  # (B, H, W)
        t = target_onehot[:, c]          # (B, H, W)
        inter = (p * t).sum(dim=(1, 2))  # (B,)
        denom = p.sum(dim=(1, 2)) + t.sum(dim=(1, 2)) + eps
        dice_c = (2.0 * inter + eps) / denom
        dices.append(dice_c)

    dice = torch.stack(dices).mean() if dices else torch.tensor(1.0, device=logits.device)

    loss = ce + (1.0 - dice)
    return loss, ce.detach(), dice.detach()


def dice_metric(logits: torch.Tensor,
                target_onehot: torch.Tensor,
                exclude_background: bool = True,
                eps: float = 1e-6):
    """
    Dice used only for logging/printing (same definition as above).
    Returns: mean_dice_scalar(float), per_class_numpy(np.ndarray)
    """
    B, K, H, W = logits.shape
    probs = logits.softmax(dim=1)
    cls_range = range(1, K) if (exclude_background and K > 1) else range(K)

    per_class = []
    for c in cls_range:
        p = probs[:, c]
        t = target_onehot[:, c]
        inter = (p * t).sum(dim=(1, 2))
        denom = p.sum(dim=(1, 2)) + t.sum(dim=(1, 2)) + eps
        dice_c = (2.0 * inter + eps) / denom  # (B,)
        per_class.append(dice_c)

    if len(per_class) == 0:
        mean_dice = 1.0
        per_cls_np = np.array([1.0], dtype=np.float32)
    else:
        per_cls_stack = torch.stack(per_class).mean(dim=1)  # (B,)
        mean_dice = per_cls_stack.mean().item()
        per_cls_np = torch.stack(per_class).mean(dim=0).detach().cpu().numpy()  # (B,)->batch mean if needed
    return mean_dice, per_cls_np


def plot_curves(history, outdir):
    os.makedirs(outdir, exist_ok=True)
    x = np.arange(1, len(history["train_loss"]) + 1)

    plt.figure()
    plt.plot(x, history["train_loss"])
    plt.plot(x, history["val_loss"])
    plt.xlabel("epoch")
    plt.ylabel("loss")
    plt.legend(["train", "val"])
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, "loss.png"))
    plt.close()

    plt.figure()
    plt.plot(x, history["train_dice"])
    plt.plot(x, history["val_dice"])
    plt.xlabel("epoch")
    plt.ylabel("mean Dice (no-bg)")
    plt.legend(["train", "val"])
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, "dice.png"))
    plt.close()


# ---------------- Main ----------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_root", type=str, required=True, help="root containing keras_png_slices_* directories")
    ap.add_argument("--num_classes", type=int, default=4)
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--out", type=str, default="runs/oasis_unet")
    ap.add_argument("--workers", type=int, default=None, help="override DataLoader workers; default auto")
    args = ap.parse_args()

    has_cuda = torch.cuda.is_available()
    auto_workers = (4 if has_cuda else 0) if args.workers is None else args.workers

    # DataLoaders (consistent with dataset.py)
    train_loader = make_loader_from_split(
        root=args.data_root, split="train", num_classes=args.num_classes,
        batch=args.batch, shuffle=True, augment=True, workers=auto_workers
    )
    val_loader = make_loader_from_split(
        root=args.data_root, split="validate", num_classes=args.num_classes,
        batch=args.batch, shuffle=False, augment=False, workers=auto_workers
    )

    device = torch.device("cuda" if has_cuda else "cpu")
    torch.backends.cudnn.benchmark = has_cuda

    # Model & Optimizer
    model = UNet(in_ch=1, n_classes=args.num_classes, base=32).to(device)
    opt = AdamW(model.parameters(), lr=args.lr, weight_decay=1e-5)

    best = -1.0
    hist = {"train_loss": [], "val_loss": [], "train_dice": [], "val_dice": []}
    os.makedirs(args.out, exist_ok=True)

    for ep in range(1, args.epochs + 1):
        # ---------------- Train ----------------
        model.train()
        tl, td = 0.0, 0.0
        ntr = 0
        for img, mask in train_loader:
            img, mask = img.to(device), mask.to(device)  # mask: one-hot (B,K,H,W)
            bs = img.size(0)
            ntr += bs

            opt.zero_grad(set_to_none=True)
            logits = model(img)                         # (B,K,H,W)
            logits = upsample_to_target(logits, mask)   # align spatial size

            loss, ce_val, dice_val = ce_dice_loss(logits, mask, exclude_background=True)
            loss.backward()
            opt.step()

            tl += loss.item() * bs
            td += dice_val.item() * bs

        tl /= max(1, ntr)
        td /= max(1, ntr)

        # ---------------- Validate ----------------
        model.eval()
        vl, vd = 0.0, 0.0
        nva = 0
        with torch.no_grad():
            for img, mask in val_loader:
                img, mask = img.to(device), mask.to(device)
                bs = img.size(0)
                nva += bs

                logits = model(img)
                logits = upsample_to_target(logits, mask)

                vloss, vce, vdice = ce_dice_loss(logits, mask, exclude_background=True)
                vl += vloss.item() * bs
                vd += vdice.item() * bs

        vl /= max(1, nva)
        vd /= max(1, nva)

        hist["train_loss"].append(tl)
        hist["val_loss"].append(vl)
        hist["train_dice"].append(td)
        hist["val_dice"].append(vd)

        print(f"[epoch {ep:03d}] loss {tl:.4f}/{vl:.4f}  dice(no-bg) {td:.4f}/{vd:.4f}")

        # Save best by val dice
        if vd > best:
            best = vd
            torch.save({"model": model.state_dict()}, os.path.join(args.out, "best.pt"))

        # Update curves each epoch
        plot_curves(hist, args.out)

    print(f"best val mean Dice (no-bg) = {best:.4f}")


if __name__ == "__main__":
    main()