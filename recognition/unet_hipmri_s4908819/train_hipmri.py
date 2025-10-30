# recognition/unet_hipmri_s4908819/train_hipmri.py
import os, argparse
import numpy as np
import torch
import torch.nn.functional as F
from torch.optim import AdamW

# headless 绘图
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# === 使用 HipMRI 数据加载器与 ResUNet ===
from dataset_hipmri import make_loader_from_split
from modules_resunet import UNetRes  # 保持与训练一致的模型


# ---------------- Utils ----------------
def upsample_to_target(logits: torch.Tensor, target_onehot: torch.Tensor) -> torch.Tensor:
    """将 logits 双线性插值到与 target 相同的 HxW（只插值 logits）。"""
    if logits.shape[-2:] != target_onehot.shape[-2:]:
        logits = F.interpolate(logits, size=target_onehot.shape[-2:], mode="bilinear", align_corners=False)
    return logits


def ce_dice_loss(
    logits: torch.Tensor,
    target_onehot: torch.Tensor,
    exclude_background: bool = True,
    eps: float = 1e-6
):
    """
    CrossEntropy + (1 - mean Dice) 组合损失。
    - logits: (B, K, H, W) 原始分数（未 softmax）
    - target_onehot: (B, K, H, W) 浮点 one-hot
    返回: loss, ce_val(detached), dice_val(detached)
    """
    B, K, H, W = logits.shape
    with torch.no_grad():
        target_idx = target_onehot.argmax(dim=1).long()  # (B,H,W)

    ce = F.cross_entropy(logits, target_idx)

    probs = logits.softmax(dim=1)  # (B,K,H,W)
    cls_range = range(1, K) if (exclude_background and K > 1) else range(K)

    dices = []
    for c in cls_range:
        p = probs[:, c]                 # (B,H,W)
        t = target_onehot[:, c]         # (B,H,W)
        inter = (p * t).sum(dim=(1, 2))
        denom = p.sum(dim=(1, 2)) + t.sum(dim=(1, 2)) + eps
        dice_c = (2.0 * inter + eps) / denom
        dices.append(dice_c)

    dice = torch.stack(dices).mean() if dices else torch.tensor(1.0, device=logits.device)
    loss = ce + (1.0 - dice)
    return loss, ce.detach(), dice.detach()


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
    ap.add_argument("--data_root", type=str, required=True,
                    help="HipMRI 根目录（包含 keras_slices_data 或其上级目录）")
    ap.add_argument("--num_classes", type=int, default=4)
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--out", type=str, default="runs/hipmri_resunet")
    ap.add_argument("--workers", type=int, default=None, help="DataLoader workers（默认自动）")
    ap.add_argument("--base", type=int, default=32, help="ResUNet 的 base 通道数（16/32）")
    ap.add_argument("--resize", type=int, default=256,
                    help="可选：统一缩放到 (resize, resize)；传 0 表示不缩放")
    # 新增：二分类开关（前景 vs 背景）
    ap.add_argument("--binary", type=int, default=0,
                    help="是否启用二分类模式（前景=非0），1 启用 / 0 关闭")
    args = ap.parse_args()

    has_cuda = torch.cuda.is_available()
    auto_workers = (4 if has_cuda else 0) if args.workers is None else args.workers
    resize_to = None if args.resize in (0, None) else (args.resize, args.resize)

    print(f"[config] data_root={args.data_root}  num_classes={args.num_classes}  binary={args.binary}")
    print(f"[config] epochs={args.epochs}  batch={args.batch}  lr={args.lr}  base={args.base}  resize={args.resize}")
    print(f"[config] workers={auto_workers}  device={'cuda' if has_cuda else 'cpu'}")

    # DataLoaders（与 dataset_hipmri.py 一致）
    train_loader = make_loader_from_split(
        root=args.data_root, split="train", num_classes=args.num_classes,
        batch=args.batch, shuffle=True, augment=True, workers=auto_workers,
        resize_to=resize_to, binary=bool(args.binary)
    )
    val_loader = make_loader_from_split(
        root=args.data_root, split="validate", num_classes=args.num_classes,
        batch=args.batch, shuffle=False, augment=False, workers=auto_workers,
        resize_to=resize_to, binary=bool(args.binary)
    )

    device = torch.device("cuda" if has_cuda else "cpu")
    torch.backends.cudnn.benchmark = has_cuda

    # Model & Optimizer（使用残差 UNet）
    model = UNetRes(in_ch=1, n_classes=args.num_classes, base=args.base).to(device)
    opt = AdamW(model.parameters(), lr=args.lr, weight_decay=1e-5)

    best = -1.0
    hist = {"train_loss": [], "val_loss": [], "train_dice": [], "val_dice": []}
    os.makedirs(args.out, exist_ok=True)

    for ep in range(1, args.epochs + 1):
        # ---------------- Train ----------------
        model.train()
        tl, td = 0.0, 0.0
        ntr = 0
        for img, mask in train_loader:            # mask: one-hot (B,K,H,W)
            img, mask = img.to(device), mask.to(device)
            bs = img.size(0)
            ntr += bs

            opt.zero_grad(set_to_none=True)
            logits = model(img)                   # (B,K,H,W)
            logits = upsample_to_target(logits, mask)

            # 二分类与多分类均以 no-bg Dice 作为优化/监控指标（K=2 时即前景通道）
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

        # 以验证 Dice 作为 best
        if vd > best:
            best = vd
            torch.save(
                {
                    "model": model.state_dict(),
                    "epoch": ep,
                    "val_mean_dice_no_bg": best,
                    "num_classes": args.num_classes,
                    "base": args.base,
                    "resize": args.resize,
                    "binary": int(args.binary),
                },
                os.path.join(args.out, "best.pt")
            )

        # 每个 epoch 更新曲线
        plot_curves(hist, args.out)

    print(f"best val mean Dice (no-bg) = {best:.4f}")


if __name__ == "__main__":
    main()
