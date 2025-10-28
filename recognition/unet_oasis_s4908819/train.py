# # recognition/unet_oasis_s4908819/train.py
# import os, argparse, time
# import numpy as np
# import torch
# import torch.nn.functional as F
# from torch.optim import AdamW
# from torch.utils.data import DataLoader

# # Use a headless-friendly backend for matplotlib
# import matplotlib
# matplotlib.use("Agg")
# import matplotlib.pyplot as plt

# from dataset import make_loader
# from modules import UNet  # 注意：与 modules.py 中的类名一致


# def upsample_to_target(logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
#     """Resize logits to the same HxW as target via bilinear interpolation (only resize logits)."""
#     if logits.shape[-2:] != target.shape[-2:]:
#         logits = F.interpolate(logits, size=target.shape[-2:], mode="bilinear", align_corners=False)
#     return logits


# def dice_score(pred, target, eps=1e-6):
#     """
#     pred: logits, shape [N, C, H, W]
#     target: one-hot float mask, shape [N, C, H, W]
#     返回 (mean_dice_scalar, per_class_numpy)
#     """
#     probs = torch.sigmoid(pred)
#     probs = (probs > 0.5).float()
#     intersect = (probs * target).sum(dim=(0, 2, 3))
#     denom = probs.sum(dim=(0, 2, 3)) + target.sum(dim=(0, 2, 3)) + eps
#     per_class = (2 * intersect / denom)
#     return per_class.mean().item(), per_class.detach().cpu().numpy()


# def bce_dice_loss(logits, target, eps=1e-6):
#     """
#     多类别 one-hot 情况下的 BCE + Dice 组合损失。
#     target 需为 [N, C, H, W]、float、取值 {0,1}。
#     """
#     bce = F.binary_cross_entropy_with_logits(logits, target)
#     probs = torch.sigmoid(logits)
#     intersect = (probs * target).sum(dim=(0, 2, 3))
#     denom = probs.sum(dim=(0, 2, 3)) + target.sum(dim=(0, 2, 3)) + eps
#     dice = 1 - (2 * intersect / denom).mean()
#     return bce + dice


# def plot_curves(history, outdir):
#     os.makedirs(outdir, exist_ok=True)
#     x = np.arange(1, len(history["train_loss"]) + 1)

#     plt.figure()
#     plt.plot(x, history["train_loss"])
#     plt.plot(x, history["val_loss"])
#     plt.xlabel("epoch")
#     plt.ylabel("loss")
#     plt.legend(["train", "val"])
#     plt.tight_layout()
#     plt.savefig(os.path.join(outdir, "loss.png"))
#     plt.close()

#     plt.figure()
#     plt.plot(x, history["train_dice"])
#     plt.plot(x, history["val_dice"])
#     plt.xlabel("epoch")
#     plt.ylabel("mean Dice")
#     plt.legend(["train", "val"])
#     plt.tight_layout()
#     plt.savefig(os.path.join(outdir, "dice.png"))
#     plt.close()


# def main():
#     ap = argparse.ArgumentParser()
#     ap.add_argument("--data_root", type=str, required=True, help="root containing images/ and masks/")
#     ap.add_argument("--num_classes", type=int, default=4)
#     ap.add_argument("--epochs", type=int, default=20)
#     ap.add_argument("--batch", type=int, default=16)
#     ap.add_argument("--lr", type=float, default=1e-3)
#     ap.add_argument("--out", type=str, default="runs/oasis_unet")
#     ap.add_argument("--workers", type=int, default=None, help="override DataLoader workers; default auto")
#     args = ap.parse_args()

#     img_dir = os.path.join(args.data_root, "images")
#     mask_dir = os.path.join(args.data_root, "masks")

#     # 自动设置 workers & pin_memory
#     has_cuda = torch.cuda.is_available()
#     auto_workers = (4 if has_cuda else 0) if args.workers is None else args.workers
#     pin_mem = has_cuda  # 只有 CUDA 时才启用以避免无谓警告

#     train_loader = make_loader(
#         img_dir, mask_dir, args.num_classes,
#         batch=args.batch, shuffle=True, augment=True,
#         workers=auto_workers, pin_memory=pin_mem
#     )
#     # 这里简化：同源目录上做一次“验证”遍历（用于 sanity check）
#     val_loader = make_loader(
#         img_dir, mask_dir, args.num_classes,
#         batch=args.batch, shuffle=False, augment=False,
#         workers=max(1, auto_workers // 2), pin_memory=pin_mem
#     )

#     device = torch.device("cuda" if has_cuda else "cpu")
#     torch.backends.cudnn.benchmark = has_cuda

#     # 与 modules.py 对齐
#     model = UNet(in_ch=1, n_classes=args.num_classes, base=32).to(device)
#     opt = AdamW(model.parameters(), lr=args.lr)

#     best = -1.0
#     hist = {"train_loss": [], "val_loss": [], "train_dice": [], "val_dice": []}
#     os.makedirs(args.out, exist_ok=True)

#     for ep in range(1, args.epochs + 1):
#         # ---------------- Train ----------------
#         model.train()
#         tl, td = 0.0, 0.0
#         ntr = 0
#         for img, mask in train_loader:
#             img, mask = img.to(device), mask.to(device)
#             bs = img.size(0)
#             ntr += bs

#             opt.zero_grad(set_to_none=True)
#             logits = model(img)
#             # 对齐到 mask 的空间尺寸（只插值 logits）
#             logits = upsample_to_target(logits, mask)
#             loss = bce_dice_loss(logits, mask)
#             loss.backward()
#             opt.step()

#             tl += loss.item() * bs
#             md, _ = dice_score(logits.detach(), mask)
#             td += md * bs

#         tl /= max(1, ntr)
#         td /= max(1, ntr)

#         # ---------------- Validate ----------------
#         model.eval()
#         vl, vd = 0.0, 0.0
#         nva = 0
#         with torch.no_grad():
#             for img, mask in val_loader:
#                 img, mask = img.to(device), mask.to(device)
#                 bs = img.size(0)
#                 nva += bs

#                 logits = model(img)
#                 # 验证同样对齐 logits 与 mask
#                 logits = upsample_to_target(logits, mask)
#                 vl += bce_dice_loss(logits, mask).item() * bs
#                 md, _ = dice_score(logits, mask)
#                 vd += md * bs

#         vl /= max(1, nva)
#         vd /= max(1, nva)

#         hist["train_loss"].append(tl)
#         hist["val_loss"].append(vl)
#         hist["train_dice"].append(td)
#         hist["val_dice"].append(vd)

#         print(f"[epoch {ep:03d}] loss {tl:.4f}/{vl:.4f}  dice {td:.4f}/{vd:.4f}")

#         # Save best
#         if vd > best:
#             best = vd
#             torch.save({"model": model.state_dict()}, os.path.join(args.out, "best.pt"))

#         # Update curves each epoch
#         plot_curves(hist, args.out)

#     print(f"best val mean Dice = {best:.4f}")


# if __name__ == "__main__":
#     main()

# recognition/unet_oasis_s4908819/train.py
import os, argparse, time
import numpy as np
import torch
import torch.nn.functional as F
from torch.optim import AdamW
from torch.utils.data import DataLoader

# Use a headless-friendly backend for matplotlib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# 原：from dataset import make_loader
from dataset import make_loader_from_split
from modules import UNet  # 注意：与 modules.py 中的类名一致


def upsample_to_target(logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Resize logits to the same HxW as target via bilinear interpolation (only resize logits)."""
    if logits.shape[-2:] != target.shape[-2:]:
        logits = F.interpolate(logits, size=target.shape[-2:], mode="bilinear", align_corners=False)
    return logits


def dice_score(pred, target, eps=1e-6):
    """
    pred: logits, shape [N, C, H, W]
    target: one-hot float mask, shape [N, C, H, W]
    返回 (mean_dice_scalar, per_class_numpy)
    """
    probs = torch.sigmoid(pred)
    probs = (probs > 0.5).float()
    intersect = (probs * target).sum(dim=(0, 2, 3))
    denom = probs.sum(dim=(0, 2, 3)) + target.sum(dim=(0, 2, 3)) + eps
    per_class = (2 * intersect / denom)
    return per_class.mean().item(), per_class.detach().cpu().numpy()


def bce_dice_loss(logits, target, eps=1e-6):
    """
    多类别 one-hot 情况下的 BCE + Dice 组合损失。
    target 需为 [N, C, H, W]、float、取值 {0,1}。
    """
    bce = F.binary_cross_entropy_with_logits(logits, target)
    probs = torch.sigmoid(logits)
    intersect = (probs * target).sum(dim=(0, 2, 3))
    denom = probs.sum(dim=(0, 2, 3)) + target.sum(dim=(0, 2, 3)) + eps
    dice = 1 - (2 * intersect / denom).mean()
    return bce + dice


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
    plt.ylabel("mean Dice")
    plt.legend(["train", "val"])
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, "dice.png"))
    plt.close()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_root", type=str, required=True, help="root containing images/ and masks/")
    ap.add_argument("--num_classes", type=int, default=4)
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--out", type=str, default="runs/oasis_unet")
    ap.add_argument("--workers", type=int, default=None, help="override DataLoader workers; default auto")
    args = ap.parse_args()

    img_dir = os.path.join(args.data_root, "images")
    mask_dir = os.path.join(args.data_root, "masks")

    # 自动设置 workers & pin_memory
    has_cuda = torch.cuda.is_available()
    auto_workers = (4 if has_cuda else 0) if args.workers is None else args.workers
    pin_mem = has_cuda  # 只有 CUDA 时才启用以避免无谓警告

    # 在构建 DataLoader 的地方改成：
    train_loader = make_loader_from_split(
        root=args.data_root, split="train", num_classes=args.num_classes,
        batch=args.batch, shuffle=True, augment=True, workers=args.workers
    )
    val_loader = make_loader_from_split(
        root=args.data_root, split="validate", num_classes=args.num_classes,
        batch=args.batch, shuffle=False, augment=False, workers=args.workers
    )

    device = torch.device("cuda" if has_cuda else "cpu")
    torch.backends.cudnn.benchmark = has_cuda

    # 与 modules.py 对齐
    model = UNet(in_ch=1, n_classes=args.num_classes, base=32).to(device)
    opt = AdamW(model.parameters(), lr=args.lr)

    best = -1.0
    hist = {"train_loss": [], "val_loss": [], "train_dice": [], "val_dice": []}
    os.makedirs(args.out, exist_ok=True)

    for ep in range(1, args.epochs + 1):
        # ---------------- Train ----------------
        model.train()
        tl, td = 0.0, 0.0
        ntr = 0
        for img, mask in train_loader:
            img, mask = img.to(device), mask.to(device)
            bs = img.size(0)
            ntr += bs

            opt.zero_grad(set_to_none=True)
            logits = model(img)
            # 对齐到 mask 的空间尺寸（只插值 logits）
            logits = upsample_to_target(logits, mask)
            loss = bce_dice_loss(logits, mask)
            loss.backward()
            opt.step()

            tl += loss.item() * bs
            md, _ = dice_score(logits.detach(), mask)
            td += md * bs

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
                # 验证同样对齐 logits 与 mask
                logits = upsample_to_target(logits, mask)
                vl += bce_dice_loss(logits, mask).item() * bs
                md, _ = dice_score(logits, mask)
                vd += md * bs

        vl /= max(1, nva)
        vd /= max(1, nva)

        hist["train_loss"].append(tl)
        hist["val_loss"].append(vl)
        hist["train_dice"].append(td)
        hist["val_dice"].append(vd)

        print(f"[epoch {ep:03d}] loss {tl:.4f}/{vl:.4f}  dice {td:.4f}/{vd:.4f}")

        # Save best
        if vd > best:
            best = vd
            torch.save({"model": model.state_dict()}, os.path.join(args.out, "best.pt"))

        # Update curves each epoch
        plot_curves(hist, args.out)

    print(f"best val mean Dice = {best:.4f}")


if __name__ == "__main__":
    main()
