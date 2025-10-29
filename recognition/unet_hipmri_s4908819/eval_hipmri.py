# recognition/unet_hipmri_s4908819/eval_hipmri.py
import argparse, os, numpy as np, torch
import torch.nn.functional as F
from dataset_hipmri import make_loader_from_split
from modules_resunet import UNetRes  # 只使用 ResUNet

def _strip_module_prefix(state_dict):
    """去掉 DataParallel 保存时的 'module.' 前缀"""
    if not any(k.startswith("module.") for k in state_dict.keys()):
        return state_dict
    return {k.replace("module.", "", 1): v for k, v in state_dict.items()}

@torch.no_grad()
def dice_per_class(logits: torch.Tensor, target_onehot: torch.Tensor,
                   exclude_background: bool = True, eps: float = 1e-6) -> np.ndarray:
    """
    logits: (B,K,H,W), target_onehot: (B,K,H,W)
    return: np.ndarray, shape = (K-1,) 或 (K,)
    """
    probs = F.softmax(logits, dim=1)
    K = probs.shape[1]
    cls_range = range(1, K) if (exclude_background and K > 1) else range(K)

    per_class_batch = []
    for c in cls_range:
        p = probs[:, c:c+1, :, :]   # (B,1,H,W)
        t = target_onehot[:, c:c+1, :, :]
        inter = (p * t).sum(dim=(1, 2, 3))
        denom = p.sum(dim=(1, 2, 3)) + t.sum(dim=(1, 2, 3)) + eps
        d = (2 * inter + eps) / denom  # (B,)
        per_class_batch.append(d)

    if len(per_class_batch) == 0:
        return np.array([1.0], dtype=np.float32)

    pc = torch.stack(per_class_batch, dim=0).mean(dim=1).cpu().numpy()  # (K-1,) or (K,)
    return pc


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data_root', required=True, help='根目录，包含 keras_png_slices_* 与 *_seg_*')
    ap.add_argument('--weights', required=True, help='模型权重 best.pt')
    ap.add_argument('--num_classes', type=int, default=2, help='类别数（含背景）')
    ap.add_argument('--binary', type=int, default=1, help='1=二分类(前景vs背景)，0=多分类')
    ap.add_argument('--prostate_idx', type=int, default=1, help='多分类时：去背景后的前列腺通道索引(1..K-1)')
    ap.add_argument('--batch', type=int, default=8)
    ap.add_argument('--workers', type=int, default=2)
    ap.add_argument('--base', type=int, default=32, help='ResUNet 宽度基数（与你训练时一致）')
    ap.add_argument('--resize', type=int, default=256, help='评测时将图片/掩码缩放到的尺寸')
    ap.add_argument('--device', type=str, default='cuda')
    args = ap.parse_args()

    dev = torch.device(args.device if torch.cuda.is_available() else 'cpu')

    # DataLoader：使用 test split；把 binary 传入以确保数据集按二/多分类读取
    dl_te = make_loader_from_split(
        root=args.data_root, split='test', num_classes=args.num_classes,
        batch=args.batch, shuffle=False, augment=False, workers=args.workers,
        resize_to=(args.resize, args.resize), binary=bool(args.binary)
    )

    # 仅使用 ResUNet
    model = UNetRes(n_classes=args.num_classes, base=args.base).to(dev)

    # 加载权重：支持 {'model': state_dict} 或直接 state_dict；自动去掉 DataParallel 前缀
    ckpt = torch.load(args.weights, map_location=dev)
    state = ckpt.get('model', ckpt)
    state = _strip_module_prefix(state)

    try:
        model.load_state_dict(state, strict=True)
    except RuntimeError as e:
        print(f"[warn] strict=True 加载失败：{e}\n[warn] 回退到 strict=False 继续加载（请确认模型结构与训练一致）")
        model.load_state_dict(state, strict=False)

    model.eval()

    # 累计每个 batch 的 per-class dice
    pcs = []
    for img, mask in dl_te:
        img, mask = img.to(dev), mask.to(dev)  # mask: (B,K,H,W) one-hot
        logits = model(img)                    # (B,K,H,W)
        if logits.shape[-2:] != mask.shape[-2:]:
            logits = F.interpolate(logits, size=mask.shape[-2:], mode='bilinear', align_corners=False)
        pc = dice_per_class(logits, mask, exclude_background=True)  # (K-1,) 或 (1,)
        pcs.append(pc)

    pcs = np.stack(pcs, axis=0)       # (num_batches, K-1)
    per_class_dice = pcs.mean(axis=0) # (K-1,)
    mean_no_bg = per_class_dice.mean()

    print("== Test Dice (no-bg) ==")
    for i, d in enumerate(per_class_dice, start=1):  # 从1开始，对应去背景后的索引
        print(f"class_{i}: {d:.4f}")
    print(f"mean (no-bg): {mean_no_bg:.4f}")

    # 前列腺 Dice 报告
    if args.num_classes == 2 or args.binary == 1:
        # 二分类：no-bg 只有一个类，即前景=前列腺
        prostate_dice = per_class_dice[0]
    else:
        idx = args.prostate_idx
        if 1 <= idx <= len(per_class_dice):
            prostate_dice = per_class_dice[idx - 1]
        else:
            print(f"\n[warning] prostate_idx={idx} 超出范围（1..{len(per_class_dice)}）")
            return

    print(f"\n>> PROSTATE Dice (test) = {prostate_dice:.4f}  "
          f"{'(OK ≥ 0.75)' if prostate_dice >= 0.75 else '(NOT YET < 0.75)'}")


if __name__ == "__main__":
    main()
