# recognition/unet_hipmri_s4908819/eval_hipmri.py
import argparse, os, numpy as np, torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from dataset_hipmri import make_loader_from_split
from modules_resunet import UNetRes  # 若你用的是标准UNet就改成 from modules import UNet

@torch.no_grad()
def dice_per_class(logits, target_onehot, exclude_background=True, eps=1e-6):
    # logits: (B,K,H,W), target_onehot: (B,K,H,W)
    probs = F.softmax(logits, dim=1)
    K = probs.shape[1]
    cls_range = range(1, K) if (exclude_background and K > 1) else range(K)

    per_class_batch = []
    for c in cls_range:
        # 保留通道维度 -> (B,1,H,W)
        p = probs[:, c:c+1, :, :]
        t = target_onehot[:, c:c+1, :, :]

        inter = (p * t).sum(dim=(1, 2, 3))
        denom = p.sum(dim=(1, 2, 3)) + t.sum(dim=(1, 2, 3)) + eps
        d = (2 * inter + eps) / denom    # (B,)
        per_class_batch.append(d)

    if len(per_class_batch) == 0:
        return np.array([1.0], dtype=np.float32)

    pc = torch.stack(per_class_batch, dim=0).mean(dim=1).cpu().numpy()  # (K-1,) or (K,)
    return pc


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data_root', required=True)   # 指向包含 keras_png_slices_* 的根目录
    ap.add_argument('--weights', required=True)     # 训练时保存的 best.pt
    ap.add_argument('--num_classes', type=int, default=4)
    ap.add_argument('--batch', type=int, default=8)
    ap.add_argument('--workers', type=int, default=2)
    ap.add_argument('--base', type=int, default=32)         # 与训练一致
    ap.add_argument('--resize', type=int, default=256)      # 与训练一致
    ap.add_argument('--prostate_idx', type=int, default=1)  # 前列腺类别在 [1..K-1] 的索引（去背景后）
    ap.add_argument('--device', type=str, default='cuda')
    args = ap.parse_args()

    dev = torch.device(args.device if torch.cuda.is_available() else 'cpu')

    # DataLoader：使用 test split
    dl_te = make_loader_from_split(
        root=args.data_root, split='test', num_classes=args.num_classes,
        batch=args.batch, shuffle=False, augment=False, workers=args.workers,
        resize_to=(args.resize, args.resize)
    )

    # 模型（与你训练时一致：UNetRes 或 UNet）
    model = UNetRes(n_classes=args.num_classes, base=args.base).to(dev)
    state = torch.load(args.weights, map_location=dev)
    model.load_state_dict(state.get('model', state))
    model.eval()

    # 累计每个 batch 的 per-class dice，再求全数据均值
    pcs = []  # list of np.array per-batch
    for img, mask in dl_te:
        img, mask = img.to(dev), mask.to(dev)
        logits = model(img)  # (B,K,H,W)
        # 与 mask 尺寸对齐
        if logits.shape[-2:] != mask.shape[-2:]:
            logits = F.interpolate(logits, size=mask.shape[-2:], mode='bilinear', align_corners=False)
        pc = dice_per_class(logits, mask, exclude_background=True)  # 长度 K-1
        pcs.append(pc)

    pcs = np.stack(pcs, axis=0)  # (num_batches, K-1)
    per_class_dice = pcs.mean(axis=0)  # (K-1,)
    mean_no_bg = per_class_dice.mean()

    print("== Test Dice (no-bg) ==")
    for i, d in enumerate(per_class_dice, start=1):  # 从1开始对应去背景后的索引
        print(f"class_{i}: {d:.4f}")
    print(f"mean (no-bg): {mean_no_bg:.4f}")

    # 报告前列腺 Dice
    idx = args.prostate_idx
    if 1 <= idx <= len(per_class_dice):
        prostate_dice = per_class_dice[idx-1]
        print(f"\n>> PROSTATE Dice (test) = {prostate_dice:.4f}  "
              f"{'(OK ≥ 0.75)' if prostate_dice >= 0.75 else '(NOT YET < 0.75)'}")
    else:
        print(f"\n[warning] prostate_idx={idx} 超出范围，当前 no-bg 类别数={len(per_class_dice)}")
if __name__ == "__main__":
    main()
