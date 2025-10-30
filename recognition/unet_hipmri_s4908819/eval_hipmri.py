# recognition/unet_hipmri_s4908819/eval_hipmri.py
import os, argparse, numpy as np, torch
import torch.nn.functional as F
from dataset_hipmri import make_loader_from_split
from modules_resunet import UNetRes  # 只使用 ResUNet
from typing import Tuple

# ---------------- Utils ----------------
def _strip_module_prefix(state_dict):
    """去掉 DataParallel 保存时的 'module.' 前缀"""
    if not any(k.startswith("module.") for k in state_dict.keys()):
        return state_dict
    return {k.replace("module.", "", 1): v for k, v in state_dict.items()}

def set_seed(seed: int = 42):
    import random
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

@torch.no_grad()
def soft_dice_per_class(
    logits: torch.Tensor, target_onehot: torch.Tensor,
    exclude_background: bool = True, eps: float = 1e-6
) -> torch.Tensor:
    """
    logits: (B,K,H,W), target_onehot: (B,K,H,W) one-hot
    return: (K-1,) or (K,) —— 按 batch 平均后的 per-class Dice（soft 版）
    """
    probs = F.softmax(logits, dim=1)
    K = probs.shape[1]
    cls_range = range(1, K) if (exclude_background and K > 1) else range(K)
    per_class = []
    for c in cls_range:
        p = probs[:, c:c+1]   # (B,1,H,W)
        t = target_onehot[:, c:c+1]
        inter = (p * t).sum(dim=(1,2,3))
        denom = p.sum(dim=(1,2,3)) + t.sum(dim=(1,2,3)) + eps
        d = (2*inter + eps) / denom  # (B,)
        per_class.append(d.mean())   # 标量
    return torch.stack(per_class, dim=0) if per_class else torch.tensor([1.0], device=logits.device)

@torch.no_grad()
def hard_iou_per_class(
    logits: torch.Tensor, target_onehot: torch.Tensor,
    exclude_background: bool = True, eps: float = 1e-6
) -> torch.Tensor:
    """
    基于硬预测（argmax）计算 per-class IoU；返回 (K-1,) or (K,)
    """
    K = logits.shape[1]
    pred_lbl = torch.argmax(logits, dim=1, keepdim=False)  # (B,H,W)
    # one-hot 化预测
    pred_1h = F.one_hot(pred_lbl, num_classes=K).permute(0,3,1,2).float()  # (B,K,H,W)
    cls_range = range(1, K) if (exclude_background and K > 1) else range(K)
    ious = []
    for c in cls_range:
        p = pred_1h[:, c:c+1]  # (B,1,H,W)
        t = target_onehot[:, c:c+1]
        inter = (p * t).sum(dim=(1,2,3))
        union = (p + t - p*t).sum(dim=(1,2,3))
        iou = ((inter + eps) / (union + eps)).mean()  # 标量
        ious.append(iou)
    return torch.stack(ious, dim=0) if ious else torch.tensor([1.0], device=logits.device)

def upsample_to_target(logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """将 logits 双线性插值到与 target 相同的 HxW。"""
    if logits.shape[-2:] != target.shape[-2:]:
        logits = F.interpolate(logits, size=target.shape[-2:], mode="bilinear", align_corners=False)
    return logits

def make_vis_rows(img: torch.Tensor, mask_1h: torch.Tensor, pred_1h: torch.Tensor) -> np.ndarray:
    """
    img: (1,H,W) in [0,1]
    mask_1h, pred_1h: (K,H,W) one-hot（已去背景后可视化第1通道）
    返回 (H, W*3) 的拼接灰度阵列：IMG | GT | PRED
    """
    x = img[0].clamp(0,1).cpu().numpy()
    # 可视化前景=所有非背景通道并在[0,1]显示
    gt = mask_1h[1:].sum(0).clamp(0,1).cpu().numpy() if mask_1h.shape[0] > 1 else mask_1h[0].cpu().numpy()
    pr = pred_1h[1:].sum(0).clamp(0,1).cpu().numpy() if pred_1h.shape[0] > 1 else pred_1h[0].cpu().numpy()
    H, W = x.shape
    canvas = np.zeros((H, W*3), dtype=np.float32)
    canvas[:, 0:W]   = x
    canvas[:, W:2*W] = gt
    canvas[:, 2*W:]  = pr
    return (canvas * 255.0).astype(np.uint8)

def save_vis_grid(save_dir: str, basename: str, grid: np.ndarray):
    os.makedirs(save_dir, exist_ok=True)
    from imageio.v2 import imwrite
    imwrite(os.path.join(save_dir, f"{basename}.png"), grid)

# ---------------- Main ----------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data_root', required=True, help='根目录，包含 keras_png_slices_* 与 *_seg_*')
    ap.add_argument('--weights', required=True, help='模型权重 best.pt')
    ap.add_argument('--num_classes', type=int, default=2, help='类别数（含背景）')
    ap.add_argument('--binary', type=int, default=1, help='1=二分类(前景vs背景)，0=多分类')
    ap.add_argument('--prostate_label', type=int, default=2, help='二分类时：把该标签映射为前景(默认2=前列腺)')
    ap.add_argument('--batch', type=int, default=8)
    ap.add_argument('--workers', type=int, default=2)
    ap.add_argument('--base', type=int, default=32, help='ResUNet 宽度基数（与你训练时一致）')
    ap.add_argument('--resize', type=int, default=256, help='评测时将图片/掩码缩放到的尺寸')
    ap.add_argument('--device', type=str, default='cuda')
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--save_vis', type=int, default=0, help='>0 则保存该数量的可视化（IMG|GT|PRED）PNG')
    ap.add_argument('--vis_out', type=str, default='runs/hipmri_eval_vis', help='可视化输出目录')
    args = ap.parse_args()

    set_seed(args.seed)
    dev = torch.device(args.device if torch.cuda.is_available() else 'cpu')

    # --- DataLoader：test split；binary/prostate_label/resize_to 与数据集对齐 ---
    dl_te = make_loader_from_split(
        root=args.data_root, split='test', num_classes=args.num_classes,
        batch=args.batch, shuffle=False, augment=False, workers=args.workers,
        resize_to=(args.resize, args.resize), binary=bool(args.binary),
        prostate_label=args.prostate_label
    )

    # --- Model ---
    model = UNetRes(n_classes=args.num_classes, base=args.base).to(dev)

    # --- Load Weights (兼容 DataParallel/ckpt包装) ---
    ckpt = torch.load(args.weights, map_location=dev)
    state = ckpt.get('model', ckpt)
    state = _strip_module_prefix(state)
    try:
        model.load_state_dict(state, strict=True)
    except RuntimeError as e:
        print(f"[warn] strict=True 加载失败：{e}\n[warn] 回退到 strict=False 继续加载（请确认模型结构与训练一致）")
        model.load_state_dict(state, strict=False)
    model.eval()

    # --- Eval Loop ---
    dice_batches = []  # (K-1,) per batch
    iou_batches  = []  # (K-1,) per batch (硬预测)
    saved = 0

    for img, mask in dl_te:
        # img: (B,1,H,W) in [0,1], mask: (B,K,H,W) one-hot
        img, mask = img.to(dev), mask.to(dev)
        logits = model(img)                      # (B,K,h,w)
        logits = upsample_to_target(logits, mask)

        # soft Dice（去背景）
        dice_pc = soft_dice_per_class(logits, mask, exclude_background=True)  # (K-1,)
        dice_batches.append(dice_pc.unsqueeze(0))  # (1,K-1)

        # IoU（基于 argmax 硬预测）
        iou_pc = hard_iou_per_class(logits, mask, exclude_background=True)    # (K-1,)
        iou_batches.append(iou_pc.unsqueeze(0))

        # 可视化若需要
        if args.save_vis and saved < args.save_vis:
            with torch.no_grad():
                K = logits.shape[1]
                pred_lbl = torch.argmax(logits, dim=1)  # (B,H,W)
                pred_1h  = F.one_hot(pred_lbl, num_classes=K).permute(0,3,1,2).float()
                # 取 batch 的第一张
                vis = make_vis_rows(img[0], mask[0], pred_1h[0])
                save_vis_grid(args.vis_out, f"sample_{saved:03d}", vis)
                saved += 1

    dice_batches = torch.cat(dice_batches, dim=0) if dice_batches else torch.zeros(1, args.num_classes-1)
    iou_batches  = torch.cat(iou_batches,  dim=0) if iou_batches  else torch.zeros(1, args.num_classes-1)

    per_class_dice = dice_batches.mean(dim=0).cpu().numpy()  # (K-1,)
    per_class_iou  = iou_batches.mean(dim=0).cpu().numpy()   # (K-1,)
    mean_dice_no_bg = per_class_dice.mean()
    mean_iou_no_bg  = per_class_iou.mean()

    # --- Report ---
    print("== Test Dice (no-bg) ==")
    for i, d in enumerate(per_class_dice, start=1):
        print(f"class_{i}: {d:.4f}")
    print(f"mean (no-bg): {mean_dice_no_bg:.4f}")

    print("\n== Test IoU (no-bg, hard pred) ==")
    for i, j in enumerate(per_class_iou, start=1):
        print(f"class_{i}: {j:.4f}")
    print(f"mean (no-bg): {mean_iou_no_bg:.4f}")

    # 前列腺 Dice 报告（对 binary：no-bg 的第一个类即前景=前列腺）
    if args.num_classes == 2 or args.binary == 1:
        prostate_dice = per_class_dice[0]
    else:
        # 多分类时需要你根据数据集中“去背景后的索引”来指定，这里默认 1 即第一个非背景类
        prostate_dice = per_class_dice[0]

    print(f"\n>> PROSTATE Dice (test) = {prostate_dice:.4f}  "
          f"{'(OK ≥ 0.75)' if prostate_dice >= 0.75 else '(NOT YET < 0.75)'}")

if __name__ == "__main__":
    main()
