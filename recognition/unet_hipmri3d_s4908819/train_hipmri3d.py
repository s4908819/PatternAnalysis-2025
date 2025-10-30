import os, argparse, torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from dataset_hipmri3d import HipMRI3D
from modules_unet3d import UNet3D

def dice_per_class(logits, target, eps=1e-6):
    # logits: (B,K,D,H,W); target: (B,K,D,H,W)
    prob = torch.sigmoid(logits)
    num = 2*(prob*target).sum(dim=(2,3,4))
    den = (prob+target).sum(dim=(2,3,4)) + eps
    return (num/den).mean(dim=0)  # (K,)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--root', type=str, required=True)   # .../HipMRI_Study_open
    ap.add_argument('--out',  type=str, default='./runs_hipmri3d')
    ap.add_argument('--epochs', type=int, default=100)
    ap.add_argument('--batch',  type=int, default=1)
    ap.add_argument('--workers',type=int, default=2)
    ap.add_argument('--shape',  type=str, default='64,128,128')
    ap.add_argument('--classes',type=int, default=2)
    ap.add_argument('--lr',     type=float, default=1e-3)
    ap.add_argument('--device', type=str, default='cuda')
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    D,H,W = map(int, args.shape.split(','))

    tr = HipMRI3D(args.root, 'train', args.classes, (D,H,W))
    va = HipMRI3D(args.root, 'validate', args.classes, (D,H,W))
    tl = DataLoader(tr, batch_size=args.batch, shuffle=True,  num_workers=args.workers, pin_memory=True)
    vl = DataLoader(va, batch_size=1,         shuffle=False, num_workers=args.workers, pin_memory=True)

    dev = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    net = UNet3D(1, args.classes, base=16).to(dev)
    opt = torch.optim.AdamW(net.parameters(), lr=args.lr)

    best = -1
    for ep in range(1, args.epochs+1):
        net.train(); tloss=0.0
        for x,y in tl:
            x,y = x.to(dev), y.to(dev)
            z = net(x)
            bce = F.binary_cross_entropy_with_logits(z, y)
            dsc = 1.0 - dice_per_class(z,y).mean()
            loss = bce + dsc
            opt.zero_grad(); loss.backward(); opt.step()
            tloss += loss.item()*x.size(0)
        tloss/=len(tr)

        net.eval(); dsum=None
        with torch.no_grad():
            for x,y in vl:
                x,y = x.to(dev), y.to(dev)
                z = net(x)
                d = dice_per_class(z,y)  # (K,)
                dsum = d if dsum is None else dsum + d
        dmean = (dsum/len(va)).cpu().numpy()
        prostate = dmean[1] if args.classes>1 else dmean[0]
        print(f"[{ep:03d}] trainLoss={tloss:.4f}  meanDice={dmean.mean():.3f}  prostateDice={prostate:.3f}")

        if prostate > best:
            best = prostate
            torch.save({'model': net.state_dict(), 'epoch': ep, 'dice': best},
                       os.path.join(args.out,'best.pt'))
            print(f"  ↳ saved best.pt (prostateDice={best:.3f})")

if __name__ == "__main__":
    main()
