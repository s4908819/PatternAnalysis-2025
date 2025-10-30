import os, argparse, torch
import numpy as np
from PIL import Image
from dataset_hipmri3d import HipMRI3D
from modules_unet3d import UNet3D

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--root', type=str, required=True)
    ap.add_argument('--ckpt', type=str, required=True)
    ap.add_argument('--save', type=str, default='./preds3d')
    ap.add_argument('--shape', type=str, default='64,128,128')
    ap.add_argument('--classes', type=int, default=2)
    ap.add_argument('--device', type=str, default='cuda')
    args = ap.parse_args()

    os.makedirs(args.save, exist_ok=True)
    D,H,W = map(int, args.shape.split(','))

    ds = HipMRI3D(args.root, 'test', args.classes, (D,H,W))
    dev = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    net = UNet3D(1, args.classes, base=16).to(dev)
    state = torch.load(args.ckpt, map_location=dev)
    net.load_state_dict(state['model']); net.eval()

    with torch.no_grad():
        for i in range(min(len(ds), 10)):
            x,y = ds[i]
            z = net(x.unsqueeze(0).to(dev))              # (1,K,D,H,W)
            prob = torch.sigmoid(z)[0].cpu().numpy()     # (K,D,H,W)
            pred = prob.argmax(0).astype(np.uint8)       # (D,H,W)
            mid = pred.shape[0]//2
            Image.fromarray((x.numpy()[0,mid]*127+128).clip(0,255).astype(np.uint8)).save(os.path.join(args.save,f"img_{i:03d}.png"))
            Image.fromarray((y.numpy()[:,mid].argmax(0)).astype(np.uint8)).save(os.path.join(args.save,f"gt_{i:03d}.png"))
            Image.fromarray(pred[mid]).save(os.path.join(args.save,f"pred_{i:03d}.png"))

if __name__ == "__main__":
    main()
