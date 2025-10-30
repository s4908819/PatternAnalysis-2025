import torch
import torch.nn as nn
import torch.nn.functional as F

def CBR(cin, cout, k=3, s=1, p=1):
    return nn.Sequential(
        nn.Conv3d(cin, cout, k, s, p, bias=False),
        nn.BatchNorm3d(cout),
        nn.ReLU(inplace=True)
    )

class DoubleConv(nn.Module):
    def __init__(self, c1, c2):
        super().__init__()
        self.net = nn.Sequential(CBR(c1,c2), CBR(c2,c2))
    def forward(self, x): return self.net(x)

class UNet3D(nn.Module):
    def __init__(self, in_ch=1, out_ch=2, base=16):
        super().__init__()
        self.inc = DoubleConv(in_ch, base)
        self.d1  = nn.Sequential(nn.MaxPool3d(2), DoubleConv(base, base*2))
        self.d2  = nn.Sequential(nn.MaxPool3d(2), DoubleConv(base*2, base*4))
        self.d3  = nn.Sequential(nn.MaxPool3d(2), DoubleConv(base*4, base*8))
        self.u1  = nn.ConvTranspose3d(base*8, base*4, 2, 2)
        self.c1  = DoubleConv(base*8, base*4)
        self.u2  = nn.ConvTranspose3d(base*4, base*2, 2, 2)
        self.c2  = DoubleConv(base*4, base*2)
        self.u3  = nn.ConvTranspose3d(base*2, base,   2, 2)
        self.c3  = DoubleConv(base*2, base)
        self.out = nn.Conv3d(base, out_ch, 1)

    def forward(self, x):
        x0 = self.inc(x)
        x1 = self.d1(x0)
        x2 = self.d2(x1)
        x3 = self.d3(x2)
        x  = self.u1(x3)
        x  = self.c1(torch.cat([x, x2], dim=1))
        x  = self.u2(x)
        x  = self.c2(torch.cat([x, x1], dim=1))
        x  = self.u3(x)
        x  = self.c3(torch.cat([x, x0], dim=1))
        return self.out(x)
