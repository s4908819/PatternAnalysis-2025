# recognition/unet_oasis_s4908819/modules.py
import torch
import torch.nn as nn
import torch.nn.functional as F

class ConvBNReLU(nn.Module):
    def __init__(self, c_in, c_out, ks=3, p=1):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(c_in, c_out, ks, padding=p, bias=False),
            nn.BatchNorm2d(c_out),
            nn.ReLU(inplace=True),
            nn.Conv2d(c_out, c_out, ks, padding=p, bias=False),
            nn.BatchNorm2d(c_out),
            nn.ReLU(inplace=True)
        )
    def forward(self, x): return self.block(x)

class Down(nn.Module):
    def __init__(self, c_in, c_out):
        super().__init__()
        self.pool = nn.MaxPool2d(2)
        self.conv = ConvBNReLU(c_in, c_out)
    def forward(self, x): return self.conv(self.pool(x))

class Up(nn.Module):
    def __init__(self, c_in, c_out):
        super().__init__()
        self.up = nn.ConvTranspose2d(c_in, c_in//2, 2, stride=2)
        self.conv = ConvBNReLU(c_in, c_out)
    def forward(self, x, skip):
        x = self.up(x)
        # pad if needed
        diffY = skip.size(2) - x.size(2)
        diffX = skip.size(3) - x.size(3)
        x = F.pad(x, [diffX//2, diffX-diffX//2, diffY//2, diffY-diffY//2])
        x = torch.cat([skip, x], dim=1)
        return self.conv(x)

class UNet2D(nn.Module):
    def __init__(self, in_ch=1, num_classes=4, base=32):
        super().__init__()
        self.inc = ConvBNReLU(in_ch, base)
        self.d1  = Down(base, base*2)
        self.d2  = Down(base*2, base*4)
        self.d3  = Down(base*4, base*8)
        self.bottleneck = ConvBNReLU(base*8, base*16)
        self.u3  = Up(base*16, base*8)
        self.u2  = Up(base*8,  base*4)
        self.u1  = Up(base*4,  base*2)
        self.out = nn.Conv2d(base*2, num_classes, 1)
    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.d1(x1)
        x3 = self.d2(x2)
        x4 = self.d3(x3)
        xb = self.bottleneck(x4)
        x  = self.u3(xb, x3)
        x  = self.u2(x,  x2)
        x  = self.u1(x,  x1)
        return self.out(x)  # logits
