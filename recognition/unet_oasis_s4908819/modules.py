# recognition/unet_oasis_s4908819/modules.py
import torch
import torch.nn as nn
import torch.nn.functional as F


class DoubleConv(nn.Module):
    """(Conv-BN-ReLU) x2"""
    def __init__(self, in_ch: int, out_ch: int, ks: int = 3, pad: int = 1):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, ks, padding=pad, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, ks, padding=pad, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class Down(nn.Module):
    """MaxPool2d + DoubleConv"""
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.pool = nn.MaxPool2d(2)
        self.conv = DoubleConv(in_ch, out_ch)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(self.pool(x))


class Up(nn.Module):
    """
    Upsampling block:
      in_ch   : input channels from a deeper layer
      skip_ch : skip connection feature channels
      out_ch  : output channels for this layer
    Design notes:
      First upsample in_ch to out_ch;
      After concatenation, total channels = out_ch(upsampled) + skip_ch;
      Then fuse via DoubleConv(out_ch + skip_ch -> out_ch).
    """
    def __init__(self, in_ch: int, skip_ch: int, out_ch: int):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_ch, out_ch, kernel_size=2, stride=2)
        self.conv = DoubleConv(out_ch + skip_ch, out_ch)

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = self.up(x)
        # Align to skip’s spatial dimensions if mismatch due to rounding
        if x.size(-1) != skip.size(-1) or x.size(-2) != skip.size(-2):
            diffY = skip.size(-2) - x.size(-2)
            diffX = skip.size(-1) - x.size(-1)
            x = F.pad(
                x,
                [diffX // 2, diffX - diffX // 2,
                 diffY // 2, diffY - diffY // 2]
            )
        # Concatenate along channel dimension: [N, C, H, W]
        x = torch.cat([x, skip], dim=1)
        return self.conv(x)


class OutConv(nn.Module):
    def __init__(self, in_ch: int, n_classes: int):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, n_classes, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class UNet(nn.Module):
    """
    Standard symmetric U-Net (3 downsamples + bottleneck + 3 upsamples)
    - in_ch:    input channels (OASIS grayscale=1; if stacked PNGs as RGB, use 3)
    - n_classes:number of classes (e.g., 3 or 4 depending on mask)
    - base:     base channel width (32 or 64 both fine)
    Encoder: base, 2b, 4b, 8b, 16b
    Decoder: 16b->8b (skip 8b), 8b->4b (skip 4b), 4b->2b (skip 2b)
    """
    def __init__(self, in_ch: int = 1, n_classes: int = 3, base: int = 32):
        super().__init__()
        # Encoder
        self.inc = DoubleConv(in_ch, base)           # -> b
        self.down1 = Down(base, base * 2)            # -> 2b
        self.down2 = Down(base * 2, base * 4)        # -> 4b
        self.down3 = Down(base * 4, base * 8)        # -> 8b
        self.bottleneck = DoubleConv(base * 8, base * 16)  # -> 16b

        # Decoder (explicitly specify skip_ch)
        self.up3 = Up(in_ch=base * 16, skip_ch=base * 8, out_ch=base * 8)  # 16b -> 8b
        self.up2 = Up(in_ch=base * 8,  skip_ch=base * 4, out_ch=base * 4)  # 8b  -> 4b
        self.up1 = Up(in_ch=base * 4,  skip_ch=base * 2, out_ch=base * 2)  # 4b  -> 2b

        self.outc = OutConv(base * 2, n_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Encoder
        x1 = self.inc(x)       # b
        x2 = self.down1(x1)    # 2b
        x3 = self.down2(x2)    # 4b
        x4 = self.down3(x3)    # 8b
        xb = self.bottleneck(x4)  # 16b

        # Decoder
        x = self.up3(xb, x4)   # -> 8b
        x = self.up2(x,  x3)   # -> 4b
        x = self.up1(x,  x2)   # -> 2b
        logits = self.outc(x)  # -> n_classes
        return logits
