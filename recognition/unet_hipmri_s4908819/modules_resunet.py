# recognition/unet_hipmri_s4908819/modules_resunet.py
import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------- Basic Blocks ----------------
class ConvBNReLU(nn.Module):
    """Conv2d -> BatchNorm2d -> ReLU"""
    def __init__(self, in_ch: int, out_ch: int, ks: int = 3, stride: int = 1, pad: int = 1):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, kernel_size=ks, stride=stride, padding=pad, bias=False)
        self.bn   = nn.BatchNorm2d(out_ch)
        self.act  = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.bn(self.conv(x)))


class ResBlock(nn.Module):
    """
    Residual block: ConvBNReLU × 2 + shortcut.
    - If in_ch != out_ch, use a 1x1 conv to project the residual to match channels.
    """
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.conv1 = ConvBNReLU(in_ch, out_ch, ks=3, pad=1)
        self.conv2 = ConvBNReLU(out_ch, out_ch, ks=3, pad=1)
        self.proj  = None if in_ch == out_ch else nn.Conv2d(in_ch, out_ch, kernel_size=1, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        idt = x if self.proj is None else self.proj(x)
        y   = self.conv2(self.conv1(x))
        return F.relu(y + idt, inplace=True)


# ---------------- Encoder / Decoder ----------------
class Down(nn.Module):
    """MaxPool2d + ResBlock"""
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.pool = nn.MaxPool2d(2)
        self.block = ResBlock(in_ch, out_ch)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(self.pool(x))


class Up(nn.Module):
    """
    Upsampling block (keeps the same interface as the original Up):
      in_ch   : input channels from the deeper layer
      skip_ch : skip-connection feature channels to concatenate
      out_ch  : output channels for this level
    Steps:
      1) Deconvolution upsamples in_ch to out_ch;
      2) Concatenate with skip along channel dim (out_ch + skip_ch);
      3) Fuse with ResBlock(out_ch + skip_ch -> out_ch).
    """
    def __init__(self, in_ch: int, skip_ch: int, out_ch: int):
        super().__init__()
        self.up   = nn.ConvTranspose2d(in_ch, out_ch, kernel_size=2, stride=2)
        self.fuse = ResBlock(out_ch + skip_ch, out_ch)

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = self.up(x)
        # If spatial sizes differ, pad to align with skip
        if x.size(-1) != skip.size(-1) or x.size(-2) != skip.size(-2):
            diffY = skip.size(-2) - x.size(-2)
            diffX = skip.size(-1) - x.size(-1)
            x = F.pad(
                x,
                [diffX // 2, diffX - diffX // 2,
                 diffY // 2, diffY - diffY // 2]
            )
        x = torch.cat([x, skip], dim=1)
        return self.fuse(x)


class OutConv(nn.Module):
    def __init__(self, in_ch: int, n_classes: int):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, n_classes, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


# ---------------- Residual U-Net ----------------
class UNetRes(nn.Module):
    """
    Residual symmetric U-Net (structure matches your existing U-Net):
      Encoder: base, 2b, 4b, 8b, 16b
      Decoder: 16b->8b (skip 8b), 8b->4b (skip 4b), 4b->2b (skip 2b)
    - in_ch:     input channels (grayscale = 1)
    - n_classes: number of classes
    - base:      base channel width (default 32; use 16 if memory is tight)
    """
    def __init__(self, in_ch: int = 1, n_classes: int = 3, base: int = 32):
        super().__init__()
        # Encoder
        self.inc   = ResBlock(in_ch, base)             # -> b
        self.down1 = Down(base, base * 2)              # -> 2b
        self.down2 = Down(base * 2, base * 4)          # -> 4b
        self.down3 = Down(base * 4, base * 8)          # -> 8b
        self.bottleneck = ResBlock(base * 8, base * 16)  # -> 16b

        # Decoder (keep the same constructor signature as the original Up)
        self.up3  = Up(in_ch=base * 16, skip_ch=base * 8, out_ch=base * 8)  # 16b -> 8b
        self.up2  = Up(in_ch=base * 8,  skip_ch=base * 4, out_ch=base * 4)  # 8b  -> 4b
        self.up1  = Up(in_ch=base * 4,  skip_ch=base * 2, out_ch=base * 2)  # 4b  -> 2b

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
