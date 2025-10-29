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
    残差块：ConvBNReLU × 2 + shortcut
    - 若 in_ch != out_ch，用 1x1 卷积对残差分支做投影以匹配通道
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
    上采样块（与原版 Up 的接口保持一致）：
      in_ch   : 来自更深层的输入通道
      skip_ch : 与之拼接的 skip 特征通道
      out_ch  : 本层输出通道
    步骤：
      1) 反卷积把 in_ch 上采样到 out_ch；
      2) 与 skip 在通道维拼接（out_ch + skip_ch）；
      3) 用 ResBlock( out_ch + skip_ch -> out_ch ) 融合。
    """
    def __init__(self, in_ch: int, skip_ch: int, out_ch: int):
        super().__init__()
        self.up   = nn.ConvTranspose2d(in_ch, out_ch, kernel_size=2, stride=2)
        self.fuse = ResBlock(out_ch + skip_ch, out_ch)

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = self.up(x)
        # 若空间尺寸不齐，pad 对齐到 skip
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
    残差版对称 U-Net（结构与你现有 UNet 保持一致）：
      编码： base, 2b, 4b, 8b, 16b
      解码： 16b->8b (skip 8b), 8b->4b (skip 4b), 4b->2b (skip 2b)
    - in_ch:     输入通道（灰度=1）
    - n_classes: 类别数
    - base:      基础通道宽度（默认 32；显存紧张可用 16）
    """
    def __init__(self, in_ch: int = 1, n_classes: int = 3, base: int = 32):
        super().__init__()
        # Encoder
        self.inc   = ResBlock(in_ch, base)             # -> b
        self.down1 = Down(base, base * 2)              # -> 2b
        self.down2 = Down(base * 2, base * 4)          # -> 4b
        self.down3 = Down(base * 4, base * 8)          # -> 8b
        self.bottleneck = ResBlock(base * 8, base * 16)  # -> 16b

        # Decoder（保持与原 Up 相同的构造签名）
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
