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
    上采样块：
      in_ch   : 来自更深层的输入通道
      skip_ch : 与之拼接的 skip 特征通道
      out_ch  : 本层输出通道
    设计要点：
      先把 in_ch 上采样到 out_ch；
      拼接后通道 = out_ch(上采样) + skip_ch；
      再用 DoubleConv( out_ch + skip_ch -> out_ch ) 消化融合。
    """
    def __init__(self, in_ch: int, skip_ch: int, out_ch: int):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_ch, out_ch, kernel_size=2, stride=2)
        self.conv = DoubleConv(out_ch + skip_ch, out_ch)

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = self.up(x)
        # 若维度不齐（整除误差），对齐到 skip 的空间尺寸
        if x.size(-1) != skip.size(-1) or x.size(-2) != skip.size(-2):
            diffY = skip.size(-2) - x.size(-2)
            diffX = skip.size(-1) - x.size(-1)
            x = F.pad(
                x,
                [diffX // 2, diffX - diffX // 2,
                 diffY // 2, diffY - diffY // 2]
            )
        # 通道维拼接： [N, C, H, W]
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
    标准对称 U-Net（3 个下采样 + bottleneck + 3 个上采样）
    - in_ch:    输入通道（OASIS 灰度=1；若你把 PNG 堆成 RGB，则传 3）
    - n_classes:类别数（按你的 mask 来设，比如 3/4）
    - base:     基础通道宽度（32/64 均可）
    编码： base, 2b, 4b, 8b, 16b
    解码： 16b->8b (skip 8b), 8b->4b (skip 4b), 4b->2b (skip 2b)
    """
    def __init__(self, in_ch: int = 1, n_classes: int = 3, base: int = 32):
        super().__init__()
        # Encoder
        self.inc = DoubleConv(in_ch, base)           # -> b
        self.down1 = Down(base, base * 2)            # -> 2b
        self.down2 = Down(base * 2, base * 4)        # -> 4b
        self.down3 = Down(base * 4, base * 8)        # -> 8b
        self.bottleneck = DoubleConv(base * 8, base * 16)  # -> 16b

        # Decoder（显式指定 skip_ch）
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
