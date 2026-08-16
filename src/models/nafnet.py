"""
NAFNet-style restoration backbone: U-Net hierarchy, gated activation-free
blocks (SimpleGate, no GELU/ReLU), simplified channel attention (global
pool + 1x1 conv, no self-attention/transformer blocks), fused pixel-shuffle
upsampling head.

Single forward pass handles denoise + deblur + super-resolution jointly:
most computation happens at the LR input resolution (efficient for
inference speed), and the network only upsamples once, at the very end,
via PixelShuffle.

Scale factor: both of this problem's degradation pairs (512->256 and
256->128) are exactly 2x downsampling, so there is no need for a scale
flag or shape-based branching - the network is fully convolutional and
always upsamples by a fixed factor of 2 via PixelShuffle(2). Feeding it a
128x128 input yields 256x256 output; feeding it a 256x256 input (if
512-scale data becomes available later) yields 512x512 output, with no
architecture change required.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class LayerNorm2d(nn.Module):
    def __init__(self, channels: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(channels))
        self.bias = nn.Parameter(torch.zeros(channels))
        self.eps = eps

    def forward(self, x):
        mu = x.mean(1, keepdim=True)
        var = x.var(1, keepdim=True, unbiased=False)
        x = (x - mu) / torch.sqrt(var + self.eps)
        return x * self.weight[None, :, None, None] + self.bias[None, :, None, None]


class SimpleGate(nn.Module):
    def forward(self, x):
        x1, x2 = x.chunk(2, dim=1)
        return x1 * x2


class SimplifiedChannelAttention(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv2d(channels, channels, kernel_size=1)

    def forward(self, x):
        return x * self.conv(self.pool(x))


class NAFBlock(nn.Module):
    def __init__(self, c: int, dw_expand: int = 2, ffn_expand: int = 2):
        super().__init__()
        dw_channel = c * dw_expand
        self.norm1 = LayerNorm2d(c)
        self.conv1 = nn.Conv2d(c, dw_channel, kernel_size=1)
        self.conv2 = nn.Conv2d(dw_channel, dw_channel, kernel_size=3, padding=1, groups=dw_channel)
        self.sg1 = SimpleGate()
        self.sca = SimplifiedChannelAttention(dw_channel // 2)
        self.conv3 = nn.Conv2d(dw_channel // 2, c, kernel_size=1)

        ffn_channel = c * ffn_expand
        self.norm2 = LayerNorm2d(c)
        self.conv4 = nn.Conv2d(c, ffn_channel, kernel_size=1)
        self.sg2 = SimpleGate()
        self.conv5 = nn.Conv2d(ffn_channel // 2, c, kernel_size=1)

        self.beta = nn.Parameter(torch.zeros((1, c, 1, 1)))
        self.gamma = nn.Parameter(torch.zeros((1, c, 1, 1)))

    def forward(self, x):
        y = self.conv1(self.norm1(x))
        y = self.conv2(y)
        y = self.sg1(y)
        y = self.sca(y)
        y = self.conv3(y)
        x = x + y * self.beta

        y = self.conv4(self.norm2(x))
        y = self.sg2(y)
        y = self.conv5(y)
        x = x + y * self.gamma
        return x


class NAFNetSR(nn.Module):
    def __init__(self, img_channel: int = 1, width: int = 32,
                 enc_blk_nums=(1, 1, 1, 2), middle_blk_num: int = 2,
                 dec_blk_nums=(1, 1, 1, 1), upscale: int = 2):
        super().__init__()
        self.upscale = upscale
        self.intro = nn.Conv2d(img_channel, width, kernel_size=3, padding=1)

        self.encoders = nn.ModuleList()
        self.downs = nn.ModuleList()
        chan = width
        for num in enc_blk_nums:
            self.encoders.append(nn.Sequential(*[NAFBlock(chan) for _ in range(num)]))
            self.downs.append(nn.Conv2d(chan, chan * 2, kernel_size=2, stride=2))
            chan *= 2

        self.middle = nn.Sequential(*[NAFBlock(chan) for _ in range(middle_blk_num)])

        self.ups = nn.ModuleList()
        self.decoders = nn.ModuleList()
        for num in dec_blk_nums:
            self.ups.append(nn.Sequential(
                nn.Conv2d(chan, chan * 2, kernel_size=1, bias=False),
                nn.PixelShuffle(2),
            ))
            chan //= 2
            self.decoders.append(nn.Sequential(*[NAFBlock(chan) for _ in range(num)]))

        # fused pixel-shuffle upsampling head: single extra upscale at the very
        # end, after the U-Net has restored back to input resolution
        self.up_head = nn.Sequential(
            nn.Conv2d(chan, img_channel * (upscale ** 2), kernel_size=3, padding=1),
            nn.PixelShuffle(upscale),
        )

        self.padder_size = 2 ** len(enc_blk_nums)

    def _pad_to_multiple(self, x):
        _, _, h, w = x.shape
        mod = self.padder_size
        pad_h = (mod - h % mod) % mod
        pad_w = (mod - w % mod) % mod
        return F.pad(x, (0, pad_w, 0, pad_h), mode="reflect")

    def forward(self, inp: torch.Tensor) -> torch.Tensor:
        _, _, h, w = inp.shape
        x = self._pad_to_multiple(inp)

        x = self.intro(x)
        skips = []
        for encoder, down in zip(self.encoders, self.downs):
            x = encoder(x)
            skips.append(x)
            x = down(x)

        x = self.middle(x)

        for decoder, up, skip in zip(self.decoders, self.ups, reversed(skips)):
            x = up(x)
            x = x + skip
            x = decoder(x)

        out = self.up_head(x)

        base = F.interpolate(inp, scale_factor=self.upscale, mode="bilinear", align_corners=False)
        out = out[:, :, : h * self.upscale, : w * self.upscale] + base
        return out
