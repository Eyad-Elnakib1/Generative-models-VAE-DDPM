# Repository file: src/genmodels/nn_blocks.py
"""Building blocks shared by the DDPM U-Net and the convolutional VAE.

The blocks follow the official DDPM implementation (Ho et al., 2020,
github.com/hojonathanho/diffusion, `diffusion_tf/models/unet.py`): a Wide-ResNet
style residual block with GroupNorm + Swish, single-head self-attention, and
TF-style "fan_avg" variance-scaling initialisation (with near-zero init for the
last layer of every residual branch).
"""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F


def default_init_(module: nn.Module, scale: float = 1.0) -> nn.Module:
    """TF `variance_scaling(scale, 'fan_avg', 'uniform')` == Xavier-uniform with gain sqrt(scale)."""
    scale = max(scale, 1e-10)  # the official code uses 1e-10 for "zero" init
    nn.init.xavier_uniform_(module.weight, gain=math.sqrt(scale))
    if module.bias is not None:
        nn.init.zeros_(module.bias)
    return module


def conv3x3(in_ch, out_ch, stride=1, init_scale=1.0):
    return default_init_(nn.Conv2d(in_ch, out_ch, 3, stride=stride, padding=1), init_scale)


def nin(in_ch, out_ch, init_scale=1.0):
    """Network-in-network == 1x1 convolution."""
    return default_init_(nn.Conv2d(in_ch, out_ch, 1), init_scale)


def linear(in_f, out_f, init_scale=1.0):
    return default_init_(nn.Linear(in_f, out_f), init_scale)


def normalize(ch: int) -> nn.GroupNorm:
    return nn.GroupNorm(num_groups=32, num_channels=ch, eps=1e-6)


def timestep_embedding(t: torch.Tensor, dim: int) -> torch.Tensor:
    """Transformer sinusoidal embedding, as in the official `get_timestep_embedding`."""
    assert dim % 2 == 0
    half = dim // 2
    freqs = torch.exp(-math.log(10000) * torch.arange(half, device=t.device, dtype=torch.float32) / (half - 1))
    args = t.float()[:, None] * freqs[None]
    return torch.cat([torch.sin(args), torch.cos(args)], dim=1)


class Downsample(nn.Module):
    def __init__(self, ch: int, with_conv: bool = True):
        super().__init__()
        self.with_conv = with_conv
        if with_conv:
            # TF 'SAME' padding for a stride-2 3x3 conv pads (0, 1) on each spatial axis.
            self.conv = default_init_(nn.Conv2d(ch, ch, 3, stride=2, padding=0))

    def forward(self, x):
        if self.with_conv:
            return self.conv(F.pad(x, (0, 1, 0, 1)))
        return F.avg_pool2d(x, 2)


class Upsample(nn.Module):
    def __init__(self, ch: int, with_conv: bool = True):
        super().__init__()
        self.conv = conv3x3(ch, ch) if with_conv else None

    def forward(self, x):
        x = F.interpolate(x, scale_factor=2.0, mode="nearest")
        return self.conv(x) if self.conv is not None else x


class ResnetBlock(nn.Module):
    """GN -> Swish -> conv -> (+ time embedding) -> GN -> Swish -> dropout -> conv, plus skip.

    `temb_ch=0` gives the unconditional variant used by the VAE.
    """

    def __init__(self, in_ch: int, out_ch: int, temb_ch: int = 0, dropout: float = 0.0):
        super().__init__()
        self.norm1 = normalize(in_ch)
        self.conv1 = conv3x3(in_ch, out_ch)
        self.temb_proj = linear(temb_ch, out_ch) if temb_ch > 0 else None
        self.norm2 = normalize(out_ch)
        self.dropout = nn.Dropout(dropout)
        self.conv2 = conv3x3(out_ch, out_ch, init_scale=0.0)
        self.skip = nin(in_ch, out_ch) if in_ch != out_ch else nn.Identity()

    def forward(self, x, temb=None):
        h = self.conv1(F.silu(self.norm1(x)))
        if self.temb_proj is not None:
            h = h + self.temb_proj(F.silu(temb))[:, :, None, None]
        h = self.conv2(self.dropout(F.silu(self.norm2(h))))
        return self.skip(x) + h


class AttnBlock(nn.Module):
    """Single-head spatial self-attention with a residual connection."""

    def __init__(self, ch: int):
        super().__init__()
        self.norm = normalize(ch)
        self.q, self.k, self.v = nin(ch, ch), nin(ch, ch), nin(ch, ch)
        self.proj_out = nin(ch, ch, init_scale=0.0)

    def forward(self, x):
        B, C, H, W = x.shape
        h = self.norm(x)
        q = self.q(h).reshape(B, C, H * W).transpose(1, 2)  # B, HW, C
        k = self.k(h).reshape(B, C, H * W).transpose(1, 2)
        v = self.v(h).reshape(B, C, H * W).transpose(1, 2)
        # Same maths as softmax(q k^T / sqrt(C)) v, using the fused kernel when available.
        h = F.scaled_dot_product_attention(q[:, None], k[:, None], v[:, None])[:, 0]
        h = h.transpose(1, 2).reshape(B, C, H, W)
        return x + self.proj_out(h)
