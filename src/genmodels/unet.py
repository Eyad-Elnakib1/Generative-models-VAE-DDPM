# Repository file: src/genmodels/unet.py
"""DDPM noise-prediction U-Net (Ho et al., 2020, Appendix B).

PixelCNN++-style U-Net on a Wide-ResNet backbone with GroupNorm, four feature-map
resolutions (32x32 -> 4x4), two residual blocks per resolution, self-attention at
16x16 and a sinusoidal timestep embedding added into every residual block. With the
CIFAR-10 settings (ch=128, ch_mult=(1, 2, 2, 2)) it has 35.7M parameters, matching
the paper.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from .nn_blocks import (AttnBlock, Downsample, ResnetBlock, Upsample, conv3x3,
                        linear, normalize, timestep_embedding)


class UNet(nn.Module):
    def __init__(self, in_ch=3, out_ch=3, ch=128, ch_mult=(1, 2, 2, 2), num_res_blocks=2,
                 attn_resolutions=(16,), dropout=0.1, resamp_with_conv=True, resolution=32):
        super().__init__()
        self.ch = ch
        temb_ch = ch * 4
        self.temb = nn.Sequential(linear(ch, temb_ch), nn.SiLU(), linear(temb_ch, temb_ch))
        self.conv_in = conv3x3(in_ch, ch)

        # Downsampling path; `skip_chs` records the channels of every stored activation.
        self.down = nn.ModuleList()
        skip_chs, cur_ch, res = [ch], ch, resolution
        for level, mult in enumerate(ch_mult):
            blocks, attns = nn.ModuleList(), nn.ModuleList()
            for _ in range(num_res_blocks):
                blocks.append(ResnetBlock(cur_ch, ch * mult, temb_ch, dropout))
                cur_ch = ch * mult
                attns.append(AttnBlock(cur_ch) if res in attn_resolutions else nn.Identity())
                skip_chs.append(cur_ch)
            stage = nn.Module()
            stage.blocks, stage.attns = blocks, attns
            if level != len(ch_mult) - 1:
                stage.downsample = Downsample(cur_ch, resamp_with_conv)
                skip_chs.append(cur_ch)
                res //= 2
            self.down.append(stage)

        self.mid_block1 = ResnetBlock(cur_ch, cur_ch, temb_ch, dropout)
        self.mid_attn = AttnBlock(cur_ch)
        self.mid_block2 = ResnetBlock(cur_ch, cur_ch, temb_ch, dropout)

        # Upsampling path: num_res_blocks + 1 blocks per level, each consuming one skip.
        self.up = nn.ModuleList()
        for level, mult in reversed(list(enumerate(ch_mult))):
            blocks, attns = nn.ModuleList(), nn.ModuleList()
            for _ in range(num_res_blocks + 1):
                blocks.append(ResnetBlock(cur_ch + skip_chs.pop(), ch * mult, temb_ch, dropout))
                cur_ch = ch * mult
                attns.append(AttnBlock(cur_ch) if res in attn_resolutions else nn.Identity())
            stage = nn.Module()
            stage.blocks, stage.attns = blocks, attns
            if level != 0:
                stage.upsample = Upsample(cur_ch, resamp_with_conv)
                res *= 2
            self.up.insert(0, stage)
        assert not skip_chs

        self.norm_out = normalize(cur_ch)
        self.conv_out = conv3x3(cur_ch, out_ch, init_scale=0.0)

    def forward(self, x, t):
        temb = self.temb(timestep_embedding(t, self.ch))
        hs = [self.conv_in(x)]
        for stage in self.down:
            for block, attn in zip(stage.blocks, stage.attns):
                hs.append(attn(block(hs[-1], temb)))
            if hasattr(stage, "downsample"):
                hs.append(stage.downsample(hs[-1]))

        h = self.mid_block2(self.mid_attn(self.mid_block1(hs[-1], temb)), temb)

        for stage in reversed(self.up):
            for block, attn in zip(stage.blocks, stage.attns):
                h = attn(block(torch.cat([h, hs.pop()], dim=1), temb))
            if hasattr(stage, "upsample"):
                h = stage.upsample(h)
        assert not hs
        return self.conv_out(F.silu(self.norm_out(h)))
