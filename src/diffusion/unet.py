"""Conditional U-Net predicting the noise added to a noisy image."""
from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from src.diffusion.blocks import (
    AttentionBlock,
    ClassEmbedding,
    Downsample,
    ResBlock,
    SinusoidalTimeEmbedding,
    Upsample,
)


class UNet(nn.Module):
    def __init__(
        self,
        image_size: int = 96,
        in_channels: int = 3,
        base_channels: int = 64,
        channel_mults: tuple[int, ...] = (1, 2, 4, 4),
        num_res_blocks: int = 3,
        attn_resolutions: tuple[int, ...] = (12, 6),
        time_emb_dim: int = 256,
        class_emb_dim: int = 256,
        num_classes: int = 18,
        num_groups: int = 8,
    ) -> None:
        super().__init__()
        assert time_emb_dim == class_emb_dim, "time_emb_dim must equal class_emb_dim"
        emb_dim = time_emb_dim

        self.time_emb = SinusoidalTimeEmbedding(time_emb_dim)
        self.time_mlp = nn.Sequential(
            nn.Linear(time_emb_dim, time_emb_dim * 4),
            nn.SiLU(),
            nn.Linear(time_emb_dim * 4, time_emb_dim),
        )
        self.class_emb = ClassEmbedding(num_classes, class_emb_dim)

        self.stem = nn.Conv2d(in_channels, base_channels, kernel_size=3, padding=1)

        self.encoder_levels = nn.ModuleList()
        self.encoder_downs = nn.ModuleList()

        ch = base_channels
        resolution = image_size

        for mult in channel_mults:
            out_ch = base_channels * mult
            level_blocks = nn.ModuleList()
            for _ in range(num_res_blocks):
                level_blocks.append(ResBlock(ch, out_ch, emb_dim, num_groups))
                ch = out_ch
            if resolution in attn_resolutions:
                level_blocks.append(AttentionBlock(ch, num_groups))
            self.encoder_levels.append(level_blocks)
            self.encoder_downs.append(Downsample(ch))
            resolution //= 2

        self.mid_block1 = ResBlock(ch, ch, emb_dim, num_groups)
        self.mid_attn = AttentionBlock(ch, num_groups)
        self.mid_block2 = ResBlock(ch, ch, emb_dim, num_groups)

        self.decoder_ups = nn.ModuleList()
        self.decoder_levels = nn.ModuleList()

        for level in reversed(range(len(channel_mults))):
            target_ch = base_channels * channel_mults[level]
            skip_ch = target_ch

            self.decoder_ups.append(Upsample(ch))
            resolution *= 2

            level_blocks = nn.ModuleList()
            for _ in range(num_res_blocks):
                level_blocks.append(ResBlock(ch + skip_ch, target_ch, emb_dim, num_groups))
                ch = target_ch
            if resolution in attn_resolutions:
                level_blocks.append(AttentionBlock(ch, num_groups))
            self.decoder_levels.append(level_blocks)

        self.final_norm = nn.GroupNorm(num_groups, ch)
        self.final_conv = nn.Conv2d(ch, in_channels, kernel_size=3, padding=1)

    def forward(self, x: Tensor, t: Tensor, c: Tensor) -> Tensor:
        emb = self.time_mlp(self.time_emb(t)) + self.class_emb(c)
        h = self.stem(x)

        skips: list[Tensor] = []
        for level, blocks in enumerate(self.encoder_levels):
            for block in blocks:
                if isinstance(block, ResBlock):
                    h = block(h, emb)
                    skips.append(h)
                else:
                    h = block(h)
            h = self.encoder_downs[level](h)

        h = self.mid_block1(h, emb)
        h = self.mid_attn(h)
        h = self.mid_block2(h, emb)

        for level, blocks in enumerate(self.decoder_levels):
            h = self.decoder_ups[level](h)
            for block in blocks:
                if isinstance(block, ResBlock):
                    h = torch.cat([h, skips.pop()], dim=1)
                    h = block(h, emb)
                else:
                    h = block(h)

        return self.final_conv(F.silu(self.final_norm(h)))
