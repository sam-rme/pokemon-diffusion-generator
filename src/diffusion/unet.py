"""Conditional U-Net predicting the noise added to a noisy image."""
from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from src.diffusion.blocks import (
    AttentionBlock,
    ClassEmbedding,
    CrossAttentionResBlock,
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
        cond_mode: str = "additive",
        selective_cond: bool = False,
    ) -> None:
        super().__init__()
        assert time_emb_dim == class_emb_dim, "time_emb_dim must equal class_emb_dim"
        if cond_mode not in ("additive", "cross_attn"):
            raise ValueError(f"unknown cond_mode: {cond_mode!r}")
        self.cond_mode = cond_mode
        self.selective_cond = selective_cond
        emb_dim = time_emb_dim

        self.time_emb = SinusoidalTimeEmbedding(time_emb_dim)
        self.time_mlp = nn.Sequential(
            nn.Linear(time_emb_dim, time_emb_dim * 4),
            nn.SiLU(),
            nn.Linear(time_emb_dim * 4, time_emb_dim),
        )
        self.class_emb = ClassEmbedding(num_classes, class_emb_dim)

        self.stem = nn.Conv2d(in_channels, base_channels, kernel_size=3, padding=1)

        def make_block(in_ch: int, out_ch: int) -> nn.Module:
            if cond_mode == "cross_attn":
                return CrossAttentionResBlock(in_ch, out_ch, time_emb_dim, class_emb_dim, num_groups)
            return ResBlock(in_ch, out_ch, emb_dim, num_groups)

        self.encoder_levels = nn.ModuleList()
        self.encoder_downs = nn.ModuleList()

        ch = base_channels
        resolution = image_size

        for mult in channel_mults:
            out_ch = base_channels * mult
            level_blocks = nn.ModuleList()
            for _ in range(num_res_blocks):
                level_blocks.append(make_block(ch, out_ch))
                ch = out_ch
            if resolution in attn_resolutions:
                level_blocks.append(AttentionBlock(ch, num_groups))
            self.encoder_levels.append(level_blocks)
            self.encoder_downs.append(Downsample(ch))
            resolution //= 2

        self.mid_block1 = make_block(ch, ch)
        self.mid_attn = AttentionBlock(ch, num_groups)
        self.mid_block2 = make_block(ch, ch)

        self.decoder_ups = nn.ModuleList()
        self.decoder_levels = nn.ModuleList()

        for level in reversed(range(len(channel_mults))):
            target_ch = base_channels * channel_mults[level]
            skip_ch = target_ch

            self.decoder_ups.append(Upsample(ch))
            resolution *= 2

            level_blocks = nn.ModuleList()
            for _ in range(num_res_blocks):
                level_blocks.append(make_block(ch + skip_ch, target_ch))
                ch = target_ch
            if resolution in attn_resolutions:
                level_blocks.append(AttentionBlock(ch, num_groups))
            self.decoder_levels.append(level_blocks)

        self.final_norm = nn.GroupNorm(num_groups, ch)
        self.final_conv = nn.Conv2d(ch, in_channels, kernel_size=3, padding=1)

    def _apply_block(
        self,
        block: nn.Module,
        h: Tensor,
        time_emb: Tensor,
        class_emb: Tensor | None,
    ) -> Tensor:
        """Dispatch to the right call signature based on the block type."""
        if isinstance(block, CrossAttentionResBlock):
            return block(h, time_emb, class_emb)
        if isinstance(block, ResBlock):
            emb = time_emb if class_emb is None else time_emb + class_emb
            return block(h, emb)
        return block(h)

    def forward(self, x: Tensor, t: Tensor, c: Tensor) -> Tensor:
        time_emb = self.time_mlp(self.time_emb(t))
        class_emb = self.class_emb(c)

        # In selective_cond mode, encoder + bottleneck see no class info.
        # The decoder always receives full conditioning.
        encoder_class = None if self.selective_cond else class_emb

        h = self.stem(x)

        skips: list[Tensor] = []
        for level, blocks in enumerate(self.encoder_levels):
            for block in blocks:
                if isinstance(block, AttentionBlock):
                    h = block(h)
                else:
                    h = self._apply_block(block, h, time_emb, encoder_class)
                    skips.append(h)
            h = self.encoder_downs[level](h)

        h = self._apply_block(self.mid_block1, h, time_emb, encoder_class)
        h = self.mid_attn(h)
        h = self._apply_block(self.mid_block2, h, time_emb, encoder_class)

        for level, blocks in enumerate(self.decoder_levels):
            h = self.decoder_ups[level](h)
            for block in blocks:
                if isinstance(block, AttentionBlock):
                    h = block(h)
                else:
                    h = torch.cat([h, skips.pop()], dim=1)
                    h = self._apply_block(block, h, time_emb, class_emb)

        return self.final_conv(F.silu(self.final_norm(h)))
