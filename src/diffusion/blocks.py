"""Building blocks for the conditional U-Net."""
from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import Tensor, nn


class SinusoidalTimeEmbedding(nn.Module):
    def __init__(self, dim: int) -> None:
        super().__init__()
        self.dim = dim

    def forward(self, t: Tensor) -> Tensor:
        half_dim = self.dim // 2
        freqs = torch.exp(
            -math.log(10000) * torch.arange(half_dim, device=t.device) / (half_dim - 1)
        )
        args = t.unsqueeze(-1).float() * freqs
        return torch.cat([args.sin(), args.cos()], dim=-1)


class ClassEmbedding(nn.Module):
    def __init__(self, num_classes: int, dim: int) -> None:
        super().__init__()
        self.emb = nn.Embedding(num_classes + 1, dim)

    def forward(self, c: Tensor) -> Tensor:
        return self.emb(c)


class ResBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        emb_dim: int,
        num_groups: int = 8,
    ) -> None:
        super().__init__()
        self.norm1 = nn.GroupNorm(num_groups, in_channels)
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.emb_proj = nn.Linear(emb_dim, out_channels)
        self.norm2 = nn.GroupNorm(num_groups, out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.shortcut = (
            nn.Conv2d(in_channels, out_channels, kernel_size=1)
            if in_channels != out_channels
            else nn.Identity()
        )
        self.out_channels = out_channels

    def forward(self, x: Tensor, emb: Tensor) -> Tensor:
        h = self.conv1(F.silu(self.norm1(x)))
        h = h + self.emb_proj(emb).reshape(emb.shape[0], self.out_channels, 1, 1)
        h = self.conv2(F.silu(self.norm2(h)))
        return h + self.shortcut(x)


class CrossAttention(nn.Module):
    """Cross-attention from spatial positions (queries) to a small set of class tokens."""

    def __init__(
        self,
        channels: int,
        cond_dim: int,
        num_heads: int = 4,
        num_tokens: int = 4,
    ) -> None:
        super().__init__()
        self.num_heads = num_heads
        self.num_tokens = num_tokens
        self.head_dim = channels // num_heads
        self.token_proj = nn.Linear(cond_dim, num_tokens * channels)
        self.to_q = nn.Conv2d(channels, channels, kernel_size=1)
        self.to_kv = nn.Linear(channels, 2 * channels)
        self.proj_out = nn.Conv2d(channels, channels, kernel_size=1)

    def forward(self, x: Tensor, cond: Tensor) -> Tensor:
        B, C, H, W = x.shape
        N = H * W

        tokens = self.token_proj(cond).reshape(B, self.num_tokens, C)
        q = self.to_q(x).flatten(2).transpose(1, 2)
        k, v = self.to_kv(tokens).chunk(2, dim=-1)

        q = q.reshape(B, N, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.reshape(B, self.num_tokens, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.reshape(B, self.num_tokens, self.num_heads, self.head_dim).transpose(1, 2)

        scores = q @ k.transpose(-2, -1) / (self.head_dim ** 0.5)
        weights = scores.softmax(dim=-1)
        out = weights @ v

        out = out.transpose(1, 2).reshape(B, N, C).transpose(1, 2).reshape(B, C, H, W)
        return self.proj_out(out)


class CrossAttentionResBlock(nn.Module):
    """ResBlock with additive time injection and cross-attention class injection."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        time_emb_dim: int,
        class_emb_dim: int,
        num_groups: int = 8,
        num_tokens: int = 4,
        num_heads: int = 4,
    ) -> None:
        super().__init__()
        self.norm1 = nn.GroupNorm(num_groups, in_channels)
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.time_proj = nn.Linear(time_emb_dim, out_channels)
        self.norm2 = nn.GroupNorm(num_groups, out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.norm_attn = nn.GroupNorm(num_groups, out_channels)
        self.cross_attn = CrossAttention(out_channels, class_emb_dim, num_heads, num_tokens)
        self.shortcut = (
            nn.Conv2d(in_channels, out_channels, kernel_size=1)
            if in_channels != out_channels
            else nn.Identity()
        )
        self.out_channels = out_channels

    def forward(self, x: Tensor, time_emb: Tensor, class_emb: Tensor | None = None) -> Tensor:
        B = x.shape[0]
        h = self.conv1(F.silu(self.norm1(x)))
        h = h + self.time_proj(time_emb).reshape(B, self.out_channels, 1, 1)
        h = self.conv2(F.silu(self.norm2(h)))
        if class_emb is not None:
            h = h + self.cross_attn(self.norm_attn(h), class_emb)
        return h + self.shortcut(x)


class AttentionBlock(nn.Module):
    def __init__(self, channels: int, num_groups: int = 8) -> None:
        super().__init__()
        self.norm = nn.GroupNorm(num_groups, channels)
        self.qkv = nn.Conv2d(channels, 3 * channels, kernel_size=1)
        self.proj_out = nn.Conv2d(channels, channels, kernel_size=1)
        self.channels = channels

    def forward(self, x: Tensor) -> Tensor:
        B, C, H, W = x.shape
        h = self.norm(x)
        q, k, v = self.qkv(h).chunk(3, dim=1)

        q = q.flatten(2).transpose(1, 2)
        k = k.flatten(2).transpose(1, 2)
        v = v.flatten(2).transpose(1, 2)

        scores = q @ k.transpose(-2, -1) / (self.channels ** 0.5)
        weights = scores.softmax(dim=-1)
        out = weights @ v

        out = out.transpose(1, 2).reshape(B, C, H, W)
        return x + self.proj_out(out)


class Downsample(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.conv = nn.Conv2d(channels, channels, kernel_size=3, stride=2, padding=1)

    def forward(self, x: Tensor) -> Tensor:
        return self.conv(x)


class Upsample(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.conv = nn.Conv2d(channels, channels, kernel_size=3, padding=1)

    def forward(self, x: Tensor) -> Tensor:
        x = F.interpolate(x, scale_factor=2, mode="nearest")
        return self.conv(x)
