"""Noise schedule (linear or cosine)."""
from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor


@dataclass
class NoiseSchedule:
    T: int
    betas: Tensor
    alphas: Tensor
    alphas_cumprod: Tensor
    sqrt_alphas_cumprod: Tensor
    sqrt_one_minus_alphas_cumprod: Tensor


def linear_betas(T: int) -> Tensor:
    return torch.linspace(1e-4, 0.02, T)


def cosine_betas(T: int, s: float = 0.008) -> Tensor:
    t = torch.linspace(0, T, T + 1)
    f = torch.cos(((t / T + s) / (1 + s)) * torch.pi / 2) ** 2
    alphas_cumprod = f / f[0]
    betas = 1 - alphas_cumprod[1:] / alphas_cumprod[:-1]
    return betas.clamp(max=0.999)


def make_schedule(T: int = 1000, kind: str = "cosine") -> NoiseSchedule:
    if kind == "cosine":
        betas = cosine_betas(T)
    elif kind == "linear":
        betas = linear_betas(T)
    else:
        raise ValueError(f"unknown kind: {kind!r} (expected 'linear' or 'cosine')")

    alphas = 1.0 - betas
    alphas_cumprod = torch.cumprod(alphas, dim=0)
    return NoiseSchedule(
        T=T,
        betas=betas,
        alphas=alphas,
        alphas_cumprod=alphas_cumprod,
        sqrt_alphas_cumprod=alphas_cumprod.sqrt(),
        sqrt_one_minus_alphas_cumprod=(1.0 - alphas_cumprod).sqrt(),
    )
