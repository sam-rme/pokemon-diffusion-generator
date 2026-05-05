"""Diffusion forward process, training loss, and DDIM sampling with CFG."""
from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from src.diffusion.schedule import NoiseSchedule
from src.diffusion.unet import UNet


def _extract(arr: Tensor, t: Tensor, x_shape: tuple[int, ...]) -> Tensor:
    out = arr.gather(0, t)
    return out.reshape(t.shape[0], *([1] * (len(x_shape) - 1)))


class Diffusion(nn.Module):
    def __init__(
        self,
        model: UNet,
        schedule: NoiseSchedule,
        cfg_drop_prob: float = 0.1,
        num_classes: int = 18,
    ) -> None:
        super().__init__()
        self.model = model
        self.cfg_drop_prob = cfg_drop_prob
        self.num_classes = num_classes
        self.T = schedule.T

        self.register_buffer("betas", schedule.betas)
        self.register_buffer("alphas_cumprod", schedule.alphas_cumprod)
        self.register_buffer("sqrt_alphas_cumprod", schedule.sqrt_alphas_cumprod)
        self.register_buffer("sqrt_one_minus_alphas_cumprod", schedule.sqrt_one_minus_alphas_cumprod)

    def q_sample(self, x_0: Tensor, t: Tensor, noise: Tensor) -> Tensor:
        sqrt_ac = _extract(self.sqrt_alphas_cumprod, t, x_0.shape)
        sqrt_one_ac = _extract(self.sqrt_one_minus_alphas_cumprod, t, x_0.shape)
        return sqrt_ac * x_0 + sqrt_one_ac * noise

    def loss(self, x_0: Tensor, c: Tensor) -> Tensor:
        B = x_0.shape[0]
        device = x_0.device

        t = torch.randint(0, self.T, (B,), device=device)
        noise = torch.randn_like(x_0)
        x_t = self.q_sample(x_0, t, noise)

        mask = torch.rand(B, device=device) < self.cfg_drop_prob
        c_dropped = torch.where(mask, self.num_classes, c)

        eps_pred = self.model(x_t, t, c_dropped)
        return F.mse_loss(eps_pred, noise)

    @torch.no_grad()
    def sample(
        self,
        c: Tensor,
        image_size: int = 96,
        num_steps: int = 50,
        guidance_scale: float = 3.0,
    ) -> Tensor:
        B = c.shape[0]
        device = c.device

        timesteps = torch.linspace(self.T - 1, 0, num_steps + 1).long().to(device)
        x = torch.randn(B, 3, image_size, image_size, device=device)
        null_c = torch.full_like(c, self.num_classes)

        for t_now, t_next in zip(timesteps[:-1], timesteps[1:]):
            t_batch = t_now.expand(B)

            eps_cond = self.model(x, t_batch, c)
            eps_uncond = self.model(x, t_batch, null_c)
            eps = eps_uncond + guidance_scale * (eps_cond - eps_uncond)

            ac_t = _extract(self.alphas_cumprod, t_batch, x.shape)
            ac_next = _extract(self.alphas_cumprod, t_next.expand(B), x.shape)
            x0_pred = (x - (1 - ac_t).sqrt() * eps) / ac_t.sqrt()
            x = ac_next.sqrt() * x0_pred + (1 - ac_next).sqrt() * eps

        return x
