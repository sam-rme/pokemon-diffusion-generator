"""Diffusion model training loop."""
from __future__ import annotations

from pathlib import Path

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

import wandb

from src.diffusion.diffusion import Diffusion
from src.diffusion.ema import EMA


def _save_checkpoint(
    path: Path,
    diffusion: Diffusion,
    ema: EMA,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    global_step: int,
) -> None:
    torch.save(
        {
            "model": diffusion.model.state_dict(),
            "ema": ema.state_dict(),
            "optimizer": optimizer.state_dict(),
            "epoch": epoch,
            "global_step": global_step,
        },
        path,
    )


def _generate_grid(
    diffusion_for_buffers: Diffusion,
    ema: EMA,
    classes: list[int],
    samples_per_class: int,
    image_size: int,
    num_steps: int,
    guidance_scale: float,
    device: torch.device,
) -> torch.Tensor:
    live_model = diffusion_for_buffers.model
    diffusion_for_buffers.model = ema.ema_model
    diffusion_for_buffers.model.eval()
    try:
        c = torch.tensor(classes, device=device).repeat_interleave(samples_per_class)
        samples = diffusion_for_buffers.sample(
            c=c,
            image_size=image_size,
            num_steps=num_steps,
            guidance_scale=guidance_scale,
        )
    finally:
        diffusion_for_buffers.model = live_model
        diffusion_for_buffers.model.train()
    return samples


def _samples_to_wandb_images(samples: torch.Tensor) -> list[wandb.Image]:
    imgs = samples.detach().cpu().clamp(-1, 1)
    imgs = (imgs + 1) * 127.5
    imgs = imgs.permute(0, 2, 3, 1).byte().numpy()
    return [wandb.Image(img) for img in imgs]


def train(
    diffusion: Diffusion,
    ema: EMA,
    optimizer: torch.optim.Optimizer,
    loader: DataLoader,
    device: torch.device,
    *,
    num_epochs: int,
    image_size: int,
    grad_clip: float | None,
    sample_every_epochs: int,
    sample_classes: list[int],
    samples_per_class: int,
    sample_num_steps: int,
    sample_guidance_scale: float,
    checkpoint_every_epochs: int,
    checkpoint_dir: Path,
    start_epoch: int = 0,
    start_global_step: int = 0,
) -> None:
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    diffusion.train()
    global_step = start_global_step

    print(f"Training {num_epochs} epochs on {device} | {len(loader)} batches/epoch")
    if start_epoch > 0:
        print(f"Resuming from epoch {start_epoch}")
    print()

    for epoch in range(start_epoch, num_epochs):
        epoch_loss = 0.0
        pbar = tqdm(loader, desc=f"epoch {epoch:>3}/{num_epochs}", leave=False)
        for x, c in pbar:
            x = x.to(device)
            c = c.to(device)

            loss = diffusion.loss(x, c)

            optimizer.zero_grad()
            loss.backward()
            if grad_clip is not None:
                torch.nn.utils.clip_grad_norm_(diffusion.parameters(), grad_clip)
            optimizer.step()
            ema.update(diffusion.model)

            loss_value = loss.item()
            epoch_loss += loss_value
            pbar.set_postfix(loss=f"{loss_value:.4f}")

            wandb.log({"train/loss": loss_value, "epoch": epoch}, step=global_step)
            global_step += 1

        avg_loss = epoch_loss / len(loader)
        print(f"epoch {epoch:>3} | avg loss = {avg_loss:.4f}")
        wandb.log({"train/epoch_loss": avg_loss, "epoch": epoch}, step=global_step)

        is_last = epoch == num_epochs - 1

        if epoch % sample_every_epochs == 0 or is_last:
            samples = _generate_grid(
                diffusion_for_buffers=diffusion,
                ema=ema,
                classes=sample_classes,
                samples_per_class=samples_per_class,
                image_size=image_size,
                num_steps=sample_num_steps,
                guidance_scale=sample_guidance_scale,
                device=device,
            )
            wandb.log(
                {"samples": _samples_to_wandb_images(samples), "epoch": epoch},
                step=global_step,
            )

        if epoch % checkpoint_every_epochs == 0 or is_last:
            ckpt_path = checkpoint_dir / f"epoch_{epoch:04d}.pt"
            _save_checkpoint(
                path=ckpt_path,
                diffusion=diffusion,
                ema=ema,
                optimizer=optimizer,
                epoch=epoch,
                global_step=global_step,
            )
