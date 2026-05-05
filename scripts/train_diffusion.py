"""Train the diffusion model from configs/config.yaml."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import torch
from omegaconf import OmegaConf
from torch.utils.data import DataLoader

import wandb

from src.data import PokemonDataset
from src.diffusion.diffusion import Diffusion
from src.diffusion.ema import EMA
from src.diffusion.schedule import make_schedule
from src.diffusion.train import train
from src.diffusion.unet import UNet


def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--resume", type=str, default=None,
                        help="Path to a checkpoint .pt to resume training from")
    args = parser.parse_args()

    cfg = OmegaConf.load(PROJECT_ROOT / "configs" / "config.yaml")
    torch.manual_seed(cfg.training.seed)
    device = get_device()
    print(f"Using device: {device}")

    dataset = PokemonDataset(
        csv_path=PROJECT_ROOT / cfg.data.csv_path,
        images_dir=PROJECT_ROOT / cfg.data.images_dir,
        image_size=cfg.data.image_size,
    )
    loader = DataLoader(
        dataset=dataset,
        batch_size=cfg.training.batch_size,
        shuffle=True,
        num_workers=2,
        pin_memory=True,
        drop_last=True,
    )

    unet = UNet(
        image_size=cfg.data.image_size,
        base_channels=cfg.unet.base_channels,
        channel_mults=tuple(cfg.unet.channel_mults),
        num_res_blocks=cfg.unet.num_res_blocks,
        attn_resolutions=tuple(cfg.unet.attn_resolutions),
        time_emb_dim=cfg.unet.time_emb_dim,
        class_emb_dim=cfg.unet.class_emb_dim,
        num_classes=cfg.data.num_classes,
        num_groups=cfg.unet.num_groups,
    )
    schedule = make_schedule(T=cfg.diffusion.T, kind=cfg.diffusion.schedule)
    diffusion = Diffusion(
        model=unet,
        schedule=schedule,
        cfg_drop_prob=cfg.training.cfg_drop_prob,
        num_classes=cfg.data.num_classes,
    ).to(device)

    ema = EMA(diffusion.model, decay=cfg.training.ema_decay)
    optimizer = torch.optim.AdamW(diffusion.parameters(), lr=cfg.training.lr)

    start_epoch = 0
    start_global_step = 0
    if args.resume:
        ckpt = torch.load(args.resume, map_location=device)
        diffusion.model.load_state_dict(ckpt["model"])
        ema.ema_model.load_state_dict(ckpt["ema"])
        optimizer.load_state_dict(ckpt["optimizer"])
        start_epoch = ckpt["epoch"] + 1
        start_global_step = ckpt["global_step"]
        print(f"Loaded checkpoint from {args.resume} (resuming at epoch {start_epoch})")

    wandb.init(
        project="pokemon-genesis",
        config=OmegaConf.to_container(cfg, resolve=True),
    )

    train(
        diffusion=diffusion,
        ema=ema,
        optimizer=optimizer,
        loader=loader,
        device=device,
        num_epochs=cfg.training.epochs,
        image_size=cfg.data.image_size,
        grad_clip=cfg.training.grad_clip,
        sample_every_epochs=cfg.training.sample_every_epochs,
        sample_classes=list(cfg.sampling.classes_to_sample),
        samples_per_class=cfg.sampling.samples_per_class,
        sample_num_steps=cfg.sampling.num_steps,
        sample_guidance_scale=cfg.sampling.guidance_scale,
        checkpoint_every_epochs=cfg.training.checkpoint_every_epochs,
        checkpoint_dir=PROJECT_ROOT / cfg.paths.checkpoint_dir,
        start_epoch=start_epoch,
        start_global_step=start_global_step,
    )

    wandb.finish()


if __name__ == "__main__":
    main()
