"""Generate Pokemon samples from a trained checkpoint."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import torch
from matplotlib import pyplot as plt
from omegaconf import OmegaConf

from src.data import TYPES
from src.diffusion.diffusion import Diffusion
from src.diffusion.schedule import make_schedule
from src.diffusion.unet import UNet


def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", type=str, required=True, help="Path to checkpoint .pt")
    parser.add_argument("--classes", type=str, default="fire,grass,water,bug",
                        help="Comma-separated type names (e.g. 'fire,water')")
    parser.add_argument("--samples_per_class", type=int, default=4)
    parser.add_argument("--guidance_scale", type=float, default=3.0)
    parser.add_argument("--num_steps", type=int, default=200)
    parser.add_argument("--cond_mode", type=str, default="additive",
                        help="'additive' (run 1/2) or 'cross_attn' (run 3)")
    parser.add_argument("--selective_cond", action="store_true")
    parser.add_argument("--out", type=str, default="generated.png")
    args = parser.parse_args()

    cfg = OmegaConf.load(PROJECT_ROOT / "configs" / "config.yaml")
    device = get_device()
    print(f"Using device: {device}")

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
        cond_mode=args.cond_mode,
        selective_cond=args.selective_cond,
    ).to(device)

    ckpt = torch.load(args.ckpt, map_location=device)
    unet.load_state_dict(ckpt["ema"])
    unet.eval()
    print(f"Loaded checkpoint (epoch {ckpt.get('epoch', '?')})")

    schedule = make_schedule(T=cfg.diffusion.T, kind=cfg.diffusion.schedule)
    diffusion = Diffusion(unet, schedule, num_classes=cfg.data.num_classes).to(device)

    class_names = [name.strip() for name in args.classes.split(",")]
    type_to_id = {t: i for i, t in enumerate(TYPES)}
    class_ids = [type_to_id[name] for name in class_names]
    print(f"Generating for: {class_names} (ids: {class_ids})")

    fig, axes = plt.subplots(
        len(class_ids),
        args.samples_per_class,
        figsize=(args.samples_per_class * 2.5, len(class_ids) * 2.5),
        squeeze=False,
    )
    for row, (cls, name) in enumerate(zip(class_ids, class_names)):
        c = torch.full((args.samples_per_class,), cls, device=device)
        samples = diffusion.sample(
            c=c,
            image_size=cfg.data.image_size,
            num_steps=args.num_steps,
            guidance_scale=args.guidance_scale,
        )
        for col, s in enumerate(samples):
            img = (s.clamp(-1, 1).permute(1, 2, 0).cpu().numpy() + 1) / 2
            axes[row, col].imshow(img)
            axes[row, col].axis("off")
            if col == 0:
                axes[row, col].set_ylabel(name, rotation=0, labelpad=40, fontsize=11)

    plt.tight_layout()
    plt.savefig(args.out, dpi=120, bbox_inches="tight")
    print(f"Saved to {args.out}")


if __name__ == "__main__":
    main()
