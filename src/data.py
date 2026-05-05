"""Pokemon dataset."""
from __future__ import annotations

from pathlib import Path

import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms


TYPES = (
    "bug", "dark", "dragon", "electric", "fairy", "fighting", "fire",
    "flying", "ghost", "grass", "ground", "ice", "normal", "poison",
    "psychic", "rock", "steel", "water",
)
TYPE_TO_ID = {t: i for i, t in enumerate(TYPES)}


def composite_on_white(img_rgba: Image.Image) -> Image.Image:
    background = Image.new("RGB", img_rgba.size, (255, 255, 255))
    background.paste(img_rgba, mask=img_rgba.split()[3])
    return background


class PokemonDataset(Dataset):
    def __init__(
        self,
        csv_path: Path,
        images_dir: Path,
        image_size: int = 96,
    ) -> None:
        self.df = pd.read_csv(csv_path)
        self.images_dir = images_dir
        self.transform = transforms.Compose([
            transforms.Lambda(composite_on_white),
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5] * 3, std=[0.5] * 3),
        ])

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int]:
        row = self.df.iloc[idx]
        img_path = self.images_dir / f"{int(row.id):04d}.png"
        img = Image.open(img_path).convert("RGBA")
        return self.transform(img), TYPE_TO_ID[row.type1]
