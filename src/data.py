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
        expand_dual_type: bool = True,
    ) -> None:
        df = pd.read_csv(csv_path)

        if expand_dual_type and "type2" in df.columns:
            rows = []
            for _, row in df.iterrows():
                rows.append({"id": row["id"], "name": row["name"], "type": row["type1"]})
                if pd.notna(row["type2"]):
                    rows.append({"id": row["id"], "name": row["name"], "type": row["type2"]})
            self.df = pd.DataFrame(rows).reset_index(drop=True)
        else:
            self.df = df.rename(columns={"type1": "type"}).reset_index(drop=True)

        self.images_dir = images_dir
        self.transform = transforms.Compose([
            transforms.Lambda(composite_on_white),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5] * 3, std=[0.5] * 3),
        ])

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int]:
        row = self.df.iloc[idx]
        img_path = self.images_dir / f"{int(row['id']):04d}.png"
        img = Image.open(img_path).convert("RGBA")
        return self.transform(img), TYPE_TO_ID[row["type"]]
