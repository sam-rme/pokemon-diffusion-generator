"""Compute evaluation metrics for the trained pipeline."""
import numpy as np
from PIL import Image
from pathlib import Path

IMAGES_DIR = Path('data/images')

dark_files = []
for f in sorted(IMAGES_DIR.glob('*.png')):
    img = np.array(Image.open(f).convert('RGBA'))
    alpha = img[..., 3]
    visible = alpha > 0
    if visible.sum() == 0:
        dark_files.append(f.name)
        continue
    rgb_mean = img[visible][..., :3].mean()
    if rgb_mean < 50:  # mostly dark
        dark_files.append((f.name, rgb_mean))

print(f"Dark/placeholder sprites: {len(dark_files)}")
for x in dark_files[:30]:
    print(f"  {x}")
