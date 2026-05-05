"""Download Pokemon sprites and metadata from PokeAPI."""
from __future__ import annotations

import struct
from pathlib import Path
from typing import Optional

import pandas as pd
import requests
from PIL import Image
from tqdm import tqdm


POKEAPI_BASE = "https://pokeapi.co/api/v2"
DATA_DIR = Path(__file__).parent.parent / "data"
IMAGES_DIR = DATA_DIR / "images"
METADATA_PATH = DATA_DIR / "metadata.csv"
N_POKEMON = 1025


def fetch_pokemon_data(pokemon_id: int) -> Optional[dict]:
    url = f"{POKEAPI_BASE}/pokemon/{pokemon_id}"
    response = requests.get(url, timeout=10)
    if response.status_code != 200:
        return None
    data = response.json()
    return {
        "id": pokemon_id,
        "name": data["name"],
        "type1": data["types"][0]["type"]["name"],
        "sprite_url": data["sprites"]["front_default"],
    }


def _strip_iccp(path: Path) -> None:
    # A few PokeAPI PNGs ship with a broken iCCP chunk; rewrite the file
    # without that chunk so PIL stops refusing to open it.
    data = path.read_bytes()
    if data[:8] != b"\x89PNG\r\n\x1a\n":
        return
    out = data[:8]
    i = 8
    while i < len(data):
        length = struct.unpack(">I", data[i:i + 4])[0]
        chunk_type = data[i + 4:i + 8]
        chunk_total = 4 + 4 + length + 4
        if chunk_type != b"iCCP":
            out += data[i:i + chunk_total]
        i += chunk_total
    path.write_bytes(out)


def download_sprite(url: str, out_path: Path) -> bool:
    response = requests.get(url, timeout=10)
    if response.status_code != 200:
        return False
    out_path.write_bytes(response.content)
    try:
        Image.open(out_path).load()
    except Exception:
        _strip_iccp(out_path)
    return True


def main():
    IMAGES_DIR.mkdir(parents=True, exist_ok=True)

    metadata_rows: list[dict] = []
    for pokemon_id in tqdm(range(1, N_POKEMON + 1), desc="Pokemon"):
        info = fetch_pokemon_data(pokemon_id)
        if info is None or info["sprite_url"] is None:
            continue
        out_path = IMAGES_DIR / f"{pokemon_id:04d}.png"
        if download_sprite(info["sprite_url"], out_path):
            metadata_rows.append({
                "id": info["id"],
                "name": info["name"],
                "type1": info["type1"],
            })

    df = pd.DataFrame(metadata_rows)
    df.to_csv(METADATA_PATH, index=False)
    print(f"\n{len(df)} Pokemon saved to {METADATA_PATH}")
    print(f"Images stored under {IMAGES_DIR}")


if __name__ == "__main__":
    main()
