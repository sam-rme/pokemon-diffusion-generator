"""Validate every PNG under data/images and strip broken iCCP chunks in place."""
from __future__ import annotations

import struct
from pathlib import Path

from PIL import Image


IMAGES_DIR = Path(__file__).parent.parent / "data" / "images"


def strip_iccp(path: Path) -> None:
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


def main():
    fixed = []
    for f in sorted(IMAGES_DIR.glob("*.png")):
        try:
            Image.open(f).convert("RGBA").load()
        except Exception:
            strip_iccp(f)
            fixed.append(f.name)
    print(f"Fixed {len(fixed)} files: {fixed}")


if __name__ == "__main__":
    main()
