#!/usr/bin/env python3
"""Render icon.ico (Windows, 16-256 px) and icon.icns (macOS) from the app icon with Pillow.

    python make_icons.py OUT_DIR [SOURCE_PNG]

The source defaults to audio_cleaner_desktop/assets/icon.png. The PyInstaller spec reads the result from
$ACW_ICON_DIR.
"""
from __future__ import annotations

import sys
from pathlib import Path

from PIL import Image

DEFAULT_SOURCE = Path(__file__).resolve().parents[1] / "audio_cleaner_desktop" / "assets" / "icon.png"
ICO_SIZES = [(16, 16), (24, 24), (32, 32), (48, 48), (64, 64), (128, 128), (256, 256)]
ICNS_SIZES = [16, 32, 64, 128, 256, 512, 1024]


def square(img: Image.Image, size: int) -> Image.Image:
    img = img.convert("RGBA")
    side = max(img.size)
    canvas = Image.new("RGBA", (side, side), (0, 0, 0, 0))
    canvas.paste(img, ((side - img.width) // 2, (side - img.height) // 2))
    return canvas.resize((size, size), Image.Resampling.LANCZOS)


def main(argv: list[str]) -> int:
    if not 2 <= len(argv) <= 3:
        print(__doc__, file=sys.stderr)
        return 2
    out = Path(argv[1])
    src = Path(argv[2]) if len(argv) == 3 else DEFAULT_SOURCE
    if not src.is_file():
        print(f"make_icons: source icon not found: {src}", file=sys.stderr)
        return 1
    out.mkdir(parents=True, exist_ok=True)
    base = Image.open(src)
    if min(base.size) < 512:
        print(f"make_icons: warning: {src.name} is only {base.size[0]}x{base.size[1]}; large icon sizes are upscaled", file=sys.stderr)
    big = square(base, 1024)
    big.save(out / "icon.ico", format="ICO", sizes=ICO_SIZES)
    # Pillow writes an .icns with one image per entry of append_images / the sizes it knows.
    big.save(out / "icon.icns", format="ICNS", append_images=[square(base, s) for s in ICNS_SIZES[:-1]])
    big.save(out / "icon.png")
    print(f"icons written to {out}: icon.ico, icon.icns, icon.png")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
