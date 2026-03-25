"""Resize images so the shorter side is a target size (default 1024) and save to a new directory."""

import argparse
from pathlib import Path

from PIL import Image, ImageOps

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png"}


def compress_image(input_path: Path, output_path: Path, short_side: int) -> None:
    img = ImageOps.exif_transpose(Image.open(input_path).convert("RGB"))
    w, h = img.size
    if min(w, h) <= short_side:
        img.save(output_path, quality=85)
        return
    if h < w:
        new_h = short_side
        new_w = int(round(w * short_side / h))
    else:
        new_w = short_side
        new_h = int(round(h * short_side / w))
    img = img.resize((new_w, new_h), Image.LANCZOS)
    img.save(output_path, quality=85)


def main() -> None:
    parser = argparse.ArgumentParser(description="Resize images for smaller file size.")
    parser.add_argument("input", type=Path, help="Input file or directory.")
    parser.add_argument("output", type=Path, help="Output directory.")
    parser.add_argument(
        "--short-side", type=int, default=1024, help="Target shorter side."
    )
    args = parser.parse_args()

    args.output.mkdir(parents=True, exist_ok=True)

    if args.input.is_dir():
        paths = sorted(
            p for p in args.input.iterdir() if p.suffix.lower() in IMAGE_EXTENSIONS
        )
    else:
        paths = [args.input]

    for path in paths:
        out = args.output / path.name
        compress_image(path, out, args.short_side)
        print(f"{path} -> {out}")


if __name__ == "__main__":
    main()
