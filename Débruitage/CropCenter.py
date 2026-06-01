#!/usr/bin/env python3
"""
Découpe un carré centré sur une image et sauvegarde le résultat.

Usage :
  python CropCenter.py <image.png>
  python CropCenter.py <image.png> --size 128 --output crop.png
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2


def crop_center(input_path: str, output_path: str, size: int = 64) -> None:
    img = cv2.imread(input_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        sys.exit(f"ERREUR : impossible de lire {input_path}")

    H, W = img.shape
    cy, cx = H // 2, W // 2
    half = size // 2

    y0, y1 = max(cy - half, 0), min(cy + half, H)
    x0, x1 = max(cx - half, 0), min(cx + half, W)

    crop = img[y0:y1, x0:x1]
    cv2.imwrite(output_path, crop)

    print(f"  Source  : {input_path}  ({W}×{H})")
    print(f"  Crop    : [{x0}:{x1}, {y0}:{y1}]  → {crop.shape[1]}×{crop.shape[0]} px")
    print(f"  Sortie  : {output_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Découpe un carré centré sur une image.")
    parser.add_argument("input", help="Image d'entrée (.png / .tiff / .jpg).")
    parser.add_argument("--size",   type=int, default=64, help="Taille du carré en pixels (défaut : 64).")
    parser.add_argument("--output", default=None, help="Image de sortie (défaut : <input>_crop64.png).")
    args = parser.parse_args()

    inp = Path(args.input)
    out = Path(args.output) if args.output else inp.with_stem(f"{inp.stem}_crop{args.size}")
    crop_center(str(inp), str(out), size=args.size)


if __name__ == "__main__":
    main()
