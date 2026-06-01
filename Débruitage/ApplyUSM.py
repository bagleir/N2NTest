#!/usr/bin/env python3
"""
Applique un filtre Unsharp Mask (USM) sur toutes les frames d'une vidéo et sauvegarde le résultat.

Usage :
  python ApplyUSM.py <video.avi>
  python ApplyUSM.py <video.avi> --sigma 2.0 --strength 2.0 --output sortie.avi
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np
from skimage.filters import unsharp_mask as ski_usm


def apply_usm_video(
    input_path: str,
    output_path: str,
    sigma: float = 2.0,
    strength: float = 2.0,
) -> None:
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        sys.exit(f"ERREUR : impossible d'ouvrir {input_path}")

    fps    = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total  = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    fourcc = cv2.VideoWriter_fourcc(*"XVID")
    writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height), isColor=False)

    print(f"  Entrée  : {input_path}  ({total} frames, {width}×{height}, {fps:.1f} fps)")
    print(f"  Sortie  : {output_path}")
    print(f"  USM     : sigma={sigma}  strength={strength}")
    print()

    count = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if frame.ndim == 3:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        sharpened = ski_usm(frame, radius=sigma, amount=strength, preserve_range=True)
        out = np.clip(sharpened, 0, 255).astype(np.uint8)
        writer.write(out)

        count += 1
        pct = count / max(total, 1)
        bar = "█" * int(45 * pct) + "░" * (45 - int(45 * pct))
        print(f"\r  [{bar}] {count}/{total}", end="", flush=True)

    print()
    cap.release()
    writer.release()
    print(f"\n  Terminé — {count} frames traitées → {output_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Applique USM sur toutes les frames d'une vidéo.")
    parser.add_argument("input", help="Vidéo d'entrée (.avi).")
    parser.add_argument("--output", default=None, help="Vidéo de sortie (défaut : <input>_usm.avi).")
    parser.add_argument("--sigma",    type=float, default=2.0, help="Rayon Gaussien (défaut : 2.0).")
    parser.add_argument("--strength", type=float, default=2.0, help="Intensité USM (défaut : 2.0).")
    args = parser.parse_args()

    inp = Path(args.input)
    out = Path(args.output) if args.output else inp.with_stem(inp.stem + "_usm")

    apply_usm_video(str(inp), str(out), sigma=args.sigma, strength=args.strength)


if __name__ == "__main__":
    main()
