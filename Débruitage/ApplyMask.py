#!/usr/bin/env python3
"""
Applique mask.png sur toutes les frames d'une vidéo (pixels hors masque → 0).

Usage :
  python ApplyMask.py <video.avi> <mask.png>
  python ApplyMask.py <video.avi> <mask.png> --output sortie.avi
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from mask_detection import load_mask


def apply_mask_video(input_path: str, mask_path: str, output_path: str) -> None:
    mask = load_mask(mask_path)
    mask_bool = mask.astype(bool)

    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        sys.exit(f"ERREUR : impossible d'ouvrir {input_path}")

    fps    = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total  = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    fourcc = cv2.VideoWriter_fourcc(*"XVID")
    writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height), isColor=False)

    print(f"  Entrée : {input_path}  ({total} frames, {width}×{height})")
    print(f"  Masque : {mask_path}")
    print(f"  Sortie : {output_path}\n")

    count = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if frame.ndim == 3:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        out = np.zeros_like(frame)
        out[mask_bool] = frame[mask_bool]
        writer.write(out)
        count += 1
        pct = count / max(total, 1)
        bar = "█" * int(45 * pct) + "░" * (45 - int(45 * pct))
        print(f"\r  [{bar}] {count}/{total}", end="", flush=True)

    print()
    cap.release()
    writer.release()
    print(f"\n  Terminé — {count} frames → {output_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Applique mask.png sur toutes les frames d'une vidéo.")
    parser.add_argument("input",  help="Vidéo d'entrée (.avi).")
    parser.add_argument("mask",   help="Chemin vers mask.png.")
    parser.add_argument("--output", default=None, help="Vidéo de sortie (défaut : <input>_masked.avi).")
    args = parser.parse_args()

    inp = Path(args.input)
    out = Path(args.output) if args.output else inp.with_stem(inp.stem + "_masked")
    apply_mask_video(str(inp), args.mask, str(out))


if __name__ == "__main__":
    main()
