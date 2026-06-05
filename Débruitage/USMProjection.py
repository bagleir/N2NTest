#!/usr/bin/env python3
"""
Pipeline USM → Projection moyenne (sans prétraitement, sans masque).

Pipeline : vidéo brute → upscale 1024×1024 (Lanczos) → USM par frame → moyenne temporelle.

Usage :
  python USMProjection.py video.avi
  python USMProjection.py video.avi --output-dir resultats/
  python USMProjection.py dossier/
  python USMProjection.py dossier/ --output-dir resultats/
  python USMProjection.py video.avi --sigma 2.0 --strength 2.0 --target-size 1024

Dépendances :
  pip install opencv-python numpy scikit-image
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np
from skimage.filters import unsharp_mask as ski_usm


TARGET_SIZE  = 1024
USM_SIGMA    = 2.0
USM_STRENGTH = 2.0


def process_video(
    video_path: Path,
    output_dir: Path,
    sigma: float,
    strength: float,
    target_size: int,
) -> None:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Impossible d'ouvrir : {video_path}")

    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"  {video_path.name}  ({total} frames) …", end="", flush=True)

    acc   = np.zeros((target_size, target_size), dtype=np.float64)
    count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if frame.ndim == 3:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        frame = cv2.resize(frame, (target_size, target_size), interpolation=cv2.INTER_LANCZOS4)
        frame = ski_usm(frame, radius=sigma, amount=strength, preserve_range=True)
        acc  += np.clip(frame, 0, 255)
        count += 1

    cap.release()

    if count == 0:
        print("  ERREUR : aucune frame lue")
        return

    mean_img = np.clip(acc / count, 0, 255).astype(np.uint8)

    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / f"{video_path.stem}.png"
    cv2.imwrite(str(out_path), mean_img)
    print(f"  OK  →  {out_path.name}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Upscale → USM par frame → projection moyenne. Sans masque ni prétraitement.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Exemples :\n"
            "  python USMProjection.py video.avi\n"
            "  python USMProjection.py video.avi --output-dir resultats/\n"
            "  python USMProjection.py dossier/\n"
            "  python USMProjection.py dossier/ --output-dir resultats/\n"
        ),
    )
    parser.add_argument("input",
        help="Vidéo .avi ou dossier contenant des vidéos .avi.")
    parser.add_argument("--output-dir", default=None,
        help="Dossier de sortie (défaut : <input>_usm_projection/).")
    parser.add_argument("--sigma",       type=float, default=USM_SIGMA,
        help=f"Rayon Gaussien USM (défaut : {USM_SIGMA}).")
    parser.add_argument("--strength",    type=float, default=USM_STRENGTH,
        help=f"Intensité USM (défaut : {USM_STRENGTH}).")
    parser.add_argument("--target-size", type=int,   default=TARGET_SIZE,
        help=f"Résolution après upscale (défaut : {TARGET_SIZE}).")
    args = parser.parse_args()

    inp = Path(args.input)
    if not inp.exists():
        sys.exit(f"ERREUR : chemin introuvable : {inp}")

    out_dir = Path(args.output_dir) if args.output_dir else inp.parent / f"{inp.stem}_usm_projection"

    if inp.is_file():
        videos = [inp]
    else:
        videos = sorted(inp.glob("*.avi"))
        if not videos:
            sys.exit(f"ERREUR : aucune vidéo .avi dans {inp}")

    print(f"\n  Sortie  : {out_dir}")
    print(f"  USM     : sigma={args.sigma}  strength={args.strength}")
    print(f"  Upscale : {args.target_size}×{args.target_size}\n")

    failed = []
    for video_path in videos:
        try:
            process_video(video_path, out_dir, args.sigma, args.strength, args.target_size)
        except Exception as exc:
            print(f"  ERREUR {video_path.name} : {exc}")
            failed.append(video_path.name)

    print(f"\n  {len(videos) - len(failed)}/{len(videos)} vidéo(s) traitée(s).")
    if failed:
        print(f"  Échecs : {', '.join(failed)}")


if __name__ == "__main__":
    main()
