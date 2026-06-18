#!/usr/bin/env python3
"""Coupes verticales d'une image de projection + tracé des profils d'intensité.

Exemples :
    python coupes_verticales.py proj.png
    python coupes_verticales.py proj.png --x 0.3 0.5 0.7 --band 2 -o profils.png
    python coupes_verticales.py proj.png --x 200 512 800   # positions en pixels
"""
import argparse
import numpy as np
import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def vertical_profiles(img: np.ndarray, x_positions, band: int = 2):
    """Retourne {x: profil} ; profil = moyenne des colonnes [x-band, x+band]."""
    H, W = img.shape
    out = {}
    for x in x_positions:
        x = int(round(x * W)) if 0 < x <= 1 else int(round(x))  # fraction ou pixel
        x = max(0, min(W - 1, x))
        out[x] = img[:, max(0, x - band):x + band + 1].mean(axis=1)
    return out


def plot_vertical_cuts(image_path, x_positions=(0.3, 0.5, 0.7), band=2,
                       out_path="coupes_verticales.png"):
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(image_path)
    img = img.astype(np.float32)
    H, W = img.shape

    profiles = vertical_profiles(img, x_positions, band)
    colors = plt.cm.tab10(np.linspace(0, 1, max(3, len(profiles))))

    fig = plt.figure(figsize=(15, 6))
    gs = fig.add_gridspec(1, 2, width_ratios=[1, 1.3], wspace=0.18)

    axI = fig.add_subplot(gs[0])
    axI.imshow(img, cmap="gray", vmin=0, vmax=255)
    axI.set_title("Projection + coupes verticales")
    axI.axis("off")

    axP = fig.add_subplot(gs[1])
    y = np.arange(H)
    for (x, prof), c in zip(profiles.items(), colors):
        axI.axvline(x, color=c, lw=1.5)
        axP.plot(prof, y, color=c, lw=1.2, label=f"x={x} ({100 * x // W}%)")
    axP.invert_yaxis()
    axP.set_xlabel("Intensité (0-255)")
    axP.set_ylabel("Position verticale (ligne)")
    axP.set_title("Profils d'intensité (coupes verticales)")
    axP.legend(loc="lower right")
    axP.grid(alpha=0.3)

    plt.savefig(out_path, dpi=95, bbox_inches="tight")
    plt.close()
    print(f"-> {out_path}")
    return profiles


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("image")
    ap.add_argument("--x", type=float, nargs="+", default=[0.3, 0.5, 0.7],
                    help="Positions des coupes : fractions (0-1) ou pixels (>1)")
    ap.add_argument("--band", type=int, default=2,
                    help="Demi-largeur de moyennage (px) pour lisser le profil")
    ap.add_argument("-o", "--out", default="coupes_verticales.png")
    a = ap.parse_args()
    plot_vertical_cuts(a.image, a.x, a.band, a.out)