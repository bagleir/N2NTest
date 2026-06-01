#!/usr/bin/env python3
"""
USM agressif par frame  →  projection moyenne/médiane  →  upscale 1024×1024.

Pipeline :
  1. Lire la vidéo prétraitée en streaming (par chunks).
  2. Appliquer skimage.filters.unsharp_mask sur chaque frame
     (strength=2, radius configurable — défaut 2.0 pour L≈1280 +882%).
  3. Écrire chaque frame USM dans une vidéo AVI de sortie (même FPS, même codec).
  4. Accumuler :
       mean   → somme float64 cumulée  (peu de mémoire, exact)
       median → réservoir de 200 frames échantillonnées aléatoirement
  5. Sauvegarder les deux projections en PNG + TIFF à la résolution native.
  6. Upscaler à 1024×1024 par interpolation Lanczos (cv2.INTER_LANCZOS4)
     et sauvegarder.

Usage :
  python ProjectionUSM.py <video.avi>
  python ProjectionUSM.py <video.avi> --mask mask.png --output-dir sorties/
  python ProjectionUSM.py <video.avi> --sigma 2.0 --strength 2.0
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import Iterator

import cv2
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).parent))
from mask_detection import load_mask
from UnsharpMask import apply_unsharp_mask


# ─────────────────────────────────────────────────────────────────────────────
# Paramètres par défaut
# ─────────────────────────────────────────────────────────────────────────────

USM_RADIUS   = 2.0   # rayon Gaussien (pixels) — calibrer via UnsharpMask.py --calibrate
USM_STRENGTH = 2.0   # intensité du rehaussement — résultat observé : L≈1280, +882%
CHUNK_SIZE   = 50    # frames par chunk  (50 × 512² × float32 ≈ 26 MB)
RESERVOIR    = 200   # taille réservoir pour la médiane
TARGET_SIZE  = 1024  # résolution finale après upscale (pixels)


# ─────────────────────────────────────────────────────────────────────────────
# Utilitaires internes
# ─────────────────────────────────────────────────────────────────────────────


def _progress(current: int, total: int, width: int = 45) -> None:
    pct = current / max(total, 1)
    bar = "█" * int(width * pct) + "░" * (width - int(width * pct))
    print(f"\r  [{bar}] {current}/{total} ({pct*100:.0f}%)", end="", flush=True)
    if current >= total:
        print()


def _count_frames(path: str | Path) -> int:
    cap = cv2.VideoCapture(str(path))
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    return max(n, 0)


def _iter_chunks(
    path: str | Path,
    chunk_size: int = CHUNK_SIZE,
) -> Iterator[tuple[np.ndarray, int]]:
    """Yield (chunk_uint8, frames_read_so_far).  Pas de chargement total en RAM."""
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise FileNotFoundError(f"Cannot open: {path}")
    buf: list[np.ndarray] = []
    count = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            if buf:
                yield np.stack(buf, axis=0), count
            break
        if frame.ndim == 3:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        buf.append(frame)
        count += 1
        if len(buf) == chunk_size:
            yield np.stack(buf, axis=0), count
            buf = []
    cap.release()


def _normalize(img: np.ndarray, mask: np.ndarray | None) -> np.ndarray:
    """Étire la dynamique [0,255] à l'intérieur du masque ; 0 en dehors."""
    out = np.zeros(img.shape, dtype=np.uint8)
    roi = mask.astype(bool) if mask is not None else np.ones(img.shape, dtype=bool)
    if not roi.any():
        return out
    vals = img[roi]
    lo, hi = float(vals.min()), float(vals.max())
    if hi > lo:
        scaled = (img.astype(np.float64) - lo) / (hi - lo) * 255.0
    else:
        scaled = np.zeros_like(img, dtype=np.float64)
    out[roi] = np.clip(scaled[roi], 0, 255).astype(np.uint8)
    return out


def _get_fps(path: str | Path) -> float:
    cap = cv2.VideoCapture(str(path))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    cap.release()
    return fps


def _save(img: np.ndarray, path: Path, tiff: bool = False) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), img)
    if tiff:
        cv2.imwrite(str(path.with_suffix(".tiff")), img)


# ─────────────────────────────────────────────────────────────────────────────
# Cœur du traitement
# ─────────────────────────────────────────────────────────────────────────────


def process(
    video_path: str,
    mask: np.ndarray | None,
    output_dir: str,
    sigma: float = USM_RADIUS,
    strength: float = USM_STRENGTH,
    chunk_size: int = CHUNK_SIZE,
    reservoir_size: int = RESERVOIR,
    target_size: int = TARGET_SIZE,
) -> dict:
    """
    Applique USM sur toutes les frames, calcule mean + médiane, upscale 1024×1024.

    Args:
        video_path:     Vidéo prétraitée (.avi).
        mask:           Masque binaire uint8 (H×W) ou None (traite tout le cadre).
        output_dir:     Dossier de sortie.
        sigma:          Rayon Gaussien pour skimage USM.
        strength:       Intensité USM (2.0 correspond à L≈1280 +882%).
        chunk_size:     Frames par chunk (mémoire).
        reservoir_size: Taille du réservoir pour la médiane.
        target_size:    Résolution de sortie après upscale (target_size × target_size).

    Returns:
        dict avec 'mean_native', 'median_native', 'mean_upscaled', 'median_upscaled'
        (chemins Path) et 'processing_time_s', 'n_frames'.
    """
    t0 = time.perf_counter()
    total = _count_frames(video_path)
    fps   = _get_fps(video_path)
    out   = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    p_video_usm = out / "video_usm.avi"
    fourcc = cv2.VideoWriter_fourcc(*"XVID")

    sum_acc:   np.ndarray | None = None   # float64 (H, W)
    reservoir: np.ndarray | None = None   # float32 (R, H, W)
    writer:    cv2.VideoWriter | None = None
    n_total  = 0
    rng = np.random.default_rng(seed=42)

    print(f"\n  USM  radius={sigma}  strength={strength}  →  "
          f"{total} frames  (chunks={chunk_size})")
    print(f"  Médiane : réservoir de {reservoir_size} frames aléatoires")
    print(f"  Vidéo USM → {p_video_usm}")
    print()

    for chunk_u8, count in _iter_chunks(video_path, chunk_size):
        n_chunk, H, W = chunk_u8.shape

        if sum_acc is None:
            sum_acc   = np.zeros((H, W), dtype=np.float64)
            reservoir = np.empty((reservoir_size, H, W), dtype=np.float32)
            writer    = cv2.VideoWriter(str(p_video_usm), fourcc, fps, (W, H), isColor=False)

        # ── USM sur chaque frame du chunk ─────────────────────────────────
        _mask = mask if mask is not None else np.ones((H, W), dtype=np.uint8) * 255
        usm_chunk = np.empty_like(chunk_u8, dtype=np.float32)
        for i in range(n_chunk):
            frame_usm = apply_unsharp_mask(chunk_u8[i], _mask, sigma=sigma, strength=strength)
            usm_chunk[i] = frame_usm.astype(np.float32)
            writer.write(frame_usm)

        # ── Accumulation mean ─────────────────────────────────────────────
        sum_acc += usm_chunk.sum(axis=0).astype(np.float64)

        # ── Réservoir pour médiane (Vitter Algorithm R) ───────────────────
        for i in range(n_chunk):
            g = n_total + i
            if g < reservoir_size:
                reservoir[g] = usm_chunk[i]
            else:
                j = int(rng.integers(0, g + 1))
                if j < reservoir_size:
                    reservoir[j] = usm_chunk[i]

        n_total += n_chunk
        _progress(count, total)

    _progress(total, total)
    if writer is not None:
        writer.release()

    # ── Calcul final ──────────────────────────────────────────────────────────
    actual_res = min(n_total, reservoir_size)
    print(f"\n  Calcul mean / médiane … ", end="", flush=True)

    mean_raw   = (sum_acc / n_total).astype(np.float32)
    median_raw = np.median(reservoir[:actual_res], axis=0).astype(np.float32)

    mean_img   = _normalize(mean_raw,   mask)
    median_img = _normalize(median_raw, mask)

    elapsed = time.perf_counter() - t0
    print(f"done  ({elapsed:.1f} s)")

    # ── Sauvegarde résolution native ──────────────────────────────────────────
    p_mean_nat   = out / "projection_mean_native.png"
    p_median_nat = out / "projection_median_native.png"
    _save(mean_img,   p_mean_nat,   tiff=True)
    _save(median_img, p_median_nat, tiff=True)

    # ── Upscale → 1024×1024 (Lanczos) ────────────────────────────────────────
    mean_up   = cv2.resize(mean_img,   (target_size, target_size),
                           interpolation=cv2.INTER_LANCZOS4)
    median_up = cv2.resize(median_img, (target_size, target_size),
                           interpolation=cv2.INTER_LANCZOS4)

    p_mean_up   = out / f"projection_mean_{target_size}px.png"
    p_median_up = out / f"projection_median_{target_size}px.png"
    _save(mean_up,   p_mean_up,   tiff=True)
    _save(median_up, p_median_up, tiff=True)

    # ── Grille de comparaison ─────────────────────────────────────────────────
    _comparison_figure(
        mean_nat   = mean_img,
        median_nat = median_img,
        mean_up    = mean_up,
        median_up  = median_up,
        sigma      = sigma,
        strength   = strength,
        n_frames   = n_total,
        target_size= target_size,
        output_path= out / "projection_comparison.png",
    )

    results = {
        "video_usm":      p_video_usm,
        "mean_native":    p_mean_nat,
        "median_native":  p_median_nat,
        "mean_upscaled":  p_mean_up,
        "median_upscaled":p_median_up,
        "n_frames":       n_total,
        "processing_time_s": round(elapsed, 2),
    }

    print(f"\n  ── Résultats ─────────────────────────────────────────────────────")
    print(f"  Frames traitées   : {n_total}")
    print(f"  Réservoir utilisé : {actual_res} / {reservoir_size}")
    print(f"  Temps total       : {elapsed:.1f} s")
    print(f"\n  Vidéo USM (toutes frames) → {p_video_usm}")
    print(f"\n  Projections natives ({H}×{W}) :")
    print(f"    Mean    PNG/TIFF  → {p_mean_nat} / .tiff")
    print(f"    Médiane PNG/TIFF  → {p_median_nat} / .tiff")
    print(f"\n  Upscalées ({target_size}×{target_size} Lanczos) :")
    print(f"    Mean    PNG/TIFF  → {p_mean_up} / .tiff")
    print(f"    Médiane PNG/TIFF  → {p_median_up} / .tiff")
    print(f"\n  Grille comparaison → {out}/projection_comparison.png")

    return results


# ─────────────────────────────────────────────────────────────────────────────
# Visualisation
# ─────────────────────────────────────────────────────────────────────────────


def _comparison_figure(
    mean_nat: np.ndarray,
    median_nat: np.ndarray,
    mean_up: np.ndarray,
    median_up: np.ndarray,
    sigma: float,
    strength: float,
    n_frames: int,
    target_size: int,
    output_path: Path,
) -> None:
    """
    Grille 2×2 :
      Col 0 — résolution native  |  Col 1 — upscalée 1024×1024
      Lig 0 — moyenne            |  Lig 1 — médiane
    """
    fig, axes = plt.subplots(
        2, 2,
        figsize=(11, 10),
        gridspec_kw={"hspace": 0.06, "wspace": 0.04},
    )
    panels = [
        (0, 0, mean_nat,   f"Mean  ({mean_nat.shape[0]}×{mean_nat.shape[1]})"),
        (0, 1, mean_up,    f"Mean upscalée ({target_size}×{target_size})"),
        (1, 0, median_nat, f"Médiane  ({median_nat.shape[0]}×{median_nat.shape[1]})"),
        (1, 1, median_up,  f"Médiane upscalée ({target_size}×{target_size})"),
    ]
    for r, c, img, title in panels:
        axes[r, c].imshow(img, cmap="gray", vmin=0, vmax=255, interpolation="nearest")
        axes[r, c].axis("off")
        axes[r, c].set_title(title, fontsize=9, fontweight="bold")

    plt.suptitle(
        f"Projection temporelle après USM par frame\n"
        f"USM : radius={sigma}  strength={strength}  —  {n_frames} frames traitées\n"
        f"Upscale : Lanczos  {mean_nat.shape[0]}×{mean_nat.shape[1]} → {target_size}×{target_size}",
        fontsize=10,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(str(output_path), dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Grille comparaison → {output_path}")


# ─────────────────────────────────────────────────────────────────────────────
# Preview frame du milieu
# ─────────────────────────────────────────────────────────────────────────────


def preview_midframe(
    video_path: str,
    mask: np.ndarray | None,
    output_path: str,
    sigma: float = USM_RADIUS,
    strength: float = USM_STRENGTH,
    target_size: int = TARGET_SIZE,
) -> None:
    """
    Lit la frame du milieu de la vidéo et sauvegarde une grille 2×2 :
      Col 0 — résolution native   |  Col 1 — upscalée (target_size × target_size)
      Lig 0 — sans USM (original) |  Lig 1 — avec USM

    Permet de juger l'effet USM + upscale sans lancer le traitement complet.
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise FileNotFoundError(f"Cannot open: {video_path}")

    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    mid   = max(0, total // 2)
    cap.set(cv2.CAP_PROP_POS_FRAMES, mid)
    ret, frame = cap.read()
    cap.release()
    if not ret:
        raise RuntimeError(f"Impossible de lire la frame {mid} de {video_path}")

    if frame.ndim == 3:
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    _mask = mask if mask is not None else np.ones(frame.shape, dtype=np.uint8) * 255
    frame_usm = apply_unsharp_mask(frame, _mask, sigma=sigma, strength=strength)

    H, W = frame.shape
    frame_up     = cv2.resize(frame,     (target_size, target_size), interpolation=cv2.INTER_LANCZOS4)
    frame_usm_up = cv2.resize(frame_usm, (target_size, target_size), interpolation=cv2.INTER_LANCZOS4)

    fig, axes = plt.subplots(
        2, 2,
        figsize=(12, 11),
        gridspec_kw={"hspace": 0.05, "wspace": 0.03},
    )

    panels = [
        (0, 0, frame,        f"Original  ({W}×{H})"),
        (0, 1, frame_up,     f"Original upscalée ({target_size}×{target_size})"),
        (1, 0, frame_usm,    f"USM  ({W}×{H})  σ={sigma} s={strength}"),
        (1, 1, frame_usm_up, f"USM upscalée ({target_size}×{target_size})"),
    ]
    for r, c, img, title in panels:
        axes[r, c].imshow(img, cmap="gray", vmin=0, vmax=255, interpolation="nearest")
        axes[r, c].axis("off")
        axes[r, c].set_title(title, fontsize=9, fontweight="bold")

    row_labels = ["Sans USM", "Avec USM"]
    for r, label in enumerate(row_labels):
        axes[r, 0].set_ylabel(label, fontsize=10, rotation=90,
                              ha="center", va="center", labelpad=45)

    plt.suptitle(
        f"Preview — frame {mid}/{total}  |  USM : sigma={sigma}  strength={strength}\n"
        f"Upscale Lanczos : {W}×{H} → {target_size}×{target_size}",
        fontsize=10,
    )
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Preview frame {mid} → {output_path}")


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "USM agressif par frame → projection mean/médiane → upscale 1024×1024.\n\n"
            "Étapes :\n"
            "  1. Applique skimage USM (strength=2) sur chaque frame\n"
            "  2. Calcule la moyenne et la médiane temporelle\n"
            "  3. Sauvegarde en PNG + TIFF natif et upscalé 1024×1024"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("input",
        help="Vidéo prétraitée (.avi) ou dossier la contenant (cherche step10_usm.avi).")
    parser.add_argument("--mask", default=None,
        help="Chemin vers mask.png (défaut : mask.png à côté de la vidéo, optionnel).")
    parser.add_argument("--output-dir", default=None,
        help="Dossier de sortie (défaut : même dossier que la vidéo).")
    parser.add_argument("--sigma", type=float, default=USM_RADIUS,
        help=f"Rayon Gaussien USM en pixels (défaut : {USM_RADIUS}).")
    parser.add_argument("--strength", type=float, default=USM_STRENGTH,
        help=f"Intensité USM — 2.0 correspond à L≈1280 +882%% (défaut : {USM_STRENGTH}).")
    parser.add_argument("--chunk-size", type=int, default=CHUNK_SIZE,
        help=f"Frames par chunk (défaut : {CHUNK_SIZE}).")
    parser.add_argument("--reservoir-size", type=int, default=RESERVOIR,
        help=f"Réservoir pour la médiane (défaut : {RESERVOIR}).")
    parser.add_argument("--target-size", type=int, default=TARGET_SIZE,
        help=f"Résolution de sortie upscalée (défaut : {TARGET_SIZE}).")
    parser.add_argument("--preview", action="store_true",
        help="Mode rapide : compare frame du milieu sans/avec USM (pas de projection complète).")
    args = parser.parse_args()

    inp = Path(args.input)
    if inp.is_dir():
        video_in  = inp / "step10_usm.avi"
        mask_path = inp / "mask.png"
        out_dir   = inp
    else:
        video_in  = inp
        mask_path = Path(args.mask) if args.mask else inp.parent / "mask.png"
        out_dir   = Path(args.output_dir) if args.output_dir else inp.parent

    if not video_in.exists():
        sys.exit(f"ERREUR : vidéo introuvable : {video_in}")

    mask: np.ndarray | None = None
    if mask_path.exists():
        mask = load_mask(str(mask_path))
        print(f"  Masque chargé : {mask_path}")
    else:
        print(f"  Aucun masque trouvé ({mask_path}) — traitement sur le cadre complet.")

    print(f"\nEntrée  : {video_in}")
    print(f"Sortie  : {out_dir}")

    if args.preview:
        out_preview = out_dir / "preview_usm_midframe.png"
        preview_midframe(
            video_path  = str(video_in),
            mask        = mask,
            output_path = str(out_preview),
            sigma       = args.sigma,
            strength    = args.strength,
            target_size = args.target_size,
        )
    else:
        process(
            video_path     = str(video_in),
            mask           = mask,
            output_dir     = str(out_dir),
            sigma          = args.sigma,
            strength       = args.strength,
            chunk_size     = args.chunk_size,
            reservoir_size = args.reservoir_size,
            target_size    = args.target_size,
        )

    print("\nDone.")


if __name__ == "__main__":
    main()
