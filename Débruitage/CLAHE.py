#!/usr/bin/env python3
"""
Step [9] — CLAHE + Unsharp Mask post-processing for denoised ocular vascular videos.

Expects the output of the temporal median filter (step [7]) + mask.png.
Can be run standalone (see __main__) or imported by the full pipeline.

Workflow:
  1. Run with --calibrate to generate the visual calibration grid.
  2. Open the grid image, pick the best thumbnail visually.
  3. Re-run with --clip-limit, --tile, --usm-strength to process the full video.
"""
from __future__ import annotations

import sys
import time
from itertools import product
from pathlib import Path

import cv2
import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).parent))
from mask_detection import load_mask


# ── Parameters (edit here) ────────────────────────────────────────────────────

CLAHE_CLIP_LIMIT      = 2.0        # contrast amplification ceiling
CLAHE_TILE_GRID_SIZE  = (16, 16)   # adaptive histogram tile size in pixels (width, height)
USM_KERNEL_SIZE       = 3          # Gaussian blur kernel size for USM (odd integer)
USM_STRENGTH          = 0.5        # unsharp mask blending weight

# Mask cleanup parameters
MASK_EROSION_PX       = 4          # pixels to erode mask before CLAHE (avoids border artefacts)
BORDER_SMOOTH_PX      = 10         # width of the border blending zone (px from mask edge)
BORDER_SMOOTH_SIGMA   = 3.0        # Gaussian sigma for border smoothing (proportional to BORDER_SMOOTH_PX)

FPS                     = 30.0     # acquisition frame rate (used for FFT axis)
CARDIAC_FREQ_MIN        = 0.5      # Hz — lower bound of cardiac band
CARDIAC_FREQ_MAX        = 3.0      # Hz — upper bound of cardiac band
PULSATION_AMP_THRESHOLD = 0.80     # warn if cardiac amplitude drops below this fraction


# ── Internal helpers ──────────────────────────────────────────────────────────


def _read_video_gray(path: str | Path) -> tuple[np.ndarray, float]:
    """Read all frames from a video into a uint8 (N, H, W) array."""
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise FileNotFoundError(f"Cannot open video: {path}")
    fps = cap.get(cv2.CAP_PROP_FPS) or FPS
    frames: list[np.ndarray] = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if frame.ndim == 3:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        frames.append(frame)
    cap.release()
    if not frames:
        raise ValueError(f"No frames read from {path}")
    return np.stack(frames, axis=0), fps


def _write_video(frames: np.ndarray, output_path: str | Path, fps: float) -> None:
    """Write a uint8 (N, H, W) array to a grayscale AVI file."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    N, H, W = frames.shape
    fourcc = cv2.VideoWriter_fourcc(*"XVID")
    writer = cv2.VideoWriter(str(output_path), fourcc, fps, (W, H), isColor=False)
    for i in range(N):
        writer.write(frames[i])
    writer.release()


def _local_contrast(frame: np.ndarray, mask: np.ndarray, tile_size: int = 16) -> float:
    """
    Mean std of pixel values inside non-empty tile_size×tile_size tiles overlapping the mask.
    Used as a proxy for local contrast — higher = more vessel detail visible.
    """
    H, W = frame.shape
    mask_bool = mask.astype(bool)
    stds: list[float] = []
    for y in range(0, H, tile_size):
        for x in range(0, W, tile_size):
            tile_mask = mask_bool[y : y + tile_size, x : x + tile_size]
            if tile_mask.any():
                pixels = frame[y : y + tile_size, x : x + tile_size][tile_mask]
                stds.append(float(pixels.std()))
    return float(np.mean(stds)) if stds else 0.0


def _apply_usm_inplace(
    frame_f32: np.ndarray,
    mask_bool: np.ndarray,
    ksize: int,
    strength: float,
) -> np.ndarray:
    """Apply USM on a float32 frame, restricted to mask. Returns uint8."""
    blurred = cv2.GaussianBlur(frame_f32, (ksize, ksize), 0)
    sharpened = frame_f32 + strength * (frame_f32 - blurred)
    out = np.zeros(frame_f32.shape, dtype=np.uint8)
    out[mask_bool] = np.clip(sharpened[mask_bool], 0, 255).astype(np.uint8)
    return out


def _erode_mask(mask: np.ndarray, pixels: int) -> np.ndarray:
    """
    Érode le masque de 'pixels' pixels avec un élément structurant elliptique.
    Utilisé pour exclure les bords de transition avant d'appliquer le CLAHE,
    évitant ainsi les artéfacts noirs aux frontières du masque.
    """
    if pixels <= 0:
        return mask
    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (2 * pixels + 1, 2 * pixels + 1)
    )
    return cv2.erode(mask, kernel, iterations=1)


def _smooth_mask_border(
    frame: np.ndarray,       # uint8 (H, W)
    mask: np.ndarray,        # uint8 (H, W)
    border_px: int = 10,
    sigma: float   = 0.5,
) -> np.ndarray:
    """
    Applique un léger flou Gaussien (sigma) uniquement sur la bande de 'border_px'
    pixels à l'intérieur du bord du masque.

    Zone de lissage = pixels dans le masque original qui disparaissent après
    une érosion de border_px px → anneau intérieur du masque.

    Lisse la transition bord sans toucher au centre de l'image.
    """
    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (2 * border_px + 1, 2 * border_px + 1)
    )
    inner_mask = cv2.erode(mask, kernel, iterations=1)
    # Anneau = pixels dans mask mais pas dans inner_mask
    border_zone = (mask > 0) & (inner_mask == 0)

    if not border_zone.any():
        return frame

    # Taille du kernel : au moins 3, impair, proportionnel à sigma
    ksize = max(3, int(np.ceil(6 * sigma)) | 1)  # | 1 = force impair
    blurred = cv2.GaussianBlur(frame.astype(np.float32), (ksize, ksize), sigma)

    result = frame.astype(np.float32).copy()
    result[border_zone] = blurred[border_zone]
    return np.clip(result, 0, 255).astype(np.uint8)


# ── Core processing functions ─────────────────────────────────────────────────


def apply_clahe(
    frame: np.ndarray,
    mask: np.ndarray,
    clip_limit: float = CLAHE_CLIP_LIMIT,
    tile_grid_size: tuple = CLAHE_TILE_GRID_SIZE,
    erosion_px: int = MASK_EROSION_PX,
) -> np.ndarray:
    """
    Apply CLAHE only inside the mask region.

    Le masque est légèrement érodé avant application pour éviter que le CLAHE
    traite les pixels de bord de transition → supprime les points noirs aux bordures.
    Pixels outside the original mask remain 0. Input must be uint8.
    Retourne la frame rehaussée uint8.
    """
    clahe       = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
    eroded_mask = _erode_mask(mask, erosion_px)
    eroded_bool = eroded_mask.astype(bool)
    # Fill non-mask area with the mean of masked pixels so the CLAHE tile
    # histograms are not dominated by the black background (0-bias fix).
    fill_val    = int(frame[eroded_bool].mean()) if eroded_bool.any() else 128
    filled      = frame.copy()
    filled[~eroded_bool] = fill_val
    enhanced    = clahe.apply(filled)
    out         = np.zeros_like(frame)
    out[eroded_bool] = enhanced[eroded_bool]
    return out


def apply_unsharp_mask(
    frame: np.ndarray,
    mask: np.ndarray,
    kernel_size: int = USM_KERNEL_SIZE,
    strength: float = USM_STRENGTH,
) -> np.ndarray:
    """
    Apply Unsharp Mask only inside the mask region.

    formula : output = frame + strength × (frame − gaussian_blur(frame))
    Clippe les valeurs entre 0 et 255 après application.
    Retourne la frame renforcée uint8.
    """
    ksize = kernel_size if kernel_size % 2 == 1 else kernel_size + 1
    return _apply_usm_inplace(
        frame.astype(np.float32),
        mask.astype(bool),
        ksize,
        strength,
    )


def apply_clahe_usm_pipeline(
    video_path: str,
    mask: np.ndarray,
    output_path: str,
    clahe_clip_limit: float = CLAHE_CLIP_LIMIT,
    clahe_tile_grid_size: tuple = CLAHE_TILE_GRID_SIZE,
    usm_kernel_size: int = USM_KERNEL_SIZE,
    usm_strength: float = USM_STRENGTH,
    mask_erosion_px: int = MASK_EROSION_PX,
    border_smooth_px: int = BORDER_SMOOTH_PX,
    border_smooth_sigma: float = BORDER_SMOOTH_SIGMA,
) -> dict:
    """
    Apply CLAHE then USM to all frames of a video, with 3 border-cleanup steps:

      Étape A — Érosion du masque avant CLAHE (mask_erosion_px)
        Le CLAHE n'opère que sur les pixels de l'intérieur érodé → évite
        que l'histogramme local soit perturbé par les pixels de transition.

      Étape B — Réapplication du masque original après CLAHE + USM
        Force frame[masque == 0] = 0 → élimine les points noirs résiduels
        apparus aux bords pendant le traitement.

      Étape C — Lissage Gaussien de la bande de bord (border_smooth_px, sigma)
        Appliqué uniquement sur l'anneau border_smooth_px px à l'intérieur
        du bord → transition douce sans toucher au centre.

    CLAHE object is pre-created and reused across frames for speed.
    Target: < 3 ms per frame.

    Returns:
        dict with:
          'mean_contrast_before' : float
          'mean_contrast_after'  : float
          'processing_time_ms'   : float
    """
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise FileNotFoundError(f"Cannot open video: {video_path}")
    fps = cap.get(cv2.CAP_PROP_FPS) or FPS
    N   = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    W   = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H   = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"  Loading video …  {N} frames @ {fps:.1f} fps  ({H}×{W})")

    clahe = cv2.createCLAHE(
        clipLimit=clahe_clip_limit,
        tileGridSize=clahe_tile_grid_size,
    )
    ksize = usm_kernel_size if usm_kernel_size % 2 == 1 else usm_kernel_size + 1

    eroded_mask   = _erode_mask(mask, mask_erosion_px)
    eroded_bool   = eroded_mask.astype(bool)
    original_bool = mask.astype(bool)

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*"XVID")
    writer = cv2.VideoWriter(str(output_path), fourcc, fps, (W, H), isColor=False)

    # Sample every ~5% of frames for contrast metrics
    sample_interval = max(1, N // 20)
    contrast_before_samples: list[float] = []
    contrast_after_samples:  list[float] = []
    frame_times: list[float] = []
    i = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if frame.ndim == 3:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        if i % sample_interval == 0:
            contrast_before_samples.append(_local_contrast(frame, mask))

        t0 = time.perf_counter()

        # Étape A — CLAHE sur le masque érodé (histogramme non biaisé par le fond noir)
        fill_val    = int(frame[eroded_bool].mean()) if eroded_bool.any() else 128
        filled      = frame.copy()
        filled[~eroded_bool] = fill_val
        enhanced    = clahe.apply(filled)
        after_clahe = np.zeros_like(frame)
        after_clahe[eroded_bool] = enhanced[eroded_bool]

        # USM — renforcement des bords (sur le masque érodé)
        after_usm = _apply_usm_inplace(
            after_clahe.astype(np.float32), eroded_bool, ksize, usm_strength
        )

        # Étape B — Réapplication du masque original : force 0 hors masque
        after_usm[~original_bool] = 0

        # Étape C — Lissage gaussien uniquement sur la bande de bord
        out_frame = _smooth_mask_border(
            after_usm, mask, border_smooth_px, border_smooth_sigma
        )

        frame_times.append((time.perf_counter() - t0) * 1_000.0)

        if i % sample_interval == 0:
            contrast_after_samples.append(_local_contrast(out_frame, mask))

        writer.write(out_frame)
        i += 1

    cap.release()
    writer.release()

    contrast_before = float(np.mean(contrast_before_samples)) if contrast_before_samples else 0.0
    contrast_after  = float(np.mean(contrast_after_samples))  if contrast_after_samples  else 0.0
    avg_ms          = float(np.mean(frame_times))              if frame_times             else 0.0

    print(f"\n  ── CLAHE + Unsharp Mask (avec nettoyage de bord) ───────────")
    print(f"  clipLimit            : {clahe_clip_limit}")
    print(f"  tileGridSize         : {clahe_tile_grid_size}")
    print(f"  USM kernel / force   : {ksize}×{ksize}  /  {usm_strength}")
    print(f"  Érosion masque       : {mask_erosion_px} px")
    print(f"  Lissage bord         : {border_smooth_px} px  σ={border_smooth_sigma}")
    print(f"  Temps moyen/frame    : {avg_ms:.2f} ms  {'[OK]' if avg_ms < 3.0 else '[LENT — > 3 ms]'}")
    print(f"  Contraste local avant: {contrast_before:.3f}")
    print(f"  Contraste local après: {contrast_after:.3f}")
    if contrast_before > 0:
        print(f"  Gain contraste       : {contrast_after / contrast_before:.2f}×")
    print(f"  Sortie               : {output_path}")

    return {
        "mean_contrast_before": contrast_before,
        "mean_contrast_after":  contrast_after,
        "processing_time_ms":   avg_ms,
    }


# ── Calibration & visualization ───────────────────────────────────────────────


def calibration_grid(
    video_path: str,
    mask: np.ndarray,
    output_path: str,
) -> None:
    """
    OUTIL DE CALIBRATION VISUELLE — le plus important de ce fichier.

    Prend la frame du milieu de la vidéo et génère une grille testant toutes
    les combinaisons de :
      clipLimit    : [1.5, 2.0, 3.0]
      tileGridSize : [8×8, 16×16, 32×32]
      usm_strength : [0.3, 0.5, 0.8]

    Grid layout : 9 lignes (clipLimit × tileGridSize) × 4 colonnes
      Col 0          : frame originale (référence)
      Cols 1, 2, 3   : usm_strength = 0.3, 0.5, 0.8

    Chaque vignette est annotée avec ses paramètres et le contraste local mesuré.
    Sauvegarder en haute résolution pour choisir les meilleurs paramètres
    avant de traiter toute la vidéo.
    """
    CLIP_LIMITS   = [1.5, 2.0, 3.0]
    TILE_SIZES    = [(8, 8), (16, 16), (32, 32)]
    USM_STRENGTHS = [0.3, 0.5, 0.8]
    USM_KERNEL    = USM_KERNEL_SIZE if USM_KERNEL_SIZE % 2 == 1 else USM_KERNEL_SIZE + 1

    frames, _ = _read_video_gray(video_path)
    mid = frames[len(frames) // 2]

    # Pré-calcul du masque érodé (même traitement que le pipeline)
    eroded_mask  = _erode_mask(mask, MASK_EROSION_PX)
    eroded_bool  = eroded_mask.astype(bool)
    original_bool = mask.astype(bool)

    combos = list(product(CLIP_LIMITS, TILE_SIZES))   # 9 lignes
    n_rows = len(combos)
    n_cols = len(USM_STRENGTHS) + 1                   # +1 colonne "Référence"

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4.2 * n_cols, 3.8 * n_rows))

    for row, (clip, tile) in enumerate(combos):
        # Colonne 0 : référence non traitée
        ax = axes[row, 0]
        ax.imshow(mid, cmap="gray", vmin=0, vmax=255)
        ax.axis("off")
        if row == 0:
            ax.set_title("Référence\n(non traité)", fontsize=8, fontweight="bold", color="seagreen")
        ax.set_ylabel(
            f"clip={clip}  tile={tile[0]}×{tile[1]}",
            fontsize=7, rotation=0, ha="right", va="center", labelpad=60,
        )

        # Pré-calcul CLAHE (Étape A : masque érodé, histogramme non biaisé)
        clahe       = cv2.createCLAHE(clipLimit=clip, tileGridSize=tile)
        fill_val    = int(mid[eroded_bool].mean()) if eroded_bool.any() else 128
        mid_filled  = mid.copy()
        mid_filled[~eroded_bool] = fill_val
        after_clahe = np.zeros_like(mid)
        after_clahe[eroded_bool] = clahe.apply(mid_filled)[eroded_bool]
        f32 = after_clahe.astype(np.float32)

        for col_idx, strength in enumerate(USM_STRENGTHS):
            blurred = cv2.GaussianBlur(f32, (USM_KERNEL, USM_KERNEL), 0)
            # USM sur masque érodé
            after_usm = np.zeros_like(mid)
            after_usm[eroded_bool] = np.clip(
                f32 + strength * (f32 - blurred), 0, 255
            )[eroded_bool].astype(np.uint8)
            # Étape B : masque original
            after_usm[~original_bool] = 0
            # Étape C : lissage de bord
            out = _smooth_mask_border(after_usm, mask, BORDER_SMOOTH_PX, BORDER_SMOOTH_SIGMA)

            ax = axes[row, col_idx + 1]
            ax.imshow(out, cmap="gray", vmin=0, vmax=255)
            ax.axis("off")

            if row == 0:
                ax.set_title(
                    f"usm_strength = {strength}\n(kernel={USM_KERNEL})",
                    fontsize=8, fontweight="bold",
                )

            contrast = _local_contrast(out, mask)
            ax.text(
                0.02, 0.02,
                f"C={contrast:.1f}",
                transform=ax.transAxes,
                fontsize=6, color="yellow", va="bottom",
                bbox=dict(facecolor="black", alpha=0.4, pad=1, linewidth=0),
            )

    plt.suptitle(
        "Grille de calibration CLAHE + USM\n"
        "Lignes : clipLimit × tileGridSize  |  Colonnes : usm_strength\n"
        "C = contraste local (std sur tuiles 16×16)  —  chercher vaisseaux nets sans halos ni bruit",
        fontsize=10,
    )
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Grille de calibration → {output_path}")


def compare_full_pipeline(
    video_brute: str,
    video_preprocessed: str,
    video_temporal_median: str,
    video_final: str,
    mask: np.ndarray,
    output_path: str,
    n_samples: int = 5,
    crop_size: int = 150,
    crop_origin: tuple[int, int] = (180, 180),
) -> None:
    """
    Grille de comparaison du pipeline complet sur n_samples frames.

    Layout : 2×n_samples lignes × 4 colonnes
      Lignes paires  : frame complète (brute, prétraité, median, final)
      Lignes impaires: crop 150×150 px sur une zone riche en petits vaisseaux

    Le rectangle rouge sur la frame complète indique la zone croppée.

    Args:
        crop_origin : (y, x) coin supérieur-gauche du crop — à ajuster sur
                      une zone riche en petits vaisseaux de la vidéo.
    """
    col_titles = [
        "Vidéo brute",
        "Après prétraitement\n(étapes 1-6)",
        "Après median\ntemporel",
        "Après CLAHE + USM\n(final)",
    ]

    vids: list[np.ndarray] = []
    for path in (video_brute, video_preprocessed, video_temporal_median, video_final):
        arr, _ = _read_video_gray(path)
        vids.append(arr)

    N_max   = min(len(v) for v in vids)
    indices = np.linspace(0, N_max - 1, n_samples, dtype=int)

    cy, cx = crop_origin
    ch = cw = crop_size

    fig, axes = plt.subplots(
        2 * n_samples, 4,
        figsize=(16, 5.2 * n_samples),
        gridspec_kw={"hspace": 0.08, "wspace": 0.04},
    )

    for row, idx in enumerate(indices):
        for col, vid in enumerate(vids):
            frame = vid[min(idx, len(vid) - 1)]

            # Ligne paire : frame complète avec rectangle de crop
            ax_full = axes[2 * row, col]
            ax_full.imshow(frame, cmap="gray", vmin=0, vmax=255)
            ax_full.add_patch(
                plt.Rectangle((cx, cy), cw, ch,
                               linewidth=1.2, edgecolor="red", facecolor="none")
            )
            ax_full.axis("off")
            if row == 0:
                ax_full.set_title(col_titles[col], fontsize=9, fontweight="bold")
            if col == 0:
                ax_full.set_ylabel(f"Frame {idx}", fontsize=8)

            # Ligne impaire : crop
            ax_crop = axes[2 * row + 1, col]
            crop = frame[cy : cy + ch, cx : cx + cw]
            ax_crop.imshow(crop, cmap="gray", vmin=0, vmax=255)
            ax_crop.axis("off")
            if col == 0 and row == 0:
                ax_crop.set_ylabel(f"Crop {crop_size}px", fontsize=8)

    plt.suptitle(
        f"Comparaison pipeline complet — {n_samples} frames échantillons\n"
        f"Lignes paires : frame complète  |  Lignes impaires : crop {crop_size}×{crop_size} px"
        f"  (zone vasculaire — gain le plus visible)",
        fontsize=10,
    )
    plt.savefig(output_path, dpi=120, bbox_inches="tight")
    plt.close()
    print(f"  Grille comparaison pipeline → {output_path}")


def verify_pulsation_preserved(
    video_temporal_median: str,
    video_final: str,
    mask: np.ndarray,
    output_path: str,
    fps: float = FPS,
) -> dict:
    """
    Vérifie que la pulsation cardiaque est intacte après CLAHE + USM.

    Superpose les courbes de luminosité moyenne par frame des deux vidéos
    ainsi que leurs spectres FFT. Émet un avertissement si l'amplitude
    cardiaque chute sous PULSATION_AMP_THRESHOLD.

    Returns:
        dict with 'cardiac_amplitude_ratio' (float) and 'dominant_freq_hz' (float)
    """
    mask_bool = mask.astype(bool)

    pre, fps_det = _read_video_gray(video_temporal_median)
    fin, _       = _read_video_gray(video_final)
    fps = fps_det or fps

    N = min(len(pre), len(fin))
    t = np.arange(N)

    lum_pre = np.array([pre[i][mask_bool].mean() for i in range(N)], dtype=np.float64)
    lum_fin = np.array([fin[i][mask_bool].mean() for i in range(N)], dtype=np.float64)

    # De-trend : retirer DC + dérive linéaire pour isoler la composante cardiaque
    lum_pre -= np.polyval(np.polyfit(t, lum_pre, 1), t)
    lum_fin -= np.polyval(np.polyfit(t, lum_fin, 1), t)

    freqs   = np.fft.rfftfreq(N, d=1.0 / fps)
    fft_pre = np.abs(np.fft.rfft(lum_pre))
    fft_fin = np.abs(np.fft.rfft(lum_fin))

    cardiac = (freqs >= CARDIAC_FREQ_MIN) & (freqs <= CARDIAC_FREQ_MAX)

    if not np.any(cardiac):
        print("  AVERTISSEMENT : résolution temporelle insuffisante pour la bande cardiaque.")
        return {"cardiac_amplitude_ratio": float("nan"), "dominant_freq_hz": float("nan")}

    amp_pre  = fft_pre[cardiac].max()
    amp_fin  = fft_fin[cardiac].max()
    ratio    = float(amp_fin / amp_pre) if amp_pre > 0 else float("nan")
    dom_freq = float(freqs[cardiac][np.argmax(fft_pre[cardiac])])
    preserved = ratio >= PULSATION_AMP_THRESHOLD

    print(f"\n  ── Préservation de la pulsation (CLAHE+USM) ────────────────")
    print(f"  Fréquence cardiaque dominante : {dom_freq:.2f} Hz")
    print(f"  Amplitude cardiaque (median)  : {amp_pre:.4f}")
    print(f"  Amplitude cardiaque (final)   : {amp_fin:.4f}")
    print(f"  Ratio d'amplitude             : {ratio * 100:.1f} %")
    if preserved:
        print(f"  Pulsation préservée           : OUI  "
              f"({ratio*100:.1f}% ≥ {PULSATION_AMP_THRESHOLD*100:.0f}%)")
    else:
        print(f"  Pulsation préservée           : NON  "
              f"({ratio*100:.1f}% < {PULSATION_AMP_THRESHOLD*100:.0f}%)")
        print(f"  → Réduire CLAHE_CLIP_LIMIT (actuellement {CLAHE_CLIP_LIMIT})")

    time_axis = t / fps
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))

    ax1.plot(time_axis, lum_pre, lw=0.8, color="steelblue", label="Après median temporel")
    ax1.plot(time_axis, lum_fin, lw=0.8, color="coral", ls="--", label="Après CLAHE+USM")
    ax1.set_xlabel("Temps (s)")
    ax1.set_ylabel("Luminosité moyenne centrée")
    ax1.set_title("Signal de luminosité dans le masque (dérivé)")
    ax1.legend(fontsize=9)

    ax2.plot(freqs, fft_pre, lw=0.8, color="steelblue", label="Après median temporel")
    ax2.plot(freqs, fft_fin, lw=0.8, color="coral", ls="--", label="Après CLAHE+USM")
    ax2.axvspan(CARDIAC_FREQ_MIN, CARDIAC_FREQ_MAX, alpha=0.12, color="green",
                label=f"Bande cardiaque [{CARDIAC_FREQ_MIN}–{CARDIAC_FREQ_MAX} Hz]")
    ax2.axvline(dom_freq, color="green", ls=":", lw=1.0,
                label=f"f_dom = {dom_freq:.2f} Hz")
    ax2.set_xlabel("Fréquence (Hz)")
    ax2.set_ylabel("Amplitude FFT")
    ax2.set_xlim(0, min(15.0, fps / 2))
    ax2.set_title(
        f"Spectre fréquentiel — ratio amplitude cardiaque = {ratio*100:.1f}%"
        f"  {'[OK]' if preserved else '[ATTENTION : atténuation]'}"
    )
    ax2.legend(fontsize=9)

    plt.tight_layout()
    plt.savefig(output_path, dpi=120, bbox_inches="tight")
    plt.close()
    print(f"  Graphique pulsation → {output_path}")

    return {"cardiac_amplitude_ratio": ratio, "dominant_freq_hz": dom_freq}


# ── CLI helpers ───────────────────────────────────────────────────────────────

_VIDEO_EXTENSIONS = {".avi", ".mp4", ".mkv", ".mov", ".mpeg", ".mpg"}


def _resolve_mask(mask_arg: str | None, video_path: Path, fallback_dir: Path | None = None) -> np.ndarray:
    """Return a loaded mask array.

    Priority: --mask argument > mask.png next to video > mask.png in fallback_dir.
    """
    if mask_arg:
        p = Path(mask_arg)
        if not p.exists():
            sys.exit(f"ERREUR : masque introuvable : {p}")
        return load_mask(str(p))
    candidates = [video_path.parent / "mask.png"]
    if fallback_dir:
        candidates.append(fallback_dir / "mask.png")
    for p in candidates:
        if p.exists():
            return load_mask(str(p))
    sys.exit(
        f"ERREUR : masque introuvable. Fournissez --mask <chemin> "
        f"ou placez mask.png dans {video_path.parent}"
    )


def _run_single(video_in: Path, mask: np.ndarray, out_dir: Path, args) -> None:
    """Process one video and save outputs to out_dir."""
    tile_size = (args.tile, args.tile)
    stem      = video_in.stem

    if args.calibrate:
        out_calib = out_dir / f"{stem}_calibration_grid.png"
        print(f"\n  Mode calibration — grille → {out_calib}\n")
        calibration_grid(str(video_in), mask, str(out_calib))
        print(
            "\n  ── Comment utiliser la grille ──────────────────────────────\n"
            "  1. Ouvrir la grille à 100% dans un viewer.\n"
            "  2. Chercher la vignette où les petits vaisseaux sont nets,\n"
            "     sans halos et sans bruit amplifié dans le fond noir.\n"
            "  3. Lire les paramètres annotés (clip, tile, usm_strength).\n"
            "  4. Relancer sans --calibrate avec ces valeurs :\n"
            "     python CLAHE.py <input> --clip-limit X --tile Y --usm-strength Z"
        )
        return

    out_video = out_dir / f"{stem}_clahe_usm.avi"
    out_puls  = out_dir / f"{stem}_pulsation.png"

    print(f"    Sortie  : {out_video}")
    print(f"    clip={args.clip_limit}  tile={tile_size}  "
          f"usm_kernel={args.usm_kernel}  usm_strength={args.usm_strength}\n")

    apply_clahe_usm_pipeline(
        video_path           = str(video_in),
        mask                 = mask,
        output_path          = str(out_video),
        clahe_clip_limit     = args.clip_limit,
        clahe_tile_grid_size = tile_size,
        usm_kernel_size      = args.usm_kernel,
        usm_strength         = args.usm_strength,
    )

    verify_pulsation_preserved(
        video_temporal_median = str(video_in),
        video_final           = str(out_video),
        mask                  = mask,
        output_path           = str(out_puls),
    )

    if getattr(args, "original", None) and getattr(args, "preprocessed", None):
        out_compare = out_dir / f"{stem}_pipeline_comparison.png"
        compare_full_pipeline(
            video_brute           = args.original,
            video_preprocessed    = args.preprocessed,
            video_temporal_median = str(video_in),
            video_final           = str(out_video),
            mask                  = mask,
            output_path           = str(out_compare),
            crop_origin           = (args.crop_y, args.crop_x),
        )


# ── CLI entry point ───────────────────────────────────────────────────────────


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description=(
            "Step [9] — CLAHE + Unsharp Mask post-processing.\n\n"
            "Modes :\n"
            "  Vidéo unique  : python CLAHE.py <vidéo_ou_dossier_pipeline>\n"
            "  Dossier entier: python CLAHE.py <dossier_videos> --batch\n\n"
            "Workflow recommandé :\n"
            "  1. python CLAHE.py <input> --calibrate\n"
            "  2. Ouvrir la grille de calibration, choisir les paramètres\n"
            "  3. python CLAHE.py <input> --clip-limit X --tile Y --usm-strength Z"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "input",
        help=(
            "Vidéo, dossier pipeline (contenant step7_temporal_median.avi + mask.png), "
            "ou dossier de vidéos (avec --batch)."
        ),
    )
    parser.add_argument(
        "--mask", default=None, metavar="PATH",
        help="Chemin vers le masque (défaut : mask.png dans le même dossier que la vidéo)",
    )
    parser.add_argument(
        "--output", default=None, metavar="DIR",
        help="Dossier de sortie (défaut : même dossier que la vidéo entrante)",
    )
    parser.add_argument(
        "--batch", action="store_true",
        help=(
            "Traiter toutes les vidéos du dossier input. "
            "Un masque commun peut être fourni via --mask ; "
            "sinon mask.png est cherché dans chaque sous-dossier puis dans input."
        ),
    )
    parser.add_argument(
        "--calibrate", action="store_true",
        help="Générer la grille de calibration visuelle (ne traite pas toute la vidéo)",
    )
    parser.add_argument(
        "--clip-limit", type=float, default=CLAHE_CLIP_LIMIT,
        help=f"clipLimit CLAHE (défaut {CLAHE_CLIP_LIMIT})",
    )
    parser.add_argument(
        "--tile", type=int, default=CLAHE_TILE_GRID_SIZE[0],
        help=f"Taille des tuiles CLAHE en pixels (défaut {CLAHE_TILE_GRID_SIZE[0]})",
    )
    parser.add_argument(
        "--usm-kernel", type=int, default=USM_KERNEL_SIZE,
        help=f"Taille du kernel USM (défaut {USM_KERNEL_SIZE})",
    )
    parser.add_argument(
        "--usm-strength", type=float, default=USM_STRENGTH,
        help=f"Intensité USM (défaut {USM_STRENGTH})",
    )
    parser.add_argument(
        "--original", default=None,
        help="Vidéo brute originale (pour compare_full_pipeline, mode vidéo unique)",
    )
    parser.add_argument(
        "--preprocessed", default=None,
        help="Vidéo après prétraitement étapes 1-6 (pour compare_full_pipeline, mode vidéo unique)",
    )
    parser.add_argument(
        "--crop-y", type=int, default=180,
        help="Coordonnée Y du coin supérieur-gauche du crop (défaut 180)",
    )
    parser.add_argument(
        "--crop-x", type=int, default=180,
        help="Coordonnée X du coin supérieur-gauche du crop (défaut 180)",
    )
    args = parser.parse_args()

    inp     = Path(args.input)
    out_dir = Path(args.output) if args.output else None

    # ── Mode batch : traiter toutes les vidéos d'un dossier ──────────────────
    if args.batch:
        if not inp.is_dir():
            sys.exit("ERREUR : --batch requiert un dossier en entrée")

        videos = sorted(f for f in inp.iterdir() if f.suffix.lower() in _VIDEO_EXTENSIONS)
        if not videos:
            sys.exit(f"ERREUR : aucune vidéo ({', '.join(_VIDEO_EXTENSIONS)}) trouvée dans {inp}")

        batch_out = out_dir or inp
        batch_out.mkdir(parents=True, exist_ok=True)

        print(f"\n[9] CLAHE + USM — batch : {len(videos)} vidéo(s) dans {inp}")
        print(f"    Sortie   : {batch_out}")
        if args.mask:
            print(f"    Masque   : {args.mask} (commun)")

        skipped = 0
        for video_in in videos:
            print(f"\n  ── {video_in.name} " + "─" * max(0, 50 - len(video_in.name)))
            try:
                mask = _resolve_mask(args.mask, video_in, fallback_dir=inp)
            except SystemExit as e:
                print(f"  IGNORÉ : {e}")
                skipped += 1
                continue

            print(f"    Entrée : {video_in}")
            _run_single(video_in, mask, batch_out, args)

        print(f"\n[9] Batch terminé — {len(videos) - skipped}/{len(videos)} vidéo(s) traitée(s) → {batch_out}")

    # ── Mode vidéo unique ─────────────────────────────────────────────────────
    else:
        if inp.is_dir():
            video_in       = inp / "step7_temporal_median.avi"
            default_out    = out_dir or inp
        else:
            video_in       = inp
            default_out    = out_dir or inp.parent

        if not video_in.exists():
            sys.exit(f"ERREUR : vidéo introuvable : {video_in}")

        default_out.mkdir(parents=True, exist_ok=True)
        mask = _resolve_mask(args.mask, video_in)

        print(f"\n[9] CLAHE + Unsharp Mask")
        print(f"    Entrée  : {video_in}")
        print(f"    Masque  : {args.mask or (video_in.parent / 'mask.png')}")
        print(f"    Sortie  : {default_out}")

        _run_single(video_in, mask, default_out, args)
