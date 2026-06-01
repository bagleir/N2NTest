#!/usr/bin/env python3
"""
Step [10] — Unsharp Mask (USM) post-processing for denoised ocular vascular videos.

Applied after CLAHE (step [9]) to selectively reinforce vessel edges and fine
details without amplifying the homogeneous background.

Uses skimage.filters.unsharp_mask — internally computes:
    output = clip(image + amount × (image − gaussian(image, sigma)), 0, 1)
mapped back to uint8 [0, 255].

Only pixels inside the circular retinal mask are processed; outside pixels stay 0.

Workflow:
  1. python UnsharpMask.py <video.avi> [mask.png] [output_dir] --calibrate
     → opens step10_usm_calibration.png, pick best (sigma, strength).
  2. python UnsharpMask.py <video.avi> [mask.png] [output_dir]
     --sigma S --strength A
     → processes full video + saves comparison grid.
"""
from __future__ import annotations

import sys
import time
import warnings
from itertools import product
from pathlib import Path

import cv2
import numpy as np
from skimage.filters import laplace as ski_laplace
from skimage.filters import unsharp_mask as ski_unsharp_mask

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).parent))
from mask_detection import load_mask


# ── Parameters (edit here, never hardcode elsewhere) ─────────────────────────

USM_SIGMA    = 1.0   # Gaussian radius passed to skimage.filters.unsharp_mask
USM_STRENGTH = 0.5   # sharpening amount (≡ strength), [0.0 – 2.0]

# Automatic validation thresholds
SHARPNESS_INCREASE_MIN = 0.01   # relative — warn if Laplacian var gain < this fraction
NOISE_CORRELATION_MAX  = 0.3    # Pearson |r| — warn if residual correlates with noise
TIMING_BUDGET_MS       = 2.0    # warn if mean frame time exceeds this


# ── Internal helpers ──────────────────────────────────────────────────────────


def _laplacian_variance(frame: np.ndarray, mask: np.ndarray) -> float:
    """Variance of the Laplacian response inside the mask (sharpness proxy)."""
    lap = ski_laplace(frame.astype(np.float64))
    mask_bool = mask.astype(bool)
    return float(lap[mask_bool].var()) if mask_bool.any() else 0.0


def _pearson_r(a: np.ndarray, b: np.ndarray) -> float:
    """Pearson correlation between two 1-D float arrays."""
    a = a.astype(np.float64)
    b = b.astype(np.float64)
    a_c = a - a.mean()
    b_c = b - b.mean()
    denom = np.sqrt((a_c ** 2).sum() * (b_c ** 2).sum())
    return float(np.dot(a_c, b_c) / denom) if denom > 0 else 0.0


def _read_video_gray(path: str | Path) -> tuple[np.ndarray, float]:
    """Read all frames from a video into a uint8 (N, H, W) array."""
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise FileNotFoundError(f"Cannot open video: {path}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
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


# ── Core processing ───────────────────────────────────────────────────────────


def apply_unsharp_mask(
    frame: np.ndarray,
    mask: np.ndarray,
    sigma: float = USM_SIGMA,
    strength: float = USM_STRENGTH,
) -> np.ndarray:
    """
    Apply Unsharp Mask only inside the mask region.

    Delegates to skimage.filters.unsharp_mask(radius=sigma, amount=strength).
    The kernel support is derived automatically from sigma (≈ 6σ + 1 pixels),
    so no explicit kernel_size is needed.

    Args:
        frame:    Input uint8 grayscale frame (H, W).
        mask:     Binary mask (H, W) uint8 from step [1].
        sigma:    Gaussian radius (std dev in pixels).
        strength: Sharpening amount in [0.0, 1.0].

    Returns:
        Sharpened uint8 frame; pixels outside the mask are 0.
    """
    mask_bool = mask.astype(bool)
    sharpened = ski_unsharp_mask(frame, radius=sigma, amount=strength, preserve_range=True)
    out = np.zeros(frame.shape, dtype=np.uint8)
    out[mask_bool] = np.clip(sharpened[mask_bool], 0, 255).astype(np.uint8)
    return out


def apply_usm_to_video(
    video_path: str,
    mask: np.ndarray,
    output_path: str,
    sigma: float = USM_SIGMA,
    strength: float = USM_STRENGTH,
) -> dict:
    """
    Apply USM to all frames of a video with automatic validation.

    Validation checks performed:
      1. Laplacian variance must increase — warns if gain < SHARPNESS_INCREASE_MIN.
      2. |Pearson r| between (after − before) and noise residual (Laplacian of input)
         must stay below NOISE_CORRELATION_MAX — warns if strength amplifies noise.
      3. Mean processing time must stay below TIMING_BUDGET_MS.

    Args:
        video_path:  Path to the input .avi video.
        mask:        Binary mask (H, W) uint8 from step [1].
        output_path: Destination .avi path.
        sigma:       Gaussian radius passed to skimage.filters.unsharp_mask.
        strength:    Sharpening amount.

    Returns:
        dict with keys:
          'mean_sharpness_before' : float — mean Laplacian variance (sampled frames).
          'mean_sharpness_after'  : float — idem after USM.
          'processing_time_ms'    : float — mean ms per frame.
    """
    print("  Loading video …", end=" ", flush=True)
    frames, fps = _read_video_gray(video_path)
    N, H, W = frames.shape
    print(f"{N} frames @ {fps:.1f} fps  ({H}×{W})")

    mask_bool = mask.astype(bool)
    output_frames = np.empty_like(frames)
    frame_times: list[float] = []

    for i in range(N):
        t0 = time.perf_counter()
        sharpened = ski_unsharp_mask(frames[i], radius=sigma, amount=strength, preserve_range=True)
        out = np.zeros((H, W), dtype=np.uint8)
        out[mask_bool] = np.clip(sharpened[mask_bool], 0, 255).astype(np.uint8)
        output_frames[i] = out
        frame_times.append((time.perf_counter() - t0) * 1_000.0)

    # ── Metrics on a representative sample ───────────────────────────────────
    sample_idx = np.linspace(0, N - 1, min(20, N), dtype=int)
    sharp_before = float(np.mean([_laplacian_variance(frames[i], mask) for i in sample_idx]))
    sharp_after  = float(np.mean([_laplacian_variance(output_frames[i], mask) for i in sample_idx]))
    avg_ms       = float(np.mean(frame_times))

    # ── Validation 1 : sharpness must increase ────────────────────────────────
    relative_gain = (sharp_after - sharp_before) / (sharp_before + 1e-9)
    if relative_gain < SHARPNESS_INCREASE_MIN:
        warnings.warn(
            f"USM TOO WEAK: Laplacian variance gain = {relative_gain*100:.2f}% "
            f"< {SHARPNESS_INCREASE_MIN*100:.0f}%. "
            f"Try increasing strength (current: {strength}) or sigma.",
            UserWarning,
            stacklevel=2,
        )

    # ── Validation 2 : residual must not correlate with noise ─────────────────
    mid_idx      = sample_idx[len(sample_idx) // 2]
    residual     = (output_frames[mid_idx].astype(np.float64)
                    - frames[mid_idx].astype(np.float64))[mask_bool].ravel()
    noise_est    = ski_laplace(frames[mid_idx].astype(np.float64))[mask_bool].ravel()
    r_noise      = abs(_pearson_r(residual, noise_est))

    if r_noise > NOISE_CORRELATION_MAX:
        warnings.warn(
            f"USM STRENGTH TOO HIGH: residual–noise |r| = {r_noise:.3f} "
            f"> {NOISE_CORRELATION_MAX}. "
            "The difference contains noise texture, not just edges. "
            f"Reduce strength (current: {strength}).",
            UserWarning,
            stacklevel=2,
        )

    # ── Validation 3 : timing budget ─────────────────────────────────────────
    if avg_ms > TIMING_BUDGET_MS:
        warnings.warn(
            f"USM SLOW: mean time = {avg_ms:.2f} ms/frame > budget {TIMING_BUDGET_MS} ms. "
            "Use sigma ≤ 1.0 for speed.",
            UserWarning,
            stacklevel=2,
        )

    _write_video(output_frames, output_path, fps)

    print(f"\n  ── Unsharp Mask ─────────────────────────────────────────────")
    print(f"  sigma={sigma}  strength={strength}")
    print(f"  Netteté avant     : {sharp_before:.2f}")
    print(f"  Netteté après     : {sharp_after:.2f}")
    print(f"  Gain Laplacien    : {relative_gain*100:+.1f}%")
    print(f"  Bruit |r|         : {r_noise:.3f}  "
          f"({'OK' if r_noise <= NOISE_CORRELATION_MAX else 'ATTENTION'})")
    print(f"  Temps moyen/frame : {avg_ms:.2f} ms  "
          f"({'OK' if avg_ms <= TIMING_BUDGET_MS else 'LENT'})")
    print(f"  Sortie            : {output_path}")

    return {
        "mean_sharpness_before": sharp_before,
        "mean_sharpness_after":  sharp_after,
        "processing_time_ms":    avg_ms,
    }


# ── Calibration & comparison ──────────────────────────────────────────────────


def calibration_grid_usm(
    video_path: str,
    mask: np.ndarray,
    output_path: str,
) -> None:
    """
    Visual calibration grid — test all combinations of sigma × strength.

    Grid: 3 rows (sigma) × 4 columns (strength) = 12 thumbnails.
      sigma    : [1.0, 2.0, 3.0]
      strength : [0.5, 1.0, 1.5, 2.0]

    Each cell contains:
      - full 512×512 result
      - crop 150×150 px centred on the image (vessel-rich zone in retinal imaging)
      - sharpness score L (Laplacian variance inside mask) + % gain vs reference

    Saved at high resolution for side-by-side visual comparison.
    """
    SIGMAS    = [2.0, 4.0, 6.0]
    STRENGTHS = [1.0, 2.0, 3.0, 5.0]

    frames, _ = _read_video_gray(video_path)
    H, W = frames.shape[1], frames.shape[2]
    mid = frames[len(frames) // 2]

    # Crop centred on the image — typically vessel-rich for retinal imaging
    crop_h = crop_w = 150
    cy = max(0, (H - crop_h) // 2)
    cx = max(0, (W - crop_w) // 2)

    mask_bool    = mask.astype(bool)
    ref_sharpness = _laplacian_variance(mid, mask)

    n_rows = len(SIGMAS)
    n_cols = len(STRENGTHS)

    # Each sigma row has 2 sub-rows: full view + crop
    fig, axes = plt.subplots(
        2 * n_rows, n_cols + 1,
        figsize=(3.8 * (n_cols + 1), 4.2 * n_rows),
        gridspec_kw={"hspace": 0.04, "wspace": 0.03},
    )

    for row, sig in enumerate(SIGMAS):
        # Column 0: reference (unchanged)
        for sub, img in [(0, mid), (1, mid[cy:cy+crop_h, cx:cx+crop_w])]:
            ax = axes[2*row + sub, 0]
            ax.imshow(img, cmap="gray", vmin=0, vmax=255)
            ax.axis("off")
            if row == 0 and sub == 0:
                ax.set_title("Référence", fontsize=8, fontweight="bold", color="seagreen")
            if sub == 0:
                ax.set_ylabel(
                    f"σ={sig}",
                    fontsize=8, rotation=0, ha="right", va="center", labelpad=40,
                )
            if sub == 1 and row == 0:
                ax.set_ylabel("crop 150px", fontsize=6, rotation=0,
                              ha="right", va="center", labelpad=50)

        for col_idx, strength in enumerate(STRENGTHS):
            sharpened = ski_unsharp_mask(mid, radius=sig, amount=strength, preserve_range=True)
            out = np.zeros((H, W), dtype=np.uint8)
            out[mask_bool] = np.clip(sharpened[mask_bool], 0, 255).astype(np.uint8)
            crop      = out[cy:cy+crop_h, cx:cx+crop_w]
            sharpness = _laplacian_variance(out, mask)
            gain_pct  = (sharpness - ref_sharpness) / (ref_sharpness + 1e-9) * 100

            for sub, img in [(0, out), (1, crop)]:
                ax = axes[2*row + sub, col_idx + 1]
                ax.imshow(img, cmap="gray", vmin=0, vmax=255)
                ax.axis("off")
                if row == 0 and sub == 0:
                    ax.set_title(f"strength={strength}", fontsize=8, fontweight="bold")
                if sub == 0:
                    ax.text(
                        0.02, 0.97,
                        f"L={sharpness:.0f}  {gain_pct:+.0f}%",
                        transform=ax.transAxes,
                        fontsize=6, color="yellow", va="top",
                        bbox=dict(facecolor="black", alpha=0.45, pad=1, linewidth=0),
                    )

    plt.suptitle(
        "Grille de calibration USM  —  12 combinaisons\n"
        "Lignes : sigma (rayon Gaussien)  |  Colonnes : strength (intensité)\n"
        "L = variance Laplacien dans le masque (netteté)  |  % = gain vs référence\n"
        "Chercher : vaisseaux fins nets, pas de halo, fond homogène inchangé",
        fontsize=10,
    )
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Grille de calibration → {output_path}")


def compare_usm(
    video_before: str,
    video_after: str,
    mask: np.ndarray,
    output_path: str,
    n_samples: int = 5,
    diff_amplification: float = 5.0,
) -> None:
    """
    Diagnostic comparison grid: n_samples evenly-spaced frames × 3 rows.

    Rows:
      0 — Avant USM
      1 — Après USM
      2 — Différence ×diff_amplification (centred on 128)

    A correct USM shows only thin bright edges in the difference panel.
    If background grain is visible in the difference → strength is too high.

    Args:
        video_before:       Path to the video before USM.
        video_after:        Path to the video after USM.
        mask:               Binary mask (H, W) uint8 from step [1].
        output_path:        Destination PNG path.
        n_samples:          Number of sample frames (default 5).
        diff_amplification: Multiplier for the difference panel (default 5).
    """
    frames_before, _ = _read_video_gray(video_before)
    frames_after,  _ = _read_video_gray(video_after)

    N = min(len(frames_before), len(frames_after))
    indices   = np.linspace(0, N - 1, n_samples, dtype=int)
    mask_bool = mask.astype(bool)

    fig, axes = plt.subplots(
        3, n_samples,
        figsize=(4.0 * n_samples, 10),
        gridspec_kw={"hspace": 0.05, "wspace": 0.03},
    )
    if n_samples == 1:
        axes = axes.reshape(3, 1)

    row_labels = ["Avant USM", "Après USM", f"Différence ×{diff_amplification:.0f}"]

    for col, idx in enumerate(indices):
        bef = frames_before[idx]
        aft = frames_after[idx]

        diff_f   = (aft.astype(np.float32) - bef.astype(np.float32)) * diff_amplification
        diff_img = np.clip(diff_f + 128, 0, 255).astype(np.uint8)
        diff_img[~mask_bool] = 0   # zero outside mask for a clean diff panel

        for row, img in enumerate([bef, aft, diff_img]):
            ax = axes[row, col]
            ax.imshow(img, cmap="gray", vmin=0, vmax=255)
            ax.axis("off")
            ax.set_title(f"Frame {idx}", fontsize=8)
            if col == 0:
                ax.set_ylabel(row_labels[row], fontsize=9, rotation=0,
                              ha="right", va="center", labelpad=60)

        for row, src in [(0, bef), (1, aft)]:
            s = _laplacian_variance(src, mask)
            axes[row, col].text(
                0.02, 0.02,
                f"L={s:.0f}",
                transform=axes[row, col].transAxes,
                fontsize=7, color="yellow", va="bottom",
                bbox=dict(facecolor="black", alpha=0.4, pad=1, linewidth=0),
            )

    plt.suptitle(
        "Comparaison Avant / Après USM\n"
        "Différence ×5 : seuls les bords fins doivent apparaître — "
        "si du bruit de fond est visible → strength trop élevé",
        fontsize=10,
    )
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Grille de comparaison → {output_path}")


# ── CLI entry point ───────────────────────────────────────────────────────────


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(
        description=(
            "Step [10] — Unsharp Mask post-processing.\n\n"
            "Workflow recommandé :\n"
            "  1. python UnsharpMask.py <video.avi> --calibrate\n"
            "  2. Ouvrir step10_usm_calibration.png, choisir les meilleurs paramètres\n"
            "  3. python UnsharpMask.py <video.avi> --sigma S --strength A"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("input",
        help="Vidéo .avi à traiter (output step [9] CLAHE) ou dossier la contenant.")
    parser.add_argument("--mask", default=None,
        help="Chemin vers mask.png (défaut : mask.png dans le même dossier).")
    parser.add_argument("--output-dir", default=None,
        help="Dossier de sortie (défaut : même dossier que la vidéo).")
    parser.add_argument("--calibrate", action="store_true",
        help="Générer la grille de calibration 12 combinaisons (sans traiter la vidéo).")
    parser.add_argument("--sigma", type=float, default=USM_SIGMA,
        help=f"Rayon Gaussien (défaut {USM_SIGMA}).")
    parser.add_argument("--strength", type=float, default=USM_STRENGTH,
        help=f"Intensité du rehaussement 0–1 (défaut {USM_STRENGTH}).")
    args = parser.parse_args()

    inp = Path(args.input)
    if inp.is_dir():
        video_in  = inp / "step9_clahe_usm.avi"
        mask_path = inp / "mask.png"
        out_dir   = inp
    else:
        video_in  = inp
        mask_path = Path(args.mask) if args.mask else inp.parent / "mask.png"
        out_dir   = Path(args.output_dir) if args.output_dir else inp.parent

    if not video_in.exists():
        sys.exit(f"ERREUR : vidéo introuvable : {video_in}")
    if not mask_path.exists():
        sys.exit(f"ERREUR : masque introuvable : {mask_path}")

    out_dir.mkdir(parents=True, exist_ok=True)
    mask = load_mask(str(mask_path))

    print(f"\n[10] Unsharp Mask")
    print(f"     Entrée  : {video_in}")
    print(f"     Masque  : {mask_path}")

    if args.calibrate:
        out_calib = out_dir / "step10_usm_calibration.png"
        print(f"\n  Mode calibration — grille 12 combinaisons → {out_calib}\n")
        calibration_grid_usm(str(video_in), mask, str(out_calib))
        print(
            "\n  ── Comment utiliser la grille ──────────────────────────────\n"
            "  1. Ouvrir step10_usm_calibration.png à 100% dans un viewer d'images.\n"
            "  2. Chaque ligne = sigma, chaque colonne = strength.\n"
            "  3. Chercher la vignette où :\n"
            "       • les vaisseaux fins sont plus nets (score L élevé)\n"
            "       • pas de halo blanc autour des vaisseaux\n"
            "       • le fond noir reste homogène (pas de bruit amplifié)\n"
            "  4. Lire les paramètres de la vignette choisie.\n"
            "  5. Relancer sans --calibrate avec ces valeurs :\n"
            "     python UnsharpMask.py <video.avi> --sigma S --strength A"
        )
    else:
        out_video = out_dir / "step10_usm.avi"
        out_cmp   = out_dir / "step10_usm_comparison.png"

        print(f"     Sortie  : {out_video}")
        print(f"     sigma={args.sigma}  strength={args.strength}\n")

        apply_usm_to_video(
            video_path  = str(video_in),
            mask        = mask,
            output_path = str(out_video),
            sigma       = args.sigma,
            strength    = args.strength,
        )

        compare_usm(
            video_before = str(video_in),
            video_after  = str(out_video),
            mask         = mask,
            output_path  = str(out_cmp),
        )

    print("\nDone.")


if __name__ == "__main__":
    main()
