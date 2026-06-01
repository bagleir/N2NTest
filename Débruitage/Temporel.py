#!/usr/bin/env python3
"""
Step [7] — Temporal median filter for denoising preprocessed ocular vascular videos.

Expects the output directory of Pretraitement.py (step6_corrected.avi + mask.png).
Can be run standalone (see __main__) or imported by the full pipeline.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
from scipy.signal.windows import gaussian

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).parent))
from mask_detection import load_mask


# ── Parameters (edit here) ────────────────────────────────────────────────────

WINDOW_SIZE               = 5      # temporal window width in frames (odd, ≤ 7)
GAUSSIAN_WEIGHTS          = True   # True → weighted median; False → uniform median
FPS                       = 30.0   # acquisition frame rate (used for FFT axis)
CARDIAC_FREQ_MIN          = 0.5    # Hz — lower bound of cardiac band
CARDIAC_FREQ_MAX          = 3.0    # Hz — upper bound of cardiac band
PULSATION_AMP_THRESHOLD   = 0.80   # warn if cardiac amplitude drops below this fraction


# ── Internal helpers ──────────────────────────────────────────────────────────


def _read_video_gray(path: str | Path) -> tuple[np.ndarray, float]:
    """
    Read all frames from a video into a float32 array.

    Returns:
        frames : np.ndarray (N, H, W) float32 in [0, 255]
        fps    : frame rate reported by the container
    """
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
        frames.append(frame.astype(np.float32))
    cap.release()

    if not frames:
        raise ValueError(f"No frames read from {path}")

    return np.stack(frames, axis=0), fps


def _write_video(frames: np.ndarray, output_path: str | Path, fps: float) -> None:
    """Write a float32 (N, H, W) array to a grayscale AVI file."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    N, H, W = frames.shape
    fourcc = cv2.VideoWriter_fourcc(*"XVID")
    writer = cv2.VideoWriter(str(output_path), fourcc, fps, (W, H), isColor=False)

    for i in range(N):
        writer.write(np.clip(frames[i], 0, 255).astype(np.uint8))
    writer.release()


def _estimate_noise_laplacian(frame: np.ndarray, mask: Optional[np.ndarray] = None) -> float:
    """
    Estimate noise σ via the Laplacian method (Immerkaer 1996).
    Operates only inside the mask when provided.
    """
    lap = cv2.Laplacian(frame.astype(np.float64), cv2.CV_64F)
    values = lap[mask.astype(bool)] if mask is not None else lap.ravel()
    if values.size == 0:
        return 0.0
    # Normalisation Immerkaer: sigma ≈ sqrt(pi/2) * MAD / (6 * n_px)
    return float(np.sqrt(np.pi / 2.0) * np.sum(np.abs(values)) / (6.0 * values.size))


def _build_gaussian_weights(window_size: int) -> np.ndarray:
    """Return normalized Gaussian weights centered on the middle frame."""
    sigma = window_size / 4.0
    w = gaussian(window_size, sigma).astype(np.float32)
    return w / w.sum()


def _weighted_median(window: np.ndarray, weights: np.ndarray) -> np.ndarray:
    """
    Vectorized weighted median along the temporal axis.

    Args:
        window  : float32 (T, H, W) — T frames in the current window
        weights : float32 (T,)      — normalized weights (sum to 1)

    Returns:
        np.ndarray (H, W) — weighted median at each pixel
    """
    # Sort each pixel's values across time; track which frame index lands where
    sort_idx     = np.argsort(window, axis=0)                        # (T, H, W)
    sorted_vals  = np.take_along_axis(window, sort_idx, axis=0)     # (T, H, W)

    # Assign each sorted slot the weight of the frame it came from
    sorted_w     = weights[sort_idx]                                 # (T, H, W)

    # Weighted median = first slot where cumulative weight reaches 0.5
    cum_w        = np.cumsum(sorted_w, axis=0)                       # (T, H, W)
    rank         = np.argmax(cum_w >= 0.5, axis=0)                  # (H, W)

    H, W = window.shape[1], window.shape[2]
    ri = np.arange(H, dtype=np.intp)[:, None]
    ci = np.arange(W, dtype=np.intp)[None, :]
    return sorted_vals[rank, ri, ci]                                 # (H, W)


# ── Core filter ───────────────────────────────────────────────────────────────


def temporal_median_filter(
    video_path: str,
    mask: np.ndarray,
    output_path: str,
    window_size: int = WINDOW_SIZE,
    gaussian_weights: bool = GAUSSIAN_WEIGHTS,
) -> dict:
    """
    Apply temporal median filter to a preprocessed video.

    The filter operates only inside the circular mask.  At the borders of the
    video the window is made asymmetric (only available frames are used) rather
    than padding with repeated frames.

    Args:
        video_path       : Path to the input (preprocessed) video.
        mask             : Binary mask (H, W) uint8, 255 inside the useful zone.
        output_path      : Destination path for the filtered video.
        window_size      : Number of frames in the temporal window (odd, ≤ 7).
        gaussian_weights : If True use Gaussian-weighted median, else uniform.

    Returns:
        dict with:
          'noise_reduction_ratio' : float, σ_before / σ_after
          'mean_diff_per_frame'   : np.ndarray (N,), mean |diff| per frame
          'processing_time_ms'    : float, mean processing time per frame (ms)
    """
    if window_size % 2 == 0:
        raise ValueError(f"window_size must be odd, got {window_size}")
    if window_size > 7:
        raise ValueError(f"window_size must be ≤ 7, got {window_size}")

    half      = window_size // 2
    mask_bool = mask.astype(bool)
    w_full    = _build_gaussian_weights(window_size) if gaussian_weights else None

    # ── Load all frames into RAM ──────────────────────────────────────────────
    print("  Loading video …", end=" ", flush=True)
    frames, fps = _read_video_gray(video_path)
    N, H, W = frames.shape
    print(f"{N} frames @ {fps:.1f} fps  ({H}×{W})")

    # ── Noise before filtering (sample 20 evenly-spaced frames) ──────────────
    sample_idx = np.linspace(0, N - 1, min(20, N), dtype=int)
    sigma_before = float(np.mean([_estimate_noise_laplacian(frames[i], mask) for i in sample_idx]))

    # ── Frame-by-frame processing ─────────────────────────────────────────────
    filtered       = np.empty_like(frames)
    mean_diff      = np.empty(N, dtype=np.float32)
    frame_times: list[float] = []

    for i in range(N):
        t0 = time.perf_counter()

        # Asymmetric window: only available frames at the borders
        i_start = max(0, i - half)
        i_end   = min(N, i + half + 1)
        window  = frames[i_start:i_end]          # (T, H, W), T ≤ window_size

        if gaussian_weights and w_full is not None:
            # Trim the precomputed weights to match the actual window length
            left_cut = max(0, half - i)
            w        = w_full[left_cut: left_cut + (i_end - i_start)]
            w        = w / w.sum()               # renormalize after trimming
            result   = _weighted_median(window, w)
        else:
            result = np.median(window, axis=0).astype(np.float32)

        # Copy original outside mask, filtered inside
        out = frames[i].copy()
        out[mask_bool] = result[mask_bool]
        filtered[i]    = out

        dt = (time.perf_counter() - t0) * 1_000.0
        frame_times.append(dt)

        diff = np.abs(frames[i][mask_bool] - out[mask_bool])
        mean_diff[i] = float(diff.mean()) if diff.size else 0.0

    # ── Noise after filtering ─────────────────────────────────────────────────
    sigma_after = float(np.mean([_estimate_noise_laplacian(filtered[i], mask) for i in sample_idx]))

    # ── Write output ──────────────────────────────────────────────────────────
    _write_video(filtered, output_path, fps)

    avg_ms = float(np.mean(frame_times))
    ratio  = (sigma_before / sigma_after) if sigma_after > 0 else float("inf")

    # ── Terminal summary ──────────────────────────────────────────────────────
    print(f"\n  ── Temporal median filter ──────────────────────────────────")
    print(f"  Fenêtre             : {window_size} frames"
          f"  ({'Gaussienne' if gaussian_weights else 'uniforme'})")
    print(f"  Temps moyen/frame   : {avg_ms:.2f} ms")
    print(f"  Sigma bruit (avant) : {sigma_before:.4f}")
    print(f"  Sigma bruit (après) : {sigma_after:.4f}")
    print(f"  Ratio de réduction  : {ratio:.2f}×")
    print(f"  Sortie              : {output_path}")

    return {
        "noise_reduction_ratio": ratio,
        "mean_diff_per_frame":   mean_diff,
        "processing_time_ms":    avg_ms,
    }


# ── Visualization ─────────────────────────────────────────────────────────────


def compare_before_after(
    video_original: str,
    video_preprocessed: str,
    video_filtered: str,
    mask: np.ndarray,
    output_path: str,
    n_samples: int = 5,
) -> None:
    """
    Save a grid (n_samples rows × 4 columns) comparing the full pipeline.

    Columns:
      (1) Vidéo brute originale
      (2) Vidéo prétraitée (étapes 1-6)
      (3) Vidéo après median temporel
      (4) Différence prétraité vs filtré ×5
    """
    orig, _  = _read_video_gray(video_original)
    pre,  _  = _read_video_gray(video_preprocessed)
    filt, _  = _read_video_gray(video_filtered)

    N = min(len(pre), len(filt))
    indices = np.linspace(0, N - 1, n_samples, dtype=int)

    col_titles = [
        "Original brut",
        "Prétraité (étapes 1-6)",
        "Après median temporel",
        "Diff. prétraité→filtré ×5",
    ]
    cmaps = ["gray", "gray", "gray", "hot"]

    fig, axes = plt.subplots(n_samples, 4, figsize=(16, n_samples * 4))

    for row, idx in enumerate(indices):
        orig_idx = min(idx, len(orig) - 1)
        diff_amp = np.clip(np.abs(pre[idx] - filt[idx]) * 5, 0, 255).astype(np.uint8)
        imgs = [
            orig[orig_idx].astype(np.uint8),
            pre[idx].astype(np.uint8),
            filt[idx].astype(np.uint8),
            diff_amp,
        ]
        for col, (img, cmap) in enumerate(zip(imgs, cmaps)):
            ax = axes[row, col]
            ax.imshow(img, cmap=cmap, vmin=0, vmax=255)
            ax.axis("off")
            if row == 0:
                ax.set_title(col_titles[col], fontsize=9, fontweight="bold")
            if col == 0:
                ax.set_ylabel(f"Frame {idx}", fontsize=8)

    plt.suptitle("Comparaison pipeline complet — étapes 1 → 7", fontsize=11)
    plt.tight_layout()
    plt.savefig(output_path, dpi=120, bbox_inches="tight")
    plt.close()
    print(f"  Grille de comparaison → {output_path}")


def visualize_noise_reduction(stats_dict: dict, output_path: str) -> None:
    """
    Two-panel diagnostic plot for the filter results:
      (1) Mean diff per frame — should be flat (no temporal vascular structure removed)
      (2) Histogram of diff values — should be symmetric and zero-centered (no bias)
    """
    diffs = stats_dict["mean_diff_per_frame"]
    ratio = stats_dict.get("noise_reduction_ratio", float("nan"))
    N     = len(diffs)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

    ax1.plot(np.arange(N), diffs, linewidth=0.7, color="steelblue")
    ax1.axhline(diffs.mean(), color="crimson", linestyle="--", linewidth=1.0,
                label=f"Moyenne = {diffs.mean():.3f}")
    ax1.fill_between(np.arange(N),
                     diffs.mean() - diffs.std(),
                     diffs.mean() + diffs.std(),
                     alpha=0.15, color="steelblue")
    ax1.set_xlabel("Indice de frame")
    ax1.set_ylabel("Différence moyenne (niveaux de gris)")
    ax1.set_title("Diff. prétraité → filtré par frame\n"
                  "(courbe plate = bruit supprimé, pas de signal vasculaire)")
    ax1.legend(fontsize=8)

    ax2.hist(diffs, bins=40, color="coral", edgecolor="white", linewidth=0.4)
    ax2.axvline(diffs.mean(), color="darkred", linestyle="--", linewidth=1.0,
                label=f"μ = {diffs.mean():.3f}")
    ax2.axvline(0, color="black", linestyle=":", linewidth=0.8)
    ax2.set_xlabel("Valeur de différence")
    ax2.set_ylabel("Fréquence")
    ax2.set_title(f"Histogramme des différences\n"
                  f"(ratio réduction bruit = {ratio:.2f}×  —  centré sur 0 = pas de biais)")
    ax2.legend(fontsize=8)

    plt.tight_layout()
    plt.savefig(output_path, dpi=120, bbox_inches="tight")
    plt.close()
    print(f"  Graphique réduction bruit → {output_path}")


def compute_pulsation_preservation(
    video_preprocessed: str,
    video_filtered: str,
    mask: np.ndarray,
    output_path: str,
    fps: float = FPS,
) -> dict:
    """
    Validate that the cardiac pulsation is preserved after temporal filtering.

    Computes mean luminosity per frame inside the mask, de-trends both curves,
    then compares the FFT amplitude in the cardiac band [0.5–3.0 Hz].
    Issues a WARNING if amplitude drops below PULSATION_AMP_THRESHOLD (80 %).

    Returns:
        dict with 'cardiac_amplitude_ratio' (float) and 'dominant_freq_hz' (float)
    """
    mask_bool = mask.astype(bool)

    pre,  fps_detected = _read_video_gray(video_preprocessed)
    filt, _            = _read_video_gray(video_filtered)
    fps = fps_detected or fps

    N = min(len(pre), len(filt))
    t = np.arange(N)

    lum_pre  = np.array([pre[i][mask_bool].mean()  for i in range(N)], dtype=np.float64)
    lum_filt = np.array([filt[i][mask_bool].mean() for i in range(N)], dtype=np.float64)

    # De-trend: remove DC + linear drift before FFT so the cardiac peak is visible
    lum_pre  -= np.polyval(np.polyfit(t, lum_pre,  1), t)
    lum_filt -= np.polyval(np.polyfit(t, lum_filt, 1), t)

    freqs    = np.fft.rfftfreq(N, d=1.0 / fps)
    fft_pre  = np.abs(np.fft.rfft(lum_pre))
    fft_filt = np.abs(np.fft.rfft(lum_filt))

    cardiac  = (freqs >= CARDIAC_FREQ_MIN) & (freqs <= CARDIAC_FREQ_MAX)

    if not np.any(cardiac):
        print("  AVERTISSEMENT : résolution temporelle insuffisante pour la bande cardiaque.")
        return {"cardiac_amplitude_ratio": float("nan"), "dominant_freq_hz": float("nan")}

    amp_pre   = fft_pre[cardiac].max()
    amp_filt  = fft_filt[cardiac].max()
    ratio     = float(amp_filt / amp_pre) if amp_pre > 0 else float("nan")
    dom_freq  = float(freqs[cardiac][np.argmax(fft_pre[cardiac])])
    preserved = ratio >= PULSATION_AMP_THRESHOLD

    print(f"\n  ── Préservation de la pulsation ────────────────────────────")
    print(f"  Fréquence cardiaque dominante : {dom_freq:.2f} Hz")
    print(f"  Amplitude cardiaque (avant)   : {amp_pre:.4f}")
    print(f"  Amplitude cardiaque (après)   : {amp_filt:.4f}")
    print(f"  Ratio d'amplitude             : {ratio * 100:.1f} %")
    if preserved:
        print(f"  Pulsation préservée           : OUI  ({ratio * 100:.1f}% ≥ {PULSATION_AMP_THRESHOLD*100:.0f}%)")
    else:
        print(f"  Pulsation préservée           : NON  ({ratio * 100:.1f}% < {PULSATION_AMP_THRESHOLD*100:.0f}%)")
        print(f"  → Réduire WINDOW_SIZE (actuellement {WINDOW_SIZE})")

    # ── Plot ──────────────────────────────────────────────────────────────────
    time_axis = t / fps
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))

    ax1.plot(time_axis, lum_pre,  lw=0.8, color="steelblue", label="Prétraité")
    ax1.plot(time_axis, lum_filt, lw=0.8, color="coral",     label="Après filtrage", ls="--")
    ax1.set_xlabel("Temps (s)")
    ax1.set_ylabel("Luminosité moyenne centrée")
    ax1.set_title("Signal de luminosité dans le masque (dérivé)")
    ax1.legend(fontsize=9)

    ax2.plot(freqs, fft_pre,  lw=0.8, color="steelblue", label="Prétraité")
    ax2.plot(freqs, fft_filt, lw=0.8, color="coral",     label="Après filtrage", ls="--")
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


# ── Single-video processing helper ───────────────────────────────────────────


def _process_one(
    video_pre: Path,
    mask_path: Path,
    out_dir: Path,
    stem: str,
    window_size: int,
    use_gaussian: bool,
    original_path: Optional[Path] = None,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    out_video   = out_dir / f"{stem}_step7_temporal_median.avi"
    out_stats   = out_dir / f"{stem}_step7_noise_reduction.png"
    out_puls    = out_dir / f"{stem}_step7_pulsation.png"
    out_compare = (out_dir / f"{stem}_step7_comparison.png") if original_path else None

    mask = load_mask(str(mask_path))

    print(f"\n[7] Temporal median filter — {stem}")
    print(f"    Entrée  : {video_pre}")
    print(f"    Masque  : {mask_path}")
    print(f"    Sortie  : {out_video}")
    print(f"    Fenêtre : {window_size} frames  ({'Gaussienne' if use_gaussian else 'uniforme'})\n")

    stats = temporal_median_filter(
        video_path       = str(video_pre),
        mask             = mask,
        output_path      = str(out_video),
        window_size      = window_size,
        gaussian_weights = use_gaussian,
    )

    visualize_noise_reduction(stats, str(out_stats))

    compute_pulsation_preservation(
        video_preprocessed = str(video_pre),
        video_filtered     = str(out_video),
        mask               = mask,
        output_path        = str(out_puls),
    )

    if original_path and out_compare:
        compare_before_after(
            video_original     = str(original_path),
            video_preprocessed = str(video_pre),
            video_filtered     = str(out_video),
            mask               = mask,
            output_path        = str(out_compare),
        )


# ── CLI entry point ───────────────────────────────────────────────────────────


def _find_mask(video_path: Path) -> Optional[Path]:
    """
    Locate the mask for a given video file.

    Search order:
      1. <stem>_mask.png  beside the video  (Pretraitement default mode)
      2. mask.png         beside the video  (Pretraitement --info mode, single-video call)
      3. mask.png         in the parent dir (Pretraitement --info mode, subdir layout)
    """
    candidates = [
        video_path.parent / f"{video_path.stem}_mask.png",
        video_path.parent / "mask.png",
        video_path.parent.parent / "mask.png",
    ]
    for c in candidates:
        if c.exists():
            return c
    return None


def _is_pretreat_dir(p: Path) -> bool:
    return (p / "step6_corrected.avi").exists()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Step [7] — Temporal median filter for preprocessed ocular videos.",
    )
    parser.add_argument(
        "input",
        help=(
            "Dossier contenant les vidéos prétraitées par Pretraitement.py "
            "(mode défaut : <stem>.avi + <stem>_mask.png ; mode --info : sous-dossiers <stem>/), "
            "ou chemin direct vers une vidéo .avi."
        ),
    )
    parser.add_argument(
        "--output", default=None,
        help=(
            "Dossier de sortie pour toutes les vidéos filtrées. "
            "Par défaut : même dossier que l'entrée."
        ),
    )
    parser.add_argument(
        "--window", type=int, default=WINDOW_SIZE,
        help=f"Taille de la fenêtre temporelle (impair, ≤ 7 ; défaut {WINDOW_SIZE})",
    )
    parser.add_argument(
        "--no-gaussian", action="store_true",
        help="Désactiver la pondération gaussienne (median uniforme)",
    )
    parser.add_argument(
        "--original", default=None,
        help=(
            "Vidéo unique : chemin vers la vidéo brute originale. "
            "Mode batch : dossier contenant les vidéos brutes (nommées <stem>.avi)."
        ),
    )
    args = parser.parse_args()

    inp          = Path(args.input)
    use_gaussian = not args.no_gaussian

    if not inp.exists():
        sys.exit(f"ERREUR : chemin introuvable : {inp}")

    # ── Vidéo unique passée directement ──────────────────────────────────────
    if inp.is_file():
        mask_path = _find_mask(inp)
        if mask_path is None:
            sys.exit(f"ERREUR : aucun masque trouvé pour {inp}")
        stem    = inp.stem
        out_dir = Path(args.output) if args.output else inp.parent
        orig    = Path(args.original) if args.original else None
        _process_one(inp, mask_path, out_dir, stem, args.window, use_gaussian, orig)
        sys.exit(0)

    # ── Dossier en entrée ────────────────────────────────────────────────────
    # Collect every .avi that is not itself a step7 output
    videos = sorted(
        f for f in inp.glob("*.avi") if "_step7_" not in f.name
    )

    # Also handle Pretraitement --info layout: subdirs with step6_corrected.avi
    info_subdirs = sorted(d for d in inp.iterdir() if d.is_dir() and _is_pretreat_dir(d))
    for sub in info_subdirs:
        videos.append(sub / "step6_corrected.avi")

    if not videos:
        sys.exit(f"ERREUR : aucune vidéo prétraitée trouvée dans {inp}")

    out_dir       = Path(args.output) if args.output else inp
    originals_dir = Path(args.original) if args.original else None

    print(f"Mode batch : {len(videos)} vidéo(s) détectée(s) dans {inp}")
    print(f"Dossier de sortie : {out_dir}\n")

    failed: list[str] = []
    for video_pre in videos:
        stem      = video_pre.stem if video_pre.stem != "step6_corrected" else video_pre.parent.name
        mask_path = _find_mask(video_pre)
        if mask_path is None:
            print(f"  [IGNORÉ] masque introuvable pour : {video_pre.name}")
            failed.append(stem)
            continue
        orig = None
        if originals_dir:
            candidate = originals_dir / f"{stem}.avi"
            if candidate.exists():
                orig = candidate
            else:
                print(f"  [AVERTISSEMENT] vidéo originale introuvable : {candidate}")
        try:
            _process_one(video_pre, mask_path, out_dir, stem, args.window, use_gaussian, orig)
        except Exception as exc:
            print(f"  [ERREUR] {stem} : {exc}")
            failed.append(stem)

    print(f"\n── Bilan batch ──────────────────────────────────────────────")
    print(f"  Traitées avec succès : {len(videos) - len(failed)}/{len(videos)}")
    if failed:
        print(f"  Échecs              : {', '.join(failed)}")
