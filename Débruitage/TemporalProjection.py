#!/usr/bin/env python3
"""
Step [11] — Temporal projections for retinal vascular imaging.

Generates static anatomical maps from the fully-processed video
(pre-processing → temporal median → N2N → CLAHE → USM, step [10]).

Memory-efficient: streams the video in chunks of 50 frames and uses
reservoir sampling for percentile / median (never >reservoir_size frames
in RAM simultaneously). 639 × 512×512 × float32 ≈ 670 MB is never
allocated as a whole.

Projections:
  mean       — average over all frames;          SNR gain ≈ √N
  max        — pixel-wise maximum;               captures peak vessel dilation
  percentile — robust max at P ∈ {75,85,90,95}; standard clinical alternative
  median     — most robust;                      may erase rarely-visible vessels

Usage (recommended workflow):
  python TemporalProjection.py <step10_usm.avi> [--mask mask.png]
      [--output-dir DIR] [--usm-amount 1.5] [--best percentile_90]
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Iterator

import cv2
import numpy as np
from skimage.filters import laplace as ski_laplace
from skimage.filters import unsharp_mask as ski_unsharp_mask

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).parent))
from mask_detection import load_mask


# ─────────────────────────────────────────────────────────────────────────────
# Internal helpers
# ─────────────────────────────────────────────────────────────────────────────


def _progress(current: int, total: int, prefix: str = "", width: int = 40) -> None:
    """Print an in-place ASCII progress bar."""
    pct = current / max(total, 1)
    filled = int(width * pct)
    bar = "█" * filled + "░" * (width - filled)
    print(f"\r  {prefix}[{bar}] {current}/{total} ({pct*100:.0f}%)",
          end="", flush=True)
    if current >= total:
        print()


def _count_frames(video_path: str | Path) -> int:
    cap = cv2.VideoCapture(str(video_path))
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    return max(n, 0)


def _iter_frames_chunked(
    video_path: str | Path,
    chunk_size: int = 50,
) -> Iterator[tuple[np.ndarray, int]]:
    """
    Yield (chunk_float32, cumulative_frame_count) without loading the full video.

    chunk_float32 has shape (n, H, W) with values in [0, 255].
    cumulative_frame_count counts how many frames have been yielded so far
    (suitable for a progress bar denominator = total_frames).
    """
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise FileNotFoundError(f"Cannot open video: {video_path}")

    chunk: list[np.ndarray] = []
    count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            if chunk:
                yield np.stack(chunk, axis=0).astype(np.float32), count
            break
        if frame.ndim == 3:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        chunk.append(frame)
        count += 1
        if len(chunk) == chunk_size:
            yield np.stack(chunk, axis=0).astype(np.float32), count
            chunk = []

    cap.release()


def _normalize_uint8(img: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Stretch the dynamic range *inside the mask* to [0, 255]; zero outside."""
    mask_bool = mask.astype(bool)
    out = np.zeros(img.shape, dtype=np.uint8)
    if not mask_bool.any():
        return out
    vals = img[mask_bool]
    vmin, vmax = float(vals.min()), float(vals.max())
    if vmax > vmin:
        scaled = (img.astype(np.float64) - vmin) / (vmax - vmin) * 255.0
    else:
        scaled = np.zeros_like(img, dtype=np.float64)
    out[mask_bool] = np.clip(scaled[mask_bool], 0, 255).astype(np.uint8)
    return out


# ─────────────────────────────────────────────────────────────────────────────
# Core projection engine
# ─────────────────────────────────────────────────────────────────────────────


def compute_projections(
    video_path: str,
    mask: np.ndarray,
    output_dir: str,
    percentiles: list[int] | None = None,
    chunk_size: int = 50,
    reservoir_size: int = 200,
) -> dict:
    """
    Compute all temporal projections in a *single* streaming pass over the video.

    Algorithm per accumulator type:
      mean       — float64 cumulative sum, divided by total count at the end.
      max        — element-wise running maximum updated chunk by chunk.
      percentile — Vitter's reservoir sampling (Algorithm R): maintains a random
                   sample of reservoir_size frames uniformly drawn from the stream,
                   then np.percentile is applied on that sample at the end.
      median     — same reservoir; np.median applied at the end.

    All output images are uint8 with the dynamic range stretched within the mask.

    Args:
        video_path:     Path to the processed .avi video (step [10] output).
        mask:           Binary mask (H × W) uint8 from step [1].
        output_dir:     Directory where PNG/TIFF/JSON will be written.
        percentiles:    List of percentile values; default [75, 85, 90, 95].
        chunk_size:     Frames per chunk (keep ≤ 50 to stay under 200 MB/chunk).
        reservoir_size: Number of frames kept for percentile/median estimation.

    Returns:
        dict with keys:
          'mean', 'max', 'median'         — np.ndarray uint8 (H × W)
          'percentile_75', ..._85, ..._90, ..._95
                                          — np.ndarray uint8 (H × W)
          'n_frames_used'                 — int
          'processing_time_s'             — float
    """
    if percentiles is None:
        percentiles = [75, 85, 90, 95]

    t0 = time.perf_counter()
    total = _count_frames(video_path)

    sum_acc: np.ndarray | None = None   # float64 (H, W)
    max_acc: np.ndarray | None = None   # float32 (H, W)
    reservoir: np.ndarray | None = None  # float32 (reservoir_size, H, W)
    n_total = 0
    rng = np.random.default_rng(seed=42)

    print(f"  Streaming {total} frames  (chunks={chunk_size}, "
          f"reservoir={reservoir_size}) …")

    for chunk, count in _iter_frames_chunked(video_path, chunk_size):
        n_chunk, H, W = chunk.shape

        # Lazy initialisation on first chunk
        if sum_acc is None:
            sum_acc  = np.zeros((H, W), dtype=np.float64)
            max_acc  = np.full((H, W), -np.inf, dtype=np.float32)
            reservoir = np.empty((reservoir_size, H, W), dtype=np.float32)

        # ── mean accumulator ─────────────────────────────────────────────────
        sum_acc += chunk.sum(axis=0).astype(np.float64)

        # ── max accumulator ──────────────────────────────────────────────────
        np.maximum(max_acc, chunk.max(axis=0), out=max_acc)

        # ── reservoir sampling (Vitter's Algorithm R) ────────────────────────
        for i in range(n_chunk):
            g = n_total + i            # global 0-based frame index
            if g < reservoir_size:
                reservoir[g] = chunk[i]
            else:
                j = int(rng.integers(0, g + 1))
                if j < reservoir_size:
                    reservoir[j] = chunk[i]

        n_total += n_chunk
        _progress(count, total, prefix="Frames ")

    # Ensure the bar reaches 100% if CAP_PROP_FRAME_COUNT was off
    _progress(total, total, prefix="Frames ")

    actual_res = min(n_total, reservoir_size)
    res = reservoir[:actual_res]   # (actual_res, H, W)

    print(f"\n  → {n_total} frames processed, reservoir filled with {actual_res} samples")
    print("  Computing statistics … ", end="", flush=True)

    mean_raw   = (sum_acc / n_total).astype(np.float32)
    median_raw = np.median(res, axis=0).astype(np.float32)

    results: dict = {
        "mean":              _normalize_uint8(mean_raw,  mask),
        "max":               _normalize_uint8(max_acc,   mask),
        "median":            _normalize_uint8(median_raw, mask),
        "n_frames_used":     n_total,
        "processing_time_s": round(time.perf_counter() - t0, 2),
    }

    for p in percentiles:
        pct_raw = np.percentile(res, p, axis=0).astype(np.float32)
        results[f"percentile_{p}"] = _normalize_uint8(pct_raw, mask)

    print(f"done  ({results['processing_time_s']:.1f} s)")

    return results


# ─────────────────────────────────────────────────────────────────────────────
# Post-processing
# ─────────────────────────────────────────────────────────────────────────────


def apply_usm_to_projection(
    projection: np.ndarray,
    mask: np.ndarray,
    radius: float = 1.0,
    amount: float = 1.0,
) -> np.ndarray:
    """
    Apply Unsharp Mask on a projection image via skimage.filters.unsharp_mask.

    Because the projection has already benefited from N-frame noise averaging,
    its SNR is much higher than a single frame — amount can safely reach 1.5–2.0
    without amplifying residual grain.

    Args:
        projection: uint8 grayscale projection (H × W).
        mask:       Binary mask (H × W) uint8 from step [1].
        radius:     Gaussian std-dev in pixels passed to skimage.
        amount:     Sharpening strength; recommended range 1.0–2.0 for projections.

    Returns:
        Sharpened uint8 image; pixels outside the mask are 0.
    """
    mask_bool = mask.astype(bool)
    sharpened = ski_unsharp_mask(
        projection, radius=radius, amount=amount, preserve_range=True
    )
    out = np.zeros_like(projection, dtype=np.uint8)
    out[mask_bool] = np.clip(sharpened[mask_bool], 0, 255).astype(np.uint8)
    return out


# ─────────────────────────────────────────────────────────────────────────────
# Quality scoring
# ─────────────────────────────────────────────────────────────────────────────


def vessel_enhancement_score(
    projection: np.ndarray,
    mask: np.ndarray,
) -> dict[str, float]:
    """
    Compute three complementary quality metrics inside the mask.

    Metrics:
      local_contrast — mean std over 16×16 non-overlapping tiles that are at
                       least half covered by the mask.  Measures fine-grain
                       spatial variation; higher = more small-vessel detail.
      sharpness      — variance of the Laplacian inside the mask.
                       Classic focus / edge-richness proxy; higher = sharper edges.
      snr            — (mean of top-20% pixels − mean of bottom-20% pixels)
                         / std of all mask pixels.
                       Quantifies vessel-to-background separation regardless of
                       whether vessels are bright or dark in the modality.
      combined_score — geometric mean of the three metrics (all ≥ 0), comparable
                       across projections computed from the same video.

    Returns:
        dict with 'local_contrast', 'sharpness', 'snr', 'combined_score'.
    """
    mask_bool = mask.astype(bool)
    img = projection.astype(np.float64)
    tile = 16
    H, W = img.shape

    # ── local contrast ───────────────────────────────────────────────────────
    stds: list[float] = []
    for r in range(0, H - tile + 1, tile):
        for c in range(0, W - tile + 1, tile):
            tm = mask_bool[r:r + tile, c:c + tile]
            if tm.sum() >= (tile * tile) // 2:
                stds.append(float(img[r:r + tile, c:c + tile][tm].std()))
    local_contrast = float(np.mean(stds)) if stds else 0.0

    # ── sharpness (Laplacian variance) ───────────────────────────────────────
    lap = ski_laplace(img)
    sharpness = float(lap[mask_bool].var()) if mask_bool.any() else 0.0

    # ── SNR ──────────────────────────────────────────────────────────────────
    vals = img[mask_bool]
    if vals.size > 10:
        p20 = float(np.percentile(vals, 20))
        p80 = float(np.percentile(vals, 80))
        sigma = float(vals.std()) + 1e-9
        snr = (vals[vals >= p80].mean() - vals[vals <= p20].mean()) / sigma
    else:
        snr = 0.0

    combined = float(
        (max(local_contrast, 1e-9) * max(sharpness, 1e-9) * max(snr, 1e-9)) ** (1 / 3)
    )

    return {
        "local_contrast": round(local_contrast, 3),
        "sharpness":      round(sharpness,      3),
        "snr":            round(float(snr),     4),
        "combined_score": round(combined,       4),
    }


def rank_projections(
    projections: dict,
    mask: np.ndarray,
) -> list[tuple[str, dict]]:
    """
    Score all projections, print a ranked table, return sorted list.

    Returns:
        [(name, score_dict), ...] sorted by combined_score descending.
    """
    proj_keys = [
        k for k in projections
        if k not in ("n_frames_used", "processing_time_s")
    ]

    scores: list[tuple[str, dict]] = [
        (name, vessel_enhancement_score(projections[name], mask))
        for name in proj_keys
    ]
    scores.sort(key=lambda x: x[1]["combined_score"], reverse=True)

    hdr = f"  {'#':<4} {'Projection':<20} {'Contraste local':<18} {'Netteté':<14} {'SNR':<10} Score combiné"
    sep = f"  {'─'*4} {'─'*20} {'─'*18} {'─'*14} {'─'*10} {'─'*14}"
    print("\n  ── Classement des projections ──────────────────────────────────")
    print(hdr)
    print(sep)
    for rank, (name, s) in enumerate(scores, 1):
        marker = " ◀ meilleure" if rank == 1 else ""
        print(
            f"  {rank:<4} {name:<20} "
            f"{s['local_contrast']:<18.2f}"
            f"{s['sharpness']:<14.2f}"
            f"{s['snr']:<10.3f}"
            f"{s['combined_score']:.4f}{marker}"
        )
    print()

    return scores


# ─────────────────────────────────────────────────────────────────────────────
# Visualisation
# ─────────────────────────────────────────────────────────────────────────────


def comparison_grid(
    projections: dict,
    mask: np.ndarray,
    output_path: str,
    usm_radius: float = 1.0,
    usm_amount: float = 1.0,
    crop_size: int = 200,
) -> None:
    """
    Save a grid comparing every projection type with and without USM.

    Layout (one row per projection type):
      Col 0 — Original projection          (full 512×512)
      Col 1 — + USM                        (full 512×512)
      Col 2 — Crop {crop_size}px (original)
      Col 3 — Crop {crop_size}px (+ USM)

    The crop is centred on the image — typically the most vessel-rich zone
    in retinal imaging; differences between projections are most visible there.
    Each cell is annotated with its projection name and the combined score.
    """
    # Ordered: mean, median, max, percentile_75, 85, 90, 95
    def _sort_key(k: str) -> tuple:
        order = {"mean": 0, "median": 1, "max": 2}
        if k in order:
            return (order[k], 0)
        if k.startswith("percentile_"):
            return (3, int(k.split("_")[1]))
        return (4, 0)

    proj_keys = sorted(
        [k for k in projections if k not in ("n_frames_used", "processing_time_s")],
        key=_sort_key,
    )

    # Pre-compute scores for annotations
    scores_map = {
        name: vessel_enhancement_score(projections[name], mask)
        for name in proj_keys
    }

    H, W = mask.shape[:2]
    cy = max(0, (H - crop_size) // 2)
    cx = max(0, (W - crop_size) // 2)

    n_rows = len(proj_keys)
    n_cols = 4
    fig, axes = plt.subplots(
        n_rows, n_cols,
        figsize=(4.2 * n_cols, 3.0 * n_rows),
        gridspec_kw={"hspace": 0.06, "wspace": 0.03},
    )
    if n_rows == 1:
        axes = axes[np.newaxis, :]

    for c, title in enumerate([
        "Original",
        f"USM  r={usm_radius} a={usm_amount}",
        f"Crop {crop_size}px",
        f"Crop + USM",
    ]):
        axes[0, c].set_title(title, fontsize=9, fontweight="bold")

    for row, name in enumerate(proj_keys):
        img = projections[name]
        usm = apply_usm_to_projection(img, mask, radius=usm_radius, amount=usm_amount)
        score = scores_map[name]["combined_score"]

        panels = [
            img,
            usm,
            img[cy:cy + crop_size, cx:cx + crop_size],
            usm[cy:cy + crop_size, cx:cx + crop_size],
        ]

        for col, panel in enumerate(panels):
            ax = axes[row, col]
            ax.imshow(panel, cmap="gray", vmin=0, vmax=255, interpolation="nearest")
            ax.axis("off")
            ax.text(
                0.02, 0.97,
                f"{name}\nscore={score:.3f}",
                transform=ax.transAxes, fontsize=5.5, color="yellow", va="top",
                bbox=dict(facecolor="black", alpha=0.55, pad=1.5, linewidth=0),
            )

        axes[row, 0].set_ylabel(
            name, fontsize=8, rotation=0, ha="right", va="center", labelpad=65,
        )

    plt.suptitle(
        "Grille des projections temporelles — Original | + USM | Crop | Crop + USM\n"
        f"USM : radius={usm_radius}  amount={usm_amount}  —  "
        f"crop {crop_size}×{crop_size}px centré (zone vasculaire)",
        fontsize=10,
    )

    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(str(out), dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Grille de comparaison        → {out}")


def compare_projection_vs_frame(
    video_path: str,
    projections: dict,
    mask: np.ndarray,
    output_path: str,
    best_key: str = "percentile_90",
    usm_radius: float = 1.0,
    usm_amount: float = 1.0,
    crop_size: int = 200,
    raw_video_path: str | None = None,
) -> None:
    """
    3-panel comparison showing the gain of temporal projection over a single frame.

    Panels:
      0 — Single frame from the processed video (mid-sequence)
          or raw frame if raw_video_path is supplied
      1 — Same frame after full pipeline (the processed video)
      2 — Best projection + USM

    Two rows: full 512×512 view and 200×200 crop centred on the image.

    Args:
        raw_video_path: Optional path to the step-0 (raw) video.  When supplied
                        panel 0 shows the raw frame and panel 1 the processed frame.
                        When omitted, panel 0 and 1 both come from video_path
                        (useful to show projection gain alone).
    """
    def _read_mid(path: str) -> np.ndarray | None:
        cap = cv2.VideoCapture(path)
        n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.set(cv2.CAP_PROP_POS_FRAMES, max(0, n // 2))
        ret, fr = cap.read()
        cap.release()
        if not ret:
            return None
        return cv2.cvtColor(fr, cv2.COLOR_BGR2GRAY) if fr.ndim == 3 else fr

    processed_frame = _read_mid(video_path)
    if processed_frame is None:
        print(f"  WARN: could not read mid frame from {video_path}")
        return

    if raw_video_path is not None:
        raw_frame = _read_mid(raw_video_path)
        if raw_frame is None:
            raw_frame = processed_frame
        panel0_label = "Frame brute (step 0)"
    else:
        raw_frame = processed_frame
        panel0_label = "Frame pipeline complet"

    # Pick projection
    if best_key not in projections:
        best_key = next(
            k for k in projections
            if k not in ("n_frames_used", "processing_time_s")
        )
    best_proj = projections[best_key]
    usm_proj  = apply_usm_to_projection(best_proj, mask, radius=usm_radius, amount=usm_amount)

    H, W = mask.shape[:2]
    cy = max(0, (H - crop_size) // 2)
    cx = max(0, (W - crop_size) // 2)

    panels_full = [raw_frame, processed_frame, usm_proj]
    panels_crop = [
        raw_frame[cy:cy + crop_size, cx:cx + crop_size],
        processed_frame[cy:cy + crop_size, cx:cx + crop_size],
        usm_proj[cy:cy + crop_size, cx:cx + crop_size],
    ]
    labels = [
        panel0_label,
        "Frame pipeline complet" if raw_video_path else "Même frame",
        f"{best_key} + USM",
    ]

    fig, axes = plt.subplots(
        2, 3,
        figsize=(5.0 * 3, 9.0),
        gridspec_kw={"hspace": 0.06, "wspace": 0.03},
    )
    for col, (full, crop, lbl) in enumerate(zip(panels_full, panels_crop, labels)):
        axes[0, col].imshow(full, cmap="gray", vmin=0, vmax=255, interpolation="nearest")
        axes[0, col].axis("off")
        axes[0, col].set_title(lbl, fontsize=9, fontweight="bold")

        axes[1, col].imshow(crop, cmap="gray", vmin=0, vmax=255, interpolation="nearest")
        axes[1, col].axis("off")

    for r, rl in enumerate(["Vue globale", f"Crop {crop_size}×{crop_size}px"]):
        axes[r, 0].set_ylabel(rl, fontsize=9, rotation=0, ha="right",
                              va="center", labelpad=75)

    plt.suptitle(
        "Frame unique  vs  Projection temporelle\n"
        "Les petits vaisseaux invisibles sur une seule frame "
        "apparaissent dans la projection",
        fontsize=10,
    )
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(str(out), dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Frame vs projection          → {out}")


# ─────────────────────────────────────────────────────────────────────────────
# Save results
# ─────────────────────────────────────────────────────────────────────────────


def save_projections(
    projections: dict,
    mask: np.ndarray,
    output_dir: str,
    best_key: str | None = None,
    usm_radius: float = 1.0,
    usm_amount: float = 1.0,
) -> str:
    """
    Write all results to disk and return the key of the best projection.

    Outputs:
      proj_<name>.png            — all projection types (PNG, high quality)
      proj_<best>_usm_best.tiff  — best projection + USM (TIFF, lossless)
      projection_scores.json     — quality scores for each projection
    """
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    proj_keys = [
        k for k in projections
        if k not in ("n_frames_used", "processing_time_s")
    ]

    scores_data: dict[str, dict] = {}
    for name in proj_keys:
        scores_data[name] = vessel_enhancement_score(projections[name], mask)
        cv2.imwrite(str(out / f"proj_{name}.png"), projections[name])

    if best_key is None or best_key not in projections:
        best_key = max(scores_data, key=lambda k: scores_data[k]["combined_score"])

    best_usm = apply_usm_to_projection(
        projections[best_key], mask, usm_radius, usm_amount
    )
    tiff_path = out / f"proj_{best_key}_usm_best.tiff"
    cv2.imwrite(str(tiff_path), best_usm)

    json_payload = {
        "best_projection":   best_key,
        "usm_params":        {"radius": usm_radius, "amount": usm_amount},
        "n_frames_used":     projections.get("n_frames_used"),
        "processing_time_s": projections.get("processing_time_s"),
        "scores":            scores_data,
    }
    json_path = out / "projection_scores.json"
    with open(json_path, "w") as fh:
        json.dump(json_payload, fh, indent=2)

    print(f"  PNG (toutes proj.)           → {out}/proj_*.png")
    print(f"  TIFF lossless (meilleure)    → {tiff_path}")
    print(f"  Scores JSON                  → {json_path}")
    print(f"  Meilleure projection         → {best_key}")

    return best_key


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Step [11] — Temporal projections for retinal vascular video.\n\n"
            "Computes mean / max / percentile / median projections in a single\n"
            "memory-efficient streaming pass and saves PNG, TIFF, and JSON scores."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "input",
        help="Processed .avi video (step [10] USM output) or its parent directory.",
    )
    parser.add_argument(
        "--mask", default=None,
        help="Path to mask.png (default: mask.png next to the video).",
    )
    parser.add_argument(
        "--output-dir", default=None,
        help="Output directory (default: same directory as the video).",
    )
    parser.add_argument(
        "--raw-video", default=None,
        help="Optional path to the raw (step 0) video for the frame-vs-projection "
             "comparison panel.",
    )
    parser.add_argument(
        "--percentiles", nargs="+", type=int, default=[75, 85, 90, 95],
        metavar="P",
        help="Percentile values to compute (default: 75 85 90 95).",
    )
    parser.add_argument(
        "--chunk-size", type=int, default=50,
        help="Frames per streaming chunk (default: 50 ≈ 130 MB/chunk at 512²).",
    )
    parser.add_argument(
        "--reservoir-size", type=int, default=200,
        help="Reservoir sample size for percentile / median (default: 200).",
    )
    parser.add_argument(
        "--usm-radius", type=float, default=1.0,
        help="Gaussian radius for final USM on the projection (default: 1.0).",
    )
    parser.add_argument(
        "--usm-amount", type=float, default=1.0,
        help="Sharpening amount for final USM — up to 2.0 is safe on projections "
             "(default: 1.0).",
    )
    parser.add_argument(
        "--best", default=None,
        help="Force a specific projection as 'best' for TIFF output "
             "(e.g. percentile_90).  Auto-detected by score if omitted.",
    )
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
    if not mask_path.exists():
        sys.exit(f"ERREUR : masque introuvable : {mask_path}")

    out_dir.mkdir(parents=True, exist_ok=True)
    mask = load_mask(str(mask_path))

    print(f"\n[11] Projections temporelles")
    print(f"     Entrée        : {video_in}")
    print(f"     Masque        : {mask_path}")
    print(f"     Sortie        : {out_dir}")
    print(f"     Percentiles   : {args.percentiles}")
    print(f"     Chunk size    : {args.chunk_size}")
    print(f"     Réservoir     : {args.reservoir_size}")
    print(f"     USM radius    : {args.usm_radius}  amount : {args.usm_amount}")
    print()

    # ── 1. Compute projections ────────────────────────────────────────────────
    projections = compute_projections(
        video_path     = str(video_in),
        mask           = mask,
        output_dir     = str(out_dir),
        percentiles    = args.percentiles,
        chunk_size     = args.chunk_size,
        reservoir_size = args.reservoir_size,
    )

    # ── 2. Rank ───────────────────────────────────────────────────────────────
    ranked   = rank_projections(projections, mask)
    best_key = args.best if args.best else ranked[0][0]

    # ── 3. Save PNG / TIFF / JSON ─────────────────────────────────────────────
    best_key = save_projections(
        projections = projections,
        mask        = mask,
        output_dir  = str(out_dir),
        best_key    = best_key,
        usm_radius  = args.usm_radius,
        usm_amount  = args.usm_amount,
    )

    # ── 4. Visual grids ───────────────────────────────────────────────────────
    comparison_grid(
        projections = projections,
        mask        = mask,
        output_path = str(out_dir / "step11_projection_grid.png"),
        usm_radius  = args.usm_radius,
        usm_amount  = args.usm_amount,
    )

    compare_projection_vs_frame(
        video_path      = str(video_in),
        projections     = projections,
        mask            = mask,
        output_path     = str(out_dir / "step11_frame_vs_projection.png"),
        best_key        = best_key,
        usm_radius      = args.usm_radius,
        usm_amount      = args.usm_amount,
        raw_video_path  = args.raw_video,
    )

    print(f"\n  ── Résumé final ────────────────────────────────────────────────")
    print(f"  Meilleure projection  : {best_key}")
    print(f"  TIFF lossless         : {out_dir}/proj_{best_key}_usm_best.tiff")
    print(f"  Grille comparaison    : {out_dir}/step11_projection_grid.png")
    print(f"  Frame vs projection   : {out_dir}/step11_frame_vs_projection.png")
    print(f"  Scores JSON           : {out_dir}/projection_scores.json")
    print("\nDone.")


if __name__ == "__main__":
    main()
