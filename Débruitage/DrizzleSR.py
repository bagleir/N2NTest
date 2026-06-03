#!/usr/bin/env python3
"""
Step [8] — Temporal super-resolution via Drizzle algorithm.

Reconstructs a 1024×1024 image from ~639 stabilized frames (512×512) by
exploiting the natural sub-pixel shifts between frames (residual motion
after rigid stabilization: ~0.1–0.5 px).

Functions:
  estimate_subpixel_shifts   — phase-correlation shift estimation per frame
  drizzle_combine            — vectorized Drizzle accumulation → HR image
  temporal_sr_drizzle        — full pipeline (reference → shifts → drizzle)
  visualize_shifts           — 3-panel shift diagnostic plot
  compare_sr_methods         — Drizzle vs bicubic vs Lanczos comparison grid
  analyze_subpixel_diversity — sub-pixel coverage map and diversity check

Usage (standalone):
  python DrizzleSR.py <video.avi> [output_dir]

  video.avi  : stabilized video (output of step [4] / step [7])
  output_dir : defaults to <video_stem>_drizzle_sr/

  Expects in output_dir (or video parent dir):
    mask.png              — binary circular mask (step [1])
    step2_corrected.json  — corrupted frame log  (step [2])
"""
from __future__ import annotations

import json
import sys
import time
import warnings
from pathlib import Path
from typing import Optional

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import yaml
from skimage.registration import phase_cross_correlation
from tqdm import tqdm

from mask_detection import load_mask


# ── Default parameters (overridden by config.yaml sr: section) ───────────────

SR_SCALE_FACTOR     = 2       # output = input × scale_factor
SR_DROP_SIZE        = 0.7     # drizzle kernel size in HR pixels
SR_UPSAMPLE_FACTOR  = 100     # sub-pixel precision = 1/upsample_factor px
SR_N_FRAMES_MAX     = None    # None = use all valid frames
SR_MIN_FRAMES       = 20      # minimum valid frames to proceed
SR_MAX_SHIFT_PX     = 2.0     # reject frame if total shift > N px
SR_MAX_ERROR        = 0.3     # reject frame if phase-corr error > threshold
SR_N_REFERENCE      = 60      # frames used to build reference median


# ── Config loader ─────────────────────────────────────────────────────────────


def _load_sr_config(config_path: str | Path | None = None) -> dict:
    """Load sr: section from config.yaml, falling back to module defaults."""
    defaults = {
        "scale_factor":    SR_SCALE_FACTOR,
        "drop_size":       SR_DROP_SIZE,
        "upsample_factor": SR_UPSAMPLE_FACTOR,
        "n_frames_max":    SR_N_FRAMES_MAX,
        "min_frames":      SR_MIN_FRAMES,
    }
    if config_path is None:
        config_path = Path(__file__).parent / "config.yaml"
    config_path = Path(config_path)
    if not config_path.exists():
        return defaults
    with open(config_path) as f:
        cfg = yaml.safe_load(f) or {}
    return {**defaults, **cfg.get("sr", {})}


# ── Video helpers ─────────────────────────────────────────────────────────────


def _open_video(video_path: str | Path) -> cv2.VideoCapture:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise FileNotFoundError(f"Cannot open video: {video_path}")
    return cap


def _read_gray_frame(cap: cv2.VideoCapture, idx: int) -> Optional[np.ndarray]:
    cap.set(cv2.CAP_PROP_POS_FRAMES, float(idx))
    ret, frame = cap.read()
    if not ret:
        return None
    if frame.ndim == 3:
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    return frame


def _compute_reference_frame(
    video_path: str | Path,
    mask: np.ndarray,
    corrupted_frames: list[int],
    n_frames: int = SR_N_REFERENCE,
) -> np.ndarray:
    """Temporal median of the first n_frames healthy frames inside the mask."""
    cap = _open_video(video_path)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    collected: list[np.ndarray] = []
    idx = 0
    while len(collected) < n_frames and idx < total:
        if idx not in corrupted_frames:
            frame = _read_gray_frame(cap, idx)
            if frame is not None:
                collected.append(frame.astype(np.float32))
        idx += 1
    cap.release()
    if not collected:
        raise ValueError("No healthy frames found to build reference.")
    stack = np.stack(collected, axis=0)
    return np.median(stack, axis=0).astype(np.float32)


# ── Step 1 — Sub-pixel shift estimation ──────────────────────────────────────


def estimate_subpixel_shifts(
    video_path: str | Path,
    reference_frame: np.ndarray,
    mask: np.ndarray,
    corrupted_frames: list[int] | None = None,
    upsample_factor: int = SR_UPSAMPLE_FACTOR,
    max_shift_px: float = SR_MAX_SHIFT_PX,
    max_error: float = SR_MAX_ERROR,
) -> dict:
    """
    Estimate sub-pixel shifts between each frame and the reference frame.

    Uses phase cross-correlation (scikit-image) with 1/upsample_factor px
    precision, restricted to the mask region to avoid background bias.

    Returns:
        dict with keys:
          - 'shifts'          : list of (shift_y, shift_x) float tuples
          - 'errors'          : list of float estimation errors
          - 'valid_frames'    : list of valid frame indices
          - 'rejected_frames' : list of (index, reason) tuples
    """
    video_path = Path(video_path)
    corrupted = set(corrupted_frames or [])

    cap = _open_video(video_path)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # Crop reference to mask bounding box for speed
    ys, xs = np.where(mask > 0)
    y0, y1 = int(ys.min()), int(ys.max()) + 1
    x0, x1 = int(xs.min()), int(xs.max()) + 1
    ref_crop = reference_frame[y0:y1, x0:x1].astype(np.float32)

    shifts: list[tuple[float, float]] = []
    errors: list[float] = []
    valid_frames: list[int] = []
    rejected_frames: list[tuple[int, str]] = []

    for i in tqdm(range(total), desc="Estimating shifts", unit="frame"):
        if i in corrupted:
            rejected_frames.append((i, "corrupted"))
            shifts.append((0.0, 0.0))
            errors.append(1.0)
            continue

        frame = _read_gray_frame(cap, i)
        if frame is None:
            rejected_frames.append((i, "read_error"))
            shifts.append((0.0, 0.0))
            errors.append(1.0)
            continue

        frame_crop = frame[y0:y1, x0:x1].astype(np.float32)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            shift, error, _ = phase_cross_correlation(
                ref_crop,
                frame_crop,
                upsample_factor=upsample_factor,
                normalization=None,
            )

        sy, sx = float(shift[0]), float(shift[1])
        total_shift = float(np.hypot(sy, sx))
        error = float(error)

        if total_shift > max_shift_px:
            rejected_frames.append((i, f"shift={total_shift:.2f}px > {max_shift_px}px"))
            shifts.append((sy, sx))
            errors.append(error)
            continue
        if error > max_error:
            rejected_frames.append((i, f"error={error:.3f} > {max_error}"))
            shifts.append((sy, sx))
            errors.append(error)
            continue

        shifts.append((sy, sx))
        errors.append(error)
        valid_frames.append(i)

    cap.release()
    return {
        "shifts":          shifts,
        "errors":          errors,
        "valid_frames":    valid_frames,
        "rejected_frames": rejected_frames,
    }


# ── Step 2 — Drizzle accumulation ────────────────────────────────────────────


def drizzle_combine(
    video_path: str | Path,
    shifts: list[tuple[float, float]],
    valid_frames: list[int],
    mask: np.ndarray,
    scale_factor: int = SR_SCALE_FACTOR,
    drop_size: float = SR_DROP_SIZE,
    output_path: str | Path | None = None,
) -> np.ndarray:
    """
    Combine valid frames into a high-resolution image via Drizzle.

    Each LR pixel is projected onto the HR grid using its sub-pixel shift.
    The kernel footprint (drop_size) controls sharpness vs. coverage:
      0.5 → sharpest, needs 50+ frames
      0.7 → good balance for 20–50 frames  (default)
      1.0 → equivalent to upscaled mean

    Fully vectorized: all pixels of one frame processed as matrix operations.
    """
    h_lr, w_lr = mask.shape
    h_hr = h_lr * scale_factor
    w_hr = w_lr * scale_factor

    output_grid = np.zeros((h_hr, w_hr), dtype=np.float64)
    weight_grid = np.zeros((h_hr, w_hr), dtype=np.float64)

    # Mask pixel coordinates (LR)
    mask_ys, mask_xs = np.where(mask > 0)  # shape (N_px,)
    n_mask_px = len(mask_ys)

    # Half-size of the drizzle kernel in HR pixels
    half = drop_size * scale_factor / 2.0

    cap = _open_video(video_path)

    for frame_idx in tqdm(valid_frames, desc="Drizzle accumulation", unit="frame"):
        frame = _read_gray_frame(cap, frame_idx)
        if frame is None:
            continue
        frame_f = frame.astype(np.float64)

        sy, sx = shifts[frame_idx]

        # HR coordinates of each masked LR pixel center after shift
        x_hr = (mask_xs + sx) * scale_factor   # shape (N_px,)
        y_hr = (mask_ys + sy) * scale_factor

        # Bounding box in HR for the kernel footprint
        x_min_arr = np.floor(x_hr - half).astype(np.int32)
        x_max_arr = np.ceil( x_hr + half).astype(np.int32)
        y_min_arr = np.floor(y_hr - half).astype(np.int32)
        y_max_arr = np.ceil( y_hr + half).astype(np.int32)

        # Clamp to HR grid bounds
        x_min_arr = np.clip(x_min_arr, 0, w_hr - 1)
        x_max_arr = np.clip(x_max_arr, 0, w_hr - 1)
        y_min_arr = np.clip(y_min_arr, 0, h_hr - 1)
        y_max_arr = np.clip(y_max_arr, 0, h_hr - 1)

        pixel_values = frame_f[mask_ys, mask_xs]

        # Iterate over the small kernel window (at most ceil(drop_size*scale)² iters)
        # Typical: 2×2 = 4 iterations for drop_size=0.7, scale=2
        kernel_span = int(np.ceil(drop_size * scale_factor)) + 1
        for dy in range(kernel_span):
            for dx in range(kernel_span):
                # HR pixel coordinates for this kernel offset
                px = x_min_arr + dx   # shape (N_px,)
                py = y_min_arr + dy

                # Mask out pixels outside the kernel window
                valid = (
                    (px >= 0) & (px < w_hr) &
                    (py >= 0) & (py < h_hr) &
                    (px <= x_max_arr) &
                    (py <= y_max_arr)
                )
                if not np.any(valid):
                    continue

                vx = px[valid]
                vy = py[valid]
                vval = pixel_values[valid]
                vcx = x_hr[valid]
                vcy = y_hr[valid]

                # Overlap = intersection area between kernel square and HR pixel square
                # Kernel: [xcenter-half, xcenter+half] × [ycenter-half, ycenter+half]
                # HR pixel: [vx, vx+1] × [vy, vy+1]
                ox = np.minimum(vcx + half, vx + 1.0) - np.maximum(vcx - half, vx.astype(np.float64))
                oy = np.minimum(vcy + half, vy + 1.0) - np.maximum(vcy - half, vy.astype(np.float64))
                overlap = np.clip(ox, 0, None) * np.clip(oy, 0, None)

                np.add.at(output_grid, (vy, vx), vval * overlap)
                np.add.at(weight_grid, (vy, vx), overlap)

    cap.release()

    # Normalize by accumulated weights
    covered = weight_grid > 0
    result = np.zeros((h_hr, w_hr), dtype=np.float32)
    result[covered] = (output_grid[covered] / weight_grid[covered]).astype(np.float32)

    # Inpaint uncovered HR pixels (weight == 0) within the upscaled mask
    mask_hr = cv2.resize(mask, (w_hr, h_hr), interpolation=cv2.INTER_NEAREST)
    uncovered_in_mask = (mask_hr > 0) & (~covered)
    if np.any(uncovered_in_mask):
        inpaint_mask = uncovered_in_mask.astype(np.uint8) * 255
        result_u8 = np.clip(result, 0, 255).astype(np.uint8)
        result_u8 = cv2.inpaint(result_u8, inpaint_mask, inpaintRadius=3, flags=cv2.INPAINT_TELEA)
        result = result_u8.astype(np.float32)

    # Apply upscaled mask
    result[mask_hr == 0] = 0

    result_u8 = np.clip(result, 0, 255).astype(np.uint8)

    if output_path is not None:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(output_path.with_suffix(".png")), result_u8)
        cv2.imwrite(str(output_path.with_suffix(".tiff")), result_u8)
        print(f"  Saved : {output_path.with_suffix('.png')}")
        print(f"  Saved : {output_path.with_suffix('.tiff')}")

    return result_u8


# ── Step 3 — Full pipeline ────────────────────────────────────────────────────


def temporal_sr_drizzle(
    video_path: str | Path,
    mask: np.ndarray,
    corrupted_frames: list[int],
    output_path: str | Path,
    scale_factor: int = SR_SCALE_FACTOR,
    drop_size: float = SR_DROP_SIZE,
    upsample_factor: int = SR_UPSAMPLE_FACTOR,
    n_frames_max: int | None = SR_N_FRAMES_MAX,
    min_frames: int = SR_MIN_FRAMES,
) -> dict:
    """
    Full temporal super-resolution pipeline.

    1. Build reference frame (median of first 60 healthy frames)
    2. Estimate sub-pixel shifts for all frames
    3. Filter and rank valid frames by estimation error
    4. Run Drizzle accumulation
    5. Save SR image as .png and .tiff
    """
    t_start = time.perf_counter()
    video_path = Path(video_path)
    output_path = Path(output_path)

    print("\n── Step 1/4 : Computing reference frame …")
    reference = _compute_reference_frame(video_path, mask, corrupted_frames)

    print("── Step 2/4 : Estimating sub-pixel shifts …")
    shifts_dict = estimate_subpixel_shifts(
        video_path, reference, mask,
        corrupted_frames=corrupted_frames,
        upsample_factor=upsample_factor,
    )

    valid_frames = shifts_dict["valid_frames"]
    n_rejected   = len(shifts_dict["rejected_frames"])

    if len(valid_frames) < min_frames:
        raise RuntimeError(
            f"Only {len(valid_frames)} valid frames (min={min_frames}). "
            "Check shift/error thresholds or video quality."
        )

    # Rank by estimation error (ascending) and optionally cap
    if n_frames_max is not None and n_frames_max < len(valid_frames):
        errors     = shifts_dict["errors"]
        valid_frames = sorted(valid_frames, key=lambda i: errors[i])[:n_frames_max]
        print(f"   Using top {n_frames_max} frames (ranked by error).")

    print(f"── Step 3/4 : Drizzle on {len(valid_frames)} frames …")
    sr_image = drizzle_combine(
        video_path,
        shifts=shifts_dict["shifts"],
        valid_frames=valid_frames,
        mask=mask,
        scale_factor=scale_factor,
        drop_size=drop_size,
        output_path=output_path,
    )

    t_end = time.perf_counter()

    # Shift statistics (valid frames only)
    used_shifts = np.array([shifts_dict["shifts"][i] for i in valid_frames])
    mean_sy, mean_sx = float(used_shifts[:, 0].mean()), float(used_shifts[:, 1].mean())
    std_sy,  std_sx  = float(used_shifts[:, 0].std()),  float(used_shifts[:, 1].std())

    results = {
        "n_frames_used":     len(valid_frames),
        "n_frames_rejected": n_rejected,
        "mean_shift_x":      mean_sx,
        "mean_shift_y":      mean_sy,
        "std_shift_x":       std_sx,
        "std_shift_y":       std_sy,
        "processing_time_s": t_end - t_start,
        "shifts_dict":       shifts_dict,
        "sr_image":          sr_image,
    }

    print(f"\n── Step 4/4 : Validation …")
    _validate_sr(sr_image, mask, video_path, results)

    return results


# ── Validation ────────────────────────────────────────────────────────────────


def _laplacian_sharpness(img: np.ndarray) -> float:
    return float(cv2.Laplacian(img.astype(np.float32), cv2.CV_32F).var())


def _local_contrast(img: np.ndarray, tile: int = 16) -> float:
    h, w = img.shape[:2]
    stds = []
    for y in range(0, h - tile, tile):
        for x in range(0, w - tile, tile):
            stds.append(float(img[y:y + tile, x:x + tile].std()))
    return float(np.mean(stds)) if stds else 0.0


def _validate_sr(
    sr_image: np.ndarray,
    mask: np.ndarray,
    video_path: str | Path,
    results: dict,
) -> None:
    """Print validation metrics and warn about potential hallucinations."""
    mask_hr = cv2.resize(mask, (sr_image.shape[1], sr_image.shape[0]),
                         interpolation=cv2.INTER_NEAREST)
    h_lr, w_lr = mask.shape
    h_hr, w_hr = sr_image.shape

    # Build bicubic upscale of reference for comparison
    cap = _open_video(video_path)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    ref_idx = min(30, total - 1)
    ref_frame = _read_gray_frame(cap, ref_idx)
    cap.release()

    bicubic = cv2.resize(ref_frame, (w_hr, h_hr), interpolation=cv2.INTER_CUBIC)

    sharp_bic   = _laplacian_sharpness(bicubic)
    sharp_lan   = _laplacian_sharpness(
        cv2.resize(ref_frame, (w_hr, h_hr), interpolation=cv2.INTER_LANCZOS4)
    )
    sharp_sr    = _laplacian_sharpness(sr_image)

    results["sharpness_bicubic"]  = sharp_bic
    results["sharpness_lanczos"]  = sharp_lan
    results["sharpness_drizzle"]  = sharp_sr

    if sharp_sr <= sharp_bic:
        print(
            "  WARNING: Drizzle sharpness not better than bicubic. "
            "Try drop_size=0.5 or increase n_frames_max."
        )

    # Hallucination check: large structures in SR that don't exist in mean proj
    cap = _open_video(video_path)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    acc = np.zeros((h_lr, w_lr), np.float64)
    n_acc = 0
    for i in range(0, total, max(1, total // 60)):
        f = _read_gray_frame(cap, i)
        if f is not None:
            acc += f.astype(np.float64)
            n_acc += 1
    cap.release()
    mean_proj = (acc / max(n_acc, 1)).astype(np.uint8)
    mean_hr   = cv2.resize(mean_proj, (w_hr, h_hr), interpolation=cv2.INTER_CUBIC)
    diff      = cv2.absdiff(sr_image, mean_hr).astype(np.float32)
    diff[mask_hr == 0] = 0
    hallucination_score = float(diff.max())
    results["hallucination_score"] = hallucination_score
    if hallucination_score > 80:
        print(
            f"  WARNING: max pixel diff SR vs mean projection = {hallucination_score:.0f} "
            "> 80. Some structures may be hallucinated."
        )

    # ── Terminal report ───────────────────────────────────────────────────────
    pct_lan = (sharp_sr / sharp_lan - 1) * 100 if sharp_lan > 0 else 0.0
    print("\n" + "─" * 53)
    print("  RÉSULTATS SUPER-RÉSOLUTION DRIZZLE")
    print(f"  Frames utilisées     : {results['n_frames_used']}")
    print(f"  Frames rejetées      : {results['n_frames_rejected']}")
    print(f"  Décalage moyen       : dx={results['mean_shift_x']:+.2f}px  dy={results['mean_shift_y']:+.2f}px")
    print(f"  Std décalages        : σx={results['std_shift_x']:.2f}px  σy={results['std_shift_y']:.2f}px")
    print(f"  Netteté (Laplacien)  :")
    print(f"    Bicubique          : {sharp_bic:.1f}")
    print(f"    Lanczos            : {sharp_lan:.1f}")
    print(f"    Drizzle            : {sharp_sr:.1f}  ({pct_lan:+.0f}% vs Lanczos)")
    print(f"  Temps total          : {results['processing_time_s']:.1f}s")
    print("─" * 53 + "\n")


# ── Utility : shift visualization ─────────────────────────────────────────────


def visualize_shifts(
    shifts_dict: dict,
    output_path: str | Path,
) -> None:
    """
    3-panel diagnostic plot:
      1. Scatter (shift_x, shift_y) — valid=blue, rejected=red
      2. Histogram of shift amplitudes
      3. Temporal evolution of shift_x and shift_y
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    all_shifts = np.array(shifts_dict["shifts"])   # (N, 2) — (sy, sx)
    all_errors = np.array(shifts_dict["errors"])
    valid_set  = set(shifts_dict["valid_frames"])
    n_total    = len(all_shifts)

    valid_mask = np.array([i in valid_set for i in range(n_total)])
    sx_all = all_shifts[:, 1]
    sy_all = all_shifts[:, 0]
    amplitudes = np.hypot(sx_all, sy_all)

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle("Sub-pixel shift analysis", fontsize=13, fontweight="bold")

    # Panel 1 — scatter
    ax = axes[0]
    ax.scatter(sx_all[~valid_mask], sy_all[~valid_mask],
               c="red", s=10, alpha=0.5, label="Rejected")
    ax.scatter(sx_all[valid_mask],  sy_all[valid_mask],
               c="steelblue", s=10, alpha=0.6, label="Valid")
    ax.axhline(0, color="gray", lw=0.7, ls="--")
    ax.axvline(0, color="gray", lw=0.7, ls="--")
    ax.set_xlabel("shift_x (px)")
    ax.set_ylabel("shift_y (px)")
    ax.set_title("Shift scatter")
    ax.legend(fontsize=9)
    ax.set_aspect("equal")

    # Panel 2 — amplitude histogram
    ax = axes[1]
    ax.hist(amplitudes[valid_mask], bins=40, color="steelblue",
            alpha=0.8, edgecolor="white", label="Valid")
    ax.hist(amplitudes[~valid_mask], bins=40, color="red",
            alpha=0.5, edgecolor="white", label="Rejected")
    ax.set_xlabel("Shift amplitude (px)")
    ax.set_ylabel("Frame count")
    ax.set_title("Amplitude distribution")
    ax.legend(fontsize=9)

    # Panel 3 — temporal evolution
    ax = axes[2]
    frames_idx = np.arange(n_total)
    ax.plot(frames_idx, sx_all, lw=0.8, color="steelblue", label="shift_x")
    ax.plot(frames_idx, sy_all, lw=0.8, color="tomato",    label="shift_y")
    # Mark rejected
    rej_idx = [i for i, v in enumerate(valid_mask) if not v]
    if rej_idx:
        ax.scatter(rej_idx, sx_all[rej_idx], c="steelblue", s=15, marker="x")
        ax.scatter(rej_idx, sy_all[rej_idx], c="tomato",    s=15, marker="x")
    ax.axhline(0, color="gray", lw=0.7, ls="--")
    ax.set_xlabel("Frame index")
    ax.set_ylabel("Shift (px)")
    ax.set_title("Temporal shift evolution")
    ax.legend(fontsize=9)

    plt.tight_layout()
    plt.savefig(str(output_path), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved : {output_path}")


# ── Utility : method comparison ───────────────────────────────────────────────


def compare_sr_methods(
    video_path: str | Path,
    mask: np.ndarray,
    sr_result: np.ndarray,
    shifts_dict: dict,
    output_path: str | Path,
    crop_size: int = 200,
) -> None:
    """
    4-column comparison grid:
      Nearest  |  Bicubic  |  Lanczos  |  Drizzle
    Shown at full image scale AND a 200×200 crop on a vessel-rich zone.
    Sharpness (Laplacian variance) and local contrast printed per method.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    cap = _open_video(video_path)
    ref_frame = _read_gray_frame(cap, 0)
    cap.release()

    h_hr = sr_result.shape[0]
    w_hr = sr_result.shape[1]

    nearest  = cv2.resize(ref_frame, (w_hr, h_hr), interpolation=cv2.INTER_NEAREST)
    bicubic  = cv2.resize(ref_frame, (w_hr, h_hr), interpolation=cv2.INTER_CUBIC)
    lanczos  = cv2.resize(ref_frame, (w_hr, h_hr), interpolation=cv2.INTER_LANCZOS4)
    drizzle  = sr_result

    methods = [
        ("Nearest ×2",  nearest),
        ("Bicubic ×2",  bicubic),
        ("Lanczos ×2",  lanczos),
        ("Drizzle SR",  drizzle),
    ]

    # Find crop center on the vessel-rich zone inside the mask
    mask_hr = cv2.resize(mask, (w_hr, h_hr), interpolation=cv2.INTER_NEAREST)
    ys, xs  = np.where(mask_hr > 0)
    cy_c = int(ys.mean())
    cx_c = int(xs.mean())
    half_c = crop_size // 2
    cy0 = max(0, cy_c - half_c)
    cx0 = max(0, cx_c - half_c)
    cy1 = min(h_hr, cy0 + crop_size)
    cx1 = min(w_hr, cx0 + crop_size)

    fig, axes = plt.subplots(2, 4, figsize=(22, 11))
    fig.suptitle("Super-resolution method comparison", fontsize=14, fontweight="bold")

    for col, (name, img) in enumerate(methods):
        sharp   = _laplacian_sharpness(img)
        contrast = _local_contrast(img)
        label   = f"{name}\nSharp={sharp:.0f}  Contr={contrast:.1f}"

        # Row 0 — full image
        axes[0, col].imshow(img, cmap="gray", vmin=0, vmax=255)
        axes[0, col].set_title(label, fontsize=9)
        axes[0, col].axis("off")
        # Crop rectangle indicator
        rect = mpatches.Rectangle(
            (cx0, cy0), cx1 - cx0, cy1 - cy0,
            linewidth=1.5, edgecolor="red", facecolor="none"
        )
        axes[0, col].add_patch(rect)

        # Row 1 — crop
        crop = img[cy0:cy1, cx0:cx1]
        axes[1, col].imshow(crop, cmap="gray", vmin=0, vmax=255)
        axes[1, col].set_title(f"Crop ({crop_size}×{crop_size})", fontsize=9)
        axes[1, col].axis("off")

    axes[0, 0].set_ylabel("Full image", fontsize=10)
    axes[1, 0].set_ylabel(f"Crop {crop_size}px", fontsize=10)

    plt.tight_layout()
    plt.savefig(str(output_path), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved : {output_path}")


# ── Utility : sub-pixel diversity analysis ────────────────────────────────────


def analyze_subpixel_diversity(
    shifts_dict: dict,
    output_path: str | Path,
    grid_size: int = 10,
) -> dict:
    """
    Analyze whether sub-pixel shifts are sufficiently diverse for SR.

    Divides the fractional shift space [0,1)×[0,1) into a grid_size×grid_size
    grid and counts how many cells are covered by at least one valid frame.
    Coverage < 50% triggers a WARNING.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    valid_set = set(shifts_dict["valid_frames"])
    all_shifts = shifts_dict["shifts"]

    if not valid_set:
        print("  WARNING: no valid frames — cannot analyze diversity.")
        return {"coverage": 0.0}

    valid_shifts = np.array([all_shifts[i] for i in sorted(valid_set)])
    # Fractional parts only
    frac_y = valid_shifts[:, 0] % 1.0
    frac_x = valid_shifts[:, 1] % 1.0

    coverage_map = np.zeros((grid_size, grid_size), dtype=np.int32)
    for fy, fx in zip(frac_y, frac_x):
        gy = min(int(fy * grid_size), grid_size - 1)
        gx = min(int(fx * grid_size), grid_size - 1)
        coverage_map[gy, gx] += 1

    n_covered  = int(np.sum(coverage_map > 0))
    n_total    = grid_size * grid_size
    coverage   = n_covered / n_total

    if coverage < 0.5:
        print(
            f"  WARNING: sub-pixel coverage = {coverage:.0%} < 50%. "
            "Shifts may not be diverse enough for optimal SR."
        )
    else:
        print(f"  Sub-pixel coverage : {coverage:.0%} ({n_covered}/{n_total} cells).")

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle(
        f"Sub-pixel diversity — coverage {coverage:.0%} ({n_covered}/{n_total} cells)",
        fontsize=12, fontweight="bold"
    )

    # Panel 1 — fractional shift scatter
    ax = axes[0]
    ax.scatter(frac_x, frac_y, s=8, alpha=0.5, c="steelblue")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xlabel("frac(shift_x)")
    ax.set_ylabel("frac(shift_y)")
    ax.set_title("Fractional shift distribution")
    ax.set_aspect("equal")

    # Draw grid
    for g in np.linspace(0, 1, grid_size + 1):
        ax.axhline(g, color="gray", lw=0.4, ls="--")
        ax.axvline(g, color="gray", lw=0.4, ls="--")

    # Panel 2 — coverage heatmap
    ax = axes[1]
    im = ax.imshow(coverage_map, origin="lower", cmap="YlOrRd",
                   extent=[0, 1, 0, 1], aspect="equal")
    ax.set_xlabel("frac(shift_x)")
    ax.set_ylabel("frac(shift_y)")
    ax.set_title("Coverage map (frames per cell)")
    fig.colorbar(im, ax=ax, fraction=0.046)

    plt.tight_layout()
    plt.savefig(str(output_path), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved : {output_path}")

    return {
        "coverage":     coverage,
        "n_covered":    n_covered,
        "coverage_map": coverage_map,
    }


# ── Lanczos standalone ────────────────────────────────────────────────────────


def lanczos_sr(
    video_path: str | Path,
    mask: np.ndarray,
    corrupted_frames: list[int],
    output_path: str | Path,
    scale_factor: int = SR_SCALE_FACTOR,
    n_reference: int = SR_N_REFERENCE,
) -> np.ndarray:
    """
    Single-image Lanczos upscaling using the temporal median reference frame.

    Much faster than Drizzle (no shift estimation, no accumulation) but uses
    only one frame worth of information — no real super-resolution.

    Returns the upscaled image as uint8 (H×W).
    Saves <output_path>.png and <output_path>.tiff.
    """
    video_path  = Path(video_path)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    print("\n── Lanczos SR : computing temporal median reference …")
    ref = _compute_reference_frame(video_path, mask, corrupted_frames, n_frames=n_reference)

    h_lr, w_lr = ref.shape
    h_hr = h_lr * scale_factor
    w_hr = w_lr * scale_factor

    print(f"── Lanczos SR : upscaling {w_lr}×{h_lr} → {w_hr}×{h_hr} …")
    result = cv2.resize(ref.astype(np.uint8), (w_hr, h_hr), interpolation=cv2.INTER_LANCZOS4)

    # Apply upscaled mask
    mask_hr = cv2.resize(mask, (w_hr, h_hr), interpolation=cv2.INTER_NEAREST)
    result[mask_hr == 0] = 0

    cv2.imwrite(str(output_path.with_suffix(".png")),  result)
    cv2.imwrite(str(output_path.with_suffix(".tiff")), result)
    print(f"  Saved : {output_path.with_suffix('.png')}")
    print(f"  Saved : {output_path.with_suffix('.tiff')}")

    return result


# ── Shared helpers for CLI ────────────────────────────────────────────────────


def _load_mask_for_video(video_path: Path, explicit_mask: Path | None, output_dir: Path) -> np.ndarray:
    """Find and load a mask for the given video, in priority order."""
    candidates: list[Path] = []
    if explicit_mask is not None:
        candidates.append(explicit_mask)
    candidates += [
        output_dir / "mask.png",
        video_path.parent / "mask.png",
        video_path.parent.parent / "mask.png",
        Path(__file__).parent.parent / "PretraitementIntegrale" / "mask.png",
    ]
    for mp in candidates:
        if mp.exists():
            m = load_mask(mp)
            print(f"  Mask : {mp}")
            return m
    raise FileNotFoundError(f"mask.png introuvable. Candidats testés :\n  " + "\n  ".join(str(c) for c in candidates))


def _load_corrupted_frames_for_video(video_path: Path, output_dir: Path) -> list[int]:
    for cf_path in [output_dir / "step2_corrected.json", video_path.parent / "step2_corrected.json"]:
        if cf_path.exists():
            with open(cf_path) as f:
                data = json.load(f)
            frames = data.get("corrupted_frames", [])
            print(f"  Corrupted frames log : {cf_path} ({len(frames)} frames)")
            return frames
    return []


def _process_one_video(
    video_path: Path,
    output_dir: Path,
    method: str,
    explicit_mask: Path | None,
    cfg: dict,
    drop_size: float,
) -> None:
    """Run Drizzle, Lanczos, or both on a single video and save results."""
    output_dir.mkdir(parents=True, exist_ok=True)
    stem = video_path.stem

    mask             = _load_mask_for_video(video_path, explicit_mask, output_dir)
    corrupted_frames = _load_corrupted_frames_for_video(video_path, output_dir)

    drizzle_result: np.ndarray | None = None
    shifts_dict:    dict | None       = None

    if method in ("drizzle", "both"):
        results = temporal_sr_drizzle(
            video_path      = video_path,
            mask            = mask,
            corrupted_frames= corrupted_frames,
            output_path     = output_dir / f"{stem}_drizzle_sr",
            scale_factor    = cfg["scale_factor"],
            drop_size       = drop_size,
            upsample_factor = cfg["upsample_factor"],
            n_frames_max    = cfg["n_frames_max"],
            min_frames      = cfg["min_frames"],
        )
        drizzle_result = results["sr_image"]
        shifts_dict    = results["shifts_dict"]

        print("── Generating Drizzle visualizations …")
        visualize_shifts(shifts_dict, output_dir / f"{stem}_shifts_analysis.png")
        analyze_subpixel_diversity(shifts_dict, output_dir / f"{stem}_subpixel_diversity.png")

    if method in ("lanczos", "both"):
        lanczos_sr(
            video_path      = video_path,
            mask            = mask,
            corrupted_frames= corrupted_frames,
            output_path     = output_dir / f"{stem}_lanczos_sr",
            scale_factor    = cfg["scale_factor"],
        )

    # Comparison grid when both are available (or at least Drizzle for baseline)
    if drizzle_result is not None and shifts_dict is not None:
        compare_sr_methods(
            video_path, mask, drizzle_result, shifts_dict,
            output_dir / f"{stem}_method_comparison.png",
        )


# ── CLI entry point ───────────────────────────────────────────────────────────


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(
        description="Super-résolution temporelle — Drizzle et/ou Lanczos.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Résultats produits par vidéo :\n"
            "  Drizzle  → <stem>_drizzle_sr.png / .tiff\n"
            "             shifts_analysis.png       — scatter/histogramme des décalages\n"
            "             subpixel_diversity.png    — couverture sous-pixel\n"
            "             method_comparison.png     — Nearest / Bicubic / Lanczos / Drizzle\n"
            "  Lanczos  → <stem>_lanczos_sr.png / .tiff\n"
            "             (médiane temporelle des 60 premières frames, upscale Lanczos)\n\n"
            "Exemples :\n"
            "  python DrizzleSR.py video.avi\n"
            "  python DrizzleSR.py video.avi --output resultats/ --method drizzle\n"
            "  python DrizzleSR.py video.avi --method lanczos\n"
            "  python DrizzleSR.py video.avi --method both\n"
            "  python DrizzleSR.py dossier_videos/ --output sr_out/ --mask mon_masque.png\n"
            "  python DrizzleSR.py dossier_videos/ --method both --drop-size 0.5\n"
        ),
    )
    parser.add_argument("input",
        help="Vidéo .avi ou dossier contenant des vidéos .avi.")
    parser.add_argument("--output", default=None,
        help="Dossier de sortie. Défaut : <stem>_drizzle_sr/ (vidéo) ou <input>_sr/ (dossier).")
    parser.add_argument("--method", choices=["drizzle", "lanczos", "both"], default="drizzle",
        help="Méthode à appliquer (défaut : drizzle).")
    parser.add_argument("--mask", default=None,
        help="Chemin vers un masque PNG prédéfini (partagé par toutes les vidéos).")
    parser.add_argument("--drop-size", type=float, default=None,
        help=f"Taille du noyau Drizzle en pixels HR (défaut config.yaml : {SR_DROP_SIZE}). "
             "0.5=net, 0.7=équilibré, 1.0=moyenne.")
    args = parser.parse_args()

    inp          = Path(args.input)
    explicit_mask: Path | None = Path(args.mask) if args.mask else None

    if explicit_mask is not None and not explicit_mask.exists():
        sys.exit(f"ERREUR : masque introuvable : {explicit_mask}")
    if not inp.exists():
        sys.exit(f"ERREUR : chemin introuvable : {inp}")

    cfg       = _load_sr_config(Path(__file__).parent / "config.yaml")
    drop_size = args.drop_size if args.drop_size is not None else cfg["drop_size"]

    # Collect videos
    if inp.is_file():
        videos = [inp]
        default_out = inp.parent / f"{inp.stem}_sr"
    else:
        videos = sorted(inp.glob("*.avi"))
        if not videos:
            sys.exit(f"ERREUR : aucune vidéo .avi trouvée dans {inp}")
        default_out = inp.parent / f"{inp.name}_sr"

    out_root = Path(args.output) if args.output else default_out

    print(f"\n  Méthode    : {args.method}")
    print(f"  Vidéos     : {len(videos)}")
    print(f"  Sortie     : {out_root}")
    print(f"  Drop-size  : {drop_size}  (Drizzle uniquement)")

    failed: list[str] = []
    for video_path in videos:
        stem = video_path.stem
        print(f"\n{'═'*60}")
        print(f"  {video_path.name}")
        print(f"{'═'*60}")
        try:
            _process_one_video(video_path, out_root, args.method, explicit_mask, cfg, drop_size)
        except Exception as exc:
            print(f"  [ERREUR] {stem} : {exc}")
            failed.append(stem)

    print(f"\n{'═'*60}")
    print(f"  Terminé : {len(videos) - len(failed)}/{len(videos)} vidéo(s) réussie(s)")
    if failed:
        print(f"  Échecs  : {', '.join(failed)}")
    print(f"  Résultats → {out_root}")


if __name__ == "__main__":
    main()
