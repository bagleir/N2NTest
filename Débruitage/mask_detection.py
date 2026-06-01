#!/usr/bin/env python3
"""
Step [1] — Circular optical-disk mask detection.

Detects the circular field-of-view boundary once per video and returns
a reusable binary mask for all downstream pipeline stages.
"""
from __future__ import annotations

import cv2
import numpy as np
from pathlib import Path


# ── Helpers ──────────────────────────────────────────────────────────────────


def _odd(n: int) -> int:
    """Return n if odd, n+1 if even (kernel sizes must be odd)."""
    return n if n % 2 == 1 else n + 1


# ── Core detection ───────────────────────────────────────────────────────────


def detect_circular_mask(
    video_path: str | Path,
    n_sample_frames: int = 20,
    margin_px: int = 5,
) -> dict:
    """
    Detect the circular optical field-of-view mask from a video.

    Strategy:
      1. Sample n evenly-spaced frames across the whole video.
      2. Compute their temporal mean → stable image with reduced noise.
      3. Otsu threshold → binary image highlighting the lit disk.
      4. Find the largest closed contour.
      5. Fit the minimum enclosing circle on that contour.
      6. Erode by margin_px pixels to avoid edge artefacts.

    Args:
        video_path:      Path to the .avi video file.
        n_sample_frames: Number of evenly-spaced frames to average (default 20).
        margin_px:       Erosion margin in pixels to pull the mask edge inward (default 5).

    Returns:
        dict with keys:
          - 'mask'     : np.ndarray (H, W) uint8 — 255 inside useful zone, 0 outside.
          - 'center'   : tuple[int, int] (cx, cy) in pixels.
          - 'radius'   : float, effective radius in pixels (after erosion).
          - 'coverage' : float, fraction [0–1] of image pixels inside the mask.

    Raises:
        FileNotFoundError: if video_path cannot be opened.
        ValueError:        if the detected circle is < 30 % or > 95 % of the image area.
    """
    video_path = Path(video_path)

    # ── 1. Sample frames ──────────────────────────────────────────────────────
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise FileNotFoundError(f"Cannot open video: {video_path}")

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))

    n_sample_frames = min(n_sample_frames, total_frames)
    indices = np.linspace(0, total_frames - 1, n_sample_frames, dtype=int)

    accumulator = np.zeros((h, w), dtype=np.float64)
    count = 0

    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, float(idx))
        ret, frame = cap.read()
        if not ret:
            continue
        if frame.ndim == 3:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        accumulator += frame.astype(np.float64)
        count += 1

    cap.release()

    if count == 0:
        raise ValueError("Could not read any frame from the video.")

    mean_frame = (accumulator / count).astype(np.uint8)

    # ── 2. Blur → Otsu → close ────────────────────────────────────────────────
    # A large Gaussian blur erases vessel detail and exposes only the
    # circular vignette shape. Without this, Otsu latches onto bright vessels
    # instead of the disk boundary.
    blur_k = _odd(min(h, w) // 8)           # ~63 px for 512×512
    blurred = cv2.GaussianBlur(mean_frame, (blur_k, blur_k), 0)

    _, binary = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Morphological closing fills small holes left by dark vessel shadows.
    close_k = _odd(min(h, w) // 16)         # ~31 px for 512×512
    kernel_close = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (close_k, close_k)
    )
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel_close, iterations=2)

    # ── 3. Largest contour ────────────────────────────────────────────────────
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        raise ValueError("No contour found after Otsu thresholding.")

    largest = max(contours, key=cv2.contourArea)

    # ── 4. Minimum enclosing circle ───────────────────────────────────────────
    (cx_f, cy_f), radius_f = cv2.minEnclosingCircle(largest)
    cx, cy = int(round(cx_f)), int(round(cy_f))
    radius = float(radius_f)

    # ── 5. Validate coverage ──────────────────────────────────────────────────
    total_pixels = h * w
    raw_coverage = (np.pi * radius ** 2) / total_pixels

    if raw_coverage < 0.30:
        raise ValueError(
            f"Detected circle too small: coverage={raw_coverage:.1%} < 30 %. "
            f"center=({cx}, {cy}), radius={radius:.1f} px. "
            "Check that the video contains a clear circular optical disk."
        )
    if raw_coverage > 0.95:
        raise ValueError(
            f"Detected circle too large: coverage={raw_coverage:.1%} > 95 %. "
            f"center=({cx}, {cy}), radius={radius:.1f} px. "
            "Otsu may have included background noise — try increasing n_sample_frames."
        )

    # ── 6. Build mask + erosion margin ────────────────────────────────────────
    mask = np.zeros((h, w), dtype=np.uint8)
    cv2.circle(mask, (cx, cy), int(radius), 255, thickness=-1)

    if margin_px > 0:
        kernel_size = 2 * margin_px + 1
        kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (kernel_size, kernel_size)
        )
        mask = cv2.erode(mask, kernel, iterations=1)
        radius = max(0.0, radius - margin_px)

    coverage = float(np.count_nonzero(mask)) / total_pixels

    return {
        "mask": mask,
        "center": (cx, cy),
        "radius": radius,
        "coverage": coverage,
    }


# ── I/O ──────────────────────────────────────────────────────────────────────


def save_mask(mask: np.ndarray, output_path: str | Path) -> None:
    """Save a binary mask as a PNG file (lossless)."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output_path), mask)


def load_mask(mask_path: str | Path) -> np.ndarray:
    """Load a previously saved mask PNG as a (H, W) uint8 array."""
    mask_path = Path(mask_path)
    mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
    if mask is None:
        raise FileNotFoundError(f"Cannot load mask: {mask_path}")
    return mask


# ── Visualisation ────────────────────────────────────────────────────────────


def visualize_mask(
    video_path: str | Path,
    mask: np.ndarray,
    output_path: str | Path,
    n_preview_frames: int = 3,
) -> None:
    """
    Save a validation PNG: n_preview_frames panels showing sampled video frames
    with the detected circle boundary overlaid in red.

    Args:
        video_path:       Source video.
        mask:             Binary mask from detect_circular_mask().
        output_path:      Destination PNG path.
        n_preview_frames: Number of evenly-spaced frames to display (default 3).
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    video_path = Path(video_path)

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise FileNotFoundError(f"Cannot open video: {video_path}")

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    sample_indices = np.linspace(0, total_frames - 1, n_preview_frames, dtype=int)

    frames: list[tuple[int, np.ndarray]] = []
    for idx in sample_indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, float(idx))
        ret, frame = cap.read()
        if not ret:
            continue
        if frame.ndim == 3:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        frames.append((int(idx), frame))

    cap.release()

    # Recover circle geometry from the (possibly eroded) mask
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        (cx_f, cy_f), r_f = cv2.minEnclosingCircle(max(contours, key=cv2.contourArea))
        cx, cy, r = int(round(cx_f)), int(round(cy_f)), int(round(r_f))
    else:
        cx = cy = r = 0

    fig, axes = plt.subplots(1, len(frames), figsize=(6 * len(frames), 6))
    if len(frames) == 1:
        axes = [axes]

    for ax, (frame_idx, frame) in zip(axes, frames):
        rgb = cv2.cvtColor(frame, cv2.COLOR_GRAY2RGB)
        if r > 0:
            cv2.circle(rgb, (cx, cy), r, (255, 0, 0), thickness=2)
            cv2.drawMarker(rgb, (cx, cy), (255, 0, 0), cv2.MARKER_CROSS, markerSize=14, thickness=2)
        ax.imshow(rgb)
        ax.set_title(f"Frame {frame_idx}", fontsize=11)
        ax.axis("off")

    h_img, w_img = mask.shape
    coverage = float(np.count_nonzero(mask)) / (h_img * w_img)
    fig.suptitle(
        f"Mask validation — center=({cx}, {cy})  radius={r} px  coverage={coverage:.1%}",
        fontsize=12,
        fontweight="bold",
    )
    plt.tight_layout()
    plt.savefig(str(output_path), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved : {output_path}")


# ── CLI entry point ───────────────────────────────────────────────────────────


def main() -> None:
    import sys

    if len(sys.argv) < 2:
        print("Usage: python mask_detection.py <video.avi> [output_dir]")
        sys.exit(1)

    video_path = Path(sys.argv[1])
    output_dir = Path(sys.argv[2]) if len(sys.argv) > 2 else video_path.parent

    print(f"Video    : {video_path.name}")
    print("Detecting circular mask …")

    result = detect_circular_mask(video_path)

    cx, cy = result["center"]
    r      = result["radius"]
    cov    = result["coverage"]

    print(f"  center   : ({cx}, {cy}) px")
    print(f"  radius   : {r:.1f} px")
    print(f"  coverage : {cov:.1%}")

    mask_path = output_dir / f"mask_{video_path.stem}.png"
    save_mask(result["mask"], mask_path)
    print(f"  Mask saved : {mask_path}")

    viz_path = output_dir / f"mask_validation_{video_path.stem}.png"
    visualize_mask(video_path, result["mask"], viz_path)
    print("\nDone.")


if __name__ == "__main__":
    main()
