#!/usr/bin/env python3
"""
Step [2] — Bad frame detection and replacement.

Detects corrupted frames (flashes and blinks/occlusions) in ocular vascular
imaging videos and replaces them by linear interpolation of neighboring clean
frames, within the circular mask region only.
"""
from __future__ import annotations

import json
import warnings
from pathlib import Path

import cv2
import numpy as np
from scipy import stats

from mask_detection import detect_circular_mask, load_mask


# ── Internal helpers ──────────────────────────────────────────────────────────


def _read_video_frames(video_path: Path) -> tuple[list[np.ndarray], dict]:
    """Read all frames from a video and return (grayscale frames, metadata)."""
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise FileNotFoundError(f"Cannot open video: {video_path}")

    meta = {
        "fps":          cap.get(cv2.CAP_PROP_FPS),
        "width":        int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
        "height":       int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
        "fourcc":       int(cap.get(cv2.CAP_PROP_FOURCC)),
        "total_frames": int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
    }

    frames: list[np.ndarray] = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if frame.ndim == 3:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        frames.append(frame)

    cap.release()
    return frames, meta


def _masked_mean(frame: np.ndarray, mask: np.ndarray) -> float:
    """Return the mean pixel value inside the mask region."""
    pixels = frame[mask > 0].astype(np.float64)
    return float(pixels.mean()) if len(pixels) > 0 else 0.0


def _masked_percentile(frame: np.ndarray, mask: np.ndarray, percentile: float) -> float:
    """
    Return a high percentile of within-mask pixel values.

    Using the 99th percentile instead of the maximum makes the statistic robust
    to isolated hot pixels while remaining sensitive to partial flashes that
    affect only a sub-region of the optical disk.
    """
    pixels = frame[mask > 0].astype(np.float64)
    return float(np.percentile(pixels, percentile)) if len(pixels) > 0 else 0.0


def _masked_correlation(frame_a: np.ndarray, frame_b: np.ndarray, mask: np.ndarray) -> float:
    """Return the Pearson correlation between two frames inside the mask."""
    px_a = frame_a[mask > 0].astype(np.float64)
    px_b = frame_b[mask > 0].astype(np.float64)
    if len(px_a) < 2:
        return 1.0
    r, _ = stats.pearsonr(px_a, px_b)
    return float(r) if not np.isnan(r) else 1.0


# ── Core detection ────────────────────────────────────────────────────────────


def detect_corrupted_frames(
    video_path: str | Path,
    mask: np.ndarray,
    std_threshold: float = 3.0,
    correlation_threshold: float = 0.5,
    local_percentile: float = 99.0,
) -> dict:
    """
    Detect corrupted frames in a video using luminosity and correlation criteria.

    Flash detection covers both full-frame flashes (mean luminosity spike) and
    partial flashes (high-percentile spike), because bright artefacts may affect
    only a sub-region of the optical disk without raising the global mean much.

    Args:
        video_path:            Path to the source .avi video.
        mask:                  Binary mask (H, W) uint8 from step [1].
        std_threshold:         Standard-deviation multiplier above/below the mean
                               to flag a frame as flash/blink (default 3.0).
        correlation_threshold: Minimum Pearson correlation with the previous frame;
                               below this, the frame is flagged as occlusion (default 0.5).
        local_percentile:      Percentile of within-mask pixels used for partial
                               flash detection (default 99.0 — top 1 % of pixels).

    Returns:
        dict with keys:
          - 'flash_frames'     : list[int] — too-bright frame indices.
          - 'blink_frames'     : list[int] — too-dark frame indices.
          - 'occlusion_frames' : list[int] — low-correlation frame indices.
          - 'all_corrupted'    : list[int] — deduplicated sorted union of all above.
          - 'luminosity_curve' : np.ndarray — mean luminosity per frame.
          - 'local_max_curve'  : np.ndarray — local percentile per frame.
          - 'correlation_curve': np.ndarray — correlation with previous frame (NaN at index 0).
          - 'local_percentile' : float — value used for local_percentile (for traceability).

    Raises:
        FileNotFoundError: if video_path cannot be opened.
        UserWarning:       if more than 20 % of frames are corrupted.
    """
    video_path = Path(video_path)
    frames, _ = _read_video_frames(video_path)
    n = len(frames)

    # ── Per-frame statistics ──────────────────────────────────────────────────
    lum       = np.array([_masked_mean(f, mask) for f in frames])
    local_max = np.array([_masked_percentile(f, mask, local_percentile) for f in frames])

    corr = np.full(n, np.nan)
    for i in range(1, n):
        corr[i] = _masked_correlation(frames[i], frames[i - 1], mask)

    # ── Detection thresholds ──────────────────────────────────────────────────
    lum_mean, lum_std = lum.mean(), lum.std()
    loc_mean, loc_std = local_max.mean(), local_max.std()

    flash_threshold_global = lum_mean + std_threshold * lum_std
    flash_threshold_local  = loc_mean + std_threshold * loc_std
    blink_threshold        = lum_mean - std_threshold * lum_std

    # Full-frame flashes OR partial flashes (high-percentile spike)
    flash_frames: list[int] = sorted(
        set(np.where(lum > flash_threshold_global)[0].tolist()) |
        set(np.where(local_max > flash_threshold_local)[0].tolist())
    )
    blink_frames:    list[int] = np.where(lum < blink_threshold)[0].tolist()
    occlusion_frames: list[int] = np.where(corr < correlation_threshold)[0].tolist()

    all_corrupted = sorted(set(flash_frames) | set(blink_frames) | set(occlusion_frames))

    # ── >20 % guard ───────────────────────────────────────────────────────────
    ratio = len(all_corrupted) / n
    if ratio > 0.20:
        warnings.warn(
            f"{len(all_corrupted)}/{n} frames ({ratio:.1%}) detected as corrupted. "
            "This exceeds 20 %. Review the thresholds before continuing.",
            UserWarning,
            stacklevel=2,
        )

    return {
        "flash_frames":      flash_frames,
        "blink_frames":      blink_frames,
        "occlusion_frames":  occlusion_frames,
        "all_corrupted":     all_corrupted,
        "luminosity_curve":  lum,
        "local_max_curve":   local_max,
        "correlation_curve": corr,
        "local_percentile":  local_percentile,
    }


# ── Replacement helpers ───────────────────────────────────────────────────────


def _make_video_writer(
    path: Path,
    fps: float,
    width: int,
    height: int,
) -> cv2.VideoWriter:
    """
    Create a VideoWriter that is guaranteed to be open, trying codecs in order.

    The input video codec is intentionally NOT reused: many medical-imaging codecs
    support decoding but not encoding in OpenCV, causing a silent write failure
    (the writer object is created but isOpened() is False and no frames are stored).
    """
    for fourcc_str in ("XVID", "MJPG", "mp4v"):
        fourcc = cv2.VideoWriter_fourcc(*fourcc_str)
        writer = cv2.VideoWriter(
            str(path), fourcc, fps, (width, height), isColor=True
        )
        if writer.isOpened():
            return writer
        writer.release()
    raise RuntimeError(
        f"Could not open VideoWriter for {path}. "
        "Tried XVID, MJPG, mp4v — none worked. "
        "Check that OpenCV was built with video-write support."
    )


def _collect_clean_frames(
    frames: list[np.ndarray],
    corrupted_set: set[int],
    start: int,
    n: int,
    direction: int,
) -> list[np.ndarray]:
    """
    Walk from `start` in `direction` (+1 or -1) and collect up to `n` clean frames.

    Skips any index in corrupted_set.  Returns float64 arrays.
    """
    result: list[np.ndarray] = []
    i = start
    while 0 <= i < len(frames) and len(result) < n:
        if i not in corrupted_set:
            result.append(frames[i].astype(np.float64))
        i += direction
    return result


# ── Core replacement ──────────────────────────────────────────────────────────


def replace_corrupted_frames(
    video_path: str | Path,
    mask: np.ndarray,
    corrupted_info: dict,
    output_path: str | Path,
    n_avg: int = 5,
    run_padding: int = 5,
) -> None:
    """
    Replace corrupted frames inside the mask region by temporal averaging.

    For each corrupted frame, the replacement is the pixel-wise mean of the n_avg
    nearest clean frames before it plus the n_avg nearest clean frames after it.

    run_padding dilates every detected corrupted run by that many frames on both
    sides before replacement.  This is necessary because flashes typically have a
    gradual onset / fade-out: the transitional frames are only slightly above the
    detection threshold and may be missed, producing a "double-flash" artifact
    (clean core surrounded by two residual bright edges).  Padding ensures the
    full transition is replaced alongside the core.

    Pixels outside the mask are left unchanged.
    The corrected video is saved as .avi (XVID codec preferred, MJPG fallback).
    A JSON log of all replaced frames is written alongside the output video.

    Args:
        video_path:   Path to the original .avi video.
        mask:         Binary mask (H, W) uint8 from step [1].
        corrupted_info: dict returned by detect_corrupted_frames().
        output_path:  Destination .avi path for the corrected video.
        n_avg:        Clean frames to average on each side (default 5).
        run_padding:  Extra frames added on each side of every detected run to
                      cover flash transitions (default 5).

    Raises:
        FileNotFoundError: if video_path cannot be opened.
        RuntimeError:      if no working video codec is available for writing.
    """
    video_path  = Path(video_path)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    frames, meta = _read_video_frames(video_path)
    n                    = len(frames)
    originally_corrupted = set(corrupted_info["all_corrupted"])

    # Dilate every corrupted run by run_padding frames on each side to cover
    # flash onset / fade-out transitions that fall below the detection threshold.
    corrupted_set = set(originally_corrupted)
    if run_padding > 0:
        for idx in originally_corrupted:
            for p in range(1, run_padding + 1):
                if idx - p >= 0:
                    corrupted_set.add(idx - p)
                if idx + p < n:
                    corrupted_set.add(idx + p)

    mask_bool = mask > 0

    # float64 working copies — only corrupted frames will be overwritten
    corrected: list[np.ndarray] = [f.copy().astype(np.float64) for f in frames]
    ref_counts: dict[int, int]  = {}

    for idx in corrupted_set:
        before = _collect_clean_frames(frames, corrupted_set, idx - 1, n_avg, -1)
        after  = _collect_clean_frames(frames, corrupted_set, idx + 1, n_avg, +1)
        ref    = before + after
        ref_counts[idx] = len(ref)

        if not ref:
            warnings.warn(
                f"Frame {idx}: no clean reference found anywhere — frame left unchanged.",
                UserWarning,
                stacklevel=2,
            )
            continue

        replacement = np.mean(ref, axis=0)              # pixel-wise mean of reference frames
        corrected[idx][mask_bool] = replacement[mask_bool]

    # ── Write output video ────────────────────────────────────────────────────
    writer = _make_video_writer(
        output_path, meta["fps"], meta["width"], meta["height"]
    )

    flash_set = set(corrupted_info["flash_frames"])
    blink_set = set(corrupted_info["blink_frames"])
    replacement_log: dict[str, dict] = {}

    for idx, frame_f in enumerate(corrected):
        frame_u8  = np.clip(frame_f, 0, 255).astype(np.uint8)
        frame_bgr = cv2.cvtColor(frame_u8, cv2.COLOR_GRAY2BGR)
        writer.write(frame_bgr)
        if idx in corrupted_set:
            is_padding = idx not in originally_corrupted
            frame_type = (
                "transition" if is_padding else
                "flash"      if idx in flash_set else
                "blink"      if idx in blink_set else
                "occlusion"
            )
            replacement_log[str(idx)] = {
                "type":        frame_type,
                "n_refs_used": ref_counts.get(idx, 0),
            }

    writer.release()
    print(f"  Saved corrected video : {output_path}")

    log_path = output_path.with_suffix(".json")
    with open(log_path, "w") as fh:
        json.dump(replacement_log, fh, indent=2)
    print(f"  Saved replacement log : {log_path}")


# ── Visualisation ─────────────────────────────────────────────────────────────


def visualize_corrupted(
    corrupted_info: dict,
    output_path: str | Path,
) -> None:
    """
    Save a PNG with two plots:
      - Top: per-frame mean luminosity and local-percentile curve, with
             flash frames marked in red and blink/occlusion frames in blue.
      - Bottom: frame-to-frame Pearson correlation with the detection threshold.

    Args:
        corrupted_info: dict returned by detect_corrupted_frames().
        output_path:    Destination PNG path.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    lum        = corrupted_info["luminosity_curve"]
    local_max  = corrupted_info["local_max_curve"]
    corr       = corrupted_info["correlation_curve"]
    percentile = corrupted_info.get("local_percentile", 99)
    n          = len(lum)

    flash_set   = set(corrupted_info["flash_frames"])
    blink_set   = set(corrupted_info["blink_frames"]) | set(corrupted_info["occlusion_frames"])

    fig, axes = plt.subplots(2, 1, figsize=(14, 8), sharex=True)

    # ── Top: luminosity ───────────────────────────────────────────────────────
    ax = axes[0]
    ax.plot(lum,       color="gray",   linewidth=0.8, label="Mean luminosity")
    ax.plot(local_max, color="orange", linewidth=0.8, alpha=0.7,
            label=f"Local max (p{percentile:.0f})")

    for idx in flash_set:
        ax.axvline(idx, color="red",  alpha=0.35, linewidth=0.8)
    for idx in blink_set - flash_set:
        ax.axvline(idx, color="blue", alpha=0.35, linewidth=0.8)

    # Legend proxies
    ax.axvline(-1, color="red",  alpha=0.9, linewidth=1.5, label="Flash frames")
    ax.axvline(-1, color="blue", alpha=0.9, linewidth=1.5, label="Blink / occlusion")

    lum_mean, lum_std = lum.mean(), lum.std()
    loc_mean, loc_std = local_max.mean(), local_max.std()
    ax.axhline(lum_mean + 3 * lum_std, color="red",    linestyle="--", linewidth=0.8, alpha=0.6)
    ax.axhline(lum_mean - 3 * lum_std, color="blue",   linestyle="--", linewidth=0.8, alpha=0.6)
    ax.axhline(loc_mean + 3 * loc_std, color="orange",  linestyle="--", linewidth=0.8, alpha=0.6)

    ax.set_ylabel("Pixel value (0–255)")
    ax.set_title(
        f"Luminosity — {len(flash_set)} flash, "
        f"{len(corrupted_info['blink_frames'])} blink, "
        f"{len(corrupted_info['occlusion_frames'])} occlusion"
    )
    ax.legend(loc="upper right", fontsize=9)
    ax.set_xlim(0, n - 1)

    # ── Bottom: correlation ───────────────────────────────────────────────────
    ax2 = axes[1]
    valid = ~np.isnan(corr)
    ax2.plot(np.where(valid)[0], corr[valid], color="steelblue", linewidth=0.8)

    for idx in corrupted_info["occlusion_frames"]:
        ax2.axvline(idx, color="purple", alpha=0.35, linewidth=0.8)

    corr_threshold = 0.5   # display only — actual threshold stored in corrupted_info
    ax2.axhline(corr_threshold, color="blue", linestyle="--", linewidth=0.8, alpha=0.6,
                label=f"Threshold ({corr_threshold})")
    ax2.set_ylabel("Pearson r")
    ax2.set_xlabel("Frame index")
    ax2.set_title("Frame-to-frame correlation")
    ax2.set_ylim(-0.1, 1.05)
    ax2.legend(loc="lower right", fontsize=9)

    plt.tight_layout()
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(str(output_path), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved : {output_path}")


def compare_frames(
    video_original: str | Path,
    video_corrected: str | Path,
    corrupted_info: dict,
    output_path: str | Path,
    max_panels: int = 20,
) -> None:
    """
    Save a grid image comparing original vs corrected frames side-by-side for
    each replaced frame (up to max_panels pairs).

    Rows:    0 = original (flash → red title, blink → blue, occlusion → purple)
             1 = corrected
    Columns: one per replaced frame (capped at max_panels).

    Args:
        video_original:  Path to the original video.
        video_corrected: Path to the corrected video.
        corrupted_info:  dict returned by detect_corrupted_frames().
        output_path:     Destination PNG path.
        max_panels:      Maximum number of frame pairs to display (default 20).
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    corrupted = corrupted_info["all_corrupted"]
    if not corrupted:
        print("  No corrupted frames — nothing to compare.")
        return

    indices     = corrupted[:max_panels]
    n_panels    = len(indices)
    flash_set   = set(corrupted_info["flash_frames"])
    blink_set   = set(corrupted_info["blink_frames"])

    def _read_frame(cap: cv2.VideoCapture, idx: int) -> np.ndarray | None:
        cap.set(cv2.CAP_PROP_POS_FRAMES, float(idx))
        ret, frame = cap.read()
        if not ret:
            return None
        return cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if frame.ndim == 3 else frame

    cap_orig = cv2.VideoCapture(str(video_original))
    cap_corr = cv2.VideoCapture(str(video_corrected))

    fig, axes = plt.subplots(2, n_panels, figsize=(3 * n_panels, 6))
    if n_panels == 1:
        axes = axes.reshape(2, 1)

    for col, idx in enumerate(indices):
        orig = _read_frame(cap_orig, idx)
        corr = _read_frame(cap_corr, idx)

        color = (
            "red"    if idx in flash_set else
            "blue"   if idx in blink_set else
            "purple"
        )
        kind = (
            "flash"     if idx in flash_set else
            "blink"     if idx in blink_set else
            "occlusion"
        )

        for row, (img, title) in enumerate([
            (orig, f"#{idx}\n({kind})"),
            (corr, f"#{idx}\ncorrected"),
        ]):
            ax = axes[row, col]
            if img is not None:
                ax.imshow(img, cmap="gray", vmin=0, vmax=255)
            else:
                ax.text(0.5, 0.5, "N/A", ha="center", va="center", transform=ax.transAxes)
            ax.set_title(title, fontsize=8, color=color if row == 0 else "black")
            ax.axis("off")

    cap_orig.release()
    cap_corr.release()

    shown = f"{n_panels}/{len(corrupted)}" if n_panels < len(corrupted) else str(n_panels)
    fig.suptitle(
        f"Original vs corrected — {len(corrupted)} frames replaced (showing {shown})",
        fontsize=11,
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
        print(
            "Usage: python BadFrameDétection.py <video.avi> [mask.png] [output_dir]\n"
            "  mask.png   : pre-computed mask PNG from step [1] (optional)\n"
            "  output_dir : directory for outputs (default: same folder as the video)",
        )
        sys.exit(1)

    video_path = Path(sys.argv[1])
    output_dir = Path(sys.argv[3]) if len(sys.argv) > 3 else Path.cwd()

    # ── Mask ──────────────────────────────────────────────────────────────────
    if len(sys.argv) > 2 and Path(sys.argv[2]).exists():
        mask = load_mask(sys.argv[2])
        print(f"Mask loaded : {sys.argv[2]}")
    else:
        print("Detecting circular mask …")
        result = detect_circular_mask(video_path)
        mask   = result["mask"]
        print(f"  coverage: {result['coverage']:.1%}")

    def _fmt(lst: list[int]) -> str:
        preview = lst[:10]
        suffix  = "…" if len(lst) > 10 else ""
        return f"{len(lst):4d}  {preview}{suffix}"

    # ── Iterative detect → replace loop ──────────────────────────────────────
    # After each replacement pass, re-run detection on the corrected video.
    # Flash transitions that survived the previous pass become visible as a new
    # detection once the bright core is gone.  Repeat until no corrupted frames
    # remain or max_iterations is reached.
    MAX_ITER      = 10
    current_path  = video_path          # input for this iteration
    corrected_path = output_dir / f"corrected_{video_path.stem}.avi"
    all_info_per_iter: list[dict] = []

    for iteration in range(1, MAX_ITER + 1):
        print(f"\n── Iteration {iteration} / {MAX_ITER}  (input: {current_path.name}) ──")

        info  = detect_corrupted_frames(current_path, mask)
        total = len(info["luminosity_curve"])
        all_info_per_iter.append(info)

        print(f"  Flash frames      : {_fmt(info['flash_frames'])}")
        print(f"  Blink frames      : {_fmt(info['blink_frames'])}")
        print(f"  Occlusion frames  : {_fmt(info['occlusion_frames'])}")
        print(f"  Total corrupted   : {len(info['all_corrupted'])} / {total}")

        if not info["all_corrupted"]:
            print("  No corrupted frames detected — stopping.")
            break

        # >20 % guard — only ask on first iteration
        if iteration == 1:
            ratio = len(info["all_corrupted"]) / total
            if ratio > 0.20:
                answer = input(
                    f"\n  WARNING: {ratio:.1%} of frames are corrupted. Continue? [y/N] "
                )
                if answer.strip().lower() != "y":
                    print("Aborted.")
                    sys.exit(0)

        iter_out = output_dir / f"corrected_{video_path.stem}_iter{iteration}.avi"
        replace_corrupted_frames(current_path, mask, info, iter_out)
        current_path = iter_out          # next iteration reads the corrected video

    else:
        print(f"\n  Reached max iterations ({MAX_ITER}) — stopping.")

    # Final corrected video = last iter output (rename for clean output name)
    if current_path != video_path:
        import shutil
        shutil.copy2(current_path, corrected_path)
        print(f"  Final video copied to : {corrected_path}")

    # ── Visualise first iteration analysis ───────────────────────────────────
    viz_path = output_dir / f"corrupted_analysis_{video_path.stem}.png"
    visualize_corrupted(all_info_per_iter[0], viz_path)

    # ── Compare original vs final ─────────────────────────────────────────────
    compare_path = output_dir / f"comparison_{video_path.stem}.png"
    compare_frames(video_path, corrected_path, all_info_per_iter[0], compare_path)

    print("\nDone.")


if __name__ == "__main__":
    main()
