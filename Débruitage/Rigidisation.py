#!/usr/bin/env python3
"""
Step [4] — Rigid registration (translation + rotation stabilization).

Aligns all frames to a robust temporal-median reference frame using the
ECC (Enhanced Correlation Coefficient) algorithm with MOTION_EUCLIDEAN
model (translation + rotation only, no shear or scaling).

Functions:
  compute_reference_frame  — temporal median of first N healthy frames
  estimate_motion          — per-frame ECC registration with fallback
  apply_stabilization      — warp and mask all frames, write .avi
  save_motion / load_motion — persist transforms as .npz + .json sidecar
  visualize_motion          — translation X/Y and rotation curves
  compare_stabilization     — side-by-side original / stabilized / diff grid

Usage (standalone):
  python Rigidisation.py <video.avi> [output_dir]

  video.avi  : FPN-corrected video (output of step [3])
  output_dir : defaults to <video_stem>_stabilized/

  Expects in output_dir (or video parent dir):
    mask.png              — binary circular mask (step [1])
    step2_corrected.json  — corrupted frame log  (step [2])
"""
from __future__ import annotations

import json
import sys
import warnings
from pathlib import Path

import cv2
import numpy as np

from mask_detection import load_mask


# ── ECC parameters (edit here) ────────────────────────────────────────────────

ECC_MAX_ITERATIONS      = 200      # max optimizer iterations per frame
ECC_TERMINATION_EPS     = 1e-6     # convergence threshold
ECC_GAUSS_FILT_SIZE     = 5        # Gaussian pre-blur kernel size (must be odd; 0 = disabled)
ECC_MAX_TRANSLATION_PX  = 50.0     # reject transform if |tx| or |ty| exceeds this (px)
ECC_MAX_ROTATION_DEG    = 10.0     # reject transform if |angle| exceeds this (degrees)
ECC_N_REFERENCE_FRAMES  = 60       # number of healthy frames used for reference median


# ── Internal helpers ──────────────────────────────────────────────────────────


def _make_video_writer(
    path: Path,
    fps: float,
    width: int,
    height: int,
) -> cv2.VideoWriter:
    """Try XVID → MJPG → mp4v; return the first working VideoWriter."""
    for fourcc_str in ("XVID", "MJPG", "mp4v"):
        fourcc = cv2.VideoWriter_fourcc(*fourcc_str)
        writer = cv2.VideoWriter(str(path), fourcc, fps, (width, height), isColor=True)
        if writer.isOpened():
            return writer
        writer.release()
    raise RuntimeError(
        f"Could not open VideoWriter for {path}. "
        "Tried XVID, MJPG, mp4v — none worked. "
        "Check that OpenCV was built with video-write support."
    )


def _warp_to_angle(warp: np.ndarray) -> float:
    """Extract rotation angle in degrees from a 2×3 Euclidean warp matrix."""
    return float(np.degrees(np.arctan2(float(warp[1, 0]), float(warp[0, 0]))))


def _read_gray(cap: cv2.VideoCapture, idx: int) -> np.ndarray | None:
    """Seek to frame idx and return it as uint8 grayscale, or None on failure."""
    cap.set(cv2.CAP_PROP_POS_FRAMES, float(idx))
    ret, frame = cap.read()
    if not ret:
        return None
    if frame.ndim == 3:
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    return frame


def _progress(i: int, total: int, label: str = "") -> None:
    bar_len = 30
    filled  = int(bar_len * (i + 1) / total)
    bar     = "█" * filled + "░" * (bar_len - filled)
    pct     = (i + 1) / total * 100
    print(f"\r  {label} [{bar}] {i+1}/{total} {pct:.0f}%", end="", flush=True)
    if i + 1 == total:
        print()


# ── Core functions ────────────────────────────────────────────────────────────


def compute_reference_frame(
    video_path: str | Path,
    mask: np.ndarray,
    corrupted_frames: list[int],
    n_frames: int = ECC_N_REFERENCE_FRAMES,
) -> np.ndarray:
    """
    Compute the reference frame as a temporal median of the first n_frames
    healthy (non-corrupted) frames.

    Using the median rather than the mean ensures a single anomalous frame
    cannot bias the reference.  Restricting to the first n_frames healthy
    frames avoids including drift that would shift the target.

    Args:
        video_path:       Source video (FPN-corrected).
        mask:             Binary mask (uint8) from step [1].
        corrupted_frames: List of frame indices to skip (from step [2]).
        n_frames:         Maximum number of healthy frames to use.

    Returns:
        float32 (H, W) reference frame.

    Raises:
        FileNotFoundError: if the video cannot be opened.
        ValueError:        if no healthy frame can be read.
    """
    video_path = Path(video_path)
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise FileNotFoundError(f"Cannot open video: {video_path}")

    total      = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    skip       = set(corrupted_frames)
    candidates = [i for i in range(total) if i not in skip][:n_frames]

    if not candidates:
        cap.release()
        raise ValueError("No healthy frames available to compute the reference.")

    stack: list[np.ndarray] = []
    for idx in candidates:
        frame = _read_gray(cap, idx)
        if frame is not None:
            stack.append(frame.astype(np.float32))

    cap.release()

    if not stack:
        raise ValueError("Could not read any healthy frame from the video.")

    reference = np.median(np.stack(stack, axis=0), axis=0).astype(np.float32)
    print(f"  Reference: median of {len(stack)} healthy frames "
          f"(target {n_frames}, video total {total})")
    return reference


def estimate_motion(
    video_path: str | Path,
    reference_frame: np.ndarray,
    mask: np.ndarray,
    corrupted_frames: list[int],
) -> dict:
    """
    Estimate per-frame rigid motion (dx, dy, rotation) with ECC.

    For each frame, cv2.findTransformECC estimates the 2×3 Euclidean warp
    matrix W such that warpAffine(frame, W) ≈ reference_frame.

    Failure handling:
      - Corrupted frames (from step [2]): inherit previous transform.
      - ECC exception or aberrant result (|tx|>50 px or |rotation|>10°):
        inherit previous transform and record the frame in failed_frames.

    Args:
        video_path:       Source video (FPN-corrected).
        reference_frame:  float32 (H, W) reference from compute_reference_frame().
        mask:             Binary mask (uint8).
        corrupted_frames: Frame indices to skip.

    Returns:
        dict with keys:
          'transforms'     : list[np.ndarray]  — 2×3 float32 warp per frame
          'translations_x' : np.ndarray float32
          'translations_y' : np.ndarray float32
          'rotations'      : np.ndarray float32 (degrees)
          'failed_frames'  : list[int]
    """
    video_path = Path(video_path)
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise FileNotFoundError(f"Cannot open video: {video_path}")

    total    = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    skip     = set(corrupted_frames)
    mask_bool = mask > 0

    # Prepare reference: pre-blur and mask
    ref = reference_frame.astype(np.float32)
    if ECC_GAUSS_FILT_SIZE > 0:
        k   = ECC_GAUSS_FILT_SIZE | 1        # ensure odd
        ref = cv2.GaussianBlur(ref, (k, k), 0)
    ref_masked           = ref.copy()
    ref_masked[~mask_bool] = 0.0

    criteria = (
        cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT,
        ECC_MAX_ITERATIONS,
        ECC_TERMINATION_EPS,
    )

    transforms:    list[np.ndarray] = []
    tx_list:       list[float]      = []
    ty_list:       list[float]      = []
    angle_list:    list[float]      = []
    failed_frames: list[int]        = []
    prev_warp = np.eye(2, 3, dtype=np.float32)

    cap.set(cv2.CAP_PROP_POS_FRAMES, 0.0)

    for i in range(total):
        ret, frame = cap.read()

        # Unreadable frame: fall back to previous transform
        if not ret:
            transforms.append(prev_warp.copy())
            tx_list.append(float(prev_warp[0, 2]))
            ty_list.append(float(prev_warp[1, 2]))
            angle_list.append(_warp_to_angle(prev_warp))
            failed_frames.append(i)
            _progress(i, total, "Estimating motion")
            continue

        if frame.ndim == 3:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        frame_f = frame.astype(np.float32)

        # Corrupted frame: inherit previous transform silently
        if i in skip:
            transforms.append(prev_warp.copy())
            tx_list.append(float(prev_warp[0, 2]))
            ty_list.append(float(prev_warp[1, 2]))
            angle_list.append(_warp_to_angle(prev_warp))
            _progress(i, total, "Estimating motion")
            continue

        # Pre-blur and mask the current frame
        if ECC_GAUSS_FILT_SIZE > 0:
            k       = ECC_GAUSS_FILT_SIZE | 1
            frame_f = cv2.GaussianBlur(frame_f, (k, k), 0)
        frame_masked           = frame_f.copy()
        frame_masked[~mask_bool] = 0.0

        warp    = prev_warp.copy()
        failed  = False

        try:
            _, warp = cv2.findTransformECC(
                ref_masked,
                frame_masked,
                warp,
                cv2.MOTION_EUCLIDEAN,
                criteria,
            )
            tx    = float(warp[0, 2])
            ty    = float(warp[1, 2])
            angle = _warp_to_angle(warp)

            if (abs(tx) > ECC_MAX_TRANSLATION_PX
                    or abs(ty) > ECC_MAX_TRANSLATION_PX
                    or abs(angle) > ECC_MAX_ROTATION_DEG):
                failed = True

        except cv2.error:
            failed = True

        if failed:
            warp = prev_warp.copy()
            failed_frames.append(i)

        transforms.append(warp.copy())
        tx_list.append(float(warp[0, 2]))
        ty_list.append(float(warp[1, 2]))
        angle_list.append(_warp_to_angle(warp))
        prev_warp = warp

        _progress(i, total, "Estimating motion")

    cap.release()

    fail_rate = len(failed_frames) / total if total > 0 else 0.0
    if fail_rate > 0.10:
        warnings.warn(
            f"Registration failed on {len(failed_frames)}/{total} frames "
            f"({fail_rate:.1%}). Consider checking video quality or ECC parameters.",
            RuntimeWarning,
            stacklevel=2,
        )

    return {
        "transforms":     transforms,
        "translations_x": np.array(tx_list,    dtype=np.float32),
        "translations_y": np.array(ty_list,    dtype=np.float32),
        "rotations":      np.array(angle_list, dtype=np.float32),
        "failed_frames":  failed_frames,
    }


def apply_stabilization(
    video_path: str | Path,
    mask: np.ndarray,
    motion_dict: dict,
    output_path: str | Path,
) -> None:
    """
    Apply the estimated rigid transforms to every frame and save the result.

    After warping, pixels outside the mask are zeroed so downstream steps
    never see edge artefacts introduced by the warp.

    Args:
        video_path:   Source video (FPN-corrected).
        mask:         Binary mask (uint8).
        motion_dict:  Output of estimate_motion() or load_motion().
        output_path:  Destination .avi path.
    """
    video_path  = Path(video_path)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise FileNotFoundError(f"Cannot open video: {video_path}")

    total  = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps    = float(cap.get(cv2.CAP_PROP_FPS)) or 30.0
    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    writer    = _make_video_writer(output_path, fps, width, height)
    transforms = motion_dict["transforms"]
    mask_bool  = mask > 0

    cap.set(cv2.CAP_PROP_POS_FRAMES, 0.0)

    for i in range(total):
        ret, frame = cap.read()
        if not ret:
            break

        if frame.ndim == 3:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        if i < len(transforms):
            warped = cv2.warpAffine(
                frame,
                transforms[i],
                (width, height),
                flags=cv2.INTER_LINEAR,
                borderMode=cv2.BORDER_CONSTANT,
                borderValue=0,
            )
        else:
            warped = frame.copy()

        warped[~mask_bool] = 0
        writer.write(cv2.cvtColor(warped, cv2.COLOR_GRAY2BGR))

        _progress(i, total, "Applying stabilization")

    cap.release()
    writer.release()


# ── I/O ──────────────────────────────────────────────────────────────────────


def save_motion(motion_dict: dict, output_path: str | Path) -> None:
    """
    Save motion data to a .npz file (for fast reloading) and a .json sidecar
    (human-readable per-frame log).

    Args:
        motion_dict: Output of estimate_motion().
        output_path: Destination .npz path.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    transforms_arr = np.stack(motion_dict["transforms"], axis=0)   # (N, 2, 3)
    np.savez(
        str(output_path),
        transforms    = transforms_arr,
        translations_x = motion_dict["translations_x"],
        translations_y = motion_dict["translations_y"],
        rotations      = motion_dict["rotations"],
        failed_frames  = np.array(motion_dict["failed_frames"], dtype=np.int32),
    )

    # JSON sidecar: one entry per frame for manual inspection
    json_path   = output_path.with_suffix(".json")
    failed_set  = set(motion_dict["failed_frames"])
    tx          = motion_dict["translations_x"]
    ty          = motion_dict["translations_y"]
    rot         = motion_dict["rotations"]

    log: dict = {
        "n_total":       len(tx),
        "n_failed":      len(failed_set),
        "failed_frames": motion_dict["failed_frames"],
        "frames":        {},
    }
    for i in range(len(tx)):
        log["frames"][str(i)] = {
            "tx":     round(float(tx[i]),  3),
            "ty":     round(float(ty[i]),  3),
            "angle":  round(float(rot[i]), 4),
            "failed": i in failed_set,
        }
    with open(json_path, "w") as fh:
        json.dump(log, fh, indent=2)

    print(f"  Motion saved : {output_path.name}  +  {json_path.name}")


def load_motion(motion_path: str | Path) -> dict:
    """
    Reload motion data previously saved by save_motion().

    Args:
        motion_path: Path to the .npz file.

    Returns:
        dict matching the structure returned by estimate_motion().
    """
    motion_path    = Path(motion_path)
    data           = np.load(str(motion_path))
    transforms_arr = data["transforms"]              # (N, 2, 3)

    return {
        "transforms":     [transforms_arr[i] for i in range(len(transforms_arr))],
        "translations_x": data["translations_x"],
        "translations_y": data["translations_y"],
        "rotations":      data["rotations"],
        "failed_frames":  data["failed_frames"].tolist(),
    }


# ── Visualisation ─────────────────────────────────────────────────────────────


def visualize_motion(motion_dict: dict, output_path: str | Path) -> None:
    """
    Save a figure with three time-series curves: translation X, translation Y,
    and rotation angle.  Failed frames are marked as vertical red lines.

    Args:
        motion_dict: Output of estimate_motion() or load_motion().
        output_path: Destination PNG.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    output_path = Path(output_path)
    tx     = motion_dict["translations_x"]
    ty     = motion_dict["translations_y"]
    rot    = motion_dict["rotations"]
    failed = set(motion_dict["failed_frames"])
    n      = len(tx)
    frames = np.arange(n)

    fig, axes = plt.subplots(3, 1, figsize=(14, 8), sharex=True)

    series = [
        (tx,  "Translation X (px)", "steelblue"),
        (ty,  "Translation Y (px)", "darkorange"),
        (rot, "Rotation (°)",       "seagreen"),
    ]

    for ax, (data, ylabel, color) in zip(axes, series):
        ax.plot(frames, data, color=color, linewidth=0.8, zorder=2)
        ax.axhline(0, color="black", linewidth=0.5, linestyle="--", zorder=1)
        ax.set_ylabel(ylabel, fontsize=10)
        ax.grid(True, alpha=0.3, zorder=0)
        for fi in failed:
            ax.axvline(fi, color="red", alpha=0.35, linewidth=0.9, zorder=3)

    if failed:
        # One labelled entry in the legend (top subplot only)
        axes[0].axvline(
            next(iter(failed)), color="red", alpha=0.35, linewidth=0.9,
            label=f"Failed frames ({len(failed)})",
        )
        axes[0].legend(loc="upper right", fontsize=9)

    axes[-1].set_xlabel("Frame index", fontsize=10)

    pct = len(failed) / n * 100 if n > 0 else 0.0
    fig.suptitle(
        f"Motion estimation — {n} frames  |  Failed: {len(failed)} ({pct:.1f}%)",
        fontsize=12, fontweight="bold",
    )
    plt.tight_layout()
    plt.savefig(str(output_path), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved : {output_path}")


def compare_stabilization(
    video_original:   str | Path,
    video_stabilized: str | Path,
    motion_dict:      dict,
    output_path:      str | Path,
    reference_frame:  np.ndarray | None = None,
    n_samples:        int = 5,
) -> None:
    """
    Save a comparison grid: original / stabilized / diff×5 vs reference,
    plus a 100×100 crop on a vessel-rich zone for each sample frame.

    Row layout:
      Row 0 — Original frame
      Row 1 — Stabilized frame
      Row 2 — |stabilized − reference| × 5  (highlights residual motion)
      Row 3 — 100×100 crop (original | stabilized side-by-side)

    The crop location is chosen as the 100×100 window with the highest
    temporal standard deviation across the stabilized samples (vessel-rich
    regions have high temporal contrast).

    Args:
        video_original:   Source video before stabilization.
        video_stabilized: Video produced by apply_stabilization().
        motion_dict:      Used to annotate failed frames.
        output_path:      Destination PNG.
        reference_frame:  Reference used during registration (optional).
                          When None, the mean of the stabilized samples is used.
        n_samples:        Number of evenly-spaced frames to display (default 5).
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    video_original   = Path(video_original)
    video_stabilized = Path(video_stabilized)
    output_path      = Path(output_path)

    cap_o = cv2.VideoCapture(str(video_original))
    cap_s = cv2.VideoCapture(str(video_stabilized))

    total   = int(cap_o.get(cv2.CAP_PROP_FRAME_COUNT))
    indices = np.linspace(0, total - 1, n_samples, dtype=int).tolist()

    orig_frames: list[tuple[int, np.ndarray]] = []
    stab_frames: list[np.ndarray]             = []

    for idx in indices:
        fo = _read_gray(cap_o, idx)
        fs = _read_gray(cap_s, idx)
        if fo is not None and fs is not None:
            orig_frames.append((idx, fo))
            stab_frames.append(fs)

    cap_o.release()
    cap_s.release()

    n = len(orig_frames)
    if n == 0:
        print("  compare_stabilization: no frames could be read — skipping.")
        return

    # Reference image for the diff row
    if reference_frame is not None:
        ref = reference_frame.astype(np.float32)
    else:
        ref = np.mean([f.astype(np.float32) for f in stab_frames], axis=0)

    # Find the 100×100 crop with highest temporal std (vessel-rich)
    std_map          = np.std([f.astype(np.float32) for f in stab_frames], axis=0)
    h_img, w_img     = std_map.shape
    best, crop_y, crop_x = -1.0, h_img // 4, w_img // 4
    for yy in range(50, h_img - 150, 20):
        for xx in range(50, w_img - 150, 20):
            score = float(std_map[yy: yy + 100, xx: xx + 100].mean())
            if score > best:
                best, crop_y, crop_x = score, yy, xx

    failed_set  = set(motion_dict.get("failed_frames", []))
    row_labels  = [
        "Original",
        "Stabilized",
        "Diff ×5 vs ref",
        "Crop 100 px\n(orig | stab)",
    ]

    fig, axes = plt.subplots(4, n, figsize=(4 * n, 16))
    if n == 1:
        axes = axes.reshape(-1, 1)

    for col, ((fidx, orig_f), stab_f) in enumerate(zip(orig_frames, stab_frames)):
        diff      = np.clip(np.abs(stab_f.astype(np.float32) - ref) * 5, 0, 255).astype(np.uint8)
        orig_crop = orig_f[crop_y: crop_y + 100, crop_x: crop_x + 100]
        stab_crop = stab_f[crop_y: crop_y + 100, crop_x: crop_x + 100]
        both_crop = np.concatenate([orig_crop, stab_crop], axis=1)

        for row, img in enumerate([orig_f, stab_f, diff, both_crop]):
            ax = axes[row, col]
            ax.imshow(img, cmap="gray", vmin=0, vmax=255)
            ax.axis("off")

        tag   = " [FAIL]" if fidx in failed_set else ""
        color = "red" if tag else "black"
        axes[0, col].set_title(f"Frame {fidx}{tag}", fontsize=9, color=color)

    for row, label in enumerate(row_labels):
        axes[row, 0].set_ylabel(label, fontsize=9, rotation=0, labelpad=90, va="center")

    fig.suptitle("Stabilization comparison", fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.savefig(str(output_path), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved : {output_path}")


# ── CLI entry point ───────────────────────────────────────────────────────────


def main() -> None:
    if len(sys.argv) < 2:
        print(
            "Usage:\n"
            "  python Rigidisation.py <video.avi> [output_dir]\n"
            "\n"
            "  video.avi  : FPN-corrected video (step [3] output)\n"
            "  output_dir : defaults to ./preprocessed/<video_stem>/\n"
            "\n"
            "Expected in output_dir (or video parent dir):\n"
            "  mask.png              — binary circular mask  (step [1])\n"
            "  step2_corrected.json  — corrupted frame log   (step [2])\n"
        )
        sys.exit(1)

    video_path = Path(sys.argv[1])
    if not video_path.exists():
        print(f"Error: file not found — {video_path}")
        sys.exit(1)

    output_dir = (
        Path(sys.argv[2]) if len(sys.argv) > 2
        else Path.cwd() / "preprocessed" / video_path.stem
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    # ── Mask ─────────────────────────────────────────────────────────────────
    for candidate in [output_dir / "mask.png", video_path.parent / "mask.png"]:
        if candidate.exists():
            mask = load_mask(candidate)
            print(f"  Mask loaded : {candidate}")
            break
    else:
        print(f"Error: mask.png not found in {output_dir} or {video_path.parent}. "
              "Run step [1] first.")
        sys.exit(1)

    # ── Corrupted frames ──────────────────────────────────────────────────────
    corrupted_frames: list[int] = []
    for candidate in [
        output_dir / "step2_corrected.json",
        video_path.parent / "step2_corrected.json",
    ]:
        if candidate.exists():
            with open(candidate) as fh:
                corrupted_frames = [int(k) for k in json.load(fh).keys()]
            print(f"  Corrupted frames : {len(corrupted_frames)} loaded from {candidate.name}")
            break

    # ── Reference frame ───────────────────────────────────────────────────────
    print("\n── Computing reference frame …")
    reference = compute_reference_frame(video_path, mask, corrupted_frames)

    # ── Motion estimation (cached) ────────────────────────────────────────────
    motion_npz = output_dir / "motion.npz"
    if motion_npz.exists():
        print(f"\n── Loading cached motion : {motion_npz.name}")
        motion_dict = load_motion(motion_npz)
    else:
        print("\n── Estimating motion …")
        motion_dict = estimate_motion(video_path, reference, mask, corrupted_frames)
        save_motion(motion_dict, motion_npz)

    n_f = len(motion_dict["failed_frames"])
    n_t = len(motion_dict["transforms"])
    print(f"  Failed : {n_f}/{n_t} ({n_f/n_t:.1%})")

    # ── Apply stabilization (cached) ─────────────────────────────────────────
    stab_path = output_dir / "step4_stabilized.avi"
    if stab_path.exists():
        print(f"\n── Reusing cached stabilized video : {stab_path.name}")
    else:
        print("\n── Applying stabilization …")
        apply_stabilization(video_path, mask, motion_dict, stab_path)
        print(f"  Saved : {stab_path}")

    # ── Visualisations ────────────────────────────────────────────────────────
    motion_plot = output_dir / "motion_plot.png"
    if not motion_plot.exists():
        print("\n── Motion plot …")
        visualize_motion(motion_dict, motion_plot)

    stab_cmp = output_dir / "stabilization_comparison.png"
    if not stab_cmp.exists():
        print("\n── Comparison grid …")
        compare_stabilization(
            video_path, stab_path, motion_dict, stab_cmp,
            reference_frame=reference,
        )

    print(f"\nDone. Output : {output_dir}")


if __name__ == "__main__":
    main()
