#!/usr/bin/env python3
"""
Step [3] — Fixed Pattern Noise (FPN) estimation and correction.

FPN is sensor-fixed structured noise that appears at the same pixels in every
frame.  It must be removed before stabilisation because registration would
shift the pattern, making it impossible to correct afterwards.

Key constraint for ocular imaging: the eye barely moves between frames, so the
temporal mean contains sharp vessel structures.  A naive mean-minus-blur with
a small sigma will detect vessels as "fixed pattern" and erase them.

Correct approach: use a very large Gaussian sigma (≥ 50 px) so that the blur
captures low-frequency vessel structure, leaving only the high-frequency
pixel-to-pixel sensor noise in the residual.

Two safety checks prevent vessel erasure:
  1. |correlation(pattern, mean_frame)| > FPN_CORRELATION_THRESHOLD → vessel
     content leaked into the pattern; correction is blocked.
  2. pattern_std > FPN_MAX_VALID_STD after applying the chosen sigma → no
     detectable sensor FPN; correction is skipped.
"""
from __future__ import annotations

import warnings
from pathlib import Path

import cv2
import numpy as np
from scipy.ndimage import gaussian_filter

from mask_detection import load_mask


# ── Thresholds (modify here, never hardcode elsewhere) ────────────────────────

FPN_STD_MILD     = 1.0   # std > this → correction recommended
FPN_STD_MODERATE = 2.0   # std > this → correction important
FPN_STD_SEVERE   = 4.0   # std > this → correction mandatory

# A valid sensor-noise pattern should have low std (pixel-to-pixel jitter only).
# If std remains above this threshold even with a large sigma, the video likely
# has no detectable FPN — correction is skipped and a warning is emitted.
FPN_MAX_VALID_STD = 3.0

# If |Pearson r| between the estimated pattern and the temporal mean exceeds
# this, vessel structures have leaked into the pattern — correction is blocked.
FPN_CORRELATION_THRESHOLD = 0.3

FPN_GAUSSIAN_SIGMA = 60.0  # default blur sigma; must exceed vessel widths (≥ 50 px)

MIN_CLEAN_FRAMES = 50    # below this count → estimation reliability warning


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


def _severity_label(
    std: float,
    mild: float,
    moderate: float,
    severe: float,
) -> str:
    if std >= severe:
        return "severe"
    if std >= moderate:
        return "moderate"
    if std >= mild:
        return "mild"
    return "none"


# ── Core estimation ───────────────────────────────────────────────────────────


def estimate_fpn(
    video_path: str | Path,
    mask: np.ndarray,
    corrupted_frames: list[int],
    gaussian_sigma: float = FPN_GAUSSIAN_SIGMA,
    fpn_std_mild: float = FPN_STD_MILD,
    fpn_std_moderate: float = FPN_STD_MODERATE,
    fpn_std_severe: float = FPN_STD_SEVERE,
    max_valid_std: float = FPN_MAX_VALID_STD,
    correlation_threshold: float = FPN_CORRELATION_THRESHOLD,
    min_clean_frames: int = MIN_CLEAN_FRAMES,
) -> dict:
    """
    Estimate the Fixed Pattern Noise on the full video.

    The pattern is the high-frequency residual of the temporal mean: a very
    large Gaussian blur (sigma ≥ 50 px) acts as a low-pass filter that retains
    vessel structures in the background while the difference isolates only the
    high-frequency pixel-to-pixel sensor noise.

    Two safety checks are applied before marking FPN as detected:
      1. Vessel-leakage check: |Pearson r(pattern, mean_frame)| inside the mask.
         If > correlation_threshold the sigma is too small — vessels leaked into
         the pattern.  fpn_detected is forced to False and a warning is emitted.
      2. Amplitude sanity check: if pattern_std > max_valid_std even with a
         large sigma, there is likely no detectable FPN on this sensor.
         fpn_detected is forced to False and a warning is emitted.

    Args:
        video_path:            Path to the .avi video (output of step [2]).
        mask:                  Binary mask (H, W) uint8 from step [1].
        corrupted_frames:      Frame indices to exclude (from step [2]).
        gaussian_sigma:        Sigma (px) for the low-pass background blur.
                               Must be large enough to cover vessel widths
                               (≥ 50 px recommended for 512×512 imaging).
        fpn_std_mild:          Std threshold for mild severity.
        fpn_std_moderate:      Std threshold for moderate severity.
        fpn_std_severe:        Std threshold for severe severity.
        max_valid_std:         Upper bound on pattern_std for a valid sensor
                               pattern.  Above this, no correction is applied.
        correlation_threshold: Max tolerated |r| between pattern and mean_frame.
                               Above this, vessel leakage is declared.
        min_clean_frames:      Warn when fewer clean frames are available.

    Returns:
        dict with keys:
          - 'pattern'             : np.ndarray float32 (H, W).
          - 'mean_frame'          : np.ndarray float32 (H, W) — temporal mean.
          - 'pattern_std'         : float — std of pattern inside mask.
          - 'pattern_correlation' : float — |r| between pattern and mean_frame.
          - 'vessel_leakage'      : bool  — True if correlation > threshold.
          - 'fpn_detected'        : bool  — True only when safe to correct.
          - 'severity'            : str   — 'none'/'mild'/'moderate'/'severe'.
          - 'n_clean'             : int   — clean frames used.
          - 'gaussian_sigma'      : float — sigma used (traceability).

    Raises:
        FileNotFoundError: if video_path cannot be opened.
    """
    video_path = Path(video_path)
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise FileNotFoundError(f"Cannot open video: {video_path}")

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))

    corrupted_set = set(corrupted_frames)
    accumulator   = np.zeros((h, w), dtype=np.float64)
    n_clean       = 0

    for idx in range(total_frames):
        ret, frame = cap.read()
        if not ret:
            break
        if idx in corrupted_set:
            continue
        if frame.ndim == 3:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        accumulator += frame.astype(np.float64)
        n_clean += 1

    cap.release()

    if n_clean == 0:
        raise ValueError("No clean frames available to estimate FPN.")

    if n_clean < min_clean_frames:
        warnings.warn(
            f"Only {n_clean} clean frames available (< {min_clean_frames}). "
            "FPN estimate may be unreliable.",
            UserWarning,
            stacklevel=2,
        )

    mean_frame = (accumulator / n_clean).astype(np.float32)

    # Low-pass background: sigma must be large enough to include vessel widths
    # so that the residual contains only pixel-to-pixel sensor jitter.
    background = gaussian_filter(mean_frame, sigma=gaussian_sigma).astype(np.float32)
    pattern    = (mean_frame - background).astype(np.float32)

    # ── Statistics inside mask only ───────────────────────────────────────────
    mask_bool   = mask > 0
    pattern_std = float(pattern[mask_bool].std()) if mask_bool.any() else 0.0

    # Pearson correlation between pattern and mean_frame inside the mask.
    # A high |r| means vessel structure survived into the pattern (sigma too small).
    if mask_bool.any():
        p_vals = pattern[mask_bool].astype(np.float64)
        m_vals = mean_frame[mask_bool].astype(np.float64)
        p_c    = p_vals - p_vals.mean()
        m_c    = m_vals - m_vals.mean()
        denom  = np.sqrt((p_c ** 2).sum() * (m_c ** 2).sum())
        raw_corr = float(np.dot(p_c, m_c) / denom) if denom > 0 else 0.0
    else:
        raw_corr = 0.0
    pattern_correlation = abs(raw_corr)

    # ── Safety check 1 : vessel leakage ──────────────────────────────────────
    vessel_leakage = pattern_correlation > correlation_threshold
    if vessel_leakage:
        warnings.warn(
            f"FPN VESSEL LEAKAGE: |correlation(pattern, mean_frame)| = "
            f"{pattern_correlation:.3f} > {correlation_threshold}. "
            f"gaussian_sigma={gaussian_sigma} px is too small — vessel structures "
            "have leaked into the estimated pattern. "
            "Increase gaussian_sigma (try 80–120 px) or skip FPN correction. "
            "Correction will NOT be applied.",
            UserWarning,
            stacklevel=2,
        )

    # ── Safety check 2 : amplitude sanity ────────────────────────────────────
    amplitude_invalid = pattern_std > max_valid_std
    if amplitude_invalid and not vessel_leakage:
        warnings.warn(
            f"FPN NOT DETECTABLE: pattern_std={pattern_std:.3f} > "
            f"max_valid_std={max_valid_std} with sigma={gaussian_sigma} px. "
            "This video likely has no measurable sensor FPN. "
            "Correction will NOT be applied.",
            UserWarning,
            stacklevel=2,
        )

    # FPN is safe to correct only when the pattern is plausibly sensor noise
    fpn_detected = (
        pattern_std >= fpn_std_mild
        and not vessel_leakage
        and not amplitude_invalid
    )
    severity = _severity_label(pattern_std, fpn_std_mild, fpn_std_moderate, fpn_std_severe)

    return {
        "pattern":             pattern,
        "mean_frame":          mean_frame,
        "pattern_std":         pattern_std,
        "pattern_correlation": pattern_correlation,
        "vessel_leakage":      vessel_leakage,
        "fpn_detected":        fpn_detected,
        "severity":            severity,
        "n_clean":             n_clean,
        "gaussian_sigma":      gaussian_sigma,
    }


# ── Core correction ───────────────────────────────────────────────────────────


def correct_fpn(
    video_path: str | Path,
    mask: np.ndarray,
    fpn_dict: dict,
    output_path: str | Path,
) -> None:
    """
    Subtract the estimated FPN pattern from every frame inside the mask.

    Pixel values are clipped to [0, 255] after subtraction.  Pixels outside
    the mask are written unchanged.  The output codec follows the same fallback
    chain as step [2]: XVID → MJPG → mp4v.

    Args:
        video_path:  Path to the .avi video to correct.
        mask:        Binary mask (H, W) uint8 from step [1].
        fpn_dict:    dict returned by estimate_fpn().
        output_path: Destination .avi path.

    Raises:
        FileNotFoundError: if video_path cannot be opened.
        RuntimeError:      if no working video codec is available for writing.
    """
    video_path  = Path(video_path)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise FileNotFoundError(f"Cannot open video: {video_path}")

    fps    = cap.get(cv2.CAP_PROP_FPS)
    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Hard block: never apply correction when vessel leakage was detected.
    if fpn_dict.get("vessel_leakage", False):
        raise ValueError(
            "Refusing FPN correction: vessel leakage detected in pattern "
            f"(|correlation|={fpn_dict.get('pattern_correlation', 'N/A'):.3f}). "
            "Call estimate_fpn() with a larger gaussian_sigma to separate "
            "vascular structure from sensor noise before correcting."
        )

    writer    = _make_video_writer(output_path, fps, width, height)
    pattern   = fpn_dict["pattern"].astype(np.float32)
    mask_bool = mask > 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if frame.ndim == 3:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY).astype(np.float32)
        else:
            gray = frame.astype(np.float32)

        gray[mask_bool] -= pattern[mask_bool]
        gray = np.clip(gray, 0, 255).astype(np.uint8)

        writer.write(cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR))

    cap.release()
    writer.release()
    print(f"  Saved corrected video : {output_path}")


# ── Persistence ───────────────────────────────────────────────────────────────


def save_fpn_pattern(fpn_dict: dict, output_path: str | Path) -> None:
    """Save the estimated pattern array to a .npy file."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(str(output_path), fpn_dict["pattern"])
    print(f"  Saved FPN pattern : {output_path}")


def load_fpn_pattern(path: str | Path) -> np.ndarray:
    """Load a previously saved FPN pattern from a .npy file."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Cannot load FPN pattern: {path}")
    return np.load(str(path)).astype(np.float32)


# ── Visualisation ─────────────────────────────────────────────────────────────


def visualize_fpn(
    fpn_dict: dict,
    mask: np.ndarray,
    output_path: str | Path,
    amplification: float = 10.0,
) -> None:
    """
    Save a 3-panel diagnostic PNG:
      1. Pattern amplified ×amplification for naked-eye visibility.
      2. Pattern in false colour (RdBu colormap, centred on 0).
      3. Histogram of pattern values inside the mask.

    Args:
        fpn_dict:      dict returned by estimate_fpn().
        mask:          Binary mask (H, W) uint8 from step [1].
        output_path:   Destination PNG path.
        amplification: Multiplier for the raw-pattern panel (default 10).
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    pattern   = fpn_dict["pattern"]
    std       = fpn_dict["pattern_std"]
    severity  = fpn_dict["severity"]
    mask_bool = mask > 0

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # Panel 1 — amplified pattern (grey)
    amp = np.clip(pattern * amplification + 128, 0, 255).astype(np.uint8)
    axes[0].imshow(amp, cmap="gray", vmin=0, vmax=255)
    axes[0].set_title(f"Pattern ×{amplification:.0f} (grey)")
    axes[0].axis("off")

    # Panel 2 — false colour centred on 0
    vmax = max(float(np.abs(pattern[mask_bool]).max()), 1e-6) if mask_bool.any() else 1.0
    im   = axes[1].imshow(pattern, cmap="RdBu", vmin=-vmax, vmax=vmax)
    axes[1].set_title("Pattern (RdBu, centred 0)")
    axes[1].axis("off")
    plt.colorbar(im, ax=axes[1], fraction=0.046, pad=0.04)

    # Panel 3 — histogram inside mask
    vals = pattern[mask_bool] if mask_bool.any() else pattern.ravel()
    axes[2].hist(vals, bins=100, color="steelblue", edgecolor="none")
    axes[2].axvline(0, color="black", linewidth=0.8, linestyle="--")
    axes[2].set_xlabel("Pattern value (pixel offset)")
    axes[2].set_ylabel("Count")
    axes[2].set_title("Distribution inside mask")

    corr    = fpn_dict.get("pattern_correlation", float("nan"))
    leakage = fpn_dict.get("vessel_leakage", False)
    status  = "VESSEL LEAKAGE — correction blocked" if leakage else (
              "not detectable" if not fpn_dict["fpn_detected"] else "detected"
    )
    fig.suptitle(
        f"FPN analysis — std={std:.3f}  |r|={corr:.3f}  "
        f"severity={severity}  status={status}\n"
        f"n_clean={fpn_dict['n_clean']}  σ_blur={fpn_dict['gaussian_sigma']} px",
        fontsize=11,
        fontweight="bold",
    )
    plt.tight_layout()
    plt.savefig(str(output_path), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved : {output_path}")


def compare_fpn_correction(
    video_path_original: str | Path,
    video_path_corrected: str | Path,
    fpn_dict: dict,
    output_path: str | Path,
    n_samples: int = 3,
    diff_amplification: float = 10.0,
) -> None:
    """
    Save a grid showing n_samples frames: original, corrected, difference ×10.

    If FPN was not detected, a clear message image is written instead.

    Args:
        video_path_original:  Path to the original video.
        video_path_corrected: Path to the FPN-corrected video.
        fpn_dict:             dict returned by estimate_fpn().
        output_path:          Destination PNG path.
        n_samples:            Number of evenly-spaced frames to display (default 3).
        diff_amplification:   Multiplier for the difference panel (default 10).
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if not fpn_dict["fpn_detected"]:
        leakage = fpn_dict.get("vessel_leakage", False)
        corr    = fpn_dict.get("pattern_correlation", float("nan"))
        if leakage:
            msg = (
                f"VESSEL LEAKAGE — correction blocked\n"
                f"|correlation(pattern, mean)| = {corr:.3f} "
                f"> threshold {FPN_CORRELATION_THRESHOLD}\n"
                f"gaussian_sigma={fpn_dict['gaussian_sigma']} px is too small.\n"
                "Increase sigma to separate vessels from sensor noise."
            )
        else:
            msg = (
                f"No FPN detected / not correctable\n"
                f"pattern_std={fpn_dict['pattern_std']:.3f}  "
                f"|r|={corr:.3f}  "
                f"severity={fpn_dict['severity']}\n"
                "Correction was not applied."
            )
        fig, ax = plt.subplots(figsize=(8, 4))
        color = "firebrick" if leakage else "black"
        ax.text(0.5, 0.5, msg, ha="center", va="center",
                fontsize=13, color=color, transform=ax.transAxes)
        ax.axis("off")
        plt.savefig(str(output_path), dpi=100, bbox_inches="tight")
        plt.close(fig)
        print(f"  Saved (no correction) : {output_path}")
        return

    cap_orig = cv2.VideoCapture(str(video_path_original))
    cap_corr = cv2.VideoCapture(str(video_path_corrected))

    total = int(cap_orig.get(cv2.CAP_PROP_FRAME_COUNT))
    indices = np.linspace(0, total - 1, n_samples, dtype=int)

    def _read(cap: cv2.VideoCapture, idx: int) -> np.ndarray | None:
        cap.set(cv2.CAP_PROP_POS_FRAMES, float(idx))
        ret, frame = cap.read()
        if not ret:
            return None
        return cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if frame.ndim == 3 else frame

    fig, axes = plt.subplots(3, n_samples, figsize=(5 * n_samples, 12))
    if n_samples == 1:
        axes = axes.reshape(3, 1)

    row_labels = ["Original", "Corrected", f"Difference ×{diff_amplification:.0f}"]

    for col, idx in enumerate(indices):
        orig = _read(cap_orig, idx)
        corr = _read(cap_corr, idx)

        diff: np.ndarray | None = None
        if orig is not None and corr is not None:
            diff_f = (orig.astype(np.float32) - corr.astype(np.float32)) * diff_amplification
            diff   = np.clip(diff_f + 128, 0, 255).astype(np.uint8)

        for row, (img, label) in enumerate(zip([orig, corr, diff], row_labels)):
            ax = axes[row, col]
            if img is not None:
                ax.imshow(img, cmap="gray", vmin=0, vmax=255)
            else:
                ax.text(0.5, 0.5, "N/A", ha="center", va="center", transform=ax.transAxes)
            if col == 0:
                ax.set_ylabel(label, fontsize=11)
            ax.set_title(f"Frame {idx}", fontsize=9)
            ax.axis("off")

    cap_orig.release()
    cap_corr.release()

    fig.suptitle(
        f"FPN correction — std={fpn_dict['pattern_std']:.3f}  "
        f"severity={fpn_dict['severity']}",
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
        print(
            "Usage: python PatternNoise.py <video.avi> [mask.png] [output_dir]\n"
            "  mask.png   : pre-computed mask PNG from step [1] (optional)\n"
            "  output_dir : directory for outputs (default: same folder as the video)"
        )
        sys.exit(1)

    video_path = Path(sys.argv[1])
    output_dir = Path(sys.argv[3]) if len(sys.argv) > 3 else Path.cwd()
    output_dir.mkdir(parents=True, exist_ok=True)

    # ── Mask ──────────────────────────────────────────────────────────────────
    if len(sys.argv) > 2 and Path(sys.argv[2]).exists():
        mask = load_mask(sys.argv[2])
        print(f"Mask loaded : {sys.argv[2]}")
    else:
        from mask_detection import detect_circular_mask
        print("Detecting circular mask …")
        result = detect_circular_mask(video_path)
        mask   = result["mask"]
        print(f"  coverage: {result['coverage']:.1%}")

    # ── FPN pattern: reuse cached .npy if present ──────────────────────────
    pattern_path = output_dir / f"fpn_pattern_{video_path.stem}.npy"

    if pattern_path.exists():
        print(f"Loading cached FPN pattern : {pattern_path}")
        cached_pattern = load_fpn_pattern(pattern_path)
        mask_bool      = mask > 0
        pattern_std    = float(cached_pattern[mask_bool].std()) if mask_bool.any() else 0.0
        severity       = _severity_label(
            pattern_std, FPN_STD_MILD, FPN_STD_MODERATE, FPN_STD_SEVERE
        )
        fpn_dict = {
            "pattern":             cached_pattern,
            "mean_frame":          cached_pattern,  # not available from cache
            "pattern_std":         pattern_std,
            "pattern_correlation": float("nan"),    # not available from cache
            "vessel_leakage":      False,
            "fpn_detected":        pattern_std >= FPN_STD_MILD,
            "severity":            severity,
            "n_clean":             -1,
            "gaussian_sigma":      FPN_GAUSSIAN_SIGMA,
        }
    else:
        print("Estimating FPN …")
        # No corrupted-frame info at this call site — pass empty list.
        # In a full pipeline, pass the list from step [2].
        fpn_dict = estimate_fpn(video_path, mask, corrupted_frames=[])
        save_fpn_pattern(fpn_dict, pattern_path)

    print(f"  pattern std : {fpn_dict['pattern_std']:.4f}")
    print(f"  severity    : {fpn_dict['severity']}")
    print(f"  FPN detected: {fpn_dict['fpn_detected']}")

    # ── Visualise pattern ─────────────────────────────────────────────────────
    viz_path = output_dir / f"fpn_pattern_{video_path.stem}.png"
    visualize_fpn(fpn_dict, mask, viz_path)

    # ── Correct only if FPN detected ─────────────────────────────────────────
    corrected_path = output_dir / f"fpn_corrected_{video_path.stem}.avi"

    if fpn_dict["fpn_detected"]:
        print("Correcting FPN …")
        correct_fpn(video_path, mask, fpn_dict, corrected_path)
    else:
        print("No FPN detected — skipping correction.")
        import shutil
        shutil.copy2(video_path, corrected_path)
        print(f"  Original copied to : {corrected_path}")

    # ── Comparison grid ───────────────────────────────────────────────────────
    cmp_path = output_dir / f"fpn_comparison_{video_path.stem}.png"
    compare_fpn_correction(video_path, corrected_path, fpn_dict, cmp_path)

    print("\nDone.")


if __name__ == "__main__":
    main()
