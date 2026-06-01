#!/usr/bin/env python3
"""
Step [6] — Banding detection and correction (periodic line artefacts).

Banding = horizontal or vertical periodic lines caused by the sensor readout
circuit.  In the 2D FFT of a banding image, these appear as bright peaks along
the central column (horizontal banding) or central row (vertical banding) of
the magnitude spectrum.

Strategy:
  DETECTION  : average FFT magnitude over n_sample_frames; extract the 1-D
               profiles along the central column and row; find statistically
               significant peaks outside the DC guard zone.
  CORRECTION : for each detected frequency, apply a narrow notch filter
               (2*notch_width+1 pixels wide) in the shifted FFT and reconstruct
               via IFFT.

CRITICAL: the notch must not touch DC-region frequencies (large structures /
main vessels) and must be at most 2–3 pixels wide in FFT space to preserve
vascular detail.

Usage (standalone):
  python Banding.py <video.avi> [output_dir]

  Expects in output_dir (or video parent dir):
    mask.png             — binary circular mask    (step [1])
    step2_corrected.json — corrupted-frame log     (step [2])
  Reads  : step5_corrected.avi  (output of step [5])
  Writes : step6_corrected.avi, banding_info.json,
           banding_fft.png, banding_comparison.png (if banding found)
"""
from __future__ import annotations

import json
import shutil
import sys
import warnings
from pathlib import Path

import cv2
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.signal import find_peaks

from mask_detection import load_mask


# ── Tuning parameters (edit here) ─────────────────────────────────────────────

N_SAMPLE_FRAMES      = 50    # frames averaged for FFT analysis
PEAK_SIGMA_THRESHOLD = 5.0   # peak detection: height > median + k·σ in profile
PEAK_MIN_HEIGHT_LOG  = 1.5   # minimum height above baseline in log1p units
DC_GUARD_PX          = 16    # min distance from DC centre to consider a peak
NOTCH_WIDTH          = 2     # half-width of notch in FFT bins (total 2w+1 = 5)
SSIM_MIN             = 0.95  # minimum SSIM between original and corrected
VASCULAR_CORR_MAX    = 0.20  # max correlation between diff map and vessel mean


# ── Internal helpers ──────────────────────────────────────────────────────────


def _make_video_writer(
    path: Path, fps: float, width: int, height: int
) -> cv2.VideoWriter:
    for fourcc_str in ("XVID", "MJPG", "mp4v"):
        fourcc = cv2.VideoWriter_fourcc(*fourcc_str)
        writer = cv2.VideoWriter(str(path), fourcc, fps, (width, height), isColor=True)
        if writer.isOpened():
            return writer
        writer.release()
    raise RuntimeError(f"Cannot open VideoWriter for {path}.")


def _read_gray(cap: cv2.VideoCapture, idx: int) -> np.ndarray | None:
    """Random-access read of one frame as uint8 grayscale."""
    cap.set(cv2.CAP_PROP_POS_FRAMES, float(idx))
    ret, frame = cap.read()
    if not ret:
        return None
    return cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if frame.ndim == 3 else frame


def _compute_ssim(
    img1: np.ndarray, img2: np.ndarray, mask: np.ndarray
) -> float:
    """Structural similarity between two uint8 images restricted to the mask."""
    a = img1[mask > 0].astype(np.float64)
    b = img2[mask > 0].astype(np.float64)
    mu_a, mu_b = a.mean(), b.mean()
    var_a  = ((a - mu_a) ** 2).mean()
    var_b  = ((b - mu_b) ** 2).mean()
    cov_ab = ((a - mu_a) * (b - mu_b)).mean()
    L = 255.0
    c1 = (0.01 * L) ** 2
    c2 = (0.03 * L) ** 2
    num = (2 * mu_a * mu_b + c1) * (2 * cov_ab + c2)
    den = (mu_a ** 2 + mu_b ** 2 + c1) * (var_a + var_b + c2)
    return float(num / den) if den > 0 else 1.0


def _pearson_corr(a: np.ndarray, b: np.ndarray) -> float:
    """Pearson correlation between two flat arrays."""
    a_c = a - a.mean()
    b_c = b - b.mean()
    denom = float(np.std(a_c) * np.std(b_c))
    return float(np.mean(a_c * b_c) / denom) if denom > 1e-9 else 0.0


def _find_peaks_in_profile(
    profile: np.ndarray,
    dc_center: int,
    sigma_threshold: float = PEAK_SIGMA_THRESHOLD,
    dc_guard: int = DC_GUARD_PX,
    min_height_log: float = PEAK_MIN_HEIGHT_LOG,
) -> list[int]:
    """
    Return shifted-FFT indices of significant peaks in a 1-D log-magnitude
    profile.  Only the positive-frequency half (index > dc_center) is returned;
    the symmetric counterpart is handled by _build_notch_mask.

    Args:
        profile:         1-D array (log1p FFT magnitude).
        dc_center:       Index of DC component in profile.
        sigma_threshold: Peak height threshold as multiples of noise σ.
        dc_guard:        Bins around DC that are excluded from detection.
        min_height_log:  Absolute minimum height above baseline in log1p units.

    Returns:
        List of positive-half peak indices.
    """
    N = len(profile)
    valid = np.ones(N, dtype=bool)
    valid[max(0, dc_center - dc_guard): dc_center + dc_guard + 1] = False

    baseline  = float(np.median(profile[valid]))
    noise_std = float(np.std(profile[valid]))

    height_thr = max(
        baseline + sigma_threshold * noise_std,
        baseline + min_height_log,
    )

    peaks, _ = find_peaks(profile, height=height_thr, distance=3)

    return [
        int(p) for p in peaks
        if p > dc_center and abs(p - dc_center) > dc_guard
    ]


def _build_notch_mask(
    H: int,
    W: int,
    horizontal_freqs: list[int],
    vertical_freqs: list[int],
    notch_width: int = NOTCH_WIDTH,
) -> np.ndarray:
    """
    Build a boolean mask (shifted-FFT coordinates) of bins to zero out.

    Horizontal banding peaks (row indices) produce a narrow rectangle around
    kx=0 (column W//2) at rows r and H−r.
    Vertical banding peaks (column indices) produce a narrow rectangle around
    ky=0 (row H//2) at columns c and W−c.

    The DC neighbourhood is never included regardless of input.

    Returns:
        np.ndarray (H, W) bool — True = zero this bin.
    """
    notch = np.zeros((H, W), dtype=bool)
    cy, cx = H // 2, W // 2
    w = notch_width

    def _sl(n: int, center: int) -> slice:
        return slice(max(0, center - w), min(n, center + w + 1))

    for r in horizontal_freqs:
        sym = (H - r) % H
        for row in {r, sym}:
            notch[_sl(H, row), _sl(W, cx)] = True

    for c in vertical_freqs:
        sym = (W - c) % W
        for col in {c, sym}:
            notch[_sl(H, cy), _sl(W, col)] = True

    # Safety: never notch the DC region
    notch[
        max(0, cy - DC_GUARD_PX): cy + DC_GUARD_PX + 1,
        max(0, cx - DC_GUARD_PX): cx + DC_GUARD_PX + 1,
    ] = False

    return notch


def _apply_notch(frame_gray: np.ndarray, notch_shifted: np.ndarray) -> np.ndarray:
    """
    Apply a pre-built notch mask (shifted-FFT coordinates) to one frame.

    Zeroes the identified FFT bins and reconstructs via IFFT.
    Conjugate symmetry is preserved because _build_notch_mask always zeros
    both a peak and its complex-conjugate counterpart.

    Args:
        frame_gray:    uint8 grayscale frame.
        notch_shifted: bool (H, W), True = zero this shifted FFT bin.

    Returns:
        Corrected uint8 grayscale frame.
    """
    F       = np.fft.fft2(frame_gray.astype(np.float32))
    F_shift = np.fft.fftshift(F)
    F_shift[notch_shifted] = 0.0
    corrected = np.real(np.fft.ifft2(np.fft.ifftshift(F_shift)))
    return np.clip(corrected, 0, 255).astype(np.uint8)


def _severity_label(n_peaks: int, max_amplitude: float) -> str:
    if n_peaks == 0:
        return "none"
    if n_peaks == 1 and max_amplitude < 2.0:
        return "mild"
    if n_peaks <= 3 and max_amplitude < 5.0:
        return "moderate"
    return "severe"


# ── Main API ──────────────────────────────────────────────────────────────────


def detect_banding(
    video_path: str,
    mask: np.ndarray,
    corrupted_frames: list,
    n_sample_frames: int = N_SAMPLE_FRAMES,
) -> dict:
    """
    Detect banding artefacts by averaging 2-D FFT over sampled frames.

    The FFT is computed on each sampled frame (masked), shifted so DC is at
    centre, and the magnitudes are averaged in log1p scale.  Peaks in the
    centre column (horizontal banding) and centre row (vertical banding) of
    the average spectrum are then detected using a threshold of
    PEAK_SIGMA_THRESHOLD × σ above the profile's noise floor.

    Args:
        video_path:       Path to input video (output of step [5]).
        mask:             Binary circular mask (from step [1]).
        corrupted_frames: Frame indices to skip (from step [2]).
        n_sample_frames:  Number of frames to average (default 50).

    Returns:
        dict with keys:
          'banding_detected' : bool
          'horizontal_freqs' : list[int] — shifted-FFT row indices (H banding)
          'vertical_freqs'   : list[int] — shifted-FFT col indices (V banding)
          'severity'         : 'none' | 'mild' | 'moderate' | 'severe'
          'mean_fft'         : np.ndarray (H, W) log1p-magnitude, fftshifted
          'peak_amplitudes'  : dict {label: float} — amplitude above baseline
    """
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise FileNotFoundError(f"Cannot open video: {video_path}")

    total  = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    H      = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    W      = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))

    skip   = set(corrupted_frames)
    avail  = [i for i in range(total) if i not in skip]
    n_samp = min(n_sample_frames, len(avail))
    step   = max(1, len(avail) // n_samp)
    sample = avail[::step][:n_samp]

    mask_f = (mask > 0).astype(np.float32)
    accum  = np.zeros((H, W), dtype=np.float64)
    n_ok   = 0

    for idx in sample:
        frame = _read_gray(cap, idx)
        if frame is None:
            continue
        F     = np.fft.fft2(frame.astype(np.float32) * mask_f)
        accum += np.abs(np.fft.fftshift(F))
        n_ok  += 1

    cap.release()

    if n_ok == 0:
        raise ValueError("No valid frames available for banding detection.")

    # log1p for better dynamic range; shape (H, W), DC at centre
    mean_fft = np.log1p(accum / n_ok)

    cy, cx = H // 2, W // 2

    h_peaks = _find_peaks_in_profile(mean_fft[:, cx], dc_center=cy)
    v_peaks = _find_peaks_in_profile(mean_fft[cy, :], dc_center=cx)

    baseline_h = float(np.median(mean_fft[:, cx]))
    baseline_v = float(np.median(mean_fft[cy, :]))

    peak_amplitudes: dict = {}
    for r in h_peaks:
        peak_amplitudes[f"H_row_{r}"] = float(mean_fft[r, cx] - baseline_h)
    for c in v_peaks:
        peak_amplitudes[f"V_col_{c}"] = float(mean_fft[cy, c] - baseline_v)

    n_peaks = len(h_peaks) + len(v_peaks)
    max_amp = max(peak_amplitudes.values(), default=0.0)

    return {
        "banding_detected": bool(h_peaks or v_peaks),
        "horizontal_freqs": h_peaks,
        "vertical_freqs":   v_peaks,
        "severity":         _severity_label(n_peaks, max_amp),
        "mean_fft":         mean_fft,
        "peak_amplitudes":  peak_amplitudes,
    }


def correct_banding(
    video_path: str,
    mask: np.ndarray,
    banding_dict: dict,
    output_path: str,
    notch_width: int = NOTCH_WIDTH,
) -> None:
    """
    Apply notch filters to the frequencies identified by detect_banding().

    If no banding was detected, copies the video without modification and logs
    clearly that no correction was applied.

    Validation after correction:
      - mean SSIM over sampled frames must exceed SSIM_MIN (default 0.95);
        raises ValueError if not (filter too aggressive).
      - Pearson correlation between the mean difference image and the vessel
        mean image must be below VASCULAR_CORR_MAX; issues a warning if not.

    Args:
        video_path:   Input video path (output of step [5]).
        mask:         Binary circular mask.
        banding_dict: Output of detect_banding().
        output_path:  Destination .avi path (same codec as input).
        notch_width:  Half-width of notch in FFT bins (≤ 3 recommended).

    Raises:
        ValueError: if mean SSIM drops below SSIM_MIN after correction.
    """
    output_path = str(output_path)

    if not banding_dict["banding_detected"]:
        print("  [Banding] No banding detected — copying video unchanged.")
        shutil.copy2(str(video_path), output_path)
        return

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise FileNotFoundError(f"Cannot open video: {video_path}")

    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    H     = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    W     = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    fps   = cap.get(cv2.CAP_PROP_FPS) or 30.0

    notch  = _build_notch_mask(
        H, W,
        banding_dict["horizontal_freqs"],
        banding_dict["vertical_freqs"],
        notch_width=notch_width,
    )
    writer = _make_video_writer(Path(output_path), fps, W, H)

    # Validation accumulators
    check_step    = max(1, total // 20)
    ssim_samples: list[float] = []
    vessel_accum  = np.zeros((H, W), dtype=np.float64)
    diff_accum    = np.zeros((H, W), dtype=np.float64)
    n_check       = 0

    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    for i in range(total):
        ret, frame_bgr = cap.read()
        if not ret:
            break

        frame_gray = (
            cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
            if frame_bgr.ndim == 3 else frame_bgr
        )
        corrected = _apply_notch(frame_gray, notch)
        writer.write(cv2.cvtColor(corrected, cv2.COLOR_GRAY2BGR))

        if i % check_step == 0:
            orig_f = frame_gray.astype(np.float64)
            corr_f = corrected.astype(np.float64)
            vessel_accum += orig_f
            diff_accum   += np.abs(corr_f - orig_f)
            ssim_samples.append(_compute_ssim(frame_gray, corrected, mask))
            n_check += 1

    writer.release()
    cap.release()

    if n_check == 0:
        warnings.warn("[Banding] Could not compute validation metrics.")
        return

    # ── SSIM check ────────────────────────────────────────────────────────────
    mean_ssim = float(np.mean(ssim_samples))
    if mean_ssim < SSIM_MIN:
        Path(output_path).unlink(missing_ok=True)
        raise ValueError(
            f"[Banding] Filter too aggressive: SSIM={mean_ssim:.4f} < {SSIM_MIN}. "
            "Reduce notch_width or loosen SSIM_MIN."
        )

    # ── Vascular contamination check ──────────────────────────────────────────
    vessel_mean = vessel_accum / n_check
    mean_diff   = diff_accum   / n_check
    mask_px     = mask > 0

    corr = _pearson_corr(
        mean_diff[mask_px],
        vessel_mean[mask_px],
    )
    if abs(corr) > VASCULAR_CORR_MAX:
        warnings.warn(
            f"[Banding] Difference image correlates with vessel map "
            f"(corr={corr:.3f} > threshold {VASCULAR_CORR_MAX}). "
            "The notch may have removed vascular content — inspect manually."
        )

    print(
        f"  [Banding] Correction applied — "
        f"SSIM={mean_ssim:.4f}  vessel_corr={corr:.3f}  "
        f"severity={banding_dict['severity']}  "
        f"H={banding_dict['horizontal_freqs']}  V={banding_dict['vertical_freqs']}"
    )


# ── Visualisation ─────────────────────────────────────────────────────────────


def visualize_fft(banding_dict: dict, output_path: str) -> None:
    """
    Save a 2-panel diagnostic PNG.

    Left  — log-scale FFT magnitude (fftshifted) with detected peaks marked in
            red (horizontal) / orange (vertical) and notch zones outlined in
            lime green.
    Right — centre-column and centre-row profiles with peak markers and DC
            guard region shaded.

    Args:
        banding_dict: Output of detect_banding().
        output_path:  Destination PNG path.
    """
    mean_fft = banding_dict["mean_fft"]
    h_freqs  = banding_dict["horizontal_freqs"]
    v_freqs  = banding_dict["vertical_freqs"]

    H, W    = mean_fft.shape
    cy, cx  = H // 2, W // 2
    w       = NOTCH_WIDTH

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # ── Left: 2D spectrum ─────────────────────────────────────────────────────
    ax = axes[0]
    im = ax.imshow(
        mean_fft, cmap="inferno", aspect="equal",
        vmin=np.percentile(mean_fft, 5),
        vmax=np.percentile(mean_fft, 99.5),
    )
    plt.colorbar(im, ax=ax, fraction=0.03, label="log1p(magnitude)")
    ax.set_title("Average FFT magnitude (log scale)", fontsize=11)
    ax.set_xlabel("kx  (bin)")
    ax.set_ylabel("ky  (bin)")
    ax.plot(cx, cy, "w+", markersize=10, markeredgewidth=1.5)   # DC marker

    for r in h_freqs:
        sym = (H - r) % H
        for row in {r, sym}:
            ax.axhline(row, color="red", linewidth=0.7, alpha=0.7)
            rect = plt.Rectangle(
                (cx - w - 0.5, row - w - 0.5), 2 * w + 1, 2 * w + 1,
                linewidth=1.2, edgecolor="lime", facecolor="none",
            )
            ax.add_patch(rect)

    for c in v_freqs:
        sym = (W - c) % W
        for col in {c, sym}:
            ax.axvline(col, color="orange", linewidth=0.7, alpha=0.7)
            rect = plt.Rectangle(
                (col - w - 0.5, cy - w - 0.5), 2 * w + 1, 2 * w + 1,
                linewidth=1.2, edgecolor="lime", facecolor="none",
            )
            ax.add_patch(rect)

    # ── Right: profiles ───────────────────────────────────────────────────────
    ax2 = axes[1]
    profile_h = mean_fft[:, cx]
    profile_v = mean_fft[cy, :]
    xs_h = np.arange(H) - cy
    xs_v = np.arange(W) - cx

    ax2.plot(xs_h, profile_h, color="royalblue",  lw=0.9, label="Centre-column (H banding)")
    ax2.plot(xs_v, profile_v, color="darkorange",  lw=0.9, label="Centre-row   (V banding)")

    for r in h_freqs:
        f = r - cy
        ax2.axvline( f, color="royalblue", ls="--", lw=0.8)
        ax2.axvline(-f, color="royalblue", ls="--", lw=0.8)
    for c in v_freqs:
        f = c - cx
        ax2.axvline( f, color="darkorange", ls="--", lw=0.8)
        ax2.axvline(-f, color="darkorange", ls="--", lw=0.8)

    ax2.axvspan(-DC_GUARD_PX, DC_GUARD_PX, alpha=0.12, color="gray", label="DC guard")
    ax2.set_title("Centre-axis FFT profiles", fontsize=11)
    ax2.set_xlabel("Frequency bin (relative to DC)")
    ax2.set_ylabel("log1p(magnitude)")
    ax2.legend(fontsize=8)
    ax2.grid(True, alpha=0.3)

    fig.suptitle(
        f"Banding analysis — detected={banding_dict['banding_detected']}  "
        f"severity={banding_dict['severity']}  "
        f"H_peaks={h_freqs}  V_peaks={v_freqs}",
        fontsize=11, fontweight="bold",
    )
    plt.tight_layout()
    plt.savefig(str(output_path), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  [Banding] FFT visualization saved: {output_path}")


def compare_banding_correction(
    video_original: str,
    video_corrected: str,
    banding_dict: dict,
    output_path: str,
    n_frames: int = 3,
) -> None:
    """
    Save a comparison grid (only when banding was detected).

    Layout: n_frames rows × 3 columns
      Col 0: Original frame
      Col 1: Corrected frame
      Col 2: Difference ×10  centred at 128 (gray=no change, bright=added,
             dark=removed) — should show only horizontal/vertical lines,
             NOT vessel structures.

    Args:
        video_original:  Input to step [6] (step5_corrected.avi).
        video_corrected: Output of step [6] (step6_corrected.avi).
        banding_dict:    Output of detect_banding().
        output_path:     Destination PNG path.
        n_frames:        Number of sample frames (default 3).
    """
    if not banding_dict["banding_detected"]:
        print("  [Banding] No banding — skipping comparison visualization.")
        return

    cap_o = cv2.VideoCapture(str(video_original))
    cap_c = cv2.VideoCapture(str(video_corrected))

    if not cap_o.isOpened() or not cap_c.isOpened():
        warnings.warn("[Banding] Cannot open one of the videos for comparison.")
        cap_o.release()
        cap_c.release()
        return

    total   = int(cap_o.get(cv2.CAP_PROP_FRAME_COUNT))
    indices = np.linspace(0, total - 1, n_frames, dtype=int)

    fig, axes = plt.subplots(n_frames, 3, figsize=(15, 5 * n_frames))
    if n_frames == 1:
        axes = axes[np.newaxis, :]

    col_titles = ["Original", "Corrected", "Difference ×10  (128=no change)"]
    for j, t in enumerate(col_titles):
        axes[0, j].set_title(t, fontsize=11, fontweight="bold")

    for row, fidx in enumerate(indices):
        orig = _read_gray(cap_o, int(fidx))
        corr = _read_gray(cap_c, int(fidx))
        if orig is None or corr is None:
            continue

        diff_display = np.clip(
            128 + 10 * (corr.astype(np.int16) - orig.astype(np.int16)),
            0, 255,
        ).astype(np.uint8)

        for j, img in enumerate((orig, corr, diff_display)):
            axes[row, j].imshow(img, cmap="gray", vmin=0, vmax=255)
            axes[row, j].set_ylabel(f"frame {fidx}", fontsize=8)
            axes[row, j].axis("off")

    cap_o.release()
    cap_c.release()

    fig.suptitle(
        f"Banding correction — severity={banding_dict['severity']}  "
        f"H={banding_dict['horizontal_freqs']}  V={banding_dict['vertical_freqs']}\n"
        "Difference column must show only periodic lines — no vessel structures.",
        fontsize=10, fontweight="bold",
    )
    plt.tight_layout()
    plt.savefig(str(output_path), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  [Banding] Comparison saved: {output_path}")


# ── CLI ───────────────────────────────────────────────────────────────────────


def main() -> None:
    if len(sys.argv) < 2:
        print(
            "Usage:\n"
            "  python Banding.py <video.avi> [output_dir]\n"
            "\n"
            "  video.avi  : step5_corrected.avi (output of step [5])\n"
            "  output_dir : defaults to <video_stem>_banding/\n"
            "\n"
            "  Expects mask.png and (optionally) step2_corrected.json in\n"
            "  output_dir or the video's parent directory.\n"
        )
        sys.exit(1)

    video_path = Path(sys.argv[1])
    if not video_path.exists():
        print(f"Error: file not found — {video_path}")
        sys.exit(1)

    output_dir = (
        Path(sys.argv[2]) if len(sys.argv) > 2
        else video_path.parent / f"{video_path.stem}_banding"
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    # ── Load mask ──────────────────────────────────────────────────────────────
    mask_candidates = [output_dir / "mask.png", video_path.parent / "mask.png"]
    mask_path = next((p for p in mask_candidates if p.exists()), None)
    if mask_path is None:
        print("Error: mask.png not found. Run step [1] first.")
        sys.exit(1)
    mask = load_mask(mask_path)
    print(f"Mask loaded: {mask_path}")

    # ── Load corrupted frame list ──────────────────────────────────────────────
    corrupted_frames: list[int] = []
    for jp in [output_dir / "step2_corrected.json", video_path.parent / "step2_corrected.json"]:
        if jp.exists():
            with open(jp) as fh:
                corrupted_frames = [int(k) for k in json.load(fh).keys()]
            print(f"Corrupted frames loaded: {len(corrupted_frames)} from {jp.name}")
            break

    # ── Detection ─────────────────────────────────────────────────────────────
    print(f"\nDetecting banding in {video_path.name} …")
    banding_dict = detect_banding(video_path, mask, corrupted_frames)

    print(
        f"  banding_detected = {banding_dict['banding_detected']}\n"
        f"  severity         = {banding_dict['severity']}\n"
        f"  horizontal_freqs = {banding_dict['horizontal_freqs']}\n"
        f"  vertical_freqs   = {banding_dict['vertical_freqs']}\n"
        f"  peak_amplitudes  = {banding_dict['peak_amplitudes']}"
    )

    # ── Save JSON (without the large mean_fft array) ──────────────────────────
    info_path = output_dir / "banding_info.json"
    saveable  = {k: v for k, v in banding_dict.items() if k != "mean_fft"}
    with open(info_path, "w") as fh:
        json.dump(saveable, fh, indent=2)
    print(f"  Info saved: {info_path}")

    # ── FFT visualisation ──────────────────────────────────────────────────────
    visualize_fft(banding_dict, str(output_dir / "banding_fft.png"))

    # ── Correction ────────────────────────────────────────────────────────────
    output_video = output_dir / "step6_corrected.avi"
    print("\nCorrecting banding …")
    correct_banding(str(video_path), mask, banding_dict, str(output_video))

    # ── Comparison ────────────────────────────────────────────────────────────
    if banding_dict["banding_detected"]:
        compare_banding_correction(
            str(video_path), str(output_video),
            banding_dict, str(output_dir / "banding_comparison.png"),
        )

    print(f"\nDone.  Output video : {output_video}")


if __name__ == "__main__":
    main()
