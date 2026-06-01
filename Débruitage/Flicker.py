#!/usr/bin/env python3
"""
Step [5] — Flicker and brightness drift correction.

Two distinct artefacts are corrected separately:
  • DRIFT  : slow, monotone luminosity trend (low-frequency, estimated by
             a degree-2 polynomial fit on the mean-luminosity curve).
  • FLICKER: rapid, irregular frame-to-frame luminosity oscillations
             (high-frequency residual after drift removal, estimated by
             a Butterworth low-pass filter on the residual curve).

A final contrast normalisation recalibrates the output to a fixed
reference brightness, compensating for the contrast loss introduced by
the warp-and-crop in step [4].

CRITICAL — cardiac pulsation preservation
The vascular pulsation produces legitimate luminosity oscillations at
~1–2 Hz (0.033–0.067 × Nyquist at 30 fps).  The Butterworth cut-off is
set at 0.025 × Nyquist so these oscillations are left untouched.

Usage (standalone):
  python Flicker.py <video.avi> [output_dir]

  video.avi  : stabilised video (output of step [4])
  output_dir : defaults to <video_stem>_flicker/

  Expects in output_dir (or video parent dir):
    mask.png              — binary circular mask  (step [1])
    step2_corrected.json  — corrupted frame log   (step [2])
"""
from __future__ import annotations

import json
import sys
import warnings
from pathlib import Path

import cv2
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt

from mask_detection import load_mask


# ── Module-level tuning parameters (edit here) ────────────────────────────────

BUTTERWORTH_ORDER  = 4      # Butterworth filter order
BUTTERWORTH_CUTOFF = 0.025  # cut-off as fraction of Nyquist (preserves ≥1 Hz pulsation)
POLY_ORDER         = 2      # polynomial order for drift estimation

DRIFT_R2_THRESHOLD   = 0.5   # minimum R² to declare drift detected
FLICKER_STD_MIN      = 0.3   # minimum normalised std to declare flicker detected

N_REFERENCE_FRAMES   = 60    # healthy frames used for contrast reference

PULSATION_HZ_LOW  = 0.5   # valid cardiac pulsation band (Hz)
PULSATION_HZ_HIGH = 3.0


# ── Internal helpers ──────────────────────────────────────────────────────────


def _make_video_writer(
    path: Path,
    fps: float,
    width: int,
    height: int,
) -> cv2.VideoWriter:
    for fourcc_str in ("XVID", "MJPG", "mp4v"):
        fourcc = cv2.VideoWriter_fourcc(*fourcc_str)
        writer = cv2.VideoWriter(str(path), fourcc, fps, (width, height), isColor=True)
        if writer.isOpened():
            return writer
        writer.release()
    raise RuntimeError(
        f"Could not open VideoWriter for {path}. "
        "Tried XVID, MJPG, mp4v — none worked."
    )


def _progress(i: int, total: int, label: str = "") -> None:
    bar_len = 30
    filled  = int(bar_len * (i + 1) / total)
    bar     = "█" * filled + "░" * (bar_len - filled)
    pct     = (i + 1) / total * 100
    print(f"\r  {label} [{bar}] {i+1}/{total} {pct:.0f}%", end="", flush=True)
    if i + 1 == total:
        print()


def _r2_score(y: np.ndarray, y_fit: np.ndarray) -> float:
    ss_res = float(np.sum((y - y_fit) ** 2))
    ss_tot = float(np.sum((y - y.mean()) ** 2))
    if ss_tot == 0.0:
        return 1.0
    return 1.0 - ss_res / ss_tot


def _lowpass(signal: np.ndarray, cutoff: float, order: int) -> np.ndarray:
    """Zero-phase Butterworth low-pass filter. cutoff is fraction of Nyquist."""
    if len(signal) < 2 * order + 1:
        return signal.copy()
    # filtfilt needs padlen < signal length; use default or clamp
    b, a = butter(order, cutoff, btype="low")
    padlen = min(3 * max(len(a), len(b)), len(signal) - 1)
    return filtfilt(b, a, signal, padlen=padlen)


def _mean_in_mask(frame: np.ndarray, mask_bool: np.ndarray) -> float:
    pixels = frame.astype(np.float64)[mask_bool]
    return float(pixels.mean()) if pixels.size else 0.0


def _percentile99_in_mask(frame: np.ndarray, mask_bool: np.ndarray) -> float:
    pixels = frame.astype(np.float64)[mask_bool]
    return float(np.percentile(pixels, 99)) if pixels.size else 1.0


# ── Core analysis ─────────────────────────────────────────────────────────────


def analyze_luminosity(
    video_path: str | Path,
    mask: np.ndarray,
    corrupted_frames: list[int],
    butterworth_cutoff: float = BUTTERWORTH_CUTOFF,
    poly_order: int = POLY_ORDER,
    fps: float | None = None,
) -> dict:
    """
    Analyse the luminosity curve of a video and detect drift and flicker.

    Args:
        video_path:        Path to the stabilised video (step [4] output).
        mask:              Binary mask uint8 from step [1].
        corrupted_frames:  Frame indices to exclude from analysis (step [2]).
        butterworth_cutoff: Low-pass cut-off as fraction of Nyquist.
        poly_order:        Polynomial order for drift estimation.
        fps:               Frame rate override (read from video if None).

    Returns:
        dict with keys:
          'luminosity_curve'   np.ndarray  — mean luminosity per frame
          'drift_detected'     bool
          'drift_r2'           float       — R² of polynomial fit
          'drift_amplitude'    float       — peak-to-peak drift (levels)
          'flicker_std'        float       — std of normalised flicker residual
          'flicker_detected'   bool
          'pulsation_freq_hz'  float       — dominant FFT frequency in valid band
          'fps'                float
          'n_frames'           int
          'drift_poly'         np.ndarray  — polynomial values per frame
          'flicker_envelope'   np.ndarray  — low-pass envelope after drift removal
    """
    video_path = Path(video_path)
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise FileNotFoundError(f"Cannot open video: {video_path}")

    n_total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if fps is None:
        fps = float(cap.get(cv2.CAP_PROP_FPS)) or 30.0

    mask_bool = mask > 0
    skip = set(corrupted_frames)
    luminosity_curve = np.zeros(n_total, dtype=np.float64)

    print(f"  Analysing luminosity ({n_total} frames) …")
    for i in range(n_total):
        ret, frame = cap.read()
        if not ret:
            break
        if frame.ndim == 3:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        if i in skip:
            luminosity_curve[i] = np.nan
        else:
            luminosity_curve[i] = _mean_in_mask(frame, mask_bool)
        _progress(i, n_total, "luminosity")
    cap.release()

    # Interpolate NaN (corrupted frames) for filtering
    x_all = np.arange(n_total, dtype=np.float64)
    valid_mask = ~np.isnan(luminosity_curve)
    if valid_mask.sum() < 2:
        warnings.warn("Not enough valid frames for luminosity analysis.")
        lum_interp = luminosity_curve.copy()
    else:
        lum_interp = np.interp(x_all, x_all[valid_mask], luminosity_curve[valid_mask])
    # Restore NaN positions for downstream use, but work on interp for fitting
    luminosity_interp = lum_interp.copy()

    # ── Drift estimation (polynomial fit) ─────────────────────────────────────
    coeffs    = np.polyfit(x_all, luminosity_interp, poly_order)
    drift_poly = np.polyval(coeffs, x_all)
    drift_r2   = _r2_score(luminosity_interp, drift_poly)
    drift_amplitude = float(drift_poly.max() - drift_poly.min())
    drift_detected  = drift_r2 > DRIFT_R2_THRESHOLD and drift_amplitude > 1.0

    # ── Flicker estimation (residual after drift removal) ─────────────────────
    if drift_detected:
        drift_mean   = float(drift_poly.mean())
        drift_norm   = drift_poly / (drift_mean + 1e-9)
        residual_lum = luminosity_interp / (drift_norm + 1e-9)
    else:
        residual_lum = luminosity_interp.copy()

    flicker_envelope = _lowpass(residual_lum, butterworth_cutoff, BUTTERWORTH_ORDER)
    flicker_residual = residual_lum - flicker_envelope
    flicker_std      = float(np.std(flicker_residual[valid_mask]) /
                             (np.mean(residual_lum[valid_mask]) + 1e-9))
    flicker_detected = flicker_std > FLICKER_STD_MIN

    # ── Pulsation frequency (FFT on residual after flicker envelope removal) ──
    pulsation_freq_hz = _dominant_pulsation_freq(residual_lum, fps,
                                                  PULSATION_HZ_LOW,
                                                  PULSATION_HZ_HIGH)

    print(f"  drift: r²={drift_r2:.3f}  amp={drift_amplitude:.2f}  detected={drift_detected}")
    print(f"  flicker: std={flicker_std:.4f}  detected={flicker_detected}")
    print(f"  pulsation: {pulsation_freq_hz:.3f} Hz")

    return {
        "luminosity_curve":  luminosity_curve,       # NaN at corrupted frames
        "luminosity_interp": luminosity_interp,       # fully interpolated
        "drift_detected":    drift_detected,
        "drift_r2":          drift_r2,
        "drift_amplitude":   drift_amplitude,
        "drift_poly":        drift_poly,
        "flicker_std":       flicker_std,
        "flicker_detected":  flicker_detected,
        "flicker_envelope":  flicker_envelope,
        "pulsation_freq_hz": pulsation_freq_hz,
        "fps":               fps,
        "n_frames":          n_total,
        "residual_lum":      residual_lum,
    }


def _dominant_pulsation_freq(
    signal: np.ndarray,
    fps: float,
    hz_low: float,
    hz_high: float,
) -> float:
    """Return the dominant FFT frequency in [hz_low, hz_high], or 0.0 if absent."""
    n = len(signal)
    if n < 8:
        return 0.0
    spectrum = np.abs(np.fft.rfft(signal - signal.mean()))
    freqs    = np.fft.rfftfreq(n, d=1.0 / fps)
    band     = (freqs >= hz_low) & (freqs <= hz_high)
    if not band.any():
        return 0.0
    return float(freqs[band][np.argmax(spectrum[band])])


# ── Core correction ───────────────────────────────────────────────────────────


def correct_drift_and_flicker(
    video_path: str | Path,
    mask: np.ndarray,
    luminosity_dict: dict,
    reference_frame: np.ndarray,
    output_path: str | Path,
    butterworth_cutoff: float = BUTTERWORTH_CUTOFF,
    poly_order: int = POLY_ORDER,
) -> None:
    """
    Apply drift + flicker correction and final contrast normalisation,
    then write the result to output_path.

    Correction factors are applied multiplicatively per frame:
      corrected = original / drift_factor / flicker_factor * contrast_scale

    Args:
        video_path:        Input stabilised video.
        mask:              Binary mask from step [1].
        luminosity_dict:   Output of analyze_luminosity().
        reference_frame:   Float32 reference image for contrast calibration.
        output_path:       Destination .avi path.
        butterworth_cutoff: Low-pass cut-off (fraction of Nyquist).
        poly_order:        Polynomial order for drift.
    """
    video_path  = Path(video_path)
    output_path = Path(output_path)

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise FileNotFoundError(f"Cannot open video: {video_path}")

    n_total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps     = float(cap.get(cv2.CAP_PROP_FPS)) or 30.0
    width   = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height  = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    mask_bool        = mask > 0
    drift_detected   = luminosity_dict["drift_detected"]
    flicker_detected = luminosity_dict["flicker_detected"]
    drift_poly       = luminosity_dict["drift_poly"]       # shape (n_total,)
    flicker_envelope = luminosity_dict["flicker_envelope"] # shape (n_total,)
    residual_lum     = luminosity_dict["residual_lum"]

    # ── Build per-frame correction factors ────────────────────────────────────
    drift_mean = float(drift_poly.mean())

    if drift_detected:
        drift_factors = drift_poly / (drift_mean + 1e-9)
    else:
        drift_factors = np.ones(n_total, dtype=np.float64)

    if flicker_detected:
        envelope_mean = float(flicker_envelope.mean())
        flicker_factors = flicker_envelope / (envelope_mean + 1e-9)
    else:
        flicker_factors = np.ones(n_total, dtype=np.float64)

    # ── Contrast reference from reference_frame ───────────────────────────────
    ref_p99 = _percentile99_in_mask(reference_frame, mask_bool)
    if ref_p99 <= 0:
        ref_p99 = 1.0

    # ── Compute final luminosity after drift+flicker for contrast scaling ──
    # We compute a representative mean for contrast scaling post-correction.
    # The target luminosity comes from the corrected residual divided by
    # flicker factors, normalised so the mean equals ref_p99.

    # ── Write corrected video ─────────────────────────────────────────────────
    writer = _make_video_writer(output_path, fps, width, height)

    # Second pass: measure p99 of the corrected frames to set global scale
    # (done in a single forward pass; we accumulate per-frame p99 estimates)
    corrected_p99_samples: list[float] = []
    corrected_frames_buf: list[np.ndarray] = []

    print(f"  Correcting {n_total} frames …")
    for i in range(n_total):
        ret, frame = cap.read()
        if not ret:
            break
        if frame.ndim == 3:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY).astype(np.float32)
        else:
            gray = frame.astype(np.float32)

        corrected = gray / (drift_factors[i] * flicker_factors[i] + 1e-9)
        corrected_p99_samples.append(_percentile99_in_mask(corrected, mask_bool))
        corrected_frames_buf.append(corrected)
        _progress(i, n_total, "pass-1 ")

    cap.release()

    # Global contrast scale: median of per-frame p99 → target ref_p99
    median_p99 = float(np.median(corrected_p99_samples)) if corrected_p99_samples else 1.0
    if median_p99 <= 0:
        median_p99 = 1.0
    contrast_scale = ref_p99 / median_p99

    print(f"  Contrast scale: {contrast_scale:.4f}  "
          f"(target p99={ref_p99:.1f}, corrected median p99={median_p99:.1f})")

    # Apply contrast scale and write
    corrected_luminosity: list[float] = []
    print(f"  Writing output …")
    for i, corrected in enumerate(corrected_frames_buf):
        scaled = corrected * contrast_scale
        scaled = np.clip(scaled, 0, 255).astype(np.uint8)
        corrected_luminosity.append(_mean_in_mask(scaled, mask_bool))
        bgr = cv2.cvtColor(scaled, cv2.COLOR_GRAY2BGR)
        writer.write(bgr)
        _progress(i, len(corrected_frames_buf), "pass-2 ")

    writer.release()

    # ── Validation ────────────────────────────────────────────────────────────
    lum_orig = luminosity_dict["luminosity_interp"]
    _validate_correction(
        lum_orig,
        np.array(corrected_luminosity),
        luminosity_dict["fps"],
        luminosity_dict["pulsation_freq_hz"],
    )

    # ── Save correction curves for traceability ───────────────────────────────
    npz_path = output_path.with_suffix(".npz")
    np.savez_compressed(
        npz_path,
        drift_factors=drift_factors,
        flicker_factors=flicker_factors,
        corrected_luminosity=np.array(corrected_luminosity),
        original_luminosity=lum_orig,
        drift_poly=drift_poly,
        flicker_envelope=flicker_envelope,
        contrast_scale=np.array([contrast_scale]),
    )
    print(f"  Correction curves saved: {npz_path.name}")
    print(f"  Output: {output_path}")


def _validate_correction(
    original: np.ndarray,
    corrected: np.ndarray,
    fps: float,
    expected_pulsation_hz: float,
) -> None:
    # Pulsation still present?
    puls_after = _dominant_pulsation_freq(corrected, fps,
                                          PULSATION_HZ_LOW, PULSATION_HZ_HIGH)
    if expected_pulsation_hz > 0 and puls_after == 0.0:
        warnings.warn(
            "CRITICAL: cardiac pulsation disappeared after correction "
            f"(was {expected_pulsation_hz:.3f} Hz). "
            "Consider raising butterworth_cutoff.",
            stacklevel=3,
        )
    else:
        print(f"  Pulsation check: {expected_pulsation_hz:.3f} Hz → {puls_after:.3f} Hz  ✓")

    # Std reduced?
    std_orig  = float(np.std(original))
    std_corr  = float(np.std(corrected))
    if std_corr >= std_orig:
        print(f"  WARNING: luminosity std not reduced "
              f"(before={std_orig:.3f}, after={std_corr:.3f}). "
              "Correction may not have improved the video.")
    else:
        print(f"  Luminosity std: {std_orig:.3f} → {std_corr:.3f}  ✓")

    # Mean close to reference?
    mean_ratio = float(np.mean(corrected)) / (float(np.mean(original)) + 1e-9)
    if abs(mean_ratio - 1.0) > 0.05:
        print(f"  WARNING: final mean luminosity differs from original by "
              f"{abs(mean_ratio - 1.0):.1%} (>{5:.0f}%)")
    else:
        print(f"  Mean luminosity ratio: {mean_ratio:.3f}  ✓")


# ── Visualisations ────────────────────────────────────────────────────────────


def visualize_luminosity_analysis(
    luminosity_dict: dict,
    output_path: str | Path,
) -> None:
    """
    4-panel figure:
      [0] Raw luminosity curve + polynomial drift estimate
      [1] Residual after drift removal + Butterworth envelope
      [2] Corrected luminosity (after drift + flicker correction)
      [3] FFT of raw luminosity — cardiac pulsation peak must be visible
    """
    output_path = Path(output_path)
    fps   = luminosity_dict["fps"]
    n     = luminosity_dict["n_frames"]
    t     = np.arange(n) / fps

    lum_raw   = luminosity_dict["luminosity_interp"]
    drift     = luminosity_dict["drift_poly"]
    residual  = luminosity_dict["residual_lum"]
    envelope  = luminosity_dict["flicker_envelope"]
    flicker   = residual - envelope

    # Corrected luminosity (approximate — not the actual video output)
    drift_mean = float(drift.mean())
    drift_factors   = drift / (drift_mean + 1e-9) if luminosity_dict["drift_detected"] else np.ones(n)
    env_mean        = float(envelope.mean())
    flicker_factors = envelope / (env_mean + 1e-9) if luminosity_dict["flicker_detected"] else np.ones(n)
    corrected_approx = lum_raw / (drift_factors * flicker_factors + 1e-9)

    fig, axes = plt.subplots(4, 1, figsize=(14, 14), constrained_layout=True)
    fig.suptitle("Luminosity analysis — drift & flicker", fontsize=13)

    # Panel 0 — raw + drift
    ax = axes[0]
    ax.plot(t, lum_raw, lw=0.8, color="steelblue", label="raw luminosity")
    if luminosity_dict["drift_detected"]:
        ax.plot(t, drift, lw=1.8, color="crimson",
                label=f"drift poly (R²={luminosity_dict['drift_r2']:.3f})")
    ax.set_ylabel("Mean luminosity (levels)")
    ax.set_title("Raw luminosity + drift estimate")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # Panel 1 — residual after drift + flicker envelope
    ax = axes[1]
    ax.plot(t, residual, lw=0.8, color="steelblue", label="residual after drift")
    ax.plot(t, envelope, lw=1.8, color="orange",
            label=f"low-pass envelope (cutoff={BUTTERWORTH_CUTOFF}×Nyq)")
    ax.plot(t, flicker, lw=0.6, color="gray", alpha=0.7,
            label=f"flicker (std={luminosity_dict['flicker_std']:.4f})")
    ax.set_ylabel("Luminosity (levels)")
    ax.set_title("Residual after drift removal + Butterworth flicker envelope")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # Panel 2 — corrected luminosity
    ax = axes[2]
    ax.plot(t, lum_raw, lw=0.6, color="steelblue", alpha=0.5, label="original")
    ax.plot(t, corrected_approx, lw=0.9, color="green", label="corrected (approx)")
    ax.set_ylabel("Luminosity (levels)")
    ax.set_xlabel("Time (s)")
    ax.set_title("Luminosity before vs after correction (approximate)")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # Panel 3 — FFT
    ax = axes[3]
    freqs   = np.fft.rfftfreq(n, d=1.0 / fps)
    spectrum = np.abs(np.fft.rfft(lum_raw - lum_raw.mean()))
    ax.plot(freqs, spectrum, lw=0.8, color="purple")
    ax.axvspan(PULSATION_HZ_LOW, PULSATION_HZ_HIGH, color="green", alpha=0.15,
               label=f"pulsation band [{PULSATION_HZ_LOW}–{PULSATION_HZ_HIGH} Hz]")
    pf = luminosity_dict["pulsation_freq_hz"]
    if pf > 0:
        ax.axvline(pf, color="red", lw=1.2, linestyle="--",
                   label=f"dominant pulsation {pf:.3f} Hz")
    ax.set_xlim(0, min(10.0, freqs[-1]))
    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel("|FFT|")
    ax.set_title("FFT of raw luminosity curve")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    fig.savefig(str(output_path), dpi=110)
    plt.close(fig)
    print(f"  Luminosity analysis plot: {output_path.name}")


def compare_correction(
    video_original: str | Path,
    video_corrected: str | Path,
    luminosity_dict: dict,
    output_path: str | Path,
    n_samples: int = 5,
) -> None:
    """
    Comparison figure:
      Grid of n_samples frame triplets (original | corrected | diff×5)
      + overlay plot of luminosity curves before / after correction.
    """
    video_original  = Path(video_original)
    video_corrected = Path(video_corrected)
    output_path     = Path(output_path)

    cap_o = cv2.VideoCapture(str(video_original))
    cap_c = cv2.VideoCapture(str(video_corrected))
    n_total = int(cap_o.get(cv2.CAP_PROP_FRAME_COUNT))
    fps     = luminosity_dict["fps"]

    sample_indices = np.linspace(0, n_total - 1, n_samples, dtype=int).tolist()

    rows = n_samples
    fig, axes = plt.subplots(rows + 1, 3, figsize=(14, 3 * rows + 5),
                             constrained_layout=True)
    fig.suptitle("Drift/flicker correction — original vs corrected", fontsize=12)

    for row, idx in enumerate(sample_indices):
        # Read frames
        cap_o.set(cv2.CAP_PROP_POS_FRAMES, float(idx))
        cap_c.set(cv2.CAP_PROP_POS_FRAMES, float(idx))
        _, fo = cap_o.read()
        _, fc = cap_c.read()

        if fo is not None and fo.ndim == 3:
            fo = cv2.cvtColor(fo, cv2.COLOR_BGR2GRAY)
        if fc is not None and fc.ndim == 3:
            fc = cv2.cvtColor(fc, cv2.COLOR_BGR2GRAY)

        lum_o = float(fo.mean()) if fo is not None else 0.0
        lum_c = float(fc.mean()) if fc is not None else 0.0
        t_s   = idx / fps

        ax = axes[row, 0]
        if fo is not None:
            ax.imshow(fo, cmap="gray", vmin=0, vmax=255)
        ax.set_title(f"Original  t={t_s:.1f}s  μ={lum_o:.1f}", fontsize=8)
        ax.axis("off")

        ax = axes[row, 1]
        if fc is not None:
            ax.imshow(fc, cmap="gray", vmin=0, vmax=255)
        ax.set_title(f"Corrected  μ={lum_c:.1f}", fontsize=8)
        ax.axis("off")

        ax = axes[row, 2]
        if fo is not None and fc is not None:
            diff = np.clip((fc.astype(np.int16) - fo.astype(np.int16)) * 5 + 128,
                           0, 255).astype(np.uint8)
            ax.imshow(diff, cmap="RdBu_r", vmin=0, vmax=255)
        ax.set_title("Diff × 5", fontsize=8)
        ax.axis("off")

    cap_o.release()
    cap_c.release()

    # Bottom row: luminosity curves overlay
    ax_lum = axes[-1, :]
    for a in ax_lum:
        a.set_visible(False)
    # Replace the 3 bottom cells with a single merged axes
    gs = axes[-1, 0].get_gridspec()
    ax_merge = fig.add_subplot(gs[-1, :])
    t_axis = np.arange(luminosity_dict["n_frames"]) / fps
    ax_merge.plot(t_axis, luminosity_dict["luminosity_interp"],
                  lw=0.8, color="steelblue", alpha=0.7, label="original")
    # Load corrected curve from .npz if available
    npz_path = video_corrected.with_suffix(".npz")
    if npz_path.exists():
        data = np.load(npz_path)
        if "corrected_luminosity" in data:
            ax_merge.plot(t_axis[:len(data["corrected_luminosity"])],
                          data["corrected_luminosity"],
                          lw=0.9, color="green", label="corrected")
    ax_merge.set_xlabel("Time (s)")
    ax_merge.set_ylabel("Mean luminosity")
    ax_merge.set_title("Luminosity curves — before vs after correction")
    ax_merge.legend(fontsize=8)
    ax_merge.grid(True, alpha=0.3)

    fig.savefig(str(output_path), dpi=110)
    plt.close(fig)
    print(f"  Comparison figure: {output_path.name}")


# ── Standalone entry-point ────────────────────────────────────────────────────


def _run_standalone(
    video_path: Path,
    output_dir: Path,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load mask
    mask_candidates = [
        output_dir / "mask.png",
        video_path.parent / "mask.png",
    ]
    mask_path = next((p for p in mask_candidates if p.exists()), None)
    if mask_path is None:
        print("Error: mask.png not found. Run step [1] first.")
        sys.exit(1)
    mask = load_mask(mask_path)

    # Load corrupted frames
    corrupted_frames: list[int] = []
    log_candidates = [
        output_dir / "step2_corrected.json",
        video_path.parent / "step2_corrected.json",
    ]
    for lp in log_candidates:
        if lp.exists():
            with open(lp) as fh:
                corrupted_frames = [int(k) for k in json.load(fh).keys()]
            break

    # Reference frame (temporal median of first N_REFERENCE_FRAMES healthy frames)
    from Rigidisation import compute_reference_frame
    reference_frame = compute_reference_frame(
        video_path, mask, corrupted_frames, n_frames=N_REFERENCE_FRAMES
    )

    # Analysis
    lum_dict = analyze_luminosity(video_path, mask, corrupted_frames)

    # Visualisation
    viz_path = output_dir / "step5_luminosity_analysis.png"
    visualize_luminosity_analysis(lum_dict, viz_path)

    # Correction
    out_video = output_dir / "step5_corrected.avi"
    if out_video.exists():
        print(f"  Reusing cached step-5 video: {out_video.name}")
    else:
        correct_drift_and_flicker(
            video_path, mask, lum_dict, reference_frame, out_video
        )

    # Comparison
    cmp_path = output_dir / "step5_comparison.png"
    if not cmp_path.exists():
        compare_correction(video_path, out_video, lum_dict, cmp_path)

    print(f"\n  Done. Output: {out_video}")


def main() -> None:
    if len(sys.argv) < 2:
        print(
            "Usage:\n"
            "  python Flicker.py <video.avi> [output_dir]\n"
            "\n"
            "video.avi  : stabilised video (step [4] output)\n"
            "output_dir : defaults to <video_stem>_flicker/\n"
        )
        sys.exit(1)

    video_path  = Path(sys.argv[1])
    output_dir  = Path(sys.argv[2]) if len(sys.argv) > 2 else \
                  video_path.parent / (video_path.stem + "_flicker")

    if not video_path.exists():
        print(f"Error: file not found — {video_path}")
        sys.exit(1)

    _run_standalone(video_path, output_dir)


if __name__ == "__main__":
    main()
