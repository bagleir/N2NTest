#!/usr/bin/env python3
"""
Full preprocessing pipeline for ocular vascular imaging videos.

Steps applied in order for each video:
  [1] Circular mask detection
  [2] Bad frame detection and replacement (iterative until clean)
  [3] Fixed Pattern Noise estimation and correction

Output layout per video  (inside output_dir/<video_stem>/):
  mask.png                        — binary circular mask
  step2_corrected.avi             — video after bad-frame replacement
  step2_corrected.json            — replacement log (frame indices)
  fpn_pattern.npy                 — cached FPN pattern (reused on next run)
  final.avi                       — FPN-corrected or step2 copy (input to step [4])
  motion.npz                      — cached rigid transforms (reused on next run)
  motion.json                     — per-frame transform log (human-readable)
  motion_plot.png                 — translation X/Y and rotation curves
  stabilization_comparison.png    — original vs stabilized frames + crop
  step4_stabilized.avi            — stabilized video (input to step [5])
  step5_corrected.avi             — drift+flicker corrected video (input to step [6])
  step5_corrected.npz             — correction curves for traceability
  step5_luminosity_analysis.png   — 4-panel luminosity + FFT diagnostic
  step5_comparison.png            — original vs corrected frames grid
  banding_info.json               — banding detection results (step [6])
  banding_fft.png                 — FFT spectrum with detected peaks and notches
  banding_comparison.png          — before/after grid (only if banding found)
  step6_corrected.avi             — final output (banding corrected or copy)
  summary.txt                     — human-readable run report

Usage:
  # Default mode — final video only, no intermediate files
  python Pretraitement.py video.avi [output.avi]
  python Pretraitement.py folder/   [output_dir/]

  # Info mode — full diagnostics (intermediate videos, PNGs, JSON logs)
  python Pretraitement.py --info video.avi [output_dir/]
  python Pretraitement.py --info folder/   [output_dir/]
"""
from __future__ import annotations

import shutil
import sys
import tempfile
import time
import traceback
import warnings
from pathlib import Path

import cv2
import numpy as np

# ── Pipeline modules ──────────────────────────────────────────────────────────
from mask_detection import detect_circular_mask, save_mask
from BadFrameDétection import (
    detect_corrupted_frames,
    replace_corrupted_frames,
)
from PatternNoise import (
    estimate_fpn,
    correct_fpn,
    save_fpn_pattern,
    load_fpn_pattern,
    _severity_label,
    FPN_STD_MILD,
    FPN_STD_MODERATE,
    FPN_STD_SEVERE,
)
from Rigidisation import (
    compute_reference_frame,
    estimate_motion,
    apply_stabilization,
    save_motion,
    load_motion,
    visualize_motion,
    compare_stabilization,
    ECC_MAX_ITERATIONS,
    ECC_TERMINATION_EPS,
    ECC_GAUSS_FILT_SIZE,
    ECC_MAX_TRANSLATION_PX,
    ECC_MAX_ROTATION_DEG,
    ECC_N_REFERENCE_FRAMES,
)
from Flicker import (
    analyze_luminosity,
    correct_drift_and_flicker,
    visualize_luminosity_analysis,
    compare_correction,
    BUTTERWORTH_CUTOFF,
    POLY_ORDER,
    N_REFERENCE_FRAMES as FLICKER_N_REFERENCE_FRAMES,
)
from Banding import (
    detect_banding,
    correct_banding,
    visualize_fft as visualize_banding_fft,
    compare_banding_correction,
    N_SAMPLE_FRAMES as BANDING_N_SAMPLE_FRAMES,
    NOTCH_WIDTH as BANDING_NOTCH_WIDTH,
)


# ── Pipeline parameters (edit here) ──────────────────────────────────────────

# Step [1] — mask
MASK_N_SAMPLE_FRAMES = 20
MASK_MARGIN_PX       = 5

# Step [2] — bad frames
BAD_STD_THRESHOLD         = 3.0
BAD_CORRELATION_THRESHOLD = 0.5
BAD_MAX_ITERATIONS        = 10
BAD_N_AVG                 = 5
BAD_RUN_PADDING           = 5

# Step [3] — FPN
FPN_GAUSSIAN_SIGMA = 60.0   # must exceed vessel widths; see PatternNoise.FPN_GAUSSIAN_SIGMA

# Step [4] — ECC registration (values mirror Rigidisation module defaults)
STAB_MAX_ITERATIONS     = ECC_MAX_ITERATIONS
STAB_TERMINATION_EPS    = ECC_TERMINATION_EPS
STAB_GAUSS_FILT_SIZE    = ECC_GAUSS_FILT_SIZE
STAB_MAX_TRANSLATION_PX = ECC_MAX_TRANSLATION_PX
STAB_MAX_ROTATION_DEG   = ECC_MAX_ROTATION_DEG
STAB_N_REFERENCE_FRAMES = ECC_N_REFERENCE_FRAMES

# Step [5] — Flicker/drift correction (mirrors Flicker module defaults)
FLICKER_BUTTERWORTH_CUTOFF = BUTTERWORTH_CUTOFF
FLICKER_POLY_ORDER         = POLY_ORDER

# Step [6] — Banding correction (mirrors Banding module defaults)
BANDING_N_SAMPLE = BANDING_N_SAMPLE_FRAMES
BANDING_NOTCH    = BANDING_NOTCH_WIDTH


# ── Helpers ───────────────────────────────────────────────────────────────────


def _header(title: str) -> None:
    bar = "─" * 60
    print(f"\n{bar}")
    print(f"  {title}")
    print(bar)


def _sub(msg: str) -> None:
    print(f"  {msg}")


def _find_videos(path: Path) -> list[Path]:
    """Return all .avi files under path (or path itself if it is a file)."""
    if path.is_file():
        return [path]
    return sorted(path.rglob("*.avi"))


# ── Single-video pipeline ─────────────────────────────────────────────────────


def preprocess_video(
    video_path: Path,
    output_dir: Path,
    info_mode: bool = True,
    output_video_path: Path | None = None,
    # step [1]
    n_sample_frames: int = MASK_N_SAMPLE_FRAMES,
    margin_px: int = MASK_MARGIN_PX,
    # step [2]
    std_threshold: float = BAD_STD_THRESHOLD,
    correlation_threshold: float = BAD_CORRELATION_THRESHOLD,
    max_iterations: int = BAD_MAX_ITERATIONS,
    n_avg: int = BAD_N_AVG,
    run_padding: int = BAD_RUN_PADDING,
    # step [3]
    gaussian_sigma: float = FPN_GAUSSIAN_SIGMA,
    # step [4]
    stab_n_reference_frames: int = STAB_N_REFERENCE_FRAMES,
    # step [5]
    flicker_butterworth_cutoff: float = FLICKER_BUTTERWORTH_CUTOFF,
    flicker_poly_order: int = FLICKER_POLY_ORDER,
    # step [6]
    banding_n_sample: int = BANDING_N_SAMPLE,
    banding_notch_width: int = BANDING_NOTCH,
) -> dict:
    """
    Run the full preprocessing pipeline on one video.

    Intermediate files are cached: if a file already exists it is reused,
    so re-running the script after a crash is safe and fast.

    Args:
        video_path: Path to the source .avi.
        output_dir: Destination directory for all outputs of this video.
        ...        : Step-specific tuning parameters (see module constants).

    Returns:
        Summary dict with keys:
          'video'                : str   — source video name
          'mask_coverage'        : float
          'n_corrupted'          : int   — total replaced frames (all iterations)
          'fpn_severity'         : str
          'fpn_std'              : float
          'n_failed_registration': int   — frames where ECC failed
          'drift_detected'       : bool
          'drift_r2'             : float
          'flicker_detected'     : bool
          'flicker_std'          : float
          'pulsation_freq_hz'    : float
          'banding_detected'     : bool
          'banding_severity'     : str
          'final_video'          : str   — path to final output video
          'elapsed_s'            : float
          'status'               : 'ok' | 'error'
          'error'                : str   — only present on failure

    Modes:
        info_mode=True  (default): creates output_dir with every intermediate
                        file, visualisation PNG, JSON log, and summary.txt.
        info_mode=False: uses a temporary directory for all intermediate files;
                        only the final corrected video is written to
                        output_video_path.  No diagnostics are produced.
    """
    t0       = time.time()
    stem     = video_path.stem
    _tmp_ctx = None

    if info_mode:
        output_dir.mkdir(parents=True, exist_ok=True)
        working = output_dir
    else:
        _tmp_ctx = tempfile.TemporaryDirectory()
        working  = Path(_tmp_ctx.name)

    summary: dict = {
        "video":                 str(video_path),
        "mask_coverage":         0.0,
        "n_corrupted":           0,
        "fpn_severity":          "unknown",
        "fpn_std":               0.0,
        "n_failed_registration": 0,
        "drift_detected":        False,
        "drift_r2":              0.0,
        "flicker_detected":      False,
        "flicker_std":           0.0,
        "pulsation_freq_hz":     0.0,
        "banding_detected":      False,
        "banding_severity":      "none",
        "final_video":           "",
        "elapsed_s":             0.0,
        "status":                "ok",
    }

    try:
        # ── Step [1] : Mask ───────────────────────────────────────────────────
        _header(f"[1/6] Mask detection — {stem}")

        mask_path = working / "mask.png"

        if mask_path.exists():
            _sub(f"Reusing cached mask : {mask_path.name}")
            mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
            coverage = float(np.count_nonzero(mask)) / mask.size
        else:
            mask_result = detect_circular_mask(
                video_path,
                n_sample_frames=n_sample_frames,
                margin_px=margin_px,
            )
            mask     = mask_result["mask"]
            coverage = mask_result["coverage"]
            if info_mode:
                save_mask(mask, mask_path)
            _sub(f"center={mask_result['center']}  radius={mask_result['radius']:.1f} px"
                 f"  coverage={coverage:.1%}")

        summary["mask_coverage"] = coverage

        # ── Step [2] : Bad frames (iterative) ────────────────────────────────
        _header(f"[2/6] Bad frame correction — {stem}")

        step2_out = working / "step2_corrected.avi"

        total_corrupted = 0

        if step2_out.exists():
            _sub(f"Reusing cached step-2 video : {step2_out.name}")
            # Reload corrupted count from the JSON log if present
            import json
            log_path = step2_out.with_suffix(".json")
            if log_path.exists():
                with open(log_path) as fh:
                    log = json.load(fh)
                total_corrupted = len(log)
                _sub(f"Log shows {total_corrupted} replaced frames")
        else:
            current_input = video_path
            iter_paths: list[Path] = []

            for iteration in range(1, max_iterations + 1):
                _sub(f"Iteration {iteration}/{max_iterations}  "
                     f"(input: {current_input.name})")

                with warnings.catch_warnings(record=True) as caught:
                    warnings.simplefilter("always")
                    info = detect_corrupted_frames(
                        current_input, mask,
                        std_threshold=std_threshold,
                        correlation_threshold=correlation_threshold,
                    )
                    for w in caught:
                        _sub(f"  WARNING: {w.message}")

                n = len(info["all_corrupted"])
                total = len(info["luminosity_curve"])

                _sub(f"  flash={len(info['flash_frames'])}  "
                     f"blink={len(info['blink_frames'])}  "
                     f"occlusion={len(info['occlusion_frames'])}  "
                     f"total={n}/{total}")

                if n == 0:
                    _sub("  No corrupted frames — stopping iterations.")
                    break

                total_corrupted += n

                iter_path = working / f"_iter{iteration}.avi"
                iter_paths.append(iter_path)

                replace_corrupted_frames(
                    current_input, mask, info, iter_path,
                    n_avg=n_avg, run_padding=run_padding,
                )
                current_input = iter_path

            else:
                _sub(f"Reached max iterations ({max_iterations}).")

            # Rename last iter output to canonical step2 name
            if current_input != video_path:
                shutil.copy2(current_input, step2_out)
                _sub(f"Final step-2 video : {step2_out.name}")
                # Remove intermediate iter files
                for p in iter_paths[:-1]:
                    p.unlink(missing_ok=True)
                if iter_paths:
                    iter_paths[-1].unlink(missing_ok=True)
            else:
                # No corrupted frames at all — just copy original
                shutil.copy2(video_path, step2_out)
                _sub("No iterations needed — original copied as step-2 output.")

        summary["n_corrupted"] = total_corrupted

        # ── Step [3] : FPN ────────────────────────────────────────────────────
        _header(f"[3/6] FPN correction — {stem}")

        fpn_npy   = working / "fpn_pattern.npy"
        final_out = working / "final.avi"

        # Reload or estimate pattern
        if fpn_npy.exists():
            _sub(f"Reusing cached FPN pattern : {fpn_npy.name}")
            cached = load_fpn_pattern(fpn_npy)
            mask_bool   = mask > 0
            pattern_std = float(cached[mask_bool].std()) if mask_bool.any() else 0.0
            severity    = _severity_label(
                pattern_std, FPN_STD_MILD, FPN_STD_MODERATE, FPN_STD_SEVERE
            )
            fpn_dict = {
                "pattern":        cached,
                "pattern_std":    pattern_std,
                "fpn_detected":   pattern_std >= FPN_STD_MILD,
                "severity":       severity,
                "n_clean":        -1,
                "gaussian_sigma": gaussian_sigma,
            }
        else:
            # Pass corrupted-frame list so the estimator ignores bad frames.
            # We reload from the step-2 JSON log when available.
            corrupted_frames: list[int] = []
            import json
            log_path = step2_out.with_suffix(".json")
            if log_path.exists():
                with open(log_path) as fh:
                    corrupted_frames = [int(k) for k in json.load(fh).keys()]

            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                fpn_dict = estimate_fpn(
                    step2_out, mask,
                    corrupted_frames=corrupted_frames,
                    gaussian_sigma=gaussian_sigma,
                )
                for w in caught:
                    _sub(f"  WARNING: {w.message}")

            if info_mode:
                save_fpn_pattern(fpn_dict, fpn_npy)

        _sub(f"pattern_std={fpn_dict['pattern_std']:.4f}  "
             f"severity={fpn_dict['severity']}  "
             f"detected={fpn_dict['fpn_detected']}")

        summary["fpn_severity"] = fpn_dict["severity"]
        summary["fpn_std"]      = fpn_dict["pattern_std"]

        if final_out.exists():
            _sub(f"Reusing cached final video : {final_out.name}")
        else:
            if fpn_dict["fpn_detected"]:
                _sub("Applying FPN correction …")
                correct_fpn(step2_out, mask, fpn_dict, final_out)
            else:
                _sub("No FPN detected — copying step-2 output as final.")
                shutil.copy2(step2_out, final_out)

        # ── Step [4] : Rigid stabilization ────────────────────────────────────
        _header(f"[4/6] Rigid stabilization — {stem}")

        stab_out   = working / "step4_stabilized.avi"
        motion_npz = working / "motion.npz"
        motion_plt = working / "motion_plot.png"
        stab_cmp   = working / "stabilization_comparison.png"

        # Reload corrupted frame list from step-2 log
        corrupted_frames_s4: list[int] = []
        import json as _json
        log_path_s4 = step2_out.with_suffix(".json")
        if log_path_s4.exists():
            with open(log_path_s4) as fh:
                corrupted_frames_s4 = [int(k) for k in _json.load(fh).keys()]

        # Reference frame (always recomputed — fast, median of 60 frames)
        _sub("Computing reference frame …")
        reference_frame = compute_reference_frame(
            final_out, mask, corrupted_frames_s4,
            n_frames=stab_n_reference_frames,
        )

        # Motion estimation (cached)
        if motion_npz.exists():
            _sub(f"Reusing cached motion : {motion_npz.name}")
            motion_dict = load_motion(motion_npz)
        else:
            _sub("Estimating motion …")
            motion_dict = estimate_motion(
                final_out, reference_frame, mask, corrupted_frames_s4,
            )
            if info_mode:
                save_motion(motion_dict, motion_npz)

        n_failed = len(motion_dict["failed_frames"])
        n_total  = len(motion_dict["transforms"])
        _sub(f"Failed : {n_failed}/{n_total} ({n_failed/n_total:.1%})")
        summary["n_failed_registration"] = n_failed

        if n_failed > n_total * 0.10:
            _sub(f"WARNING: >10% registration failures — check video quality.")

        # Apply stabilization (cached)
        if stab_out.exists():
            _sub(f"Reusing cached stabilized video : {stab_out.name}")
        else:
            _sub("Applying stabilization …")
            apply_stabilization(final_out, mask, motion_dict, stab_out)

        # Visualisations (info mode only)
        if info_mode:
            if not motion_plt.exists():
                visualize_motion(motion_dict, motion_plt)
            if not stab_cmp.exists():
                compare_stabilization(
                    video_path, stab_out, motion_dict, stab_cmp,
                    reference_frame=reference_frame,
                )

        # ── Step [5] : Flicker / drift correction ────────────────────────────
        _header(f"[5/6] Flicker & drift correction — {stem}")

        step5_out  = working / "step5_corrected.avi"
        step5_viz  = working / "step5_luminosity_analysis.png"
        step5_cmp  = working / "step5_comparison.png"

        # Luminosity analysis (always recomputed unless cached npz exists)
        lum_dict: dict | None = None

        if step5_out.exists():
            _sub(f"Reusing cached step-5 video: {step5_out.name}")
            # Try to reload summary metrics from the correction npz
            npz_path = step5_out.with_suffix(".npz")
            if npz_path.exists():
                _sub(f"Loading correction metrics from {npz_path.name}")
        else:
            _sub("Analysing luminosity …")
            lum_dict = analyze_luminosity(
                stab_out, mask, corrupted_frames_s4,
                butterworth_cutoff=flicker_butterworth_cutoff,
                poly_order=flicker_poly_order,
            )

            if info_mode and not step5_viz.exists():
                visualize_luminosity_analysis(lum_dict, step5_viz)

            _sub("Correcting drift + flicker …")
            correct_drift_and_flicker(
                stab_out, mask, lum_dict, reference_frame, step5_out,
                butterworth_cutoff=flicker_butterworth_cutoff,
                poly_order=flicker_poly_order,
            )

        if lum_dict is not None:
            summary["drift_detected"]    = lum_dict["drift_detected"]
            summary["drift_r2"]          = lum_dict["drift_r2"]
            summary["flicker_detected"]  = lum_dict["flicker_detected"]
            summary["flicker_std"]       = lum_dict["flicker_std"]
            summary["pulsation_freq_hz"] = lum_dict["pulsation_freq_hz"]

            _sub(f"drift={lum_dict['drift_detected']}  r²={lum_dict['drift_r2']:.3f}  "
                 f"flicker={lum_dict['flicker_detected']}  std={lum_dict['flicker_std']:.4f}  "
                 f"pulsation={lum_dict['pulsation_freq_hz']:.3f} Hz")

        if info_mode and not step5_cmp.exists() and lum_dict is not None:
            compare_correction(stab_out, step5_out, lum_dict, step5_cmp)

        # ── Step [6] : Banding correction ─────────────────────────────────────
        _header(f"[6/6] Banding correction — {stem}")

        step6_out    = working / "step6_corrected.avi"
        banding_json = working / "banding_info.json"
        banding_fft  = working / "banding_fft.png"
        banding_cmp  = working / "banding_comparison.png"

        if step6_out.exists():
            _sub(f"Reusing cached step-6 video: {step6_out.name}")
            if banding_json.exists():
                import json as _bj
                with open(banding_json) as fh:
                    _bi = _bj.load(fh)
                summary["banding_detected"] = _bi.get("banding_detected", False)
                summary["banding_severity"] = _bi.get("severity", "none")
        else:
            _sub("Detecting banding …")
            banding_dict = detect_banding(
                step5_out, mask, corrupted_frames_s4,
                n_sample_frames=banding_n_sample,
            )

            if info_mode:
                import json as _bj
                banding_saveable = {k: v for k, v in banding_dict.items() if k != "mean_fft"}
                with open(banding_json, "w") as fh:
                    _bj.dump(banding_saveable, fh, indent=2)
                if not banding_fft.exists():
                    visualize_banding_fft(banding_dict, str(banding_fft))

            _sub(
                f"banding={banding_dict['banding_detected']}  "
                f"severity={banding_dict['severity']}  "
                f"H_peaks={banding_dict['horizontal_freqs']}  "
                f"V_peaks={banding_dict['vertical_freqs']}"
            )

            _sub("Applying banding correction …")
            correct_banding(
                str(step5_out), mask, banding_dict, str(step6_out),
                notch_width=banding_notch_width,
            )

            if info_mode and banding_dict["banding_detected"] and not banding_cmp.exists():
                compare_banding_correction(
                    str(step5_out), str(step6_out), banding_dict, str(banding_cmp)
                )

            summary["banding_detected"] = banding_dict["banding_detected"]
            summary["banding_severity"] = banding_dict["severity"]

        if info_mode:
            summary["final_video"] = str(step6_out)
        elif output_video_path is not None:
            output_video_path.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(str(step6_out), str(output_video_path))
            mask_dest = output_video_path.parent / f"{stem}_mask.png"
            if mask_path.exists():
                shutil.copy2(str(mask_path), str(mask_dest))
            summary["final_video"] = str(output_video_path)
        else:
            summary["final_video"] = str(step6_out)

    except Exception:
        summary["status"] = "error"
        summary["error"]  = traceback.format_exc()
        print(f"\n  ERROR processing {video_path.name}:\n{summary['error']}")

    summary["elapsed_s"] = round(time.time() - t0, 1)

    if info_mode:
        _write_summary(summary, working / "summary.txt")

    if _tmp_ctx is not None:
        _tmp_ctx.cleanup()

    return summary


# ── Summary report ────────────────────────────────────────────────────────────


def _write_summary(summary: dict, path: Path) -> None:
    lines = [
        "Preprocessing summary",
        "=" * 40,
        f"Video                : {summary['video']}",
        f"Status               : {summary['status']}",
        f"Mask coverage        : {summary['mask_coverage']:.1%}",
        f"Replaced frames      : {summary['n_corrupted']}",
        f"FPN severity         : {summary['fpn_severity']}",
        f"FPN std              : {summary['fpn_std']:.4f}",
        f"Failed registration  : {summary['n_failed_registration']}",
        f"Drift detected       : {summary['drift_detected']}  (R²={summary['drift_r2']:.3f})",
        f"Flicker detected     : {summary['flicker_detected']}  (std={summary['flicker_std']:.4f})",
        f"Pulsation            : {summary['pulsation_freq_hz']:.3f} Hz",
        f"Banding detected     : {summary['banding_detected']}  (severity={summary['banding_severity']})",
        f"Final output         : {summary['final_video']}",
        f"Elapsed              : {summary['elapsed_s']} s",
    ]
    if "error" in summary:
        lines += ["", "Error:", summary["error"]]
    path.write_text("\n".join(lines) + "\n")


# ── Folder pipeline ───────────────────────────────────────────────────────────


def preprocess_folder(
    folder_path: Path,
    output_root: Path,
    info_mode: bool = True,
    **kwargs,
) -> list[dict]:
    """
    Apply preprocess_video to every .avi found under folder_path.

    info_mode=True : each video gets output_root/<video_stem>/ with all files.
    info_mode=False: each final video is placed flat in output_root/<stem>.avi.

    Args:
        folder_path: Directory to scan recursively for .avi files.
        output_root: Root output directory.
        info_mode:   True = full diagnostics; False = final videos only.
        **kwargs:    Forwarded to preprocess_video (step tuning params).

    Returns:
        List of summary dicts (one per video).
    """
    videos = _find_videos(folder_path)
    if not videos:
        print(f"No .avi files found under: {folder_path}")
        return []

    print(f"Found {len(videos)} video(s) under {folder_path}")

    if not info_mode:
        output_root.mkdir(parents=True, exist_ok=True)

    summaries: list[dict] = []
    for i, vid in enumerate(videos, 1):
        print(f"\n{'═' * 60}")
        print(f"  Video {i}/{len(videos)} : {vid.name}")
        print(f"{'═' * 60}")
        if info_mode:
            out_dir = output_root / vid.stem
            summaries.append(
                preprocess_video(vid, out_dir, info_mode=True, **kwargs)
            )
        else:
            out_video = output_root / f"{vid.stem}.avi"
            summaries.append(
                preprocess_video(
                    vid, Path("."), info_mode=False,
                    output_video_path=out_video, **kwargs,
                )
            )

    # ── Global report (info mode only) ───────────────────────────────────────
    _header("Global report")
    ok  = [s for s in summaries if s["status"] == "ok"]
    err = [s for s in summaries if s["status"] == "error"]
    print(f"  Processed : {len(summaries)}  OK : {len(ok)}  Errors : {len(err)}")
    for s in summaries:
        tag = "✓" if s["status"] == "ok" else "✗"
        line = (
            f"  {tag}  {Path(s['video']).name:<40}"
            f"  {s['elapsed_s']} s"
        )
        if info_mode:
            line = (
                f"  {tag}  {Path(s['video']).name:<40}"
                f"  fpn={s['fpn_severity']:<8}"
                f"  bad={s['n_corrupted']:>4} frames"
                f"  regfail={s['n_failed_registration']:>4}"
                f"  drift={'Y' if s['drift_detected'] else 'N'}"
                f"  flicker={'Y' if s['flicker_detected'] else 'N'}"
                f"  banding={'Y' if s['banding_detected'] else 'N'}({s['banding_severity']})"
                f"  {s['elapsed_s']} s"
            )
        print(line)

    if info_mode:
        report_path = output_root / "pipeline_report.txt"
        with open(report_path, "w") as fh:
            fh.write("Pipeline report\n" + "=" * 60 + "\n")
            for s in summaries:
                tag = "OK   " if s["status"] == "ok" else "ERROR"
                fh.write(
                    f"[{tag}]  {Path(s['video']).name}\n"
                    f"       mask={s['mask_coverage']:.1%}  "
                    f"bad={s['n_corrupted']}  "
                    f"fpn={s['fpn_severity']}(std={s['fpn_std']:.3f})  "
                    f"regfail={s['n_failed_registration']}  "
                    f"drift={s['drift_detected']}(r2={s['drift_r2']:.2f})  "
                    f"flicker={s['flicker_detected']}(std={s['flicker_std']:.4f})  "
                    f"pulsation={s['pulsation_freq_hz']:.2f}Hz  "
                    f"banding={s['banding_detected']}({s['banding_severity']})  "
                    f"{s['elapsed_s']} s\n"
                )
                if "error" in s:
                    fh.write(f"       {s['error'][:200]}\n")
        print(f"\n  Report saved : {report_path}")

    return summaries


# ── CLI ───────────────────────────────────────────────────────────────────────


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(
        prog="Pretraitement.py",
        description="Preprocessing pipeline for ocular vascular imaging videos.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Modes:\n"
            "  Default     : produces only the final corrected .avi — no intermediate\n"
            "                files, no diagnostics, no subdirectories.\n"
            "  --info      : full diagnostic mode — intermediate videos, visualisation\n"
            "                PNGs, JSON logs, summary.txt, and pipeline_report.txt.\n\n"
            "Examples:\n"
            "  python Pretraitement.py video.avi\n"
            "    → video_final.avi next to the input\n\n"
            "  python Pretraitement.py video.avi output.avi\n"
            "    → final video saved as output.avi\n\n"
            "  python Pretraitement.py --info video.avi\n"
            "    → ./preprocessed/video_stem/ with all diagnostics\n\n"
            "  python Pretraitement.py --info video.avi results/\n"
            "    → results/video_stem/ with all diagnostics\n\n"
            "  python Pretraitement.py folder/\n"
            "    → folder_final/<stem>.avi for each video (no intermediate files)\n\n"
            "  python Pretraitement.py --info folder/ results/\n"
            "    → results/<stem>/ for each video with all diagnostics\n"
        ),
    )
    parser.add_argument("input",  help=".avi video file or folder of videos")
    parser.add_argument(
        "output", nargs="?",
        help=(
            "Default mode: output .avi path (single video) or output directory (folder). "
            "--info mode: output directory (per-video subdirs are created inside)."
        ),
    )
    parser.add_argument(
        "--info", action="store_true",
        help="Save all intermediate files, visualisations, and diagnostic reports.",
    )

    args       = parser.parse_args()
    input_path = Path(args.input)
    info_mode  = args.info

    if not input_path.exists():
        print(f"Error: path not found — {input_path}")
        sys.exit(1)

    if input_path.is_file():
        if info_mode:
            out_root = Path(args.output) if args.output else Path.cwd() / "preprocessed"
            out_dir  = out_root / input_path.stem
            summary  = preprocess_video(input_path, out_dir, info_mode=True)
        else:
            out_video = (
                Path(args.output) if args.output
                else input_path.parent / f"{input_path.stem}_final.avi"
            )
            summary = preprocess_video(
                input_path, Path("."), info_mode=False,
                output_video_path=out_video,
            )
            print(f"\n  Final video : {summary['final_video']}")
        sys.exit(0 if summary["status"] == "ok" else 1)

    else:
        if info_mode:
            out_root  = Path(args.output) if args.output else Path.cwd() / "preprocessed"
            summaries = preprocess_folder(input_path, out_root, info_mode=True)
        else:
            out_root  = (
                Path(args.output) if args.output
                else input_path.parent / f"{input_path.stem}_final"
            )
            summaries = preprocess_folder(input_path, out_root, info_mode=False)
            print(f"\n  Final videos saved to: {out_root}")
        sys.exit(1 if any(s["status"] == "error" for s in summaries) else 0)


if __name__ == "__main__":
    main()
