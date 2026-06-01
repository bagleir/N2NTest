#!/usr/bin/env python3
"""
Pipeline d'inférence FastDVDnet N2N pour le débruitage de vidéos vasculaires.

Enchaînement :
    1. Charger la vidéo prétraitée (étape 6 — PretraitementIntegrale/)
    2. Charger le masque (étape 1)
    3. Appliquer temporal median filter (étape 7)
    4. Appliquer FastDVDnet N2N
    5. Sauvegarder la vidéo débruitée en .avi

Usage :
    python inference.py --video path/to/preprocessed.avi
                        [--config config.yaml]
                        [--checkpoint checkpoints/best_model.pth]
                        [--output denoised.avi]
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch
import yaml

SCRIPT_DIR   = Path(__file__).resolve().parent   # Stage/Débruitage/
PROJECT_ROOT = SCRIPT_DIR.parent                 # Stage/

from mask_detection import load_mask
from Temporel import temporal_median_filter
from model import FastDVDnet

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)


# ── I/O vidéo ─────────────────────────────────────────────────────────────────

def _read_video_gray(path: Path) -> tuple[np.ndarray, float, tuple[int, int]]:
    """Retourne (N, H, W) uint8, fps, (W, H)."""
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise FileNotFoundError(f"Impossible d'ouvrir : {path}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    w   = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h   = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frames: list[np.ndarray] = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if frame.ndim == 3:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        frames.append(frame)
    cap.release()
    if not frames:
        raise RuntimeError(f"Aucune frame : {path}")
    return np.stack(frames), fps, (w, h)


def _save_video(frames: np.ndarray, path: Path, fps: float, size: tuple[int, int]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"XVID"), fps, size, isColor=False)
    for f in frames:
        writer.write(f)
    writer.release()


# ── Inférence ─────────────────────────────────────────────────────────────────

def denoise_video(
    frames:          np.ndarray,
    mask:            np.ndarray,
    model:           FastDVDnet,
    sigma:           float,
    device:          torch.device,
    mixed_precision: bool = True,
) -> tuple[np.ndarray, float, float]:
    """
    Applique FastDVDnet frame par frame (réflexion aux bords).

    Retourne :
        denoised      : (N, H, W) uint8
        mean_ms       : temps moyen par frame (ms)
        peak_vram_mb  : VRAM maximale utilisée (MB)
    """
    N, H, W    = frames.shape
    sigma_norm = sigma / 255.0
    sigma_t    = torch.full((1, 1, H, W), sigma_norm, dtype=torch.float32, device=device)
    frames_f32 = frames.astype(np.float32)
    denoised   = np.empty_like(frames)
    times: list[float] = []

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    model.eval()
    with torch.no_grad():
        for t in range(N):
            # Réflexion aux bords pour les 5 frames
            idxs  = [max(0, min(N - 1, t + k)) for k in range(-2, 3)]
            clip  = np.stack([frames_f32[i] for i in idxs])
            clip_t = torch.from_numpy(clip / 255.0).unsqueeze(0).to(device)

            t0 = time.perf_counter()
            if mixed_precision and device.type == "cuda":
                with torch.cuda.amp.autocast():
                    out_t = model(clip_t, sigma_t)
            else:
                out_t = model(clip_t, sigma_t)
            if device.type == "cuda":
                torch.cuda.synchronize()
            times.append((time.perf_counter() - t0) * 1000.0)

            out_np     = (out_t[0, 0].cpu().numpy() * 255.0).clip(0, 255).astype(np.uint8)
            denoised[t] = np.where(mask > 0, out_np, frames[t])

    mean_ms  = float(np.mean(times[5:])) if len(times) > 5 else float(np.mean(times))
    peak_vram = torch.cuda.max_memory_allocated(device) / 1e6 if device.type == "cuda" else 0.0
    return denoised, mean_ms, peak_vram


# ── Grille de comparaison ──────────────────────────────────────────────────────

def generate_comparison_grid(
    preprocessed: np.ndarray,
    median:       np.ndarray,
    n2n:          np.ndarray,
    output_path:  Path,
    n_frames:     int = 5,
    crop_size:    int = 150,
    mask:         np.ndarray | None = None,
) -> None:
    """
    Grille de comparaison :
      Ligne 1 : N2N frames sélectionnées (avec overlay diff)
      Ligne 2 : Crops 150×150 sur zone riche en petits vaisseaux
                (Prétraité | Median | N2N | Diff×5)
      Ligne 3 : Courbes de luminosité temporelle (vérification pulsation)
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    N, H, W = preprocessed.shape
    tidx = np.linspace(0, N - 1, n_frames, dtype=int)

    # Zone la plus riche en vaisseaux (variance max dans masque)
    best_var, br, bc = -1.0, H // 2 - crop_size // 2, W // 2 - crop_size // 2
    rng = np.random.default_rng(0)
    if mask is not None:
        for _ in range(1000):
            r = rng.integers(0, H - crop_size)
            c = rng.integers(0, W - crop_size)
            if mask[r:r+crop_size, c:c+crop_size].mean() / 255.0 < 0.90:
                continue
            v = float(median[:, r:r+crop_size, c:c+crop_size].var())
            if v > best_var:
                best_var, br, bc = v, r, c

    n_cols = max(n_frames, 4)
    fig, axes = plt.subplots(3, n_cols, figsize=(4 * n_cols, 11))

    # ── Ligne 1 : frames N2N + overlay diff ───────────────────────────────
    for fi, frame_i in enumerate(tidx):
        ax = axes[0, fi]
        ax.imshow(n2n[frame_i], cmap="gray", vmin=0, vmax=255)
        diff = np.abs(n2n[frame_i].astype(float) - median[frame_i].astype(float)) * 5
        ax.imshow(np.clip(diff, 0, 255).astype(np.uint8), cmap="hot", alpha=0.35, vmin=0, vmax=255)
        ax.set_title(f"N2N t={frame_i}", fontsize=8)
        ax.axis("off")
    for fi in range(n_frames, n_cols):
        axes[0, fi].axis("off")

    # ── Ligne 2 : crops comparatifs ───────────────────────────────────────
    mid = N // 2
    sources_crop = [
        (preprocessed, "Prétraité"),
        (median,        "Median"),
        (n2n,           "N2N"),
    ]
    for si, (src, lbl) in enumerate(sources_crop):
        axes[1, si].imshow(src[mid, br:br+crop_size, bc:bc+crop_size], cmap="gray", vmin=0, vmax=255)
        axes[1, si].set_title(lbl, fontsize=8)
        axes[1, si].axis("off")
    diff_crop = np.abs(n2n[mid, br:br+crop_size, bc:bc+crop_size].astype(float) -
                       median[mid, br:br+crop_size, bc:bc+crop_size].astype(float)) * 5
    axes[1, 3].imshow(np.clip(diff_crop, 0, 255).astype(np.uint8), cmap="hot", vmin=0, vmax=255)
    axes[1, 3].set_title("Diff×5", fontsize=8)
    axes[1, 3].axis("off")
    for ci in range(4, n_cols):
        axes[1, ci].axis("off")

    # ── Ligne 3 : courbes de luminosité temporelle ─────────────────────────
    t_axis = np.arange(N) / 30.0
    for si, (src, lbl, col) in enumerate([
        (preprocessed, "Prétraité", "#5bc8af"),
        (median,        "Median",    "#e8a838"),
        (n2n,           "N2N",       "#e84040"),
    ]):
        curve = src[:, br:br+crop_size, bc:bc+crop_size].mean(axis=(1, 2))
        axes[2, si].plot(t_axis, curve, color=col, lw=0.9)
        axes[2, si].set_title(f"Luminosité — {lbl}", fontsize=8)
        axes[2, si].set_xlabel("Temps (s)", fontsize=7)
        axes[2, si].set_ylabel("Intensité moy.", fontsize=7)
    for ci in range(3, n_cols):
        axes[2, ci].axis("off")

    fig.suptitle("Comparaison pipeline de débruitage", fontsize=12)
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=120)
    plt.close(fig)
    log.info("Grille de comparaison → %s", output_path)


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="Inférence FastDVDnet N2N")
    parser.add_argument("--video",      required=True)
    parser.add_argument("--config",     default=str(SCRIPT_DIR / "config.yaml"))
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--output",     default=None)
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    video_path = Path(args.video)
    if not video_path.exists():
        log.error("Vidéo introuvable : %s", video_path)
        sys.exit(1)

    ckpt_path = Path(args.checkpoint or (PROJECT_ROOT / cfg["inference"]["checkpoint"]))
    if not ckpt_path.exists():
        log.error("Checkpoint introuvable : %s", ckpt_path)
        sys.exit(1)

    mask_path = PROJECT_ROOT / cfg["data"]["mask_path"]
    out_dir   = PROJECT_ROOT / cfg["inference"]["output_dir"]
    out_path  = Path(args.output) if args.output else (out_dir / (video_path.stem + "_n2n.avi"))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log.info("Device : %s", device)

    # ── Chargement du modèle ──────────────────────────────────────────────
    model = FastDVDnet(features=cfg["model"]["features"]).to(device)
    ckpt  = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    log.info("Modèle chargé — epoch %d | val loss=%.5f",
             ckpt.get("epoch", 0), ckpt.get("val_loss", float("nan")))

    mask = load_mask(mask_path)

    # ── Étape 3 : temporal median filter ─────────────────────────────────
    log.info("Étape 3 : temporal median filter…")
    tm_cfg      = cfg["inference"].get("temporal_median", {})
    median_path = out_path.with_name(out_path.stem + "_tmpmed.avi")
    temporal_median_filter(
        video_path       = str(video_path),
        mask             = mask,
        output_path      = str(median_path),
        window_size      = tm_cfg.get("window_size", 5),
        gaussian_weights = tm_cfg.get("gaussian_weights", True),
    )

    # ── Étape 4 : FastDVDnet ──────────────────────────────────────────────
    log.info("Étape 4 : débruitage FastDVDnet…")
    median_frames, fps, size = _read_video_gray(median_path)
    denoised, mean_ms, vram_mb = denoise_video(
        frames          = median_frames,
        mask            = mask,
        model           = model,
        sigma           = cfg["data"]["sigma_noise"],
        device          = device,
        mixed_precision = cfg["training"]["mixed_precision"],
    )
    log.info(
        "Inférence : %.2f ms/frame → %.1f FPS%s",
        mean_ms, 1000.0 / max(mean_ms, 1e-3),
        f" | VRAM peak : {vram_mb:.0f} MB" if vram_mb > 0 else "",
    )
    if mean_ms > 33.3:
        log.warning("⚠ Dépasse le budget 33 ms/frame (objectif 30 fps)")

    # ── Étape 5 : sauvegarde ─────────────────────────────────────────────
    log.info("Étape 5 : sauvegarde → %s", out_path)
    _save_video(denoised, out_path, fps, size)

    # ── Grille de comparaison ─────────────────────────────────────────────
    if cfg["inference"]["comparison"]["enabled"]:
        log.info("Génération de la grille de comparaison…")
        preproc_frames, _, _ = _read_video_gray(video_path)
        generate_comparison_grid(
            preprocessed = preproc_frames,
            median       = median_frames,
            n2n          = denoised,
            output_path  = out_path.with_name(out_path.stem + "_comparison.png"),
            n_frames     = cfg["inference"]["comparison"]["n_frames"],
            crop_size    = cfg["inference"]["comparison"]["crop_size"],
            mask         = mask,
        )

    if median_path.exists():
        median_path.unlink()

    log.info("Terminé.")


if __name__ == "__main__":
    main()
