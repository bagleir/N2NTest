#!/usr/bin/env python3
"""
Comparaison objective du bruit entre deux vidéos vasculaires oculaires.

Usage :
    python compare_noise.py video_A.avi video_B.avi mask.png
    python compare_noise.py video_A.avi video_B.avi mask.png --output_dir résultats/
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).parent))
from mask_detection import load_mask

# ─────────────────────────────────────────────────────────────────────────────
# Paramètres modifiables
# ─────────────────────────────────────────────────────────────────────────────

N_SAMPLE_FRAMES    = 20    # frames échantillons pour les métriques
TILE_SIZE          = 32    # taille des tuiles pour la carte de bruit locale (px)
HIGH_FREQ_THRESH   = 0.3   # seuil hautes fréquences (fraction de Nyquist, 0–1)
LOW_FREQ_THRESH    = 0.1   # seuil basses fréquences (fraction de Nyquist, 0–1)
SNR_DILATION_PX    = 5     # rayon (px) pour définir le fond proche des vaisseaux
TEMPORAL_VIZ_N     = 5     # paires dans la visualisation temporelle
CROP_SIZE          = 150   # taille du crop de zone vasculaire (px)


# ─────────────────────────────────────────────────────────────────────────────
# Utilitaires
# ─────────────────────────────────────────────────────────────────────────────

def _bar(current: int, total: int, width: int = 40) -> str:
    pct    = current / max(total, 1)
    filled = int(width * pct)
    return f"[{'█' * filled}{'░' * (width - filled)}] {current}/{total}"


def _step(title: str) -> None:
    print(f"\n{'─' * 62}\n  {title}\n{'─' * 62}")


# ─────────────────────────────────────────────────────────────────────────────
# Chargement des frames
# ─────────────────────────────────────────────────────────────────────────────

def load_sample_frames(path: str, n: int = N_SAMPLE_FRAMES) -> tuple[list[np.ndarray], int]:
    """Charge n frames uniformément réparties. Retourne (frames_float32, total)."""
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise RuntimeError(f"Impossible d'ouvrir : {path}")
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total < 1:
        raise RuntimeError(f"Vidéo vide ou non lisible : {path}")

    indices = np.linspace(0, total - 1, min(n, total), dtype=int)
    frames: list[np.ndarray] = []
    print(f"  {Path(path).name}  ({total} frames, {len(indices)} échantillons)")
    for i, idx in enumerate(indices):
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
        ret, frame = cap.read()
        if not ret:
            print(f"\n  AVERT : frame {idx} illisible, ignorée")
            continue
        if frame.ndim == 3:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        frames.append(frame.astype(np.float32))
        print(f"\r    {_bar(i + 1, len(indices))}", end="", flush=True)
    print()
    cap.release()
    return frames, total


def load_consecutive_pairs(
    path: str, n: int = TEMPORAL_VIZ_N, total: int = 0
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Charge n paires (frame[i], frame[i+1]) uniformément réparties."""
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        return []
    if total < 1:
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    indices = np.linspace(0, max(total - 2, 0), min(n, max(total - 1, 1)), dtype=int)
    pairs: list[tuple[np.ndarray, np.ndarray]] = []
    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
        ret1, f1 = cap.read()
        ret2, f2 = cap.read()
        if not (ret1 and ret2):
            continue
        if f1.ndim == 3:
            f1 = cv2.cvtColor(f1, cv2.COLOR_BGR2GRAY)
        if f2.ndim == 3:
            f2 = cv2.cvtColor(f2, cv2.COLOR_BGR2GRAY)
        pairs.append((f1.astype(np.float32), f2.astype(np.float32)))
    cap.release()
    return pairs


# ─────────────────────────────────────────────────────────────────────────────
# Métriques
# ─────────────────────────────────────────────────────────────────────────────

def estimate_noise_sigma(frames: list[np.ndarray], mask: np.ndarray) -> dict:
    """
    Pour chaque frame dans le masque :
        lap = cv2.Laplacian(frame, cv2.CV_64F)
        sigma = sqrt(mean(lap²) / 6) * sqrt(π/2)
    Retourne :
    - 'sigma_mean'   : float, moyenne sur toutes les frames
    - 'sigma_std'    : float, variabilité inter-frames
    - 'sigma_curve'  : np.ndarray, sigma par frame
    """
    mask_bool = mask.astype(bool)
    coeff     = np.sqrt(np.pi / 2.0)
    sigmas    = []
    for frame in frames:
        lap   = cv2.Laplacian(frame, cv2.CV_64F)
        roi   = lap[mask_bool]
        sigma = np.sqrt(np.mean(roi ** 2) / 6.0) * coeff
        sigmas.append(float(sigma))
    arr = np.array(sigmas)
    return {
        'sigma_mean':  float(np.mean(arr)),
        'sigma_std':   float(np.std(arr)),
        'sigma_curve': arr,
    }


def estimate_snr(frames: list[np.ndarray], mask: np.ndarray) -> dict:
    """
    Détection des vaisseaux par seuillage Otsu dans le masque circulaire.
    SNR = intensité_moyenne_vaisseaux / std_fond_proche_des_vaisseaux.
    Retourne :
    - 'snr_mean'    : float
    - 'snr_curve'   : np.ndarray
    - 'vessel_mask' : np.ndarray (bool, masque moyen des vaisseaux détectés)
    """
    mask_bool = mask.astype(bool)
    kernel    = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (2 * SNR_DILATION_PX + 1, 2 * SNR_DILATION_PX + 1)
    )
    snrs:         list[float]       = []
    vessel_acc    = np.zeros(frames[0].shape, dtype=np.float32)

    for frame in frames:
        roi_u8     = np.clip(frame[mask_bool], 0, 255).astype(np.uint8).reshape(-1, 1)
        thresh, _  = cv2.threshold(roi_u8, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        vessel     = (frame >= float(thresh)) & mask_bool
        dilated    = cv2.dilate(vessel.astype(np.uint8), kernel).astype(bool)
        background = dilated & ~vessel & mask_bool

        v_mean = float(np.mean(frame[vessel]))     if vessel.any()     else np.nan
        bg_std = float(np.std(frame[background]))  if background.any() else np.nan

        if not np.isnan(v_mean) and not np.isnan(bg_std) and bg_std > 0:
            snrs.append(v_mean / bg_std)
        else:
            snrs.append(np.nan)
        vessel_acc += vessel.astype(np.float32)

    arr = np.array(snrs)
    return {
        'snr_mean':    float(np.nanmean(arr)),
        'snr_curve':   arr,
        'vessel_mask': (vessel_acc / len(frames)) > 0.5,
    }


def estimate_temporal_noise(frames: list[np.ndarray], mask: np.ndarray) -> dict:
    """
    Pour chaque paire de frames consécutives dans la liste :
        diff = frame(t+1) - frame(t)  dans le masque
        std_diff = std(diff)
    Retourne :
    - 'temporal_std_mean'  : float
    - 'temporal_std_curve' : np.ndarray
    """
    mask_bool = mask.astype(bool)
    stds: list[float] = []
    for i in range(len(frames) - 1):
        diff = frames[i + 1] - frames[i]
        stds.append(float(np.std(diff[mask_bool])))
    arr = np.array(stds) if stds else np.array([0.0])
    return {
        'temporal_std_mean':  float(np.mean(arr)),
        'temporal_std_curve': arr,
    }


def estimate_power_spectrum(frames: list[np.ndarray], mask: np.ndarray) -> dict:
    """
    FFT 2D moyenne + profil radial moyen.
    Retourne :
    - 'mean_fft'        : np.ndarray, FFT 2D moyenne (magnitudes²)
    - 'radial_profile'  : np.ndarray, profil radial
    - 'high_freq_power' : float, puissance normalisée > HIGH_FREQ_THRESH × Nyquist
    - 'low_freq_power'  : float, puissance normalisée < LOW_FREQ_THRESH  × Nyquist
    - 'noise_floor'     : float, niveau plancher HF
    """
    mask_bool = mask.astype(bool)
    H, W      = frames[0].shape
    fft_acc   = np.zeros((H, W), dtype=np.float64)

    for frame in frames:
        f            = frame.copy()
        f[~mask_bool] = 0.0
        fft_acc      += np.abs(np.fft.fftshift(np.fft.fft2(f))) ** 2
    mean_fft = fft_acc / len(frames)

    cy, cx = H // 2, W // 2
    y, x   = np.indices((H, W))
    r_int  = np.sqrt((x - cx) ** 2 + (y - cy) ** 2).astype(int)
    r_max  = min(H, W) // 2

    radial = np.zeros(r_max, dtype=np.float64)
    cnt    = np.zeros(r_max, dtype=np.int64)
    valid  = r_int < r_max
    np.add.at(radial, r_int[valid], mean_fft[valid])
    np.add.at(cnt,    r_int[valid], 1)
    nonzero         = cnt > 0
    radial[nonzero] /= cnt[nonzero]

    freqs      = np.arange(r_max) / max(H, W)   # cycles/pixel, Nyquist ≈ 0.5
    nyquist    = 0.5
    high_mask  = freqs > HIGH_FREQ_THRESH * nyquist
    low_mask   = freqs < LOW_FREQ_THRESH  * nyquist
    total_p    = radial.sum() + 1e-12

    return {
        'mean_fft':        mean_fft,
        'radial_profile':  radial,
        '_freqs':          freqs,
        'high_freq_power': float(radial[high_mask].sum() / total_p),
        'low_freq_power':  float(radial[low_mask].sum()  / total_p),
        'noise_floor':     float(radial[high_mask].mean()) if high_mask.any() else 0.0,
    }


def compute_noise_map(
    frames: list[np.ndarray], mask: np.ndarray, tile_size: int = TILE_SIZE
) -> np.ndarray:
    """
    Carte H×W du sigma local par tuiles tile_size×tile_size (méthode Laplacien).
    """
    H, W      = frames[0].shape
    mask_bool = mask.astype(bool)
    noise_map = np.zeros((H, W), dtype=np.float32)
    coeff     = np.sqrt(np.pi / 2.0)

    tiles_y    = list(range(0, H, tile_size))
    tiles_x    = list(range(0, W, tile_size))
    total_tiles = len(tiles_y) * len(tiles_x)
    tile_count  = 0

    for ty in tiles_y:
        for tx in tiles_x:
            tile_count += 1
            print(f"\r    {_bar(tile_count, total_tiles)}", end="", flush=True)

            tile_m = mask_bool[ty:ty + tile_size, tx:tx + tile_size]
            if tile_m.sum() < max(4, tile_m.size * 0.1):
                continue

            tile_sigmas: list[float] = []
            for frame in frames:
                tile = frame[ty:ty + tile_size, tx:tx + tile_size]
                if tile.shape[0] < 3 or tile.shape[1] < 3:
                    continue
                lap = cv2.Laplacian(tile, cv2.CV_64F)
                roi = lap[tile_m]
                if len(roi) < 4:
                    continue
                tile_sigmas.append(
                    float(np.sqrt(np.mean(roi ** 2) / 6.0) * coeff)
                )
            if tile_sigmas:
                noise_map[ty:ty + tile_size, tx:tx + tile_size] = np.mean(tile_sigmas)
    print()
    return noise_map


# ─────────────────────────────────────────────────────────────────────────────
# Visualisations
# ─────────────────────────────────────────────────────────────────────────────

def save_metrics_table(metrics_a: dict, metrics_b: dict, output_dir: Path) -> None:
    """noise_metrics_comparison.png + .json"""
    rows = [
        ("Sigma Laplacien",        metrics_a['sigma']['sigma_mean'],
                                    metrics_b['sigma']['sigma_mean'],       False),
        ("SNR vaisseaux",           metrics_a['snr']['snr_mean'],
                                    metrics_b['snr']['snr_mean'],            True),
        ("Std diff temporelle",     metrics_a['temporal']['temporal_std_mean'],
                                    metrics_b['temporal']['temporal_std_mean'], False),
        ("Puissance HF normalisée", metrics_a['fft']['high_freq_power'],
                                    metrics_b['fft']['high_freq_power'],    False),
    ]

    fig, ax = plt.subplots(figsize=(11, 3))
    ax.axis("off")

    json_data: dict = {}
    cell_text: list[list[str]] = []
    for label, va, vb, higher_better in rows:
        gain_pct = (vb - va) / abs(va) * 100 if va != 0 else 0.0
        is_ok    = (gain_pct > 0) if higher_better else (gain_pct < 0)
        icon     = "✅" if is_ok else "⚠️"
        sign     = "+" if gain_pct >= 0 else ""
        cell_text.append([label, f"{va:.3f}", f"{vb:.3f}", f"{sign}{gain_pct:.0f}% {icon}"])
        json_data[label] = {"video_A": round(va, 4), "video_B": round(vb, 4),
                            "gain_pct": round(gain_pct, 1)}

    t = ax.table(
        cellText=cell_text,
        colLabels=["Métrique", "Vidéo A", "Vidéo B", "Gain (%)"],
        loc="center",
        cellLoc="center",
    )
    t.auto_set_font_size(False)
    t.set_fontsize(11)
    t.scale(1, 2.5)
    for j in range(4):
        t[0, j].set_facecolor("#2c3e50")
        t[0, j].set_text_props(color="white", fontweight="bold")
    for i in range(1, len(rows) + 1):
        for j in range(4):
            t[i, j].set_facecolor("#f8f9fa" if i % 2 == 0 else "white")

    plt.tight_layout()
    out_png = output_dir / "noise_metrics_comparison.png"
    plt.savefig(str(out_png), dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  → {out_png.name}")

    out_json = output_dir / "noise_metrics_comparison.json"
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(json_data, f, indent=2, ensure_ascii=False)
    print(f"  → {out_json.name}")


def save_sigma_curves(
    metrics_a: dict, metrics_b: dict, name_a: str, name_b: str, output_dir: Path
) -> None:
    """sigma_curves_comparison.png — 3 panneaux superposant vidéo A et B."""
    datasets = [
        ("Sigma Laplacien",
         metrics_a['sigma']['sigma_curve'],        metrics_b['sigma']['sigma_curve']),
        ("SNR vaisseaux",
         metrics_a['snr']['snr_curve'],             metrics_b['snr']['snr_curve']),
        ("Std temporelle",
         metrics_a['temporal']['temporal_std_curve'], metrics_b['temporal']['temporal_std_curve']),
    ]
    color_a, color_b = "#e74c3c", "#2980b9"

    fig, axes = plt.subplots(3, 1, figsize=(12, 9))
    for ax, (title, ca, cb) in zip(axes, datasets):
        ax.plot(np.arange(len(ca)), ca, color=color_a, linewidth=1.8,
                label=name_a, alpha=0.9)
        ax.plot(np.arange(len(cb)), cb, color=color_b, linewidth=1.8,
                label=name_b, alpha=0.9, linestyle="--")
        ax.set_ylabel(title, fontsize=9)
        ax.legend(fontsize=8, loc="upper right")
        ax.grid(alpha=0.3)
        ax.tick_params(labelsize=8)
    axes[-1].set_xlabel("Index frame échantillon", fontsize=9)
    plt.suptitle("Évolution des métriques de bruit par frame",
                 fontsize=11, fontweight="bold")
    plt.tight_layout()

    out = output_dir / "sigma_curves_comparison.png"
    plt.savefig(str(out), dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  → {out.name}")


def save_noise_maps(
    map_a: np.ndarray, map_b: np.ndarray, name_a: str, name_b: str, output_dir: Path
) -> None:
    """noise_map_comparison.png — cartes côte à côte + différence."""
    vmax     = max(map_a.max(), map_b.max(), 1e-6)
    diff_map = map_a - map_b
    absmax   = max(np.abs(diff_map).max(), 1e-6)

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    for ax, img, title in [
        (axes[0], map_a,   f"Bruit local — {name_a}"),
        (axes[1], map_b,   f"Bruit local — {name_b}"),
    ]:
        im = ax.imshow(img, cmap="hot", vmin=0, vmax=vmax)
        ax.set_title(title, fontsize=9)
        ax.axis("off")
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="sigma")

    im_d = axes[2].imshow(diff_map, cmap="RdBu_r", vmin=-absmax, vmax=absmax)
    axes[2].set_title("Différence (A − B)\nRouge = gain débruitage", fontsize=9)
    axes[2].axis("off")
    plt.colorbar(im_d, ax=axes[2], fraction=0.046, pad=0.04, label="sigma A−B")

    plt.suptitle(f"Cartes de bruit local (sigma Laplacien, tuiles {TILE_SIZE}×{TILE_SIZE})",
                 fontsize=11, fontweight="bold")
    plt.tight_layout()

    out = output_dir / "noise_map_comparison.png"
    plt.savefig(str(out), dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  → {out.name}")


def save_temporal_diff_viz(
    pairs_a: list[tuple[np.ndarray, np.ndarray]],
    pairs_b: list[tuple[np.ndarray, np.ndarray]],
    name_a: str, name_b: str,
    mask_bool: np.ndarray,
    output_dir: Path,
) -> None:
    """temporal_diff_comparison.png — grille 5 paires × 3 colonnes."""
    n = min(len(pairs_a), len(pairs_b), TEMPORAL_VIZ_N)
    if n == 0:
        print("  AVERT : aucune paire temporelle disponible")
        return

    fig, axes = plt.subplots(n, 3, figsize=(13, 3.4 * n),
                              gridspec_kw={"wspace": 0.03, "hspace": 0.06})
    if n == 1:
        axes = axes[np.newaxis, :]

    col_titles = [f"diff×10 — {name_a}", f"diff×10 — {name_b}", "Différence des diffs"]
    for j, title in enumerate(col_titles):
        axes[0, j].set_title(title, fontsize=9, fontweight="bold")

    def _masked(img: np.ndarray) -> np.ndarray:
        out = img.copy()
        out[~mask_bool] = 0.0
        return out

    for i in range(n):
        fa0, fa1 = pairs_a[i]
        fb0, fb1 = pairs_b[i]
        diff_a   = _masked((fa1 - fa0) * 10.0)
        diff_b   = _masked((fb1 - fb0) * 10.0)
        diff_d   = _masked(diff_a - diff_b)

        for j, (panel, cmap) in enumerate([
            (diff_a, "gray"), (diff_b, "gray"), (diff_d, "RdBu_r")
        ]):
            vm = max(np.abs(panel).max(), 1.0)
            axes[i, j].imshow(panel, cmap=cmap, vmin=-vm, vmax=vm)
            axes[i, j].axis("off")
        axes[i, 0].set_ylabel(f"Paire {i + 1}", fontsize=8)

    plt.suptitle(
        "Différences temporelles (frame t+1 − t) ×10 — grain fort = bruit élevé",
        fontsize=10, fontweight="bold",
    )
    out = output_dir / "temporal_diff_comparison.png"
    plt.savefig(str(out), dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  → {out.name}")


def save_fft_comparison(
    fft_a: dict, fft_b: dict, name_a: str, name_b: str, output_dir: Path
) -> None:
    """fft_comparison.png — spectres 2D + profil radial annoté."""
    fig = plt.figure(figsize=(17, 5))
    gs  = fig.add_gridspec(1, 3, wspace=0.08)

    for i, (fft_d, name) in enumerate([(fft_a, name_a), (fft_b, name_b)]):
        ax = fig.add_subplot(gs[i])
        ax.imshow(np.log1p(fft_d['mean_fft']), cmap="inferno")
        ax.set_title(f"Spectre FFT (log) — {name}", fontsize=9)
        ax.axis("off")

    ax3    = fig.add_subplot(gs[2])
    freqs  = fft_a['_freqs']
    nyq    = 0.5
    ax3.semilogy(freqs, fft_a['radial_profile'] + 1,
                 color="#e74c3c", linewidth=1.8, label=name_a)
    ax3.semilogy(fft_b['_freqs'], fft_b['radial_profile'] + 1,
                 color="#2980b9", linewidth=1.8, label=name_b, linestyle="--")
    ax3.axvspan(HIGH_FREQ_THRESH * nyq, freqs[-1],
                alpha=0.13, color="orange", label=f"Bruit HF (>{HIGH_FREQ_THRESH} Nyq)")
    ax3.axvspan(0, LOW_FREQ_THRESH * nyq,
                alpha=0.13, color="green",  label=f"Signal BF (<{LOW_FREQ_THRESH} Nyq)")
    ax3.set_xlabel("Fréquence (cycles/pixel)", fontsize=9)
    ax3.set_ylabel("Puissance (log)", fontsize=9)
    ax3.set_title("Profil radial moyen", fontsize=9)
    ax3.legend(fontsize=7)
    ax3.grid(alpha=0.3)
    ax3.tick_params(labelsize=8)

    plt.suptitle("Analyse spectrale — FFT 2D moyenne", fontsize=11, fontweight="bold")
    out = output_dir / "fft_comparison.png"
    plt.savefig(str(out), dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  → {out.name}")


def _vessel_crop_center(
    vessel_mask: np.ndarray, crop: int, shape: tuple[int, int]
) -> tuple[int, int]:
    """Centre du crop avec la plus haute densité de vaisseaux (sliding window)."""
    H, W   = shape
    half   = crop // 2
    step   = max(crop // 4, 1)
    best_score, best_cy, best_cx = -1, H // 2, W // 2
    for cy in range(half, H - half, step):
        for cx in range(half, W - half, step):
            score = int(vessel_mask[cy - half:cy + half, cx - half:cx + half].sum())
            if score > best_score:
                best_score, best_cy, best_cx = score, cy, cx
    return best_cy, best_cx


def save_frame_crops(
    frames_a: list[np.ndarray], frames_b: list[np.ndarray],
    vessel_mask: np.ndarray, mask_bool: np.ndarray,
    name_a: str, name_b: str, output_dir: Path,
) -> None:
    """frame_comparison_crops.png — 5 frames × 5 colonnes."""
    n       = min(len(frames_a), len(frames_b), 5)
    indices = np.linspace(0, n - 1, n, dtype=int)
    half    = CROP_SIZE // 2
    cy, cx  = _vessel_crop_center(vessel_mask, CROP_SIZE, frames_a[0].shape)
    H, W    = frames_a[0].shape

    fig, axes = plt.subplots(n, 5, figsize=(19, 3.6 * n),
                              gridspec_kw={"wspace": 0.04, "hspace": 0.06})
    if n == 1:
        axes = axes[np.newaxis, :]

    col_titles = [f"Frame — {name_a}", f"Frame — {name_b}",
                  "Diff (A−B)×5", f"Crop — {name_a}", f"Crop — {name_b}"]
    for j, title in enumerate(col_titles):
        axes[0, j].set_title(title, fontsize=8, fontweight="bold")

    for row, idx in enumerate(indices):
        fa    = frames_a[idx]
        fb    = frames_b[idx]
        diff  = np.clip((fa - fb) * 5.0 + 128.0, 0.0, 255.0)
        y0, y1 = max(cy - half, 0), min(cy + half, H)
        x0, x1 = max(cx - half, 0), min(cx + half, W)
        crop_a = fa[y0:y1, x0:x1]
        crop_b = fb[y0:y1, x0:x1]
        for col, panel in enumerate([fa, fb, diff, crop_a, crop_b]):
            axes[row, col].imshow(panel, cmap="gray", vmin=0, vmax=255,
                                  interpolation="nearest")
            axes[row, col].axis("off")
        axes[row, 0].set_ylabel(f"Éch. {idx}", fontsize=7)

    plt.suptitle("Comparaison frames — zone vasculaire", fontsize=10, fontweight="bold")
    out = output_dir / "frame_comparison_crops.png"
    plt.savefig(str(out), dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  → {out.name}")


# ─────────────────────────────────────────────────────────────────────────────
# Rapport terminal
# ─────────────────────────────────────────────────────────────────────────────

def print_report(
    metrics_a: dict, metrics_b: dict,
    name_a: str, name_b: str,
    total_a: int, total_b: int,
    output_dir: str,
) -> None:
    rows = [
        ("Sigma Laplacien",        metrics_a['sigma']['sigma_mean'],
                                    metrics_b['sigma']['sigma_mean'],       False),
        ("SNR vaisseaux",           metrics_a['snr']['snr_mean'],
                                    metrics_b['snr']['snr_mean'],            True),
        ("Std temporelle",          metrics_a['temporal']['temporal_std_mean'],
                                    metrics_b['temporal']['temporal_std_mean'], False),
        ("Puissance HF",            metrics_a['fft']['high_freq_power'],
                                    metrics_b['fft']['high_freq_power'],    False),
    ]
    sep    = "─" * 65
    all_ok = True

    print(f"\n{sep}")
    print("  RAPPORT COMPARAISON DE BRUIT")
    print(f"  Vidéo A : {name_a:<40} ({total_a} frames)")
    print(f"  Vidéo B : {name_b:<40} ({total_b} frames)")
    print(sep)
    print(f"  {'Métrique':<28}  {'Vidéo A':>8}  {'Vidéo B':>8}  {'Gain':>9}")
    print(sep)

    for label, va, vb, higher_better in rows:
        gain_pct = (vb - va) / abs(va) * 100 if va != 0 else 0.0
        is_ok    = (gain_pct > 0) if higher_better else (gain_pct < 0)
        if not is_ok:
            all_ok = False
        icon = "✅" if is_ok else "⚠️"
        sign = "+" if gain_pct >= 0 else ""
        print(f"  {label:<28}  {va:>8.2f}  {vb:>8.2f}  {sign}{gain_pct:>5.0f}% {icon}")

    print(sep)
    verdict = (
        "VERDICT : débruitage efficace sur toutes les métriques ✅"
        if all_ok else
        "VERDICT : résultats mitigés — vérifier les métriques ⚠️"
    )
    print(f"  {verdict}")
    print(sep)
    print(f"  Visualisations sauvegardées dans : {output_dir}")
    print(f"{sep}\n")


# ─────────────────────────────────────────────────────────────────────────────
# Orchestration principale
# ─────────────────────────────────────────────────────────────────────────────

def run_comparison(path_a: str, path_b: str, mask_path: str, output_dir: str) -> None:
    out    = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    name_a = Path(path_a).name
    name_b = Path(path_b).name

    _step("1/7 — Chargement du masque")
    mask      = load_mask(mask_path)
    mask_bool = mask.astype(bool)
    print(f"  Masque : {mask_path}  ({mask.shape[0]}×{mask.shape[1]})")

    _step("2/7 — Chargement des frames")
    try:
        frames_a, total_a = load_sample_frames(path_a)
    except RuntimeError as e:
        sys.exit(f"ERREUR vidéo A : {e}")
    try:
        frames_b, total_b = load_sample_frames(path_b)
    except RuntimeError as e:
        sys.exit(f"ERREUR vidéo B : {e}")

    n        = min(len(frames_a), len(frames_b))
    frames_a = frames_a[:n]
    frames_b = frames_b[:n]
    if total_a != total_b:
        print(f"  AVERT : nombre de frames différent ({total_a} vs {total_b}) "
              f"— utilisation du minimum ({n} échantillons)")

    _step("3/7 — Métriques vidéo A")
    sigma_a = estimate_noise_sigma(frames_a, mask)
    print(f"  sigma_mean = {sigma_a['sigma_mean']:.3f}")
    snr_a   = estimate_snr(frames_a, mask)
    print(f"  snr_mean   = {snr_a['snr_mean']:.2f}")
    temp_a  = estimate_temporal_noise(frames_a, mask)
    print(f"  temporal_std_mean = {temp_a['temporal_std_mean']:.3f}")
    fft_a   = estimate_power_spectrum(frames_a, mask)
    print(f"  high_freq_power = {fft_a['high_freq_power']:.4f}")

    _step("4/7 — Métriques vidéo B")
    sigma_b = estimate_noise_sigma(frames_b, mask)
    print(f"  sigma_mean = {sigma_b['sigma_mean']:.3f}")
    snr_b   = estimate_snr(frames_b, mask)
    print(f"  snr_mean   = {snr_b['snr_mean']:.2f}")
    temp_b  = estimate_temporal_noise(frames_b, mask)
    print(f"  temporal_std_mean = {temp_b['temporal_std_mean']:.3f}")
    fft_b   = estimate_power_spectrum(frames_b, mask)
    print(f"  high_freq_power = {fft_b['high_freq_power']:.4f}")

    metrics_a = {'sigma': sigma_a, 'snr': snr_a, 'temporal': temp_a, 'fft': fft_a}
    metrics_b = {'sigma': sigma_b, 'snr': snr_b, 'temporal': temp_b, 'fft': fft_b}

    _step("5/7 — Cartes de bruit locales")
    print("  Carte vidéo A...")
    map_a = compute_noise_map(frames_a, mask)
    print("  Carte vidéo B...")
    map_b = compute_noise_map(frames_b, mask)

    _step("6/7 — Chargement paires temporelles")
    pairs_a = load_consecutive_pairs(path_a, TEMPORAL_VIZ_N, total_a)
    pairs_b = load_consecutive_pairs(path_b, TEMPORAL_VIZ_N, total_b)
    print(f"  Paires chargées : A={len(pairs_a)}, B={len(pairs_b)}")

    _step("7/7 — Génération des visualisations")
    save_metrics_table(metrics_a, metrics_b, out)
    save_sigma_curves(metrics_a, metrics_b, name_a, name_b, out)
    save_noise_maps(map_a, map_b, name_a, name_b, out)
    save_temporal_diff_viz(pairs_a, pairs_b, name_a, name_b, mask_bool, out)
    save_fft_comparison(fft_a, fft_b, name_a, name_b, out)
    save_frame_crops(frames_a, frames_b, snr_a['vessel_mask'], mask_bool,
                     name_a, name_b, out)

    print_report(metrics_a, metrics_b, name_a, name_b, total_a, total_b, output_dir)


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Comparaison objective du bruit entre deux vidéos vasculaires.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Exemple :\n"
            "  python compare_noise.py orig.avi denoised.avi mask.png\n"
            "  python compare_noise.py orig.avi denoised.avi mask.png --output_dir résultats/"
        ),
    )
    parser.add_argument("video_a",      help="Vidéo de référence (ex. originale).")
    parser.add_argument("video_b",      help="Vidéo à comparer (ex. débruitée).")
    parser.add_argument("mask",         help="Masque circulaire (.png).")
    parser.add_argument("--output_dir", default="noise_comparison",
                        help="Dossier de sortie (défaut : noise_comparison/).")
    args = parser.parse_args()

    for fpath, label in [(args.video_a, "video_a"), (args.video_b, "video_b"),
                         (args.mask, "mask")]:
        if not Path(fpath).exists():
            sys.exit(f"ERREUR : fichier introuvable ({label}) : {fpath}")

    print(f"\n  Vidéo A    : {args.video_a}")
    print(f"  Vidéo B    : {args.video_b}")
    print(f"  Masque     : {args.mask}")
    print(f"  Sortie     : {args.output_dir}")

    run_comparison(args.video_a, args.video_b, args.mask, args.output_dir)


if __name__ == "__main__":
    main()
