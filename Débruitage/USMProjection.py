#!/usr/bin/env python3
"""
Pipeline USM → Projection (sans prétraitement).

Pipeline : vidéo brute → upscale → USM par frame → projection temporelle.
Équivalent au cas 5 de FullComparison.py, mais sans aucun prétraitement préalable.

Supporte une vidéo unique ou un dossier entier de vidéos.

Usage :
  python USMProjection.py video.avi
  python USMProjection.py video.avi --output-dir resultats/
  python USMProjection.py dossier/
  python USMProjection.py dossier/ --output-dir resultats/
  python USMProjection.py video.avi --mask mask.png --sigma 2.0 --strength 2.0

Dépendances :
  pip install opencv-python numpy scikit-image matplotlib
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Iterator

import cv2
import numpy as np
from skimage.filters import unsharp_mask as ski_usm

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# ─────────────────────────────────────────────────────────────────────────────
# Paramètres par défaut
# ─────────────────────────────────────────────────────────────────────────────

USM_SIGMA    = 2.0
USM_STRENGTH = 2.0
TARGET_SIZE  = 1024
CHUNK_SIZE   = 50
RESERVOIR    = 200
PERCENTILES  = [75, 85, 90, 95]


# ─────────────────────────────────────────────────────────────────────────────
# Masque circulaire
# ─────────────────────────────────────────────────────────────────────────────


def _odd(n: int) -> int:
    return n if n % 2 == 1 else n + 1


def detect_circular_mask(
    video_path: str | Path,
    n_sample: int = 20,
    margin_px: int = 5,
) -> np.ndarray:
    """
    Détecte le masque circulaire du champ visuel à partir de la vidéo.
    Retourne un tableau uint8 (H×W) avec 255 dans la zone utile, 0 en dehors.
    Lève ValueError si aucun cercle plausible n'est trouvé.
    """
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise FileNotFoundError(f"Impossible d'ouvrir : {video_path}")

    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    h     = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    w     = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    n_sample = min(n_sample, total)
    indices  = np.linspace(0, total - 1, n_sample, dtype=int)

    acc, count = np.zeros((h, w), dtype=np.float64), 0
    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, float(idx))
        ret, frame = cap.read()
        if not ret:
            continue
        if frame.ndim == 3:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        acc += frame.astype(np.float64)
        count += 1
    cap.release()

    if count == 0:
        raise ValueError("Aucune frame lisible dans la vidéo.")

    mean_frame = (acc / count).astype(np.uint8)
    blur_k     = _odd(min(h, w) // 8)
    blurred    = cv2.GaussianBlur(mean_frame, (blur_k, blur_k), 0)
    _, binary  = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    close_k = _odd(min(h, w) // 16)
    kernel  = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (close_k, close_k))
    binary  = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel, iterations=2)

    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        raise ValueError("Aucun contour trouvé pour la détection du masque.")

    largest         = max(contours, key=cv2.contourArea)
    (cx_f, cy_f), r = cv2.minEnclosingCircle(largest)
    cx, cy          = int(round(cx_f)), int(round(cy_f))
    coverage        = (np.pi * r ** 2) / (h * w)

    if coverage < 0.30:
        raise ValueError(
            f"Cercle détecté trop petit (couverture={coverage:.1%}). "
            "Vérifiez que la vidéo contient un disque optique circulaire visible."
        )
    if coverage > 0.95:
        raise ValueError(
            f"Cercle détecté trop grand (couverture={coverage:.1%}). "
            "Otsu a peut-être inclus du bruit de fond."
        )

    mask = np.zeros((h, w), dtype=np.uint8)
    cv2.circle(mask, (cx, cy), int(r), 255, thickness=-1)

    if margin_px > 0:
        ksz    = 2 * margin_px + 1
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ksz, ksz))
        mask   = cv2.erode(mask, kernel, iterations=1)

    return mask


def _load_mask(path: str | Path) -> np.ndarray:
    mask = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if mask is None:
        raise FileNotFoundError(f"Impossible de charger le masque : {path}")
    return mask


# ─────────────────────────────────────────────────────────────────────────────
# Utilitaires vidéo
# ─────────────────────────────────────────────────────────────────────────────


def _count_frames(path: str | Path) -> int:
    cap = cv2.VideoCapture(str(path))
    n   = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    return max(n, 0)


def _iter_chunks(
    path: str | Path,
    chunk_size: int = CHUNK_SIZE,
) -> Iterator[tuple[np.ndarray, int]]:
    """Génère (chunk_uint8, frames_lues_total) sans charger la vidéo entière."""
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise FileNotFoundError(f"Impossible d'ouvrir : {path}")
    buf: list[np.ndarray] = []
    count = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            if buf:
                yield np.stack(buf, axis=0), count
            break
        if frame.ndim == 3:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        buf.append(frame)
        count += 1
        if len(buf) == chunk_size:
            yield np.stack(buf, axis=0), count
            buf = []
    cap.release()


def _progress(current: int, total: int, prefix: str = "") -> None:
    pct    = current / max(total, 1)
    filled = int(40 * pct)
    bar    = "█" * filled + "░" * (40 - filled)
    print(f"\r  {prefix}[{bar}] {current}/{total} ({pct*100:.0f}%)",
          end="", flush=True)
    if current >= total:
        print()


# ─────────────────────────────────────────────────────────────────────────────
# Normalisation
# ─────────────────────────────────────────────────────────────────────────────


def _normalize_uint8(img: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Étire la dynamique dans le masque vers [0, 255] ; 0 hors masque."""
    mask_bool = mask.astype(bool)
    out = np.zeros(img.shape, dtype=np.uint8)
    if not mask_bool.any():
        return out
    vals = img[mask_bool]
    vmin, vmax = float(vals.min()), float(vals.max())
    if vmax > vmin:
        scaled = (img.astype(np.float64) - vmin) / (vmax - vmin) * 255.0
    else:
        scaled = np.zeros_like(img, dtype=np.float64)
    out[mask_bool] = np.clip(scaled[mask_bool], 0, 255).astype(np.uint8)
    return out


# ─────────────────────────────────────────────────────────────────────────────
# Cœur du traitement : upscale + USM + projection en une seule passe
# ─────────────────────────────────────────────────────────────────────────────


def process_video(
    video_path: str | Path,
    mask: np.ndarray | None,
    output_dir: str | Path,
    sigma: float     = USM_SIGMA,
    strength: float  = USM_STRENGTH,
    target_size: int = TARGET_SIZE,
    chunk_size: int  = CHUNK_SIZE,
    reservoir_size: int = RESERVOIR,
    percentiles: list[int] | None = None,
) -> dict:
    """
    Pipeline : upscale chaque frame → USM → accumulation pour projection temporelle.

    Les projections sont calculées à la résolution upscalée (target_size × target_size).
    Le masque est également upscalé si nécessaire.

    Retourne un dict avec les chemins des projections PNG sauvegardées.
    """
    if percentiles is None:
        percentiles = PERCENTILES

    video_path = Path(video_path)
    out        = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    total = _count_frames(video_path)
    t0    = time.perf_counter()

    # Upscale du masque si fourni
    if mask is not None:
        mask_up      = cv2.resize(mask, (target_size, target_size),
                                  interpolation=cv2.INTER_NEAREST)
        mask_bool_up = mask_up.astype(bool)
    else:
        mask_up      = np.ones((target_size, target_size), dtype=np.uint8) * 255
        mask_bool_up = np.ones((target_size, target_size), dtype=bool)

    print(f"\n  Pipeline : upscale {target_size}×{target_size}  →  USM (σ={sigma}, s={strength})  →  projection")
    print(f"  Vidéo   : {video_path.name}  ({total} frames)")
    print(f"  Sortie  : {out}")
    print()

    # Accumulateurs
    sum_acc:   np.ndarray | None = None   # float64 (H, W)
    max_acc:   np.ndarray | None = None   # float32 (H, W)
    reservoir: np.ndarray | None = None   # float32 (R, H, W)
    n_total = 0
    rng = np.random.default_rng(seed=42)

    for chunk_u8, count in _iter_chunks(video_path, chunk_size):
        n_chunk = chunk_u8.shape[0]

        # Upscale + USM sur chaque frame du chunk
        usm_frames: list[np.ndarray] = []
        for i in range(n_chunk):
            frame = cv2.resize(chunk_u8[i], (target_size, target_size),
                               interpolation=cv2.INTER_LANCZOS4)
            sharpened = ski_usm(frame, radius=sigma, amount=strength, preserve_range=True)
            usm_frames.append(np.clip(sharpened, 0, 255).astype(np.float32))

        chunk_up = np.stack(usm_frames, axis=0)  # (n_chunk, H, W) float32

        # Initialisation paresseuse
        if sum_acc is None:
            H, W    = target_size, target_size
            sum_acc  = np.zeros((H, W), dtype=np.float64)
            max_acc  = np.full((H, W), -np.inf, dtype=np.float32)
            reservoir = np.empty((reservoir_size, H, W), dtype=np.float32)

        # Accumulateurs
        sum_acc += chunk_up.sum(axis=0).astype(np.float64)
        np.maximum(max_acc, chunk_up.max(axis=0), out=max_acc)

        # Réservoir (Vitter Algorithm R) pour percentile / médiane
        for i in range(n_chunk):
            g = n_total + i
            if g < reservoir_size:
                reservoir[g] = chunk_up[i]
            else:
                j = int(rng.integers(0, g + 1))
                if j < reservoir_size:
                    reservoir[j] = chunk_up[i]

        n_total += n_chunk
        _progress(count, total, prefix="Frames ")

    _progress(total, total, prefix="Frames ")

    actual_res = min(n_total, reservoir_size)
    res        = reservoir[:actual_res]

    print(f"\n  Calcul des statistiques ({n_total} frames, réservoir={actual_res}) … ",
          end="", flush=True)

    mean_raw   = (sum_acc / n_total).astype(np.float32)
    median_raw = np.median(res, axis=0).astype(np.float32)

    projections: dict[str, np.ndarray] = {
        "mean":   _normalize_uint8(mean_raw,  mask_up),
        "max":    _normalize_uint8(max_acc,   mask_up),
        "median": _normalize_uint8(median_raw, mask_up),
    }
    for p in percentiles:
        pct_raw = np.percentile(res, p, axis=0).astype(np.float32)
        projections[f"percentile_{p}"] = _normalize_uint8(pct_raw, mask_up)

    elapsed = time.perf_counter() - t0
    print(f"done  ({elapsed:.1f} s)")

    # Sauvegarde PNG + TIFF
    saved: dict[str, Path] = {}
    for name, img in projections.items():
        p_png  = out / f"projection_{name}.png"
        p_tiff = out / f"projection_{name}.tiff"
        cv2.imwrite(str(p_png),  img)
        cv2.imwrite(str(p_tiff), img)
        saved[name] = p_png

    # Scores qualité
    scores = {name: _quality_score(img, mask_up) for name, img in projections.items()}
    best   = max(scores, key=lambda k: scores[k]["combined_score"])

    # JSON résumé
    json_path = out / "projection_scores.json"
    with open(json_path, "w") as fh:
        json.dump({
            "best_projection":   best,
            "usm_params":        {"sigma": sigma, "strength": strength},
            "target_size":       target_size,
            "n_frames":          n_total,
            "processing_time_s": round(elapsed, 2),
            "scores":            scores,
        }, fh, indent=2)

    # Grille de comparaison
    _comparison_grid(projections, mask_up, out / "projection_grid.png",
                     best=best)

    print(f"\n  ── Résultats ──────────────────────────────────────────────────")
    print(f"  Frames traitées   : {n_total}")
    print(f"  Meilleure proj.   : {best}  (score={scores[best]['combined_score']:.4f})")
    print(f"  PNG  (toutes)     : {out}/projection_*.png")
    print(f"  Grille            : {out}/projection_grid.png")
    print(f"  Scores JSON       : {json_path}")
    print(f"  Temps total       : {elapsed:.1f} s")

    return {
        "saved":         saved,
        "best":          best,
        "n_frames":      n_total,
        "elapsed_s":     round(elapsed, 2),
    }


# ─────────────────────────────────────────────────────────────────────────────
# Score qualité (contraste local, netteté, SNR)
# ─────────────────────────────────────────────────────────────────────────────


def _quality_score(img: np.ndarray, mask: np.ndarray) -> dict:
    from skimage.filters import laplace as ski_laplace

    mask_bool = mask.astype(bool)
    f         = img.astype(np.float64)
    tile      = 16
    H, W      = f.shape

    # Contraste local
    stds: list[float] = []
    for r in range(0, H - tile + 1, tile):
        for c in range(0, W - tile + 1, tile):
            tm = mask_bool[r:r + tile, c:c + tile]
            if tm.sum() >= (tile * tile) // 2:
                stds.append(float(f[r:r + tile, c:c + tile][tm].std()))
    local_contrast = float(np.mean(stds)) if stds else 0.0

    # Netteté (variance du laplacien)
    sharpness = float(ski_laplace(f)[mask_bool].var()) if mask_bool.any() else 0.0

    # SNR
    vals = f[mask_bool]
    if vals.size > 10:
        p20  = float(np.percentile(vals, 20))
        p80  = float(np.percentile(vals, 80))
        sig  = float(vals.std()) + 1e-9
        snr  = (vals[vals >= p80].mean() - vals[vals <= p20].mean()) / sig
    else:
        snr = 0.0

    combined = float(
        (max(local_contrast, 1e-9) * max(sharpness, 1e-9) * max(snr, 1e-9)) ** (1 / 3)
    )
    return {
        "local_contrast": round(local_contrast, 3),
        "sharpness":      round(sharpness,      3),
        "snr":            round(float(snr),     4),
        "combined_score": round(combined,       4),
    }


# ─────────────────────────────────────────────────────────────────────────────
# Grille de comparaison visuelle
# ─────────────────────────────────────────────────────────────────────────────


def _comparison_grid(
    projections: dict[str, np.ndarray],
    mask: np.ndarray,
    output_path: Path,
    best: str = "mean",
    crop_size: int = 200,
) -> None:
    def _sort_key(k: str) -> tuple:
        order = {"mean": 0, "median": 1, "max": 2}
        if k in order:
            return (order[k], 0)
        if k.startswith("percentile_"):
            return (3, int(k.split("_")[1]))
        return (4, 0)

    keys  = sorted(projections.keys(), key=_sort_key)
    n     = len(keys)
    H, W  = mask.shape[:2]
    cy    = max(0, (H - crop_size) // 2)
    cx    = max(0, (W - crop_size) // 2)

    fig, axes = plt.subplots(
        n, 2,
        figsize=(8, 3.2 * n),
        gridspec_kw={"hspace": 0.05, "wspace": 0.03},
    )
    if n == 1:
        axes = axes[np.newaxis, :]

    axes[0, 0].set_title("Projection complète", fontsize=9, fontweight="bold")
    axes[0, 1].set_title(f"Zoom central {crop_size}×{crop_size}", fontsize=9, fontweight="bold")

    for row, name in enumerate(keys):
        img  = projections[name]
        crop = img[cy:cy + crop_size, cx:cx + crop_size]
        marker = " ◀" if name == best else ""
        label  = f"{name}{marker}"

        for col, panel in enumerate([img, crop]):
            ax = axes[row, col]
            ax.imshow(panel, cmap="gray", vmin=0, vmax=255, interpolation="nearest")
            ax.axis("off")
        axes[row, 0].set_ylabel(label, fontsize=8, rotation=0, ha="right",
                                va="center", labelpad=70)

    plt.suptitle(
        "Projections temporelles — USM → Projection (sans prétraitement)\n"
        "◀ = meilleure projection (score combiné)",
        fontsize=9,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(str(output_path), dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Grille            : {output_path}")


# ─────────────────────────────────────────────────────────────────────────────
# Gestion masque : chargement ou détection automatique
# ─────────────────────────────────────────────────────────────────────────────


def _get_mask(
    video_path: Path,
    mask_arg: str | None,
    output_dir: Path,
    verbose: bool = True,
) -> np.ndarray | None:
    """
    Résolution du masque dans cet ordre :
      1. Chemin fourni via --mask
      2. mask.png à côté de la vidéo
      3. Détection automatique (cercle) + sauvegarde
      4. None si la détection échoue (traitement sur le cadre complet)
    """
    # 1. Masque explicite
    if mask_arg:
        p = Path(mask_arg)
        if not p.exists():
            sys.exit(f"ERREUR : masque introuvable : {p}")
        if verbose:
            print(f"  Masque          : {p}  (fourni)")
        return _load_mask(p)

    # 2. mask.png à côté
    candidate = video_path.parent / "mask.png"
    if candidate.exists():
        if verbose:
            print(f"  Masque          : {candidate}  (trouvé à côté de la vidéo)")
        return _load_mask(candidate)

    # 3. Détection automatique
    if verbose:
        print(f"  Masque          : auto-détection depuis la vidéo …", end="", flush=True)
    try:
        mask        = detect_circular_mask(video_path)
        mask_saved  = output_dir / f"{video_path.stem}_mask.png"
        output_dir.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(mask_saved), mask)
        if verbose:
            print(f" OK  (sauvegardé : {mask_saved.name})")
        return mask
    except Exception as exc:
        if verbose:
            print(f"\n  AVERTISSEMENT : détection masque échouée ({exc})")
            print("  → Traitement sur le cadre complet (sans masque).")
        return None


# ─────────────────────────────────────────────────────────────────────────────
# Traitement d'une vidéo ou d'un dossier
# ─────────────────────────────────────────────────────────────────────────────


def _step(title: str) -> None:
    print(f"\n{'─'*62}\n  {title}\n{'─'*62}")


def run_single(
    video_path: Path,
    output_dir: Path,
    mask_arg: str | None,
    sigma: float,
    strength: float,
    target_size: int,
    chunk_size: int,
    reservoir_size: int,
) -> None:
    _step(f"Traitement : {video_path.name}")
    print(f"  Vidéo           : {video_path}")
    print(f"  Sortie          : {output_dir}")
    print(f"  USM             : sigma={sigma}  strength={strength}")
    print(f"  Upscale         : {target_size}×{target_size} (Lanczos)")

    mask = _get_mask(video_path, mask_arg, output_dir)
    process_video(
        video_path     = video_path,
        mask           = mask,
        output_dir     = output_dir,
        sigma          = sigma,
        strength       = strength,
        target_size    = target_size,
        chunk_size     = chunk_size,
        reservoir_size = reservoir_size,
    )


def run_folder(
    folder: Path,
    output_dir: Path,
    mask_arg: str | None,
    sigma: float,
    strength: float,
    target_size: int,
    chunk_size: int,
    reservoir_size: int,
) -> None:
    videos = sorted(folder.glob("*.avi"))
    if not videos:
        sys.exit(f"ERREUR : aucune vidéo .avi dans {folder}")

    print(f"\n  Mode dossier : {len(videos)} vidéo(s)  →  {output_dir}\n")
    failed: list[str] = []
    t_global = time.perf_counter()

    for video_path in videos:
        vid_out = output_dir / video_path.stem
        try:
            run_single(video_path, vid_out, mask_arg, sigma, strength,
                       target_size, chunk_size, reservoir_size)
        except Exception as exc:
            print(f"\n  [ERREUR] {video_path.name} : {exc}")
            failed.append(video_path.name)

    elapsed = time.perf_counter() - t_global
    print(f"\n{'═'*62}")
    print(f"  DOSSIER TERMINÉ : {len(videos) - len(failed)}/{len(videos)} vidéo(s)  "
          f"en {elapsed:.0f} s")
    if failed:
        print(f"  Échecs  : {', '.join(failed)}")
    print(f"  Résultats → {output_dir}")
    print(f"{'═'*62}\n")


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Pipeline USM → Projection sans prétraitement.\n\n"
            "Applique directement : upscale Lanczos → USM par frame → projection temporelle.\n"
            "Équivalent au cas 5 de FullComparison.py.\n\n"
            "Dépendances : pip install opencv-python numpy scikit-image matplotlib"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Exemples :\n"
            "  python USMProjection.py video.avi\n"
            "  python USMProjection.py video.avi --output-dir resultats/\n"
            "  python USMProjection.py dossier/\n"
            "  python USMProjection.py dossier/ --output-dir resultats/\n"
            "  python USMProjection.py video.avi --mask mask.png --sigma 2.0 --strength 2.0\n"
            "  python USMProjection.py video.avi --target-size 512\n"
        ),
    )
    parser.add_argument(
        "input",
        help="Vidéo .avi ou dossier contenant des vidéos .avi.",
    )
    parser.add_argument(
        "--output-dir", default=None,
        help=(
            "Dossier de sortie.\n"
            "  Vidéo unique : défaut = <input>_usm_projection/\n"
            "  Dossier      : défaut = <input>_usm_projection/"
        ),
    )
    parser.add_argument(
        "--mask", default=None,
        help=(
            "Chemin vers un masque PNG à utiliser.\n"
            "Si absent, cherche mask.png à côté de la vidéo, puis auto-détecte."
        ),
    )
    parser.add_argument(
        "--sigma", type=float, default=USM_SIGMA,
        help=f"Rayon Gaussien USM en pixels (défaut : {USM_SIGMA}).",
    )
    parser.add_argument(
        "--strength", type=float, default=USM_STRENGTH,
        help=f"Intensité USM (défaut : {USM_STRENGTH}).",
    )
    parser.add_argument(
        "--target-size", type=int, default=TARGET_SIZE,
        help=f"Résolution de sortie après upscale (défaut : {TARGET_SIZE}).",
    )
    parser.add_argument(
        "--chunk-size", type=int, default=CHUNK_SIZE,
        help=f"Frames par chunk de traitement (défaut : {CHUNK_SIZE}).",
    )
    parser.add_argument(
        "--reservoir-size", type=int, default=RESERVOIR,
        help=f"Taille du réservoir pour percentile/médiane (défaut : {RESERVOIR}).",
    )
    args = parser.parse_args()

    inp = Path(args.input)
    if not inp.exists():
        sys.exit(f"ERREUR : chemin introuvable : {inp}")

    if args.output_dir:
        out_dir = Path(args.output_dir)
    else:
        out_dir = inp.parent / f"{inp.stem}_usm_projection"

    if inp.is_file():
        if inp.suffix.lower() != ".avi":
            sys.exit(f"ERREUR : seules les vidéos .avi sont supportées ({inp.suffix})")
        run_single(
            video_path     = inp,
            output_dir     = out_dir,
            mask_arg       = args.mask,
            sigma          = args.sigma,
            strength       = args.strength,
            target_size    = args.target_size,
            chunk_size     = args.chunk_size,
            reservoir_size = args.reservoir_size,
        )
        print(f"\n{'═'*62}")
        print(f"  TERMINÉ  →  {out_dir}")
        print(f"{'═'*62}\n")

    else:
        run_folder(
            folder         = inp,
            output_dir     = out_dir,
            mask_arg       = args.mask,
            sigma          = args.sigma,
            strength       = args.strength,
            target_size    = args.target_size,
            chunk_size     = args.chunk_size,
            reservoir_size = args.reservoir_size,
        )


if __name__ == "__main__":
    main()
