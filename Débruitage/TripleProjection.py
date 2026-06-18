#!/usr/bin/env python3
"""
Sextuple pipeline — vidéos + projections 1024 + grille comparative.
Sans prétraitement, sans masque.

Pipelines produits :
  A. vidéo brute → upscale 1024 → <stem>_base_1024.avi                                → proj_base.png
  B. vidéo brute → upscale 1024 → USM → <stem>_usm_1024.avi                           → proj_usm.png
  C. vidéo brute → N2N → upscale 1024 → USM → <stem>_n2n_usm_1024.avi                → proj_n2n_usm.png
  D. N2N upscalé → USM → CLAHE → <stem>_n2n_usm_clahe_1024.avi                        → proj_n2n_usm_clahe.png
  E. N2N upscalé → Frangi → USM → <stem>_n2n_frangi_usm_1024.avi                      → proj_n2n_frangi_usm.png
  F. N2N upscalé → projection → NLM → Frangi blend (70/30) → CLAHE                    → proj_n2n_nlm_frangi_clahe.png

  + comparison_grid.png : grille 6 colonnes

Usage :
  python TripleProjection.py video.avi
  python TripleProjection.py video.avi --output-dir resultats/
  python TripleProjection.py video.avi --n2n-video video_n2n.avi --output-dir resultats/
  python TripleProjection.py video.avi --checkpoint checkpoints/best.pth
  python TripleProjection.py video.avi --sigma 1.5 --strength 3.0 --target-size 1024

Dépendances :
  pip install opencv-python numpy scikit-image matplotlib
"""
from __future__ import annotations

import argparse
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import cv2
import numpy as np
from skimage.filters import unsharp_mask as ski_usm
from skimage.filters import frangi as ski_frangi
from skimage.restoration import denoise_nl_means, estimate_sigma

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parent))
from CLAHE import apply_clahe_usm_pipeline


SDIR = Path(__file__).resolve().parent

DEFAULT_TARGET_SIZE  = 1024
DEFAULT_USM_SIGMA    = 2.0
DEFAULT_USM_STRENGTH = 2.0
DEFAULT_CLAHE_CLIP   = 1.0
DEFAULT_CLAHE_TILE   = 16
DEFAULT_CLAHE_USM    = 0.5
DEFAULT_FRANGI_NLM_H = 0.8


# ─────────────────────────────────────────────────────────────────────────────
# Utilitaires
# ─────────────────────────────────────────────────────────────────────────────


def _step(title: str) -> None:
    print(f"\n{'─'*62}\n  {title}\n{'─'*62}")


def _progress(current: int, total: int, width: int = 45) -> None:
    pct    = current / max(total, 1)
    filled = int(width * pct)
    bar    = "█" * filled + "░" * (width - filled)
    print(f"\r  [{bar}] {current}/{total}", end="", flush=True)
    if current >= total:
        print()


def upscale_video(input_path: Path, output_path: Path, size: int) -> None:
    """Upscale toutes les frames à size×size (Lanczos), grayscale."""
    cap = cv2.VideoCapture(str(input_path))
    if not cap.isOpened():
        raise RuntimeError(f"Impossible d'ouvrir : {input_path}")
    fps    = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total  = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fourcc = cv2.VideoWriter_fourcc(*"XVID")
    writer: cv2.VideoWriter | None = None
    count  = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if frame.ndim == 3:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        up = cv2.resize(frame, (size, size), interpolation=cv2.INTER_LANCZOS4)
        if writer is None:
            writer = cv2.VideoWriter(str(output_path), fourcc, fps, (size, size), isColor=False)
        writer.write(up)
        count += 1
        _progress(count, total)
    cap.release()
    if writer:
        writer.release()
    print(f"  → {output_path.name}  ({count} frames, {size}×{size})")


def apply_usm_video(
    input_path: Path,
    output_path: Path,
    sigma: float,
    strength: float,
) -> None:
    """Applique USM frame à frame et sauvegarde la vidéo résultante."""
    cap = cv2.VideoCapture(str(input_path))
    if not cap.isOpened():
        raise RuntimeError(f"Impossible d'ouvrir : {input_path}")
    fps    = cap.get(cv2.CAP_PROP_FPS) or 30.0
    w      = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h      = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total  = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fourcc = cv2.VideoWriter_fourcc(*"XVID")
    writer = cv2.VideoWriter(str(output_path), fourcc, fps, (w, h), isColor=False)
    count  = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if frame.ndim == 3:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        sharpened = ski_usm(frame, radius=sigma, amount=strength, preserve_range=True)
        writer.write(np.clip(sharpened, 0, 255).astype(np.uint8))
        count += 1
        _progress(count, total)
    cap.release()
    writer.release()
    print(f"  → {output_path.name}  ({count} frames)")


def apply_frangi_usm_video(
    input_path: Path,
    output_path: Path,
    sigma: float,
    strength: float,
) -> None:
    """Applique Frangi vesselness (normalisé → [0,255]) puis USM frame à frame."""
    cap    = cv2.VideoCapture(str(input_path))
    if not cap.isOpened():
        raise RuntimeError(f"Impossible d'ouvrir : {input_path}")
    fps    = cap.get(cv2.CAP_PROP_FPS) or 30.0
    w      = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h      = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total  = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fourcc = cv2.VideoWriter_fourcc(*"XVID")
    writer = cv2.VideoWriter(str(output_path), fourcc, fps, (w, h), isColor=False)
    count  = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if frame.ndim == 3:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        frame_f = frame.astype(np.float64) / 255.0
        vessel  = ski_frangi(frame_f, sigmas=range(1, 4), black_ridges=False)
        vmax = vessel.max()
        if vmax > 0:
            vessel /= vmax
        vessel_u8 = (vessel * 255).clip(0, 255).astype(np.uint8)
        sharpened = ski_usm(vessel_u8, radius=sigma, amount=strength, preserve_range=True)
        writer.write(np.clip(sharpened, 0, 255).astype(np.uint8))
        count += 1
        _progress(count, total)
    cap.release()
    writer.release()
    print(f"  → {output_path.name}  ({count} frames)")


def n2n_proj_nlm_frangi_clahe(
    proj: np.ndarray,
    nlm_h_factor: float = 0.8,
    blend_frangi: float = 0.3,
    clahe_clip: float = 1.5,
    clahe_tile: int = 32,
) -> np.ndarray:
    """
    Pipeline image (sur la projection N2N) :
      NLM léger → Frangi vesselness → blend (70 % NLM + 30 % Frangi) → CLAHE
    """
    proj_f    = proj.astype(np.float32) / 255.0
    sigma_est = float(estimate_sigma(proj_f))
    denoised  = denoise_nl_means(
        proj_f, h=nlm_h_factor * sigma_est,
        fast_mode=True, patch_size=5, patch_distance=6,
    )
    denoised_u8 = (denoised * 255).clip(0, 255).astype(np.uint8)

    denoised_f64 = denoised.astype(np.float64)
    vessel = ski_frangi(denoised_f64, sigmas=range(1, 4), black_ridges=False)
    vmax = vessel.max()
    if vmax > 0:
        vessel /= vmax
    vessel_u8 = (vessel * 255).clip(0, 255).astype(np.uint8)

    blended = (
        (1.0 - blend_frangi) * denoised_u8.astype(np.float32)
        + blend_frangi * vessel_u8.astype(np.float32)
    ).clip(0, 255).astype(np.uint8)

    clahe_op = cv2.createCLAHE(clipLimit=clahe_clip, tileGridSize=(clahe_tile, clahe_tile))
    return clahe_op.apply(blended)


def mean_projection(video_path: Path, size: int) -> np.ndarray:
    """
    Projection temporelle moyenne.
    Upscale Lanczos à size×size si la vidéo n'est pas déjà à cette résolution.
    Retourne un tableau uint8 (size, size).
    """
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Impossible d'ouvrir : {video_path}")
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    acc   = np.zeros((size, size), dtype=np.float64)
    count = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if frame.ndim == 3:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        fh, fw = frame.shape[:2]
        if fh != size or fw != size:
            frame = cv2.resize(frame, (size, size), interpolation=cv2.INTER_LANCZOS4)
        acc  += frame.astype(np.float64)
        count += 1
        _progress(count, max(total, 1))
    cap.release()
    if count == 0:
        raise RuntimeError(f"Aucune frame lue dans {video_path}")
    return np.clip(acc / count, 0, 255).astype(np.uint8)


def _find_checkpoint() -> str | None:
    """Cherche un checkpoint N2N dans les emplacements habituels."""
    try:
        import yaml as _yaml
        with open(SDIR / "config.yaml") as f:
            cfg = _yaml.safe_load(f)
        rel = cfg.get("inference", {}).get("checkpoint", "")
        if rel:
            abs_p = (SDIR.parent / rel).resolve()
            if abs_p.is_file():
                return str(abs_p)
    except Exception:
        pass
    for candidate in [
        SDIR / "checkpoints" / "best_model.pth",
        SDIR / "checkpoints" / "last.pth",
        SDIR.parent / "checkpoints" / "best_model.pth",
        SDIR.parent / "checkpoints" / "last.pth",
        SDIR / "best_model.pth",
        SDIR / "last.pth",
    ]:
        if candidate.is_file():
            return str(candidate)
    return None


def run_n2n(input_path: Path, output_path: Path, checkpoint: str) -> bool:
    """
    Lance inference.py en sous-processus avec un masque blanc temporaire
    adapté à la résolution de la vidéo d'entrée.
    """
    cap = cv2.VideoCapture(str(input_path))
    w   = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h   = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()
    if w == 0 or h == 0:
        print(f"  ERREUR : impossible de lire les dimensions de {input_path}")
        return False

    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
        tmp_mask = Path(tmp.name)
    cv2.imwrite(str(tmp_mask), np.full((h, w), 255, dtype=np.uint8))

    print(f"  Lancement N2N : {input_path.name} → {output_path.name}")
    try:
        result = subprocess.run([
            sys.executable, str(SDIR / "inference.py"),
            "--video",      str(input_path),
            "--checkpoint", checkpoint,
            "--mask",       str(tmp_mask),
            "--output",     str(output_path),
        ])
        if result.returncode != 0:
            print(f"  ERREUR N2N (code {result.returncode})")
            return False
        return True
    finally:
        tmp_mask.unlink(missing_ok=True)


def comparison_grid(
    proj_base:        np.ndarray,
    proj_usm:         np.ndarray,
    proj_n2n_usm:     np.ndarray | None,
    proj_clahe:       np.ndarray | None,
    proj_frangi_usm:  np.ndarray | None,
    proj_nlm_frangi:  np.ndarray | None,
    output_path:      Path,
    sigma:            float,
    strength:         float,
    target_size:      int,
    crop_size:        int = 256,
) -> None:
    """Grille comparative : N colonnes × 2 lignes (image complète / zoom centré)."""
    cases: list[tuple[str, np.ndarray]] = [
        ("Base\n(projection)",    proj_base),
        ("USM\n(projection)",     proj_usm),
    ]
    if proj_n2n_usm is not None:
        cases.append(("N2N → USM\n(projection)", proj_n2n_usm))
    if proj_clahe is not None:
        cases.append(("N2N → USM → CLAHE\n(projection)", proj_clahe))
    if proj_frangi_usm is not None:
        cases.append(("N2N → Frangi → USM\n(projection)", proj_frangi_usm))
    if proj_nlm_frangi is not None:
        cases.append(("N2N → NLM → Frangi\n→ CLAHE\n(projection)", proj_nlm_frangi))

    n   = len(cases)
    fig, axes = plt.subplots(
        2, n,
        figsize=(4.5 * n, 10.0),
        gridspec_kw={"hspace": 0.06, "wspace": 0.03},
    )
    if n == 1:
        axes = axes.reshape(2, 1)

    for col, (label, img) in enumerate(cases):
        H, W = img.shape
        cy, cx = H // 2, W // 2
        half   = crop_size // 2
        crop   = img[max(cy - half, 0):min(cy + half, H),
                     max(cx - half, 0):min(cx + half, W)]

        axes[0, col].imshow(img,  cmap="gray", vmin=0, vmax=255, interpolation="nearest")
        axes[0, col].axis("off")
        axes[0, col].set_title(label, fontsize=9, fontweight="bold")

        axes[1, col].imshow(crop, cmap="gray", vmin=0, vmax=255, interpolation="nearest")
        axes[1, col].axis("off")

    axes[0, 0].set_ylabel(
        "Image complète", fontsize=9, rotation=90,
        ha="center", va="center", labelpad=60,
    )
    axes[1, 0].set_ylabel(
        f"Zoom {crop_size}×{crop_size}\n(centre)",
        fontsize=9, rotation=90, ha="center", va="center", labelpad=60,
    )

    plt.suptitle(
        "Comparaison des projections temporelles — sans masque\n"
        f"Résolution : {target_size}×{target_size}  |  "
        f"USM σ={sigma}  force={strength}",
        fontsize=10,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(str(output_path), dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  → {output_path.name}")


# ─────────────────────────────────────────────────────────────────────────────
# Orchestrateur
# ─────────────────────────────────────────────────────────────────────────────


def run(
    base_video:   Path,
    output_dir:   Path,
    n2n_video:    Path | None,
    checkpoint:   str | None,
    target_size:  int,
    sigma:        float,
    strength:     float,
    clahe_clip:   float = DEFAULT_CLAHE_CLIP,
    clahe_tile:   int   = DEFAULT_CLAHE_TILE,
    clahe_usm:    float = DEFAULT_CLAHE_USM,
    frangi_nlm_h: float = DEFAULT_FRANGI_NLM_H,
) -> None:
    t0  = time.perf_counter()
    out = output_dir
    out.mkdir(parents=True, exist_ok=True)

    print(f"\n{'═'*62}")
    print(f"  Triple Projection — Base / USM / N2N+USM")
    print(f"{'═'*62}")
    print(f"  Entrée     : {base_video}")
    print(f"  Sortie     : {out}")
    print(f"  Upscale    : {target_size}×{target_size}")
    print(f"  USM        : sigma={sigma}  strength={strength}")
    print(f"  N2N vidéo  : {n2n_video or '(à calculer)'}")
    print(f"  Checkpoint : {checkpoint or '(auto-detect)'}")
    print(f"  CLAHE      : clip={clahe_clip}  tile={clahe_tile}  usm={clahe_usm}")

    stem = base_video.stem

    # ── A. Vidéo de base upscalée ────────────────────────────────────────────
    _step(f"A — Upscale vidéo de base → {target_size}×{target_size}")
    p_base = out / f"{stem}_base_{target_size}.avi"
    if p_base.exists():
        print(f"  Cache : {p_base.name}")
    else:
        upscale_video(base_video, p_base, target_size)

    # ── B. Vidéo USM ─────────────────────────────────────────────────────────
    _step(f"B — USM (sigma={sigma}  strength={strength})")
    p_usm = out / f"{stem}_usm_{target_size}.avi"
    if p_usm.exists():
        print(f"  Cache : {p_usm.name}")
    else:
        apply_usm_video(p_base, p_usm, sigma, strength)

    # ── C. N2N → upscale → USM ───────────────────────────────────────────────
    _step("C — N2N → upscale → USM")
    p_n2n_up:  Path | None = None
    p_n2n_usm: Path | None = None

    n2n_src: Path | None = None
    if n2n_video is not None:
        if n2n_video.exists():
            n2n_src = n2n_video
            print(f"  Vidéo N2N fournie : {n2n_src}")
        else:
            print(f"  AVERTISSEMENT : --n2n-video introuvable ({n2n_video}) — calcul depuis checkpoint.")

    if n2n_src is None:
        ckpt = checkpoint or _find_checkpoint()
        if ckpt is None:
            print("  AVERTISSEMENT : aucun checkpoint N2N trouvé — pipeline N2N ignoré.")
            print("  → Lancer l'entraînement : python train.py")
            print("  → Ou préciser le chemin : --checkpoint checkpoints/best_model.pth")
        else:
            p_n2n_raw = out / f"{stem}_n2n.avi"
            if p_n2n_raw.exists():
                print(f"  Cache N2N brut : {p_n2n_raw.name}")
                n2n_src = p_n2n_raw
            else:
                print(f"  Checkpoint : {ckpt}")
                if run_n2n(base_video, p_n2n_raw, ckpt) and p_n2n_raw.exists():
                    n2n_src = p_n2n_raw
                else:
                    print("  ERREUR N2N — pipeline N2N ignoré.")

    if n2n_src is not None:
        p_n2n_up = out / f"{stem}_n2n_{target_size}.avi"
        if p_n2n_up.exists():
            print(f"  Cache N2N upscalé : {p_n2n_up.name}")
        else:
            _step(f"C1 — Upscale vidéo N2N → {target_size}×{target_size}")
            upscale_video(n2n_src, p_n2n_up, target_size)

        p_n2n_usm = out / f"{stem}_n2n_usm_{target_size}.avi"
        if p_n2n_usm.exists():
            print(f"  Cache N2N→USM : {p_n2n_usm.name}")
        else:
            _step(f"C2 — USM sur N2N upscalé (sigma={sigma}  strength={strength})")
            apply_usm_video(p_n2n_up, p_n2n_usm, sigma, strength)

    # ── D. N2N → upscale → USM → CLAHE ──────────────────────────────────────
    _step(f"D — N2N → USM → CLAHE (clip={clahe_clip}  tile={clahe_tile}  usm={clahe_usm})")
    p_clahe: Path | None = None

    if p_n2n_usm is not None:
        # Réutilise p_n2n_usm déjà calculé en C
        p_clahe = out / f"{stem}_n2n_usm_clahe_{target_size}.avi"
        if p_clahe.exists():
            print(f"  Cache : {p_clahe.name}")
        else:
            full_mask = np.full((target_size, target_size), 255, dtype=np.uint8)
            apply_clahe_usm_pipeline(
                video_path           = str(p_n2n_usm),
                mask                 = full_mask,
                output_path          = str(p_clahe),
                clahe_clip_limit     = clahe_clip,
                clahe_tile_grid_size = (clahe_tile, clahe_tile),
                usm_strength         = clahe_usm,
            )
    else:
        print("  Pipeline D ignoré (N2N non disponible).")

    # ── E. N2N → Frangi → USM ────────────────────────────────────────────────
    _step("E — N2N → Frangi → USM")
    p_n2n_frangi_usm: Path | None = None

    if p_n2n_up is not None:
        p_n2n_frangi_usm = out / f"{stem}_n2n_frangi_usm_{target_size}.avi"
        if p_n2n_frangi_usm.exists():
            print(f"  Cache : {p_n2n_frangi_usm.name}")
        else:
            apply_frangi_usm_video(p_n2n_up, p_n2n_frangi_usm, sigma, strength)
    else:
        print("  Pipeline E ignoré (N2N non disponible).")

    # ── Projections temporelles moyennes ─────────────────────────────────────
    _step("Projections temporelles moyennes")

    print("  Base …")
    proj_base = mean_projection(p_base, target_size)
    p_proj_base = out / "proj_base.png"
    cv2.imwrite(str(p_proj_base), proj_base)
    print(f"  → {p_proj_base.name}")

    print("  USM …")
    proj_usm = mean_projection(p_usm, target_size)
    p_proj_usm = out / "proj_usm.png"
    cv2.imwrite(str(p_proj_usm), proj_usm)
    print(f"  → {p_proj_usm.name}")

    proj_n2n_usm: np.ndarray | None = None
    if p_n2n_usm is not None and p_n2n_usm.exists():
        print("  N2N→USM …")
        proj_n2n_usm = mean_projection(p_n2n_usm, target_size)
        p_proj_n2n_usm = out / "proj_n2n_usm.png"
        cv2.imwrite(str(p_proj_n2n_usm), proj_n2n_usm)
        print(f"  → {p_proj_n2n_usm.name}")

    proj_clahe: np.ndarray | None = None
    if p_clahe is not None and p_clahe.exists():
        print("  N2N→USM→CLAHE …")
        proj_clahe = mean_projection(p_clahe, target_size)
        p_proj_clahe = out / "proj_n2n_usm_clahe.png"
        cv2.imwrite(str(p_proj_clahe), proj_clahe)
        print(f"  → {p_proj_clahe.name}")

    proj_frangi_usm: np.ndarray | None = None
    if p_n2n_frangi_usm is not None and p_n2n_frangi_usm.exists():
        print("  N2N→Frangi→USM …")
        proj_frangi_usm = mean_projection(p_n2n_frangi_usm, target_size)
        cv2.imwrite(str(out / "proj_n2n_frangi_usm.png"), proj_frangi_usm)
        print("  → proj_n2n_frangi_usm.png")

    proj_nlm_frangi: np.ndarray | None = None
    if p_n2n_up is not None and p_n2n_up.exists():
        print("  N2N→NLM→Frangi→CLAHE (projection) …")
        proj_n2n_raw    = mean_projection(p_n2n_up, target_size)
        proj_nlm_frangi = n2n_proj_nlm_frangi_clahe(proj_n2n_raw, nlm_h_factor=frangi_nlm_h)
        cv2.imwrite(str(out / "proj_n2n_nlm_frangi_clahe.png"), proj_nlm_frangi)
        print("  → proj_n2n_nlm_frangi_clahe.png")

    # ── Grille comparative ────────────────────────────────────────────────────
    _step("Grille comparative")
    comparison_grid(
        proj_base       = proj_base,
        proj_usm        = proj_usm,
        proj_n2n_usm    = proj_n2n_usm,
        proj_clahe      = proj_clahe,
        proj_frangi_usm = proj_frangi_usm,
        proj_nlm_frangi = proj_nlm_frangi,
        output_path     = out / "comparison_grid.png",
        sigma           = sigma,
        strength        = strength,
        target_size     = target_size,
    )

    elapsed = time.perf_counter() - t0
    print(f"\n{'═'*62}")
    print(f"  TERMINÉ en {elapsed:.0f} s")
    print(f"  Résultats → {out}")
    print(f"{'═'*62}\n")


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Triple pipeline Base/USM/N2N+USM — vidéos + projections 1024 + grille.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Exemples :\n"
            "  python TripleProjection.py video.avi\n"
            "  python TripleProjection.py video.avi --output-dir resultats/\n"
            "  python TripleProjection.py video.avi --n2n-video video_n2n.avi\n"
            "  python TripleProjection.py video.avi --checkpoint checkpoints/best.pth\n"
            "  python TripleProjection.py video.avi --sigma 1.5 --strength 3.0\n"
        ),
    )
    parser.add_argument("input",
        help="Vidéo de base .avi.")
    parser.add_argument("--output-dir", default=None,
        help="Dossier de sortie (défaut : <stem>_triple_projection/ à côté de la vidéo).")
    parser.add_argument("--n2n-video", default=None,
        help="Vidéo avec N2N déjà appliqué. Si absent, l'inférence est lancée via --checkpoint.")
    parser.add_argument("--checkpoint", default=None,
        help="Checkpoint .pth du modèle N2N (défaut : auto-detect).")
    parser.add_argument("--target-size", type=int, default=DEFAULT_TARGET_SIZE,
        help=f"Résolution après upscale (défaut : {DEFAULT_TARGET_SIZE}).")
    parser.add_argument("--sigma", type=float, default=DEFAULT_USM_SIGMA,
        help=f"Rayon Gaussien USM (défaut : {DEFAULT_USM_SIGMA}).")
    parser.add_argument("--strength", type=float, default=DEFAULT_USM_STRENGTH,
        help=f"Intensité USM (défaut : {DEFAULT_USM_STRENGTH}).")
    parser.add_argument("--clahe-clip", type=float, default=DEFAULT_CLAHE_CLIP,
        help=f"clipLimit CLAHE (défaut : {DEFAULT_CLAHE_CLIP}).")
    parser.add_argument("--clahe-tile", type=int, default=DEFAULT_CLAHE_TILE,
        help=f"Taille des tuiles CLAHE en px (défaut : {DEFAULT_CLAHE_TILE}).")
    parser.add_argument("--clahe-usm", type=float, default=DEFAULT_CLAHE_USM,
        help=f"Intensité USM interne CLAHE (défaut : {DEFAULT_CLAHE_USM}).")
    parser.add_argument("--frangi-nlm-h", type=float, default=DEFAULT_FRANGI_NLM_H,
        help=f"Facteur h NLM cas F (h = facteur × sigma_estimé, défaut : {DEFAULT_FRANGI_NLM_H}).")
    args = parser.parse_args()

    base = Path(args.input)
    if not base.exists():
        sys.exit(f"ERREUR : vidéo introuvable : {base}")

    out_dir = (
        Path(args.output_dir)
        if args.output_dir
        else base.parent / f"{base.stem}_triple_projection"
    )

    n2n_video = Path(args.n2n_video) if args.n2n_video else None

    run(
        base_video   = base,
        output_dir   = out_dir,
        n2n_video    = n2n_video,
        checkpoint   = args.checkpoint,
        target_size  = args.target_size,
        sigma        = args.sigma,
        strength     = args.strength,
        clahe_clip   = args.clahe_clip,
        clahe_tile   = args.clahe_tile,
        clahe_usm    = args.clahe_usm,
        frangi_nlm_h = args.frangi_nlm_h,
    )


if __name__ == "__main__":
    main()
