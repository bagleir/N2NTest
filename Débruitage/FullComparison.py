#!/usr/bin/env python3
"""
Pipeline de comparaison complète sur une frame de référence.

Entrée : vidéo de base (non prétraitée).

Génère automatiquement les 9 cas suivants (tous avec masque) :
  1. Base avant prétraitement                       → frame FRAME
  2. Après prétraitement                            → frame FRAME
  3. Après prétraitement + USM                      → frame FRAME
  4. Après prétraitement → projection (image temporelle)
  5. Après prétraitement + USM → projection
  6. Après N2N (sur prétraité 1024×1024) → projection
  7. Après N2N (sur USM 1024×1024)       → projection
  8. Prétraitement → Drizzle SR → USM    → projection SR
  9. Prétraitement → N2N (512) → Drizzle SR → USM → projection SR  [nécessite checkpoint]

Pour chaque cas : image complète + zoom 128×128 centré.
Les projections (cas 4-7) sont upscalées à 1024×1024 (Lanczos).
Les cas 8-9 utilisent Drizzle pour la super-résolution temporelle (512→1024).

Vidéos sauvegardées dans output_dir/videos/ :
  01_base_masked.avi   02_preprocessed.avi   03_usm.avi

Usage :
  python FullComparison.py video_base.avi
  python FullComparison.py video_base.avi --output-dir resultats/ --frame 50
  python FullComparison.py video_base.avi --checkpoint checkpoints/best.pth
  python FullComparison.py video_base.avi --drop-size 0.5
"""
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import time
from pathlib import Path

import cv2
import numpy as np
from skimage.filters import unsharp_mask as ski_usm

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).parent))
from mask_detection import load_mask
from Pretraitement import preprocess_video
from ApplyUSM import apply_usm_video
from ApplyMask import apply_mask_video
from TemporalProjection import compute_projections
from DrizzleSR import temporal_sr_drizzle

PROJ_KEY    = "mean"   # clé de projection utilisée : 'mean' | 'percentile_90' | etc.
CROP_SIZE   = 128
PROJ_SIZE   = 1024    # résolution des images de projection après upscale Lanczos


# ─────────────────────────────────────────────────────────────────────────────
# Utilitaires internes
# ─────────────────────────────────────────────────────────────────────────────


def _step(title: str) -> None:
    print(f"\n{'─'*62}\n  {title}\n{'─'*62}")


def _extract_masked_frame(video_path: str, frame_idx: int, mask_bool: np.ndarray) -> np.ndarray:
    """Lit la frame frame_idx, applique le masque, retourne uint8 (H×W)."""
    cap   = cv2.VideoCapture(video_path)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    idx   = min(frame_idx, max(total - 1, 0))
    cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
    ret, frame = cap.read()
    cap.release()
    if not ret:
        raise RuntimeError(f"Impossible de lire la frame {idx} de {video_path}")
    if frame.ndim == 3:
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    out = np.zeros_like(frame)
    out[mask_bool] = frame[mask_bool]
    return out


def _upscale(img: np.ndarray, size: int = PROJ_SIZE) -> np.ndarray:
    """Upscale une image à size×size via interpolation Lanczos."""
    return cv2.resize(img, (size, size), interpolation=cv2.INTER_LANCZOS4)


def _crop_center(img: np.ndarray, size: int = CROP_SIZE) -> np.ndarray:
    H, W = img.shape[:2]
    cy, cx = H // 2, W // 2
    half = size // 2
    return img[max(cy - half, 0):min(cy + half, H),
               max(cx - half, 0):min(cx + half, W)]


def _save_case(
    img: np.ndarray,
    folder: Path,
    stem: str,
    crop_size: int = CROP_SIZE,
) -> tuple[np.ndarray, np.ndarray]:
    """Sauvegarde image + crop, retourne (img, crop) pour la grille finale."""
    folder.mkdir(parents=True, exist_ok=True)
    crop = _crop_center(img, crop_size)
    cv2.imwrite(str(folder / f"{stem}.png"),                 img)
    cv2.imwrite(str(folder / f"{stem}_crop{crop_size}.png"), crop)
    print(f"    → {folder.name}/{stem}.png  +  _crop{crop_size}.png")
    return img, crop


def _upscale_video(input_path: str, output_path: str, size: int = PROJ_SIZE) -> None:
    """Upscale toutes les frames d'une vidéo à size×size (Lanczos)."""
    cap    = cv2.VideoCapture(input_path)
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
            writer = cv2.VideoWriter(output_path, fourcc, fps, (size, size), isColor=False)
        writer.write(up)
        count += 1
        pct = count / max(total, 1)
        bar = "█" * int(40 * pct) + "░" * (40 - int(40 * pct))
        print(f"\r  [{bar}] {count}/{total}", end="", flush=True)
    print()
    cap.release()
    if writer:
        writer.release()
    print(f"  Upscale vidéo → {Path(output_path).name}  ({total} frames, {size}×{size})")


def _run_n2n(video_in: Path, video_out: Path, checkpoint: str, sdir: Path) -> bool:
    """Lance inference.py en sous-processus. Retourne True si succès."""
    print(f"  Lancement N2N : {video_in.name} → {video_out.name}")
    result = subprocess.run([
        sys.executable, str(sdir / "inference.py"),
        "--video",      str(video_in),
        "--checkpoint", checkpoint,
        "--output",     str(video_out),
    ])
    if result.returncode != 0:
        print(f"  ERREUR N2N (code {result.returncode})")
        return False
    return True


def _apply_usm_image(img: np.ndarray, sigma: float, strength: float) -> np.ndarray:
    """Applique USM sur une seule image (uint8 H×W)."""
    sharpened = ski_usm(img, radius=sigma, amount=strength, preserve_range=True)
    return np.clip(sharpened, 0, 255).astype(np.uint8)


def _load_corrupted_frames(search_dirs: list[Path]) -> list[int]:
    """Charge la liste des frames corrompues depuis step2_corrected.json."""
    for d in search_dirs:
        p = Path(d) / "step2_corrected.json"
        if p.exists():
            with open(p) as f:
                frames = json.load(f).get("corrupted_frames", [])
            print(f"  Frames corrompues : {len(frames)} (depuis {p.name})")
            return frames
    return []


def _comparison_grid(
    cases: list[tuple[str, np.ndarray, np.ndarray]],
    output_path: Path,
    frame_idx: int,
    crop_size: int,
) -> None:
    """Grille synthétique : N colonnes × 2 lignes (image complète / zoom)."""
    n = len(cases)
    fig, axes = plt.subplots(
        2, n,
        figsize=(3.6 * n, 8),
        gridspec_kw={"hspace": 0.06, "wspace": 0.03},
    )
    if n == 1:
        axes = axes.reshape(2, 1)

    for col, (label, img, crop) in enumerate(cases):
        for row, panel in enumerate([img, crop]):
            ax = axes[row, col]
            ax.imshow(panel, cmap="gray", vmin=0, vmax=255, interpolation="nearest")
            ax.axis("off")
            if row == 0:
                ax.set_title(label, fontsize=7, fontweight="bold")

    axes[0, 0].set_ylabel("Image complète",            fontsize=8, rotation=90,
                           ha="center", va="center", labelpad=55)
    axes[1, 0].set_ylabel(f"Zoom {crop_size}×{crop_size}\n(centre)",
                           fontsize=8, rotation=90, ha="center", va="center", labelpad=55)

    plt.suptitle(
        f"Comparaison pipeline — frame {frame_idx}  |  masque appliqué sur tous les cas",
        fontsize=10,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(str(output_path), dpi=150, bbox_inches="tight")
    plt.close()
    print(f"\n  Grille synthétique → {output_path}")


# ─────────────────────────────────────────────────────────────────────────────
# Orchestrateur
# ─────────────────────────────────────────────────────────────────────────────


def run_comparison(
    base_video: str,
    output_dir: str,
    frame_idx:  int   = 50,
    checkpoint: str | None = None,
    sigma:      float = 2.0,
    strength:   float = 2.0,
    crop_size:  int   = CROP_SIZE,
    drop_size:  float = 0.7,
    mask_path:  str | None = None,
) -> None:
    t0   = time.perf_counter()
    base = Path(base_video)
    out  = Path(output_dir)
    sdir = Path(__file__).parent
    vdir = out / "videos"
    vdir.mkdir(parents=True, exist_ok=True)

    cases: list[tuple[str, np.ndarray, np.ndarray]] = []

    # ── Prétraitement ─────────────────────────────────────────────────────────
    _step("Prétraitement")
    p_preproc = vdir / "02_preprocessed.avi"
    mask_png  = vdir / f"{base.stem}_mask.png"

    preproc_work = out / "_preproc_work"

    # Si un masque explicite est fourni, on l'utilise directement
    if mask_path is not None:
        src = Path(mask_path)
        if not src.exists():
            sys.exit(f"ERREUR : masque introuvable : {src}")
        if not mask_png.exists() or src.resolve() != mask_png.resolve():
            shutil.copy2(str(src), str(mask_png))
        print(f"  Masque fourni : {src}")

    if p_preproc.exists() and mask_png.exists():
        print(f"  Cache trouvé : {p_preproc.name}  +  {mask_png.name}")
    else:
        # info_mode=True : working = preproc_work, mask sauvé dans preproc_work/mask.png
        summary = preprocess_video(
            video_path = base,
            output_dir = preproc_work,
            info_mode  = True,
        )
        if summary.get("status") == "error":
            sys.exit(f"ERREUR prétraitement :\n{summary.get('error', '')}")

        # Copier la vidéo finale vers notre emplacement canonique
        final = Path(summary["final_video"])
        if not final.exists():
            sys.exit(f"ERREUR : vidéo finale introuvable : {final}")
        shutil.copy2(str(final), str(p_preproc))

        # Masque dans le dossier de travail
        mask_in_work = preproc_work / "mask.png"
        if mask_in_work.exists():
            shutil.copy2(str(mask_in_work), str(mask_png))
        else:
            candidates = list(preproc_work.rglob("mask.png"))
            if candidates:
                shutil.copy2(str(candidates[0]), str(mask_png))
            else:
                sys.exit("ERREUR : mask.png introuvable après prétraitement.")

    mask      = load_mask(str(mask_png))
    mask_bool = mask.astype(bool)
    print(f"  Masque : {mask_png}")

    # ── Vidéo de base maskée ──────────────────────────────────────────────────
    _step("Vidéo de base (masque appliqué)")
    p_base_masked = vdir / "01_base_masked.avi"
    if not p_base_masked.exists():
        apply_mask_video(str(base), str(mask_png), str(p_base_masked))
    else:
        print(f"  Cache trouvé : {p_base_masked.name}")

    # ── Upscale prétraitement → 1024×1024 ────────────────────────────────────
    _step(f"Upscale vidéo prétraitée ({PROJ_SIZE}×{PROJ_SIZE})")
    p_upscaled = vdir / "03_upscaled.avi"
    if not p_upscaled.exists():
        _upscale_video(str(p_preproc), str(p_upscaled), size=PROJ_SIZE)
    else:
        print(f"  Cache trouvé : {p_upscaled.name}")

    # Masque upscalé pour les vidéos 1024×1024
    mask_up      = cv2.resize(mask, (PROJ_SIZE, PROJ_SIZE), interpolation=cv2.INTER_NEAREST)
    mask_bool_up = mask_up.astype(bool)

    # ── Vidéo USM (sur vidéo upscalée) ───────────────────────────────────────
    _step(f"Vidéo USM sur 1024×1024  (sigma={sigma}  strength={strength})")
    p_usm = vdir / "04_usm.avi"
    if not p_usm.exists():
        apply_usm_video(str(p_upscaled), str(p_usm), sigma=sigma, strength=strength)
    else:
        print(f"  Cache trouvé : {p_usm.name}")

    # ── Cas 1 : base (512×512) ────────────────────────────────────────────────
    _step(f"Cas 1 — Base avant prétraitement (frame {frame_idx})")
    img, crop = _save_case(
        _extract_masked_frame(str(base), frame_idx, mask_bool),
        out / "cas_01_base", f"frame{frame_idx:03d}", crop_size,
    )
    cases.append(("1. Base\n(avant prétraitement)", img, crop))

    # ── Cas 2 : après prétraitement (512×512) ────────────────────────────────
    _step(f"Cas 2 — Après prétraitement (frame {frame_idx})")
    img, crop = _save_case(
        _extract_masked_frame(str(p_preproc), frame_idx, mask_bool),
        out / "cas_02_preprocessed", f"frame{frame_idx:03d}", crop_size,
    )
    cases.append(("2. Prétraitement", img, crop))

    # ── Cas 3 : après upscale + USM (1024×1024) ───────────────────────────────
    _step(f"Cas 3 — Après upscale + USM (frame {frame_idx})")
    img, crop = _save_case(
        _extract_masked_frame(str(p_usm), frame_idx, mask_bool_up),
        out / "cas_03_usm", f"frame{frame_idx:03d}", crop_size,
    )
    cases.append(("3. Upscale\n+ USM", img, crop))

    # ── Cas 4 : prétraitement → upscale → projection ─────────────────────────
    _step("Cas 4 — Prétraitement + upscale + projection")
    projs4 = compute_projections(str(p_upscaled), mask_up, str(out / "cas_04_projection"))
    img, crop = _save_case(projs4[PROJ_KEY], out / "cas_04_projection",
                           f"projection_{PROJ_KEY}", crop_size)
    cases.append((f"4. Projection\n({PROJ_KEY})", img, crop))

    # ── Cas 5 : upscale + USM → projection ───────────────────────────────────
    _step("Cas 5 — Upscale + USM + projection")
    projs5 = compute_projections(str(p_usm), mask_up, str(out / "cas_05_usm_projection"))
    img, crop = _save_case(projs5[PROJ_KEY], out / "cas_05_usm_projection",
                           f"projection_{PROJ_KEY}", crop_size)
    cases.append((f"5. USM\n+ Projection", img, crop))

    # ── Pré-calcul Drizzle SR (cas 8, indépendant du checkpoint N2N) ─────────
    # Le résultat est stocké ici et inséré dans `cases` après les cas N2N
    # pour respecter l'ordre 1-9 dans la grille de comparaison.
    _step("Cas 8 — Drizzle SR (prétraité 512→1024) → USM")
    p_drizzle_sr     = vdir / "08_drizzle_sr"
    p_drizzle_sr_usm = vdir / "08_drizzle_sr_usm.png"
    if p_drizzle_sr_usm.exists():
        print(f"  Cache trouvé : {p_drizzle_sr_usm.name}")
        _c8_img = cv2.imread(str(p_drizzle_sr_usm), cv2.IMREAD_GRAYSCALE)
    else:
        corrupted8 = _load_corrupted_frames([preproc_work, vdir, base.parent])
        sr8 = temporal_sr_drizzle(
            video_path      = p_preproc,
            mask            = mask,
            corrupted_frames= corrupted8,
            output_path     = p_drizzle_sr,
            drop_size       = drop_size,
        )
        _c8_img = _apply_usm_image(sr8["sr_image"], sigma=sigma, strength=strength)
        _c8_img[~mask_bool_up] = 0
        cv2.imwrite(str(p_drizzle_sr_usm), _c8_img)

    # ── Détection du checkpoint N2N ───────────────────────────────────────────
    ckpt = checkpoint
    if ckpt is None:
        for candidate in [
            sdir / "checkpoints" / "last.pth",
            sdir.parent / "checkpoints" / "last.pth",
            sdir / "last.pth",
        ]:
            if candidate.exists():
                ckpt = str(candidate)
                break

    _c9_img: np.ndarray | None = None   # rempli si le checkpoint est disponible

    if ckpt is None or not Path(ckpt).exists():
        print("\n  AVERTISSEMENT : checkpoint N2N introuvable — cas 6, 7 & 9 ignorés.")
        print("  Utiliser --checkpoint <chemin.pth> pour les activer.\n")
    else:
        # ── Cas 6 : N2N (sur upscaled) → projection ──────────────────────────
        _step("Cas 6 — N2N sur upscalé + projection")
        p_n2n_pre = vdir / "n2n_upscaled.avi"
        ok = True
        if not p_n2n_pre.exists():
            ok = _run_n2n(p_upscaled, p_n2n_pre, ckpt, sdir)
        else:
            print(f"  Cache trouvé : {p_n2n_pre.name}")
        if ok and p_n2n_pre.exists():
            projs6 = compute_projections(str(p_n2n_pre), mask_up, str(out / "cas_06_n2n_upscaled"))
            img, crop = _save_case(projs6[PROJ_KEY], out / "cas_06_n2n_upscaled",
                                   f"projection_{PROJ_KEY}", crop_size)
            cases.append(("6. N2N (upscalé)\n+ Projection", img, crop))

        # ── Cas 7 : N2N (sur USM upscalé) → projection ───────────────────────
        _step("Cas 7 — N2N sur USM upscalé + projection")
        p_n2n_usm = vdir / "n2n_usm.avi"
        ok = True
        if not p_n2n_usm.exists():
            ok = _run_n2n(p_usm, p_n2n_usm, ckpt, sdir)
        else:
            print(f"  Cache trouvé : {p_n2n_usm.name}")
        if ok and p_n2n_usm.exists():
            projs7 = compute_projections(str(p_n2n_usm), mask_up, str(out / "cas_07_n2n_usm"))
            img, crop = _save_case(projs7[PROJ_KEY], out / "cas_07_n2n_usm",
                                   f"projection_{PROJ_KEY}", crop_size)
            cases.append(("7. N2N (USM upscalé)\n+ Projection", img, crop))

        # ── Cas 9 : N2N (512×512) → Drizzle SR → USM (pré-calcul) ───────────
        _step("Cas 9 — N2N (prétraité 512×512) → Drizzle SR → USM")
        p_n2n_512            = vdir / "09_n2n_preproc_512.avi"
        p_drizzle_n2n_sr_usm = vdir / "09_drizzle_n2n_sr_usm.png"
        ok = True
        if not p_n2n_512.exists():
            ok = _run_n2n(p_preproc, p_n2n_512, ckpt, sdir)
        else:
            print(f"  Cache trouvé : {p_n2n_512.name}")
        if ok and p_n2n_512.exists():
            if p_drizzle_n2n_sr_usm.exists():
                print(f"  Cache trouvé : {p_drizzle_n2n_sr_usm.name}")
                _c9_img = cv2.imread(str(p_drizzle_n2n_sr_usm), cv2.IMREAD_GRAYSCALE)
            else:
                corrupted9 = _load_corrupted_frames([preproc_work, vdir, base.parent])
                sr9 = temporal_sr_drizzle(
                    video_path      = p_n2n_512,
                    mask            = mask,
                    corrupted_frames= corrupted9,
                    output_path     = vdir / "09_drizzle_n2n_sr",
                    drop_size       = drop_size,
                )
                _c9_img = _apply_usm_image(sr9["sr_image"], sigma=sigma, strength=strength)
                _c9_img[~mask_bool_up] = 0
                cv2.imwrite(str(p_drizzle_n2n_sr_usm), _c9_img)

    # ── Cas 8 → ajouté après les cas N2N pour respecter l'ordre 1-9 ──────────
    img, crop = _save_case(_c8_img, out / "cas_08_drizzle_sr_usm", "projection", crop_size)
    cases.append(("8. Drizzle SR\n→ USM", img, crop))

    # ── Cas 9 → ajouté en dernier si disponible ───────────────────────────────
    if _c9_img is not None:
        img, crop = _save_case(_c9_img, out / "cas_09_drizzle_n2n_usm", "projection", crop_size)
        cases.append(("9. N2N → Drizzle SR\n→ USM", img, crop))

    # ── Grille synthétique ────────────────────────────────────────────────────
    if cases:
        _comparison_grid(cases, out / "comparison_grid.png", frame_idx, crop_size)

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
        description="Comparaison complète du pipeline (9 cas) sur une frame de référence.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("input",
        help="Vidéo de base non prétraitée (.avi).")
    parser.add_argument("--output-dir", default=None,
        help="Dossier de sortie (défaut : <input>_comparison/).")
    parser.add_argument("--frame", type=int, default=50,
        help="Indice de la frame de référence (défaut : 50).")
    parser.add_argument("--checkpoint", default=None,
        help="Checkpoint N2N (.pth) pour les cas 6, 7 & 9.")
    parser.add_argument("--sigma",      type=float, default=2.0,
        help="Rayon Gaussien USM (défaut : 2.0).")
    parser.add_argument("--strength",   type=float, default=2.0,
        help="Intensité USM (défaut : 2.0).")
    parser.add_argument("--crop-size",  type=int, default=CROP_SIZE,
        help=f"Taille du zoom central en pixels (défaut : {CROP_SIZE}).")
    parser.add_argument("--drop-size",  type=float, default=0.7,
        help="Taille du noyau Drizzle en pixels HR (défaut : 0.7). "
             "0.5=net, 0.7=équilibré, 1.0=moyenne.")
    parser.add_argument("--mask", default=None,
        help="Masque binaire PNG à utiliser directement (défaut : auto-généré par le prétraitement).")
    args = parser.parse_args()

    inp = Path(args.input)
    if not inp.exists():
        sys.exit(f"ERREUR : vidéo introuvable : {inp}")

    out_dir = (
        Path(args.output_dir)
        if args.output_dir
        else Path.cwd() / f"{inp.stem}_comparison"
    )

    print(f"\n  Entrée      : {inp}")
    print(f"  Sortie      : {out_dir}")
    print(f"  Frame       : {args.frame}")
    print(f"  USM         : sigma={args.sigma}  strength={args.strength}")
    print(f"  Drizzle     : drop_size={args.drop_size}")
    print(f"  Checkpoint  : {args.checkpoint or 'auto-detect'}")
    print(f"  Masque      : {args.mask or 'auto-généré'}")

    run_comparison(
        base_video = str(inp),
        output_dir = str(out_dir),
        frame_idx  = args.frame,
        checkpoint = args.checkpoint,
        sigma      = args.sigma,
        strength   = args.strength,
        crop_size  = args.crop_size,
        drop_size  = args.drop_size,
        mask_path  = args.mask,
    )


if __name__ == "__main__":
    main()
