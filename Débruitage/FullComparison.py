#!/usr/bin/env python3
"""
Pipeline de comparaison complète sur une frame de référence.

Entrée : vidéo de base (non prétraitée).

Cas générés (tous avec masque) :
  1. Base avant prétraitement                → frame FRAME
  2. Après prétraitement                     → frame FRAME
  3. Après prétraitement + USM               → frame FRAME
  4. Prétraitement → upscale → projection
  5. Upscale + USM → projection
  6. N2N (sur upscalé 1024×1024) → projection              [checkpoint requis]
  7. N2N (sur USM upscalé 1024×1024) → projection          [checkpoint requis]
     └─ USM appliqué AVANT N2N dans le cas 7.
 10. N2N (512) → USM → upscale → projection                [checkpoint requis]
 11. N2N (512) → CLAHE + USM → upscale → projection        [checkpoint requis]
     └─ Cas 10/11 : N2N sur la vidéo prétraitée 512, post-traitement en 512,
        puis upscale 1024 et projection. Ordre inverse du cas 7 (post-traitement
        APRÈS le débruitage).

Pour chaque cas : image complète + zoom 128×128 centré.
Les projections sont à 1024×1024 (upscale Lanczos).

Deux grilles de comparaison produites :
  comparison_grid_pipeline.png  — cas 1, 4, 5
  comparison_grid_n2n.png       — cas 2, 5, 7, 6, 10, 11

Usage :
  python FullComparison.py video_base.avi
  python FullComparison.py video_base.avi --output-dir resultats/ --frame 50
  python FullComparison.py video_base.avi --checkpoint checkpoints/best.pth
"""
from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
import time
from pathlib import Path

import cv2
import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).parent))
from mask_detection import load_mask
from Pretraitement import preprocess_video
from ApplyUSM import apply_usm_video
from ApplyMask import apply_mask_video
from TemporalProjection import compute_projections
from CLAHE import apply_clahe_usm_pipeline

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


def _run_n2n(
    video_in:   Path,
    video_out:  Path,
    checkpoint: str,
    sdir:       Path,
    mask_path:  Path,
) -> bool:
    """
    Lance inference.py en sous-processus.

    Le masque est toujours passé en chemin absolu via --mask pour éviter
    que inference.py tente de le résoudre depuis config.yaml (chemin relatif
    qui ne correspond pas au répertoire de travail courant).

    Cas 6/7 reçoivent mask_1024.png (1024×1024) car les vidéos sont upscalées ;
    sans ça numpy lève une erreur de forme dans np.where(mask > 0, ...).
    """
    print(f"  Lancement N2N : {video_in.name} → {video_out.name}")
    print(f"  Masque         : {mask_path}")
    result = subprocess.run([
        sys.executable, str(sdir / "inference.py"),
        "--video",      str(video_in),
        "--checkpoint", checkpoint,
        "--mask",       str(mask_path.resolve()),
        "--output",     str(video_out),
    ])
    if result.returncode != 0:
        print(f"  ERREUR N2N (code {result.returncode})")
        return False
    return True




def _comparison_grid(
    cases:       list[tuple[str, np.ndarray, np.ndarray]],
    output_path: Path,
    frame_idx:   int,
    crop_size:   int,
    title:       str = "",
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

    suptitle = (f"{title}\n" if title else "") + (
        f"frame {frame_idx}  |  masque appliqué sur tous les cas"
    )
    plt.suptitle(suptitle, fontsize=10)
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
    mask_path:  str | None = None,
) -> None:
    t0   = time.perf_counter()
    base = Path(base_video)
    out  = Path(output_dir)
    sdir = Path(__file__).parent
    vdir = out / "videos"
    vdir.mkdir(parents=True, exist_ok=True)

    # Dict {numéro_cas: (label, img, crop)} — permet de former les deux grilles finales.
    case_data: dict[int, tuple[str, np.ndarray, np.ndarray]] = {}

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

    # Sauvegarder le masque 1024×1024 pour l'inférence N2N sur vidéos upscalées
    # (cas 6 et 7 : inference.py reçoit un chemin absolu et le charge tel quel).
    mask_up_path = vdir / "mask_1024.png"
    if not mask_up_path.exists():
        cv2.imwrite(str(mask_up_path), mask_up)
        print(f"  Masque 1024×1024 → {mask_up_path.name}")

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
    case_data[1] = ("1. Base\n(avant prétraitement)", img, crop)

    # ── Cas 2 : après prétraitement (512×512) ────────────────────────────────
    _step(f"Cas 2 — Après prétraitement (frame {frame_idx})")
    img, crop = _save_case(
        _extract_masked_frame(str(p_preproc), frame_idx, mask_bool),
        out / "cas_02_preprocessed", f"frame{frame_idx:03d}", crop_size,
    )
    case_data[2] = ("2. Prétraitement", img, crop)

    # ── Cas 3 : après upscale + USM (1024×1024) ───────────────────────────────
    _step(f"Cas 3 — Après upscale + USM (frame {frame_idx})")
    img, crop = _save_case(
        _extract_masked_frame(str(p_usm), frame_idx, mask_bool_up),
        out / "cas_03_usm", f"frame{frame_idx:03d}", crop_size,
    )
    case_data[3] = ("3. Upscale\n+ USM", img, crop)

    # ── Cas 4 : prétraitement → upscale → projection ─────────────────────────
    _step("Cas 4 — Prétraitement + upscale + projection")
    projs4 = compute_projections(str(p_upscaled), mask_up, str(out / "cas_04_projection"))
    img, crop = _save_case(projs4[PROJ_KEY], out / "cas_04_projection",
                           f"projection_{PROJ_KEY}", crop_size)
    case_data[4] = (f"4. Projection\n({PROJ_KEY})", img, crop)

    # ── Cas 5 : upscale + USM → projection ───────────────────────────────────
    _step("Cas 5 — Upscale + USM + projection")
    projs5 = compute_projections(str(p_usm), mask_up, str(out / "cas_05_usm_projection"))
    img, crop = _save_case(projs5[PROJ_KEY], out / "cas_05_usm_projection",
                           f"projection_{PROJ_KEY}", crop_size)
    case_data[5] = (f"5. USM\n+ Projection", img, crop)

    # ── Détection du checkpoint N2N ───────────────────────────────────────────
    # Ordre de priorité :
    #   1. checkpoint passé explicitement via --checkpoint
    #   2. config.yaml → inference.checkpoint  (best_model.pth)
    #   3. Candidats hardcodés (last.pth, best_model.pth)
    ckpt = checkpoint
    if ckpt is None:
        # Lire le checkpoint recommandé dans config.yaml (même logique qu'inference.py)
        try:
            import yaml as _yaml
            with open(sdir / "config.yaml") as _f:
                _cfg_inf = _yaml.safe_load(_f)
            _ckpt_rel = _cfg_inf.get("inference", {}).get("checkpoint", "")
            if _ckpt_rel:
                _ckpt_abs = (sdir.parent / _ckpt_rel).resolve()
                if _ckpt_abs.is_file():
                    ckpt = str(_ckpt_abs)
        except Exception:
            pass

    if ckpt is None:
        for candidate in [
            sdir / "checkpoints" / "best_model.pth",
            sdir / "checkpoints" / "last.pth",
            sdir.parent / "checkpoints" / "best_model.pth",
            sdir.parent / "checkpoints" / "last.pth",
            sdir / "best_model.pth",
            sdir / "last.pth",
        ]:
            if candidate.is_file():
                ckpt = str(candidate)
                break

    # Si le checkpoint pointe vers un dossier, chercher best_model.pth / last.pth dedans.
    if ckpt is not None and Path(ckpt).is_dir():
        for _name in ("best_model.pth", "last.pth"):
            _candidate = Path(ckpt) / _name
            if _candidate.is_file():
                ckpt = str(_candidate)
                break
        else:
            ckpt = None  # dossier sans .pth valide

    _c9_img: np.ndarray | None = None   # rempli si le checkpoint est disponible

    if ckpt is None or not Path(ckpt).is_file():
        print("\n  AVERTISSEMENT : checkpoint N2N introuvable — cas 6, 7 & 9 ignorés.")
        print("  → Lancer l'entraînement :  python train.py")
        print("  → Ou préciser le chemin : --checkpoint checkpoints/best_model.pth\n")
    else:
        print(f"  Checkpoint : {ckpt}")

        # ── Cas 6 : N2N (sur upscaled 1024) → projection ────────────────────
        # mask_up_path = masque 1024×1024 : évite l'erreur de forme dans inference.py
        _step("Cas 6 — N2N sur upscalé + projection")
        p_n2n_pre = vdir / "n2n_upscaled.avi"
        ok = True
        if not p_n2n_pre.exists():
            ok = _run_n2n(p_upscaled, p_n2n_pre, ckpt, sdir, mask_path=mask_up_path)
        else:
            print(f"  Cache trouvé : {p_n2n_pre.name}")
        if ok and p_n2n_pre.exists():
            projs6 = compute_projections(str(p_n2n_pre), mask_up, str(out / "cas_06_n2n_upscaled"))
            img, crop = _save_case(projs6[PROJ_KEY], out / "cas_06_n2n_upscaled",
                                   f"projection_{PROJ_KEY}", crop_size)
            case_data[6] = ("6. N2N (upscalé)\n+ Projection", img, crop)

        # ── Cas 7 : N2N (sur USM upscalé 1024) → projection ─────────────────
        _step("Cas 7 — N2N sur USM upscalé + projection")
        p_n2n_usm = vdir / "n2n_usm.avi"
        ok = True
        if not p_n2n_usm.exists():
            ok = _run_n2n(p_usm, p_n2n_usm, ckpt, sdir, mask_path=mask_up_path)
        else:
            print(f"  Cache trouvé : {p_n2n_usm.name}")
        if ok and p_n2n_usm.exists():
            projs7 = compute_projections(str(p_n2n_usm), mask_up, str(out / "cas_07_n2n_usm"))
            img, crop = _save_case(projs7[PROJ_KEY], out / "cas_07_n2n_usm",
                                   f"projection_{PROJ_KEY}", crop_size)
            case_data[7] = ("7. N2N (USM upscalé)\n+ Projection", img, crop)

        # ── Cas 10 & 11 : N2N 512 → post-traitement → upscale → projection ────
        # Les deux cas partagent le même résultat N2N sur la vidéo prétraitée 512.
        _step("Cas 10 & 11 — N2N (prétraité 512) → USM / CLAHE+USM → upscale → projection")
        p_n2n_512_shared = vdir / "10_11_n2n_preproc_512.avi"
        ok_n2n512 = True
        if not p_n2n_512_shared.exists():
            ok_n2n512 = _run_n2n(p_preproc, p_n2n_512_shared, ckpt, sdir, mask_path=mask_png)
        else:
            print(f"  Cache trouvé : {p_n2n_512_shared.name}")

        if ok_n2n512 and p_n2n_512_shared.exists():
            # ── Cas 10 : N2N → USM → upscale → projection ────────────────────
            p_n2n_usm_512  = vdir / "10_n2n_usm_512.avi"
            p_n2n_usm_1024 = vdir / "10_n2n_usm_1024.avi"
            if not p_n2n_usm_512.exists():
                print(f"  Cas 10 : application USM sur {p_n2n_512_shared.name}…")
                apply_usm_video(
                    str(p_n2n_512_shared), str(p_n2n_usm_512),
                    sigma=sigma, strength=strength,
                )
            else:
                print(f"  Cache trouvé : {p_n2n_usm_512.name}")
            if not p_n2n_usm_1024.exists():
                _upscale_video(str(p_n2n_usm_512), str(p_n2n_usm_1024), size=PROJ_SIZE)
            else:
                print(f"  Cache trouvé : {p_n2n_usm_1024.name}")
            projs10 = compute_projections(
                str(p_n2n_usm_1024), mask_up, str(out / "cas_10_n2n_usm")
            )
            img, crop = _save_case(
                projs10[PROJ_KEY], out / "cas_10_n2n_usm",
                f"projection_{PROJ_KEY}", crop_size,
            )
            case_data[10] = ("10. N2N → USM\n→ Projection", img, crop)

            # ── Cas 11 : N2N → CLAHE+USM → upscale → projection ──────────────
            # apply_clahe_usm_pipeline enchaîne CLAHE puis USM en une seule passe.
            p_n2n_clahe_usm_512  = vdir / "11_n2n_clahe_usm_512.avi"
            p_n2n_clahe_usm_1024 = vdir / "11_n2n_clahe_usm_1024.avi"
            if not p_n2n_clahe_usm_512.exists():
                print(f"  Cas 11 : application CLAHE+USM sur {p_n2n_512_shared.name}…")
                apply_clahe_usm_pipeline(
                    video_path  = str(p_n2n_512_shared),
                    mask        = mask,
                    output_path = str(p_n2n_clahe_usm_512),
                )
            else:
                print(f"  Cache trouvé : {p_n2n_clahe_usm_512.name}")
            if not p_n2n_clahe_usm_1024.exists():
                _upscale_video(str(p_n2n_clahe_usm_512), str(p_n2n_clahe_usm_1024), size=PROJ_SIZE)
            else:
                print(f"  Cache trouvé : {p_n2n_clahe_usm_1024.name}")
            projs11 = compute_projections(
                str(p_n2n_clahe_usm_1024), mask_up, str(out / "cas_11_n2n_clahe_usm")
            )
            img, crop = _save_case(
                projs11[PROJ_KEY], out / "cas_11_n2n_clahe_usm",
                f"projection_{PROJ_KEY}", crop_size,
            )
            case_data[11] = ("11. N2N → CLAHE+USM\n→ Projection", img, crop)

    # ── Deux grilles de comparaison ───────────────────────────────────────────
    #
    #   Grille pipeline  — vision globale sans N2N direct
    #     1 Base | 4 Projection | 5 USM+Proj
    #
    #   Grille N2N — comparaison des stratégies de débruitage (ordre utilisateur)
    #     2 Prétraité | 5 USM+Proj | 7 N2N(USM avant) | 6 N2N(upscalé)
    #     | 10 N2N→USM | 11 N2N→CLAHE+USM
    #
    # Les cas absents (checkpoint manquant) sont simplement omis.

    def _make_grid(numbers: list[int], output_path: Path, title: str) -> None:
        selected = [case_data[n] for n in numbers if n in case_data]
        if selected:
            _comparison_grid(selected, output_path, frame_idx, crop_size, title=title)
        else:
            print(f"  Aucun cas disponible pour la grille '{title}' — ignorée")

    _make_grid(
        [1, 4, 5],
        out / "comparison_grid_pipeline.png",
        "Pipeline — 1 Base  |  4 Projection  |  5 USM + Projection",
    )
    _make_grid(
        [2, 5, 7, 6, 10, 11],
        out / "comparison_grid_n2n.png",
        "N2N — 2 Prétraité  |  5 USM+Proj  |  7 N2N(USM→)  |  6 N2N(upsc)  |  10 N2N→USM  |  11 N2N→CLAHE+USM",
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
        mask_path  = args.mask,
    )


if __name__ == "__main__":
    main()
