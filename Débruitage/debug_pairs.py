#!/usr/bin/env python3
"""
Script standalone — vérifie les paires N2N AVANT l'entraînement.

Génère debug_pairs_n2n.png avec :
  Col 1 : Input (frame centrale de la fenêtre)
  Col 2 : Target (frame cible)
  Col 3 : Input - Target (SIGNÉE) — doit être symétrique et uniforme
           Rouge = Input > Target  |  Bleu = Target > Input
           Si dominante rouge → target trop sombre → BUG

Usage :
    python debug_pairs.py
    python debug_pairs.py --config config.yaml --output /tmp/debug.png
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import yaml
from torch.utils.data import DataLoader

SCRIPT_DIR   = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
sys.path.insert(0, str(SCRIPT_DIR))

from dataset import VascularVideoDataset

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser(description="Vérification des paires N2N")
    parser.add_argument("--config", default=str(SCRIPT_DIR / "config.yaml"))
    parser.add_argument("--output", default=None, help="Chemin de sortie du PNG")
    parser.add_argument("--n",      type=int, default=6, help="Nb d'exemples à afficher")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    d_cfg = cfg["data"]
    t_cfg = cfg["training"]

    raw_dirs    = d_cfg.get("video_dirs") or [d_cfg.get("video_dir", "TemporelIntegrale")]
    video_dirs  = [PROJECT_ROOT / d for d in raw_dirs]
    mask_path   = PROJECT_ROOT / d_cfg["mask_path"]
    video_pattern = d_cfg.get("video_pattern", "*.avi")

    log.info("Création du dataset (split=train, validation seule)…")
    ds = VascularVideoDataset(
        video_dirs           = video_dirs,
        mask_path            = mask_path,
        patch_size           = d_cfg["patch_size"],
        sigma_noise          = d_cfg.get("sigma_noise", 4.4),
        poisson_scale        = d_cfg.get("poisson_scale", 0.5),
        video_pattern        = video_pattern,
        split                = "train",
        train_split          = d_cfg["train_split"],
        samples_per_epoch    = t_cfg["batch_size"] * 8,
        augment              = False,
        recursive            = t_cfg.get("recursive_video_scan", True),
        force_strategy       = d_cfg.get("force_strategy", "auto"),
        ecc_max_px           = d_cfg.get("ecc_max_px", 3.0),
        min_pair_correlation = d_cfg.get("min_pair_correlation", 0.85),
        ecc_validation_n     = d_cfg.get("ecc_validation_n", 20),
        temporal_offset      = d_cfg.get("temporal_offset", 5),
    )

    dl    = DataLoader(ds, batch_size=args.n, num_workers=0, shuffle=True)
    batch = next(iter(dl))

    frames  = batch["frames"]   # (B, 5, ps, ps)
    targets = batch["target"]   # (B, 1, ps, ps)
    n       = min(args.n, frames.shape[0])

    out_dir = (Path(args.output).parent if args.output
               else PROJECT_ROOT / t_cfg["samples_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = Path(args.output) if args.output else out_dir / "debug_pairs_n2n.png"

    fig, axes = plt.subplots(n, 3, figsize=(10, 3.5 * n))
    if n == 1:
        axes = axes[np.newaxis, :]

    all_diffs: list[float] = []

    for i in range(n):
        inp  = frames[i, 2].numpy()         # frame centrale input [0,1]
        tgt  = targets[i, 0].numpy()        # target [0,1]
        diff = inp - tgt                     # SIGNÉ — révèle l'asymétrie

        mean_inp = float(inp.mean() * 255)
        mean_tgt = float(tgt.mean() * 255)
        std_inp  = float(inp.std()  * 255)
        std_tgt  = float(tgt.std()  * 255)
        mean_diff_255 = float(diff.mean() * 255)
        all_diffs.append(mean_diff_255)

        ok = abs(mean_diff_255) < 5.0
        status = "OK ✓" if ok else "PROBLÈME ✗"
        color  = "green" if ok else "red"

        # Colonne 0 : Input
        axes[i, 0].imshow(inp, cmap="gray", vmin=0, vmax=1)
        axes[i, 0].set_title(f"Input\nmoy={mean_inp:.1f}  std={std_inp:.2f}", fontsize=8)
        axes[i, 0].axis("off")

        # Colonne 1 : Target
        axes[i, 1].imshow(tgt, cmap="gray", vmin=0, vmax=1)
        axes[i, 1].set_title(f"Target\nmoy={mean_tgt:.1f}  std={std_tgt:.2f}", fontsize=8)
        axes[i, 1].axis("off")

        # Colonne 2 : Diff signée (rouge = Input > Target)
        im = axes[i, 2].imshow(diff, cmap="RdBu_r", vmin=-0.2, vmax=0.2)
        axes[i, 2].set_title(
            f"Input − Target (signé)\nΔmoy={mean_diff_255:+.1f}  {status}",
            fontsize=8, color=color,
        )
        axes[i, 2].axis("off")
        plt.colorbar(im, ax=axes[i, 2], fraction=0.046, pad=0.04)

    global_diff = float(np.mean(all_diffs))
    diagnosis = (
        "✓ Paires équilibrées (|Δ| < 5)"
        if abs(global_diff) < 5.0
        else f"✗ PROBLÈME : target {'trop sombre' if global_diff > 0 else 'trop claire'} "
             f"(Δ={global_diff:+.1f} > 5)"
    )
    fig.suptitle(
        f"DEBUG Paires N2N — Stratégie {d_cfg.get('force_strategy','auto').upper()}\n"
        f"Différence signée : rouge=Input>Target  bleu=Target>Input\n"
        f"Δ moyen global = {global_diff:+.1f}  →  {diagnosis}",
        fontsize=10,
    )
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)

    log.info("")
    log.info("─" * 60)
    log.info("PNG sauvegardé : %s", out_path)
    log.info("Δ moyen global : %+.2f  (OK si |Δ| < 5)", global_diff)
    log.info("")
    log.info("INTERPRÉTATION de la colonne 'Input − Target' :")
    log.info("  Uniforme + centré sur 0     → paires équilibrées ✓")
    log.info("  Dominante ROUGE (Δ > 5)     → target trop sombre → bug _sample_b ✗")
    log.info("  Dominante BLEUE (Δ < -5)    → input trop sombre  → bug _sample_b ✗")
    log.info("  Contours de vaisseaux visibles → paires mal alignées → activer ECC ✗")
    log.info("─" * 60)

    if abs(global_diff) >= 5.0:
        log.error("✗ PROBLÈME DÉTECTÉ — NE PAS LANCER L'ENTRAÎNEMENT")
        sys.exit(1)
    else:
        log.info("✓ Paires validées — entraînement autorisé")


if __name__ == "__main__":
    main()
