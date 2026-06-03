#!/usr/bin/env python3
"""
Boucle d'entraînement FastDVDnet N2N pour le débruitage de vidéos vasculaires.

Usage :
    python train.py [--config config.yaml] [--resume checkpoints/last.pth]
"""

from __future__ import annotations

import argparse
import logging
import math
import shutil
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import yaml
from torch.cuda.amp import GradScaler
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
from torch.utils.data import DataLoader
from tqdm import tqdm

SCRIPT_DIR   = Path(__file__).resolve().parent   # Stage/Débruitage/
PROJECT_ROOT = SCRIPT_DIR.parent                 # Stage/

from model   import FastDVDnet
from dataset import MultiScaleBatchSampler, VascularVideoDataset

logging.basicConfig(
    level    = logging.INFO,
    format   = "%(asctime)s [%(levelname)s] %(message)s",
    handlers = [logging.StreamHandler()],
)
log = logging.getLogger(__name__)


def _ram_mb() -> float:
    import os
    try:
        with open(f"/proc/{os.getpid()}/status") as f:
            for line in f:
                if line.startswith("VmRSS:"):
                    return float(line.split()[1]) / 1024.0
    except Exception:
        pass
    return 0.0


# ── Scheduler avec warmup ─────────────────────────────────────────────────────

def _build_scheduler_with_warmup(
    optimizer:      torch.optim.Optimizer,
    warmup_epochs:  int,
    total_epochs:   int,
) -> torch.optim.lr_scheduler.LRScheduler:
    """
    Epochs 1 → warmup_epochs   : LR monte linéairement de lr/20 → lr  (LinearLR).
    Epochs warmup → total       : CosineAnnealingLR de lr → 0.

    Implémenté avec SequentialLR (PyTorch ≥ 1.13).
    Si warmup_epochs == 0, retourne directement un CosineAnnealingLR.
    """
    if warmup_epochs <= 0:
        return CosineAnnealingLR(optimizer, T_max=max(total_epochs, 1))

    warmup = LinearLR(
        optimizer,
        start_factor = 1.0 / 20,
        end_factor   = 1.0,
        total_iters  = warmup_epochs,
    )
    cosine = CosineAnnealingLR(
        optimizer,
        T_max = max(total_epochs - warmup_epochs, 1),
    )
    return SequentialLR(optimizer, schedulers=[warmup, cosine], milestones=[warmup_epochs])


# ── Loss ──────────────────────────────────────────────────────────────────────

def _temporal_consistency_loss(
    outputs: list[torch.Tensor],
    inputs:  list[torch.Tensor],
) -> torch.Tensor:
    """Pénalise |out(t)-out(t-1)| > |in(t)-in(t-1)| pour éviter les discontinuités."""
    if len(outputs) < 2:
        return torch.zeros(1, device=outputs[0].device)[0]
    loss, count = torch.zeros(1, device=outputs[0].device)[0], 0
    for i in range(1, len(outputs)):
        loss += torch.relu((outputs[i] - outputs[i-1]).abs() - (inputs[i] - inputs[i-1]).abs()).mean()
        count += 1
    return loss / count


def _pulsation_preservation_loss(
    output:    torch.Tensor,
    target:    torch.Tensor,
    threshold: float,
) -> torch.Tensor:
    """Pénalise si var(output) < threshold × var(target) — préserve la pulsation cardiaque.

    Requiert batch_size ≥ 2 pour que var(dim=0) soit définie.
    Avec un seul sample (dernier batch d'une échelle si drop_last=False),
    var() renverrait NaN → on retourne 0 plutôt que de corrompre la loss.
    """
    if output.shape[0] < 2:
        return torch.zeros(1, device=output.device)[0]
    return torch.relu(threshold * target.var(dim=0).mean() - output.var(dim=0).mean())


# ── SSIM + Gradient loss natifs PyTorch (sans dépendance externe) ────────────

def _ssim_tensor(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """
    SSIM différentiable — retourne un tenseur scalaire dans [0, 1].
    Toujours float32, toujours borné → ne peut pas causer de NaN.
    """
    # Forcer float32 même sous autocast float16
    pred   = pred.detach().float() if not pred.requires_grad else pred.float()
    target = target.float()
    # Clamp entrées dans [0, 1] pour stabilité numérique
    pred   = pred.clamp(0, 1)
    target = target.clamp(0, 1)

    C1, C2 = 0.01 ** 2, 0.03 ** 2
    dev = pred.device
    coords = torch.arange(11, dtype=torch.float32, device=dev) - 5.0
    g = torch.exp(-coords ** 2 / (2 * 1.5 ** 2));  g = g / g.sum()
    k = (g.unsqueeze(0) * g.unsqueeze(1)).view(1, 1, 11, 11)

    def _c(x: torch.Tensor) -> torch.Tensor:
        return torch.nn.functional.conv2d(x, k, padding=5)

    mu_p = _c(pred);  mu_t = _c(target)
    mu_p2 = mu_p*mu_p;  mu_t2 = mu_t*mu_t;  mu_pt = mu_p*mu_t
    sig_p  = (_c(pred*pred)   - mu_p2).clamp(min=0)
    sig_t  = (_c(target*target) - mu_t2).clamp(min=0)
    sig_pt =  _c(pred*target) - mu_pt
    num   = (2*mu_pt + C1) * (2*sig_pt + C2)
    denom = (mu_p2 + mu_t2 + C1) * (sig_p + sig_t + C2)
    # Clamp la map dans [0, 1] — évite que (1-SSIM) devienne négatif ou explosif
    ssim_map = (num / denom.clamp(min=1e-8)).clamp(0.0, 1.0)
    return ssim_map.mean()


def _ssim(pred: torch.Tensor, target: torch.Tensor) -> float:
    """Version scalaire pour le monitoring (non différentiable)."""
    if torch.isnan(pred).any() or torch.isnan(target).any():
        return float("nan")
    with torch.no_grad():
        return float(_ssim_tensor(pred.float(), target.float()).item())


def _gradient_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """
    Loss sur les gradients de Sobel : force le réseau à préserver les bords fins.
    mean(|∂x(out)-∂x(tgt)| + |∂y(out)-∂y(tgt)|)  — anti-flou vasculaire.
    """
    pred   = pred.float();  target = target.float()
    dev    = pred.device
    kx = torch.tensor([[-1,0,1],[-2,0,2],[-1,0,1]],
                      dtype=torch.float32, device=dev).view(1,1,3,3) / 8.0
    ky = torch.tensor([[-1,-2,-1],[0,0,0],[1,2,1]],
                      dtype=torch.float32, device=dev).view(1,1,3,3) / 8.0
    F = torch.nn.functional.conv2d
    return ((F(pred,kx,padding=1) - F(target,kx,padding=1)).abs()
          + (F(pred,ky,padding=1) - F(target,ky,padding=1)).abs()).mean()


class N2NLoss(nn.Module):
    def __init__(
        self,
        w_l1:                float = 0.70,
        w_ssim:              float = 0.30,
        w_gradient:          float = 0.20,
        lambda_temporal:     float = 0.10,
        lambda_pulsation:    float = 0.05,
        pulsation_threshold: float = 0.70,
    ) -> None:
        super().__init__()
        self.l1       = nn.L1Loss()
        self.w_l1     = w_l1
        self.w_ssim   = w_ssim
        self.w_grad   = w_gradient
        self.lam_t    = lambda_temporal
        self.lam_p    = lambda_pulsation
        self.puls_thr = pulsation_threshold

    def forward(
        self,
        output:     torch.Tensor,
        target:     torch.Tensor,
        output_seq: list[torch.Tensor] | None = None,
        input_seq:  list[torch.Tensor] | None = None,
    ) -> tuple[torch.Tensor, dict[str, float]]:
        # Forcer float32 (AMP peut produire du float16)
        out_f  = output.float()
        tgt_f  = target.float()

        l_l1   = self.l1(out_f, tgt_f)
        l_ssim = 1.0 - _ssim_tensor(out_f, tgt_f)
        l_grad = _gradient_loss(out_f, tgt_f)
        l_temp = (
            _temporal_consistency_loss(output_seq, input_seq)
            if output_seq and input_seq
            else torch.zeros(1, device=output.device)[0]
        )
        l_puls = _pulsation_preservation_loss(out_f, tgt_f, self.puls_thr)

        total = (self.w_l1  * l_l1
               + self.w_ssim * l_ssim
               + self.w_grad * l_grad
               + self.lam_t  * l_temp
               + self.lam_p  * l_puls)
        return total, {
            "l1":       l_l1.item(),
            "ssim":     l_ssim.item(),
            "grad":     l_grad.item(),
            "temporal": l_temp.item(),
            "puls":     l_puls.item(),
        }


# ── Visualisation des paires N2N ──────────────────────────────────────────────

def _save_pair_debug(
    train_dl: DataLoader,
    out_dir:  Path,
    n_examples: int = 6,
) -> None:
    """
    Sauvegarde des exemples de paires (input central frame | target | différence)
    AVANT l'entraînement pour vérifier visuellement la cohérence des paires N2N.

    Un bon résultat : input et target montrent le même vaisseau, légèrement bruités
    différemment. Si les deux images sont identiques → stratégie B (bruit synthétique).
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out_dir.mkdir(parents=True, exist_ok=True)

    batch = next(iter(train_dl))
    frames  = batch["frames"]   # (B, 5, ps, ps)
    targets = batch["target"]   # (B, 1, ps, ps)
    n = min(n_examples, frames.shape[0])

    fig, axes = plt.subplots(n, 3, figsize=(9, 3 * n))
    if n == 1:
        axes = axes[np.newaxis, :]

    for i in range(n):
        inp = frames[i, 2].numpy()      # frame centrale de l'input
        tgt = targets[i, 0].numpy()     # target
        diff = np.abs(inp - tgt)

        for ax, img, title, cmap in zip(
            axes[i],
            [inp,              tgt,      diff * 5],
            ["Input (frame t)", "Target", "Diff × 5"],
            ["gray",           "gray",   "hot"],
        ):
            ax.imshow(img, cmap=cmap, vmin=0, vmax=1 if cmap == "gray" else None)
            ax.set_title(title, fontsize=8)
            ax.axis("off")

        # Statistiques sur la différence
        mean_diff = float(diff.mean())
        axes[i, 2].set_xlabel(
            f"mean|diff|={mean_diff:.4f}  "
            + ("→ Strat. B (diff faible = bruit synthétique)" if mean_diff < 0.01
               else "→ Strat. A (diff = bruit indépendant)"),
            fontsize=6,
        )

    fig.suptitle(
        "Vérification paires N2N — à inspecter avant de lancer l'entraînement\n"
        "Strat. A : input ≠ target (bruit indépendant) | Strat. B : target = input + bruit synthétique",
        fontsize=9,
    )
    fig.tight_layout()
    path = out_dir / "debug_pairs_n2n.png"
    fig.savefig(path, dpi=120)
    plt.close(fig)
    tqdm.write(f"  → Paires N2N sauvegardées : {path}")
    tqdm.write(f"     Ouvrir ce PNG pour vérifier visuellement la cohérence des paires.")


# ── Helpers ───────────────────────────────────────────────────────────────────

def _measure_inference_time(model: nn.Module, device: torch.device, n: int = 30) -> float:
    """Temps moyen d'inférence par frame 512×512 en ms."""
    model.eval()
    df = torch.randn(1, 5, 512, 512, device=device)
    ds = torch.full((1, 1, 512, 512), 4.4 / 255.0, device=device)
    with torch.no_grad():
        for _ in range(5):
            model(df, ds)
    if device.type == "cuda":
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    with torch.no_grad():
        for _ in range(n):
            model(df, ds)
    if device.type == "cuda":
        torch.cuda.synchronize()
    return (time.perf_counter() - t0) / n * 1000.0


def _save_samples(
    inputs:  torch.Tensor,
    outputs: torch.Tensor,
    targets: torch.Tensor,
    epoch:   int,
    out_dir: Path,
    scale:   int | None = None,
) -> None:
    """Sauvegarde jusqu'à 3 PNGs comparatifs : input bruité | output réseau | target.

    Le nom de fichier inclut l'échelle quand elle est fournie :
        epoch0010_scale128_sample1.png
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out_dir.mkdir(parents=True, exist_ok=True)
    scale_tag = f"_scale{scale}" if scale is not None else ""
    for i in range(min(3, inputs.shape[0])):
        fig, axes = plt.subplots(1, 3, figsize=(9, 3))
        for ax, img, title in zip(
            axes,
            [inputs[i, 2], outputs[i, 0], targets[i, 0]],
            ["Input (bruité)", "Output (débruité)", "Target"],
        ):
            ax.imshow(img.cpu().numpy(), cmap="gray", vmin=0, vmax=1)
            ax.set_title(title, fontsize=9)
            ax.axis("off")
        size_str = f"{scale}×{scale}" if scale is not None else ""
        fig.suptitle(f"Epoch {epoch}{f'  —  {size_str}' if size_str else ''}  —  exemple {i + 1}", fontsize=10)
        fig.tight_layout()
        fig.savefig(out_dir / f"epoch{epoch:04d}{scale_tag}_sample{i + 1}.png", dpi=100)
        plt.close(fig)


def _save_checkpoint(state: dict, path: Path, is_best: bool, best_path: Path) -> None:
    torch.save(state, path)
    if is_best:
        shutil.copy2(str(path), str(best_path))
        log.info("  → Best model → %s", best_path.name)


# ── Entraînement ──────────────────────────────────────────────────────────────

def train(config_path: str, resume_path: str | None = None) -> None:
    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log.info("Device : %s", device)
    if device.type == "cuda":
        log.info("GPU    : %s (%.0f MB VRAM)", torch.cuda.get_device_name(0),
                 torch.cuda.get_device_properties(0).total_memory / 1e6)
    log.info("RAM au démarrage : %.0f MB", _ram_mb())

    t_cfg = cfg["training"]
    d_cfg = cfg["data"]

    ckpt_dir    = PROJECT_ROOT / t_cfg["checkpoint_dir"]
    samples_dir = PROJECT_ROOT / t_cfg["samples_dir"]
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    samples_dir.mkdir(parents=True, exist_ok=True)
    log.info("Checkpoints → %s", ckpt_dir)
    log.info("Samples     → %s", samples_dir)

    # Résoudre video_dirs (supporte ancienne clé video_dir pour compatibilité)
    raw_dirs = d_cfg.get("video_dirs") or [d_cfg.get("video_dir", "TemporelIntegrale")]
    video_dirs = [PROJECT_ROOT / d for d in raw_dirs]
    video_pattern = d_cfg.get("video_pattern", "*.avi")
    mask_path = PROJECT_ROOT / d_cfg["mask_path"]
    log.info("Vidéos      → %s (pattern=%s)", [str(d) for d in video_dirs], video_pattern)
    log.info("Masque      → %s", mask_path)

    # ── Lecture de la config multi-scale ─────────────────────────────────────
    ms_raw     = d_cfg.get("multi_scale", {})
    ms_enabled = bool(ms_raw.get("enabled", False))
    ms_scales  = ms_raw.get("scales", []) if ms_enabled else []

    # Construit les tables {taille: batch_size} et {taille: grad_clip}
    batch_sizes_map: dict[int, int]   = {}
    grad_clip_map:   dict[int, float] = {}
    for s_cfg in ms_scales:
        sz = int(s_cfg["size"])
        batch_sizes_map[sz] = int(s_cfg.get("batch_size", t_cfg["batch_size"]))
        grad_clip_map[sz]   = float(s_cfg.get("grad_clip", t_cfg.get("grad_clip", 1.0)))

    # scale_configs transmis au dataset (None = mode mono-échelle rétrocompat.)
    scale_configs_list: list[dict] | None = ms_scales if ms_enabled else None

    # samples_per_epoch basé sur training.batch_size × multiplier (référence)
    spe = t_cfg["batch_size"] * t_cfg["samples_per_epoch_multiplier"]
    if ms_enabled:
        log.info(
            "Multi-scale ACTIVÉ — %d échelles : %s",
            len(ms_scales),
            "  ".join(f"{s['size']}×{s['size']}(bs={batch_sizes_map[s['size']]})"
                      for s in ms_scales),
        )
    log.info("Samples/epoch train : %d (batch_ref=%d × mult=%d)",
             spe, t_cfg["batch_size"], t_cfg["samples_per_epoch_multiplier"])

    log.info("─" * 60)
    log.info("Création du dataset TRAIN…")
    recursive = t_cfg.get("recursive_video_scan", True)
    ds_kwargs = dict(
        video_dirs           = video_dirs,
        mask_path            = mask_path,
        patch_size           = d_cfg["patch_size"],
        sigma_noise          = d_cfg.get("sigma_noise", 4.4),
        poisson_scale        = d_cfg.get("poisson_scale", 0.5),
        video_pattern        = video_pattern,
        train_split          = d_cfg["train_split"],
        recursive            = recursive,
        force_strategy        = d_cfg.get("force_strategy", "auto"),
        ecc_max_px            = d_cfg.get("ecc_max_px", 3.0),
        min_pair_correlation  = d_cfg.get("min_pair_correlation", 0.85),
        ecc_validation_n      = d_cfg.get("ecc_validation_n", 20),
        temporal_offset       = d_cfg.get("temporal_offset", 5),
        scale_configs         = scale_configs_list,
    )
    train_ds = VascularVideoDataset(
        **ds_kwargs,
        split             = "train",
        samples_per_epoch = spe,
        augment           = True,
    )
    log.info("─" * 60)
    log.info("Création du dataset VAL…")
    val_ds = VascularVideoDataset(
        **ds_kwargs,
        split             = "val",
        samples_per_epoch = max(t_cfg["batch_size"] * 64, 512),
        augment           = False,
    )
    log.info("─" * 60)
    log.info("RAM après datasets : %.0f MB", _ram_mb())

    _dl_kwargs = dict(
        num_workers = t_cfg["num_workers"],
        pin_memory  = t_cfg["pin_memory"] and device.type == "cuda",
    )

    if ms_enabled:
        # drop_last=True : évite les batches partiels (ex. 1 sample sur les 819 du 512×512)
        # qui causent var(dim=0) indéfinie → NaN dans _pulsation_preservation_loss.
        train_sampler = MultiScaleBatchSampler(train_ds, batch_sizes_map, drop_last=True)
        val_sampler   = MultiScaleBatchSampler(val_ds,   batch_sizes_map, drop_last=True)
        train_dl = DataLoader(train_ds, batch_sampler=train_sampler, **_dl_kwargs)
        val_dl   = DataLoader(val_ds,   batch_sampler=val_sampler,   **_dl_kwargs)
        # Afficher la distribution prévue pour vérification au démarrage
        train_sampler.log_distribution(n_epochs=t_cfg["epochs"])
    else:
        train_dl = DataLoader(train_ds, batch_size=t_cfg["batch_size"], shuffle=True,  **_dl_kwargs)
        val_dl   = DataLoader(val_ds,   batch_size=t_cfg["batch_size"], shuffle=False, **_dl_kwargs)

    log.info("DataLoaders créés (num_workers=%d, multi_scale=%s)",
             t_cfg["num_workers"], ms_enabled)

    model     = FastDVDnet(features=cfg["model"]["features"]).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=t_cfg["lr"])
    warmup_ep = int(t_cfg.get("warmup_epochs", 0))
    scheduler = _build_scheduler_with_warmup(optimizer, warmup_ep, t_cfg["epochs"])
    log.info(
        "Scheduler : %s (warmup=%d ep → cosine %d ep)",
        "warmup+cosine" if warmup_ep > 0 else "cosine",
        warmup_ep, t_cfg["epochs"] - warmup_ep,
    )
    criterion = N2NLoss(
        w_l1                = t_cfg.get("w_l1",             0.70),
        w_ssim              = t_cfg.get("w_ssim",           0.30),
        w_gradient          = t_cfg.get("w_gradient",       0.20),
        lambda_temporal     = t_cfg["lambda_temporal"],
        lambda_pulsation    = t_cfg["lambda_pulsation"],
        pulsation_threshold = t_cfg["pulsation_threshold"],
    )
    use_amp        = t_cfg["mixed_precision"] and device.type == "cuda"
    grad_clip      = float(t_cfg.get("grad_clip", 1.0))   # valeur par défaut / fallback
    # torch.cuda.amp.GradScaler est déprécié depuis PyTorch 2.x
    scaler         = torch.amp.GradScaler("cuda", enabled=use_amp)

    log.info("Modèle FastDVDnet : %.2fM paramètres", sum(p.numel() for p in model.parameters()) / 1e6)
    log.info("Mixed precision AMP : %s  |  grad_clip=%.1f", use_amp, grad_clip)

    def _safe_mean(lst: list[float]) -> float:
        vals = [x for x in lst if math.isfinite(x)]
        return float(np.mean(vals)) if vals else float("nan")

    start_epoch, best_val = 1, math.inf
    val_loss_history: list[float] = []   # pour détecter les oscillations

    if resume_path and Path(resume_path).exists():
        ckpt = torch.load(resume_path, map_location=device)
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        try:
            scheduler.load_state_dict(ckpt["scheduler"])
        except Exception as exc:
            # Le type de scheduler a changé (ex. cosine → warmup+cosine) :
            # on repart du début du scheduler sans planter.
            log.warning("Scheduler state incompatible avec le checkpoint (%s) — scheduler réinitialisé", exc)
        start_epoch = ckpt["epoch"] + 1
        best_val    = ckpt.get("best_val", math.inf)
        log.info("Reprise depuis epoch %d (best val=%.6f)", start_epoch - 1, best_val)

    log.info("─" * 60)
    log.info("Vérification des paires N2N avant entraînement…")
    _save_pair_debug(train_dl, samples_dir)

    log.info("─" * 60)
    log.info("Début entraînement : epochs %d → %d", start_epoch, t_cfg["epochs"])
    log.info("RAM avant 1ère epoch : %.0f MB", _ram_mb())

    n_epochs = t_cfg["epochs"]

    epoch_bar = tqdm(
        range(start_epoch, n_epochs + 1),
        desc  = "Epochs",
        unit  = "ep",
        total = n_epochs,
        initial = start_epoch - 1,
        dynamic_ncols = True,
        colour = "cyan",
    )

    for epoch in epoch_bar:
        t0 = time.perf_counter()

        # ── Train ──────────────────────────────────────────────────────────
        model.train()
        train_l, train_s                                   = [], []
        scale_train_l: dict[int, list[float]]              = defaultdict(list)
        scale_train_s: dict[int, list[float]]              = defaultdict(list)

        batch_bar = tqdm(
            train_dl,
            desc          = f"  Train E{epoch:03d}",
            leave         = False,
            unit          = "batch",
            dynamic_ncols = True,
            colour        = "green",
        )
        nan_batches = 0
        for batch in batch_bar:
            frames     = batch["frames"].to(device)
            target     = batch["target"].to(device)
            sigma      = batch["sigma"].to(device)
            # patch_size homogène dans un batch (garanti par MultiScaleBatchSampler)
            current_ps = int(batch["patch_size"][0].item())
            clip_val   = grad_clip_map.get(current_ps, grad_clip)

            optimizer.zero_grad()

            # Forward AMP (float16 si disponible) — uniquement le réseau
            with torch.amp.autocast("cuda", enabled=use_amp):
                output = model(frames, sigma)

            # Loss TOUJOURS en float32 (évite NaN avec SSIM + Sobel sous AMP)
            loss, loss_parts = criterion(output, target)

            # ── NaN guard : skip le batch si la loss est NaN/Inf ──────────
            if not torch.isfinite(loss):
                nan_batches += 1
                optimizer.zero_grad()
                if nan_batches <= 3:
                    tqdm.write(
                        f"  ⚠ NaN/Inf loss à batch {len(train_l)} "
                        f"ps={current_ps} "
                        f"(L1={loss_parts['l1']:.4f} "
                        f"ssim={loss_parts['ssim']:.4f} "
                        f"grad={loss_parts['grad']:.4f}) — batch ignoré"
                    )
                continue

            # Backward + gradient clipping (adapté à la taille) + step
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)   # doit précéder clip_grad_norm_
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=clip_val)
            scaler.step(optimizer)
            scaler.update()

            lv = loss.item()
            train_l.append(lv)
            scale_train_l[current_ps].append(lv)
            with torch.no_grad():
                sv = _ssim(output.detach(), target)
                train_s.append(sv)
                scale_train_s[current_ps].append(sv)

            batch_bar.set_postfix(
                loss = f"{lv:.4f}",
                ps   = current_ps,
                L1   = f"{loss_parts['l1']:.4f}",
                ssim = f"{loss_parts['ssim']:.4f}",
                avg  = f"{float(np.mean(train_l)):.4f}",
            )

        batch_bar.close()
        if nan_batches > 0:
            tqdm.write(f"  ⚠ Epoch {epoch} : {nan_batches} batch(s) NaN ignorés sur {len(train_l)+nan_batches}")
        scheduler.step()

        # ── Validation ────────────────────────────────────────────────────
        model.eval()
        val_l, val_s                                       = [], []
        scale_val_l: dict[int, list[float]]                = defaultdict(list)
        scale_val_s: dict[int, list[float]]                = defaultdict(list)
        saved_by_scale: dict[int, tuple]                   = {}   # ps → (in, out, tgt)

        val_bar = tqdm(
            val_dl,
            desc          = f"  Val   E{epoch:03d}",
            leave         = False,
            unit          = "batch",
            dynamic_ncols = True,
            colour        = "yellow",
        )
        with torch.no_grad():
            for batch in val_bar:
                frames     = batch["frames"].to(device)
                target     = batch["target"].to(device)
                sigma      = batch["sigma"].to(device)
                current_ps = int(batch["patch_size"][0].item())
                output     = model(frames, sigma)
                loss, _    = criterion(output, target)
                lv         = loss.item()
                sv         = _ssim(output, target)
                val_l.append(lv)
                val_s.append(sv)
                scale_val_l[current_ps].append(lv)
                scale_val_s[current_ps].append(sv)
                # Garder un exemple par échelle pour la visualisation
                if current_ps not in saved_by_scale:
                    saved_by_scale[current_ps] = (frames.cpu(), output.cpu(), target.cpu())
                val_bar.set_postfix(
                    loss = f"{lv:.4f}",
                    ps   = current_ps,
                    avg  = f"{float(np.mean(val_l)):.4f}",
                )
        val_bar.close()

        mean_tl = _safe_mean(train_l)
        mean_vl = _safe_mean(val_l)
        mean_ts = _safe_mean(train_s)
        mean_vs = _safe_mean(val_s)
        epoch_s = time.perf_counter() - t0

        val_loss_history.append(mean_vl)

        # ── Alertes de surveillance ────────────────────────────────────────
        # Alerte epoch 35 : val_loss trop haute → probable problème dataset
        if epoch == 35 and mean_vl > 0.095:
            tqdm.write(
                f"\n⚠  ALERTE epoch 35 : val_loss={mean_vl:.5f} > 0.095\n"
                f"   → Probable problème de dataset (paires incohérentes ou sigma trop faible)\n"
                f"   → Vérifier debug_pairs_n2n.png dans {samples_dir}\n"
            )

        # Alerte oscillation : std élevée ET tendance montante ET epoch suffisante
        osc_threshold = t_cfg.get("oscillation_std_threshold", 0.010)
        osc_min_epoch = t_cfg.get("oscillation_min_epoch", 40)
        stagnation_p  = t_cfg.get("stagnation_patience", 15)

        if len(val_loss_history) >= 10 and epoch >= osc_min_epoch:
            recent    = np.array(val_loss_history[-10:])
            recent_std = float(np.std(recent))
            # Pente par régression linéaire : > 0 → tendance montante
            slope = float(np.polyfit(np.arange(10), recent, 1)[0])
            if recent_std > osc_threshold and slope > 0:
                tqdm.write(
                    f"\n⚠  ALERTE oscillation (epoch {epoch}) : "
                    f"std={recent_std:.4f} > {osc_threshold}  slope={slope:+.5f} (montant)\n"
                    f"   → Réduire le LR : config.yaml → lr: 5.0e-5\n"
                    f"   → Reprendre     : python train.py --resume checkpoints/last.pth\n"
                )

        # Alerte stagnation : pas d'amélioration pendant N epochs
        if (len(val_loss_history) >= stagnation_p
                and min(val_loss_history[-stagnation_p:]) >= best_val - 1e-6
                and epoch >= osc_min_epoch):
            tqdm.write(
                f"\n⚠  ALERTE stagnation (epoch {epoch}) : "
                f"pas d'amélioration depuis {stagnation_p} epochs "
                f"(best={best_val:.5f})\n"
                f"   → Vérifier debug_pairs_n2n.png, augmenter le dataset\n"
                f"   → Ou réduire lr : config.yaml → lr: 5.0e-5\n"
            )

        # ── Mise à jour barre d'epochs ─────────────────────────────────────
        is_best = mean_vl < best_val
        if is_best:
            best_val = mean_vl

        epoch_bar.set_postfix(
            tL1   = f"{mean_tl:.4f}",
            vL1   = f"{mean_vl:.4f}",
            best  = f"{best_val:.4f}",
            tSSIM = f"{mean_ts:.3f}" if not math.isnan(mean_ts) else "—",
            vSSIM = f"{mean_vs:.3f}" if not math.isnan(mean_vs) else "—",
            lr    = f"{optimizer.param_groups[0]['lr']:.1e}",
            s     = f"{epoch_s:.0f}s",
        )

        # ── Log textuel (toutes les N epochs) ─────────────────────────────
        if epoch % t_cfg["log_every_n_epochs"] == 0 or epoch == 1:
            inf_ms = _measure_inference_time(model, device)
            tqdm.write(
                f"[E{epoch:04d}/{n_epochs}] "
                f"Train L1={mean_tl:.5f} SSIM={mean_ts:.4f} | "
                f"Val L1={mean_vl:.5f} SSIM={mean_vs:.4f} | "
                f"{inf_ms:.1f}ms/f ({1000.0/max(inf_ms,1e-3):.1f}fps) | "
                f"LR={optimizer.param_groups[0]['lr']:.1e} | "
                f"{epoch_s:.0f}s"
                + (" ★ best" if is_best else "")
            )
            # ── Détail par échelle (multi-scale uniquement) ────────────────
            if ms_enabled and (scale_train_l or scale_val_l):
                all_scales = sorted(set(list(scale_train_l) + list(scale_val_l)))
                # Calculer les % en SAMPLES (batches × batch_size),
                # pas en batches — sinon le 512 paraît sur-représenté à cause
                # de ses petits batches (batch_size=2 vs 32 pour le 128).
                total_samples_t = sum(
                    len(v) * batch_sizes_map.get(s, t_cfg.get("batch_size", 8))
                    for s, v in scale_train_l.items()
                )
                total_samples_t = max(total_samples_t, 1)
                tqdm.write("  Détail par échelle :")
                for s in all_scales:
                    tl_vals  = scale_train_l.get(s, [])
                    vl_vals  = scale_val_l.get(s, [])
                    ts_vals  = scale_train_s.get(s, [])
                    vs_vals  = scale_val_s.get(s, [])
                    n_samp   = len(tl_vals) * batch_sizes_map.get(s, t_cfg.get("batch_size", 8))
                    pct      = n_samp / total_samples_t * 100
                    sl       = _safe_mean(tl_vals)
                    ss       = _safe_mean(ts_vals)
                    vl       = _safe_mean(vl_vals)
                    vs       = _safe_mean(vs_vals)
                    tqdm.write(
                        f"    {s:3d}×{s:3d} ({pct:.0f}%) : "
                        f"Train L1={sl:.5f}  SSIM={ss:.4f} | "
                        f"Val  L1={vl:.5f}  SSIM={vs:.4f}"
                    )
            # ── Sauvegarde des exemples visuels par échelle ────────────────
            for ps, saved_ps in saved_by_scale.items():
                _save_samples(*saved_ps, epoch=epoch, out_dir=samples_dir, scale=ps)

        # ── Checkpoints ───────────────────────────────────────────────────
        state = dict(
            epoch=epoch, model=model.state_dict(),
            optimizer=optimizer.state_dict(), scheduler=scheduler.state_dict(),
            best_val=best_val, train_loss=mean_tl, val_loss=mean_vl,
            train_ssim=mean_ts, val_ssim=mean_vs,
        )
        _save_checkpoint(state, ckpt_dir / "last.pth", is_best, ckpt_dir / "best_model.pth")
        if epoch % t_cfg["save_every_n_epochs"] == 0:
            torch.save(state, ckpt_dir / f"epoch_{epoch:04d}.pth")
            tqdm.write(f"  → Checkpoint périodique sauvegardé (epoch {epoch})")

    epoch_bar.close()
    tqdm.write(f"\nEntraînement terminé. Best val loss : {best_val:.6f}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Entraînement FastDVDnet N2N pour vidéos vasculaires.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Exemples :\n"
            "  python train.py\n"
            "  python train.py --video-dirs MesDonnees/ AutreBanque/\n"
            "  python train.py --video-dirs MesDonnees/ --mask MesDonnees/mask.png\n"
            "  python train.py --epochs 50 --batch-size 4 --lr 5e-5\n"
            "  python train.py --resume checkpoints/last.pth\n"
        ),
    )
    parser.add_argument("--config",  default=str(SCRIPT_DIR / "config.yaml"),
        help="Fichier de configuration YAML (défaut : config.yaml)")
    parser.add_argument("--resume",  default=None,
        help="Reprendre depuis un checkpoint (.pth)")

    # Overrides — ces arguments écrasent les valeurs du config.yaml si fournis
    parser.add_argument("--video-dirs", nargs="+", default=None,
        help="Un ou plusieurs dossiers de vidéos d'entraînement "
             "(relatifs à la racine du projet Stage/). "
             "Écrase data.video_dirs dans config.yaml.")
    parser.add_argument("--mask", default=None,
        help="Chemin vers le masque .png "
             "(relatif à Stage/). Écrase data.mask_path.")
    parser.add_argument("--epochs", type=int, default=None,
        help="Nombre d'epochs. Écrase training.epochs.")
    parser.add_argument("--batch-size", type=int, default=None,
        help="Taille de batch. Écrase training.batch_size.")
    parser.add_argument("--lr", type=float, default=None,
        help="Learning rate initial. Écrase training.lr.")
    parser.add_argument("--checkpoint-dir", default=None,
        help="Dossier de sauvegarde des checkpoints "
             "(relatif à Stage/). Écrase training.checkpoint_dir.")

    args = parser.parse_args()

    # Charger le config de base
    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    # Appliquer les overrides CLI
    if args.video_dirs is not None:
        cfg["data"]["video_dirs"] = args.video_dirs
        log.info("Override video_dirs → %s", args.video_dirs)
    if args.mask is not None:
        cfg["data"]["mask_path"] = args.mask
        log.info("Override mask_path  → %s", args.mask)
    if args.epochs is not None:
        cfg["training"]["epochs"] = args.epochs
        log.info("Override epochs     → %d", args.epochs)
    if args.batch_size is not None:
        cfg["training"]["batch_size"] = args.batch_size
        log.info("Override batch_size → %d", args.batch_size)
    if args.lr is not None:
        cfg["training"]["lr"] = args.lr
        log.info("Override lr         → %g", args.lr)
    if args.checkpoint_dir is not None:
        cfg["training"]["checkpoint_dir"] = args.checkpoint_dir
        log.info("Override ckpt_dir   → %s", args.checkpoint_dir)

    # Écrire le config résultant dans un fichier temporaire et lancer
    import tempfile, os
    with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as tmp:
        yaml.dump(cfg, tmp)
        tmp_path = tmp.name
    try:
        train(tmp_path, args.resume)
    finally:
        os.unlink(tmp_path)


if __name__ == "__main__":
    main()
