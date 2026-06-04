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
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import yaml
from torch.cuda.amp import GradScaler
from torch.utils.data import DataLoader
from tqdm import tqdm

SCRIPT_DIR   = Path(__file__).resolve().parent   # Stage/Débruitage/
PROJECT_ROOT = SCRIPT_DIR.parent                 # Stage/

from model   import FastDVDnet
from dataset import VascularVideoDataset

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
    """Pénalise si var(output) < threshold × var(target) — préserve la pulsation cardiaque."""
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
) -> None:
    """Sauvegarde 3 PNGs comparatifs : input bruité | output réseau | target."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out_dir.mkdir(parents=True, exist_ok=True)
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
        fig.suptitle(f"Epoch {epoch} — exemple {i + 1}", fontsize=10)
        fig.tight_layout()
        fig.savefig(out_dir / f"epoch{epoch:04d}_sample{i + 1}.png", dpi=100)
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

    spe = t_cfg["batch_size"] * t_cfg["samples_per_epoch_multiplier"]
    log.info("Samples/epoch train : %d (batch=%d × mult=%d)",
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

    train_dl = DataLoader(train_ds, batch_size=t_cfg["batch_size"], shuffle=True,
                          num_workers=t_cfg["num_workers"],
                          pin_memory=t_cfg["pin_memory"] and device.type == "cuda")
    val_dl   = DataLoader(val_ds,   batch_size=t_cfg["batch_size"],
                          num_workers=t_cfg["num_workers"],
                          pin_memory=t_cfg["pin_memory"] and device.type == "cuda")
    log.info("DataLoaders créés (num_workers=%d)", t_cfg["num_workers"])

    model     = FastDVDnet(features=cfg["model"]["features"]).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=t_cfg["lr"])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=t_cfg["epochs"])
    criterion = N2NLoss(
        w_l1                = t_cfg.get("w_l1",             0.70),
        w_ssim              = t_cfg.get("w_ssim",           0.30),
        w_gradient          = t_cfg.get("w_gradient",       0.20),
        lambda_temporal     = t_cfg["lambda_temporal"],
        lambda_pulsation    = t_cfg["lambda_pulsation"],
        pulsation_threshold = t_cfg["pulsation_threshold"],
    )
    use_amp   = t_cfg["mixed_precision"] and device.type == "cuda"
    grad_clip = t_cfg.get("grad_clip", 1.0)
    scaler    = GradScaler(enabled=use_amp)

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
        scheduler.load_state_dict(ckpt["scheduler"])
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
        train_l, train_s = [], []

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
            frames = batch["frames"].to(device)
            target = batch["target"].to(device)
            sigma  = batch["sigma"].to(device)

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
                        f"(L1={loss_parts['l1']:.4f} "
                        f"ssim={loss_parts['ssim']:.4f} "
                        f"grad={loss_parts['grad']:.4f}) — batch ignoré"
                    )
                continue

            # Backward + gradient clipping + step
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)   # doit précéder clip_grad_norm_
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip)
            scaler.step(optimizer)
            scaler.update()

            train_l.append(loss.item())
            with torch.no_grad():
                train_s.append(_ssim(output.detach(), target))

            batch_bar.set_postfix(
                loss = f"{loss.item():.4f}",
                L1   = f"{loss_parts['l1']:.4f}",
                ssim = f"{loss_parts['ssim']:.4f}",
                grad = f"{loss_parts['grad']:.4f}",
                avg  = f"{float(np.mean(train_l)):.4f}",
            )

        batch_bar.close()
        if nan_batches > 0:
            tqdm.write(f"  ⚠ Epoch {epoch} : {nan_batches} batch(s) NaN ignorés sur {len(train_l)+nan_batches}")
        scheduler.step()

        # ── Validation ────────────────────────────────────────────────────
        model.eval()
        val_l, val_s, saved = [], [], None

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
                frames = batch["frames"].to(device)
                target = batch["target"].to(device)
                sigma  = batch["sigma"].to(device)
                output = model(frames, sigma)
                loss, _ = criterion(output, target)
                val_l.append(loss.item())
                val_s.append(_ssim(output, target))
                if saved is None:
                    saved = (frames.cpu(), output.cpu(), target.cpu())
                val_bar.set_postfix(
                    loss = f"{loss.item():.4f}",
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
            if saved is not None:
                _save_samples(*saved, epoch=epoch, out_dir=samples_dir)

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
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=str(SCRIPT_DIR / "config.yaml"))
    parser.add_argument("--resume", default=None)
    args = parser.parse_args()
    train(args.config, args.resume)


if __name__ == "__main__":
    main()
