"""
n2n.py  (v4 — CORRIGÉE : multi-frames + loss structurelle stable)
==================================================================
Corrections majeures :
- U-Net plus profond (base=64, depth=4)
- Loss structurelle avec dimensions correctes
- Gradient clipping
- Normalisation des données
- Appariement par luminosité robuste
- Barre de progression propre (une seule ligne)
==================================================================
"""

from __future__ import annotations
import os
import glob
import json
import numpy as np
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from skimage.filters import frangi


# =====================================================================
# MODÈLE : U-Net résiduel (base=64, depth=4)
# =====================================================================

class _DoubleConv(nn.Module):
    def __init__(self, cin, cout):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(cin, cout, 3, padding=1),
            nn.BatchNorm2d(cout),
            nn.LeakyReLU(0.1, True),
            nn.Conv2d(cout, cout, 3, padding=1),
            nn.BatchNorm2d(cout),
            nn.LeakyReLU(0.1, True)
        )

    def forward(self, x):
        return self.net(x)


class UNet(nn.Module):
    def __init__(self, in_ch=5, base=64, depth=4):
        super().__init__()
        self.downs = nn.ModuleList()
        self.ups = nn.ModuleList()
        self.upconv = nn.ModuleList()
        
        cin = in_ch
        chs = []
        for d in range(depth):
            cout = base * (2 ** d)
            self.downs.append(_DoubleConv(cin, cout))
            chs.append(cout)
            cin = cout
        
        self.bottleneck = _DoubleConv(cin, cin * 2)
        cin = cin * 2
        
        for d in reversed(range(depth)):
            cout = chs[d]
            self.upconv.append(nn.ConvTranspose2d(cin, cout, 2, stride=2))
            self.ups.append(_DoubleConv(cin, cout))
            cin = cout
        
        self.out = nn.Conv2d(cin, 1, 1)
        self.pool = nn.MaxPool2d(2)

    def forward(self, x):
        center_idx = x.shape[1] // 2
        center = x[:, center_idx:center_idx+1]
        
        skips = []
        h = x
        for down in self.downs:
            h = down(h)
            skips.append(h)
            h = self.pool(h)
        
        h = self.bottleneck(h)
        
        for upconv, conv, skip in zip(self.upconv, self.ups, reversed(skips)):
            h = upconv(h)
            if h.shape[-2:] != skip.shape[-2:]:
                h = F.interpolate(h, size=skip.shape[-2:], mode="bilinear", align_corners=False)
            h = torch.cat([skip, h], dim=1)
            h = conv(h)
        
        return center + self.out(h)


# =====================================================================
# APPARIEMENT PAR LUMINOSITÉ
# =====================================================================

def build_brightness_pairs(stack: np.ndarray, min_pair_distance: int = 8,
                           max_pair_distance: int | None = 50,
                           smooth_win: int = 5) -> np.ndarray:
    """Apparie les frames par luminosité similaire."""
    T = stack.shape[0]
    if T < 2:
        return np.array([(0, 0)], dtype=np.int32)
    
    bright = stack.reshape(T, -1).mean(axis=1)
    if smooth_win > 1:
        k = np.ones(smooth_win, np.float32) / smooth_win
        bright = np.convolve(bright, k, mode="same")
    
    pairs = []
    for i in range(T):
        valid = np.ones(T, bool)
        lo = max(0, i - min_pair_distance + 1)
        hi = min(T, i + min_pair_distance)
        valid[lo:hi] = False
        if max_pair_distance:
            far = np.abs(np.arange(T) - i) > max_pair_distance
            valid[far] = False
        if not valid.any():
            continue
        diff = np.abs(bright - bright[i])
        diff[~valid] = np.inf
        best = int(np.argmin(diff))
        if best != i:
            pairs.append((i, best))
    
    if not pairs:
        pairs = [(i, i+1) for i in range(T-1)]
    
    pairs = np.array(pairs, dtype=np.int32)
    np.random.shuffle(pairs)
    return pairs


def _consecutive_pairs(T: int, gap: int = 1) -> np.ndarray:
    idx = np.arange(0, T - gap)
    return np.stack([idx, idx + gap], axis=1).astype(np.int32)


def vesselness_map(stack: np.ndarray) -> np.ndarray:
    """Carte de vesselness [0,1] sur la moyenne temporelle."""
    ves = frangi(stack.mean(axis=0), sigmas=(1, 2, 3), black_ridges=False)
    vmax = np.percentile(ves, 99.5) + 1e-6
    return np.clip(ves / vmax, 0, 1).astype(np.float32)


# =====================================================================
# DATASET : entrée multi-frames
# =====================================================================

class N2NDataset(Dataset):
    def __init__(self, stacks, pairs_list, ves_list, offsets=(-2, -1, 0, 1, 2),
                 patch=128, samples_per_epoch=2000):
        self.stacks = [np.clip(s.astype(np.float32), 0, 1) for s in stacks]
        self.pairs = [np.asarray(p, np.int32) for p in pairs_list]
        self.ves = [np.clip(v.astype(np.float32), 0, 1) for v in ves_list]
        self.offsets = list(offsets)
        self.patch = patch
        self.n = samples_per_epoch
        
        for i, s in enumerate(self.stacks):
            print(f"  Vidéo {i}: shape={s.shape}, min={s.min():.3f}, max={s.max():.3f}, "
                  f"paires={len(self.pairs[i])}")

    def __len__(self):
        return self.n

    def __getitem__(self, _):
        vi = np.random.randint(len(self.stacks))
        stk = self.stacks[vi]
        T, H, W = stk.shape
        i, j = self.pairs[vi][np.random.randint(len(self.pairs[vi]))]
        
        y0 = np.random.randint(0, H - self.patch)
        x0 = np.random.randint(0, W - self.patch)
        ys, xs = slice(y0, y0 + self.patch), slice(x0, x0 + self.patch)
        
        idxs = [min(max(i + o, 0), T - 1) for o in self.offsets]
        inp = np.stack([np.asarray(stk[k, ys, xs], np.float32) for k in idxs], 0)
        target = np.asarray(stk[j, ys, xs], np.float32)[None]
        center = np.asarray(stk[i, ys, xs], np.float32)[None]
        ves = np.asarray(self.ves[vi][ys, xs], np.float32)[None]
        
        k = np.random.randint(4)
        flip = np.random.rand() < 0.5
        
        def aug(a):
            a = np.rot90(a, k, axes=(-2, -1))
            if flip:
                a = a[..., ::-1]
            return np.ascontiguousarray(a)
        
        inp = aug(inp)
        target = aug(target)
        center = aug(center)
        ves = aug(ves)
        
        return (torch.from_numpy(inp),
                torch.from_numpy(target),
                torch.from_numpy(center),
                torch.from_numpy(ves))


# =====================================================================
# LOSS PRÉSERVANT LA STRUCTURE
# =====================================================================

def gradient_loss(pred, target, weight, eps=1e-6):
    gx_pred = pred[..., :, 1:] - pred[..., :, :-1]
    gy_pred = pred[..., 1:, :] - pred[..., :-1, :]
    gx_target = target[..., :, 1:] - target[..., :, :-1]
    gy_target = target[..., 1:, :] - target[..., :-1, :]
    
    wx = weight[..., :, 1:]
    wy = weight[..., 1:, :]
    
    loss_x = (wx * (gx_pred - gx_target).abs()).mean()
    loss_y = (wy * (gy_pred - gy_target).abs()).mean()
    return loss_x + loss_y


def background_smoothness(pred, weight, eps=1e-6):
    gx = pred[..., :, 1:] - pred[..., :, :-1]
    gy = pred[..., 1:, :] - pred[..., :-1, :]
    
    wx = (1 - weight)[..., :, 1:]
    wy = (1 - weight)[..., 1:, :]
    
    loss_x = (wx * gx.abs()).mean()
    loss_y = (wy * gy.abs()).mean()
    return loss_x + loss_y


def structure_loss(out, target, center, ves, 
                   w_struct=4.0, w_edge=0.15, w_bg=0.05):
    weight = 1.0 + w_struct * ves
    l_n2n = (weight * (out - target).abs()).mean()
    l_edge = gradient_loss(out, center, ves)
    l_bg = background_smoothness(out, ves)
    total = l_n2n + w_edge * l_edge + w_bg * l_bg
    return total, l_n2n, l_edge, l_bg


# =====================================================================
# INFÉRENCE
# =====================================================================

@torch.no_grad()
def _infer(model, stack, offsets, device, pad_mult=8):
    model.eval()
    T, H, W = stack.shape
    out = np.empty_like(stack)
    
    ph = (pad_mult - H % pad_mult) % pad_mult
    pw = (pad_mult - W % pad_mult) % pad_mult
    
    for t in tqdm(range(T), desc="Inférence N2N", ncols=100):
        idxs = [min(max(t + o, 0), T - 1) for o in offsets]
        inp = torch.from_numpy(np.stack([stack[k] for k in idxs], 0)[None]).float().to(device)
        inp = F.pad(inp, (0, pw, 0, ph), mode="reflect")
        y = model(inp)[..., :H, :W].squeeze().cpu().numpy()
        out[t] = np.clip(y, 0, 1)
    
    return out


# =====================================================================
# ENTRAÎNEMENT AVEC BARRE DE PROGRESSION PROPRE
# =====================================================================

def _train(model, ds, val_ref, device, epochs, lr, batch):
    dl = DataLoader(ds, batch_size=batch, shuffle=True, 
                    num_workers=2, drop_last=True)
    
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        opt, max_lr=lr, epochs=epochs,
        steps_per_epoch=len(dl), pct_start=0.2
    )
    
    val_stack, val_ves, offsets = val_ref
    
    print(f"\n{'='*80}")
    print(f"DÉBUT DE L'ENTRAÎNEMENT N2N")
    print(f"  Epochs: {epochs}")
    print(f"  Batch size: {batch}")
    print(f"  Learning rate: {lr}")
    print(f"  Device: {device}")
    print(f"{'='*80}\n")
    
    for ep in range(epochs):
        model.train()
        tot = n2n_loss = edge_loss = bg_loss = 0.0
        nb = 0
        
        # Barre de progression pour les batches (une seule ligne)
        batch_bar = tqdm(dl, desc=f"Epoch {ep+1}/{epochs}",
                         leave=False, dynamic_ncols=True, mininterval=0.3)
        
        for inp, target, center, ves in batch_bar:
            inp = inp.to(device)
            target = target.to(device)
            center = center.to(device)
            ves = ves.to(device)
            
            out = model(inp)
            out = torch.clamp(out, 0, 1)
            
            loss, l1, edge, bg = structure_loss(out, target, center, ves)
            
            if torch.isnan(loss) or torch.isinf(loss):
                batch_bar.write(f"  WARNING: NaN/Inf loss à l'epoch {ep+1}")
                continue
            
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 0.1)
            opt.step()
            scheduler.step()
            
            tot += loss.item()
            n2n_loss += l1.item()
            edge_loss += edge.item()
            bg_loss += bg.item()
            nb += 1
            
            # Mise à jour de la barre (une seule ligne)
            batch_bar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'n2n': f'{l1.item():.4f}',
                'edge': f'{edge.item():.4f}'
            })
        
        # Ferme proprement la barre AVANT d'écrire le résumé,
        # sinon le print entre en conflit avec la barre et défile en boucle.
        batch_bar.close()
        
        # Calcul des moyennes
        if nb > 0:
            avg_tot = tot / nb
            avg_n2n = n2n_loss / nb
            avg_edge = edge_loss / nb
            avg_bg = bg_loss / nb
            
            # Résumé de l'epoch (tqdm.write : compatible avec d'éventuelles barres)
            tqdm.write(f"[EPOCH {ep+1:3d}/{epochs}] "
                       f"Loss: {avg_tot:.5f} | "
                       f"N2N: {avg_n2n:.5f} | "
                       f"Edge: {avg_edge:.5f} | "
                       f"BG: {avg_bg:.5f} | "
                       f"LR: {scheduler.get_last_lr()[0]:.2e}")
    
    print(f"\n{'='*80}")
    print("ENTRAÎNEMENT TERMINÉ")
    print(f"{'='*80}\n")
    
    return model


# =====================================================================
# API PRINCIPALE
# =====================================================================

def denoise_stack_n2n(stack: np.ndarray, epochs: int = 200,
                      pairing: str = "brightness",
                      n_input_frames: int = 5,
                      min_pair_distance: int = 8,
                      max_pair_distance: int | None = 50,
                      patch: int = 128, lr: float = 1e-4,
                      batch: int = 16,
                      device: str = "cuda",
                      pretrained: str | None = None,
                      model_out: str | None = None,
                      save_video: bool = False,
                      video_path: str | None = None) -> np.ndarray:
    """Débruitage N2N d'une vidéo."""
    
    # Normalisation
    stack = np.clip(stack.astype(np.float32), 0, 1)
    device = device if torch.cuda.is_available() else "cpu"
    
    half = n_input_frames // 2
    offsets = tuple(range(-half, half + 1))
    
    # Chargement pré-entraîné
    if pretrained and os.path.exists(pretrained):
        model = UNet(in_ch=n_input_frames, base=64, depth=4).to(device)
        model.load_state_dict(torch.load(pretrained, map_location=device))
        print(f"  [N2N] Modèle pré-entraîné : {pretrained}")
        result = _infer(model, stack, offsets, device)
        if save_video and video_path:
            _save_denoised_video(result, video_path)
        return result
    
    # Construction des paires
    print(f"\n[N2N] Construction des paires...")
    if pairing == "brightness":
        pairs = build_brightness_pairs(stack, min_pair_distance, max_pair_distance)
    else:
        pairs = _consecutive_pairs(stack.shape[0], gap=1)
    
    if len(pairs) == 0:
        print("  [N2N] WARNING: Pas de paires !")
        pairs = _consecutive_pairs(stack.shape[0], gap=1)
    
    print(f"  [N2N] {len(pairs)} paires ({pairing}), {n_input_frames} frames d'entrée")
    print(f"  [N2N] Distance min entre paires: {min_pair_distance}")
    
    # Vesselness
    print(f"  [N2N] Calcul de la carte de vesselness...")
    ves = vesselness_map(stack)
    
    # Dataset
    ds = N2NDataset(
        [stack], [pairs], [ves],
        offsets=offsets,
        patch=patch,
        samples_per_epoch=batch * 32
    )
    
    # Modèle
    model = UNet(in_ch=n_input_frames, base=64, depth=4).to(device)
    print(f"  [N2N] Modèle: {sum(p.numel() for p in model.parameters()):,} paramètres")
    
    # Entraînement
    val_ref = (stack, ves, offsets)
    _train(model, ds, val_ref, device, epochs, lr, batch)
    
    # Sauvegarde
    if model_out:
        torch.save(model.state_dict(), model_out)
        print(f"\n  [N2N] Modèle sauvegardé : {model_out}")
    
    # Inférence
    result = _infer(model, stack, offsets, device)
    
    # Sauvegarde de la vidéo débruitee
    if save_video and video_path:
        _save_denoised_video(result, video_path)
    
    return result


def _save_denoised_video(stack: np.ndarray, video_path: str):
    """Sauvegarde la vidéo débruitee."""
    try:
        import vessel_pipeline as vp
        vp.save_video_gray(video_path, stack)
        print(f"  [N2N] Vidéo débruitee sauvegardée : {video_path}")
    except Exception as e:
        print(f"  [N2N] Erreur lors de la sauvegarde de la vidéo : {e}")


# =====================================================================
# PRÉ-TRAITEMENT POUR DOSSIER
# =====================================================================

def preprocess_for_n2n(video_path, cache_dir, motion="euclidean",
                       optical_flow=False, min_pair_distance=8,
                       max_pair_distance=50, max_frames=None) -> dict:
    import vessel_pipeline as vp
    
    os.makedirs(cache_dir, exist_ok=True)
    name = os.path.splitext(os.path.basename(video_path))[0]
    sp = os.path.join(cache_dir, f"{name}_stack.npy")
    pp = os.path.join(cache_dir, f"{name}_pairs.npy")
    vp_ = os.path.join(cache_dir, f"{name}_ves.npy")
    
    if os.path.exists(sp) and os.path.exists(pp) and os.path.exists(vp_):
        return {"stack": sp, "pairs": pp, "ves": vp_}
    
    print(f"  Pré-traitement: {os.path.basename(video_path)}")
    
    stack = vp.load_video_gray(video_path, max_frames=max_frames)
    nm = vp.NoiseModel.estimate(stack)
    stack = nm.forward(stack)
    vlo, vhi = np.percentile(stack, [0.5, 99.9])
    stack = ((stack - vlo) / (vhi - vlo + 1e-6)).astype(np.float32)
    stack = np.clip(stack, 0, 1)
    stack, _ = vp.register_stack(stack, motion=motion, optical_flow=optical_flow)
    
    np.save(sp, stack.astype(np.float32))
    np.save(pp, build_brightness_pairs(stack, min_pair_distance, max_pair_distance))
    np.save(vp_, vesselness_map(stack))
    
    return {"stack": sp, "pairs": pp, "ves": vp_}


def train_n2n_folder(folder, model_out, cache_dir=None, glob_pattern="*.avi",
                     epochs=300, patch=128, lr=1e-4, batch=16, device="cuda",
                     n_input_frames=5, motion="euclidean", optical_flow=False,
                     min_pair_distance=8, max_pair_distance=50,
                     samples_per_epoch=4000, max_frames=None) -> str:
    """Entraîne N2N sur un dossier de vidéos."""
    
    device = device if torch.cuda.is_available() else "cpu"
    cache_dir = cache_dir or os.path.join(os.path.dirname(model_out) or ".", "n2n_cache")
    
    videos = sorted(glob.glob(os.path.join(folder, glob_pattern)))
    if not videos:
        raise FileNotFoundError(f"Aucune vidéo {glob_pattern} dans {folder}")
    
    print(f"\n[N2N-dossier] {len(videos)} vidéos trouvées")
    print(f"[N2N-dossier] Cache: {cache_dir}")
    
    stacks, pairs, vess = [], [], []
    for v in tqdm(videos, desc="Pré-traitement", ncols=100):
        try:
            info = preprocess_for_n2n(v, cache_dir, motion=motion,
                                      optical_flow=optical_flow,
                                      min_pair_distance=min_pair_distance,
                                      max_pair_distance=max_pair_distance,
                                      max_frames=max_frames)
            stacks.append(np.load(info["stack"], mmap_mode="r"))
            pairs.append(np.load(info["pairs"]))
            vess.append(np.load(info["ves"]))
        except Exception as e:
            print(f"  ERREUR sur {v}: {e}")
            continue
    
    if not stacks:
        raise ValueError("Aucune vidéo valide !")
    
    half = n_input_frames // 2
    offsets = tuple(range(-half, half + 1))
    
    ds = N2NDataset(stacks, pairs, vess, offsets=offsets,
                    patch=patch, samples_per_epoch=samples_per_epoch)
    
    model = UNet(in_ch=n_input_frames, base=64, depth=4).to(device)
    print(f"\n  Modèle: {sum(p.numel() for p in model.parameters()):,} paramètres")
    
    val = (np.asarray(stacks[0], np.float32), vess[0], offsets)
    _train(model, ds, val, device, epochs, lr, batch)
    
    torch.save(model.state_dict(), model_out)
    print(f"\n[N2N-dossier] Modèle sauvegardé : {model_out}")
    
    return model_out


# =====================================================================
# CLI
# =====================================================================

if __name__ == "__main__":
    import argparse
    import vessel_pipeline as vp
    
    ap = argparse.ArgumentParser(description="Noise2Noise multi-frames")
    sub = ap.add_subparsers(dest="cmd", required=True)
    
    pt = sub.add_parser("train")
    pt.add_argument("folder")
    pt.add_argument("-m", "--model", default="n2n_model.pt")
    pt.add_argument("--glob", default="*.avi")
    pt.add_argument("--epochs", type=int, default=300)
    pt.add_argument("--batch", type=int, default=16)
    pt.add_argument("--lr", type=float, default=1e-4)
    pt.add_argument("--frames", type=int, default=5)
    pt.add_argument("--optical-flow", action="store_true")
    pt.add_argument("--max-frames", type=int, default=None)
    pt.add_argument("--min-distance", type=int, default=8)
    pt.add_argument("--device", default="cuda")
    
    pd = sub.add_parser("denoise")
    pd.add_argument("video")
    pd.add_argument("-m", "--model", default=None)
    pd.add_argument("--epochs", type=int, default=150)
    pd.add_argument("--frames", type=int, default=5)
    pd.add_argument("--pairing", default="brightness", choices=["brightness", "consecutive"])
    pd.add_argument("--min-distance", type=int, default=8)
    pd.add_argument("--device", default="cuda")
    pd.add_argument("--save-video", action="store_true", 
                    help="Sauvegarde la vidéo débruitee")
    
    a = ap.parse_args()
    
    if a.cmd == "train":
        train_n2n_folder(
            a.folder, a.model,
            glob_pattern=a.glob,
            epochs=a.epochs,
            batch=a.batch,
            lr=a.lr,
            n_input_frames=a.frames,
            optical_flow=a.optical_flow,
            max_frames=a.max_frames,
            min_pair_distance=a.min_distance,
            device=a.device
        )
    else:
        stack = vp.load_video_gray(a.video)
        stack, _ = vp.register_stack(stack)
        stack = np.clip(stack, 0, 1)
        
        video_out = None
        if a.save_video:
            base = os.path.splitext(a.video)[0]
            video_out = f"{base}_denoised_n2n.avi"
        
        den = denoise_stack_n2n(
            stack,
            epochs=a.epochs,
            pairing=a.pairing,
            n_input_frames=a.frames,
            min_pair_distance=a.min_distance,
            pretrained=a.model,
            device=a.device,
            save_video=a.save_video,
            video_path=video_out
        )
        
        vp.save_image("n2n_projection_test.png", 
                      vp.usm_mean_projection(den, usm_when="post"))
        print("OK -> n2n_projection_test.png")