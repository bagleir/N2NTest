"""
proj_net.py — Générateur de projection vasculaire auto-supervisé
=================================================================
Idée centrale (réponse au problème de sur-lissage du N2N par frame) :

  Au lieu de débruiter FRAME PAR FRAME (où la pulsatilité sur les vaisseaux est
  vue comme du bruit et moyennée -> lissage), on entraîne le réseau à générer
  DIRECTEMENT la projection vasculaire.

  Auto-supervision « Noise2Noise au niveau projection » :
    - on coupe aléatoirement les T frames (recalées) en deux moitiés A, B
      disjointes ;
    - proj_A et proj_B estiment la MÊME image vasculaire moyenne, avec un bruit
      d'échantillonnage INDÉPENDANT ;
    - on entraîne  net(proj_A) -> proj_B.
  L'optimum N2N est alors l'image vasculaire propre. Cette fois les vaisseaux
  sont le signal COMMUN aux deux moitiés (donc préservés), et le speckle est la
  partie variable (donc supprimée). Chaque tirage A/B = une paire => données
  quasi illimitées.

  Entrée multi-canaux :  [ mean(A) , gain * std(A) ]
    - mean : structure (perfusion moyenne)
    - std  : pulsatilité temporelle (~puissance Doppler) -> localise les petits
             vaisseaux que la simple moyenne efface. Le réseau apprend à s'en
             servir pour RECONSTRUIRE le fin, pas à le lisser.
  Cible : mean(B)  (estimateur non biaisé, indépendant du bruit de A).
  Sortie résiduelle : out = mean(A) + correction.

Réutilise vessel_pipeline.py pour : lecture, VST Anscombe, recalage, sauvegarde.

CLI :
  python proj_net.py train  <dossier> -m proj.pt [--frames-cap N] [--epochs 200]
  python proj_net.py infer  <video.avi> -m proj.pt [-o sortie.png] [--ensemble 8]
"""

from __future__ import annotations
import os, glob, argparse
import numpy as np
import cv2
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from skimage.filters import frangi

import vessel_pipeline as vp

STD_GAIN = 6.0  # ramène std (~0.04) à une magnitude comparable à mean (~0.3)


# ----------------------------------------------------------------------
# Reproductibilité
# ----------------------------------------------------------------------
def seed_everything(seed: int = 1234):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _worker_init_fn(wid: int):
    base = torch.initial_seed() % (2 ** 31 - 1)
    np.random.seed((base + wid) % (2 ** 31 - 1))


# ----------------------------------------------------------------------
# U-Net résiduel (identique en esprit à n2n.py : GroupNorm, résiduel)
# ----------------------------------------------------------------------
class _DoubleConv(nn.Module):
    def __init__(self, cin, cout):
        super().__init__()
        ng = max(1, min(8, cout // 8))
        self.net = nn.Sequential(
            nn.Conv2d(cin, cout, 3, padding=1), nn.GroupNorm(ng, cout),
            nn.LeakyReLU(0.1, True),
            nn.Conv2d(cout, cout, 3, padding=1), nn.GroupNorm(ng, cout),
            nn.LeakyReLU(0.1, True),
        )

    def forward(self, x):
        return self.net(x)


class ProjUNet(nn.Module):
    """Entrée (B, in_ch, H, W) ; canal 0 = mean(A). Sortie résiduelle sur ce canal."""

    def __init__(self, in_ch=2, base=48, depth=4):
        super().__init__()
        self.downs, self.ups, self.upconv = nn.ModuleList(), nn.ModuleList(), nn.ModuleList()
        cin, chs = in_ch, []
        for d in range(depth):
            cout = base * (2 ** d)
            self.downs.append(_DoubleConv(cin, cout)); chs.append(cout); cin = cout
        self.bottleneck = _DoubleConv(cin, cin * 2); cin *= 2
        for d in reversed(range(depth)):
            cout = chs[d]
            self.upconv.append(nn.ConvTranspose2d(cin, cout, 2, stride=2))
            self.ups.append(_DoubleConv(cin, cout)); cin = cout
        self.out = nn.Conv2d(cin, 1, 1)
        self.pool = nn.MaxPool2d(2)

    def forward(self, x):
        center = x[:, 0:1]  # mean(A)
        skips, h = [], x
        for down in self.downs:
            h = down(h); skips.append(h); h = self.pool(h)
        h = self.bottleneck(h)
        for up, conv, skip in zip(self.upconv, self.ups, reversed(skips)):
            h = up(h)
            if h.shape[-2:] != skip.shape[-2:]:
                h = F.interpolate(h, size=skip.shape[-2:], mode="bilinear", align_corners=False)
            h = conv(torch.cat([skip, h], 1))
        return center + self.out(h)


# ----------------------------------------------------------------------
# Pré-traitement (VST + normalisation + recalage), avec cache
# ----------------------------------------------------------------------
def preprocess(video_path, cache_dir, motion="euclidean",
               optical_flow=False, max_frames=None) -> str:
    os.makedirs(cache_dir, exist_ok=True)
    name = os.path.splitext(os.path.basename(video_path))[0]
    sp = os.path.join(cache_dir, f"{name}_reg.npy")
    if os.path.exists(sp):
        return sp
    stack = vp.load_video_gray(video_path, max_frames=max_frames)
    nm = vp.NoiseModel.estimate(stack)
    stack = nm.forward(stack)
    vlo, vhi = np.percentile(stack, [0.5, 99.9])
    stack = np.clip((stack - vlo) / (vhi - vlo + 1e-6), 0, 1).astype(np.float32)
    stack, _ = vp.register_stack(stack, motion=motion, optical_flow=optical_flow)
    np.save(sp, np.clip(stack, 0, 1).astype(np.float32))
    return sp


def vesselness_map(mean_img: np.ndarray) -> np.ndarray:
    ves = frangi(mean_img, sigmas=(1, 2, 3), black_ridges=False)
    return np.clip(ves / (np.percentile(ves, 99.5) + 1e-6), 0, 1).astype(np.float32)


# ----------------------------------------------------------------------
# Dataset : coupe A/B aléatoire + projections par patch
# ----------------------------------------------------------------------
class ProjDataset(Dataset):
    def __init__(self, stacks, ves_list, patch=128, samples_per_epoch=4000,
                 vessel_bias=0.7):
        self.stacks = stacks                       # mmap arrays (T,H,W)
        self.ves = [np.clip(v, 0, 1).astype(np.float32) for v in ves_list]
        self.patch = patch
        self.n = samples_per_epoch
        self.vessel_bias = vessel_bias

    def __len__(self):
        return self.n

    def _sample_patch(self, ves, H, W):
        p = self.patch
        for _ in range(8):
            y0 = np.random.randint(0, H - p); x0 = np.random.randint(0, W - p)
            if np.random.rand() > self.vessel_bias:
                return y0, x0
            if ves[y0:y0 + p, x0:x0 + p].mean() > 0.04:   # patch « avec vaisseau »
                return y0, x0
        return y0, x0

    def __getitem__(self, _):
        vi = np.random.randint(len(self.stacks))
        stk = self.stacks[vi]; T, H, W = stk.shape
        ves_full = self.ves[vi]
        y0, x0 = self._sample_patch(ves_full, H, W)
        ys, xs = slice(y0, y0 + self.patch), slice(x0, x0 + self.patch)

        block = np.asarray(stk[:, ys, xs], np.float32)    # (T,p,p), lecture patch
        perm = np.random.permutation(T)
        h = T // 2
        A, B = perm[:h], perm[h:]
        meanA = block[A].mean(0); stdA = block[A].std(0)
        meanB = block[B].mean(0)
        ves = ves_full[ys, xs]

        inp = np.stack([meanA, STD_GAIN * stdA], 0).astype(np.float32)
        tgt = meanB[None].astype(np.float32)
        ves = ves[None].astype(np.float32)

        k = np.random.randint(4); flip = np.random.rand() < 0.5

        def aug(a):
            a = np.rot90(a, k, axes=(-2, -1))
            if flip:
                a = a[..., ::-1]
            return np.ascontiguousarray(a)

        return (torch.from_numpy(aug(inp)),
                torch.from_numpy(aug(tgt)),
                torch.from_numpy(aug(ves)))


# ----------------------------------------------------------------------
# Loss : N2N pondéré vaisseau + accord haute-fréquence sur la cible (non biaisée)
# ----------------------------------------------------------------------
def _grads(x):
    gx = x[..., :, 1:] - x[..., :, :-1]
    gy = x[..., 1:, :] - x[..., :-1, :]
    return gx, gy


def proj_loss(out, target, ves, w_struct=4.0, w_hf=0.5, w_bg=0.03):
    weight = 1.0 + w_struct * ves
    l_main = (weight * (out - target).abs()).mean()

    # accord des gradients OUT vs CIBLE (mean_B, non biaisée) -> pousse le détail
    # fin sans s'ancrer sur une entrée bruitée. Pondéré vaisseau.
    gxo, gyo = _grads(out); gxt, gyt = _grads(target)
    wx, wy = ves[..., :, 1:], ves[..., 1:, :]
    l_hf = (wx * (gxo - gxt).abs()).mean() + (wy * (gyo - gyt).abs()).mean()

    # léger lissage du fond (hors vaisseaux)
    gxo2, gyo2 = _grads(out)
    l_bg = ((1 - ves)[..., :, 1:] * gxo2.abs()).mean() + \
           ((1 - ves)[..., 1:, :] * gyo2.abs()).mean()

    total = l_main + w_hf * l_hf + w_bg * l_bg
    return total, l_main, l_hf, l_bg


# ----------------------------------------------------------------------
# Entraînement
# ----------------------------------------------------------------------
def _build_val_batch(ds, n=48, seed=4321):
    st = np.random.get_state(); np.random.seed(seed)
    items = [ds[i] for i in range(n)]; np.random.set_state(st)
    return (torch.stack([x[0] for x in items]),
            torch.stack([x[1] for x in items]),
            torch.stack([x[2] for x in items]))


@torch.no_grad()
def _validate(model, vb, device):
    model.eval()
    inp, tgt, ves = [t.to(device) for t in vb]
    out = model(inp)
    _, l_main, _, _ = proj_loss(out, tgt, ves)
    return float(l_main.item())


def train(model, ds, device, epochs, lr, batch, grad_clip=1.0, patience=40):
    dl = DataLoader(ds, batch_size=batch, shuffle=True, num_workers=2,
                    drop_last=True, worker_init_fn=_worker_init_fn)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.OneCycleLR(
        opt, max_lr=lr, epochs=epochs, steps_per_epoch=len(dl), pct_start=0.2)
    vb = _build_val_batch(ds)

    print(f"\n{'='*72}\nENTRAÎNEMENT proj_net | epochs={epochs} batch={batch} "
          f"lr={lr} device={device}\n{'='*72}\n")

    best, best_state, no_imp = float("inf"), None, 0
    for ep in range(epochs):
        model.train(); tot = m = hf = bg = 0.0; nb = 0
        bar = tqdm(dl, desc=f"Epoch {ep+1}/{epochs}", leave=False, dynamic_ncols=True)
        for inp, tgt, ves in bar:
            inp, tgt, ves = inp.to(device), tgt.to(device), ves.to(device)
            out = model(inp)
            loss, l_main, l_hf, l_bg = proj_loss(out, tgt, ves)
            if not torch.isfinite(loss):
                continue
            opt.zero_grad(); loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            opt.step(); sched.step()
            tot += loss.item(); m += l_main.item(); hf += l_hf.item(); bg += l_bg.item(); nb += 1
            bar.set_postfix(loss=f"{loss.item():.4f}", main=f"{l_main.item():.4f}")
        bar.close()

        val = _validate(model, vb, device)
        imp = val < best - 1e-5
        if imp:
            best, no_imp = val, 0
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            no_imp += 1
        if nb:
            tqdm.write(f"[EPOCH {ep+1:3d}/{epochs}] Loss {tot/nb:.5f} | main {m/nb:.5f} "
                       f"| hf {hf/nb:.5f} | bg {bg/nb:.5f} | VAL {val:.5f}"
                       f"{'  <-- best' if imp else ''} | LR {sched.get_last_lr()[0]:.2e}")
        if no_imp >= patience:
            tqdm.write(f"  Early stopping (best VAL={best:.5f})."); break

    if best_state is not None:
        model.load_state_dict(best_state)
        tqdm.write(f"  Meilleur modèle restauré (VAL={best:.5f}).")
    return model


def train_folder(folder, model_out, cache_dir=None, glob_pat="*.avi",
                 epochs=200, patch=128, lr=2e-4, batch=16, base=48,
                 samples_per_epoch=4000, motion="euclidean", optical_flow=False,
                 max_frames=None, device="cuda"):
    device = device if torch.cuda.is_available() else "cpu"
    cache_dir = cache_dir or os.path.join(os.path.dirname(model_out) or ".", "proj_cache")
    videos = sorted(glob.glob(os.path.join(folder, glob_pat)))
    if not videos:
        raise FileNotFoundError(f"Aucune vidéo {glob_pat} dans {folder}")
    print(f"[proj_net] {len(videos)} vidéos | cache={cache_dir}")

    stacks, vess = [], []
    for v in tqdm(videos, desc="Pré-traitement", ncols=90):
        try:
            sp = preprocess(v, cache_dir, motion=motion,
                            optical_flow=optical_flow, max_frames=max_frames)
            stk = np.load(sp, mmap_mode="r")
            stacks.append(stk)
            vess.append(vesselness_map(np.asarray(stk).mean(0)))
        except Exception as e:
            print(f"  ERREUR {v}: {e}")
    if not stacks:
        raise ValueError("Aucune vidéo valide.")
    for i, s in enumerate(stacks):
        print(f"  Vidéo {i}: {s.shape}")

    ds = ProjDataset(stacks, vess, patch=patch, samples_per_epoch=samples_per_epoch)
    model = ProjUNet(in_ch=2, base=base, depth=4).to(device)
    print(f"  Modèle : {sum(p.numel() for p in model.parameters()):,} paramètres")
    train(model, ds, device, epochs, lr, batch)
    torch.save(model.state_dict(), model_out)
    print(f"[proj_net] Modèle sauvegardé : {model_out}")
    return model_out


# ----------------------------------------------------------------------
# Inférence : génère l'image vasculaire
# ----------------------------------------------------------------------
@torch.no_grad()
def _run(model, mean_img, std_img, device, pad_mult=16):
    model.eval()
    H, W = mean_img.shape
    inp = np.stack([mean_img, STD_GAIN * std_img], 0)[None].astype(np.float32)
    t = torch.from_numpy(inp).to(device)
    ph = (pad_mult - H % pad_mult) % pad_mult
    pw = (pad_mult - W % pad_mult) % pad_mult
    t = F.pad(t, (0, pw, 0, ph), mode="reflect")
    out = model(t)[..., :H, :W].squeeze().cpu().numpy()
    return np.clip(out, 0, 1)


@torch.no_grad()
def infer_video(video_path, model_path, out_png=None, ensemble=1, base=48,
                motion="euclidean", optical_flow=False, max_frames=None,
                device="cuda", cache_dir="proj_cache", save_diff=True):
    device = device if torch.cuda.is_available() else "cpu"
    sp = preprocess(video_path, cache_dir, motion=motion,
                    optical_flow=optical_flow, max_frames=max_frames)
    stk = np.asarray(np.load(sp), np.float32)
    T = stk.shape[0]

    model = ProjUNet(in_ch=2, base=base, depth=4).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))

    if ensemble <= 1:
        # projection pleine donnée (moins bruitée que pendant l'entraînement)
        result = _run(model, stk.mean(0), stk.std(0), device)
    else:
        # moyenne de plusieurs coupes aléatoires (robuste, ~distribution train)
        acc = np.zeros(stk.shape[1:], np.float64)
        for _ in range(ensemble):
            idx = np.random.permutation(T)[: T // 2]
            sub = stk[idx]
            acc += _run(model, sub.mean(0), sub.std(0), device)
        result = (acc / ensemble).astype(np.float32)

    base_name = os.path.splitext(os.path.basename(video_path))[0]
    out_png = out_png or f"{base_name}_projnet.png"
    vp.save_image(out_png, result, bits=16)
    print(f"[proj_net] Image générée : {out_png}")

    if save_diff:
        mean_proj = stk.mean(0)
        diff = np.clip(0.5 + 4.0 * (mean_proj - result), 0, 1)  # gris=identique
        vp.save_image(out_png.replace(".png", "_diff.png"), diff)
        vp.save_image(out_png.replace(".png", "_meanref.png"), np.clip(mean_proj, 0, 1))
        print(f"[proj_net] Diagnostic : *_diff.png (vs moyenne), *_meanref.png")
    return result


# ----------------------------------------------------------------------
# CLI
# ----------------------------------------------------------------------
if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Générateur de projection auto-supervisé")
    sub = ap.add_subparsers(dest="cmd", required=True)

    pt = sub.add_parser("train")
    pt.add_argument("folder")
    pt.add_argument("-m", "--model", default="proj_net.pt")
    pt.add_argument("--glob", default="*.avi")
    pt.add_argument("--epochs", type=int, default=200)
    pt.add_argument("--batch", type=int, default=16)
    pt.add_argument("--lr", type=float, default=2e-4)
    pt.add_argument("--patch", type=int, default=128)
    pt.add_argument("--base", type=int, default=48)
    pt.add_argument("--samples", type=int, default=4000)
    pt.add_argument("--optical-flow", action="store_true")
    pt.add_argument("--max-frames", type=int, default=None)
    pt.add_argument("--device", default="cuda")

    pi = sub.add_parser("infer")
    pi.add_argument("video")
    pi.add_argument("-m", "--model", required=True)
    pi.add_argument("-o", "--out", default=None)
    pi.add_argument("--ensemble", type=int, default=8)
    pi.add_argument("--base", type=int, default=48)
    pi.add_argument("--optical-flow", action="store_true")
    pi.add_argument("--max-frames", type=int, default=None)
    pi.add_argument("--device", default="cuda")

    a = ap.parse_args()
    seed_everything(1234)
    if a.cmd == "train":
        train_folder(a.folder, a.model, glob_pat=a.glob, epochs=a.epochs,
                     batch=a.batch, lr=a.lr, patch=a.patch, base=a.base,
                     samples_per_epoch=a.samples, optical_flow=a.optical_flow,
                     max_frames=a.max_frames, device=a.device)
    else:
        infer_video(a.video, a.model, out_png=a.out, ensemble=a.ensemble,
                    base=a.base, optical_flow=a.optical_flow,
                    max_frames=a.max_frames, device=a.device)