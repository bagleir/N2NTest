"""
vessel_pipeline.py  (v2 — SANS MASQUE)
==================================================================
Débruitage + projection temporelle + rehaussement pour vidéos
d'angiographie en fluorescence (vaisseaux clairs sur fond sombre).

Changements v2 :
  * Suppression COMPLETE de la notion de masque (FOV / vaisseaux). Tout est
    traité sur l'image entière, comme le projet de référence.
  * Projection par défaut "usm_mean" : upscale Lanczos -> USM par frame ->
    moyenne temporelle (recette simple et robuste du projet de référence).
  * Rehaussement par défaut "light" (USM + gamma) ; chaîne lourde
    (tophat/CLAHE/Frangi) disponible en option 'full'.
  * Appariement N2N par luminosité (cf. n2n.py) : inspiré du projet fourni.

Étapes :
  0. Lecture vidéo (gris, float32 [0,1])
  1. (option) VST Anscombe généralisé
  2. (option) Recalage ECC
  3. Débruitage : temporal (classique) | n2n | none
  4. Projection : usm_mean (def.) | mip | mean | percentile
  5. Rehaussement : light (def.) | full | none
  6. Métriques no-reference + sauvegardes
==================================================================
"""

from __future__ import annotations
import os
import argparse
import json
import numpy as np
import cv2
from tqdm import tqdm

from skimage.filters import frangi, unsharp_mask
from skimage.restoration import denoise_nl_means, estimate_sigma, rolling_ball
from skimage.morphology import disk, white_tophat


# =====================================================================
# 0. LECTURE / ÉCRITURE
# =====================================================================
def load_video_gray(path: str, max_frames: int | None = None) -> np.ndarray:
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise IOError(f"Impossible d'ouvrir {path}")
    frames = []
    while True:
        ok, fr = cap.read()
        if not ok:
            break
        if fr.ndim == 3:
            fr = fr[..., 0]
        frames.append(fr.astype(np.float32) / 255.0)
        if max_frames and len(frames) >= max_frames:
            break
    cap.release()
    if not frames:
        raise IOError(f"Aucune frame lue dans {path}")
    return np.stack(frames, axis=0)


def save_image(path: str, img: np.ndarray, bits: int = 8):
    img = np.clip(img, 0.0, 1.0)
    if bits == 16:
        cv2.imwrite(path, (img * 65535.0 + 0.5).astype(np.uint16))
    else:
        cv2.imwrite(path, (img * 255.0 + 0.5).astype(np.uint8))


def save_video_gray(path: str, stack: np.ndarray, fps: float = 30.0):
    T, H, W = stack.shape
    vw = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"MJPG"), fps, (W, H), isColor=False)
    for t in range(T):
        vw.write((np.clip(stack[t], 0, 1) * 255 + 0.5).astype(np.uint8))
    vw.release()


# =====================================================================
# 1. MODÈLE DE BRUIT + VST (ANSCOMBE GÉNÉRALISÉ) — sans masque
# =====================================================================
class NoiseModel:
    """Var(pixel) = a * E[pixel] + b, estimé sur toute l'image."""

    def __init__(self, a: float = 1.0, b: float = 0.0):
        self.a = float(a); self.b = float(b)

    @classmethod
    def estimate(cls, stack: np.ndarray, n_bins: int = 50) -> "NoiseModel":
        mu = stack.mean(axis=0).ravel()
        var = stack.var(axis=0).ravel()
        order = np.argsort(mu)
        mu, var = mu[order], var[order]
        edges = np.linspace(mu.min(), mu.max(), n_bins + 1)
        bx, by = [], []
        for i in range(n_bins):
            sel = (mu >= edges[i]) & (mu < edges[i + 1])
            if sel.sum() < 20:
                continue
            bx.append(mu[sel].mean()); by.append(np.percentile(var[sel], 25))
        bx, by = np.array(bx), np.array(by)
        if len(bx) < 3:
            return cls(a=float(np.median(var)) * 2, b=0.0)
        A = np.vstack([bx, np.ones_like(bx)]).T
        a, b = np.linalg.lstsq(A, by, rcond=None)[0]
        return cls(a=max(a, 1e-8), b=max(b, 0.0))

    def forward(self, x: np.ndarray) -> np.ndarray:
        z = self.a * x + 3.0 / 8.0 * self.a ** 2 + self.b
        return (2.0 / self.a) * np.sqrt(np.maximum(z, 0.0))

    def inverse(self, y: np.ndarray) -> np.ndarray:
        z = (self.a / 2.0) * y
        return (z ** 2 - 3.0 / 8.0 * self.a ** 2 - self.b) / self.a


# =====================================================================
# 2. RECALAGE (sans masque)
# =====================================================================
def _reference_frame(stack: np.ndarray) -> np.ndarray:
    ref = np.median(stack, axis=0).astype(np.float32)
    return cv2.GaussianBlur(ref, (0, 0), 1.5)


def register_stack(stack: np.ndarray, motion: str = "euclidean",
                   optical_flow: bool = False,
                   ecc_iters: int = 60, ecc_eps: float = 1e-5
                   ) -> tuple[np.ndarray, list[np.ndarray]]:
    T, H, W = stack.shape
    ref = _reference_frame(stack)
    ref_s = cv2.GaussianBlur(ref, (0, 0), 1.0)
    warp_mode = {"translation": cv2.MOTION_TRANSLATION,
                 "euclidean": cv2.MOTION_EUCLIDEAN,
                 "affine": cv2.MOTION_AFFINE}[motion]
    criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, ecc_iters, ecc_eps)
    out = np.empty_like(stack); transforms = []
    for t in tqdm(range(T), desc="Recalage"):
        frame_s = cv2.GaussianBlur(stack[t], (0, 0), 1.0)
        W0 = np.eye(2, 3, dtype=np.float32)
        try:
            _, W0 = cv2.findTransformECC(ref_s, frame_s, W0, warp_mode,
                                         criteria, None, 5)
        except cv2.error:
            W0 = np.eye(2, 3, dtype=np.float32)
        warped = cv2.warpAffine(stack[t], W0, (W, H),
                                flags=cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP,
                                borderMode=cv2.BORDER_REFLECT)
        if optical_flow:
            flow = cv2.calcOpticalFlowFarneback(
                (ref * 255).astype(np.uint8), (warped * 255).astype(np.uint8),
                None, 0.5, 3, 21, 3, 5, 1.2, 0)
            gx, gy = np.meshgrid(np.arange(W), np.arange(H))
            warped = cv2.remap(warped, (gx + flow[..., 0]).astype(np.float32),
                               (gy + flow[..., 1]).astype(np.float32),
                               cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
        out[t] = warped; transforms.append(W0)
    return out, transforms


# =====================================================================
# 3. DÉBRUITAGE TEMPOREL CLASSIQUE
# =====================================================================
def temporal_denoise_stack(stack: np.ndarray, win: int = 9,
                           method: str = "trimmed") -> np.ndarray:
    T = stack.shape[0]; half = win // 2; out = np.empty_like(stack)
    for t in tqdm(range(T), desc=f"Débruitage temporel ({method})"):
        lo, hi = max(0, t - half), min(T, t + half + 1)
        block = stack[lo:hi]
        if method == "mean":
            out[t] = block.mean(0)
        elif method == "median":
            out[t] = np.median(block, 0)
        else:
            k = max(1, block.shape[0] // 5); s = np.sort(block, axis=0)
            out[t] = s[k:block.shape[0] - k].mean(0) if block.shape[0] > 2 * k else s.mean(0)
    return out


# =====================================================================
# 4. PROJECTION TEMPORELLE
# =====================================================================
def project(stack: np.ndarray, method: str = "mip",
            percentile: float = 90.0) -> np.ndarray:
    if method == "mip":
        return stack.max(axis=0)
    if method == "mean":
        return stack.mean(axis=0)
    if method == "percentile":
        return np.percentile(stack, percentile, axis=0)
    raise ValueError(method)


def usm_mean_projection(stack: np.ndarray, target_size: int = 1024,
                        usm_sigma: float = 2.0, usm_amount: float = 1.0,
                        usm_when: str = "post") -> np.ndarray:
    """Upscale Lanczos (target_size) + moyenne temporelle + USM.

    usm_when='post' (def.) : on moyenne d'ABORD les frames brutes upscalées,
        PUIS un seul USM. La moyenne supprime le bruit/les blocs JPEG par frame,
        donc l'USM n'amplifie plus l'artefact en damier de la zone saturée.
    usm_when='per_frame' : recette d'origine (USM sur chaque frame). À éviter
        ici car elle fige et amplifie les blocs du bolus saturé.
    """
    T = stack.shape[0]
    acc = np.zeros((target_size, target_size), np.float64)
    for t in tqdm(range(T), desc=f"Projection ({target_size}px, usm={usm_when})"):
        fr = cv2.resize(stack[t], (target_size, target_size),
                        interpolation=cv2.INTER_LANCZOS4)
        if usm_when == "per_frame":
            fr = unsharp_mask(np.clip(fr, 0, 1), radius=usm_sigma,
                              amount=usm_amount, preserve_range=True)
        acc += np.clip(fr, 0, 1)
    out = (acc / T).astype(np.float32)
    if usm_when == "post":
        out = np.clip(unsharp_mask(out, radius=usm_sigma, amount=usm_amount), 0, 1)
    return out


# =====================================================================
# 5. REHAUSSEMENT (sans masque)
# =====================================================================
def _normalize(img: np.ndarray, lo_p: float = 1.0, hi_p: float = 99.5) -> np.ndarray:
    lo, hi = np.percentile(img.ravel(), [lo_p, hi_p])
    if hi <= lo:
        hi = lo + 1e-6
    return np.clip((img - lo) / (hi - lo), 0, 1)


def gamma_correct(img: np.ndarray, g: float = 0.85) -> np.ndarray:
    return np.clip(np.power(np.clip(img, 0, 1), g), 0, 1)


def unsharp(img: np.ndarray, radius: float = 2.0, amount: float = 1.2) -> np.ndarray:
    return np.clip(unsharp_mask(np.clip(img, 0, 1), radius=radius, amount=amount), 0, 1)


def flat_field(img: np.ndarray, method: str = "tophat", radius: int = 60) -> np.ndarray:
    if method == "tophat":
        out = white_tophat(img.astype(np.float32), footprint=disk(radius))
    elif method == "rollingball":
        out = img - rolling_ball(img.astype(np.float32), radius=radius)
    else:
        out = img - cv2.GaussianBlur(img.astype(np.float32), (0, 0), radius / 2.0)
    return np.clip(out, 0, None)


def apply_clahe(img: np.ndarray, clip: float = 2.5, grid: int = 8,
                normalize: bool = True) -> np.ndarray:
    base = _normalize(img) if normalize else np.clip(img, 0, 1)
    u16 = (base * 65535).astype(np.uint16)
    return cv2.createCLAHE(clipLimit=clip, tileGridSize=(grid, grid)).apply(u16).astype(np.float32) / 65535.0


def estimate_sigma_mad(img: np.ndarray) -> float:
    """Écart-type du bruit ~ MAD du Laplacien (sans dépendance PyWavelets)."""
    lap = cv2.Laplacian(img.astype(np.float32), cv2.CV_32F)
    return float(1.4826 * np.median(np.abs(lap - np.median(lap))) / (6 ** 0.5) + 1e-6)


def nlm_denoise(img: np.ndarray, h_factor: float = 0.6) -> np.ndarray:
    """Non-Local Means léger pour lisser le fond (image déjà débruitée)."""
    s = estimate_sigma_mad(img)
    out = denoise_nl_means(img, h=h_factor * s, sigma=s,
                           patch_size=5, patch_distance=6, fast_mode=True)
    return np.clip(out, 0, 1)


def frangi_blend(img: np.ndarray, blend: float = 0.35,
                 sigmas=(1, 2, 3, 4, 5)) -> tuple[np.ndarray, np.ndarray]:
    ves = frangi(img, sigmas=sigmas, black_ridges=False)
    ves = _normalize(ves, 0, 99.5)
    return np.clip((1 - blend) * img + blend * ves, 0, 1), ves


def enhance(img: np.ndarray, cfg: dict) -> dict:
    """Profils :
      'soft' (def.) : normalize -> NLM léger -> Frangi (séparé) -> blend 70/30
                      -> CLAHE doux.  Recommandé (naturel, fond préservé).
      'light'       : normalize -> USM -> gamma.
      'full'        : normalize -> flat-field(top-hat) -> CLAHE -> Frangi -> USM
                      (agressif : écrase le fond, déconseillé).
      'none'        : normalize seulement.
    """
    profile = cfg.get("profile", "soft")
    outs = {}
    if profile == "none":
        outs["00_normalized"] = _normalize(img)
        outs["99_final"] = outs["00_normalized"].copy(); return outs
    if profile in ("light", "full"):
        x = _normalize(img); outs["00_normalized"] = x.copy()

    if profile == "soft":
        # Reproduit le "cas F" : NLM -> Frangi(1,2,3) normalisé par max
        # -> blend 70/30 -> CLAHE doux. PAS de re-stretch percentile.
        x = np.clip(img, 0, 1); outs["00_input"] = x.copy()
        den = nlm_denoise(x, h_factor=cfg.get("nlm_h", 0.8)); outs["10_nlm"] = den.copy()
        ves = frangi(den, sigmas=tuple(cfg.get("frangi_sigmas", (1, 2, 3))),
                     black_ridges=False)
        vmax = ves.max(); ves = (ves / vmax) if vmax > 0 else ves
        outs["20_vesselness"] = ves.copy()
        b = cfg.get("frangi_blend", 0.30)
        blended = np.clip((1 - b) * den + b * ves, 0, 1); outs["30_blend"] = blended.copy()
        x = apply_clahe(blended, cfg.get("clahe_clip", 1.5), cfg.get("clahe_grid", 32),
                        normalize=False)
        outs["99_final"] = np.clip(x, 0, 1); return outs

    if profile == "light":
        x = unsharp(x, radius=cfg.get("usm_radius", 2.0), amount=cfg.get("usm_amount", 1.2))
        outs["40_usm"] = x.copy()
        x = gamma_correct(x, g=cfg.get("gamma", 0.85))
        outs["99_final"] = x.copy(); return outs

    # full (agressif)
    x = _normalize(flat_field(x, cfg.get("ff_method", "tophat"), cfg.get("ff_radius", 60)))
    outs["10_flatfield"] = x.copy()
    x = apply_clahe(x, cfg.get("clahe_clip", 2.5), cfg.get("clahe_grid", 8))
    outs["20_clahe"] = x.copy()
    x, ves = frangi_blend(x, cfg.get("frangi_blend", 0.35),
                          tuple(cfg.get("frangi_sigmas", (1, 2, 3, 4, 5))))
    outs["30_frangi"] = x.copy(); outs["31_vesselness"] = ves
    x = unsharp(x, cfg.get("usm_radius", 2.0), cfg.get("usm_amount", 1.0))
    outs["99_final"] = np.clip(x, 0, 1); return outs


# =====================================================================
# 6. MÉTRIQUES NO-REFERENCE (sans masque)
# =====================================================================
def quality_metrics(img: np.ndarray) -> dict:
    img = np.asarray(img, dtype=np.float32)
    ves = frangi(img, sigmas=(1, 2, 3, 4), black_ridges=False)
    vthr = ves > np.percentile(ves, 90)
    bthr = ~vthr
    vmean = img[vthr].mean() if vthr.any() else 0.0
    bmean = img[bthr].mean() if bthr.any() else 0.0
    bstd = img[bthr].std() if bthr.any() else 1e-6
    gx = cv2.Sobel(img, cv2.CV_32F, 1, 0); gy = cv2.Sobel(img, cv2.CV_32F, 0, 1)
    return {
        "CNR_vaisseau_fond": float((vmean - bmean) / (bstd + 1e-6)),
        "nettete_tenengrad": float((gx ** 2 + gy ** 2).mean()),
        "reponse_frangi_totale": float(ves.sum()),
        "bruit_fond_std": float(bstd),
    }


# =====================================================================
# ORCHESTRATEUR
# =====================================================================
def run_pipeline(video_path: str, out_dir: str, cfg: dict):
    os.makedirs(out_dir, exist_ok=True)
    name = os.path.splitext(os.path.basename(video_path))[0]

    print(f"[0] Lecture {video_path}")
    stack = load_video_gray(video_path, max_frames=cfg.get("max_frames"))
    print(f"    pile {stack.shape}")

    nm = None
    if cfg.get("vst", True):
        nm = NoiseModel.estimate(stack)
        print(f"[1] Bruit : a={nm.a:.4g} b={nm.b:.4g}")
        stack = nm.forward(stack)
        vlo, vhi = np.percentile(stack, [0.5, 99.9])
        stack = ((stack - vlo) / (vhi - vlo + 1e-6)).astype(np.float32)
        _norm = (float(vlo), float(vhi))

    if cfg.get("register", True):
        print("[2] Recalage")
        stack, _ = register_stack(stack, motion=cfg.get("motion", "euclidean"),
                                  optical_flow=cfg.get("optical_flow", False))

    denoiser = cfg.get("denoiser", "temporal")
    if denoiser == "temporal":
        print("[3] Débruitage temporel")
        stack = temporal_denoise_stack(stack, win=cfg.get("temporal_win", 9),
                                       method=cfg.get("temporal_method", "trimmed"))
    elif denoiser == "n2n":
        print("[3] Débruitage Noise2Noise")
        try:
            import n2n
        except ImportError:
            raise SystemExit("n2n.py introuvable ou PyTorch non installé.")
        
        # Chemin pour la vidéo débruitee
        n2n_video_path = None
        if cfg.get("save_n2n_video", False):
            n2n_video_path = os.path.join(out_dir, f"{name}_denoised_n2n.avi")
        
        stack = n2n.denoise_stack_n2n(
            stack,
            epochs=cfg.get("n2n_epochs", 200),
            pairing=cfg.get("n2n_pairing", "brightness"),
            n_input_frames=cfg.get("n2n_input_frames", 5),
            min_pair_distance=cfg.get("n2n_min_distance", 8),
            lr=cfg.get("n2n_lr", 1e-4),
            batch=cfg.get("n2n_batch", 16),
            device=cfg.get("device", "cuda"),
            pretrained=cfg.get("n2n_pretrained"),
            model_out=os.path.join(out_dir, f"{name}_n2n.pt"),
            save_video=cfg.get("save_n2n_video", False),
            video_path=n2n_video_path
        )

    if nm is not None:
        stack = np.clip(nm.inverse(stack * (_norm[1] - _norm[0]) + _norm[0]), 0, 1)

    if cfg.get("save_stack", False):
        save_video_gray(os.path.join(out_dir, f"{name}_clean_stack.avi"), stack)

    # 4. Projection
    pmethod = cfg.get("projection", "usm_mean")
    print(f"[4] Projection : {pmethod}")
    if pmethod == "usm_mean":
        proj = usm_mean_projection(stack, target_size=cfg.get("target_size", 1024),
                                   usm_sigma=cfg.get("usm_sigma", 2.0),
                                   usm_amount=cfg.get("usm_amount_proj", 1.0),
                                   usm_when=cfg.get("usm_when", "post"))
    else:
        proj = np.clip(project(stack, pmethod, cfg.get("percentile", 90.0)), 0, 1)
    save_image(os.path.join(out_dir, f"{name}_projection.png"), proj, bits=16)
    # moyenne native 512 (référence de comparaison)
    save_image(os.path.join(out_dir, f"{name}_mean512.png"), np.clip(stack.mean(0), 0, 1))

    # 5. Rehaussement
    outs = enhance(proj, cfg.get("enhance", {}))
    for k, v in outs.items():
        save_image(os.path.join(out_dir, f"{name}_{k}.png"), v,
                   bits=16 if k == "99_final" else 8)

    # 6. Métriques
    metrics = quality_metrics(outs["99_final"])
    with open(os.path.join(out_dir, f"{name}_metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)
    print("[6] Métriques :", json.dumps(metrics, indent=2))
    print(f"Terminé -> {out_dir}")
    return outs["99_final"], metrics


# Mise à jour de DEFAULT_CFG
DEFAULT_CFG = dict(
    max_frames=None,
    vst=True,
    register=True, motion="euclidean", optical_flow=False,
    denoiser="temporal",
    temporal_win=9, temporal_method="trimmed",
    
    # Paramètres N2N
    n2n_epochs=200,
    n2n_pairing="brightness",
    n2n_input_frames=5,
    n2n_min_distance=8,
    n2n_lr=1e-4,
    n2n_batch=16,
    n2n_pretrained=None,
    save_n2n_video=False,  # NOUVEAU: sauvegarder la vidéo débruitee
    
    device="cuda",
    projection="usm_mean",
    target_size=1024, usm_sigma=2.0, usm_amount_proj=1.0, usm_when="post", percentile=90.0,
    save_stack=False,
    enhance=dict(profile="soft", nlm_h=0.8, frangi_blend=0.30,
                 clahe_clip=1.5, clahe_grid=32, frangi_sigmas=(1, 2, 3),
                 usm_radius=2.0, usm_amount=1.2, gamma=0.85,
                 ff_method="tophat", ff_radius=60),
)


def _build_cli():
    p = argparse.ArgumentParser(description="Pipeline vasculaire (sans masque)")
    p.add_argument("video")
    p.add_argument("-o", "--out", default="out_pipeline")
    p.add_argument("--denoiser", choices=["temporal", "n2n", "none"], default="temporal")
    p.add_argument("--projection", choices=["usm_mean", "mip", "mean", "percentile"],
                   default="usm_mean")
    p.add_argument("--enhance", choices=["soft", "light", "full", "none"], default="soft")
    p.add_argument("--no-register", action="store_true")
    p.add_argument("--optical-flow", action="store_true")
    p.add_argument("--motion", choices=["translation", "euclidean", "affine"],
                   default="euclidean")
    p.add_argument("--target-size", type=int, default=1024)
    p.add_argument("--save-stack", action="store_true")
    p.add_argument("--max-frames", type=int, default=None)
    p.add_argument("--device", default="cuda")
    p.add_argument("--model", default=None, help="modèle N2N pré-entraîné")
    p.add_argument("--save-n2n-video", action="store_true",
                   help="Sauvegarde la vidéo débruitee par N2N")
    return p


if __name__ == "__main__":
    a = _build_cli().parse_args()
    cfg = dict(DEFAULT_CFG)
    cfg.update(denoiser=a.denoiser, projection=a.projection,
               register=not a.no_register, optical_flow=a.optical_flow,
               motion=a.motion, target_size=a.target_size, save_stack=a.save_stack,
               max_frames=a.max_frames, device=a.device, n2n_pretrained=a.model,
               save_n2n_video=a.save_n2n_video)  # NOUVEAU
    cfg["enhance"] = dict(DEFAULT_CFG["enhance"])
    cfg["enhance"]["profile"] = a.enhance
    
    if a.model:
        cfg["denoiser"] = "n2n"
        cfg["n2n_pretrained"] = a.model
    
    run_pipeline(a.video, a.out, cfg)