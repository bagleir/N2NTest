#!/usr/bin/env python3
"""
VascularVideoDataset — Dataset PyTorch pour l'entraînement N2N FastDVDnet.

Vidéos d'entrée : dossier TemporelIntegrale/ (*_step7_temporal_median.avi)

CHARGEMENT PARESSEUX : aucune frame n'est gardée en RAM au démarrage.
Les frames sont lues sur disque uniquement au moment du tirage (clip de 5 frames).
→ consommation RAM ≈ constante quelle que soit la taille du dataset.

Deux stratégies N2N :
  Stratégie A (inter-vidéo)  : input et target = deux acquisitions indépendantes
                                du même vaisseau → bruit indépendant ✓
  Stratégie B (intra-vidéo)  : target = frame centrale + bruit synthétique Gaussien
"""

from __future__ import annotations

import logging
import os
import random
import re
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

from mask_detection import load_mask

log = logging.getLogger(__name__)


# ── Helpers ───────────────────────────────────────────────────────────────────

def _ram_mb() -> float:
    """Retourne la RAM résidente du processus courant en MB (Linux)."""
    try:
        with open(f"/proc/{os.getpid()}/status") as f:
            for line in f:
                if line.startswith("VmRSS:"):
                    return float(line.split()[1]) / 1024.0
    except Exception:
        pass
    return 0.0


def _probe_video(path: Path) -> tuple[int, float]:
    """
    Ouvre la vidéo, lit uniquement les métadonnées (n_frames, fps), ferme.
    Ne charge aucune frame en mémoire.
    """
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise FileNotFoundError(f"Impossible d'ouvrir : {path}")
    fps      = cap.get(cv2.CAP_PROP_FPS) or 30.0
    n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    # cv2 peut sous-estimer — on corrige en lisant si nécessaire
    if n_frames <= 0:
        cap = cv2.VideoCapture(str(path))
        n = 0
        while cap.read()[0]:
            n += 1
        cap.release()
        n_frames = n
    return n_frames, fps


def _read_frames_at(path: Path, start: int, count: int = 5) -> np.ndarray | None:
    """
    Lit 'count' frames consécutives à partir de l'index 'start'.
    Retourne (count, H, W) float32, ou None si la lecture échoue.
    Ouvre et referme le fichier à chaque appel (pas de handle persistant).
    """
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        return None
    cap.set(cv2.CAP_PROP_POS_FRAMES, start)
    frames: list[np.ndarray] = []
    for _ in range(count):
        ret, frame = cap.read()
        if not ret:
            break
        if frame.ndim == 3:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        frames.append(frame.astype(np.float32))
    cap.release()
    return np.stack(frames) if len(frames) == count else None


def _read_frame_at(path: Path, idx: int) -> np.ndarray | None:
    """Lit une seule frame à l'index idx. Retourne (H, W) float32 ou None."""
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        return None
    cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
    ret, frame = cap.read()
    cap.release()
    if not ret:
        return None
    if frame.ndim == 3:
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    return frame.astype(np.float32)


# ── Détection de paires ───────────────────────────────────────────────────────

def _vessel_key(stem: str) -> str:
    """
    Extrait la clé d'identité du vaisseau (tout ce qui précède '_HD').

    Exemples :
      '260310_AUZ0752_10_HD_2_M0_step7_temporal_median' → '260310_AUZ0752_10_HD'
      '260307_VAB_L_1_HD_1_M0_step7_temporal_median'    → '260307_VAB_L_1_HD'
    """
    stem = stem.replace("_step7_temporal_median", "")
    m = re.match(r"^(.+_HD)", stem)
    return m.group(1) if m else stem


def detect_pairs(paths: list[Path]) -> dict[str, list[Path]]:
    """Regroupe une liste de chemins par clé de vaisseau. Groupes ≥ 2 → candidats stratégie A."""
    groups: dict[str, list[Path]] = defaultdict(list)
    for p in paths:
        groups[_vessel_key(p.stem)].append(p)
    return dict(groups)


# ── Augmentation ──────────────────────────────────────────────────────────────

def _augment(
    clip:   np.ndarray,
    target: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Flip H/V + rotation 90°/180°/270° identiques sur clip et target."""
    if random.random() < 0.5:
        clip   = clip[:, :, ::-1].copy()
        target = target[:, ::-1].copy()
    if random.random() < 0.5:
        clip   = clip[:, ::-1].copy()
        target = target[::-1].copy()
    k = random.randint(0, 3)
    if k > 0:
        clip   = np.stack([np.rot90(f, k) for f in clip])
        target = np.rot90(target, k).copy()
    return clip, target


def _sample_valid_patch(
    mask:         np.ndarray,
    patch_size:   int,
    min_coverage: float = 0.80,
    max_tries:    int   = 100,
) -> tuple[int, int] | None:
    H, W = mask.shape
    ps   = patch_size
    if H < ps or W < ps:
        return None
    for _ in range(max_tries):
        r = random.randint(0, H - ps)
        c = random.randint(0, W - ps)
        if mask[r:r+ps, c:c+ps].mean() / 255.0 >= min_coverage:
            return r, c
    return None


# ── Alignement ECC + validation de paire ─────────────────────────────────────

def _align_with_ecc(
    src:        np.ndarray,     # frame de référence (input) float32 [0,255]
    dst:        np.ndarray,     # frame cible à aligner float32 [0,255]
    mask:       np.ndarray,     # masque uint8
    max_px:     float = 3.0,    # rejet si déplacement > max_px
    max_iter:   int   = 30,
    eps:        float = 1e-3,
) -> tuple[np.ndarray | None, float]:
    """
    Aligne dst sur src par translation ECC.

    Retourne (dst_aligné, déplacement_px) ou (None, déplacement) si échec/rejet.
    Travaille sur une version downsampleée 4× pour la vitesse.
    """
    H, W = src.shape[:2]
    scale = 4
    src_s = cv2.resize(src.astype(np.uint8), (W // scale, H // scale))
    dst_s = cv2.resize(dst.astype(np.uint8), (W // scale, H // scale))
    mask_s = cv2.resize(mask, (W // scale, H // scale))

    warp = np.eye(2, 3, dtype=np.float32)
    criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, max_iter, eps)
    try:
        _, warp = cv2.findTransformECC(
            src_s, dst_s, warp, cv2.MOTION_TRANSLATION, criteria, mask_s, 1
        )
    except cv2.error:
        return None, float("inf")

    # Remettre à l'échelle originale
    warp[0, 2] *= scale
    warp[1, 2] *= scale
    displacement = float(np.hypot(warp[0, 2], warp[1, 2]))

    if displacement > max_px:
        return None, displacement

    aligned = cv2.warpAffine(
        dst.astype(np.float32), warp, (W, H), flags=cv2.INTER_LINEAR
    )
    return aligned, displacement


def _masked_correlation(
    a:    np.ndarray,
    b:    np.ndarray,
    mask: np.ndarray,
) -> float:
    """Corrélation de Pearson entre a et b à l'intérieur du masque."""
    m = mask.astype(bool)
    af = a[m].astype(np.float64)
    bf = b[m].astype(np.float64)
    if len(af) < 10:
        return 0.0
    ac = af - af.mean();  bc = bf - bf.mean()
    denom = np.sqrt((ac ** 2).sum() * (bc ** 2).sum())
    return float(np.dot(ac, bc) / denom) if denom > 1e-8 else 0.0


# ── Wrapper vidéo (lazy) ──────────────────────────────────────────────────────

class _VideoData:
    """
    Référence paresseuse vers une vidéo.

    Stocke uniquement le chemin, le nombre de frames et le fps.
    Aucune frame n'est chargée en mémoire — toutes les lectures se font
    sur disque à la demande via _read_frames_at() / _read_frame_at().
    """

    __slots__ = ("path", "n_frames", "fps")

    def __init__(self, path: Path, n_frames: int, fps: float) -> None:
        self.path     = path
        self.n_frames = n_frames
        self.fps      = fps

    def clip(self, center: int, half: int = 2) -> np.ndarray | None:
        """Lit 5 frames depuis le disque, centrées sur 'center'."""
        lo = center - half
        if lo < 0 or center + half >= self.n_frames:
            return None
        return _read_frames_at(self.path, lo, 5)

    def frame_at(self, idx: int) -> np.ndarray | None:
        """Lit une seule frame depuis le disque."""
        return _read_frame_at(self.path, idx)


# ── Dataset ───────────────────────────────────────────────────────────────────

class VascularVideoDataset(Dataset):
    """
    Dataset PyTorch N2N pour FastDVDnet. Chargement paresseux — RAM constante.

    Chaque sample retourne :
        frames : (5, patch_size, patch_size) float32 [0, 1]
        target : (1, patch_size, patch_size) float32 [0, 1]
        sigma  : (1, patch_size, patch_size) float32
    """

    def __init__(
        self,
        video_dirs:          list[str | Path] | str | Path,
        mask_path:           str | Path,
        patch_size:          int   = 128,
        sigma_noise:         float = 4.4,
        poisson_scale:       float = 0.5,
        video_pattern:       str   = "*.avi",
        split:               str   = "train",
        train_split:         float = 0.80,
        samples_per_epoch:   int   = 4096,
        augment:             bool  = True,
        seed:                int   = 42,
        recursive:           bool  = True,
        force_strategy:       str   = "auto",   # "auto" | "A" | "B"
        ecc_max_px:           float = 3.0,
        min_pair_correlation: float = 0.85,
        ecc_validation_n:     int   = 20,
        temporal_offset:      int   = 5,        # frames entre input et target (Option 1)
    ) -> None:
        super().__init__()
        self.patch_size           = patch_size
        self.sigma_norm           = sigma_noise / 255.0
        self.poisson_scale        = poisson_scale
        self.split                = split
        self.augment              = augment and (split == "train")
        self.samples_per_epoch    = samples_per_epoch
        self.force_strategy       = force_strategy.upper()
        self.ecc_max_px           = ecc_max_px
        self.min_pair_correlation = min_pair_correlation
        self.temporal_offset      = temporal_offset

        # Normaliser video_dirs en liste de Path absolus
        if isinstance(video_dirs, (str, Path)):
            video_dirs = [video_dirs]
        dirs = [Path(d).resolve() for d in video_dirs]

        log.info("[%s] Chargement du masque : %s", split, mask_path)
        self.mask = load_mask(mask_path)
        log.info("[%s] Masque : %dx%d | RAM : %.0f MB", split, *self.mask.shape, _ram_mb())

        # ── Scan multi-dossiers ───────────────────────────────────────────
        glob_pattern = ("**/" + video_pattern) if recursive else video_pattern
        all_paths: list[Path] = []
        for d in dirs:
            found = sorted(d.glob(glob_pattern))
            log.info("[%s] %s → %d fichiers (%s)", split, d, len(found), glob_pattern)
            all_paths.extend(found)

        # Déduplication (au cas où deux dossiers se chevauchent)
        seen: set[str] = set()
        deduped: list[Path] = []
        for p in all_paths:
            if str(p) not in seen:
                seen.add(str(p))
                deduped.append(p)
        all_paths = sorted(deduped)

        log.info("[%s] Total après déduplication : %d fichiers", split, len(all_paths))

        if not all_paths:
            raise RuntimeError(
                f"Aucune vidéo trouvée (pattern='{video_pattern}') dans :\n"
                + "\n".join(f"  {d}" for d in dirs)
            )

        # Log de toutes les vidéos trouvées (avant split)
        log.info("[%s] Vidéos trouvées :", split)
        for p in all_paths:
            log.info("[%s]   %s", split, p.name)

        # ── Split train/val (vidéos entières) ─────────────────────────────
        rng = random.Random(seed)
        shuffled = list(all_paths)
        rng.shuffle(shuffled)
        n_train   = max(1, int(len(shuffled) * train_split))
        train_paths = shuffled[:n_train]
        val_paths   = shuffled[n_train:]
        split_paths = train_paths if split == "train" else val_paths
        split_set   = set(str(p) for p in split_paths)

        log.info("[%s] Split %s : %d/%d vidéos", split, split, len(split_paths), len(all_paths))
        log.info("[%s] Vidéos dans ce split :", split)
        for p in sorted(split_paths, key=lambda x: x.name):
            log.info("[%s]   ✓ %s", split, p.name)
        excluded = [p for p in all_paths if str(p) not in split_set]
        if excluded:
            log.info("[%s] Vidéos dans l'AUTRE split (%d) :", split, len(excluded))
            for p in sorted(excluded, key=lambda x: x.name):
                log.info("[%s]   ✗ %s", split, p.name)

        # ── Probe des vidéos (métadonnées seulement, pas de frames) ───────
        log.info("[%s] Lecture des métadonnées (probe rapide, aucune frame chargée)…", split)
        probed: dict[str, _VideoData] = {}
        total_frames = 0
        for i, p in enumerate(all_paths):
            if str(p) not in split_set:
                continue
            n_frames, fps = _probe_video(p)
            probed[str(p)] = _VideoData(p, n_frames, fps)
            total_frames += n_frames
            if (i + 1) % 10 == 0 or i == len(all_paths) - 1:
                log.info(
                    "[%s]   %d/%d vidéos sondées | RAM : %.0f MB",
                    split, len(probed), len(split_set), _ram_mb(),
                )

        est_gb = total_frames * 512 * 512 / 1e9
        sep = "─" * 55
        log.info("[%s] %s", split, sep)
        log.info("[%s] DATASET SUMMARY (%s)", split, split.upper())
        log.info("[%s] %s", split, sep)
        log.info("[%s] Dossiers scannés   : %d", split, len(dirs))
        for d in dirs:
            log.info("[%s]   %s", split, d)
        log.info("[%s] Vidéos trouvées    : %d  (pattern=%s)", split, len(all_paths), video_pattern)
        log.info("[%s] Vidéos %s (%.0f%%): %d",
                 split, split, (train_split if split == "train" else 1-train_split)*100, len(probed))
        for vd in sorted(probed.values(), key=lambda v: v.path.name):
            log.info("[%s]   %-55s %d frames", split, vd.path.name, vd.n_frames)
        log.info("[%s] Frames total       : %d  (~%.1f GB si float32 en RAM)",
                 split, total_frames, est_gb)
        log.info("[%s] Samples/epoch      : %d", split, samples_per_epoch)
        log.info("[%s] %s", split, sep)
        log.info("[%s] Chargement paresseux : RAM = %.0f MB", split, _ram_mb())

        # ── Détection des paires ──────────────────────────────────────────
        groups = detect_pairs(all_paths)
        self.a_pairs: list[tuple[_VideoData, _VideoData]] = []
        self.b_clips: list[_VideoData] = []
        a_vessels: list[str] = []

        for key, paths in groups.items():
            in_split = [p for p in paths if str(p) in split_set]
            if len(in_split) >= 2:
                for i, p1 in enumerate(in_split):
                    for p2 in in_split[i + 1:]:
                        self.a_pairs.append((probed[str(p1)], probed[str(p2)]))
                a_vessels.append(key)
            else:
                for p in in_split:
                    if str(p) in probed:
                        self.b_clips.append(probed[str(p)])

        log.info(
            "[%s] Stratégie A : %d paires (%d vaisseaux) | Stratégie B : %d clips",
            split, len(self.a_pairs), len(a_vessels), len(self.b_clips),
        )
        if self.a_pairs:
            log.info(
                "[%s] Exemples paires A : %s%s",
                split,
                " | ".join(
                    f"{v1.path.name[:30]} ↔ {v2.path.name[:30]}"
                    for v1, v2 in self.a_pairs[:3]
                ),
                " ..." if len(self.a_pairs) > 3 else "",
            )
        if not self.a_pairs and not self.b_clips:
            raise RuntimeError(f"[{split}] Aucune vidéo utilisable dans ce split !")

        # ── Validation ECC (échantillon de paires A) ──────────────────────
        if self.a_pairs and self.force_strategy != "B":
            n_test = min(ecc_validation_n, len(self.a_pairs))
            sample_pairs = random.sample(self.a_pairs, n_test)
            accepted, rejected_disp, rejected_corr = 0, 0, 0
            displacements: list[float] = []
            correlations:  list[float] = []

            log.info("[%s] Validation ECC sur %d paires A (downsample 4×)…", split, n_test)
            for v1, v2 in sample_pairs:
                center = min(v1.n_frames, v2.n_frames) // 2
                f1 = v1.frame_at(center)
                f2 = v2.frame_at(center)
                if f1 is None or f2 is None:
                    rejected_disp += 1
                    continue
                aligned, disp = _align_with_ecc(f1, f2, self.mask, ecc_max_px)
                if aligned is None:
                    rejected_disp += 1
                    log.debug("[%s]   ✗ déplacement=%.1f px > %.1f px  %s↔%s",
                              split, disp, ecc_max_px, v1.path.name[:30], v2.path.name[:30])
                    continue
                corr = _masked_correlation(f1, aligned, self.mask)
                if corr < min_pair_correlation:
                    rejected_corr += 1
                    log.debug("[%s]   ✗ corrélation=%.3f < %.2f  %s↔%s",
                              split, corr, min_pair_correlation, v1.path.name[:30], v2.path.name[:30])
                else:
                    accepted += 1
                    displacements.append(disp)
                    correlations.append(corr)

            log.info(
                "[%s] Résultat ECC (%d paires testées) :\n"
                "[%s]   ✓ Acceptées        : %d\n"
                "[%s]   ✗ Rejetées (disp.) : %d  (déplacement > %.1f px)\n"
                "[%s]   ✗ Rejetées (corr.) : %d  (corrélation < %.2f)\n"
                "[%s]   Déplacement moyen  : %.2f px\n"
                "[%s]   Corrélation moyenne: %.3f",
                split, n_test,
                split, accepted,
                split, rejected_disp, ecc_max_px,
                split, rejected_corr, min_pair_correlation,
                split, float(np.mean(displacements)) if displacements else float("nan"),
                split, float(np.mean(correlations))  if correlations  else float("nan"),
            )
            accept_rate = accepted / max(n_test, 1)
            if accept_rate < 0.5:
                log.warning(
                    "[%s] ⚠ Moins de 50%% des paires A sont valides (%.0f%%) !\n"
                    "[%s]   → Garder force_strategy: 'B' dans config.yaml\n"
                    "[%s]   → Les vidéos ne sont probablement pas alignées spatialement.",
                    split, accept_rate * 100, split, split,
                )

        strat_used = self.force_strategy if self.force_strategy in ("A", "B") else "auto"
        log.info("[%s] Stratégie active   : %s", split, strat_used)
        log.info("[%s] Stratégie A pairs  : %d  |  Stratégie B clips : %d",
                 split, len(self.a_pairs), len(self.b_clips))
        self._validate_pair_stats()
        log.info("[%s] Dataset prêt | %d samples/epoch | RAM : %.0f MB", split, samples_per_epoch, _ram_mb())

    def __len__(self) -> int:
        return self.samples_per_epoch

    def _sample_a(self) -> tuple[np.ndarray, np.ndarray] | None:
        """
        Tire une paire A, aligne par ECC, valide par corrélation.
        Retente jusqu'à 10 fois avant d'abandonner.
        """
        for _ in range(10):
            src1, src2 = random.choice(self.a_pairs)
            if random.random() < 0.5:
                src1, src2 = src2, src1
            n = min(src1.n_frames, src2.n_frames)
            if n < 5:
                continue
            center = random.randint(2, n - 3)
            clip   = src1.clip(center)
            target = src2.frame_at(center)
            if clip is None or target is None:
                continue

            # Alignement ECC (translation)
            aligned, disp = _align_with_ecc(
                clip[2], target, self.mask, self.ecc_max_px
            )
            if aligned is None:
                continue   # déplacement trop grand

            # Validation corrélation
            if _masked_correlation(clip[2], aligned, self.mask) < self.min_pair_correlation:
                continue   # paire incohérente

            return clip, aligned

        return None   # toutes les tentatives ont échoué → fallback sur B

    def _add_one_noise(self, frame: np.ndarray) -> np.ndarray:
        """Un seul tirage indépendant de bruit gaussien + Poisson sur frame."""
        sigma_g = self.sigma_norm * 255.0
        gaussian = np.random.normal(0.0, sigma_g, frame.shape).astype(np.float32)
        if self.poisson_scale > 0:
            lam  = np.maximum(frame * self.poisson_scale, 0.0)
            pois = (np.random.poisson(lam).astype(np.float32)
                    / max(self.poisson_scale, 1e-6)) - frame
        else:
            pois = np.zeros_like(frame)
        return np.clip(frame + gaussian + pois * 0.3, 0.0, 255.0).astype(np.float32)

    def _sample_b(self) -> tuple[np.ndarray, np.ndarray]:
        """
        Construit une paire N2N pour la stratégie B.

        Option 1 — Paire temporelle (recommandée, sans bruit synthétique) :
            Input  = fenêtre 5 frames centrée sur t  [t-2 … t+2]
            Target = frame à t + temporal_offset      (hors fenêtre → bruit indépendant)
            Condition N2N satisfaite : bruit du capteur à deux instants différents.

        Option 2 — Double bruit indépendant (fallback si vidéo trop courte) :
            base   = frame t  (bruit naturel du capteur)
            Input  = base + noise_1  (tirage indépendant 1)
            Target = base + noise_2  (tirage indépendant 2)
            Les 5 frames d'input reçoivent toutes du bruit indépendant.
            Input et Target ont le même niveau de bruit (symétrie garantie).

        JAMAIS : target = input + bruit_extra  ← crée une asymétrie et le réseau apprend à bruiter.
        """
        src    = random.choice(self.b_clips)
        offset = self.temporal_offset  # default 5

        # ── Option 1 : paire temporelle ──────────────────────────────────
        # Besoin de t ≥ 2  et  t + offset ≤ n-1  et  t ≤ n-3 (fenêtre)
        max_center = src.n_frames - 1 - offset
        if max_center > 2:
            for _ in range(20):
                center = random.randint(2, min(max_center, src.n_frames - 3))
                clip   = src.clip(center)
                target = src.frame_at(center + offset)
                if clip is not None and target is not None:
                    return clip, target  # bruit naturel du capteur, indépendant

        # ── Option 2 : double bruit indépendant (fallback) ────────────────
        clip: np.ndarray | None = None
        for _ in range(20):
            center = random.randint(2, src.n_frames - 3)
            clip   = src.clip(center)
            if clip is not None:
                break
        if clip is None:
            center = max(2, min(src.n_frames // 2, src.n_frames - 3))
            clip   = _read_frames_at(src.path, center - 2, 5)
            assert clip is not None

        base = clip[2].copy()
        # Deux tirages INDÉPENDANTS : même frame de base, bruits différents
        noisy_clip  = np.stack([self._add_one_noise(f) for f in clip])  # input bruité
        noisy_target = self._add_one_noise(base)                         # target bruité indépendamment
        return noisy_clip, noisy_target

    def _validate_pair_stats(self, n_samples: int = 10) -> None:
        """
        Vérifie que input et target ont des statistiques similaires.
        Lève une erreur si la target est significativement plus sombre que l'input
        (symptôme du bug 'target = input + bruit_extra').
        """
        mask_bool = self.mask.astype(bool)
        mean_i, mean_t, std_i, std_t = [], [], [], []

        for _ in range(n_samples):
            if self.b_clips and self.force_strategy != "A":
                clip, target = self._sample_b()
            elif self.a_pairs:
                result = self._sample_a()
                if result is None:
                    continue
                clip, target = result
            else:
                continue

            inp_vals = clip[2][mask_bool]
            tgt_vals = target[mask_bool] if target.ndim == 2 else target[mask_bool]
            mean_i.append(float(inp_vals.mean()))
            mean_t.append(float(tgt_vals.mean()))
            std_i.append(float(inp_vals.std()))
            std_t.append(float(tgt_vals.std()))

        if not mean_i:
            return

        mi = float(np.mean(mean_i));  mt = float(np.mean(mean_t))
        si = float(np.mean(std_i));   st = float(np.mean(std_t))

        log.info(
            "[%s] ── Validation statistique des paires (n=%d) ──────────",
            self.split, n_samples,
        )
        log.info("[%s]   Luminosité moy. Input  : %.2f", self.split, mi)
        log.info("[%s]   Luminosité moy. Target : %.2f  (doit être ≈ Input)", self.split, mt)
        log.info("[%s]   Std Input               : %.2f", self.split, si)
        log.info("[%s]   Std Target              : %.2f  (doit être ≈ Input)", self.split, st)
        log.info("[%s]   Δ luminosité (I-T)       : %+.2f  (OK si |Δ| < 5)", self.split, mi - mt)

        if mt < mi - 5.0:
            raise RuntimeError(
                f"\n[{self.split}] ✗ BUG CRITIQUE : luminosité target ({mt:.2f}) "
                f"< input ({mi:.2f}) − 5\n"
                f"  La target est plus dégradée que l'input → le réseau apprendrait à bruiter !\n"
                f"  → Vérifier _sample_b() dans dataset.py\n"
                f"  → Lancer : python debug_pairs.py  pour inspection visuelle"
            )

    def __getitem__(self, _: int) -> dict[str, torch.Tensor]:
        clip: np.ndarray | None = None
        target: np.ndarray | None = None

        use_a = (
            self.force_strategy == "A"
            or (self.force_strategy == "AUTO"
                and self.a_pairs
                and (not self.b_clips or random.random() < 0.70))
        )

        if use_a and self.a_pairs:
            result = self._sample_a()
            if result is not None:
                clip, target = result
            # Si ECC échoue sur toutes les tentatives → fallback B

        if clip is None:
            if self.b_clips:
                clip, target = self._sample_b()
            elif self.a_pairs:
                # Dernier recours : A sans validation (ne devrait pas arriver)
                result = self._sample_a()
                assert result is not None, "Aucune vidéo utilisable dans ce split"
                clip, target = result
            else:
                raise RuntimeError("Aucune source de données disponible")

        rc   = _sample_valid_patch(self.mask, self.patch_size)
        r, c = rc if rc is not None else (0, 0)
        ps   = self.patch_size

        clip_p   = clip[:, r:r+ps, c:c+ps]
        target_p = target[r:r+ps, c:c+ps]

        if self.augment:
            clip_p, target_p = _augment(clip_p, target_p)

        clip_t   = torch.from_numpy(clip_p   / 255.0).float()
        target_t = torch.from_numpy(target_p / 255.0).float()
        sigma_t  = torch.full((1, ps, ps), self.sigma_norm, dtype=torch.float32)

        return {
            "frames": clip_t,
            "target": target_t.unsqueeze(0),
            "sigma":  sigma_t,
        }
