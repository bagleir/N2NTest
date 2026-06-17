# Pipeline vasculaire v2 — SANS MASQUE

Débruitage + projection + rehaussement pour angiographie en fluorescence
(vaisseaux clairs sur fond sombre). **Aucune notion de masque** : tout est
traité sur l'image entière. Inspiré en partie d'un projet de référence
(appariement N2N par luminosité, projection upscale+USM+moyenne).

## Installation
```bash
pip install -r requirements.txt
pip install torch        # seulement pour la route N2N (GPU recommandé)
```

## Lancer (recette par défaut, recommandée)
Route classique, sans GPU, sans masque, projection upscale+USM+moyenne :
```bash
python vessel_pipeline.py R_260310_AUZ0752_9_p.avi -o out_v9
```
Sorties dans `out_v9/` :
- `*_projection.png`  : projection usm_mean 1024×1024 (16-bit)
- `*_99_final.png`    : image finale (projection + USM + gamma)
- `*_mean512.png`     : moyenne native 512 (comparaison)
- `*_metrics.json`    : CNR, netteté, réponse Frangi

## Options
- `--projection usm_mean|mip|mean|percentile`  (def. usm_mean)
- `--enhance light|full|none`  (def. light = USM+gamma ; full = tophat+CLAHE+Frangi+USM)
- `--target-size 1024`  (taille de l'image agrandie)
- `--no-register`  (le projet de réf. ne recale pas ; à tester si la moyenne est floue ou trop lisse)
- `--motion euclidean|affine`, `--optical-flow`
- `--denoiser temporal|n2n|none`, `--model n2n_model.pt`

## Noise2Noise (route DL)
**Appariement par luminosité** (inspiré du projet de réf.) : chaque frame est
appariée à une autre de luminosité quasi-identique mais éloignée -> même phase
cardiaque/contraste -> cibles N2N propres. C'est le défaut.

Par vidéo :
```bash
python vessel_pipeline.py video.avi -o out --denoiser n2n
```
Dossier (entraîner une fois, déployer partout) :
```bash
python n2n.py train dossier_videos -m n2n_model.pt --epochs 300
python vessel_pipeline.py video.avi -o out --model n2n_model.pt
```

## Réglages utiles (dans DEFAULT_CFG)
| Paramètre | Effet |
|---|---|
| `usm_amount_proj` | force du sharpen par frame dans la projection (def. 1.5) |
| `enhance.gamma` | éclaircit les vaisseaux faibles (def. 0.85, plus bas = plus clair) |
| `enhance.usm_amount` | sharpen final (profil light) |
| `temporal_win` | fenêtre du débruitage temporel (7–15) |
| `n2n_pairing` | 'brightness' (def.) ou 'consecutive' |

## Ce qui vient du projet de référence
1. **Zéro masque** (au plus une vignette naturelle conservée).
2. **Appariement N2N par luminosité** (vs frames consécutives).
3. **Projection upscale Lanczos + USM par frame + moyenne** (simple, robuste).
4. **Rehaussement léger** USM + gamma.

## Pistes restantes (du projet de réf., non encore portées)
- Entrée **multi-frames (5)** dans un U-Net **résiduel** (contexte temporel).
- **Loss multi-termes** : N2N pondéré vaisseaux/fond + préservation luminosité
  + préservation des bords (gradients) + lissage du fond.
Ces deux points peuvent encore améliorer le N2N si besoin.