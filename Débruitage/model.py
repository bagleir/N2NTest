#!/usr/bin/env python3
"""FastDVDnet adapté N2N — Architecture pour le débruitage de vidéos vasculaires."""

from __future__ import annotations

import torch
import torch.nn as nn


def _conv_bn_relu(in_c: int, out_c: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv2d(in_c, out_c, 3, padding=1, bias=False),
        nn.BatchNorm2d(out_c),
        nn.ReLU(inplace=True),
    )


class SubNet(nn.Module):
    """
    Sous-réseau convolutionnel partagé.

    Architecture : 4 × (Conv3x3 → BN → ReLU) → Conv3x3
    Padding=1 sur toutes les convolutions → préserve la résolution spatiale.
    """

    def __init__(self, in_channels: int, out_channels: int, features: int = 64) -> None:
        super().__init__()
        self.net = nn.Sequential(
            _conv_bn_relu(in_channels, features),
            _conv_bn_relu(features,    features),
            _conv_bn_relu(features,    features),
            _conv_bn_relu(features,    features),
            nn.Conv2d(features, out_channels, 3, padding=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class FastDVDnet(nn.Module):
    """
    FastDVDnet adapté pour l'entraînement N2N sur des vidéos vasculaires 512×512.

    Entrée  : 5 frames consécutives (déjà filtrées par le median temporel) + sigma map
    Sortie  : frame centrale débruitée

    SubNet1 (partagé) : paire de frames + sigma → feature map intermédiaire
      - Appel 1 : (t-2, t-1, σ)  → feature gauche
      - Appel 2 : (t+1, t+2, σ)  → feature droite

    SubNet2 : (feature gauche, feature droite, frame centrale) → frame débruitée
    """

    def __init__(self, features: int = 64) -> None:
        super().__init__()
        # SubNet1 : 2 frames + 1 canal sigma → 1 feature map
        self.subnet1 = SubNet(in_channels=3, out_channels=1, features=features)
        # SubNet2 : 2 feature maps + frame centrale → frame débruitée
        self.subnet2 = SubNet(in_channels=3, out_channels=1, features=features)

    def forward(
        self,
        frames: torch.Tensor,   # (B, 5, H, W) float32 normalisé [0, 1]
        sigma:  torch.Tensor,   # (B, 1, H, W) float32 — carte du niveau de bruit
    ) -> torch.Tensor:
        """
        Args:
            frames : (B, 5, H, W) — 5 frames consécutives normalisées en [0, 1]
            sigma  : (B, 1, H, W) — carte constante du niveau de bruit normalisé

        Returns:
            frame centrale débruitée (B, 1, H, W) en [0, 1]
        """
        f_tm2 = frames[:, 0:1]
        f_tm1 = frames[:, 1:2]
        f_t   = frames[:, 2:3]
        f_tp1 = frames[:, 3:4]
        f_tp2 = frames[:, 4:5]

        feat_left  = self.subnet1(torch.cat([f_tm2, f_tm1, sigma], dim=1))
        feat_right = self.subnet1(torch.cat([f_tp1, f_tp2, sigma], dim=1))

        out = self.subnet2(torch.cat([feat_left, feat_right, f_t], dim=1))
        return torch.sigmoid(out)
