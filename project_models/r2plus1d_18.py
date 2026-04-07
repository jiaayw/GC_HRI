#!/usr/bin/env python3
"""Torchvision R(2+1)D-18 wrapper for Bee-Wo video classification."""

from __future__ import annotations

from torch import nn

from torchvision.models.video import R2Plus1D_18_Weights, r2plus1d_18


def BeeWoR2Plus1D_18(
    num_classes: int = 5,
    pretrained: bool = False,
    dropout: float = 0.3,
) -> nn.Module:
    weights = R2Plus1D_18_Weights.KINETICS400_V1 if pretrained else None
    model = r2plus1d_18(weights=weights)
    in_features = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Dropout(p=dropout),
        nn.Linear(in_features, num_classes),
    )
    return model
