#!/usr/bin/env python3
"""CNN+LSTM model for Bee-Wo gesture classification."""

from __future__ import annotations

import torch
from torch import nn


class FrameEncoder(nn.Module):
    def __init__(self, embedding_dim: int = 128) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1)),
        )
        self.projection = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128, embedding_dim),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        return self.projection(x)


class BeeWoCNNLSTM(nn.Module):
    def __init__(
        self,
        num_classes: int = 5,
        embedding_dim: int = 128,
        hidden_dim: int = 128,
        num_layers: int = 1,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        self.frame_encoder = FrameEncoder(embedding_dim=embedding_dim)
        self.lstm = nn.LSTM(
            input_size=embedding_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, frames: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len, channels, height, width = frames.shape
        flattened = frames.view(batch_size * seq_len, channels, height, width)
        embeddings = self.frame_encoder(flattened)
        embeddings = embeddings.view(batch_size, seq_len, -1)
        _, (hidden, _) = self.lstm(embeddings)
        final_hidden = hidden[-1]
        return self.classifier(final_hidden)
