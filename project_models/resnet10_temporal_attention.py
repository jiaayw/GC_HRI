#!/usr/bin/env python3
"""ResNet10-3D backbone with a lightweight temporal attention head."""

from __future__ import annotations

from functools import partial

import torch
from torch import nn

from baseline_models.resnet10_3d import BasicBlock, downsample_basic_block


class TemporalAttentionHead(nn.Module):
    def __init__(
        self,
        feature_dim: int,
        num_classes: int = 5,
        attention_hidden_dim: int = 64,
        dropout: float = 0.10,
    ) -> None:
        super().__init__()
        self.attention = nn.Sequential(
            nn.Linear(feature_dim, attention_hidden_dim),
            nn.Tanh(),
            nn.Linear(attention_hidden_dim, 1),
        )
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(feature_dim, feature_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(feature_dim, num_classes),
        )

    def forward(self, temporal_features: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        attention_logits = self.attention(temporal_features).squeeze(-1)
        attention_weights = torch.softmax(attention_logits, dim=1)
        pooled = torch.sum(temporal_features * attention_weights.unsqueeze(-1), dim=1)
        logits = self.classifier(pooled)
        return logits, attention_weights


class BeeWoResNet10TemporalAttention(nn.Module):
    def __init__(
        self,
        num_classes: int = 5,
        sample_size: int = 96,
        sample_duration: int = 16,
        shortcut_type: str = "B",
        attention_hidden_dim: int = 64,
        dropout: float = 0.15,
    ) -> None:
        super().__init__()
        self.inplanes = 64
        self.conv1 = nn.Conv3d(
            3,
            64,
            kernel_size=7,
            stride=(1, 2, 2),
            padding=(3, 3, 3),
            bias=False,
        )
        self.bn1 = nn.BatchNorm3d(64)
        self.relu = nn.ReLU(inplace=True)
        # Match the baseline temporal downsampling path early for easier optimization.
        self.maxpool = nn.MaxPool3d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(BasicBlock, 64, 1, shortcut_type)
        self.layer2 = self._make_layer(BasicBlock, 128, 1, shortcut_type, stride=2)
        # Stop temporal feature extraction at layer3 so T stays short but > 1 for attention.
        self.layer3 = self._make_layer(BasicBlock, 256, 1, shortcut_type, stride=2)
        self.spatial_pool = nn.AdaptiveAvgPool3d((None, 1, 1))
        self.temporal_head = TemporalAttentionHead(
            feature_dim=256,
            num_classes=num_classes,
            attention_hidden_dim=attention_hidden_dim,
            dropout=dropout,
        )

        for module in self.modules():
            if isinstance(module, nn.Conv3d):
                nn.init.kaiming_normal_(module.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(module, nn.BatchNorm3d):
                module.weight.data.fill_(1)
                module.bias.data.zero_()

    def _make_layer(
        self,
        block: type[BasicBlock],
        planes: int,
        blocks: int,
        shortcut_type: str,
        stride: int | tuple[int, int, int] = 1,
    ) -> nn.Sequential:
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            if shortcut_type == "A":
                downsample = partial(
                    downsample_basic_block,
                    planes=planes * block.expansion,
                    stride=stride,
                )
            else:
                downsample = nn.Sequential(
                    nn.Conv3d(
                        self.inplanes,
                        planes * block.expansion,
                        kernel_size=1,
                        stride=stride,
                        bias=False,
                    ),
                    nn.BatchNorm3d(planes * block.expansion),
                )

        layers = [block(self.inplanes, planes, stride, downsample)]
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes))
        return nn.Sequential(*layers)

    def extract_temporal_features(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)

        x = self.spatial_pool(x).squeeze(-1).squeeze(-1)
        return x.transpose(1, 2).contiguous()

    def forward(
        self,
        x: torch.Tensor,
        return_attention: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        temporal_features = self.extract_temporal_features(x)
        logits, attention_weights = self.temporal_head(temporal_features)
        if return_attention:
            return logits, attention_weights
        return logits
