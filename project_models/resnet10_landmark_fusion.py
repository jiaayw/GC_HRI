#!/usr/bin/env python3
"""ResNet10-3D RGB branch fused with a temporal landmark encoder."""

from __future__ import annotations

from functools import partial

import torch
from torch import nn

from baseline_models.resnet10_3d import BasicBlock, downsample_basic_block


class ResNet10FeatureExtractor(nn.Module):
    def __init__(self, shortcut_type: str = "B") -> None:
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
        self.maxpool = nn.MaxPool3d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(BasicBlock, 64, 1, shortcut_type)
        self.layer2 = self._make_layer(BasicBlock, 128, 1, shortcut_type, stride=2)
        self.layer3 = self._make_layer(BasicBlock, 256, 1, shortcut_type, stride=2)
        self.layer4 = self._make_layer(BasicBlock, 512, 1, shortcut_type, stride=2)
        self.avgpool = nn.AdaptiveAvgPool3d((1, 1, 1))

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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        return x.flatten(1)


class LandmarkTemporalConvBranch(nn.Module):
    def __init__(self, hidden_dim: int = 128, dropout: float = 0.3) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.Conv1d(63, hidden_dim, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Conv1d(hidden_dim, hidden_dim, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        )
        self.pool = nn.AdaptiveAvgPool1d(1)

    def forward(self, landmarks: torch.Tensor) -> torch.Tensor:
        temporal_features = landmarks.transpose(1, 2).contiguous()
        encoded = self.network(temporal_features)
        return self.pool(encoded).squeeze(-1)


class BeeWoResNet10LandmarkFusion(nn.Module):
    def __init__(
        self,
        num_classes: int = 5,
        seq_len: int = 16,
        rgb_feature_dim: int = 512,
        landmark_hidden_dim: int = 128,
        dropout: float = 0.3,
    ) -> None:
        super().__init__()
        self.rgb_branch = ResNet10FeatureExtractor()
        self.landmark_branch = LandmarkTemporalConvBranch(hidden_dim=landmark_hidden_dim, dropout=dropout)
        self.rgb_projection = nn.Sequential(
            nn.Linear(rgb_feature_dim, landmark_hidden_dim),
            nn.ReLU(inplace=True),
        )
        self.classifier = nn.Sequential(
            nn.Linear(landmark_hidden_dim * 2, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(256, num_classes),
        )

    def forward(self, frames: torch.Tensor, landmarks: torch.Tensor) -> torch.Tensor:
        rgb_features = self.rgb_projection(self.rgb_branch(frames))
        landmark_features = self.landmark_branch(landmarks)
        fused = torch.cat([rgb_features, landmark_features], dim=1)
        return self.classifier(fused)
