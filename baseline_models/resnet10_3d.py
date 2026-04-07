#!/usr/bin/env python3
"""Bee-Wo 3D ResNet-10 baseline adapted from IPN-Hand."""

from __future__ import annotations

from functools import partial

import torch
from torch import nn
from torch.nn import functional as F


def conv3x3x3(in_planes: int, out_planes: int, stride: int | tuple[int, int, int] = 1) -> nn.Conv3d:
    return nn.Conv3d(
        in_planes,
        out_planes,
        kernel_size=3,
        stride=stride,
        padding=1,
        bias=False,
    )


def downsample_basic_block(x: torch.Tensor, planes: int, stride: int | tuple[int, int, int]) -> torch.Tensor:
    out = F.avg_pool3d(x, kernel_size=1, stride=stride)
    zero_pads = out.new_zeros(
        out.size(0),
        planes - out.size(1),
        out.size(2),
        out.size(3),
        out.size(4),
    )
    return torch.cat([out, zero_pads], dim=1)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(
        self,
        inplanes: int,
        planes: int,
        stride: int | tuple[int, int, int] = 1,
        downsample: nn.Module | None = None,
    ) -> None:
        super().__init__()
        self.conv1 = conv3x3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm3d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3x3(planes, planes)
        self.bn2 = nn.BatchNorm3d(planes)
        self.downsample = downsample

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)
        return out


class BeeWoResNet3D(nn.Module):
    def __init__(
        self,
        block: type[BasicBlock],
        layers: list[int],
        sample_size: int = 96,
        sample_duration: int = 16,
        shortcut_type: str = "B",
        num_classes: int = 5,
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
        self.maxpool = nn.MaxPool3d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0], shortcut_type)
        self.layer2 = self._make_layer(block, 128, layers[1], shortcut_type, stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], shortcut_type, stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], shortcut_type, stride=2)
        self.avgpool = nn.AdaptiveAvgPool3d((1, 1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)

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
        x = x.flatten(1)
        return self.fc(x)


def BeeWoResNet10_3D(
    num_classes: int = 5,
    sample_size: int = 96,
    sample_duration: int = 16,
    shortcut_type: str = "B",
) -> BeeWoResNet3D:
    return BeeWoResNet3D(
        BasicBlock,
        [1, 1, 1, 1],
        sample_size=sample_size,
        sample_duration=sample_duration,
        shortcut_type=shortcut_type,
        num_classes=num_classes,
    )
