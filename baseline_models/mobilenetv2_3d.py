#!/usr/bin/env python3
"""Bee-Wo 3D MobileNetV2 baseline adapted from IPN-Hand."""

from __future__ import annotations

import math

import torch
from torch import nn
from torch.nn import functional as F


def conv_bn(inp: int, oup: int, stride: tuple[int, int, int]) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv3d(inp, oup, kernel_size=3, stride=stride, padding=1, bias=False),
        nn.BatchNorm3d(oup),
        nn.ReLU6(inplace=True),
    )


def conv_1x1x1_bn(inp: int, oup: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv3d(inp, oup, 1, 1, 0, bias=False),
        nn.BatchNorm3d(oup),
        nn.ReLU6(inplace=True),
    )


class InvertedResidual(nn.Module):
    def __init__(self, inp: int, oup: int, stride: tuple[int, int, int], expand_ratio: int) -> None:
        super().__init__()
        hidden_dim = round(inp * expand_ratio)
        self.use_res_connect = self.stride_is_identity(stride) and inp == oup

        if expand_ratio == 1:
            self.conv = nn.Sequential(
                nn.Conv3d(hidden_dim, hidden_dim, 3, stride, 1, groups=hidden_dim, bias=False),
                nn.BatchNorm3d(hidden_dim),
                nn.ReLU6(inplace=True),
                nn.Conv3d(hidden_dim, oup, 1, 1, 0, bias=False),
                nn.BatchNorm3d(oup),
            )
        else:
            self.conv = nn.Sequential(
                nn.Conv3d(inp, hidden_dim, 1, 1, 0, bias=False),
                nn.BatchNorm3d(hidden_dim),
                nn.ReLU6(inplace=True),
                nn.Conv3d(hidden_dim, hidden_dim, 3, stride, 1, groups=hidden_dim, bias=False),
                nn.BatchNorm3d(hidden_dim),
                nn.ReLU6(inplace=True),
                nn.Conv3d(hidden_dim, oup, 1, 1, 0, bias=False),
                nn.BatchNorm3d(oup),
            )

    @staticmethod
    def stride_is_identity(stride: tuple[int, int, int]) -> bool:
        return stride == (1, 1, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.use_res_connect:
            return x + self.conv(x)
        return self.conv(x)


class BeeWoMobileNetV2_3D(nn.Module):
    def __init__(self, num_classes: int = 5, sample_size: int = 96, width_mult: float = 1.0) -> None:
        super().__init__()
        if sample_size % 16 != 0:
            raise ValueError("sample_size must be divisible by 16.")

        input_channel = int(32 * width_mult)
        last_channel = 1280 if width_mult <= 1.0 else int(1280 * width_mult)
        inverted_residual_setting = [
            [1, 16, 1, (1, 1, 1)],
            [6, 24, 2, (2, 2, 2)],
            [6, 32, 3, (2, 2, 2)],
            [6, 64, 4, (2, 2, 2)],
            [6, 96, 3, (1, 1, 1)],
            [6, 160, 3, (2, 2, 2)],
            [6, 320, 1, (1, 1, 1)],
        ]

        features: list[nn.Module] = [conv_bn(3, input_channel, (1, 2, 2))]
        for expand_ratio, channels, repeats, stride in inverted_residual_setting:
            output_channel = int(channels * width_mult)
            for block_index in range(repeats):
                block_stride = stride if block_index == 0 else (1, 1, 1)
                features.append(
                    InvertedResidual(
                        input_channel,
                        output_channel,
                        block_stride,
                        expand_ratio=expand_ratio,
                    )
                )
                input_channel = output_channel

        features.append(conv_1x1x1_bn(input_channel, last_channel))
        self.features = nn.Sequential(*features)
        self.classifier = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(last_channel, num_classes),
        )
        self._initialize_weights()

    def _initialize_weights(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Conv3d):
                fan = module.kernel_size[0] * module.kernel_size[1] * module.kernel_size[2] * module.out_channels
                module.weight.data.normal_(0, math.sqrt(2.0 / fan))
                if module.bias is not None:
                    module.bias.data.zero_()
            elif isinstance(module, nn.BatchNorm3d):
                module.weight.data.fill_(1)
                module.bias.data.zero_()
            elif isinstance(module, nn.Linear):
                module.weight.data.normal_(0, 0.01)
                module.bias.data.zero_()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = F.adaptive_avg_pool3d(x, (1, 1, 1))
        x = x.flatten(1)
        return self.classifier(x)
