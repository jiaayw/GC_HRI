"""Project-specific Bee-Wo models."""

from .cnn_lstm import BeeWoCNNLSTM
from .r2plus1d_18 import BeeWoR2Plus1D_18
from .resnet10_landmark_fusion import BeeWoResNet10LandmarkFusion
from .resnet10_temporal_attention import BeeWoResNet10TemporalAttention

__all__ = [
    "BeeWoCNNLSTM",
    "BeeWoR2Plus1D_18",
    "BeeWoResNet10TemporalAttention",
    "BeeWoResNet10LandmarkFusion",
]
