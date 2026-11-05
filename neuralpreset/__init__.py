"""Neural Preset (CVPR 2023, arXiv 2303.13511v2) —— 复现实现。

本包只做一件事：给定内容图与风格图，输出一个 3×3 全局颜色矩阵并作用到像素上。

    Y = I · M_n · M_s ,   M = P · T · Q ∈ R^{3×3}

模块划分见 REPRODUCTION_PLAN.md §3。
"""

from .dncm import DNCM
from .encoder import DualHeadEncoder
from .losses import NeuralPresetLoss, rec_loss, con_loss
from .pipeline import NeuralPreset, build_model

__all__ = [
    "DNCM",
    "DualHeadEncoder",
    "NeuralPreset",
    "NeuralPresetLoss",
    "build_model",
    "rec_loss",
    "con_loss",
]

__version__ = "0.1.0"
