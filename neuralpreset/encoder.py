"""共享主干 + 双头编码器（论文 §3.2 / G5）。

    thumb [B,3,256,256]
      → EfficientNet-B0 features → GAP → [B,1280]
      → Linear(1280,256) → d [B,16,16]   （内容码，给 normalize 用）
      → Linear(1280,256) → r [B,16,16]   （风格码，给 stylize 用）

两个头共享同一个主干，只是各自接一个线性层 —— 这是 G5 的拍板结果。

预训练权重的三级回退（见 REPRODUCTION_PLAN.md §10.3 与本次规划补充）：

    mode="local"       只读 ckpt 文件
    mode="torchvision" 只走 torchvision 自动下载
    mode="auto"        本地 ckpt 存在 → 用它；否则 torchvision；再失败 → 随机初始化并告警
    mode="none"        随机初始化

Kaggle 上把 21MB 的 efficientnet_b0 权重随数据一起打包，`auto` 就会走 local 分支，
关网也能跑。
"""

from __future__ import annotations

import os
import math
import warnings
from typing import Optional, Tuple

import torch
import torch.nn as nn

__all__ = ["DualHeadEncoder"]


def _load_state_dict(path: str) -> dict:
    obj = torch.load(path, map_location="cpu", weights_only=False)
    for key in ("state_dict", "model", "model_state_dict", "net"):
        if isinstance(obj, dict) and key in obj and isinstance(obj[key], dict):
            obj = obj[key]
            break
    if not isinstance(obj, dict):
        raise ValueError(f"无法从 {path} 里读出 state_dict")
    # 去掉 DataParallel / torch.compile 的前缀
    return {k[len("module."):] if k.startswith("module.") else k: v for k, v in obj.items()}


def _build_efficientnet_b0(weights: Optional[str]):
    """返回 torchvision 的 efficientnet_b0；`weights` 为 "imagenet" 或 None。"""
    try:
        from torchvision.models import EfficientNet_B0_Weights, efficientnet_b0
    except Exception as exc:  # pragma: no cover - 依赖缺失时的明确报错
        raise RuntimeError(
            "需要 torchvision>=0.13 的 EfficientNet-B0。"
            "Kaggle 镜像自带；本地请 pip install torchvision。"
        ) from exc

    if weights == "imagenet":
        return efficientnet_b0(weights=EfficientNet_B0_Weights.IMAGENET1K_V1)
    return efficientnet_b0(weights=None)


class DualHeadEncoder(nn.Module):
    """共享主干 + 两个线性头。

    Args:
        k: 中间维度，头的输出是 k×k。
        backbone: 目前只支持 "efficientnet_b0"。
        pretrained_mode: none / local / torchvision / auto。
        pretrained_ckpt: local / auto 模式下要读的权重路径。
    """

    def __init__(
        self,
        k: int = 16,
        backbone: str = "efficientnet_b0",
        pretrained_mode: str = "auto",
        pretrained_ckpt: Optional[str] = None,
    ) -> None:
        super().__init__()
        if backbone != "efficientnet_b0":
            raise NotImplementedError(f"暂只支持 efficientnet_b0，收到 {backbone!r}")
        if pretrained_mode not in ("none", "local", "torchvision", "auto"):
            raise ValueError(f"pretrained_mode 非法：{pretrained_mode!r}")

        self.k = int(k)
        self.backbone_name = backbone
        self.pretrained_mode = pretrained_mode
        self.weight_source = "none"

        full = self._make_backbone(pretrained_mode, pretrained_ckpt)

        # 只保留卷积特征，丢掉 ImageNet 分类头（省 1.28M 参数，见 G6）
        self.features = full.features
        self.feat_dim = int(full.classifier[1].in_features)  # b0 = 1280
        del full

        self.pool = nn.AdaptiveAvgPool2d(1)
        self.head_d = nn.Linear(self.feat_dim, k * k)
        self.head_r = nn.Linear(self.feat_dim, k * k)
        # ImageNet 预训练权重是在 normalize 过的输入上训出来的；不吃这一步，
        # GAP 特征分布会整体偏移，预训练等于白用。放在 buffer 里（不进 state_dict）。
        self.register_buffer("_in_mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1),
                             persistent=False)
        self.register_buffer("_in_std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1),
                             persistent=False)
        self._init_heads()

    # ------------------------------------------------------------------ #
    def _make_backbone(self, mode: str, ckpt: Optional[str]):
        want_local = mode in ("local", "auto") and ckpt and os.path.isfile(ckpt)
        if mode == "local" and not want_local:
            raise FileNotFoundError(f"pretrained_mode=local 但找不到权重文件：{ckpt}")

        if want_local:
            net = _build_efficientnet_b0(None)
            state = _load_state_dict(ckpt)
            missing, unexpected = net.load_state_dict(state, strict=False)
            if len(missing) > 40:  # 分类头之外大面积缺失 ⇒ 大概率不是 EfficientNet 权重
                warnings.warn(
                    f"从 {ckpt} 加载时缺失 {len(missing)} 个张量，可能不是 EfficientNet-B0 权重",
                    RuntimeWarning,
                )
            self.weight_source = f"local:{ckpt}"
            return net

        if mode in ("torchvision", "auto"):
            try:
                net = _build_efficientnet_b0("imagenet")
                self.weight_source = "torchvision:IMAGENET1K_V1"
                return net
            except Exception as exc:
                if mode == "torchvision":
                    raise
                warnings.warn(
                    f"ImageNet 权重获取失败（{type(exc).__name__}: {exc}），回退到随机初始化。",
                    RuntimeWarning,
                )

        self.weight_source = "none"
        return _build_efficientnet_b0(None)

    def _init_heads(self) -> None:
        """G7：初始时两个 DNCM 都应当近似恒等映射。

        T 由头直接给出（[B,k,k]），所以要让 T ≈ Iₖ：把 bias 设成单位阵展开，
        weight 用高斯微扰破对称。注意 weight 不能置 0（否则 T 恒为 I、r 拿不到任何信号，
        但也不至于梯度为 0 —— 输入特征非零，梯度仍会流）。

        weight 的 std 按 `1/sqrt(feat_dim)` 缩放：GAP 出来的 1280 维特征每个维度量级 O(1)，
        直接累加 1280 项会把 N(0,0.005) 放大到 ≈0.005·√1280 ≈ 0.18，视觉上已经是明显偏色。
        缩放后 T ≈ I + O(0.005)，初始化真正接近恒等。
        """
        eye_flat = torch.eye(self.k).reshape(-1)
        std = 0.005 / math.sqrt(float(self.feat_dim))
        for head in (self.head_d, self.head_r):
            nn.init.normal_(head.weight, mean=0.0, std=std)
            with torch.no_grad():
                head.bias.copy_(eye_flat)

    # ------------------------------------------------------------------ #
    def forward(self, thumb: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """thumb [B,3,256,256] → (d [B,k,k], r [B,k,k])。"""
        if thumb.dim() != 4 or thumb.shape[1] != 3:
            raise ValueError(f"thumb 期望 [B,3,H,W]，收到 {tuple(thumb.shape)}")
        f = self.features((thumb - self._in_mean) / self._in_std)   # [B, 1280, h, w]
        g = self.pool(f).flatten(1)           # [B, 1280]
        B = g.shape[0]
        d = self.head_d(g).reshape(B, self.k, self.k)
        r = self.head_r(g).reshape(B, self.k, self.k)
        return d, r

    # ------------------------------------------------------------------ #
    @torch.no_grad()
    def num_parameters(self, trainable_only: bool = True) -> int:
        ps = self.parameters()
        return sum(p.numel() for p in ps if (p.requires_grad or not trainable_only))

    def extra_repr(self) -> str:
        return (
            f"k={self.k}, backbone={self.backbone_name}, "
            f"feat_dim={self.feat_dim}, weights={self.weight_source}"
        )
