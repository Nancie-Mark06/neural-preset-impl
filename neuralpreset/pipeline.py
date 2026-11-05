"""NeuralPreset —— 把编码器与两个 DNCM 拼成论文的两阶段流程。

    Z = I_c · M_n(d)        # normalize：把内容图拉到一个「规范空间」
    Y = Z   · M_s(r)        # stylize  ：把风格 r 套上去

训练时**必须交换 r**（REPRODUCTION_PLAN.md §1.4）：

    Z_i = normalize(I_i, d_i) ,  Y_i = stylize(Z_j, r_i)
    Z_j = normalize(I_j, d_j) ,  Y_j = stylize(Z_i, r_j)

因为 I_i 与 I_j 内容相同，所以「j 的内容 + i 的风格」应当重建出 I_i。
不交换会让 r 偷偷记住像素，退化成「复制风格图」（论文式 13 的退化解）。
"""

from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from .dncm import DNCM
from .encoder import DualHeadEncoder

__all__ = ["NeuralPreset", "build_model"]


class NeuralPreset(nn.Module):
    """两阶段颜色风格迁移。

    Args:
        k: 中间维度。
        act: G1 开关，透传给两个 DNCM。
        backbone / pretrained_mode / pretrained_ckpt: 透传给编码器。
        thumb_size: 编码器输入边长（论文 256，EfficientNet-B0 的既定尺寸）。
    """

    def __init__(
        self,
        k: int = 16,
        act: str = "none",
        backbone: str = "efficientnet_b0",
        pretrained_mode: str = "auto",
        pretrained_ckpt: Optional[str] = None,
        thumb_size: int = 256,
    ) -> None:
        super().__init__()
        self.k = int(k)
        self.act = act
        self.thumb_size = int(thumb_size)

        self.encoder = DualHeadEncoder(
            k=k,
            backbone=backbone,
            pretrained_mode=pretrained_mode,
            pretrained_ckpt=pretrained_ckpt,
        )
        # 两个 DNCM 参数独立：normalize 与 stylize 学的是两种不同的映射
        self.n_dncm = DNCM(k=k, act=act)
        self.s_dncm = DNCM(k=k, act=act)

    # ------------------------------------------------------------------ #
    # 基本构件
    # ------------------------------------------------------------------ #
    def encode(self, thumb: torch.Tensor):
        """thumb [B,3,S,S] → (d, r)，各 [B,k,k]。"""
        return self.encoder(thumb)

    def normalize(self, img: torch.Tensor, d: torch.Tensor) -> torch.Tensor:
        return self.n_dncm(img, d)

    def stylize(self, z: torch.Tensor, r: torch.Tensor) -> torch.Tensor:
        return self.s_dncm(z, r)

    def to_thumb(self, img: torch.Tensor) -> torch.Tensor:
        """把任意尺寸的图缩到编码器输入尺寸（双三次 + 抗锯齿）。"""
        if img.shape[-2:] == (self.thumb_size, self.thumb_size):
            return img
        return F.interpolate(
            img,
            size=(self.thumb_size, self.thumb_size),
            mode="bicubic",
            align_corners=False,
            antialias=True,
        ).clamp(0.0, 1.0)

    # ------------------------------------------------------------------ #
    # 训练前向
    # ------------------------------------------------------------------ #
    def forward_train(self, Ii: torch.Tensor, Ij: torch.Tensor) -> dict:
        """输入一对「内容相同、颜色不同」的图，返回 Z / Y 与中间量。

        Args:
            Ii, Ij: [B,3,H,W]，[0,1]，同一张干净图的两个颜色扰动版本。

        Returns:
            dict: Zi, Zj, Yi, Yj, d_i, r_i, d_j, r_j, M_n, M_s
                  （M_n/M_s 是 3×3，用于监控奇异值 / 尺度退化）
        """
        if Ii.shape != Ij.shape:
            raise ValueError(f"Ii 与 Ij 形状必须一致：{tuple(Ii.shape)} vs {tuple(Ij.shape)}")

        Ti = self.to_thumb(Ii)
        Tj = self.to_thumb(Ij)
        d_i, r_i = self.encode(Ti)
        d_j, r_j = self.encode(Tj)

        Zi = self.normalize(Ii, d_i)
        Zj = self.normalize(Ij, d_j)

        # 交换 r —— 整套方法最容易被写错的一行
        Yi = self.stylize(Zj, r_i)
        Yj = self.stylize(Zi, r_j)

        return {
            "Zi": Zi, "Zj": Zj, "Yi": Yi, "Yj": Yj,
            "d_i": d_i, "d_j": d_j, "r_i": r_i, "r_j": r_j,
            "M_n": self.n_dncm.matrix(d_i),   # [B,3,3]
            "M_s": self.s_dncm.matrix(r_i),   # [B,3,3]
        }

    # ------------------------------------------------------------------ #
    # 推理
    # ------------------------------------------------------------------ #
    @torch.no_grad()
    def compute_preset(self, style_img: torch.Tensor) -> torch.Tensor:
        """从风格图算出一个 preset（即 r），可复用给任意多张内容图 / 视频帧。

        style_img: [B,3,H,W] → r [B,k,k]
        """
        _, r = self.encode(self.to_thumb(style_img))
        return r

    @torch.no_grad()
    def apply_preset(
        self,
        img: torch.Tensor,
        r: torch.Tensor,
        content_img: Optional[torch.Tensor] = None,
        patch: int = 512,
        clamp: bool = True,
    ) -> torch.Tensor:
        """用给定 preset 渲染一张高分辨率图（分 patch，结果与整图严格一致）。

        Args:
            img: [1,3,H,W] 内容图（[0,1]）。
            r: [1,k,k]，通常来自 `compute_preset`。
            content_img: 用于算内容码 d 的缩略图来源，默认就是 img 本身。
            patch: 每块边长；None 表示不切块。

        Note:
            d 必须由「整图缩略图」算出，而不是每块各算一次 —— 否则块与块之间会出现
            颜色接缝（DNCM 是纯逐像素映射，同一 d 下块内块外完全一致）。
        """
        if img.shape[0] != 1:
            raise ValueError("apply_preset 只接受单张图（batch=1）")
        src = img if content_img is None else content_img
        d, _ = self.encode(self.to_thumb(src))
        return self.apply_codes(img, d, r, patch=patch, clamp=clamp)

    @torch.no_grad()
    def apply_codes(
        self,
        img: torch.Tensor,
        d: torch.Tensor,
        r: torch.Tensor,
        patch: int = 512,
        clamp: bool = True,
    ) -> torch.Tensor:
        """给定 (d, r) 渲染图像，分 patch 以控制显存。"""
        B, _, H, W = img.shape
        if patch is None or (H <= patch and W <= patch):
            y = self.stylize(self.normalize(img, d), r)
        else:
            ph = math.ceil(H / patch) * patch
            pw = math.ceil(W / patch) * patch
            padded = F.pad(img, (0, pw - W, 0, ph - H), mode="replicate")
            out = torch.empty_like(padded)
            for i in range(0, ph, patch):
                for j in range(0, pw, patch):
                    blk = padded[..., i:i + patch, j:j + patch]
                    out[..., i:i + patch, j:j + patch] = self.stylize(
                        self.normalize(blk, d), r
                    )
            y = out[..., :H, :W]
        return y.clamp(0.0, 1.0) if clamp else y

    @torch.no_grad()
    def stylize_hr(self, img: torch.Tensor, patch: int = 512) -> torch.Tensor:
        """一步到位的推理：缩略图算 (d, r) → 分 patch 渲染原图。"""
        d, r = self.encode(self.to_thumb(img))
        return self.apply_codes(img, d, r, patch=patch)

    # ------------------------------------------------------------------ #
    def fold(self, T: torch.Tensor) -> torch.Tensor:
        """把 DNCM 折叠成 3×3（仅 act='none' 有效）。"""
        return self.s_dncm.matrix(T)

    def num_parameters(self, trainable_only: bool = True) -> int:
        return sum(
            p.numel() for p in self.parameters() if p.requires_grad or not trainable_only
        )


# ---------------------------------------------------------------------- #
def build_model(cfg: dict) -> NeuralPreset:
    """从配置字典构造模型。

    cfg 形如 configs/phase0.yaml 里的 `model:` 段。
    """
    m = dict(cfg or {})
    pretrained = m.get("pretrained") or {}
    return NeuralPreset(
        k=int(m.get("k", 16)),
        act=m.get("act", "none"),
        backbone=m.get("backbone", "efficientnet_b0"),
        pretrained_mode=pretrained.get("mode", "auto"),
        pretrained_ckpt=pretrained.get("ckpt"),
        thumb_size=int(m.get("thumb_size", 256)),
    )
