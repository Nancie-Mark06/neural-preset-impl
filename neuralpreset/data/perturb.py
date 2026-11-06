"""颜色扰动：造「内容相同、颜色不同」的训练对（论文 §3 / G11）。

数据流是 **PIL → PIL（LUT）→ 张量（参数化算子）**：

1. LUT 这一步在 PIL 上做 —— `pillow-lut` 的 `filter` 就是逐像素查表，
   比在 torch 里手写三线性插值更省事，也保证用的是 LUT 的原生精度；
2. 其余算子（亮度/饱和度/色相/伽马/色温/随机 3×3）在 [3,H,W]、[0,1] 张量上做，
   因为它们要叠加、要 clamp、要能便宜地在 GPU 上跑。

算子集参考 Harmonizer 的随机滤波调整（论文引用的 [39]），另加：

* **LUT**：从 LUTBank 随机抽一条，强度在 [0.6, 1.0] 之间（全强度容易过火）；
* **随机 3×3**：廉价兜底，覆盖「任意全局线性变换」

一次扰动的组合方式：先随机挑一个子集（每个算子带独立概率），按随机顺序作用，
最后 clamp 到 [0,1]。若某次采样几乎没改颜色（max|Δ| < 1e-3），补一个 gamma 重采，
避免产生「恒等对」——恒等对会让 L_con 无信息量。

所有随机量都在 `sample_params()` 里一次性抽完并记成普通 Python 数字，
所以一段扰动参数**可复现、可打印、可跨进程传递**（排查坏样本时很有用）。
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass, field
from typing import Dict, Optional, Tuple

import numpy as np
import torch
from PIL import Image

__all__ = ["PerturbConfig", "ColorPerturber", "DEFAULT_OPS", "to_tensor"]


# 每个算子的默认参数区间。调参时改这里，不要改代码逻辑。
DEFAULT_OPS: Dict[str, dict] = {
    # 亮度/对比度：x*a + b
    "brightness": {"p": 0.7, "a": (0.85, 1.15), "b": (-0.04, 0.04)},
    # 饱和度：向灰度插值
    "saturation": {"p": 0.6, "s": (0.6, 1.4)},
    # 色相：绕灰度轴旋转
    "hue": {"p": 0.4, "deg": (-15.0, 15.0)},
    # 伽马：x**γ（典型的不可线性化扰动）
    "gamma": {"p": 0.5, "g": (0.7, 1.4)},
    # 色温：R/B 反相增益（一个偏暖一个偏冷）
    "gain": {"p": 0.5, "lo": 0.92, "hi": 1.08},
    # 随机 3×3
    "matrix": {"p": 0.3, "std": 0.12},
}

_LUMA = torch.tensor([0.299, 0.587, 0.114]).view(3, 1, 1)


def to_tensor(image: Image.Image) -> torch.Tensor:
    """PIL RGB → [3,H,W] float32，[0,1]。"""
    arr = np.asarray(image, dtype=np.float32) / 255.0
    return torch.from_numpy(arr).permute(2, 0, 1).contiguous()


@dataclass
class PerturbConfig:
    """扰动策略。`ops` 里每一项都可以被单独关掉（p=0）或改区间。"""

    ops: Dict[str, dict] = field(default_factory=lambda: {k: dict(v) for k, v in DEFAULT_OPS.items()})
    lut_prob: float = 0.5            # 每条样本使用 LUT 的概率
    lut_strength: Tuple[float, float] = (0.6, 1.0)
    max_ops: int = 4                 # 单次扰动最多叠加几个参数化算子
    clamp: bool = True

    @classmethod
    def from_dict(cls, cfg: Optional[dict]) -> "PerturbConfig":
        cfg = dict(cfg or {})
        ops = {k: dict(v) for k, v in DEFAULT_OPS.items()}
        for name, override in (cfg.get("ops") or {}).items():
            if name not in ops:
                raise KeyError(f"未知扰动算子 {name!r}，可选：{sorted(ops)}")
            ops[name].update(override)
        return cls(
            ops=ops,
            lut_prob=float(cfg.get("lut_prob", 0.5)),
            lut_strength=tuple(cfg.get("lut_strength", (0.6, 1.0))),
            max_ops=int(cfg.get("max_ops", 4)),
            clamp=bool(cfg.get("clamp", True)),
        )


# ---------------------------------------------------------------------- #
# 单个算子（张量侧）
# ---------------------------------------------------------------------- #
def _hue_matrix(deg: float) -> torch.Tensor:
    """绕灰度轴的色相旋转矩阵（SVG feColorMatrix hueRotate 的标准形式）。"""
    a = math.radians(deg)
    c, s = math.cos(a), math.sin(a)
    return torch.tensor([
        [0.213 + c * 0.787 - s * 0.213, 0.715 - c * 0.715 - s * 0.715, 0.072 - c * 0.072 + s * 0.928],
        [0.213 - c * 0.213 + s * 0.143, 0.715 + c * 0.285 + s * 0.140, 0.072 - c * 0.072 - s * 0.283],
        [0.213 - c * 0.213 - s * 0.787, 0.715 - c * 0.715 + s * 0.715, 0.072 + c * 0.928 + s * 0.072],
    ], dtype=torch.float32)


class ColorPerturber:
    """把一张干净的 PIL 图变成若干「同内容、不同颜色」的 **[3,H,W] 张量**。"""

    def __init__(
        self,
        cfg: Optional[PerturbConfig] = None,
        lut_bank=None,
        seed: Optional[int] = None,
        device=None,
    ) -> None:
        self.cfg = cfg or PerturbConfig()
        if self.cfg.lut_prob > 0 and lut_bank is None:
            raise ValueError("lut_prob>0 但没有提供 LUTBank；把 lut_prob 设为 0 或先加载 LUT")
        self.lut_bank = lut_bank
        self.device = device
        self._rng = random.Random(seed)

    # ------------------------------------------------------------------ #
    def sample_params(self) -> dict:
        """采样一次扰动的参数（纯 Python 数字，可复现、可打印）。"""
        cfg = self.cfg
        chosen = []
        for name, spec in cfg.ops.items():
            if self._rng.random() < float(spec.get("p", 0.0)):
                chosen.append(name)
        if len(chosen) > cfg.max_ops:
            chosen = self._rng.sample(chosen, cfg.max_ops)
        self._rng.shuffle(chosen)

        params = {"ops": []}
        for name in chosen:
            spec = cfg.ops[name]
            if name == "brightness":
                params["ops"].append((name, {
                    "a": self._rng.uniform(*spec["a"]),
                    "b": self._rng.uniform(*spec["b"]),
                }))
            elif name == "saturation":
                params["ops"].append((name, {"s": self._rng.uniform(*spec["s"])}))
            elif name == "hue":
                params["ops"].append((name, {"deg": self._rng.uniform(*spec["deg"])}))
            elif name == "gamma":
                params["ops"].append((name, {"g": self._rng.uniform(*spec["g"])}))
            elif name == "gain":
                lo, hi = float(spec["lo"]), float(spec["hi"])
                r = self._rng.uniform(lo, hi)
                # R 增则 B 减：整体亮度大致不变，只改色温（比三通道一起缩放更有区分度）
                b = min(hi, max(lo, 2.0 - r))
                params["ops"].append((name, {"gains": [r, 1.0, b]}))
            elif name == "matrix":
                # 在这里就抽好矩阵，而不是在 apply 时用 torch.randn ——
                # 后者依赖全局 RNG，同一 seed 的两个实例会得到不同结果。
                std = self._rng.uniform(0.5, 1.0) * float(spec["std"])
                m = [[(1.0 if i == j else 0.0) + std * self._rng.gauss(0.0, 1.0)
                      for j in range(3)] for i in range(3)]
                params["ops"].append((name, {"m": m}))

        use_lut = (
            self.lut_bank is not None
            and len(self.lut_bank) > 0
            and self._rng.random() < cfg.lut_prob
        )
        params["lut"] = (
            {"handle": self.lut_bank.sample(self._rng),
             "strength": self._rng.uniform(*cfg.lut_strength)}
            if use_lut else None
        )
        return params

    # ------------------------------------------------------------------ #
    def apply_lut(self, image: Image.Image, params: dict) -> Image.Image:
        """把 params 里挑中的那条 LUT 作用到 PIL 图上（没有就原样返回）。"""
        lut_arg = params.get("lut")
        if lut_arg is None:
            return image
        return self.lut_bank.apply(lut_arg["handle"], image, lut_arg["strength"])

    def apply_ops(self, img: torch.Tensor, params: dict) -> torch.Tensor:
        """在 [3,H,W] 张量上依次作用参数化算子。"""
        x = img
        for name, arg in params.get("ops", []):
            if name == "brightness":
                x = x * arg["a"] + arg["b"]
            elif name == "saturation":
                gray = (x * _LUMA.to(x.device)).sum(dim=0, keepdim=True)
                x = gray + (x - gray) * arg["s"]
            elif name == "hue":
                m = _hue_matrix(arg["deg"]).to(x.device)
                x = torch.einsum("ij,jhw->ihw", m, x)
            elif name == "gamma":
                x = x.clamp(1e-4, 1.0) ** arg["g"]
            elif name == "gain":
                g = torch.tensor(arg["gains"], dtype=x.dtype, device=x.device).view(3, 1, 1)
                x = x * g
            elif name == "matrix":
                m = torch.tensor(arg["m"], dtype=x.dtype, device=x.device)
                x = torch.einsum("ij,jhw->ihw", m, x)
            else:  # pragma: no cover
                raise ValueError(f"未知算子 {name!r}")
            if self.cfg.clamp:
                x = x.clamp(0.0, 1.0)
        if self.cfg.clamp:
            x = x.clamp(0.0, 1.0)
        return x

    # ------------------------------------------------------------------ #
    def __call__(self, image: Image.Image) -> torch.Tensor:
        """PIL 干净图 → 一次扰动后的 [3,H,W] 张量。"""
        if not isinstance(image, Image.Image):
            raise TypeError(f"需要 PIL.Image，收到 {type(image)!r}（LUT 这一步必须在 PIL 上做）")
        base = to_tensor(image)
        for _ in range(3):
            params = self.sample_params()
            out = self.apply_ops(to_tensor(self.apply_lut(image, params)), params)
            if (out - base).abs().max().item() > 1e-3:
                return out
        # 极端情况（所有算子都没被抽中）：补一个轻 gamma
        return base.clamp(1e-4, 1.0) ** 1.15

    def sample_pair(self, image: Image.Image) -> Tuple[torch.Tensor, torch.Tensor]:
        """一次返回两个互不相同的视图 (I_i, I_j)。"""
        ii = self(image)
        ij = self(image)
        for _ in range(3):
            if (ii - ij).abs().max().item() > 1e-3:
                break
            ij = self(image)
        return ii, ij
