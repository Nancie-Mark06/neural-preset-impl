"""训练损失（论文式 5 / 7 / 8）。

    L_rec = ‖Y_i − I_i‖₁ + ‖Y_j − I_j‖₁
    L_con = ‖Z_i − Z_j‖²
    L     = L_rec + λ · L_con ,  λ = 10

两个实现细节（G3）：

* 都是**按元素求 mean**，不是 sum —— 所以 L_rec 的数值与图像分辨率、batch 大小无关，
  可以直接和论文的曲线对比数量级。
* 输入值域必须是 [0, 1]（G2）。值域错了 L 的量级会差 255 倍，
  函数里用 `assert_unit_range` 显式拒绝（对应单测 `test_loss_scale`）。
"""

from __future__ import annotations

import torch
import torch.nn as nn

__all__ = ["NeuralPresetLoss", "rec_loss", "con_loss", "assert_unit_range"]


def assert_unit_range(x: torch.Tensor, name: str = "tensor", tol: float = 1e-3) -> None:
    """挡住「忘了 /255」这类最常见的量纲错误。"""
    if not torch.is_floating_point(x):
        raise TypeError(f"{name} 必须是浮点张量，收到 {x.dtype}")
    lo, hi = x.detach().amin().item(), x.detach().amax().item()
    if lo < -tol or hi > 1.0 + tol:
        raise ValueError(
            f"{name} 的取值范围应在 [0,1]（G2），实际 [{lo:.4f}, {hi:.4f}]。"
            "如果输入是 0–255，请先除以 255。"
        )


def rec_loss(Yi: torch.Tensor, Ii: torch.Tensor,
             Yj: torch.Tensor, Ij: torch.Tensor) -> torch.Tensor:
    """L_rec = |Yi − Ii|.mean() + |Yj − Ij|.mean()。"""
    return (Yi - Ii).abs().mean() + (Yj - Ij).abs().mean()


def con_loss(Zi: torch.Tensor, Zj: torch.Tensor) -> torch.Tensor:
    """L_con = mean((Zi − Zj)²)。

    内容相同的两张视图必须落在同一个规范空间；不做平方根（论文是平方）。
    """
    return ((Zi - Zj) ** 2).mean()


class NeuralPresetLoss(nn.Module):
    """组合损失，返回各项 + 总和，方便直接写日志。

    Returns:
        dict: {"total", "rec", "con", "con_weighted"}
    """

    def __init__(self, lambda_con: float = 10.0, check_range: bool = True) -> None:
        super().__init__()
        self.lambda_con = float(lambda_con)
        self.check_range = bool(check_range)

    def forward(self, out: dict, Ii: torch.Tensor, Ij: torch.Tensor) -> dict:
        Yi, Yj = out["Yi"], out["Yj"]
        Zi, Zj = out["Zi"], out["Zj"]

        if self.check_range:
            assert_unit_range(Ii, "Ii")
            assert_unit_range(Ij, "Ij")

        rec = rec_loss(Yi, Ii, Yj, Ij)
        con = con_loss(Zi, Zj)
        con_w = self.lambda_con * con
        return {"total": rec + con_w, "rec": rec, "con": con, "con_weighted": con_w}

    def extra_repr(self) -> str:
        return f"lambda_con={self.lambda_con}"
