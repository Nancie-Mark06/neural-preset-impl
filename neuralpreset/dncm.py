"""DNCM —— Dynamic Neural Color Mapping（论文 §3）。

数学形式（REPRODUCTION_PLAN.md §1.2 命题 1）：

    Y = I · M ,   M = P · T · Q

    P : [3, k]     可学习
    T : [B, k, k]  由风格码 / 内容码在运行时给出（每个样本一个）
    Q : [k, 3]     可学习

`act="none"` 时 M 与像素无关，整个模块退化成**一个 3×3 全局颜色矩阵**，
这也是 `fold` 推理（把两阶段折成一个矩阵、4K 上 0.02s）成立的前提。

`act="relu"|"gelu"` 是 G1 的实验开关：一旦开启，中间量经过逐元素非线性，
`matrix()` 与 `forward()` 不再等价，只能逐像素算。
"""

from __future__ import annotations

import torch
import torch.nn as nn

__all__ = ["DNCM", "ACT_CHOICES"]

ACT_CHOICES = ("none", "relu", "gelu")

_ACT_FACTORY = {
    "none": nn.Identity,
    "relu": nn.ReLU,
    "gelu": nn.GELU,
}


class DNCM(nn.Module):
    """Dynamic Neural Color Mapping。

    Args:
        k: 中间维度（论文用 16）。
        act: G1 开关，见模块 docstring。
        init_std: P/Q 在主对角线之外的初始化噪声标准差（G7）。

    Shape:
        输入 img: [B, 3, H, W]，取值范围 [0, 1]（G2，不做内部归一化）
        输入 T:   [B, k, k]
        输出:     [B, 3, H, W]
    """

    def __init__(self, k: int = 16, act: str = "none", init_std: float = 0.005) -> None:
        super().__init__()
        if act not in _ACT_FACTORY:
            raise ValueError(f"act 必须是 {ACT_CHOICES} 之一，收到 {act!r}")
        if k < 3:
            raise ValueError(f"k 必须 >= 3（前 3 维用于承载恒等映射），收到 {k}")

        self.k = int(k)
        self.act_name = act
        self.act = _ACT_FACTORY[act]()

        # G7：初始时 M = P·I·Q = I₃，即模型初始为「不变色」。
        # P 取 [I₃ | 0]，Q 取 [I₃ ; 0]，二者相乘正好是单位阵。
        self.P = nn.Parameter(torch.zeros(3, k))
        self.Q = nn.Parameter(torch.zeros(k, 3))
        with torch.no_grad():
            self.P[:, :3] = torch.eye(3)
            self.Q[:3, :] = torch.eye(3)
            # 破对称：否则 P、Q 的前 3 列/行永远拿不到与其他维不同的梯度结构
            self.P.add_(init_std * torch.randn_like(self.P))
            self.Q.add_(init_std * torch.randn_like(self.Q))

    # ------------------------------------------------------------------ #
    # 前向
    # ------------------------------------------------------------------ #
    def forward(self, img: torch.Tensor, T: torch.Tensor) -> torch.Tensor:
        """逐像素应用 color mapping。

        img: [B, 3, H, W]，T: [B, k, k]。
        """
        if img.dim() != 4 or img.shape[1] != 3:
            raise ValueError(f"img 期望 [B,3,H,W]，收到 {tuple(img.shape)}")
        if T.dim() != 3 or T.shape[1:] != (self.k, self.k):
            raise ValueError(f"T 期望 [B,{self.k},{self.k}]，收到 {tuple(T.shape)}")
        if T.shape[0] not in (1, img.shape[0]):
            raise ValueError(
                f"T 的 batch（{T.shape[0]}）必须等于图像 batch（{img.shape[0]}）或为 1"
            )

        B, _, H, W = img.shape
        x = img.reshape(B, 3, H * W).transpose(1, 2)  # [B, HW, 3]
        x = x @ self.P                                # [B, HW, k]
        x = self.act(x)
        x = x @ T                                     # [B, HW, k]（bmm 语义）
        x = x @ self.Q                                # [B, HW, 3]
        return x.transpose(1, 2).reshape(B, 3, H, W)

    # ------------------------------------------------------------------ #
    # 折叠成 3×3
    # ------------------------------------------------------------------ #
    def matrix(self, T: torch.Tensor) -> torch.Tensor:
        """返回等价的 3×3 矩阵 [B, 3, 3]。

        仅在 `act="none"` 时与 `forward` 严格等价（有单测守着）。
        """
        return self.P @ T @ self.Q

    @staticmethod
    def apply_matrix(color_mat: torch.Tensor, img: torch.Tensor) -> torch.Tensor:
        """用现成的 3×3 矩阵直接作用到图像（fold 推理 / 可视化用）。

        color_mat: [B, 3, 3] 或 [3, 3]；img: [B, 3, H, W]。
        """
        if color_mat.dim() == 2:
            color_mat = color_mat.unsqueeze(0)
        B, _, H, W = img.shape
        x = img.reshape(B, 3, H * W).transpose(1, 2)      # [B, HW, 3]
        y = torch.bmm(x, color_mat.expand(B, 3, 3))       # [B, HW, 3]
        return y.transpose(1, 2).reshape(B, 3, H, W)

    # ------------------------------------------------------------------ #
    def extra_repr(self) -> str:
        return f"k={self.k}, act={self.act_name}, params={3 * self.k * 2}"
