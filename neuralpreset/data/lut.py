"""3D LUT：`.cube` → PIL 的 `Color3DLUT`，**全量常驻内存**。

LUT 是论文用来「造训练目标」的工具（G10），不是模型的一部分 ——
模型永远只输出一个 3×3 矩阵，LUT 只出现在 dataloader 里。

为什么是「全量加载 + `pillow-lut`」而不是「离线编译成 npy」：

* 308 条 LUT 在内存里约 409 MB（原生尺寸 float32）。Kaggle 16 GB 无压力；
* DataLoader 在 Linux 上用 `fork`，只读数据 copy-on-write 共享，
  所以**不是** 409 MB × worker 数；
* 原生尺寸 ⇒ 零重采样误差。之前「统一压到 33³ uint8」的方案在 64³ 那 113 条上
  平均偏差 3.2/255、p99 13/255，这一步现在完全不需要；
* 代码少 200 行：没有 npy、没有 manifest、没有中间产物，`.cube` 目录就是唯一输入。

代价：pip 装 `pillow-lut`（纯 Python wheel，`py3-none-any`，不需要编译器）。
如果以后 LUT 数量涨到几千条（5000 条 × 3 MB ≈ 15 GB），改用
`neuralpreset/data/lut_npy.py` 的后端（配置里 `lut.backend: npy`）。
"""

from __future__ import annotations

import os
import random
import time
from typing import List, Optional, Sequence, Tuple

from PIL import Image

try:  # 让「没装依赖」的报错可操作，而不是 ImportError 裸奔
    from pillow_lut import load_cube_file
except ImportError as exc:  # pragma: no cover
    raise ImportError(
        "需要 pillow-lut 来解析 .cube：pip install pillow-lut\n"
        "（Kaggle 关网时把 pillow_lut 的 wheel 放进 code dataset 再 pip install）"
    ) from exc

__all__ = ["apply_lut", "scan_cubes", "PilLUTBank"]


# ---------------------------------------------------------------------- #
# 扫描 / 应用
# ---------------------------------------------------------------------- #
def scan_cubes(cube_dir: str, recursive: bool = True) -> List[str]:
    """递归找出目录下所有 `.cube`，按路径排序（保证同样的输入得到同样的顺序）。"""
    found: List[str] = []
    for root, dirs, files in os.walk(cube_dir):
        for name in files:
            if name.lower().endswith(".cube"):
                found.append(os.path.join(root, name))
        if not recursive:
            break
    return sorted(found)


def apply_lut(lut, image: Image.Image, strength: float = 1.0) -> Image.Image:
    """把一条 LUT 作用到 PIL 图像上，`strength` 与原因做线性混合。

    Args:
        lut: `pillow_lut.load_cube_file()` 的返回值。
        image: PIL RGB 图。
        strength: 0 = 原图，1 = 完全应用。人工调色的 LUT 全强度常常「过火」，
                  训练时随机取 0.6–1.0 能让目标分布更宽。

    Returns:
        新的 PIL 图（`Image.filter` 不会原地改）。
    """
    out = image.filter(lut)
    strength = float(strength)
    if strength >= 1.0:
        return out
    if strength <= 0.0:
        return image.copy()
    return Image.blend(image, out, strength)


# ---------------------------------------------------------------------- #
# LUT 库
# ---------------------------------------------------------------------- #
class PilLUTBank:
    """一批已解析的 LUT，常驻内存。

    接口故意做得和后端无关（`sample` 拿到 handle、`apply` 用 handle 处理 PIL 图），
    这样 `ColorPerturber` 不需要知道底下是 PIL LUT 还是 uint8 npy。
    """

    def __init__(self, luts: Sequence, paths: Optional[Sequence[str]] = None) -> None:
        if len(luts) == 0:
            raise ValueError("LUTBank 为空")
        self.luts = list(luts)
        self.paths = list(paths) if paths is not None else [str(i) for i in range(len(luts))]

    def __len__(self) -> int:
        return len(self.luts)

    @property
    def names(self) -> List[str]:
        return [os.path.basename(p) for p in self.paths]

    def get(self, index: int):
        return self.luts[int(index)]

    def sample(self, rng: Optional[random.Random] = None):
        rng = rng or random
        return self.luts[rng.randrange(len(self.luts))]

    def apply(self, handle, image: Image.Image, strength: float = 1.0) -> Image.Image:
        return apply_lut(handle, image, strength)

    # ------------------------------------------------------------------ #
    @classmethod
    def from_cubes(
        cls,
        paths: Sequence[str],
        limit: Optional[int] = None,
        quiet: bool = False,
    ) -> Tuple["PilLUTBank", dict]:
        """逐个解析 `.cube`，坏文件跳过并记账（G10：LUT 来源质量是第二大风险）。"""
        paths = list(paths)
        if limit is not None:
            paths = paths[: int(limit)]
        ok_luts, ok_paths, failed = [], [], []
        t0 = time.time()
        for idx, path in enumerate(paths, 1):
            try:
                ok_luts.append(load_cube_file(path))
                ok_paths.append(path)
            except Exception as exc:  # noqa: BLE001  单条坏文件不该打断启动
                failed.append({"path": path, "error": f"{type(exc).__name__}: {exc}"})
            if not quiet and (idx % 50 == 0 or idx == len(paths)):
                print(f"[lut] 已加载 {idx}/{len(paths)}")
        stats = {
            "total": len(ok_luts),
            "failed": failed,
            "seconds": round(time.time() - t0, 2),
        }
        return cls(ok_luts, ok_paths), stats
