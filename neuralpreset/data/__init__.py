"""数据管线：COCO patch、颜色扰动、3D LUT。

四个模块各自独立，可以单拆出来做可视化 / 单测：

    lut.py       默认后端：pillow-lut，`.cube` 全量入内存、原生尺寸、零重采样
    lut_npy.py   可选后端：把 `.cube` 离线压成 uint8 npy（几千条 LUT 时才需要）
    perturb.py   颜色扰动算子链（LUT + 参数化算子），产出训练用的成对视图
    coco.py      COCO 干净 patch 数据集，__getitem__ 直接返回 (I_i, I_j)

`lut.py` 与 `lut_npy.py` 暴露的接口是平行的，选哪个由配置 `lut.backend` 决定；
`NpyLutBank` / `PilLUTBank` 都有 `sample(rng)` + `apply(handle, pil, strength)`，
所以 `perturb.py` 和数据管线对后端无感。
"""

from .coco import CocoPatchDataset
from .lut import PilLUTBank, apply_lut, scan_cubes
from .perturb import ColorPerturber, PerturbConfig, to_tensor

__all__ = [
    "CocoPatchDataset",
    "PilLUTBank",
    "apply_lut",
    "scan_cubes",
    "ColorPerturber",
    "PerturbConfig",
    "to_tensor",
]
