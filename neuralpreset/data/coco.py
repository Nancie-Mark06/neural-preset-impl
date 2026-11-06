"""COCO 干净 patch 数据集。

每个样本返回**一对**视图 (I_i, I_j)：同一块背景、两次不同颜色扰动。
这是自监督训练的全部数据来源（论文 §3，我们没有任何成对标注）。

约定：
* 清单文件（`coco10k.txt`）每行一个相对路径，**固定**下来，避免每次实验换数据造成指标抖动（U1）；
* 先把短边缩到 `size`，再随机裁 `size×size`。颜色扰动与几何裁剪可交换，
  先裁后扰动更省算力；
* **裁完还是 PIL**：LUT 那一步（`pillow-lut`）只能在 PIL 上做，
  所以数据集把 PIL patch 交给 `ColorPerturber`，由它返回 [3,S,S] 张量；
* 值域 [0,1]（G2），dtype float32。
"""

from __future__ import annotations

import os
import random
from typing import Callable, List, Optional

import torch
from PIL import Image
from torch.utils.data import Dataset

__all__ = ["CocoPatchDataset", "load_image", "read_list"]


def read_list(path: str) -> List[str]:
    with open(path, "r", encoding="utf-8") as fh:
        return [ln.strip() for ln in fh if ln.strip() and not ln.startswith("#")]


def load_image(path: str, size: Optional[int] = None, resample=Image.BICUBIC) -> Image.Image:
    """读图并（可选）把短边缩到 `size`，返回 RGB PIL。"""
    img = Image.open(path)
    img = img.convert("RGB")
    if size is not None:
        w, h = img.size
        short = min(w, h)
        if short != size:
            scale = size / float(short)
            new = (max(size, int(round(w * scale))), max(size, int(round(h * scale))))
            img = img.resize(new, resample)
    return img


class CocoPatchDataset(Dataset):
    """返回 {"ii", "ij", "clean", "name"} 的 COCO patch 数据集。

    Args:
        root: 图片根目录（清单里的路径相对它）。
        list_file: 清单文件。
        size: patch 边长（论文 256）。
        perturber: `ColorPerturber`，负责把干净 patch 变成两个视图。
        seed: 采样种子；实际每个 worker 会再叠加自己的 id。
        verify: 初始化时检查文件是否存在（大数据集上会慢，可关）。
    """

    def __init__(
        self,
        root: str,
        list_file: str,
        size: int = 256,
        perturber: Optional[Callable] = None,
        seed: int = 0,
        verify: bool = True,
        limit: Optional[int] = None,
    ) -> None:
        self.root = root
        self.size = int(size)
        self.perturber = perturber
        self.seed = int(seed)
        self.names = read_list(list_file)
        if limit is not None:
            self.names = self.names[: int(limit)]
        if not self.names:
            raise RuntimeError(f"清单为空：{list_file}")
        if verify:
            missing = [n for n in self.names[: min(len(self.names), 5000)]
                       if not os.path.isfile(os.path.join(root, n))]
            if missing:
                raise FileNotFoundError(
                    f"{len(missing)} 个文件在 {root} 下找不到，例如 {missing[0]!r}。"
                    "检查清单里的路径是否相对于 root。"
                )
        self._rng: Optional[random.Random] = None

    # ------------------------------------------------------------------ #
    def __len__(self) -> int:
        return len(self.names)

    def _get_rng(self) -> random.Random:
        # 每个 worker / 每个 epoch 用不同但可复现的随机流
        if self._rng is None:
            info = torch.utils.data.get_worker_info()
            wid = info.id if info is not None else 0
            base = (self.seed * 1_000_003 + wid * 9176 + (torch.initial_seed() >> 12)) % (2 ** 31)
            self._rng = random.Random(base)
        return self._rng

    def _load_patch(self, name: str) -> Image.Image:
        """读一张图并随机裁出 `size×size` 的 **PIL** patch。"""
        path = os.path.join(self.root, name)
        img = load_image(path, size=self.size)
        w, h = img.size
        rng = self._get_rng()
        left = rng.randrange(0, max(1, w - self.size + 1))
        top = rng.randrange(0, max(1, h - self.size + 1))
        return img.crop((left, top, left + self.size, top + self.size))

    # ------------------------------------------------------------------ #
    def __getitem__(self, index: int) -> dict:
        name = self.names[index]
        from .perturb import to_tensor  # 局部导入，避免 data 包初始化顺序问题

        pil = self._load_patch(name)
        clean = to_tensor(pil)  # [3,S,S]
        if self.perturber is None:
            return {"ii": clean, "ij": clean.clone(), "clean": clean, "name": name}
        ii, ij = self.perturber.sample_pair(pil)
        return {"ii": ii, "ij": ij, "clean": clean, "name": name}
