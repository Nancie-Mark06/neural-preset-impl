"""可选的 LUT 后端：把 `.cube` 离线编译成 uint8 `npy`，mmap 读取。

**只在两种情况下用**：

1. LUT 数量涨到几千条，全量原生尺寸常驻内存扛不住
   （5000 条 × 平均 3 MB ≈ 15 GB）；
2. 需要把 LUT 作为数据集的固定产物冻结哈希、跨实验复现。

默认后端是 `lut.py`（pillow-lut，全量入内存，零重采样误差）。
这里攒着离线路径，是为了不把后路堵死 —— 配置里 `lut.backend: npy` 才启用。

注意：本模块的 `apply_lut` 作用在 **torch 张量**上；`lut.py` 的同名函数作用在
**PIL 图像**上。`NpyLutBank` 提供与 `PilLUTBank` 相同的 `sample`/`apply` 接口，
所以数据管线不用改，代价是每次 LUT 都要 PIL↔tensor 往返一次。
"""

from __future__ import annotations

import json
import os
from typing import Iterable, Optional, Sequence, Tuple

import numpy as np
import torch
from PIL import Image

__all__ = [
    "parse_cube",
    "apply_lut",
    "resample",
    "LUTBank",
    "NpyLutBank",
    "build_lut_bank",
]

Domain = Tuple[float, float]


# ---------------------------------------------------------------------- #
# 解析
# ---------------------------------------------------------------------- #
def parse_cube(path: str) -> Optional[np.ndarray]:
    """读一个 `.cube` 文件，返回 [3, N, N, N] float32。

    返回 None 表示这个文件不能用（1D LUT、数据行数对不上、或解析失败）。
    `.cube` 的约定：数据按 **R 变化最慢、B 变化最快** 排列，
    所以 `reshape(N,N,N,3)` 的下标顺序正好是 (r, g, b)。
    """
    size = None
    values = []
    try:
        with open(path, "r", encoding="utf-8", errors="ignore") as fh:
            for raw in fh:
                line = raw.strip()
                if not line or line.startswith("#"):
                    continue
                head = line.split(None, 1)[0].upper()
                if head == "LUT_3D_SIZE":
                    size = int(line.split()[1])
                    continue
                if head == "LUT_1D_SIZE":
                    return None  # 1D LUT 不在本次范围内
                if head in ("TITLE", "DOMAIN_MIN", "DOMAIN_MAX", "LUT_IN_VIDEO_RANGE"):
                    continue
                parts = line.split()
                if len(parts) < 3:
                    continue
                try:
                    values.append((float(parts[0]), float(parts[1]), float(parts[2])))
                except ValueError:
                    continue  # 非数据行
    except OSError:
        return None

    if size is None or len(values) != size ** 3:
        return None
    arr = np.asarray(values, dtype=np.float32).reshape(size, size, size, 3)
    return np.ascontiguousarray(arr.transpose(3, 0, 1, 2))  # [3, N, N, N]


# ---------------------------------------------------------------------- #
# 插值
# ---------------------------------------------------------------------- #
def _trilinear(
    lut: torch.Tensor,
    coords: torch.Tensor,
    domain: Domain = (0.0, 1.0),
) -> torch.Tensor:
    """在 LUT 表上做三线性采样。

    Args:
        lut: [3, N, N, N]，节点值（通常是 [0,1] 的输出色）。
        coords: [3, *shape]，输入色，取值范围与 `domain` 一致。
        domain: 该 LUT 定义域，通常 (0, 1)。

    Returns:
        [3, *shape]
    """
    if lut.dim() != 4 or lut.shape[0] != 3:
        raise ValueError(f"lut 期望 [3,N,N,N]，收到 {tuple(lut.shape)}")
    if coords.shape[0] != 3:
        raise ValueError(f"coords 期望首维为 3，收到 {tuple(coords.shape)}")

    C, N = lut.shape[0], lut.shape[1]
    flat = lut.reshape(C, N ** 3).to(coords.dtype)
    shape = coords.shape[1:]
    numel = 1
    for s in shape:
        numel *= int(s)

    lo, hi = float(domain[0]), float(domain[1])
    pos = (coords - lo) / (hi - lo) * (N - 1)
    pos = pos.clamp(0.0, float(N - 1))

    i0 = pos.floor().long().clamp(0, N - 1)
    i1 = (i0 + 1).clamp(max=N - 1)
    frac = (pos - i0.to(pos.dtype)).reshape(3, numel)

    out = torch.zeros(C, numel, dtype=flat.dtype, device=flat.device)
    for dr in (0, 1):
        rr = (i1 if dr else i0)[0].reshape(-1)
        wr = frac[0] if dr else 1.0 - frac[0]
        for dg in (0, 1):
            gg = (i1 if dg else i0)[1].reshape(-1)
            wg = frac[1] if dg else 1.0 - frac[1]
            for db in (0, 1):
                bb = (i1 if db else i0)[2].reshape(-1)
                wb = frac[2] if db else 1.0 - frac[2]
                idx = ((rr * N + gg) * N + bb).reshape(1, numel).expand(C, numel)
                w = (wr * wg * wb).reshape(1, numel)
                out = out + w * flat.gather(1, idx)

    return out.reshape(C, *shape)


def apply_lut(
    img: torch.Tensor,
    lut: torch.Tensor,
    strength: float = 1.0,
    domain: Domain = (0.0, 1.0),
) -> torch.Tensor:
    """把 LUT 作用到图像张量上，`strength` 控制强度（与恒等映射线性混合）。

    Args:
        img: [3, H, W]，[0,1]。
        lut: [3, N, N, N]。
        strength: 0 = 不变，1 = 完全应用。

    Returns:
        与 img 同形状，clamp 到 [0,1]。
    """
    if img.dim() < 3 or img.shape[0] != 3:
        raise ValueError(f"img 期望 [3,H,W]，收到 {tuple(img.shape)}")
    if strength <= 0.0:
        return img.clone()
    y = _trilinear(lut, img.clamp(0.0, 1.0), domain=domain)
    if strength != 1.0:
        y = img + (y - img) * float(strength)
    return y.clamp(0.0, 1.0)


def resample(lut: torch.Tensor, size: int = 33, domain: Domain = (0.0, 1.0)) -> torch.Tensor:
    """把任意尺寸的 LUT 重采样到 `size`³，用于统一打包。

    **这一步有真实的精度损失**：实测把 64³ 压到 33³，30,000 个随机色上
    平均偏差 3.2/255、p99 13/255（16³→33³ 反而更小，因为它是升采样）。
    对「造随机训练目标」够用（模型自身残差 ~4%），对「精确复现某条 LUT」不够 ——
    想无损就用默认的 pillow 后端，或者把 `--size` 调到 64。
    """
    if lut.shape[1] == size:
        return lut
    lin = torch.linspace(float(domain[0]), float(domain[1]), size,
                         dtype=lut.dtype, device=lut.device)
    grid = torch.stack(torch.meshgrid(lin, lin, lin, indexing="ij"), dim=0)  # [3,S,S,S]
    return _trilinear(lut, grid, domain=domain)


# ---------------------------------------------------------------------- #
# LUT 库
# ---------------------------------------------------------------------- #
class LUTBank:
    """一批已重采样到统一尺寸的 LUT，存成 uint8 以省内存。

    308 条 33³ float32 = 308×3×33³×4B ≈ 133 MB；uint8 只要 33 MB。
    """

    def __init__(self, table: np.ndarray, names: Optional[Sequence[str]] = None) -> None:
        if table.ndim != 5 or table.shape[1] != 3 or table.shape[0] == 0:
            raise ValueError(f"table 期望 [M,3,S,S,S]，收到 {table.shape}")
        self.table = table
        self.names = list(names) if names is not None else [str(i) for i in range(len(table))]

    def __len__(self) -> int:
        return int(self.table.shape[0])

    @property
    def size(self) -> int:
        return int(self.table.shape[2])

    def get(self, index: int, device=None) -> torch.Tensor:
        arr = np.asarray(self.table[index], dtype=np.uint8)
        t = torch.from_numpy(arr).to(torch.float32).div_(255.0)
        if device is not None:
            t = t.to(device)
        return t  # [3,S,S,S]

    def sample(self, rng=None, device=None) -> Tuple[torch.Tensor, int]:
        rng = rng or np.random.default_rng()
        if hasattr(rng, "integers"):
            idx = int(rng.integers(0, len(self)))
        else:  # random.Random
            idx = int(rng.randrange(len(self)))
        return self.get(idx, device=device), idx

    # -- 打包 ---------------------------------------------------------- #
    @classmethod
    def from_cubes(
        cls,
        paths: Iterable[str],
        size: int = 33,
        verbose: bool = False,
    ) -> Tuple["LUTBank", dict]:
        """从一堆 `.cube` 构建 LUTBank，返回 (bank, 统计信息)。"""
        names, tables = [], []
        skipped = []
        for path in paths:
            arr = parse_cube(path)
            if arr is None:
                skipped.append(os.path.basename(path))
                continue
            t = resample(torch.from_numpy(arr), size=size)
            u8 = (t.clamp(0, 1) * 255.0).round().to(torch.uint8).numpy()
            tables.append(u8)
            names.append(os.path.basename(path))
            if verbose:
                print(f"  + {os.path.basename(path)}  ({arr.shape[1]}³ → {size}³)")
        if not tables:
            raise RuntimeError("没有可用的 LUT（全部解析失败）")
        table = np.stack(tables, axis=0)
        stats = {"total": len(names), "skipped": skipped, "size": size}
        return cls(table, names), stats

    # -- 落盘 / 读取 ---------------------------------------------------- #
    def save(self, npy_path: str, manifest_path: Optional[str] = None,
             stats: Optional[dict] = None) -> None:
        os.makedirs(os.path.dirname(os.path.abspath(npy_path)) or ".", exist_ok=True)
        np.save(npy_path, self.table)
        if manifest_path:
            meta = {"count": len(self), "size": self.size, "names": self.names}
            if stats:
                meta.update({k: v for k, v in stats.items() if k not in ("size",)})
            with open(manifest_path, "w", encoding="utf-8") as fh:
                json.dump(meta, fh, ensure_ascii=False, indent=2)

    @classmethod
    def load(cls, npy_path: str, manifest_path: Optional[str] = None,
             mmap: bool = True) -> "LUTBank":
        table = np.load(npy_path, mmap_mode="r" if mmap else None)
        names = None
        if manifest_path and os.path.isfile(manifest_path):
            with open(manifest_path, "r", encoding="utf-8") as fh:
                names = json.load(fh).get("names")
        return cls(np.asarray(table) if not mmap else table, names)


class NpyLutBank:
    """把 `LUTBank` 包成和 `PilLUTBank` 一样的接口（`sample` / `apply`）。

    `apply` 需要 PIL 进、PIL 出，中间做一次张量往返 —— 比 pillow 后端慢，
    但这样 `ColorPerturber` 和数据管线完全不用区分后端。
    """

    def __init__(self, bank: LUTBank) -> None:
        self.bank = bank

    def __len__(self) -> int:
        return len(self.bank)

    @property
    def names(self) -> list:
        return self.bank.names

    def sample(self, rng=None):
        """返回一条 LUT 的下标。`rng` 可以是 `random.Random`（DataLoader 用）或 numpy RNG。"""
        if rng is None:
            rng = np.random.default_rng()
        if hasattr(rng, "integers"):  # numpy Generator / RandomState
            return int(rng.integers(0, len(self.bank)))
        return int(rng.randrange(len(self.bank)))

    def apply(self, handle, image: Image.Image, strength: float = 1.0) -> Image.Image:
        if strength <= 0.0:
            return image.copy()
        x = torch.from_numpy(np.asarray(image, dtype=np.float32) / 255.0)
        x = x.permute(2, 0, 1).contiguous()  # [3,H,W]
        y = apply_lut(x, self.bank.get(int(handle)), strength=strength)
        arr = (y.clamp(0.0, 1.0) * 255.0).round().to(torch.uint8).permute(1, 2, 0).numpy()
        return Image.fromarray(arr, mode="RGB")


def build_lut_bank(
    cube_dir: str,
    out_npy: str,
    out_manifest: Optional[str] = None,
    size: int = 33,
    recursive: bool = True,
) -> dict:
    """扫描目录下的所有 `.cube`，打包成 npy + manifest。"""
    paths = []
    for root, _dirs, files in os.walk(cube_dir):
        for name in files:
            if name.lower().endswith(".cube"):
                paths.append(os.path.join(root, name))
        if not recursive:
            break
    paths.sort()
    bank, stats = LUTBank.from_cubes(paths, size=size, verbose=False)
    bank.save(out_npy, out_manifest, stats=stats)
    stats["out"] = out_npy
    return stats
