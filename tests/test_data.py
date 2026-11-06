"""数据侧单测：LUT（两个后端）、扰动算子、COCO patch 数据集。

同样**本机没跑过**（没有 torch / pillow-lut）。不需要任何外部数据，全部用 tmp_path 现造。

分三组：

* `lut_npy` 后端：`parse_cube` / `resample` / `LUTBank` / 张量版 `apply_lut`
  —— 纯 torch + numpy，不需要额外依赖；
* `pillow` 后端（默认）：`PilLUTBank` + `pillow_lut.LUT3D` 过滤
  —— 用 `pytest.importorskip("pillow_lut")`，没装就跳过；
* 扰动与数据集：两者都走 PIL 入口，所以只有装了 pillow-lut 才能在**有 LUT** 的模式下测，
  但纯参数化扰动不需要它。
"""

from __future__ import annotations

import numpy as np
import pytest
import torch
from PIL import Image

from neuralpreset.data.lut_npy import LUTBank, apply_lut, parse_cube, resample
from neuralpreset.data.perturb import ColorPerturber, PerturbConfig, to_tensor

ALL_OPS_OFF = {"ops": {k: {"p": 0.0} for k in
                       ("brightness", "saturation", "hue", "gamma", "gain", "matrix")}}


# ---------------------------------------------------------------------- #
# 造数据的小工具
# ---------------------------------------------------------------------- #
def identity_lut(size: int, dtype=torch.float32) -> torch.Tensor:
    t = torch.linspace(0.0, 1.0, size, dtype=dtype)
    return torch.stack(torch.meshgrid(t, t, t, indexing="ij"), dim=0)


def write_cube(path: str, size: int, fn) -> None:
    """按 .cube 的约定写文件：R 变化最慢、B 变化最快。"""
    lines = [f"LUT_3D_SIZE {size}", "DOMAIN_MIN 0 0 0", "DOMAIN_MAX 1 1 1"]
    for r in range(size):
        for g in range(size):
            for b in range(size):
                v = fn(r / (size - 1), g / (size - 1), b / (size - 1))
                lines.append("%.6f %.6f %.6f" % tuple(v))
    with open(path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines) + "\n")


def make_pil(size: int = 32, seed: int = 0) -> Image.Image:
    arr = np.random.RandomState(seed).rand(size, size, 3)
    return Image.fromarray((arr * 255).round().astype(np.uint8), mode="RGB")


# ---------------------------------------------------------------------- #
# LUT：npy 后端
# ---------------------------------------------------------------------- #
def test_parse_cube_identity_roundtrip(tmp_path) -> None:
    p = tmp_path / "ident.cube"
    write_cube(str(p), 4, lambda r, g, b: (r, g, b))
    lut = parse_cube(str(p))
    assert lut is not None and lut.shape == (3, 4, 4, 4)

    img = torch.rand(3, 8, 9)
    out = apply_lut(img, torch.from_numpy(lut))
    assert torch.allclose(out, img.clamp(0, 1), atol=1e-5), \
        f"恒等 LUT 应当不改图，最大差 {(out - img.clamp(0, 1)).abs().max():.2e}"


def test_parse_cube_rejects_malformed(tmp_path) -> None:
    p = tmp_path / "bad.cube"
    p.write_text("LUT_3D_SIZE 2\n0 0 0\n1 1 1\n", encoding="utf-8")  # 少了 6 行
    assert parse_cube(str(p)) is None

    p1 = tmp_path / "1d.cube"
    p1.write_text("LUT_1D_SIZE 2\n0 0 0\n1 1 1\n", encoding="utf-8")
    assert parse_cube(str(p1)) is None


def test_resample_keeps_identity() -> None:
    small = identity_lut(4)
    big = resample(small, 9)
    assert big.shape == (3, 9, 9, 9)
    assert torch.allclose(big, identity_lut(9), atol=1e-5)


def test_lutbank_save_load(tmp_path) -> None:
    table = (np.random.RandomState(0).rand(3, 3, 5, 5, 5) * 255).round().astype(np.uint8)
    table[0] = (identity_lut(5).numpy() * 255).round().astype(np.uint8)
    bank = LUTBank(table, ["ident", "rand1", "rand2"])

    npy = str(tmp_path / "lut.npy")
    man = str(tmp_path / "lut.json")
    bank.save(npy, man)

    loaded = LUTBank.load(npy, man)
    assert len(loaded) == 3 and loaded.size == 5
    assert loaded.names == ["ident", "rand1", "rand2"]
    assert torch.allclose(loaded.get(0), identity_lut(5), atol=1 / 255 + 1e-6)
    assert loaded.get(1).shape == (3, 5, 5, 5)


# ---------------------------------------------------------------------- #
# LUT：pillow 后端（默认）
# ---------------------------------------------------------------------- #
def test_pil_bank_identity_and_strength(tmp_path) -> None:
    pytest.importorskip("pillow_lut")
    from neuralpreset.data.lut import PilLUTBank, apply_lut as apply_lut_pil

    p = tmp_path / "ident.cube"
    write_cube(str(p), 4, lambda r, g, b: (r, g, b))
    bank, stats = PilLUTBank.from_cubes([str(p)], quiet=True)
    assert len(bank) == 1 and stats["failed"] == []

    img = make_pil(16)
    same = bank.apply(0, img, 1.0)
    diff = np.abs(np.asarray(same, dtype=np.int16) - np.asarray(img, dtype=np.int16))
    assert diff.max() <= 2, f"恒等 LUT 不应改图，最大 {diff.max()}/255"

    # strength=0 必须完全不变（走的是提前返回，不经过 PIL filter）
    zero = apply_lut_pil(bank.get(0), img, strength=0.0)
    assert np.array_equal(np.asarray(zero), np.asarray(img))


def test_pil_bank_skips_broken_cube(tmp_path) -> None:
    pytest.importorskip("pillow_lut")
    from neuralpreset.data.lut import PilLUTBank

    good = tmp_path / "good.cube"
    write_cube(str(good), 2, lambda r, g, b: (1 - r, g, b))
    bad = tmp_path / "bad.cube"
    bad.write_text("这不是 cube\n", encoding="utf-8")

    bank, stats = PilLUTBank.from_cubes([str(good), str(bad)], quiet=True)
    assert len(bank) == 1
    assert len(stats["failed"]) == 1 and stats["failed"][0]["path"].endswith("bad.cube")


# ---------------------------------------------------------------------- #
# 扰动
# ---------------------------------------------------------------------- #
def test_perturber_gives_two_different_views() -> None:
    cfg = PerturbConfig.from_dict({"lut_prob": 0.0})
    p = ColorPerturber(cfg, lut_bank=None, seed=0)
    img = make_pil(32)
    base = to_tensor(img)
    ii, ij = p.sample_pair(img)

    assert (ii - base).abs().max().item() > 1e-3, "扰动几乎没生效"
    assert (ii - ij).abs().max().item() > 1e-3, "两个视图太像，L_con 会没有信号"
    for x in (ii, ij):
        assert x.shape == base.shape
        assert x.min().item() >= 0.0 and x.max().item() <= 1.0


def test_perturber_is_seeded() -> None:
    cfg = PerturbConfig.from_dict({"lut_prob": 0.0})
    img = make_pil(16)
    a = ColorPerturber(cfg, seed=42).sample_pair(img)
    b = ColorPerturber(cfg, seed=42).sample_pair(img)
    assert torch.equal(a[0], b[0]) and torch.equal(a[1], b[1])


def test_perturber_requires_pil() -> None:
    cfg = PerturbConfig.from_dict({"lut_prob": 0.0})
    with pytest.raises(TypeError, match="PIL"):
        ColorPerturber(cfg, seed=0)(torch.rand(3, 8, 8))


def test_lut_perturbation_changes_colors(tmp_path) -> None:
    pytest.importorskip("pillow_lut")
    from neuralpreset.data.lut import PilLUTBank

    p = tmp_path / "warm.cube"
    write_cube(str(p), 2, lambda r, g, b: (min(1.0, r * 1.2), g, b * 0.8))
    bank, _ = PilLUTBank.from_cubes([str(p)], quiet=True)

    cfg = PerturbConfig.from_dict({"lut_prob": 1.0, "lut_strength": [1.0, 1.0], **ALL_OPS_OFF})
    p2 = ColorPerturber(cfg, lut_bank=bank, seed=0)
    img = Image.new("RGB", (8, 8), (128, 128, 128))
    out = p2(img)
    base = to_tensor(img)
    assert (out - base).abs().max().item() > 1e-2


# ---------------------------------------------------------------------- #
# 数据集
# ---------------------------------------------------------------------- #
def test_coco_dataset_shapes(tmp_path) -> None:
    from neuralpreset.data.coco import CocoPatchDataset

    root = tmp_path / "images"
    root.mkdir()
    names = []
    rs = np.random.RandomState(0)
    for i in range(3):
        name = f"{i:04d}.png"
        arr = (rs.rand(300, 260, 3) * 255).astype(np.uint8)  # 故意非正方形
        Image.fromarray(arr).save(root / name)
        names.append(name)
    (tmp_path / "list.txt").write_text("\n".join(names) + "\n", encoding="utf-8")

    cfg = PerturbConfig.from_dict({"lut_prob": 0.0})
    ds = CocoPatchDataset(str(root), str(tmp_path / "list.txt"), size=64,
                          perturber=ColorPerturber(cfg, seed=0), seed=0)
    assert len(ds) == 3

    item = ds[0]
    assert set(item) == {"ii", "ij", "clean", "name"}
    for key in ("ii", "ij", "clean"):
        t = item[key]
        assert t.shape == (3, 64, 64) and t.dtype == torch.float32
        assert 0.0 <= t.min() and t.max() <= 1.0
    assert (item["ii"] - item["ij"]).abs().max().item() > 1e-3

    # 不带 perturber 时两个视图相同
    ds2 = CocoPatchDataset(str(root), str(tmp_path / "list.txt"), size=64, perturber=None)
    it2 = ds2[0]
    assert torch.equal(it2["ii"], it2["clean"])
