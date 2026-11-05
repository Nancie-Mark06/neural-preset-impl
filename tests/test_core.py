"""数学性质与 pipeline 单测（M1 验收物，见 REPRODUCTION_PLAN.md §3.4）。

**这些测试本机没跑过**（本机没有 torch），第一次上 Kaggle 时先跑：

    python -m pytest tests -q

7 条（+2 条扩展）都只依赖 torch/torchvision，不需要数据、不需要权重下载：
模型一律用 `pretrained_mode="none"` 构造。
"""

from __future__ import annotations

import pytest
import torch

pytest.importorskip("torchvision")  # 骨架需要 torchvision 的 efficientnet_b0

from neuralpreset.dncm import DNCM                      # noqa: E402
from neuralpreset.losses import (                       # noqa: E402
    NeuralPresetLoss, assert_unit_range, con_loss, rec_loss,
)
from neuralpreset.pipeline import NeuralPreset          # noqa: E402

K = 8  # 测试用小维度，快


@pytest.fixture(scope="module")
def model() -> NeuralPreset:
    m = NeuralPreset(k=K, act="none", pretrained_mode="none")
    m.eval()
    return m


# 1) DNCM 等价于 3×3 矩阵 ------------------------------------------------ #
def test_dncm_matches_3x3() -> None:
    dncm = DNCM(k=K, act="none")
    img = torch.rand(2, 3, 37, 41)
    T = torch.randn(2, K, K) * 0.05

    y = dncm(img, T)
    ref = DNCM.apply_matrix(dncm.matrix(T), img)
    assert torch.allclose(y, ref, atol=1e-6), (y - ref).abs().max()


def test_act_switch_changes_result() -> None:
    img = torch.rand(1, 3, 8, 8)
    T = torch.randn(1, K, K) * 0.1
    y_none = DNCM(k=K, act="none")(img, T)
    y_relu = DNCM(k=K, act="relu")(img, T)
    assert not torch.allclose(y_none, y_relu, atol=1e-4)


# 2) 初始化近似恒等 ------------------------------------------------------ #
def test_init_near_identity(model: NeuralPreset) -> None:
    x = torch.rand(1, 3, 64, 64)
    with torch.no_grad():
        y = model.stylize_hr(x)
        d, r = model.encode(model.to_thumb(x))
    delta = (y - x).abs().max().item()
    assert delta < 0.02, f"初始映射偏离恒等 {delta:.4f}（G7：T 的 bias 应为单位阵）"

    # 折叠出来的矩阵也应接近 I₃
    m = model.fold(r)[0]
    assert (m - torch.eye(3)).abs().max().item() < 0.05


# 3) 确定性 + preset 复用 ------------------------------------------------ #
def test_preset_reuse_is_deterministic(model: NeuralPreset) -> None:
    style = torch.rand(1, 3, 48, 56)
    other = torch.rand(1, 3, 48, 56)
    with torch.no_grad():
        r1 = model.compute_preset(style)
        r2 = model.compute_preset(style)
        r3 = model.compute_preset(other)
    assert torch.equal(r1, r2)
    assert (r1 - r3).abs().max().item() > 1e-4, "不同风格图应当给出不同 preset"

    img = torch.rand(1, 3, 40, 40)
    with torch.no_grad():
        y1 = model.apply_preset(img, r1)
        y2 = model.apply_preset(img, r1)
    assert torch.equal(y1, y2)


# 4) 分块渲染 == 整图渲染 ------------------------------------------------ #
def test_patch_matches_whole(model: NeuralPreset) -> None:
    img = torch.rand(1, 3, 200, 304)
    with torch.no_grad():
        d, r = model.encode(model.to_thumb(img))
        whole = model.apply_codes(img, d, r, patch=None)
        pieces = model.apply_codes(img, d, r, patch=64)
    assert whole.shape == pieces.shape == img.shape
    assert torch.allclose(whole, pieces, atol=1e-6), \
        f"接缝或越界，最大差 {(whole - pieces).abs().max().item():.2e}"


# 5) 参数量 -------------------------------------------------------------- #
def test_param_count() -> None:
    m = NeuralPreset(k=16, pretrained_mode="none")
    n = m.num_parameters()
    assert 4.0e6 < n < 5.6e6, f"{n/1e6:.2f}M 不在预期范围（EfficientNet-B0 主干约 4.0M + 两个头）"
    print(f"[info] params = {n/1e6:.3f}M（论文 5.15M，差异见 G6）")


# 6) 损失量纲 ------------------------------------------------------------ #
def test_loss_scale() -> None:
    x = torch.rand(2, 3, 16, 16)
    zero = torch.zeros_like(x)
    assert rec_loss(x, x, x, x).item() == pytest.approx(0.0, abs=1e-7)
    assert rec_loss(x + 0.1, x, x, x).item() == pytest.approx(0.1, abs=1e-3)

    z = torch.rand(2, 4, 4)
    assert con_loss(z, z).item() == pytest.approx(0.0, abs=1e-9)

    with pytest.raises(ValueError, match=r"\[0,1\]"):
        assert_unit_range(x * 255.0, "Ii")

    crit = NeuralPresetLoss(lambda_con=10.0)
    out = {"Yi": x + 0.1, "Yj": x, "Zi": z, "Zj": z}
    parts = crit(out, x, x)
    assert set(parts) == {"total", "rec", "con", "con_weighted"}
    assert parts["total"].item() == pytest.approx(
        parts["rec"].item() + 10.0 * parts["con"].item(), rel=1e-6)


# 7) 训练前向的交换语义（最容易写错的一条） ------------------------------ #
def test_forward_train_swaps_style_code(model: NeuralPreset) -> None:
    Ii = torch.rand(2, 3, 32, 32)
    Ij = torch.rand(2, 3, 32, 32)
    with torch.no_grad():
        out = model.forward_train(Ii, Ij)
        expect_yi = model.stylize(out["Zj"], out["r_i"])
        expect_yj = model.stylize(out["Zi"], out["r_j"])
    assert torch.allclose(out["Yi"], expect_yi, atol=1e-6)
    assert torch.allclose(out["Yj"], expect_yj, atol=1e-6)
    assert out["M_n"].shape == out["M_s"].shape == (2, 3, 3)
