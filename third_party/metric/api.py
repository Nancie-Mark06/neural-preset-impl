"""论文指标（风格相似度 / 内容相似度）的可用封装。

`style_similiary/` 与 `content_similiary/` 里的 **原始文件一字未改**，保留作者版本以便溯源；
本文件只做三件事：

1. 把写死的 `./StyleSimiliaryDiscriminator.onnx`、`./ldc.pth` 换成相对本文件的绝对路径（可用参数覆盖）；
2. `structural_similarity(..., multichannel=True)` → `channel_axis=None`（`multichannel` 在
   skimage ≥ 0.23 被移除，且我们先把边缘图 squeeze 成 [H,W] 二维，语义等价）；
3. 提供 `score()` 单对接口，避免为了算 2500 对而反复起进程 / 反复 load 模型。

**关于作者代码里的 `h, w = image.size`**：看起来像宽高写反，其实没错 ——
PIL 的 `size` 是 `(W, H)`，赋给 `h, w` 后 `h` 装的是宽；紧接着
`resize(image, (w, h))` 传的 `(H, W)` 正好是 torchvision 要的顺序。变量名误导，
但**不要按"修 bug"去改**，改了才会真的把纵横比搞错。

用法::

    from third_party.metric.api import StyleScorer, ContentScorer
    ss = StyleScorer()
    print(ss.score("out/0001.png", "data/style/0001.jpg"))     # 越大越像
    cs = ContentScorer()
    print(cs.score("out/0001.png", "data/content/0001.jpg"))   # 越大越保内容
"""

from __future__ import annotations

import os
import sys
from typing import Sequence, Union

import numpy as np
import torch
from PIL import Image

__all__ = ["StyleScorer", "ContentScorer", "DEFAULT_STYLE_ONNX", "DEFAULT_LDC_PTH"]

_HERE = os.path.dirname(os.path.abspath(__file__))
_CONTENT_DIR = os.path.join(_HERE, "content_similiary")
_STYLE_DIR = os.path.join(_HERE, "style_similiary")

DEFAULT_STYLE_ONNX = os.path.join(_STYLE_DIR, "StyleSimiliaryDiscriminator.onnx")
DEFAULT_LDC_PTH = os.path.join(_CONTENT_DIR, "ldc.pth")

ImageLike = Union[str, Image.Image, torch.Tensor]


def _to_pil(img: ImageLike) -> Image.Image:
    """接受路径 / PIL / [3,H,W] 或 [1,3,H,W] 的 [0,1] 张量。"""
    if isinstance(img, str):
        return Image.open(img).convert("RGB")
    if isinstance(img, Image.Image):
        return img.convert("RGB")
    if isinstance(img, torch.Tensor):
        t = img.detach().float().cpu()
        if t.dim() == 4:
            t = t[0]
        if t.dim() != 3 or t.shape[0] != 3:
            raise ValueError(f"张量应为 [3,H,W] 或 [1,3,H,W]，收到 {tuple(img.shape)}")
        arr = (t.clamp(0, 1) * 255.0).round().to(torch.uint8).permute(1, 2, 0).numpy()
        return Image.fromarray(arr, mode="RGB")
    raise TypeError(f"不支持的图像类型：{type(img)}")


# ---------------------------------------------------------------------- #
class StyleScorer:
    """风格相似度：StyleSimiliaryDiscriminator.onnx（输入 512×512、0–255 的 float32）。

    与 `calc_style_similiary.py` 完全一致：`run(['score'], {'ref': style, 'img': result})`。
    """

    def __init__(self, onnx_path: str | None = None, providers: Sequence[str] | None = None):
        import onnxruntime  # 只在真正用到时才要求装

        path = onnx_path or DEFAULT_STYLE_ONNX
        if not os.path.isfile(path):
            raise FileNotFoundError(
                f"找不到 {path}；该文件 44.6 MB，随仓库提交，检查 .gitignore 是否误伤")
        self.onnx_path = path
        self.session = onnxruntime.InferenceSession(
            path, providers=list(providers) if providers else ["CPUExecutionProvider"])

    @staticmethod
    def _as_array(img: ImageLike) -> np.ndarray:
        return np.asarray(_to_pil(img).resize((512, 512), Image.BICUBIC)).astype(np.float32)

    def score(self, result: ImageLike, style: ImageLike) -> float:
        out = self.session.run(
            ["score"], {"ref": self._as_array(style), "img": self._as_array(result)})[0]
        return float(np.asarray(out).reshape(-1)[0])

    def score_pairs(self, results: Sequence[ImageLike], styles: Sequence[ImageLike]) -> dict:
        """按**位置**配对（调用方负责按文件名对齐），返回逐对分数与均值。"""
        if len(results) != len(styles):
            raise ValueError(f"数量不等：{len(results)} vs {len(styles)}")
        scores = [self.score(r, s) for r, s in zip(results, styles)]
        return {"mean": float(np.mean(scores)), "n": len(scores), "scores": scores}


# ---------------------------------------------------------------------- #
class ContentScorer:
    """内容相似度：LDC 边缘图 + SSIM（`calc_content_similiary.py` 的等价实现）。

    作者流程（逐步照抄，改的只是路径与 skimage API）：

    1. `h, w = image.size` → 各自截到 32 的倍数 → `resize((w, h))`（即 (H, W)）；
    2. `to_tensor * 255 - mean([103.939, 116.779, 123.68])` 送 LDC；
    3. `ldc.postprocess_edges(edges)` → [H,W] 的 uint8（0–255），除 255；
    4. 两张边缘图求 SSIM。
    """

    def __init__(self, ldc_pth: str | None = None, device: str = "cpu", ldc_dir: str | None = None):
        ckpt = ldc_pth or DEFAULT_LDC_PTH
        if not os.path.isfile(ckpt):
            raise FileNotFoundError(f"找不到 {ckpt}")
        # ldc.py 假定自己被 import 时目录里有 ldc.pth（原脚本用 './ldc.pth'）
        mod_dir = ldc_dir or _CONTENT_DIR
        if mod_dir not in sys.path:
            sys.path.insert(0, mod_dir)
        import ldc  # noqa: WPS433  (vendored，惰性 import)

        self._ldc = ldc
        self.device = torch.device(device)
        self.model = ldc.LDC()
        self.model.load_state_dict(torch.load(ckpt, map_location="cpu"))
        self.model.to(self.device).eval()
        self._mean = torch.tensor([103.939, 116.779, 123.68]).to(self.device).view(1, 3, 1, 1)

    @torch.no_grad()
    def edge(self, img: ImageLike) -> np.ndarray:
        """返回 [H,W] 的边缘图，值域约 [0,1]（float32）。"""
        pil = _to_pil(img)
        h, w = pil.size                      # 注意：作者写法如此，h 里装的是宽
        h = int(h - h % 32)
        w = int(w - w % 32)
        pil = pil.resize((w, h), Image.BICUBIC)   # (W', H')，对应 torchvision 的 (H,W)
        arr = np.asarray(pil, dtype=np.float32) / 255.0
        x = torch.from_numpy(arr).permute(2, 0, 1)[None].to(self.device) * 255.0
        edges = self.model(x - self._mean)
        avg = self._ldc.postprocess_edges(edges)              # numpy [H,W]，uint8-ish
        return (np.asarray(avg, dtype=np.float32) / 255.0)

    def score(self, result: ImageLike, content: ImageLike, data_range: float | None = None) -> float:
        from skimage.metrics import structural_similarity

        a = self.edge(result)
        b = self.edge(content)
        if a.shape != b.shape:
            raise ValueError(f"边缘图尺寸不一致 {a.shape} vs {b.shape}；"
                             "result 与 content 应同尺寸（推理时按内容图尺寸出图）")
        if data_range is not None:
            return float(structural_similarity(a, b, data_range=data_range))
        # 不传 data_range：skimage 对 float64 推断为 2.0，与作者原脚本（multichannel + 无 data_range）一致
        return float(structural_similarity(a, b))

    def score_pairs(self, results: Sequence[ImageLike], contents: Sequence[ImageLike],
                    data_range: float | None = None) -> dict:
        if len(results) != len(contents):
            raise ValueError(f"数量不等：{len(results)} vs {len(contents)}")
        scores = [self.score(r, c, data_range=data_range) for r, c in zip(results, contents)]
        return {"mean": float(np.mean(scores)), "n": len(scores), "scores": scores}
