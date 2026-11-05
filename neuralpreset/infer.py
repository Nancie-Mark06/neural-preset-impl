"""推理：一张内容图 + 一张风格图 → 一张出图。

    # 单张
    python -m neuralpreset.infer --ckpt out/last.pt --content a.jpg --style b.jpg --out out/a_b.png

    # 一张风格图刷一整个文件夹（preset 只算一次，和论文的卖点一致）
    python -m neuralpreset.infer --ckpt out/last.pt --content-dir photos/ --style b.jpg --out-dir out/b/

    # 顺便把折叠出来的 3×3 打出来（act=none 时才有意义）
    python -m neuralpreset.infer ... --matrix
"""

from __future__ import annotations

import argparse
import os
from typing import List

import numpy as np
import torch
from PIL import Image

from .evaluate import load_image_tensor, load_model, save_tensor

__all__ = ["main"]

_IMG_EXT = (".jpg", ".jpeg", ".png", ".bmp", ".webp")


def list_images(folder: str) -> List[str]:
    names = sorted(f for f in os.listdir(folder) if f.lower().endswith(_IMG_EXT))
    if not names:
        raise FileNotFoundError(f"{folder} 里没有图片")
    return names


def side_by_side(content: torch.Tensor, result: torch.Tensor, style: torch.Tensor,
                 size: int = 512) -> torch.Tensor:
    """内容 | 结果 | 风格 三宫格，存图用。"""
    import torch.nn.functional as F

    def fit(t: torch.Tensor) -> torch.Tensor:
        h, w = t.shape[-2:]
        s = size / float(max(h, w))
        if s >= 1.0:
            return t
        return F.interpolate(t, scale_factor=s, mode="bicubic", antialias=True).clamp(0, 1)

    c = fit(content)
    r = fit(result)
    st = fit(style)
    h = min(c.shape[-2], r.shape[-2], st.shape[-2])
    w = min(c.shape[-1], r.shape[-1], st.shape[-1])

    def center(t: torch.Tensor) -> torch.Tensor:
        dh = (t.shape[-2] - h) // 2
        dw = (t.shape[-1] - w) // 2
        return t[..., dh:dh + h, dw:dw + w]

    return torch.cat([center(c), center(r), center(st)], dim=-1)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Neural Preset 推理")
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--config", default=None)
    ap.add_argument("--content", default=None, help="单张内容图")
    ap.add_argument("--content-dir", default=None, help="整个文件夹")
    ap.add_argument("--style", required=True)
    ap.add_argument("--out", default=None, help="单张输出路径")
    ap.add_argument("--out-dir", default=None)
    ap.add_argument("--device", default=None)
    ap.add_argument("--patch", type=int, default=512, help="0 表示不切块")
    ap.add_argument("--matrix", action="store_true", help="打印折叠出的 3×3")
    ap.add_argument("--compare", action="store_true", help="另存 内容|结果|风格 三宫格")
    args = ap.parse_args(argv)

    if bool(args.content) == bool(args.content_dir):
        ap.error("--content 与 --content-dir 必须二选一")
    if bool(args.out) == bool(args.out_dir):
        ap.error("--out 与 --out-dir 必须二选一")

    device = torch.device(args.device or ("cuda:0" if torch.cuda.is_available() else "cpu"))
    model = load_model(args.ckpt, device, args.config)
    patch = None if args.patch == 0 else int(args.patch)

    style = load_image_tensor(args.style).to(device)
    with torch.no_grad():
        r = model.compute_preset(style)

    if args.matrix:
        with torch.no_grad():
            m = model.fold(r)
        print("[matrix] 折叠后的 3×3（Y = I · M），act =", model.act)
        print(np.array2string(m[0].detach().cpu().numpy(), precision=4, suppress_small=True))
        print("[matrix] 列和（对灰度的增益）:", m[0].sum(dim=0).detach().cpu().numpy().round(4))

    if args.content:
        content = load_image_tensor(args.content).to(device)
        with torch.no_grad():
            y = model.apply_preset(content, r, patch=patch)
        save_tensor(y, args.out)
        print(f"[out] {args.out}  ({y.shape[-1]}×{y.shape[-2]})")
        if args.compare:
            stem, ext = os.path.splitext(args.out)
            save_tensor(side_by_side(content, y, style), f"{stem}_compare{ext or '.png'}")
            print(f"[out] {stem}_compare{ext or '.png'}")
        return 0

    names = list_images(args.content_dir)
    os.makedirs(args.out_dir, exist_ok=True)
    print(f"[infer] {len(names)} 张内容图 × 1 张风格图 → {args.out_dir}")
    for idx, name in enumerate(names, 1):
        content = load_image_tensor(os.path.join(args.content_dir, name)).to(device)
        with torch.no_grad():
            y = model.apply_preset(content, r, patch=patch)
        out_path = os.path.join(args.out_dir, os.path.splitext(name)[0] + ".png")
        save_tensor(y, out_path)
        if idx % 20 == 0 or idx == len(names):
            print(f"[infer] {idx}/{len(names)}")
    print("[done]")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
