"""评测：批量出图 + 算论文的两个指标（风格相似度 / 内容相似度）。

    python -m neuralpreset.evaluate --ckpt out/last.pt \
        --content-dir /kaggle/input/val/content --style-dir /kaggle/input/val/style \
        --n-pairs 2500 --out out/eval

指标封装在 `third_party/metric/api.py`（作者的 ONNX / LDC 权重，路径已修）。

因为评测**没有配对标签**，这里只有两个数：

* `style_similarity(result, style)` —— 越大越像风格图；
* `content_similarity(result, content)` —— 越大越保住内容。

两者是此消彼长的，单看一个没有意义。所以默认还会跑一组「恒等」基线
（result 就是内容图本身）：它的内容相似度≈1、风格相似度很低，用来给上面的数一个下界。
"""

from __future__ import annotations

import argparse
import json
import os
import random
from typing import List, Optional, Tuple

import numpy as np
import torch
from PIL import Image

from .pipeline import build_model

__all__ = ["main", "load_model", "load_image_tensor", "save_tensor"]

_IMG_EXT = (".jpg", ".jpeg", ".png", ".bmp", ".webp")


# ---------------------------------------------------------------------- #
# 基础 IO
# ---------------------------------------------------------------------- #
def load_image_tensor(path: str) -> torch.Tensor:
    """读成 [1,3,H,W] 的 float32，[0,1]。"""
    img = Image.open(path).convert("RGB")
    arr = np.asarray(img, dtype=np.float32) / 255.0
    return torch.from_numpy(arr).permute(2, 0, 1)[None].contiguous()


def save_tensor(t: torch.Tensor, path: str) -> None:
    x = t.detach().float().cpu()
    if x.dim() == 4:
        x = x[0]
    arr = (x.clamp(0, 1) * 255.0).round().to(torch.uint8).permute(1, 2, 0).numpy()
    os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
    Image.fromarray(arr, mode="RGB").save(path)


def load_model(ckpt_path: str, device: torch.device, cfg_path: Optional[str] = None):
    """从 ckpt 恢复模型（ckpt 里存了 model_cfg，所以不必再给配置文件）。"""
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    mcfg = ckpt.get("model_cfg")
    if not mcfg and cfg_path:
        from .train import load_config

        mcfg = (load_config(cfg_path) or {}).get("model")
    if not mcfg:
        raise ValueError("ckpt 里没有 model_cfg，请用 --config 指定 phase0.yaml")
    model = build_model(mcfg).to(device)
    missing, unexpected = model.load_state_dict(ckpt["model"], strict=False)
    if missing:
        print(f"[ckpt] missing {len(missing)} keys, e.g. {missing[0]}")
    if unexpected:
        print(f"[ckpt] unexpected {len(unexpected)} keys, e.g. {unexpected[0]}")
    model.eval()
    print(f"[ckpt] {ckpt_path} @ epoch {ckpt.get('epoch')} "
          f"(k={model.k}, act={model.act})")
    return model


# ---------------------------------------------------------------------- #
# 配对
# ---------------------------------------------------------------------- #
def _list_images(folder: str) -> List[str]:
    names = sorted(f for f in os.listdir(folder) if f.lower().endswith(_IMG_EXT))
    if not names:
        raise FileNotFoundError(f"{folder} 里没有图片")
    return names


def build_pairs(
    content_dir: str,
    style_dir: Optional[str] = None,
    n_pairs: int = 2500,
    seed: int = 0,
    pair_file: Optional[str] = None,
) -> List[Tuple[str, str]]:
    """返回 [(content_name, style_name)]。

    * 给了 `pair_file`：每行 `内容图 风格图`（相对各自目录），顺序原样保留；
    * 否则从两个目录里**固定种子**抽样，不重复（`n_pairs` 超过组合数时自动截断）。
    """
    contents = _list_images(content_dir)
    if pair_file:
        pairs = []
        with open(pair_file, "r", encoding="utf-8") as fh:
            for ln in fh:
                ln = ln.strip()
                if not ln or ln.startswith("#"):
                    continue
                parts = ln.split()
                if len(parts) != 2:
                    raise ValueError(f"pair_file 每行应为 '内容图 风格图'，出错行：{ln!r}")
                pairs.append((parts[0], parts[1]))
        return pairs[:n_pairs] if n_pairs else pairs

    if not style_dir:
        raise ValueError("要么给 --style-dir，要么给 --pairs")
    styles = _list_images(style_dir)
    total = len(contents) * len(styles)
    take = min(n_pairs, total) if n_pairs else total
    rng = random.Random(seed)
    picked = rng.sample(range(total), take)
    return [(contents[i // len(styles)], styles[i % len(styles)]) for i in picked]


# ---------------------------------------------------------------------- #
def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Neural Preset 评测（论文指标）")
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--config", default=None, help="ckpt 里没有 model_cfg 时用")
    ap.add_argument("--content-dir", required=True)
    ap.add_argument("--style-dir", default=None)
    ap.add_argument("--pairs", default=None, help="配对清单（每行：内容图 风格图）")
    ap.add_argument("--n-pairs", type=int, default=2500)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="out/eval")
    ap.add_argument("--device", default=None)
    ap.add_argument("--patch", type=int, default=512)
    ap.add_argument("--metrics", default="both", choices=["both", "style", "content", "none"])
    ap.add_argument("--baseline", type=int, default=200,
                    help="恒等基线的对数（0=不算）；默认抽前 200 对")
    ap.add_argument("--save-renders", action="store_true", default=True)
    ap.add_argument("--no-save-renders", dest="save_renders", action="store_false")
    args = ap.parse_args(argv)

    device = torch.device(args.device or ("cuda:0" if torch.cuda.is_available() else "cpu"))
    model = load_model(args.ckpt, device, args.config)
    pairs = build_pairs(args.content_dir, args.style_dir, args.n_pairs, args.seed, args.pairs)
    os.makedirs(args.out, exist_ok=True)
    render_dir = os.path.join(args.out, "renders")
    if args.save_renders:
        os.makedirs(render_dir, exist_ok=True)
    print(f"[eval] {len(pairs)} 对，device={device}，out={args.out}")

    style_scorer = content_scorer = None
    if args.metrics in ("both", "style"):
        from third_party.metric.api import StyleScorer

        style_scorer = StyleScorer()
        print("[metric] 风格相似度：ONNX 已载入")
    if args.metrics in ("both", "content"):
        from third_party.metric.api import ContentScorer

        content_scorer = ContentScorer(device="cpu")  # LDC 很小，放 CPU 免得和生成抢显存
        print("[metric] 内容相似度：LDC 已载入")

    rows = []
    for idx, (cname, sname) in enumerate(pairs):
        cpath = os.path.join(args.content_dir, cname)
        spath = os.path.join(args.style_dir or args.content_dir, sname)
        content = load_image_tensor(cpath).to(device)
        style = load_image_tensor(spath).to(device)

        r = model.compute_preset(style)
        y = model.apply_preset(content, r, patch=args.patch)

        stem, _ = os.path.splitext(cname)
        style_stem, _ = os.path.splitext(sname)
        out_name = f"{stem}__{style_stem}.png"
        if args.save_renders:
            save_tensor(y, os.path.join(render_dir, out_name))

        row = {"index": idx, "content": cname, "style": sname, "render": out_name}
        if style_scorer is not None:
            row["style_sim"] = style_scorer.score(y, style)
        if content_scorer is not None:
            row["content_sim"] = content_scorer.score(y, content)
        rows.append(row)

        if (idx + 1) % 50 == 0 or idx + 1 == len(pairs):
            parts = [f"{idx + 1}/{len(pairs)}"]
            if "style_sim" in row:
                parts.append(f"style={np.mean([r['style_sim'] for r in rows]):.4f}")
            if "content_sim" in row:
                parts.append(f"content={np.mean([r['content_sim'] for r in rows]):.4f}")
            print("[eval] " + " ".join(parts))

    summary = {"n_pairs": len(rows), "ckpt": args.ckpt, "seed": args.seed,
               "content_dir": args.content_dir, "style_dir": args.style_dir}
    if rows and "style_sim" in rows[0]:
        summary["style_sim_mean"] = float(np.mean([r["style_sim"] for r in rows]))
    if rows and "content_sim" in rows[0]:
        summary["content_sim_mean"] = float(np.mean([r["content_sim"] for r in rows]))

    # 恒等基线：result = 内容图本身
    if args.baseline and style_scorer is not None and len(rows) > 0:
        sub = rows[: min(args.baseline, len(rows))]
        base = []
        for row in sub:
            c = load_image_tensor(os.path.join(args.content_dir, row["content"])).to(device)
            s = load_image_tensor(os.path.join(args.style_dir or args.content_dir, row["style"])).to(device)
            base.append(style_scorer.score(c, s))
        summary["baseline_style_sim_mean"] = float(np.mean(base))
        summary["baseline_n"] = len(base)
        print(f"[baseline] 恒等出图的风格相似度 = {summary['baseline_style_sim_mean']:.4f}"
              f"（n={len(base)}）—— 我们的数高于它才算学到了东西")

    with open(os.path.join(args.out, "metrics.json"), "w", encoding="utf-8") as fh:
        json.dump({"summary": summary, "pairs": rows}, fh, ensure_ascii=False, indent=2)
    with open(os.path.join(args.out, "metrics.csv"), "w", encoding="utf-8") as fh:
        cols = [c for c in ("index", "content", "style", "render", "style_sim", "content_sim")
                if c in rows[0]] if rows else []
        fh.write(",".join(cols) + "\n")
        for row in rows:
            fh.write(",".join(str(row[c]) for c in cols) + "\n")

    print("[done]")
    for k, v in summary.items():
        print(f"  {k}: {v}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
