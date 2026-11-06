"""【可选】把一堆 `.cube` 压成 `lut33_u8.npy` + `lut_manifest.json`。

**默认的 LUT 后端不需要跑这个脚本**：`lut.backend: pillow` 会把 `.cube` 全量载入
内存（308 条 ≈ 409 MB，原生尺寸、零重采样），训练时用 `pillow-lut` 直接查表。
只有在下面两种情况才跑它：

1. LUT 数量涨到几千条，全量入内存装不下（5000 条原生尺寸 ≈ 15 GB）；
2. 需要一份冻结的二进制产物做跨实验复现。

    python scripts/prepare_lut.py --cube-dir C:\\...\\cubes --out-npy data/lut33_u8.npy

产物（由 `neuralpreset/data/lut_npy.py::LUTBank` 负责读写）：

* `.npy`：uint8，形状 `[M, 3, S, S, S]`。308 条 33³ 约 33 MB；
* `.json`：`{"count", "size", "names": [...], "items": [...], "skipped": [...]}`。

**注意有损**：64³→33³ 实测平均偏差 3.2/255、p99 13/255。培训练目标够用，做精确复现不够。

解析失败 / 尺寸异常的 LUT **不会**让脚本退出，只会记进 manifest，
最后打一张汇总表 —— 毕竟第二大的风险（G10）就是 LUT 来源质量。
"""

from __future__ import annotations

import argparse
import os
import sys
from collections import Counter

import numpy as np
import torch

# 允许从仓库根直接 `python scripts/prepare_lut.py`
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from neuralpreset.data.lut_npy import LUTBank, parse_cube, resample  # noqa: E402


def scan_cubes(cube_dir: str) -> list:
    found = []
    for root, _dirs, files in os.walk(cube_dir):
        for f in files:
            if f.lower().endswith(".cube"):
                found.append(os.path.join(root, f))
    return sorted(found)


def identity_lut(size: int, dtype=torch.float32) -> torch.Tensor:
    t = torch.linspace(0.0, 1.0, size, dtype=dtype)
    return torch.stack(torch.meshgrid(t, t, t, indexing="ij"), dim=0)  # [3,S,S,S]


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="【可选】整理 .cube LUT → npy + manifest（仅 backend=npy 需要）")
    ap.add_argument("--cube-dir", required=True)
    ap.add_argument("--out-npy", default="data/lut33_u8.npy")
    ap.add_argument("--out-manifest", default=None, help="默认与 npy 同名 .json")
    ap.add_argument("--size", type=int, default=33)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--dry-run", action="store_true", help="只统计，不写文件")
    args = ap.parse_args(argv)

    manifest_path = args.out_manifest or os.path.splitext(args.out_npy)[0] + ".json"
    files = scan_cubes(args.cube_dir)
    if args.limit:
        files = files[: args.limit]
    if not files:
        print(f"[lut] {args.cube_dir} 下没有 .cube 文件")
        return 1
    print(f"[lut] 找到 {len(files)} 个 .cube，目标 {args.size}³")

    ident = identity_lut(args.size)
    ok_items, bad, ok_paths = [], [], []
    sizes = Counter()
    for idx, path in enumerate(files, 1):
        rel = os.path.relpath(path, args.cube_dir).replace("\\", "/")
        try:
            arr = parse_cube(path)
            if arr is None:
                bad.append({"name": os.path.basename(path), "path": rel, "ok": False,
                            "note": "解析失败（行数 ≠ N³、缺 LUT_3D_SIZE，或是 1D LUT）"})
                continue
            n = int(arr.shape[1])
            sizes[n] += 1
            t = resample(torch.from_numpy(arr), args.size)
            delta = float((t - ident).abs().mean())
            ok_paths.append(path)
            ok_items.append({
                "name": os.path.basename(path), "path": rel, "ok": True,
                "source_size": n,
                "mean_abs_delta_vs_identity": round(delta, 4),
                "note": "接近恒等，训练时帮助很小" if delta < 0.02 else "",
            })
        except Exception as exc:  # noqa: BLE001  单条坏文件不该中断批处理
            bad.append({"name": os.path.basename(path), "path": rel, "ok": False,
                        "note": f"{type(exc).__name__}: {exc}"})
        if idx % 50 == 0 or idx == len(files):
            print(f"[lut] {idx}/{len(files)}")

    print(f"[lut] 可用 {len(ok_items)} / 失败 {len(bad)}")
    print(f"[lut] 源尺寸分布: {dict(sizes)}")
    weak = [it for it in ok_items if it["mean_abs_delta_vs_identity"] < 0.02]
    if weak:
        print(f"[lut] ⚠ {len(weak)} 条接近恒等（Δ<0.02），例：{[w['name'] for w in weak[:3]]}")
    if bad:
        print(f"[lut] ⚠ 失败的例子：{[(b['name'], b['note'][:40]) for b in bad[:3]]}")

    if args.dry_run:
        print("[lut] --dry-run，不写文件")
        return 0 if ok_items else 1
    if not ok_items:
        print("[lut] 没有任何可用的 LUT，退出")
        return 1

    print("[lut] 构建 npy …")
    bank, stats = LUTBank.from_cubes(ok_paths, size=args.size)
    stats.update({"items": ok_items + bad, "source_sizes": dict(sizes)})
    bank.save(args.out_npy, manifest_path, stats=stats)
    print(f"[lut] 写出 {args.out_npy}（{len(bank)} 条，{bank.size}³，"
          f"{os.path.getsize(args.out_npy) / 1e6:.1f} MB）")
    print(f"[lut] 写出 {manifest_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
