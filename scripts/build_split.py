"""【可选】从 COCO 原始目录里挑出固定的 10k 训练清单 + 200 张评测用内容图。

阶段 0 **不需要**它：`train.py` 在没有清单时会递归扫描目录、按相对路径排序、
取前 `data.limit` 张（同样的 dataset 每次顺序一致，会打印名单指纹）。
只有当你想要一份"可以冻结、可以跨机器核对"的显式清单时才跑这个脚本。

    # 只写清单
    python scripts/build_split.py --coco-dir C:\\...\\train2017 --n 10000 --holdout 200 \
        --out-list data/coco10k.txt --out-val data/val_content.txt

    # 顺便把选中的图拷成扁平目录（方便打 zip 传 Kaggle）
    python scripts/build_split.py ... --copy-to data/coco10k_images

为什么要"固定清单"（U1）：COCO 有 118k 张，每次随机抽 10k 会让实验之间不可比。
这里按**文件名排序**取前 N 张（确定性），并把清单的 sha256 写进表头。
留出的 200 张从排序后的**末尾**取，保证与训练集不重叠且同样确定。
"""

from __future__ import annotations

import argparse
import hashlib
import os
import shutil
import sys

from PIL import Image

_IMG_EXT = (".jpg", ".jpeg", ".png")


def scan(coco_dir: str) -> list:
    found = []
    for root, _dirs, files in os.walk(coco_dir):
        for f in files:
            if f.lower().endswith(_IMG_EXT):
                full = os.path.join(root, f)
                found.append(os.path.relpath(full, coco_dir).replace("\\", "/"))
    return sorted(found)


def check(path: str) -> bool:
    try:
        with Image.open(path) as im:
            im.verify()
        return True
    except Exception:  # noqa: BLE001
        return False


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="生成 COCO 10k 训练清单与 val 内容清单")
    ap.add_argument("--coco-dir", required=True)
    ap.add_argument("--n", type=int, default=10000)
    ap.add_argument("--holdout", type=int, default=200)
    ap.add_argument("--out-list", default="data/coco10k.txt")
    ap.add_argument("--out-val", default="data/val_content.txt")
    ap.add_argument("--copy-to", default=None, help="把选中图片拷到该目录（扁平命名）")
    ap.add_argument("--skip-check", action="store_true", help="跳过 PIL 校验（更快）")
    args = ap.parse_args(argv)

    names = scan(args.coco_dir)
    if len(names) < args.n + args.holdout:
        print(f"[split] ⚠ 只有 {len(names)} 张，少于 {args.n}+{args.holdout}")
    train = names[: args.n]
    val = names[-args.holdout:] if args.holdout else []
    overlap = set(train) & set(val)
    if overlap:
        print(f"[split] ⚠ 训练集与留出集重叠 {len(overlap)} 张（图片总数不足时会发生）")

    if not args.skip_check:
        for label, chunk in (("train", train), ("val", val)):
            bad = [n for n in chunk if not check(os.path.join(args.coco_dir, n))]
            if bad:
                print(f"[split] ⚠ {label} 里有 {len(bad)} 张打不开，例：{bad[:3]}")

    digest = hashlib.sha256("\n".join(train).encode("utf-8")).hexdigest()
    header = (
        f"# COCO 训练子集，按文件名排序取前 {len(train)} 张\n"
        f"# source: {os.path.abspath(args.coco_dir)}\n"
        f"# names_sha256: {digest}\n"
    )
    _write(args.out_list, header, train)
    print(f"[split] {args.out_list}: {len(train)} 行，sha256={digest[:16]}…")
    if val:
        _write(args.out_val, f"# 留出 {len(val)} 张（排序后的末尾），仅用于评测\n", val)
        print(f"[split] {args.out_val}: {len(val)} 行")

    if args.copy_to:
        os.makedirs(args.copy_to, exist_ok=True)
        for label, chunk in (("train", train), ("val", val)):
            target = os.path.join(args.copy_to, label)
            os.makedirs(target, exist_ok=True)
            for idx, name in enumerate(chunk, 1):
                shutil.copy2(os.path.join(args.coco_dir, name),
                             os.path.join(target, os.path.basename(name)))
                if idx % 1000 == 0:
                    print(f"[split] copy {label} {idx}/{len(chunk)}")
        print(f"[split] 已拷贝到 {args.copy_to}")
    return 0


def _write(path: str, header: str, names: list) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        fh.write(header)
        fh.write("\n".join(names) + "\n")


if __name__ == "__main__":
    sys.exit(main())
