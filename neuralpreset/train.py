"""训练入口。

    python -m neuralpreset.train --config configs/phase0.yaml
    python -m neuralpreset.train --config configs/phase0.yaml --smoke   # 300 张 / 2 epoch / CPU

阶段 0 的验收（REPRODUCTION_PLAN.md §10.1）都靠这个脚本产出：
A2 冒烟、A3 Kaggle 上的 L_rec 曲线、A6 --resume。
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import time
from typing import Optional, Tuple

import numpy as np
import torch
import torch.nn as nn

from .data.coco import CocoPatchDataset
from .data.lut import LUTBank
from .data.perturb import ColorPerturber, PerturbConfig
from .losses import NeuralPresetLoss
from .pipeline import build_model

__all__ = ["main"]


# ---------------------------------------------------------------------- #
# 配置
# ---------------------------------------------------------------------- #
def load_config(path: str) -> dict:
    try:
        import yaml
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("需要 PyYAML：Kaggle 镜像自带，本地 pip install pyyaml") from exc
    with open(path, "r", encoding="utf-8") as fh:
        cfg = yaml.safe_load(fh)
    if not isinstance(cfg, dict):
        raise ValueError(f"配置文件格式不对：{path}")
    return cfg


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def _build_perturber(cfg: dict, device) -> ColorPerturber:
    pcfg = PerturbConfig.from_dict(cfg.get("perturb"))
    lut_cfg = cfg.get("lut") or {}
    bank = None
    if lut_cfg.get("enable") and pcfg.lut_prob > 0:
        npy = lut_cfg.get("npy")
        if not npy or not os.path.isfile(npy):
            raise FileNotFoundError(
                f"lut.enable=true 但找不到 {npy!r}；先跑 scripts/prepare_lut.py 生成 lut33_u8.npy，"
                "或把 lut.enable 设为 false。"
            )
        bank = LUTBank.load(npy, lut_cfg.get("manifest"))
        print(f"[lut] 载入 {len(bank)} 条 LUT，{bank.size}³")
    elif pcfg.lut_prob > 0:
        print("[lut] 未启用 LUT，退回纯参数化扰动")
        pcfg.lut_prob = 0.0
    return ColorPerturber(pcfg, lut_bank=bank, seed=int(cfg.get("seed", 0)), device=device)


# ---------------------------------------------------------------------- #
# 健康检查（§4.4：出现即停的四个信号）
# ---------------------------------------------------------------------- #
class HealthMonitor:
    def __init__(self) -> None:
        self.warned = set()

    def update(self, out: dict, Ii: torch.Tensor, step: int) -> dict:
        with torch.no_grad():
            z_mean = out["Zi"].mean().item()
            i_mean = Ii.mean().item()
            ratio = z_mean / max(i_mean, 1e-6)
            sv_n = torch.linalg.svdvals(out["M_n"].float()).amax(dim=-1).mean().item()
            sv_s = torch.linalg.svdvals(out["M_s"].float()).amax(dim=-1).mean().item()
            stats = {
                "z_mean": z_mean,
                "i_mean": i_mean,
                "z_over_i": ratio,
                "sv_n_max": sv_n,
                "sv_s_max": sv_s,
                "y_mean": out["Yi"].mean().item(),
                "y_std": out["Yi"].std().item(),
            }
        if ratio < 0.5 and "scale" not in self.warned:
            self.warned.add("scale")
            print(f"[health][step {step}] ⚠ mean(Z)/mean(I)={ratio:.3f} < 0.5 "
                  "⇒ 尺度退化（§9.2），这条曲线不可用，请停下来换尺度锚定方案")
        if sv_n < 0.2 and "sv" not in self.warned:
            self.warned.add("sv")
            print(f"[health][step {step}] ⚠ M_n 最大奇异值 {sv_n:.3f} < 0.2 ⇒ 同上（矩阵在塌缩）")
        if not (0.2 < stats["y_mean"] < 0.8) and "y_mean" not in self.warned:
            self.warned.add("y_mean")
            print(f"[health][step {step}] ⚠ 输出均值 {stats['y_mean']:.3f} 偏离 0.5"
                  " ⇒ 初始化或激活问题（G1/G7）")
        return stats


# ---------------------------------------------------------------------- #
def train_one_epoch(model, loader, criterion, optimizer, scaler, cfg, epoch, device,
                    log_path: Optional[str], health: HealthMonitor,
                    global_step: int) -> Tuple[int, float]:
    model.train()
    log_every = int(cfg.get("log_every", 20))
    t0 = time.time()
    running = {"total": 0.0, "rec": 0.0, "con": 0.0, "n": 0}
    epoch_rec = 0.0
    n_seen = 0

    for it, batch in enumerate(loader):
        Ii = batch["ii"].to(device, non_blocking=True)
        Ij = batch["ij"].to(device, non_blocking=True)

        with torch.autocast(device_type="cuda" if device.type == "cuda" else "cpu",
                            dtype=torch.float16, enabled=bool(cfg.get("amp", True))):
            out = model.forward_train(Ii, Ij)
            loss_parts = criterion(out, Ii, Ij)
            loss = loss_parts["total"]

        optimizer.zero_grad(set_to_none=True)
        if scaler is not None:
            scaler.scale(loss).backward()
            if cfg.get("grad_clip"):
                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(model.parameters(), float(cfg["grad_clip"]))
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            if cfg.get("grad_clip"):
                nn.utils.clip_grad_norm_(model.parameters(), float(cfg["grad_clip"]))
            optimizer.step()

        global_step += 1
        for k in ("total", "rec", "con"):
            running[k] += float(loss_parts[k].detach())
        epoch_rec += float(loss_parts["rec"].detach())
        running["n"] += 1
        n_seen += 1

        if (it + 1) % log_every == 0:
            avg = {k: running[k] / running["n"] for k in ("total", "rec", "con")}
            stats = health.update(out, Ii, global_step)
            msg = (f"[ep {epoch} it {it + 1}/{len(loader)}] "
                   f"loss={avg['total']:.4f} rec={avg['rec']:.4f} con={avg['con']:.5f} "
                   f"λcon/rec={avg['con'] * cfg.get('lambda_con', 10.0) / max(avg['rec'], 1e-8):.3f} "
                   f"|Z|/|I|={stats['z_over_i']:.3f} sv_n={stats['sv_n_max']:.3f} "
                   f"sv_s={stats['sv_s_max']:.3f} y_mean={stats['y_mean']:.3f} "
                   f"| {running['n'] / max(time.time() - t0, 1e-6):.1f} it/s")
            print(msg)
            if log_path:
                with open(log_path, "a", encoding="utf-8") as fh:
                    fh.write(json.dumps({"epoch": epoch, "step": global_step,
                                         **avg, **stats, "it_per_s": running["n"] / max(time.time() - t0, 1e-6)},
                                        ensure_ascii=False) + "\n")
            running = {"total": 0.0, "rec": 0.0, "con": 0.0, "n": 0}
            t0 = time.time()

    mean_rec = epoch_rec / max(n_seen, 1)
    print(f"[ep {epoch}] epoch 平均 rec={mean_rec:.4f}（A3 判据：相对初值下降 ≥60% 且不发散）")
    return global_step, mean_rec


# ---------------------------------------------------------------------- #
def save_ckpt(path: str, model, optimizer, scheduler, epoch: int, cfg: dict, best: float) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
    torch.save({
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict() if scheduler is not None else None,
        "epoch": epoch,
        "best": best,
        "model_cfg": cfg.get("model", {}),
        "act": (cfg.get("model") or {}).get("act", "none"),
        "k": (cfg.get("model") or {}).get("k", 16),
    }, path)
    print(f"[ckpt] saved {path}")


def load_ckpt(path: str, model, optimizer=None, scheduler=None) -> int:
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    state = ckpt.get("model", ckpt)
    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing:
        print(f"[ckpt] missing keys: {len(missing)}（例如 {missing[0]}）")
    if unexpected:
        print(f"[ckpt] unexpected keys: {len(unexpected)}（例如 {unexpected[0]}）")
    if optimizer is not None and ckpt.get("optimizer"):
        optimizer.load_state_dict(ckpt["optimizer"])
    if scheduler is not None and ckpt.get("scheduler"):
        scheduler.load_state_dict(ckpt["scheduler"])
    start = int(ckpt.get("epoch", -1)) + 1
    print(f"[ckpt] resumed from {path} @ epoch {start}")
    return start


# ---------------------------------------------------------------------- #
def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Neural Preset 训练（阶段 0）")
    ap.add_argument("--config", required=True)
    ap.add_argument("--smoke", action="store_true", help="冒烟：300 张 / 2 epoch / 关 AMP")
    ap.add_argument("--resume", default=None, help="ckpt 路径；'auto' 取 out 目录里最新的")
    ap.add_argument("--out", default=None, help="覆盖配置里的输出目录")
    ap.add_argument("--device", default=None, help="例如 cpu / cuda:0")
    ap.add_argument("--limit", type=int, default=None, help="只用清单前 N 张（调试用）")
    ap.add_argument("--epochs", type=int, default=None)
    args = ap.parse_args(argv)

    cfg = load_config(args.config)
    tcfg = dict(cfg.get("train") or {})
    dcfg = dict(cfg.get("data") or {})
    mcfg = dict(cfg.get("model") or {})

    if args.epochs is not None:
        tcfg["epochs"] = args.epochs
    if args.smoke:
        tcfg.update({"epochs": 2, "batch_size": min(int(tcfg.get("batch_size", 24)), 8), "amp": False})
        dcfg.update({"num_workers": 0})
        args.limit = args.limit or 300

    out_dir = args.out or tcfg.get("out") or "out"
    os.makedirs(out_dir, exist_ok=True)
    log_path = os.path.join(out_dir, "metrics.jsonl")

    device = torch.device(args.device or cfg.get("device") or
                          ("cuda:0" if torch.cuda.is_available() else "cpu"))
    set_seed(int(cfg.get("seed", 0)))
    print(f"[env] torch={torch.__version__} device={device} out={out_dir}")

    # ---- 模型 ----
    model = build_model(mcfg).to(device)
    n_par = model.num_parameters()
    print(f"[model] k={model.k} act={model.act} params={n_par/1e6:.2f}M "
          f"(论文 5.15M，差异见 G6) weights={model.encoder.weight_source}")

    # ---- 数据 ----
    perturber = _build_perturber(cfg, device)
    dataset = CocoPatchDataset(
        root=dcfg["root"],
        list_file=dcfg["list"],
        size=int(dcfg.get("size", 256)),
        perturber=perturber,
        seed=int(cfg.get("seed", 0)),
        limit=args.limit,
    )
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=int(tcfg.get("batch_size", 24)),
        shuffle=True,
        num_workers=int(dcfg.get("num_workers", 4)),
        pin_memory=(device.type == "cuda"),
        drop_last=True,
        persistent_workers=int(dcfg.get("num_workers", 4)) > 0,
    )
    print(f"[data] {len(dataset)} 张 → {len(loader)} iters/epoch，bs={tcfg.get('batch_size')}")

    # ---- 优化 ----
    criterion = NeuralPresetLoss(lambda_con=float(tcfg.get("lambda_con", 10.0)))
    optimizer = torch.optim.Adam(model.parameters(), lr=float(tcfg.get("lr", 3e-4)))
    scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optimizer,
        milestones=[int(m) for m in (tcfg.get("milestones") or [6])],
        gamma=float(tcfg.get("gamma", 0.1)),
    )
    use_amp = bool(tcfg.get("amp", True)) and device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=True) if use_amp else None

    start_epoch = 0
    if args.resume:
        path = args.resume
        if path == "auto":
            cands = sorted(
                (os.path.join(out_dir, f) for f in os.listdir(out_dir) if f.endswith(".pt")),
                key=os.path.getmtime,
            )
            if not cands:
                raise FileNotFoundError(f"{out_dir} 里没有 .pt 可续训")
            path = cands[-1]
        start_epoch = load_ckpt(path, model, optimizer, scheduler)

    # ---- 训练 ----
    health = HealthMonitor()
    global_step = 0
    best = math.inf
    for epoch in range(start_epoch, int(tcfg.get("epochs", 8))):
        global_step, epoch_rec = train_one_epoch(model, loader, criterion, optimizer, scaler, tcfg,
                                                 epoch, device, log_path, health, global_step)
        scheduler.step()
        save_ckpt(os.path.join(out_dir, f"ckpt_epoch{epoch:02d}.pt"),
                  model, optimizer, scheduler, epoch, cfg, best)
        if epoch_rec < best:
            best = epoch_rec
            save_ckpt(os.path.join(out_dir, "best.pt"),
                      model, optimizer, scheduler, epoch, cfg, best)
            print(f"[best] ep {epoch}: rec={best:.4f} → best.pt")
        save_ckpt(os.path.join(out_dir, "last.pt"), model, optimizer, scheduler, epoch, cfg, best)

    print("[done]")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
