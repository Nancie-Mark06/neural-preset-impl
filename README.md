# Neural Preset 复现（阶段 0）

论文：**Neural Preset for Color Style Transfer**（CVPR 2023，[arXiv:2303.13511v2](https://arxiv.org/abs/2303.13511)）的
可运行复现。规划、缺口表、验收标准在 [`REPRODUCTION_PLAN.md`](REPRODUCTION_PLAN.md)；
本文件只说**代码怎么用**。

> **状态：第一版代码，尚未运行过。**
> 写它的机器没有 torch、没有 CUDA（Python 3.14.6），所以所有"跑起来"的证据都还没有 ——
> 单元测试与冒烟验收都标了"未验证"。第一次上 Kaggle 请先按下面的顺序跑，
> 预计会有一两处路径/形状要修。

## 这东西在做什么

一张内容图 `I_c`、一张风格图 `I_s`：

```
d, r = Encoder(I_s, I_c)          # 共享 EfficientNet-B0 主干 → GAP(1280) → 两个 Linear(1280,256)
M_n  = P · T(d) · Q               # normalize：3×3
M_s  = P · T(r) · Q               # stylize  ：3×3
Y    = I_c · M_n · M_s            # 逐像素、与分辨率无关
```

关键点：`M` 是**纯 3×3 全局线性映射**（论文命题 1），所以推理时可以（a）折叠成一个矩阵，
（b）在 4K 图/视频上逐帧套用而几乎不花算力。DNCM 的中间维 `k=16` 只影响表达能力。

训练不需要任何成对标注：拿 COCO 一张干净图 `I`，随机做两次颜色扰动得 `I_i, I_j`，
要求"j 的内容 + i 的风格"重建出 `I_i`（**style code 必须交换**，否则模型会退化成复制风格图）：

```
L = ‖Y_i − I_i‖₁ + ‖Y_j − I_j‖₁ + 10 · ‖Z_i − Z_j‖²
```

扰动来源：参数化算子（亮度/对比度/饱和度/色相/伽马/色温/随机 3×3）+ 随机 3D LUT。
LUT 只出现在 dataloader 里 —— **模型从不预测 LUT**，它学的是"用 3×3 去逼近 LUT 造成的颜色变化"。

## 目录

```
neuralpreset/
  dncm.py        3×3 映射：P·T·Q，含 act 开关（G1）
  encoder.py     共享主干 + 双头，预训练权重三级回退
  pipeline.py    两阶段前向 / 分块渲染 / preset 复用
  losses.py      L_rec + λ·L_con，值域断言
  data/
    lut.py       【默认】pillow-lut 后端：.cube 全量入内存，原生尺寸、零重采样
    lut_npy.py   【可选】.cube 离线压成 uint8 npy（几千条 LUT 时才用）
    perturb.py   颜色扰动算子集（PIL 套 LUT → 张量做参数化算子）
    coco.py      COCO patch 数据集（一图两视图；无清单时递归扫描按名字取前 N 张）
  train.py       训练入口（--smoke / --resume）
  evaluate.py    批量出图 + 论文两个指标
  infer.py       单图 / 整目录推理
configs/phase0.yaml   阶段 0 唯一配置
scripts/
  prepare_lut.py      【可选】.cube 目录 → lut33_u8.npy + manifest（仅 npy 后端）
  build_split.py      【可选】COCO → 固定 10k 清单 + 200 张评测留出（想要冻结清单时才跑）
third_party/metric/   论文官方指标（ONNX + LDC 权重，原文件未改，见其 README）
tests/                20 条单测（数学性质 + 数据侧），不需要数据集
notebooks/kaggle_train.ipynb   7 个 cell 的 Kaggle 跑法
```

## 快速开始

### Kaggle（目标环境）

1. 把本仓库打成 `code.zip`（不含 `data/`），上传为私有 dataset；图片和 `.cube` 各传一个 dataset（目录层级随意）；
2. 新建 Notebook，`Accelerator: GPU T4 x2`，`Internet: On`（同时用来 `pip install pillow-lut`、下 21 MB 权重）；
3. 按 `notebooks/kaggle_train.ipynb` 从上到下跑：解压 + **自动探测输入目录** → 装 pillow-lut + 核对数据量 → **冒烟** → 全量 → 出图/指标 → 归档。

**不需要清单文件**：图片按相对路径排序后取前 `data.limit`（默认 10000）张；同一份 dataset 每次
顺序完全一样，`train.py` 会打印名单指纹（sha1 前 12 位）方便核对。训练/评测用的路径都能用命令行覆盖：

```bash
python -m neuralpreset.train --config configs/phase0.yaml \
    --data-root /kaggle/input/<图片dataset> --data-list none \
    --cube-dir /kaggle/input/<lut dataset> --out /kaggle/working/out
```

LUT 默认**不需要**预打包：`pillow` 后端在启动时把 308 个 `.cube` 读进内存（约 409 MB，
Kaggle 16 GB 内存无压力；Linux `fork` 下多个 worker 共享同一份只读内存，不会乘以 worker 数）。
只有 LUT 涨到几千条（原生尺寸 ≈15 GB）才切 `lut.backend: npy`，并先跑 `scripts/prepare_lut.py`。

冒烟是刻意排在前面的：300 张、2 epoch、`batch_size≤8`，**CPU 上一分钟**也能跑完，
它会把"路径写错、清单错、dtype 错、shape 错"这几类问题一次性暴露出来，不用等 40 分钟的训练。

### 本地（没有 GPU 也能跑的部分）

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
pip install pyyaml pillow pillow-lut numpy onnxruntime scikit-image opencv-python pytest

python -m pytest tests -q                      # 20 条单测，不需要数据（缺 pillow-lut 时跳过 4 条）
python -m neuralpreset.train --config configs/phase0.yaml --smoke --limit 300   # CPU 冒烟
```

配置里的路径默认按 Kaggle 写；本地跑冒烟时改 `data.root`，或加 `--data-root <目录> --data-list none --out <输出目录>`。

## 常用命令

```bash
# 训练（中断后用 --resume auto 接上）
python -m neuralpreset.train --config configs/phase0.yaml --out out

# 单图推理（--matrix 打印折叠出的 3×3，--compare 另存三宫格）
python -m neuralpreset.infer --ckpt out/last.pt --content a.jpg --style b.jpg --out y.png --matrix --compare

# 一张风格图刷一整个目录（preset 只算一次）
python -m neuralpreset.infer --ckpt out/last.pt --content-dir photos/ --style b.jpg --out-dir out/b/

# 评测：2500 对，出论文的两个指标 + 恒等基线
python -m neuralpreset.evaluate --ckpt out/last.pt \
    --content-dir data/val/content --style-dir data/val/style --n-pairs 2500 --out out/eval

# 数据准备（在有 COCO 原始文件的机器上）
python scripts/build_split.py --coco-dir /path/train2017 --n 10000 --holdout 200
# 可选：只有切到 lut.backend=npy 时才需要（有损：64³→33³ 平均偏 3.2/255）
python scripts/prepare_lut.py --cube-dir /path/cubes --out-npy data/lut33_u8.npy
```

## 日志里该看什么

`out/metrics.jsonl` 每 20 步一行。阶段 0 只判三件事（§4.4 的三个停机信号）：

| 字段 | 正常 | 异常意味着 |
|---|---|---|
| `rec` | 从 ~0.15 降到 <0.04 | 降不下去 ⇒ 容量/学习率/数据问题 |
| `z_over_i` | ≈1 | <0.5 ⇒ **尺度退化（G16）**，这条曲线不能用 |
| `sv_n_max` / `sv_s_max` | 0.5–2 | <0.2 ⇒ 矩阵在塌缩，同上 |
| `y_mean` | ≈0.5 | 越出 0.2–0.8 ⇒ 初始化或 act 有问题（G7/G1） |

`z_over_i` 和 `sv_*` 是这次复现里最值得盯的两个量：理论分析（§9.2）表明
`L_rec` 沿 `c→0` 单调下降，所以**模型有动机把整张图往 0 拉**。若出现，说明
G16 必须在阶段 1 解决（尺度锚定），而不是忽略。

## 与规划文档的偏离（有意为之）

1. **包放在仓库根 `neuralpreset/`，不是 `src/neuralpreset/`** —— 在仓库根直接
   `python -m neuralpreset.train` 就能用，省掉一层 `PYTHONPATH`。
   前提是**当前目录就是仓库根**：仓库里没有 `pyproject.toml`，包没被安装过，
   只有 cwd / `PYTHONPATH` 指到仓库根时才 import 得到。Kaggle notebook 把代码解压到
   `/kaggle/working/code`、而内核 cwd 是 `/kaggle/working`，所以 cell 1 会把 `CODE_DIR`
   写进 `PYTHONPATH`（`sys.path.insert` 只对内核生效，`!python -m ...` 是另一个进程）。
2. **指标修复集中在 `third_party/metric/api.py`**，作者原始文件一字未改（便于溯源对照）。
   规划 §5.1 说作者把宽高写反 —— 复核后**这是误判**，`h, w = image.size` 与
   `resize((w, h))` 互相抵消，纵横比正确；真正要改的只有 `multichannel`→`channel_axis` 和写死的路径。
3. **扰动里多了一个 `gain`（色温）算子**：作者只提"随机滤波"，我们把它具体化成
   R↑B↓ 的反相增益，比三通道同缩放更有区分度。
4. **头初始化的 weight 按 `1/√1280` 缩放**：GAP 特征 1280 维累加会把 N(0,0.005)
   放大到 ≈0.18（肉眼可见偏色），缩放后初始映射才真正接近恒等。
5. **LUT 默认走 `pillow-lut` 全量入内存，不用离线 npy**（原计划是打包 33³ uint8）：
   只有 308 条 LUT，409 MB 内存换"原生尺寸、零重采样误差"是划算的，代码还少了约 250 行；
   npy 路径保留在 `lut_npy.py` + `scripts/prepare_lut.py`，留给 LUT 数量涨上去的那天。
6. **`pillow-lut` 是新增依赖**（纯 Python wheel，Kaggle 需 `pip install`）。
   它只用于把 `.cube` 应用到 PIL 图上，模型与前向完全不依赖它。

## 还没验证 / 未解决

- [ ] `pytest tests` 全绿（本机无 torch；`pillow-lut` 那几条也没跑过）
- [ ] `pillow-lut` 在 Kaggle 上的 `pip install` 与 308 条 `.cube` 的载入耗时（预期 5–15 s）
- [ ] CPU 冒烟跑通、Kaggle T4 上 32 epoch 的 `L_rec` 曲线（A2/A3）
- [ ] 指标能出数（A5）：风格 ONNX 在本机单独测过 test_data（0.8692），但没有在 pipeline 出图上跑过
- [ ] `--resume` 续训（A6）
- [ ] **G16 尺度退化**：目前只监控、不修（规划里明确"阶段 0 假装不存在"）
- [ ] LUT 来源质量（G10）：308 条够不够；少数接近恒等的 LUT 对 L_con 没什么贡献
