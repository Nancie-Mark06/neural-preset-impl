# third_party/metric

论文官方指标代码与权重，来自论文作者仓库
[`C:\ncx\workspace\NeuralPreset\src\metric`](../../../NeuralPreset/src/metric)（论文 arXiv:2303.13511v2 附带资源）。

| 文件 | 大小 | 用途 |
|---|---|---|
| `style_similiary/StyleSimiliaryDiscriminator.onnx` | 44,649,317 B | 风格相似度判别器（ONNX，CPU 可跑） |
| `style_similiary/calc_style_similiary.py` | 2,417 B | 原 CLI（一条命令扫整个文件夹） |
| `content_similiary/ldc.pth` | 2,767,911 B | LDC 边缘检测权重 |
| `content_similiary/ldc.py` | 9,571 B | LDC 网络定义 + `postprocess_edges` |
| `content_similiary/calc_content_similiary.py` | 3,207 B | 原 CLI |

**原始文件一字未改。** 我们的改动全部集中在 `api.py`（见下），这样"作者的数"和"我们的数"
可以对照着看，出问题也容易定位是哪一边的。

## 必须改的两处（否则在原作者的脚本上跑不起来）

1. **`multichannel=True` 已从 skimage 移除**（0.23 起，等价写法 `channel_axis=2`）。
   我们先把边缘图 `squeeze` 成 `[H, W]` 二维，再用默认的 `channel_axis=None`，语义等价。
2. **写死的相对路径**：`'./StyleSimiliaryDiscriminator.onnx'`、`'./ldc.pth'`
   只能"cd 到该目录再跑"。`api.py` 用相对 `__file__` 的绝对路径，并允许参数覆盖。

## 一个**不是** bug 的地方（别改）

`calc_content_similiary.py:24-29`：

```python
h, w = image.size          # PIL 的 size 是 (W, H)，所以 h=宽、w=高
h = int(h - h % 32); w = int(w - w % 32)
image = transforms.functional.resize(image, (w, h))   # 传进去的是 (H', W')，正是 torchvision 要的顺序
```

命名确实误导（`h` 里装的是宽），但**计算是对的**：两次交换互相抵消，纵横比不变。
把它"修"成 `w, h = image.size` 反而会引入真正的宽高错位。

## 复现口径

* 风格相似度：`result` 与 `style` 都缩放到 512×512（`Image.resize` 的默认插值即 BICUBIC ——
  作者没写 resample 参数，我们照抄），转 float32、值域 0–255；
  ONNX 输入名 `ref`（风格）/ `img`（结果），输出名 `score`。越大越像。
* 内容相似度：先把图裁到 32 的倍数（**裁右下角**，不是缩放），减 ImageNet BGR 均值
  `[103.939, 116.779, 123.68]`（注意：代码里直接对 RGB 张量减这组值，是作者原样，
  我们照抄），过 LDC 得边缘图，再算 SSIM。越大越保内容。
* 两个原 CLI 都 `assert len(files_a) == len(files_b)` 并按**文件名**配对 ⇒
  `neuralpreset/evaluate.py` 出图时沿用风格图/内容图的原文件名。

## 依赖

`onnxruntime`（风格）、`torch` + `scikit-image` + `opencv-python`（内容，`ldc.py` 顶部
`import cv2`）。Kaggle 镜像都自带；本机没有也不影响写代码，只是跑不了。
