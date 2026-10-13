"""OpenFocus A/B —— 基于 OpenFocus 分析复现验证的两条融合管线。

来源与验证（详见 docs/OpenFocus_REPRODUCTION_REPORT.md）：
- 清晰度度量 = 窗口局部方差（与 OpenFocus 深度图逐像素重合 99.5%，
  真实数据 11 栈输出 MAE 0.95%–4.1% 值域）。
- Method B（深度图选择）：argmax(var) → 深度图平滑 → 按图硬拾取；
  硬拾取优于软混合（实验证明），平局取首帧。
- Method A（加权平均）：w = gauss(var)，分母带 +1 的末帧先验
  （平坦区权重全零时精确退化为末帧——复现实验误差 0.00）。
- Method C（金字塔）：7 层拉普拉斯金字塔逐带能量软混合，
  最粗带（基带）整带取末帧——实测的 OpenFocus 行为
  （"低频来自参考帧 + 高频按频段独立决策"，输出高频能量 ~1.24×输入）。

约定：输入为 0-255 域的 BGR 帧列表（uint8 或 float32），
输出经 fuse_output_dtype 归一（与 gff/dct 相同的位深契约）。
"""
import os
from typing import List, Optional, Tuple

import cv2
import numpy as np

from utils.image_utils import fuse_output_dtype


def _collect_images(input_source) -> List[np.ndarray]:
    """接受帧列表或目录路径，返回 0-255 域 BGR 帧列表。"""
    if isinstance(input_source, (list, tuple)):
        frames = [f for f in input_source if f is not None]
        if len(frames) < 2:
            raise ValueError("OpenFocus A/B fusion needs at least 2 frames")
        return frames

    if isinstance(input_source, str) and os.path.isdir(input_source):
        exts = ('.png', '.jpg', '.jpeg', '.tif', '.tiff', '.bmp')
        files = [f for f in sorted(os.listdir(input_source))
                 if f.lower().endswith(exts)]
        if len(files) < 2:
            raise ValueError(
                f"OpenFocus A/B fusion needs at least 2 images in {input_source}")
        frames = []
        for name in files:
            data = np.fromfile(os.path.join(input_source, name), dtype=np.uint8)
            img = cv2.imdecode(data, cv2.IMREAD_ANYDEPTH | cv2.IMREAD_COLOR)
            if img is None:
                raise RuntimeError(f"Unable to read image: {name}")
            if img.dtype == np.uint16:
                img = (img.astype(np.float32) * (255.0 / 65535.0))
            frames.append(img)
        return frames

    raise ValueError("input_source must be a list of frames or a directory")


def _as_float(frames) -> List[np.ndarray]:
    out = []
    for f in frames:
        arr = f if f.dtype == np.float32 else f.astype(np.float32)
        out.append(arr)
    return out


def _luminance(img: np.ndarray) -> np.ndarray:
    if img.ndim == 2:
        return img
    take = min(img.shape[2], 3)
    if take == 1:
        return img[:, :, 0]
    weights = np.array([0.114, 0.587, 0.299], dtype=np.float32)[:take]
    out = cv2.transform(img[:, :, :take], weights.reshape(1, -1))
    return out if out.ndim == 2 else out[:, :, 0]


def _window_variance(gray: np.ndarray, radius: int) -> np.ndarray:
    """窗口局部方差：var = blur(x²) − blur(x)²，box 滤波可分离、O(N)。"""
    k = max(1, int(radius)) * 2 + 1
    g = gray
    mean = cv2.blur(g, (k, k))
    mean_sq = cv2.blur(g * g, (k, k))
    return np.maximum(mean_sq - mean * mean, 0.0)


def _variance_stack(frames_f: List[np.ndarray], radius: int) -> np.ndarray:
    """逐帧方差堆栈 (N, H, W) float32。"""
    return np.stack([
        _window_variance(_luminance(f), radius) for f in frames_f
    ], axis=0)


def openfocus_a_impl(input_source, img_resize=None, kernel_size: int = 7,
                     thread_count: int = None):
    """OpenFocus - A（方差加权平均，末帧先验）。

    w_i = gauss(var_i, σ=2) ；out = (Σ w_i·I_i + I_last) / (Σ w_i + 1)。
    +1 的末帧先验保证平坦区（权重全零）精确退化为末帧——这正是
    实验中 Method A 平局行为（输出恒等于末帧）的成因。
    """
    frames = _collect_images(input_source)
    if img_resize is not None:
        w, h = img_resize
        frames = [cv2.resize(f, (int(w), int(h)),
                             interpolation=cv2.INTER_AREA) for f in frames]
    frames_f = _as_float(frames)

    radius = max(1, int(kernel_size) // 2) if kernel_size else 3
    var = _variance_stack(frames_f, radius)
    # Smoothing：对权重图做轻度高斯（对应 OpenFocus 的 Smoothing 参数）
    weights = np.stack([
        cv2.GaussianBlur(v, (0, 0), sigmaX=2.0) for v in var
    ], axis=0)

    h, w = frames_f[0].shape[:2]
    channels = frames_f[0].shape[2] if frames_f[0].ndim == 3 else 1
    acc = np.zeros((h, w, channels), dtype=np.float32)
    wsum = np.zeros((h, w, 1), dtype=np.float32)
    for i, frame in enumerate(frames_f):
        w2d = weights[i][:, :, np.newaxis]
        acc += frame * w2d
        wsum += w2d
    # 末帧先验（+1）：平坦区权重全零时输出精确等于末帧
    acc += frames_f[-1]
    wsum += 1.0
    fused = acc / wsum
    return np.clip(fused, 0, 255).astype(fuse_output_dtype(frames))


def openfocus_b_impl(input_source, img_resize=None, kernel_size: int = 7,
                     thread_count: int = None):
    """OpenFocus - B（方差深度图选择）。

    两遍结构：argmax(var) → 深度图平滑（边界 ±3 帧线性过渡）→ 硬拾取。
    np.argmax 天然平局取首帧（实验确认的行为）。
    """
    frames = _collect_images(input_source)
    if img_resize is not None:
        w, h = img_resize
        frames = [cv2.resize(f, (int(w), int(h)),
                             interpolation=cv2.INTER_AREA) for f in frames]
    frames_f = _as_float(frames)

    radius = max(1, int(kernel_size) // 2) if kernel_size else 3
    var = _variance_stack(frames_f, radius)
    n = len(frames_f)

    depth = np.argmax(var, axis=0).astype(np.float32)
    # 深度图平滑：先中值去噪点，再高斯做边界过渡（约 ±3 帧），
    # 最后取整——硬拾取（软混合经实验证明反而偏离 OpenFocus 输出）。
    depth_u8 = np.clip(depth, 0, 255).astype(np.uint8)
    depth_u8 = cv2.medianBlur(depth_u8, 3)
    depth = depth_u8.astype(np.float32)
    depth = cv2.GaussianBlur(depth, (0, 0), sigmaX=1.2)
    idx = np.clip(np.rint(depth), 0, n - 1).astype(np.int32)

    h, w = frames_f[0].shape[:2]
    # 按深度图逐像素硬拾取（含彩色/灰度）
    stack = np.stack(
        [f if f.ndim == 3 else f[:, :, np.newaxis] for f in frames_f], axis=0)
    fused = np.take_along_axis(stack, idx[np.newaxis, :, :, np.newaxis], axis=0)[0]
    if frames_f[0].ndim == 2:
        fused = fused[:, :, 0]
    return np.clip(fused, 0, 255).astype(fuse_output_dtype(frames))


# ------------------------------------------------------------------
# OpenFocus - C（Method C：拉普拉斯金字塔，移植自分析复现管线）
# ------------------------------------------------------------------

def _pyramid_gauss(img: np.ndarray, levels: int) -> List[np.ndarray]:
    """高斯金字塔：σ=1 模糊 + 2× 抽点（与复现管线一致的 Burt 型分解）。"""
    g = [img.astype(np.float32)]
    for _ in range(levels - 1):
        b = cv2.GaussianBlur(g[-1], (0, 0), 1.0)
        g.append(b[::2, ::2])
    return g


def _pyramid_expand(band: np.ndarray, shape) -> np.ndarray:
    """2× 上采样（最近邻重复 + σ=1 模糊），裁到目标形状。

    band 可以是 (h, w) 或 (h, w, C)；cv2 对 (h,w) 2D 直接可用。
    """
    up = np.repeat(np.repeat(band, 2, axis=0), 2, axis=1)
    up = up[:shape[0], :shape[1]]
    return cv2.GaussianBlur(up, (0, 0), 1.0)


def openfocus_c_impl(input_source, img_resize=None, kernel_size: int = 7,
                     thread_count: int = None, levels: int = 7):
    """OpenFocus - C：拉普拉斯金字塔融合（复现 OpenFocus Method C）。

    每个带通层的权重 = |亮度带| 的高斯平滑（wsigma=1），
    软归一化混合 (Σ W·band)/(Σ W)，能量全零处回退末帧；
    最粗的基带整带取末帧（亮度基准）。
    层数自适应：min(levels, log2(min(H,W))-1)，小图不产生退化带。
    """
    frames = _collect_images(input_source)
    if img_resize is not None:
        w, h = img_resize
        frames = [cv2.resize(f, (int(w), int(h)),
                             interpolation=cv2.INTER_AREA) for f in frames]
    frames_f = _as_float(frames)

    h, w = frames_f[0].shape[:2]
    # 小图保护：每层减半，至少留 2px；levels 至少 2（一层带通 + 基带）
    usable = max(2, min(levels, int(np.log2(max(2, min(h, w)))) - 1))
    pyrs = [_pyramid_gauss(f, usable) for f in frames_f]
    n = len(frames_f)

    out_bands = []
    for k in range(usable):
        # band: (N, hk, wk[, C]) —— 各帧第 k 层的带通/基带
        if k < usable - 1:
            bands = []
            for p in pyrs:
                up = _pyramid_expand(p[k + 1], p[k].shape[:2])
                bands.append(p[k] - up)
        else:
            bands = [p[k] for p in pyrs]
        band = np.stack(bands, axis=0).astype(np.float32)

        if k == usable - 1:
            # 基带：整带取末帧（OpenFocus 实证行为：低频来自参考帧）
            out_bands.append(band[-1].copy())
            continue

        # 权重：亮度域带通幅值，σ=1 平滑（对应 OpenFocus 的 opacity filter）。
        # 注意：cv2.GaussianBlur 会把 (N,h,w) 当成多通道图跨 N 轴混合，
        # 必须逐帧模糊。
        if band.ndim == 4:
            lum = (band * np.array([0.114, 0.587, 0.299],
                                   np.float32)).sum(axis=3)
        else:
            lum = band
        W = np.stack([cv2.GaussianBlur(np.abs(w), (0, 0), 1.0)
                      for w in lum], axis=0)
        tot = W.sum(axis=0)
        # 软归一化混合；能量全零处回退末帧
        with np.errstate(invalid="ignore", divide="ignore"):
            if band.ndim == 4:
                mixed = (W[:, :, :, np.newaxis] * band).sum(axis=0) / \
                    np.maximum(tot, 1e-9)[:, :, np.newaxis]
            else:
                mixed = (W * band).sum(axis=0) / np.maximum(tot, 1e-9)
        flat = tot <= 1e-9
        if flat.any():
            mixed[flat] = band[-1][flat]
        out_bands.append(mixed.astype(np.float32))

    # 自顶向下重建：expand + 累加
    img = out_bands[-1]
    for k in range(usable - 2, -1, -1):
        img = _pyramid_expand(img, out_bands[k].shape[:2]) + out_bands[k]

    return np.clip(img, 0, 255).astype(fuse_output_dtype(frames))
