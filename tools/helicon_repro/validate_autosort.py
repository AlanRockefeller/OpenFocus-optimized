"""帧序方向研究（2026-10-13，v1.42 开发过程中的决定性实验）。

问题起源：早前会话据 dmap 标签差 n-1 推断 "Helicon 自动反转帧序"。
本实验 + 输出侧证据给出最终结论：**Helicon 从未重排帧序**——

  ① dmap 极性列（本脚本）：11/11 栈 dA(rev) ≪ dA(file) → 若按
     "值=内部索引" 解码则全部被"反转"；但——
  ② 输出侧（MultiFocusFusion sort 对比）：asc（base=文件末帧）在
     keyboard/balls/kitchen 上 MAE 1.18/1.73/3.48%，desc（base=文件
     首帧）恶化到 2.40/3.33/3.58% → Helicon 输出亮度基底=文件末帧；
  ③ 合成栈（stackC，清晰度递减）：Helicon B 输出 DC=文件末帧 DC
     （若按"递减即反转"应为首帧）→ 无反转。

①②③ 同时成立的唯一解释：dmap 导出极性是 "65535 = 首帧" 的反向
编码（早前解码方向弄反了），不存在任何重排。

工程结论：内容式方向自动检测不可能（内容是反转变换协变的；本脚本
的 rho 列即证据：-0.98…+1.00 随机散布）。OpenFocus 的帧序方向因此
是显式选项：asc（默认=文件序=Helicon 行为）/ desc / off。
"""
import sys, os, glob
sys.path.insert(0, r'F:\Working\OpenFocus\tools\helicon_repro')
sys.path.insert(0, r'F:\Working\OpenFocus')
import numpy as np
import cv2
from PIL import Image
from fusion_methods.openfocus_ab import frame_sharpness_curve, detect_stack_flip

DATA = r'E:\ImagefusionDatasets\StackMFF_jpg\Mobile Depth\image stack'
RES = r'F:\Working\OpenFocus\tools\helicon_repro\results'

def box_blur(a, r):
    k = 2 * r + 1
    p = np.pad(a, ((r, r), (0, 0)), mode='edge'); c = np.cumsum(p, 0)
    a = (c[k - 1:] - np.concatenate([np.zeros((1,) + a.shape[1:]), c[:-k]])) / k
    p = np.pad(a, ((0, 0), (r, r)), mode='edge'); c = np.cumsum(p, 1)
    return (c[:, k - 1:] - np.concatenate([np.zeros((a.shape[0], 1)), c[:, :-k]], 1)) / k

print('%-11s %4s %7s %8s %10s %10s %8s' % (
    'stack', 'n', 'rho', 'flip?', 'dA(file)', 'dA(rev)', 'dmap?'))
agree = disagree = nodata = 0
for stack in sorted(os.listdir(DATA)):
    files = sorted(glob.glob(os.path.join(DATA, stack, '*.jpg')))
    if len(files) < 2:
        continue
    frames = [cv2.resize(np.asarray(Image.open(p).convert('RGB'), dtype=np.float32),
                         (320, 240)) for p in files]
    n = len(files)
    curve = frame_sharpness_curve(frames)
    flip = detect_stack_flip(curve)
    # 秩相关
    c = curve; ranks = np.argsort(np.argsort(c)).astype(np.float64)
    idx = np.arange(n, dtype=np.float64); idx -= idx.mean(); ranks -= ranks.mean()
    rho = float((idx * ranks).sum() / np.sqrt((idx * idx).sum() * (ranks * ranks).sum()))
    # dmap 证据
    dmp = os.path.join(RES, 'dmap_real_%s_r3_s3.tif' % stack)
    if not os.path.exists(dmp):
        print('%-11s %4d %7.2f %8s  (no dmap)' % (stack, n, rho, flip))
        nodata += 1
        continue
    dm = np.asarray(Image.open(dmp), dtype=np.float64)
    if dm.ndim == 3:
        dm = dm[..., 0]
    step = 65535.0 / (n - 1)
    heli = dm / step
    # 我的 argmax 标签（文件序，逐通道方差和，与验证管线同口径）
    E = []
    for f in frames:
        e = np.zeros(f.shape[:2], np.float64)
        for ch in range(3):
            m = box_blur(f[..., ch], 3)
            e += box_blur((f[..., ch] - m) ** 2, 3)
        E.append(e)
    j = np.stack(E).argmax(0).astype(np.float64)
    hh, ww = min(j.shape[0], heli.shape[0]), min(j.shape[1], heli.shape[1])
    j = j[(j.shape[0]-hh)//2:(j.shape[0]-hh)//2+hh, (j.shape[1]-ww)//2:(j.shape[1]-ww)//2+ww]
    heli_c = heli[(heli.shape[0]-hh)//2:(heli.shape[0]-hh)//2+hh,
                  (heli.shape[1]-ww)//2:(heli.shape[1]-ww)//2+ww]
    m = (hh > 8) & (ww > 8)
    j, hc = j[4:-4, 4:-4], heli_c[4:-4, 4:-4]
    d_file = float(np.abs(j - hc).mean())
    d_rev = float(np.abs((n - 1) - j - hc).mean())
    truth_flip = d_rev < d_file
    tag = 'AGREE' if truth_flip == flip else 'DISAGREE'
    if truth_flip == flip:
        agree += 1
    else:
        disagree += 1
    print('%-11s %4d %7.2f %8s %10.2f %10.2f  %s (truth flip=%s)' % (
        stack, n, rho, flip, d_file, d_rev, tag, truth_flip))
print('\nagree=%d disagree=%d nodata=%d' % (agree, disagree, nodata))
