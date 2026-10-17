"""审计：v1.39 移植版 (fusion_methods/openfocus_ab.py) vs 最终验证配方
(reproduce_and_compare.py 的 "FINAL chosen configuration" / real_compare.py)。

代码评审发现的候选偏差，本脚本逐条隔离量化：
  B: ① luma 方差 vs 逐通道方差求和  ② median3+gauss1.2 vs box1+gauss1.0
  A: ③ 缺第二层 box(radius) 窗口  ④ 权重高斯 σ2 vs σ3
  C: ⑤ softnorm vs max-pick(实测胜出)  ⑥ 权重 σ1 vs σ3
     ⑦ expand: np.repeat+σ1 vs bicubic×2+σ0.6(最终验证版)
     ⑧ 验证版的 W 模糊意外混了帧轴——验证 argmax 下是否可净化为逐帧 2D

用法: python audit_port_v139.py   (全部基准在 F:\\Working\\helicon-rea\\tests)
"""
import sys, os, glob
sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, r'F:\Working\OpenFocus')
import numpy as np
import cv2
from PIL import Image
from helicon_repro import gauss_blur, estimate_translation, apply_shift
from real_compare import load_rgb, metrics, DATA, OUT

ROOT = r'F:\Working\helicon-rea\tests'
from fusion_methods.openfocus_ab import (
    openfocus_a_impl, openfocus_b_impl, openfocus_c_impl)


# ---------------------------------------------------------------- helpers
def cv_base_correct(img, last, sigma=32.0):
    a = np.ascontiguousarray(img, np.float32)
    b = np.ascontiguousarray(last, np.float32)
    return a + cv2.GaussianBlur(b, (0, 0), sigma) - cv2.GaussianBlur(a, (0, 0), sigma)


def synth_metrics(mine, ref, m=6):
    hh = min(mine.shape[0], ref.shape[0]); ww = min(mine.shape[1], ref.shape[1])
    a = mine[(mine.shape[0] - hh) // 2:(mine.shape[0] - hh) // 2 + hh,
             (mine.shape[1] - ww) // 2:(mine.shape[1] - ww) // 2 + ww]
    b = ref[(ref.shape[0] - hh) // 2:(ref.shape[0] - hh) // 2 + hh,
            (ref.shape[1] - ww) // 2:(ref.shape[1] - ww) // 2 + ww]
    a, b = a[m:-m, m:-m], b[m:-m, m:-m]
    e = float(np.abs(a - b).mean()); rng = float(b.max())
    return e, e / max(rng, 1) * 100


def align_to_first(stack, margin=1):
    ref = stack[0]
    out = [ref]; shifts = [(0, 0)]
    for f in stack[1:]:
        dy, dx = estimate_translation(ref, f)
        out.append(apply_shift(f, dy, dx)); shifts.append((dy, dx))
    ys = max(abs(s[0]) for s in shifts) + margin
    xs = max(abs(s[1]) for s in shifts) + margin
    h, w = ref.shape
    sl = (slice(ys, h - ys), slice(xs, w - xs))
    return [a[sl] for a in out]


def box_blur2(a, r):
    return cv2.blur(a.astype(np.float32), (2 * r + 1, 2 * r + 1))


def gauss_cv(a, sigma):
    return cv2.GaussianBlur(a.astype(np.float32), (0, 0), sigma)


# ---------------------------------------------------------------- B variants
def var_energy_cv(frames, radius, perchannel):
    k = 2 * radius + 1
    E = []
    for f in frames:
        if perchannel and f.ndim == 3:
            e = np.zeros(f.shape[:2], np.float32)
            for c in range(f.shape[2]):
                g = f[..., c].astype(np.float32)
                m = cv2.blur(g, (k, k))
                e += cv2.blur(g * g, (k, k)) - m * m
            E.append(np.maximum(e, 0))
        else:
            g = f.mean(axis=2) if f.ndim == 3 else f.astype(np.float32)
            m = cv2.blur(g, (k, k))
            E.append(np.maximum(cv2.blur(g * g, (k, k)) - m * m, 0))
    return E


def b_pick(frames, depth_f, base_last=True):
    n = len(frames)
    idx = np.clip(np.rint(depth_f), 0, n - 1).astype(np.int32)
    stack = np.stack([f if f.ndim == 3 else f[:, :, np.newaxis] for f in frames], 0)
    out = np.take_along_axis(stack, idx[None, :, :, None], axis=0)[0]
    if frames[0].ndim == 2:
        out = out[:, :, 0]
    if base_last:
        out = cv_base_correct(out, frames[-1])
    return out


def b_variant(frames, radius=2, perchannel=True, smooth='validated'):
    E = var_energy_cv(frames, radius, perchannel)
    d = np.argmax(np.stack(E), 0).astype(np.float32)
    if smooth == 'validated':
        d = box_blur2(d, 1)
        d = gauss_cv(d, 1.0)
    elif smooth == 'port':
        du8 = np.clip(d, 0, 255).astype(np.uint8)
        d = cv2.medianBlur(du8, 3).astype(np.float32)
        d = gauss_cv(d, 1.2)
    return b_pick(frames, d)


# ---------------------------------------------------------------- A variants
def a_variant(frames, radius=3, second_box=True, sigma=3.0, base_last=True):
    k = 2 * radius + 1
    W = []
    for f in frames:
        g = f.mean(axis=2) if f.ndim == 3 else f.astype(np.float32)
        m = cv2.blur(g, (k, k))
        w = np.maximum(cv2.blur(g * g, (k, k)) - m * m, 0)
        if second_box:
            w = cv2.blur(w, (k, k))
        W.append(gauss_cv(w, sigma))
    acc = None; wsum = None
    for w, f in zip(W, frames):
        w2 = w[:, :, None] if f.ndim == 3 else w
        acc = f.astype(np.float32) * w2 if acc is None else acc + f.astype(np.float32) * w2
        wsum = w2.copy() if wsum is None else wsum + w2
    last = frames[-1]
    last = last.astype(np.float32)
    acc = acc + last; wsum = wsum + 1.0
    out = acc / wsum
    if base_last:
        out = cv_base_correct(out, last)
    return out


# ---------------------------------------------------------------- C variants
def expand_pil(band, shape):
    h, w = band.shape
    im = Image.fromarray(band.astype(np.float32), mode='F').resize(
        (w * 2, h * 2), resample=Image.BICUBIC)
    return gauss_blur(np.asarray(im, dtype=np.float64)[:shape[0], :shape[1]], 0.6)


def expand_cv(band, shape):
    up = cv2.resize(band.astype(np.float32), (band.shape[1] * 2, band.shape[0] * 2),
                    interpolation=cv2.INTER_CUBIC)
    return gauss_cv(up[:shape[0], :shape[1]], 0.6)


def c_variant(frames, levels=7, ws=3.0, mode='maxpick', expand=expand_cv,
              weird_axis_blur=False):
    def reduce_(I):
        return cv2.GaussianBlur(I.astype(np.float32), (0, 0), 1.0)[::2, ::2]

    pyrs = []
    for f in frames:
        g = [f.astype(np.float32)]
        for _ in range(levels - 1):
            g.append(reduce_(g[-1]))
        lap = [g[i] - expand(g[i + 1], g[i].shape[:2]) for i in range(levels - 1)]
        lap.append(g[-1]); pyrs.append(lap)
    out_pyr = []
    for k in range(levels):
        band = np.stack([p[k] for p in pyrs], 0).astype(np.float32)
        if k == levels - 1:
            out_pyr.append(band[-1].copy()); continue
        if weird_axis_blur:
            # 复刻 reproduce_and_compare.repro_C 的原样行为：
            # gauss_blur 对 (n,h,w) 沿帧轴+行轴卷积（历史意外，验证数字由它产生）
            W = np.asarray(gauss_blur(np.abs(band.astype(np.float64)), ws), np.float32)
        else:
            W = np.stack([gauss_cv(np.abs(b), ws) for b in band], 0)
        if mode == 'maxpick':
            idx = W.argmax(0)
            out_pyr.append(np.take_along_axis(band, idx[None], 0)[0])
        else:  # softnorm（v1.39 移植版行为）
            tot = W.sum(0)
            mixed = (W[:, :, None] * band).sum(0) / np.maximum(tot, 1e-9)[:, :, None]
            flat = tot <= 1e-9
            if flat.any():
                mixed[flat] = band[-1][flat]
            out_pyr.append(mixed)
    img = out_pyr[-1]
    for k in range(levels - 2, -1, -1):
        img = expand(img, out_pyr[k].shape[:2]) + out_pyr[k]
    return img


# ================================================================ REAL DATA B
def run_real_b():
    print('=' * 72)
    print('REAL-DATA Method B  (Helicon refs: real_*_r3_s3.tif, M=B R=3 S=3)')
    print('=' * 72)
    from real_compare import repro_B_color  # validated final config (published table)
    for stack in ['keyboard', 'balls', 'kitchen']:
        files = sorted(glob.glob(os.path.join(DATA, stack, '*.jpg')))
        frames = np.stack([load_rgb(f) for f in files]).astype(np.float32)
        ref = np.asarray(Image.open(OUT + r'\real_%s_r3_s3.tif' % stack).convert('RGB'),
                         dtype=np.float64)
        variants = {
            'validated(pub)': repro_B_color(frames.astype(np.float64), radius=2, sp=3),
            'port(current)': openfocus_b_impl(list(frames), kernel_size=7),
            'B+perchan': b_variant(frames, 2, perchannel=True, smooth='port'),
            'B+perchan+vsmooth': b_variant(frames, 2, perchannel=True, smooth='validated'),
        }
        # port needs base correction to mirror Helicon always-on semantics
        variants['port(current)'] = cv_base_correct(
            variants['port(current)'].astype(np.float32), frames[-1])
        line = '%-10s' % stack
        for name, out in variants.items():
            e, ep, p6, p12 = metrics(np.asarray(out, np.float64), ref)
            line += '  %s=%.2f%%' % (name, ep)
        print(line)


# ================================================================ SYNTHETIC
def run_synth():
    print()
    print('=' * 72)
    print('SYNTHETIC stacks (16-bit TIFF, scaled to 0-255; interior 6 metrics)')
    print('=' * 72)
STACKS = {'stackA': [ROOT + r'\stackA\a_%d.tif' % i for i in range(2)],
          'stackB': [ROOT + r'\stackB\b_%d.tif' % i for i in range(3)],
          'stackC': [ROOT + r'\stackC\c_%d.tif' % i for i in range(2)]}


def load_synth(name, extra=None):
    """栈与参考必须用同一缩放因子，否则亮度差栈会引入全局增益差。"""
    raw = [np.asarray(Image.open(p), dtype=np.float64) for p in STACKS[name]]
    mx = max(f.max() for f in raw)
    if extra is not None:
        mx = max(mx, np.asarray(Image.open(OUT + '\\' + extra), dtype=np.float64).max())
    s = 255.0 / mx
    return [f * s for f in raw]


def get_ref(tag, scale_from=None):
    r = np.asarray(Image.open(OUT + '\\' + tag), dtype=np.float64)
    mx = scale_from if scale_from is not None else r.max()
    return r * (255.0 / mx)


CASES = [
    # (stack, method, ref tag)
    ('stackA', 'A', 'A_m0.tif'),
    ('stackB', 'A', 'B_m0_r3_s3.tif'),
    ('stackB', 'B', 'B_m1_r1_s0.tif'),
    ('stackC', 'B', 'refB_stackC.tif'),
    ('stackA', 'C', 'refC_stackA.tif'),
    ('stackB', 'C', 'refC_r1_s0.tif'),
    ('stackC', 'C', 'refC_stackC.tif'),
]


def _run_synth_cases():
    for stack_name, method, tag in CASES:
        raw = [np.asarray(Image.open(p), dtype=np.float64) for p in STACKS[stack_name]]
        ref_raw = np.asarray(Image.open(OUT + '\\' + tag), dtype=np.float64)
        scale = 255.0 / max(max(f.max() for f in raw), ref_raw.max())
        frames = align_to_first([f * scale for f in raw])
        ref = ref_raw * scale
        out = {}
        if method == 'A':
            out['validated'] = a_variant(frames, 3, True, 3.0)          # box+gauss3
            out['port(current)'] = a_variant(frames, 3, False, 2.0)     # gauss2 only
        elif method == 'B':
            out['validated'] = b_variant(frames, 2, True, 'validated')
            out['port(current)'] = b_variant(frames, 2, False, 'port')
        else:
            out['validated(weird)'] = c_variant(frames, expand=expand_pil, weird_axis_blur=True)
            out['sanitized'] = c_variant(frames, expand=expand_pil)
            # 注意：不做 [:ref.shape] 左上裁切——Helicon 裁边与本地对齐裁边
            # 每栈相差 0-2px，居中裁切度量（synth_metrics）才能对齐坐标
            out['port(current)'] = openfocus_c_impl(list(frames))
            out['validated(cv2)'] = c_variant(frames, expand=expand_cv)
        line = '%-8s %s  ref=%s' % (stack_name, method, tag)
        for name, o in out.items():
            e, ep = synth_metrics(np.asarray(o, np.float64), ref)
            line += '  %s=%.2f%%' % (name, ep)
        print(line)


if __name__ == '__main__':
    run_real_b()
    run_synth()
    _run_synth_cases()
    print()
    print('done.')
