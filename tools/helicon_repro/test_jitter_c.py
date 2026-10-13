"""Test: (1) stackB residual = Helicon's own alignment jitter (~2-3px);
(2) C stackB residual = crude pyramid interpolation -> try Burt-style expand."""
import sys, os
sys.path.insert(0, os.path.dirname(__file__))
import numpy as np
from PIL import Image
from helicon_repro import (method_b_depth, smooth_depth, method_b_composite, gauss_blur,
                           base_correct, _pyramid, method_c)

ROOT = r'F:\Working\helicon-rea\tests'
OUT = ROOT + r'\out'
def load(p): return np.asarray(Image.open(p), dtype=np.float64)
def interior(a, m=6): return a[m:-m, m:-m]
def crop_to(a, shape):
    hh, ww = shape; y0, x0 = (a.shape[0]-hh)//2, (a.shape[1]-ww)//2
    return a[y0:y0+hh, x0:x0+ww]
def shift(a, dy, dx):
    return np.roll(np.roll(a, dy, 0), dx, 1)

src = [load(ROOT + rf'\stackB\b_{i}.tif') for i in range(3)]
ref = load(OUT + r'\refB_r3_s0.tif')

print('=== jitter hypothesis: pre-shift f1/f2 by (dy,dx) then run B ===')
idx, _ = method_b_depth(src, radius=2)
d = smooth_depth(crop_to(idx.astype(float), ref.shape), 3, smoothing=0, feather=1.0)
base_case = method_b_composite([crop_to(f, ref.shape) for f in src], d, 1.0, base=None)
print('  no-jitter MAE=%.2f' % np.abs(interior(base_case) - interior(ref)).mean())
for (dy, dx) in [(2, 3), (-2, -3), (2, -3), (-2, 3), (0, 3), (2, 0), (1, 1), (3, 2)]:
    st = [crop_to(src[0], ref.shape), crop_to(shift(src[1], dy, dx), ref.shape),
          crop_to(shift(src[2], dy, dx), ref.shape)]
    idx2, _ = method_b_depth(st, radius=2)
    d2 = smooth_depth(crop_to(idx2.astype(float), ref.shape), 3, smoothing=0, feather=1.0)
    out = method_b_composite(st, d2, 1.0, base=None)
    print('  jitter(%+d,%+d) MAE=%.2f' % (dy, dx, np.abs(interior(out) - interior(ref)).mean()))

print('\n=== C pyramid: Burt-style expand vs repeat+blur ===')
def expand_burt(lap, shape):
    h, w = lap.shape
    up = np.zeros((h * 2, w * 2))
    up[::2, ::2] = lap
    k = np.array([0.25, 0.5, 0.25])
    p = np.pad(up, ((1, 1), (0, 0)), mode='edge')
    up = sum(k[i] * p[i:i + up.shape[0]] for i in range(3)) * 2
    p = np.pad(up, ((0, 0), (1, 1)), mode='edge')
    up = sum(k[i] * p[:, i:i + up.shape[1]] for i in range(3)) * 2
    return up[:shape[0], :shape[1]]

def method_c2(stack, levels=3, wsigma=3.0, expand=expand_burt):
    n = len(stack)
    # build pyramids with burt reduce/expand
    def reduce(I):
        b = gauss_blur(I, 1.0)
        return b[::2, ::2]
    pyrs = []
    for f in stack:
        g = [f.astype(np.float64)]
        for _ in range(levels - 1):
            g.append(reduce(g[-1]))
        lap = [g[k] - expand_burt(g[k + 1], g[k].shape) for k in range(levels - 1)]
        lap.append(g[-1])
        pyrs.append(lap)
    out_pyr = []
    for k in range(levels):
        band = np.stack([p[k] for p in pyrs])
        if k == levels - 1:
            out_pyr.append(band[-1].copy())
            continue
        W = np.abs(band)
        W = gauss_blur(W, wsigma)
        tot = W.sum(axis=0)
        mixed = (W * band).sum(axis=0) / np.maximum(tot, 1e-9)
        flat = tot <= 1e-9
        mixed[flat] = band[-1][flat]
        out_pyr.append(mixed)
    img = out_pyr[-1]
    for k in range(levels - 2, -1, -1):
        img = expand_burt(img, out_pyr[k].shape) + out_pyr[k]
    return img

srcA = [load(ROOT + r'\stackA\a_0.tif'), load(ROOT + r'\stackA\a_1.tif')]
srcC = [load(ROOT + r'\stackC\c_0.tif'), load(ROOT + r'\stackC\c_1.tif')]
for name, stack, refk in [('C stackB', src, 'refC_r1_s0'), ('C stackC', srcC, 'refC_stackC'),
                          ('C stackA', srcA, 'refC_stackA')]:
    ref = load(OUT + '\\' + refk + '.tif')
    out = method_c2([crop_to(f, ref.shape) for f in stack], levels=3, wsigma=3.0)
    m, r = interior(out), interior(ref)
    print('  %-9s burt-expand MAE=%.2f (%.1f%% rng)  <=6=%.1f%%'
          % (name, np.abs(m - r).mean(), np.abs(m - r).mean() / r.max() * 100,
             (np.abs(m - r) <= 6).mean() * 100))
