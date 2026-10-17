import sys, os
sys.path.insert(0, os.path.dirname(__file__))
import numpy as np
from PIL import Image
from helicon_repro import _pyramid, _expand, base_correct, gauss_blur, MEASURES, window_energy

ROOT = r'F:\Working\helicon-rea\tests'
OUT = ROOT + r'\out'
def load(p): return np.asarray(Image.open(p), dtype=np.float64)
def interior(a, m=6): return a[m:-m, m:-m]
def crop_to(a, shape):
    hh, ww = shape; y0, x0 = (a.shape[0]-hh)//2, (a.shape[1]-ww)//2
    return a[y0:y0+hh, x0:x0+ww]

src = [load(ROOT + rf'\stackB\b_{i}.tif') for i in range(3)]
f0 = src[0]

print('=== pyramid identity reconstruction test (f0 -> L levels -> f0) ===')
def identity_err(expand_fn, levels):
    g = [f0]
    for _ in range(levels - 1):
        g.append(gauss_blur(g[-1], 1.0)[::2, ::2])
    lap = [g[k] - expand_fn(g[k+1], g[k].shape) for k in range(levels - 1)]
    lap.append(g[-1])
    img = lap[-1]
    for k in range(levels - 2, -1, -1):
        img = expand_fn(img, lap[k].shape) + lap[k]
    return float(np.abs(img - f0).mean())

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

def expand_pil(lap, shape, resample=Image.BICUBIC):
    h, w = lap.shape
    im = Image.fromarray(lap.astype(np.float32), mode='F').resize((w * 2, h * 2), resample=resample)
    a = np.asarray(im, dtype=np.float64)[:shape[0], :shape[1]]
    return gauss_blur(a, 0.6)

for name, fn in [('repeat+blur0.6', _expand), ('burt', expand_burt),
                 ('pil-bicubic+blur0.6', expand_pil),
                 ('pil-lanczos+blur0.6', lambda l, s: expand_pil(l, s, Image.LANCZOS))]:
    for L in [2, 3, 4]:
        print('  %-22s L=%d identity MAE=%.2f' % (name, L, identity_err(fn, L)))

# C with best expand
def c_with_expand(stack, shape, levels=3, ws=3.0, mode='maxpick', expand_fn=expand_pil):
    stacks = [crop_to(f, shape) for f in stack]
    def reduce_(I): return gauss_blur(I, 1.0)[::2, ::2]
    pyrs = []
    for f in stacks:
        g = [f]
        for _ in range(levels - 1):
            g.append(reduce_(g[-1]))
        lap = [g[k] - expand_fn(g[k + 1], g[k].shape) for k in range(levels - 1)]
        lap.append(g[-1])
        pyrs.append(lap)
    out_pyr = []
    for k in range(levels):
        band = np.stack([p[k] for p in pyrs])
        if k == levels - 1:
            out_pyr.append(band[-1].copy()); continue
        W = np.abs(band)
        if ws: W = gauss_blur(W, ws)
        if mode == 'maxpick':
            idx = W.argmax(0)
            yy, xx = np.meshgrid(np.arange(band.shape[1]), np.arange(band.shape[2]), indexing='ij')
            mixed = band[idx, yy, xx]
        else:
            tot = W.sum(0)
            mixed = (W * band).sum(0) / np.maximum(tot, 1e-9)
            mixed[tot <= 1e-9] = band[-1][tot <= 1e-9]
        out_pyr.append(mixed)
    img = out_pyr[-1]
    for k in range(levels - 2, -1, -1):
        img = expand_fn(img, out_pyr[k].shape) + out_pyr[k]
    return img

srcC = [load(ROOT + r'\stackC\c_0.tif'), load(ROOT + r'\stackC\c_1.tif')]
srcA = [load(ROOT + r'\stackA\a_0.tif'), load(ROOT + r'\stackA\a_1.tif')]
print('\n=== C with pil-bicubic expand ===')
for mode in ['maxpick', 'softnorm']:
    for L in [3, 4]:
        es = []
        for stack, refk in [(src, 'refC_r1_s0'), (srcC, 'refC_stackC'), (srcA, 'refC_stackA')]:
            ref = load(OUT + '\\' + refk + '.tif')
            out = c_with_expand(stack, ref.shape, levels=L, ws=3.0, mode=mode, expand_fn=expand_pil)
            es.append(np.abs(interior(out) - interior(ref)).mean())
        print('  %-9s L=%d  stackB=%6.2f stackC=%6.2f stackA=%6.2f' % (mode, L, *es))

# B half-pixel test
print('\n=== B: half-pixel registration test ===')
ref = load(OUT + r'\refB_r3_s0.tif')
E = np.stack([window_energy(MEASURES['var2'](f), 0) for f in src])
idx = E.argmax(0)
d = smooth_depth(crop_to(idx.astype(float), ref.shape), 3, smoothing=0, feather=1.0)
out = None
S = np.stack([crop_to(f, ref.shape) for f in src])
sel = np.clip(np.rint(d), 0, 2).astype(int)
yy, xx = np.meshgrid(np.arange(S.shape[1]), np.arange(S.shape[2]), indexing='ij')
raw = S[sel, yy, xx]
out = base_correct(raw, S[-1], 32)
def subpixel_shift(a, dy, dx):
    fy = a - np.floor(a); fy[fy == 0] = 1  # unused
    y0 = int(np.floor(dy)); x0 = int(np.floor(dx))
    fy = dy - y0; fx = dx - x0
    a0 = np.roll(np.roll(a, y0, 0), x0, 1)
    a1 = np.roll(np.roll(a, y0 + (1 if fy > 0 else 0), 0), x0, 1)
    a2 = np.roll(np.roll(a, y0, 0), x0 + (1 if fx > 0 else 0), 1)
    a3 = np.roll(np.roll(a1, 0, 0), x0 + (1 if fx > 0 else 0), 1)
    return (1-fy)*(1-fx)*np.roll(np.roll(a, y0, 0), x0, 1) + fy*(1-fx)*np.roll(np.roll(a, y0+1, 0), x0, 1) \
         + (1-fy)*fx*np.roll(np.roll(a, y0, 0), x0+1, 1) + fy*fx*np.roll(np.roll(a, y0+1, 0), x0+1, 1)
m = interior(out); r = interior(ref)
print('  raw MAE=%.2f' % np.abs(m - r).mean())
for dy, dx in [(0.5, 0), (0, 0.5), (0.5, 0.5), (-0.5, 0), (0, -0.5), (0, 2.5), (0.5, 2.5), (0, 2.0), (0.5, 2.0)]:
    ms = subpixel_shift(m, dy, dx)[8:-8, 8:-8]
    print('  shift(%+.1f,%+.1f) MAE=%.2f' % (dy, dx, np.abs(ms - r[8:-8, 8:-8]).mean()))
