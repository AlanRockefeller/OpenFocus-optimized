import sys, os
sys.path.insert(0, os.path.dirname(__file__))
import numpy as np
from PIL import Image
from helicon_repro import (method_a, method_b_depth, smooth_depth, method_b_composite,
                           method_c, base_correct)

ROOT = r'F:\Working\helicon-rea\tests'
OUT = ROOT + r'\out'
def load(p): return np.asarray(Image.open(p), dtype=np.float64)
def interior(a, m=6): return a[m:-m, m:-m]
def crop_to(a, shape):
    hh, ww = shape; y0, x0 = (a.shape[0]-hh)//2, (a.shape[1]-ww)//2
    return a[y0:y0+hh, x0:x0+ww]

src = [load(ROOT + rf'\stackB\b_{i}.tif') for i in range(3)]
srcA = [load(ROOT + r'\stackA\a_0.tif'), load(ROOT + r'\stackA\a_1.tif')]
srcC = [load(ROOT + r'\stackC\c_0.tif'), load(ROOT + r'\stackC\c_1.tif')]
def best_shift(mine, ref, R=4):
    best = (1e18, 0, 0)
    for dy in range(-R, R + 1):
        for dx in range(-R, R + 1):
            m = np.roll(np.roll(mine, dy, 0), dx, 1)
            e = float(np.abs(m[R:-R, R:-R] - ref[R:-R, R:-R]).mean())
            if e < best[0]:
                best = (e, dy, dx)
    return best

refs = {k: load(OUT + '\\' + k + '.tif') for k in
        ['refB_r3_s0', 'refB_r3_s3', 'B_m0_r3_s3', 'refC_r1_s0', 'refC_stackC', 'refC_stackA', 'refB_stackC']}

def raw_B(stack, shape, sm=0):
    idx, _ = method_b_depth(stack, radius=2)
    d = smooth_depth(crop_to(idx.astype(float), shape), len(stack), smoothing=sm, feather=1.0)
    return d

SIGMAS = [0, 8, 16, 24, 32, 48]
print('=== Method B (var2, feather=1) base_sigma sweep ===')
for sig in SIGMAS:
    es = []
    for refk, stack in [('refB_r3_s0', src), ('refB_stackC', srcC)]:
        ref = refs[refk]
        d = raw_B(stack, ref.shape)
        out = method_b_composite([crop_to(f, ref.shape) for f in stack], d, 1.0,
                                 base=('last' if sig else None), base_sigma=max(sig, 1))
        es.append(np.abs(interior(out) - interior(ref)).mean())
    print('  sigma=%2d  stackB=%6.2f  stackC=%6.2f' % (sig, es[0], es[1]))

print('=== Method A (var4) base_sigma sweep ===')
for sig in SIGMAS:
    ref = refs['B_m0_r3_s3']
    out = method_a([crop_to(f, ref.shape) for f in src], radius=3, smoothing=3, measure='var4',
                   base=('last' if sig else None), base_sigma=max(sig, 1))
    print('  sigma=%2d  stackB=%6.2f' % (sig, np.abs(interior(out) - interior(ref)).mean()))

print('=== Method C (L=3 ws=3 softnorm, base band=last) base sigma n/a; refine wsigma ===')
for ws in [1.0, 2.0, 3.0, 4.0]:
    for L in [2, 3, 4]:
        e1 = np.abs(interior(method_c([crop_to(f, refs['refC_r1_s0'].shape) for f in src],
                                      levels=L, wsigma=ws)) - interior(refs['refC_r1_s0'])).mean()
        e2 = np.abs(interior(method_c([crop_to(f, refs['refC_stackC'].shape) for f in srcC],
                                      levels=L, wsigma=ws)) - interior(refs['refC_stackC'])).mean()
        e3 = np.abs(interior(method_c([crop_to(f, refs['refC_stackA'].shape) for f in srcA],
                                      levels=L, wsigma=ws)) - interior(refs['refC_stackA'])).mean()
        print('  ws=%.0f L=%d  stackB=%6.2f stackC=%6.2f stackA=%6.2f' % (ws, L, e1, e2, e3))

# final combined numbers with best config
print('\n=== FINAL (base last, sigma=32; C L=3 ws=3) ===')
for name, refk, stack in [('B stackB', 'refB_r3_s0', src), ('B stackC', 'refB_stackC', srcC),
                          ('A stackB', 'B_m0_r3_s3', src), ('C stackB', 'refC_r1_s0', src),
                          ('C stackC', 'refC_stackC', srcC), ('C stackA', 'refC_stackA', srcA)]:
    ref = refs[refk]
    if name.startswith('B'):
        d = raw_B(stack, ref.shape, sm=0 if 'r3s0' in refk else 1)
        out = method_b_composite([crop_to(f, ref.shape) for f in stack], d, 1.0, base='last', base_sigma=32)
    elif name.startswith('A'):
        out = method_a([crop_to(f, ref.shape) for f in stack], radius=3, smoothing=3, measure='var4',
                       base='last', base_sigma=32)
    else:
        out = method_c([crop_to(f, ref.shape) for f in stack], levels=3, wsigma=3.0)
    m, r = interior(out), interior(ref)
    e = float(np.abs(m - r).mean()); p6 = float((np.abs(m - r) <= 6).mean()) * 100
    p12 = float((np.abs(m - r) <= 12).mean()) * 100
    bs, dy, dx = best_shift(m, r)
    print('  %-9s MAE=%6.2f (%4.1f%% rng)  <=6:%5.1f%%  <=12:%5.1f%%  best-shift MAE=%.2f @(%+d,%+d)'
          % (name, e, e / r.max() * 100, p6, p12, bs, dy, dx))
