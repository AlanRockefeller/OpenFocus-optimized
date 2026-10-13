import sys, os
sys.path.insert(0, os.path.dirname(__file__))
import numpy as np
from PIL import Image
from helicon_repro import method_a, method_b_depth, smooth_depth, method_b_composite, method_c

ROOT = r'F:\Working\helicon-rea\tests'
OUT = ROOT + r'\out'
def load(p): return np.asarray(Image.open(p), dtype=np.float64)
def interior(a, m=6): return a[m:-m, m:-m]
def best_shift(mine, ref, R=4):
    best = (1e18, 0, 0)
    for dy in range(-R, R + 1):
        for dx in range(-R, R + 1):
            m = np.roll(np.roll(mine, dy, 0), dx, 1)
            e = float(np.abs(m[R:-R, R:-R] - ref[R:-R, R:-R]).mean())
            if e < best[0]:
                best = (e, dy, dx)
    return best

src = [load(ROOT + rf'\stackB\b_{i}.tif') for i in range(3)]
srcA = [load(ROOT + r'\stackA\a_0.tif'), load(ROOT + r'\stackA\a_1.tif')]
srcC = [load(ROOT + r'\stackC\c_0.tif'), load(ROOT + r'\stackC\c_1.tif')]
def crop_to(a, shape):
    hh, ww = shape; y0, x0 = (a.shape[0]-hh)//2, (a.shape[1]-ww)//2
    return a[y0:y0+hh, x0:x0+ww]

mine = {}
ref = load(OUT + r'\refB_r3_s0.tif')
idx, _ = method_b_depth(src, radius=2)
d = smooth_depth(crop_to(idx.astype(float), ref.shape), 3, smoothing=0, feather=1.0)
mine['B stackB r3s0'] = (method_b_composite([crop_to(f, ref.shape) for f in src], d, 1.0), ref)
ref = load(OUT + r'\refB_r3_s3.tif')
d = smooth_depth(crop_to(idx.astype(float), ref.shape), 3, smoothing=1, feather=1.0)
mine['B stackB r3s3'] = (method_b_composite([crop_to(f, ref.shape) for f in src], d, 1.0), ref)
ref = load(OUT + r'\B_m0_r3_s3.tif')
mine['A stackB'] = (method_a([crop_to(f, ref.shape) for f in src], radius=3, smoothing=3, measure='var4'), ref)
ref = load(OUT + r'\refC_r1_s0.tif')
mine['C stackB'] = (method_c([crop_to(f, ref.shape) for f in src], levels=3, wsigma=3.0, mode='softnorm'), ref)
ref = load(OUT + r'\refC_stackC.tif')
mine['C stackC'] = (method_c([crop_to(f, ref.shape) for f in srcC], levels=3, wsigma=3.0, mode='softnorm'), ref)
ref = load(OUT + r'\refC_stackA.tif')
mine['C stackA'] = (method_c([crop_to(f, ref.shape) for f in srcA], levels=3, wsigma=3.0, mode='softnorm'), ref)
ref = load(OUT + r'\refB_stackC.tif')
d = smooth_depth(crop_to(method_b_depth(srcC, radius=2)[0].astype(float), ref.shape), 2, smoothing=0, feather=1.0)
mine['B stackC'] = (method_b_composite([crop_to(f, ref.shape) for f in srcC], d, 1.0), ref)

hdr = 'case             rng   MAE   MAE%    <=2    <=6   <=12   best-shift'
print(hdr)
rows = []
for name, (m, r) in mine.items():
    mm, rr = interior(m), interior(r)
    rng = rr.max()
    e = float(np.abs(mm - rr).mean())
    p2 = float((np.abs(mm - rr) <= 2).mean()) * 100
    p6 = float((np.abs(mm - rr) <= 6).mean()) * 100
    p12 = float((np.abs(mm - rr) <= 12).mean()) * 100
    bs, dy, dx = best_shift(mm, rr)
    print('%-16s %4.0f %6.2f %5.1f%% %5.1f%% %5.1f%% %5.1f%%   MAE=%.2f @(%+d,%+d)'
          % (name, rng, e, e / rng * 100, p2, p6, p12, bs, dy, dx))
    rows.append((name, e, p2, p6, p12))
