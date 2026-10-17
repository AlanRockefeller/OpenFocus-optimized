import sys, os
sys.path.insert(0, os.path.dirname(__file__))
import numpy as np
from PIL import Image
from helicon_repro import _pyramid, _expand, base_correct, gauss_blur

ROOT = r'F:\Working\helicon-rea\tests'
OUT = ROOT + r'\out'
def load(p): return np.asarray(Image.open(p), dtype=np.float64)
def interior(a, m=6): return a[m:-m, m:-m]
def crop_to(a, shape):
    hh, ww = shape; y0, x0 = (a.shape[0]-hh)//2, (a.shape[1]-ww)//2
    return a[y0:y0+hh, x0:x0+ww]

src = [load(ROOT + rf'\stackB\b_{i}.tif') for i in range(3)]
ref = load(OUT + r'\refB_r3_s0.tif')

def box_blur(a, r):
    if r <= 0: return a.astype(np.float64)
    k = 2*r+1
    p = np.pad(a, ((r,r),(0,0)), mode='edge'); c = np.cumsum(p, 0)
    a = (c[k-1:] - np.concatenate([np.zeros((1,)+a.shape[1:]), c[:-k]]))/k
    p = np.pad(a, ((0,0),(r,r)), mode='edge'); c = np.cumsum(p, 1)
    return (c[:, k-1:] - np.concatenate([np.zeros((a.shape[0],1)), c[:, :-k]], 1))/k

def var_measure(I, r=2):
    m = box_blur(I, r)
    return box_blur((I-m)**2, r)

E = np.stack([var_measure(f) for f in src])          # energy maps (3,H,W)
E = E[:, :ref.shape[0], :ref.shape[1]]
S = np.stack([crop_to(f, ref.shape) for f in src])
Emax = E.max(axis=0, keepdims=True)

print('=== B: soft-max blend sweep (w = exp(beta * (E - Emax))) ===')
for beta in [0.0, 1e-4, 5e-4, 1e-3, 2e-3, 5e-3, 1e-2]:
    W = np.exp(beta * (E - Emax))
    out = (W * S).sum(0) / W.sum(0)
    out = base_correct(out, S[-1], 32)
    e = np.abs(interior(out) - interior(ref)).mean()
    print('  beta=%-6.4g MAE=%.2f' % (beta, e))

print('=== B: power-weight blend sweep (w = (E+eps)^g) ===')
for g in [1, 2, 4, 8, 16, 32]:
    W = (E + 1.0) ** g
    out = (W * S).sum(0) / W.sum(0)
    out = base_correct(out, S[-1], 32)
    e = np.abs(interior(out) - interior(ref)).mean()
    print('  gamma=%-3d MAE=%.2f' % (g, e))

# C variants with base band = last
from helicon_repro import _pyramid, _expand
def c_variant(stack, shape, levels=3, ws=3.0, mode='softnorm'):
    stacks = [crop_to(f, shape) for f in stack]
    pyrs = [_pyramid(f, levels) for f in stacks]
    out_pyr = []
    for k in range(levels):
        band = np.stack([p[k] for p in pyrs])
        if k == levels - 1:
            out_pyr.append(band[-1].copy()); continue
        if mode == 'maxpick':
            idx = np.abs(band).argmax(0)
            yy, xx = np.meshgrid(np.arange(band.shape[1]), np.arange(band.shape[2]), indexing='ij')
            mixed = band[idx, yy, xx]
        elif mode == 'hardnorm':
            W = np.abs(band) ** 8
            mixed = (W * band).sum(0) / np.maximum(W.sum(0), 1e-9)
        else:
            W = np.abs(band)
            if ws: W = gauss_blur(W, ws)
            tot = W.sum(0)
            mixed = (W * band).sum(0) / np.maximum(tot, 1e-9)
            mixed[tot <= 1e-9] = band[-1][tot <= 1e-9]
        out_pyr.append(mixed)
    img = out_pyr[-1]
    for k in range(levels - 2, -1, -1):
        img = _expand(img, out_pyr[k].shape) + out_pyr[k]
    return img

print('=== C: band combine variants (stackB / stackC / stackA) ===')
srcC = [load(ROOT + r'\stackC\c_0.tif'), load(ROOT + r'\stackC\c_1.tif')]
srcA = [load(ROOT + r'\stackA\a_0.tif'), load(ROOT + r'\stackA\a_1.tif')]
for mode in ['softnorm', 'maxpick', 'hardnorm']:
    for L in [3, 4, 5]:
        es = []
        for stack, refk in [(src, 'refC_r1_s0'), (srcC, 'refC_stackC'), (srcA, 'refC_stackA')]:
            ref = load(OUT + '\\' + refk + '.tif')
            out = c_variant(stack, ref.shape, levels=L, mode=mode)
            es.append(np.abs(interior(out) - interior(ref)).mean())
        print('  %-9s L=%d  stackB=%6.2f stackC=%6.2f stackA=%6.2f' % (mode, L, *es))
