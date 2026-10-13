"""Pure-fusion comparison on pre-aligned TIFF stacks: sweep measure/composite params."""
import sys, os, glob
sys.path.insert(0, os.path.dirname(__file__))
import numpy as np
from PIL import Image
from real_compare import metrics

PREP = r'F:\Working\helicon-rea\tests\prealigned'
OUT = r'F:\Working\helicon-rea\tests\out'

def box_blur(a, r):
    if r <= 0: return a.astype(np.float64)
    k = 2*r+1
    p = np.pad(a, ((r,r),(0,0)), mode='edge'); c = np.cumsum(p, 0)
    a = (c[k-1:] - np.concatenate([np.zeros((1,)+a.shape[1:]), c[:-k]]))/k
    p = np.pad(a, ((0,0),(r,r)), mode='edge'); c = np.cumsum(p, 1)
    return (c[:, k-1:] - np.concatenate([np.zeros((a.shape[0],1)), c[:, :-k]], 1))/k

def gauss1d(a, sigma):
    if sigma <= 0: return a.astype(np.float64)
    r = max(1, int(3*sigma))
    k = np.exp(-0.5*(np.arange(-r,r+1)/sigma)**2); k /= k.sum()
    pad0 = ((r, r), (0, 0)) + ((0, 0),) * (a.ndim - 2)
    a = sum(k[i]*np.pad(a, pad0, mode='edge')[i:i+a.shape[0]] for i in range(2*r+1))
    pad1 = ((0, 0), (r, r)) + ((0, 0),) * (a.ndim - 2)
    return sum(k[i]*np.pad(a, pad1, mode='edge')[:, i:i+a.shape[1]] for i in range(2*r+1))

def base_correct(out, last, sigma):
    if sigma <= 0: return out
    return out + gauss1d(last, sigma) - gauss1d(out, sigma)

def load_stack(stack):
    files = sorted(glob.glob(os.path.join(PREP, stack, '*.tif')))
    return np.stack([np.asarray(Image.open(f), dtype=np.float64) for f in files])

def var_energy(f, radius, mode='perchannel'):
    if mode == 'luma':
        g = f.mean(axis=2)
        m = box_blur(g, radius)
        return box_blur((g-m)**2, radius)
    e = np.zeros(f.shape[:2])
    for c in range(3):
        m = box_blur(f[..., c], radius)
        e += box_blur((f[..., c]-m)**2, radius)
    return e

def repro(frames, radius=3, sp=3, measure_mode='perchannel', depth_smooth=1, feather=1.0,
          base_sigma=32.0, transition=1.0):
    E = np.stack([var_energy(f, radius, measure_mode) for f in frames])
    idx = E.argmax(0)
    n = len(frames)
    d = idx.astype(np.float64)
    if depth_smooth: d = box_blur(d, depth_smooth)
    if feather: d = gauss1d(d, float(feather))
    q = np.clip(np.rint(d), 0, n-1)
    if transition > 0:
        W = np.clip(1.0 - np.abs(np.arange(n)[:, None, None] - d[None]) / transition, 0, None)[..., None]
        S = np.stack(frames)
        tot = W.sum(0)
        out = (W * S).sum(0) / np.maximum(tot, 1e-9)
        flat = tot[..., 0] <= 1e-9
        out[flat] = frames[-1][flat]
    else:
        sel = q.astype(int)
        H, W = sel.shape
        yy, xx = np.meshgrid(np.arange(H), np.arange(W), indexing='ij')
        S = np.stack(frames)
        out = S[sel, yy, xx]
    out = base_correct(out, frames[-1], base_sigma)
    return out

if __name__ == '__main__':
    for stack in ['kitchen', 'keyboard']:
        frames = load_stack(stack)
        ref = np.asarray(Image.open(OUT + r'\pre_%s.tif' % stack).convert('RGB'), dtype=np.float64)
        print('=== %s (n=%d, ref %s) ===' % (stack, len(frames), ref.shape))
        results = []
        for mode in ['perchannel', 'luma']:
            for radius in [1, 2, 3, 4, 5]:
                for dsm, fe in [(1, 1.0), (2, 1.0), (0, 1.0)]:
                    mine = repro(frames, radius=radius, sp=3, measure_mode=mode,
                                 depth_smooth=dsm, feather=fe)
                    e, ep, p6, p12 = metrics(mine, ref)
                    results.append((e, mode, radius, dsm, fe, ep, p6, p12))
        results.sort()
        for e, mode, radius, dsm, fe, ep, p6, p12 in results[:8]:
            print('  %-10s r=%d dsm=%d fe=%.0f  MAE=%6.2f (%.2f%%)  <=6=%5.1f%% <=12=%5.1f%%'
                  % (mode, radius, dsm, fe, e, ep, p6, p12))
