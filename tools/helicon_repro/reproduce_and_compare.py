"""FINAL Helicon Focus 8.1 reproduction pipeline + comparison figures.

Chosen configuration (fitted against the original binary's CLI output):
  alignment  : FFT phase correlation, limits-bounded, crop = |shift|+1 per side
  Method A   : w_i = gauss(var_r(I_i), Smoothing); out = sum(w*I)+eps*I_last / sum(w)+eps
               then base_correct (low-freq <- last frame, sigma=32)
  Method B   : depth = argmax(var_r(I_i)) (ties -> first frame); 1/32 quantized,
               feathered; composite = selected frame, base_correct(last, 32)
  Method C   : Laplacian pyramid (PIL-bicubic expand), per-band |band| weights,
               max-pick, coarsest band = last frame
"""
import sys, os, subprocess, glob, time
sys.path.insert(0, os.path.dirname(__file__))
import numpy as np
from PIL import Image
from helicon_repro import (gauss_blur, base_correct, smooth_depth, method_a, method_c,
                           estimate_translation, apply_shift)

ROOT = r'F:\Working\helicon-rea\tests'
OUT = ROOT + r'\out'
RES = os.path.join(os.path.dirname(__file__), 'results')
HF = r'D:\Helicon Focus 8\_HeliconFocus.exe'

def load(p): return np.asarray(Image.open(p), dtype=np.float64)
def interior(a, m=6): return a[m:-m, m:-m]
def crop_to(a, shape):
    hh, ww = shape; y0, x0 = (a.shape[0]-hh)//2, (a.shape[1]-ww)//2
    return a[y0:y0+hh, x0:x0+ww]

def box_blur(a, r):
    if r <= 0: return a.astype(np.float64)
    k = 2*r+1
    p = np.pad(a, ((r,r),(0,0)), mode='edge'); c = np.cumsum(p, 0)
    a = (c[k-1:] - np.concatenate([np.zeros((1,)+a.shape[1:]), c[:-k]]))/k
    p = np.pad(a, ((0,0),(r,r)), mode='edge'); c = np.cumsum(p, 1)
    return (c[:, k-1:] - np.concatenate([np.zeros((a.shape[0],1)), c[:, :-k]], 1))/k

def var2(I):
    m = box_blur(I, 2)
    return box_blur((I-m)**2, 2)

def align_to_first(stack, margin=1):
    ref = stack[0]
    out = [ref]; shifts = [(0, 0)]
    for f in stack[1:]:
        dy, dx = estimate_translation(ref, f)
        out.append(apply_shift(f, dy, dx)); shifts.append((dy, dx))
    ys = max(abs(s[0]) for s in shifts) + margin
    xs = max(abs(s[1]) for s in shifts) + margin
    h, w = ref.shape
    sl = (slice(ys, h-ys), slice(xs, w-xs))
    return [a[sl] for a in out], shifts

def repro_B(stack, shape, radius=2, feather=1.0, base_sigma=32):
    al, _ = align_to_first(stack)
    E = np.stack([var2(f) for f in al])
    idx = E.argmax(0)
    d = smooth_depth(crop_to(idx.astype(float), shape), len(stack), smoothing=0, feather=feather)
    S = np.stack([crop_to(f, shape) for f in al])
    sel = np.clip(np.rint(d), 0, len(stack)-1).astype(int)
    yy, xx = np.meshgrid(np.arange(S.shape[1]), np.arange(S.shape[2]), indexing='ij')
    out = S[sel, yy, xx]
    return base_correct(out, S[-1], base_sigma)

def repro_A(stack, shape, radius=3, smoothing=3, base_sigma=32):
    al, _ = align_to_first(stack)
    W = np.stack([gauss_blur(box_blur((f - box_blur(f, radius)) ** 2, radius),
                             float(smoothing)) for f in al])
    S = np.stack([crop_to(f, shape) for f in al])
    W = np.stack([crop_to(w_, shape) for w_ in W])
    out = (W * S).sum(0) + S[-1]
    out /= W.sum(0) + 1.0
    return base_correct(out, S[-1], base_sigma)

def repro_C(stack, shape, levels=7, ws=3.0):
    from helicon_repro import _expand
    al, _ = align_to_first(stack)
    stacks = [crop_to(f, shape) for f in al]
    def reduce_(I): return gauss_blur(I, 1.0)[::2, ::2]
    def expand_pil(lap, shp):
        h, w = lap.shape
        im = Image.fromarray(lap.astype(np.float32), mode='F').resize((w*2, h*2), resample=Image.BICUBIC)
        return gauss_blur(np.asarray(im, dtype=np.float64)[:shp[0], :shp[1]], 0.6)
    pyrs = []
    for f in stacks:
        g = [f]
        for _ in range(levels - 1): g.append(reduce_(g[-1]))
        lap = [g[k] - expand_pil(g[k+1], g[k].shape) for k in range(levels-1)]
        lap.append(g[-1]); pyrs.append(lap)
    out_pyr = []
    for k in range(levels):
        band = np.stack([p[k] for p in pyrs])
        if k == levels - 1:
            out_pyr.append(band[-1].copy()); continue
        W = gauss_blur(np.abs(band), ws)
        idx = W.argmax(0)
        yy, xx = np.meshgrid(np.arange(band.shape[1]), np.arange(band.shape[2]), indexing='ij')
        out_pyr.append(band[idx, yy, xx])
    img = out_pyr[-1]
    for k in range(levels - 2, -1, -1):
        img = expand_pil(img, out_pyr[k].shape) + out_pyr[k]
    return img

CASES = [
    ('B stackB r3s0', 'refB_r3_s0.tif',   'stackB', 'B'),
    ('A stackB',      'B_m0_r3_s3.tif',   'stackB', 'A'),
    ('C stackB',      'refC_r1_s0.tif',   'stackB', 'C'),
    ('B stackC',      'refB_stackC.tif',  'stackC', 'B'),
    ('C stackC',      'refC_stackC.tif',  'stackC', 'C'),
    ('A stackA',      'A_m0.tif',         'stackA', 'A'),
    ('C stackA',      'refC_stackA.tif',  'stackA', 'C'),
]
STACKS = {'stackB': [ROOT + rf'\stackB\b_{i}.tif' for i in range(3)],
          'stackA': [ROOT + r'\stackA\a_0.tif', ROOT + r'\stackA\a_1.tif'],
          'stackC': [ROOT + r'\stackC\c_0.tif', ROOT + r'\stackC\c_1.tif']}

print('loading Helicon references...')
refs = {}
for name, tag, folder, _m in CASES:
    refs[name] = load(OUT + '\\' + tag)
print('done')

print('\n%-16s %6s %8s %7s %7s' % ('case', 'range', 'MAE', 'MAE%', '<=6'))
rows = []
for name, tag, folder, m in CASES:
    ref = refs[name]
    stack = [load(p) for p in STACKS[folder]]
    if m == 'B':
        mine = repro_B(stack, ref.shape)
    elif m == 'A':
        mine = repro_A(stack, ref.shape)
    else:
        mine = repro_C(stack, ref.shape)
    if mine.shape != ref.shape:
        print('  !! shape mismatch', name, mine.shape, ref.shape)
    mm, rr = interior(mine), interior(ref)
    e = float(np.abs(mm - rr).mean()); rng = rr.max()
    p6 = float((np.abs(mm - rr) <= 6).mean()) * 100
    p12 = float((np.abs(mm - rr) <= 12).mean()) * 100
    print('%-16s %6.0f %8.2f %6.1f%% %6.1f%%   <=12:%5.1f%%' % (name, rng, e, e/rng*100, p6, p12))
    rows.append((name, mine, ref))

# diff figures (correctly normalized)
def diff_fig(mine, ref, path):
    m, r = interior(mine), interior(ref)
    h, w = m.shape
    d = np.abs(m - r)
    dn = np.clip(d / max(1.0, np.percentile(d, 99)) * 255, 0, 255)
    canvas = np.zeros((h, w*3+8))
    canvas[:, :w] = m / max(1.0, r.max()) * 255
    canvas[:, w+4:2*w+4] = r / max(1.0, r.max()) * 255
    canvas[:, 2*w+8:] = dn
    Image.fromarray(np.clip(canvas, 0, 255).astype(np.uint8)).save(path)

for (name, mine, ref), suffix in zip(rows, ['B', 'A', 'C', 'BC', 'CC', 'BA', 'CA']):
    diff_fig(mine, ref, RES + rf'\final_{suffix}.png')
print('\nfigures ->', RES)
