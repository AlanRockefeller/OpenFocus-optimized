"""Read the value-probe verdicts: alignment success/topology, sort rule, Bl identity."""
import sys, os, subprocess, glob, time
sys.path.insert(0, os.path.dirname(__file__))
import numpy as np
from PIL import Image
from probe_measure import ROOT, OUT, RES, HF, gauss1d, save_rgb

rng = np.random.default_rng(55)

def load_dm(tag):
    p = RES + r'\dmap_' + tag + '.tif'
    dm = np.asarray(Image.open(p), dtype=np.float64)
    if dm.ndim == 3: dm = dm[..., 0]
    return dm

def load_out(tag):
    return np.asarray(Image.open(OUT + r'\%s.tif' % tag).convert('RGB'), dtype=np.float64)

def center_crop(a, shape):
    h, w = shape
    y0 = (a.shape[0]-h)//2; x0 = (a.shape[1]-w)//2
    return a[y0:y0+h, x0:x0+w]

print('===== C1: opposite shifts +12/-12 -> crop 13/side = reference topology =====')
print('  (chained would give 25/side)')

print('===== C3: 40px shift, crop only 9/side -> aligned or failed? =====')
H, W = 240, 400
tex = 128 + rng.normal(0, 22, (H, W))
# rebuild same frames as probe_value2 (different rng seed there, so compare content instead):
out3 = load_out('probeC3')
f0full = np.asarray(Image.open(ROOT + r'\probeC3\c3_0.jpg').convert('RGB'), dtype=np.float64)
f1full = np.asarray(Image.open(ROOT + r'\probeC3\c3_1.jpg').convert('RGB'), dtype=np.float64)
# for each output pixel: closer to f0 (aligned or selected f0) or to f1-shifted-back?
o = out3
# candidate A: output region == f0 (same coords); candidate B: == f1 rolled (unaligned f1 content)
for name, cand in [('f0 same-coords', f0full), ('f1 unaligned', f1full)]:
    c = center_crop(cand, o.shape[:2])
    d = np.abs(o - c).mean(axis=2)
    print('  |out - %-14s| mean=%.2f' % (name, d.mean()))
# also: is the output sharp? (misalignment would blur/ghost at edges)
def hf(a, s=2.0):
    g = a.mean(axis=2) if a.ndim == 3 else a
    from probe_measure import gauss1d as gb
    return float((g - gb(g, s)).std())
print('  hf: out=%.2f  f0=%.2f  f1=%.2f' % (hf(o), hf(center_crop(f0full, o.shape[:2])), hf(center_crop(f1full, o.shape[:2]))))

print('===== S probes: auto-sort rule via dmap polarity =====')
# scene: 3 disks at x=100,250,400; frame k makes disk order[k] sharp
# file order F: disk0,disk1,disk2 ; R: disk2,disk1,disk0 ; M: disk1,disk0,disk2
def disk_vote(dm_path, tag):
    dm = load_dm(tag)
    n = 3; step = 65535.0/(n-1)
    idx = np.clip((n-1) - np.rint(dm/step).astype(int), 0, n-1)  # my-order decode
    # disk centers in dmap coords (crop ~2/side, keep approx)
    dh, dw = idx.shape
    res = {}
    for name, cx in [('disk0', 100), ('disk1', 250), ('disk2', 400)]:
        x0 = max(cx-40, 5); x1 = min(cx+40, dw-5)
        v, c = np.unique(idx[100:200, x0:x1], return_counts=True)
        res[name] = int(v[np.argmax(c)])
    return res
for tag in ['probeS_F', 'probeS_R', 'probeS_M']:
    print('  %-10s winner-in-disk: %s' % (tag, disk_vote(None, tag)))
print('  expected if NO sort:  F={0,1,2}  R={2,1,0}  M={1,0,2} (file idx per disk)')
print('  expected if SORTED to disk order: F={0,1,2}  R={0,1,2}  M={0,1,2}')

print('===== Bl identity: compare Bl output vs A/B/C on same tiny stack =====')
def run(tag, mp):
    out = OUT + '\\' + tag + '.tif'
    if os.path.exists(out): os.remove(out)
    p = subprocess.run([HF, '-silent', '-save:' + out, '-mp:%d' % mp, '-rp:3', '-sp:0',
                        '-j:100', ROOT + r'\probeBl'], capture_output=True, timeout=120)
    return np.asarray(Image.open(out).convert('RGB'), dtype=np.float64)
bl = load_out('probeBl')
a = run('blA', 0); b = run('blB', 1); c = run('blC', 3)
def mae(x, y): return float(np.abs(center_crop(x, bl.shape[:2]) - center_crop(y, bl.shape[:2])).mean())
print('  |Bl-A|=%.2f  |Bl-B|=%.2f  |Bl-C|=%.2f  |B-C|=%.2f  |A-B|=%.2f  (range 0-255)'
      % (mae(bl, a), mae(bl, b), mae(bl, c), mae(b, c), mae(a, b)))
mix = (b + c) / 2
print('  |Bl - (B+C)/2| = %.2f   |Bl - max(B,C)| = %.2f'
      % (mae(bl, mix), mae(bl, np.maximum(b, c))))
