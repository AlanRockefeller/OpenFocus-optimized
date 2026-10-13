"""Structural diagnosis: B/C on stackC and stackA — where do we differ from Helicon?"""
import sys, os, subprocess, glob, time
sys.path.insert(0, os.path.dirname(__file__))
import numpy as np
from PIL import Image
from helicon_repro import method_b_depth, smooth_depth, method_b_composite, method_c

ROOT = r'F:\Working\helicon-rea\tests'
OUT = ROOT + r'\out'
RES = os.path.join(os.path.dirname(__file__), 'results')
HF = r'D:\Helicon Focus 8\_HeliconFocus.exe'
def load(p): return np.asarray(Image.open(p), dtype=np.float64)

def hf_run(tag, args, folder, want_dmap=False):
    before = set(glob.glob(OUT + '\\*_dmap.tiff'))
    p = subprocess.run([HF, '-silent', '-save:' + OUT + '\\' + tag] + args + [folder],
                       capture_output=True, timeout=600)
    time.sleep(0.6)
    dmap = None
    if want_dmap:
        new = set(glob.glob(OUT + '\\*_dmap.tiff')) - before
        for g in new:
            dst = os.path.join(RES, 'dmap_' + tag)
            os.replace(g, dst)
            dmap = dst
    return OUT + '\\' + tag, dmap

srcC = [load(ROOT + r'\stackC\c_0.tif'), load(ROOT + r'\stackC\c_1.tif')]
srcA = [load(ROOT + r'\stackA\a_0.tif'), load(ROOT + r'\stackA\a_1.tif')]

# B on stackC with dmap
outC, dC = hf_run('refB_stackC.tif', ['-mp:1', '-rp:3', '-sp:3', '-dmap'], ROOT + r'\stackC', True)
refC = load(outC)
print('B stackC: ref DC=%.1f std=%.1f | c0 DC=%.1f | c1 DC=%.1f'
      % (refC.mean(), refC.std(), srcC[0].mean(), srcC[1].mean()))
if dC:
    dm = load(dC)
    print('  helicon dmap: med=%.0f  hist=' % np.median(dm),
          (np.histogram(dm, bins=4, range=(0, 65536))[0] / dm.size * 100).round(0).tolist())
    idx, _ = method_b_depth(srcC, radius=2)
    print('  my idx hist  :', (np.bincount(idx.ravel(), minlength=2) / idx.size * 100).round(1).tolist())

# C on stackA with dmap (float depth)
outA, dA = hf_run('refC_stackA2.tif', ['-mp:3', '-rp:3', '-sp:3', '-dmap'], ROOT + r'\stackA', True)
rC2 = load(outA)
print('\nC stackA: ref DC=%.1f std=%.1f' % (rC2.mean(), rC2.std()))
if dA:
    dm = load(dA)
    print('  helicon C dmap (continuous): min=%.0f max=%.0f mean=%.0f' % (dm.min(), dm.max(), dm.mean()))
    frac1 = dm / dm.max()
    print('  frac>0.25:', float((frac1 > 0.25).mean()).__round__(3), ' frac<0.75:', float((frac1 < 0.75).mean()).__round__(3))

# my C with fallback
def crop_to(a, shape):
    hh, ww = shape; y0, x0 = (a.shape[0]-hh)//2, (a.shape[1]-ww)//2
    return a[y0:y0+hh, x0:x0+ww]
mineA = method_c([crop_to(f, rC2.shape) for f in srcA], levels=3, wsigma=3.0, mode='softnorm')
m, r = mineA[6:-6, 6:-6], rC2[6:-6, 6:-6]
print('  my C stackA (with fallback): MAE=%.2f  my DC=%.1f  within±6=%.1f%%'
      % (np.abs(m - r).mean(), m.mean(), (np.abs(m - r) <= 6).mean() * 100))
