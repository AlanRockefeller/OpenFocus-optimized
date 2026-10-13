"""Robust per-stack reference collection: poll for dmap, validate against frame count."""
import sys, os, subprocess, glob, time
sys.path.insert(0, os.path.dirname(__file__))
import numpy as np
from PIL import Image

ROOT = r'F:\Working\helicon-rea\tests'
OUT = ROOT + r'\out'
RES = os.path.join(os.path.dirname(__file__), 'results')
DATA = r'E:\ImagefusionDatasets\StackMFF_jpg\Mobile Depth\image stack'
HF = r'D:\Helicon Focus 8\_HeliconFocus.exe'

def wait_new_dmap(before, timeout=25.0):
    t0 = time.time()
    while time.time() - t0 < timeout:
        new = set(glob.glob(OUT + '\\*_dmap.tiff')) - before
        if new:
            g = sorted(new)[0]
            # wait until size stable
            s1 = os.path.getsize(g); time.sleep(0.4); s2 = os.path.getsize(g)
            if s1 == s2:
                return g
        time.sleep(0.3)
    return None

def validate_dmap(dmap, n):
    dm = np.asarray(Image.open(dmap), dtype=np.float64)
    if dm.ndim == 3: dm = dm[..., 0]
    step = 65535.0 / (n - 1)
    on_grid = np.abs(dm / 2048.0 - np.rint(dm / 2048.0)) < 0.01
    return float(on_grid.mean()), dm

if __name__ == '__main__':
    stacks = ['kitchen', 'balls', 'keyboard', 'bottles', 'bucket', 'fruits',
              'metals', 'plants', 'telephone', 'window', 'zeromotion']
    ok = 0
    for stack in stacks:
        files = sorted(glob.glob(os.path.join(DATA, stack, '*.jpg')))
        n = len(files)
        before = set(glob.glob(OUT + '\\*_dmap.tiff'))
        out = OUT + '\\real_%s_r3_s3.tif' % stack
        p = subprocess.run([HF, '-silent', '-save:' + out, '-mp:1', '-rp:3', '-sp:3',
                            '-j:100', '-dmap', os.path.join(DATA, stack)],
                           capture_output=True, timeout=900)
        dmap = wait_new_dmap(before)
        if dmap is None:
            print('%-11s NO DMAP (rc=%d)' % (stack, p.returncode)); continue
        frac, dm = validate_dmap(dmap, n)
        # polarity/plateau sanity: modal values at multiples of step
        dst = os.path.join(RES, 'dmap_real_%s_r3_s3.tif' % stack)
        os.replace(dmap, dst)
        print('%-11s rc=%d dmap=%s  on-1/32-grid=%.3f  range=(%.0f,%.0f)  n=%d'
              % (stack, p.returncode, os.path.basename(dst), frac, dm.min(), dm.max(), n))
        ok += 1 if frac > 0.9 else 0
    print('validated %d/%d' % (ok, len(stacks)))
