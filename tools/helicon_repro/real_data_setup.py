"""Real-data validation on StackMFF 'Mobile Depth' stacks.
Runs Helicon CLI for references, then our reproduction, and compares."""
import sys, os, subprocess, glob, time
sys.path.insert(0, os.path.dirname(__file__))
import numpy as np
from PIL import Image

ROOT = r'F:\Working\helicon-rea\tests'
DATA = r'E:\ImagefusionDatasets\StackMFF_jpg\Mobile Depth\image stack'
OUT = ROOT + r'\out'
RES = os.path.join(os.path.dirname(__file__), 'results')
HF = r'D:\Helicon Focus 8\_HeliconFocus.exe'

def load_rgb(p): return np.asarray(Image.open(p).convert('RGB'), dtype=np.float64)

def hf_run(stack_dir, tag, mp, rp, sp, want_dmap=True, timeout=900):
    before = set(glob.glob(OUT + '\\*_dmap.tiff'))
    out = OUT + '\\' + tag + '.tif'
    p = subprocess.run([HF, '-silent', '-save:' + out,
                        '-mp:%d' % mp, '-rp:%d' % rp, '-sp:%d' % sp,
                        '-j:100'] + (['-dmap'] if want_dmap else []) + [stack_dir],
                       capture_output=True, timeout=timeout)
    time.sleep(0.6)
    dmap = None
    if want_dmap:
        new = set(glob.glob(OUT + '\\*_dmap.tiff')) - before
        for g in new:
            dst = os.path.join(RES, 'dmap_' + tag + '.tif')
            os.replace(g, dst); dmap = dst
    return out, dmap, p.returncode

if __name__ == '__main__':
    # survey
    for d in sorted(os.listdir(DATA)):
        fs = sorted(glob.glob(os.path.join(DATA, d, '*')))
        im = Image.open(fs[0])
        print('%-12s n=%3d %s %sx%s' % (d, len(fs), os.path.splitext(fs[0])[1], im.width, im.height))
    # reference runs
    for stack, rp, sp in [('kitchen', 3, 3), ('balls', 3, 3), ('keyboard', 3, 3)]:
        t0 = time.time()
        out, dmap, rc = hf_run(os.path.join(DATA, stack), 'real_%s_r%d_s%d' % (stack, rp, sp), 1, rp, sp)
        img = Image.open(out)
        print('%-10s rc=%d %s  shape=%s  %.1fs  dmap=%s'
              % (stack, rc, os.path.basename(out), img.size, time.time() - t0, 'Y' if dmap else '-'))
