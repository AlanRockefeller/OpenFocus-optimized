"""Isolate variables: (A) TIFF input kills JPEG-decode differences;
(B) pre-aligned stacks make Helicon's alignment a no-op -> pure fusion residual."""
import sys, os, glob, subprocess, time
sys.path.insert(0, os.path.dirname(__file__))
import numpy as np
from PIL import Image
from real_compare import load_rgb, align_stack_subpix, metrics, DATA, OUT, RES

HF = r'D:\Helicon Focus 8\_HeliconFocus.exe'
PREP = r'F:\Working\helicon-rea\tests\prealigned'
os.makedirs(PREP, exist_ok=True)

def shift_bilinear(a, dy, dx):
    """bilinear shift with edge clamp (no wrap)"""
    import math
    y0, x0 = int(math.floor(dy)), int(math.floor(dx))
    fy, fx = dy - y0, dx - x0
    H, W = a.shape[:2]
    def gather(yy, xx):
        yy = np.clip(yy, 0, H-1); xx = np.clip(xx, 0, W-1)
        return a[yy, xx]
    ys0 = np.arange(H) - y0; xs0 = np.arange(W) - x0
    Y0, X0 = np.meshgrid(ys0, xs0, indexing='ij')
    Y1, X1 = Y0 - 1, X0 - 1
    return ((1-fy)*(1-fx)*gather(Y0, X0) + fy*(1-fx)*gather(Y1, X0)
            + (1-fy)*fx*gather(Y0, X1) + fy*fx*gather(Y1, X1))

def align_rgb_full(frames, min_shift=0.05):
    """returns aligned frames (float64) + per-frame shifts (subpixel)"""
    from real_compare import estimate_translation_subpix
    gray = frames.mean(axis=3)
    ref = gray[0]
    out = [frames[0]]; shifts = [(0.0, 0.0)]
    for k in range(1, len(frames)):
        dy, dx = estimate_translation_subpix(ref, gray[k])
        if abs(dy) < min_shift and abs(dx) < min_shift:
            dy = dx = 0.0
        if dy == 0.0 and dx == 0.0:
            out.append(frames[k])
        else:
            moved = np.stack([shift_bilinear(frames[k, ..., c], dy, dx) for c in range(3)], axis=2)
            out.append(moved)
        shifts.append((dy, dx))
    return out, shifts

def save_tiff_stack(frames_u8, folder):
    os.makedirs(folder, exist_ok=True)
    for i, f in enumerate(frames_u8):
        Image.fromarray(f).save(os.path.join(folder, 'f%03d.tif' % i))
    return folder

def hf_render(stack_dir, out_tif, mp=1, rp=3, sp=3, dmap=True, timeout=900):
    before = set(glob.glob(OUT + '\\*_dmap.tiff'))
    p = subprocess.run([HF, '-silent', '-save:' + out_tif, '-mp:%d' % mp,
                        '-rp:%d' % rp, '-sp:%d' % sp, '-j:100']
                       + (['-dmap'] if dmap else []) + [stack_dir],
                       capture_output=True, timeout=timeout)
    time.sleep(0.5)
    dmap = None
    if dmap:
        t0 = time.time()
        while time.time() - t0 < 20:
            new = set(glob.glob(OUT + '\\*_dmap.tiff')) - before
            if new:
                g = sorted(new)[0]
                s1 = os.path.getsize(g); time.sleep(0.3); s2 = os.path.getsize(g)
                if s1 == s2:
                    dst = out_tif.replace('.tif', '_dmap.tif')
                    os.replace(g, dst); dmap = dst
                    break
            time.sleep(0.3)
    return out_tif, dmap, p.returncode

if __name__ == '__main__':
    for stack in ['kitchen', 'keyboard']:
        files = sorted(glob.glob(os.path.join(DATA, stack, '*.jpg')))
        frames = np.stack([load_rgb(f) for f in files])
        aligned, shifts = align_rgb_full(frames)
        print('%s shifts: %s' % (stack, ['(%+.2f,%+.2f)' % s for s in shifts]))
        u8 = [np.clip(f, 0, 255).astype(np.uint8) for f in aligned]
        folder = save_tiff_stack(u8, os.path.join(PREP, stack))
        out, dmap, rc = hf_render(folder, OUT + '\\pre_%s.tif' % stack)
        print('  helicon on pre-aligned TIFF: rc=%d out=%s dmap=%s' % (rc, os.path.basename(out), 'Y' if dmap else '-'))
