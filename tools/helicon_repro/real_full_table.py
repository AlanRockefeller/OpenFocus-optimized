"""Run all 11 Mobile Depth stacks + diagnostics (depth agreement, alignment shifts, diff map)."""
import sys, os, glob, time
sys.path.insert(0, os.path.dirname(__file__))
import numpy as np
from PIL import Image
from real_compare import (load_rgb, repro_B_color, metrics, DATA, OUT, RES)
from helicon_repro import smooth_depth

def box_blur(a, r):
    if r <= 0: return a.astype(np.float64)
    k = 2*r+1
    p = np.pad(a, ((r,r),(0,0)), mode='edge'); c = np.cumsum(p, 0)
    a = (c[k-1:] - np.concatenate([np.zeros((1,)+a.shape[1:]), c[:-k]]))/k
    p = np.pad(a, ((0,0),(r,r)), mode='edge'); c = np.cumsum(p, 1)
    return (c[:, k-1:] - np.concatenate([np.zeros((a.shape[0],1)), c[:, :-k]], 1))/k

def depth_agreement(frames, dmap_path, radius=3, sp=3):
    from real_compare import align_stack_subpix
    al = align_stack_subpix(frames)
    Elist = []
    for f in al:
        e = np.zeros(f.shape[:2])
        for c in range(3):
            m = box_blur(f[..., c], radius)
            e += box_blur((f[..., c]-m)**2, radius)
        Elist.append(e)
    idx = np.stack(Elist).argmax(0)
    n = len(frames)
    dm = np.asarray(Image.open(dmap_path), dtype=np.float64)
    if dm.ndim == 3: dm = dm[..., 0]
    step = 65535.0 / (n - 1)
    # continuous depth (0 = last frame side, n-1 = first frame side)
    heli_cont = dm / step
    mine_cont = (n - 1) - smooth_depth(idx.astype(float), n, smoothing=1, feather=1.0)
    dh, dw = heli_cont.shape
    sh, sw = mine_cont.shape
    ys, xs = (sh - dh)//2, (sw - dw)//2
    mine = mine_cont[ys:ys+dh, xs:xs+dw]
    dabs = np.abs(mine - heli_cont)
    within05 = float((dabs <= 0.5).mean())
    within1 = float((dabs <= 1.0).mean())
    return within05, within1, float(dabs.mean()), heli_cont, mine

if __name__ == '__main__':
    all_stacks = ['kitchen', 'balls', 'keyboard', 'bottles', 'bucket', 'fruits',
                  'metals', 'plants', 'telephone', 'window', 'zeromotion']
    print('== full table (radius=2, sp=3) ==')
    print('%-11s %7s %7s %7s %7s %12s %10s' % ('stack', 'MAE', 'MAE%', '<=6', '<=12', 'depth<=0.5', 'mean|dD|'))
    for stack in all_stacks:
        files = sorted(glob.glob(os.path.join(DATA, stack, '*.jpg')))
        frames = np.stack([load_rgb(f) for f in files])
        ref_path = OUT + r'\real_%s_r3_s3.tif' % stack
        if not os.path.exists(ref_path):
            print('%-11s  (no reference)' % stack); continue
        ref = np.asarray(Image.open(ref_path).convert('RGB'), dtype=np.float64)
        mine = repro_B_color(frames, radius=2, sp=3)
        e, ep, p6, p12 = metrics(mine, ref)
        dmp = RES + r'\dmap_real_%s_r3_s3.tif' % stack
        w05 = w1 = md = float('nan')
        if os.path.exists(dmp):
            try:
                w05, w1, md, _, _ = depth_agreement(frames, dmp, radius=2)
            except Exception as ex:
                pass
        print('%-11s %7.2f %6.2f%% %6.1f%% %6.1f%% %11.1f%% %10.2f' % (stack, e, ep, p6, p12, w05*100, md))
