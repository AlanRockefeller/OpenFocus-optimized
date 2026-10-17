"""System identification of Helicon's focus measure (calc_diff):
P1 pre-blur discriminator, P2 window footprint vs rp, P3 channel aggregation.
Each probe is designed so candidate measures DISAGREE; Helicon's dmap votes."""
import sys, os, subprocess, glob, time
sys.path.insert(0, os.path.dirname(__file__))
import numpy as np
from PIL import Image

ROOT = r'F:\Working\helicon-rea\tests'
OUT = ROOT + r'\out'
RES = r'F:\Working\OpenFocus\tools\helicon_repro\results'
HF = r'D:\Helicon Focus 8\_HeliconFocus.exe'
os.makedirs(RES, exist_ok=True)

def gauss1d(a, sigma):
    if sigma <= 0: return a.astype(np.float64)
    r = max(1, int(3*sigma))
    k = np.exp(-0.5*(np.arange(-r, r+1)/sigma)**2); k /= k.sum()
    p = np.pad(a, ((r, r), (0, 0)), mode='edge')
    a = sum(k[i]*p[i:i+a.shape[0]] for i in range(2*r+1))
    p = np.pad(a, ((0, 0), (r, r)), mode='edge')
    return sum(k[i]*p[:, i:i+a.shape[1]] for i in range(2*r+1))

def box(a, r):
    if r <= 0: return a.astype(np.float64)
    k = 2*r+1
    p = np.pad(a, ((r, r), (0, 0)), mode='edge'); c = np.cumsum(p, 0)
    a = (c[k-1:] - np.concatenate([np.zeros((1,)+a.shape[1:]), c[:-k]]))/k
    p = np.pad(a, ((0, 0), (r, r)), mode='edge'); c = np.cumsum(p, 1)
    return (c[:, k-1:] - np.concatenate([np.zeros((a.shape[0], 1)), c[:, :-k]], 1))/k

def shift_bilinear(a, d):
    x0 = int(np.floor(d)); f = d - x0
    lo = np.roll(a, x0, axis=1); hi = np.roll(a, x0+1, axis=1)
    return (1-f)*lo + f*hi

def save_rgb(a, path): Image.fromarray(np.clip(a, 0, 255).astype(np.uint8)).save(path)

rng = np.random.default_rng(11)
H, W = 384, 512

# ---------------- P1: pre-blur discriminator ----------------
# 5 strips x ~100px. texture = white noise sigma22.
# frames: sharp | blur(sigma1.2 -> var~0.056*s2) | shifted d=0.25/0.5/0.75 (var 0.625/0.5/0.625 s2)
# raw-variance measure: S2 (d=0.5) -> blur wins.  preblurred(>=0.5) measure: shifted wins everywhere.
T = 128 + rng.normal(0, 22, (H, W))
Tb = gauss1d(T, 1.2)
def strip_replace(img, x0, x1, content):
    out = img.copy(); out[:, x0:x1] = content[:, x0:x1]; return out
f_sharp = np.repeat(T[..., None], 3, axis=2)
f_blur  = np.repeat(Tb[..., None], 3, axis=2)
frames1 = [f_sharp, f_blur]
deltas = [(100, 200, 0.25), (200, 300, 0.5), (300, 400, 0.75)]
for x0, x1, d in deltas:
    shifted = shift_bilinear(T, d)
    img = np.repeat(Tb[..., None], 3, axis=2)          # blur elsewhere
    img[:, x0:x1] = shifted[:, x0:x1, None]            # shifted texture in this strip
    frames1.append(img)
d1 = os.path.join(ROOT, 'probeP1'); os.makedirs(d1, exist_ok=True)
for i, f in enumerate(frames1): save_rgb(f, os.path.join(d1, 'p1_%d.jpg' % i))

# ---------------- P2: window footprint (bleed) vs rp ----------------
# fA: noise disk r=40 at (192,256) sigma30 + background sigma5.  fB: background only.
# winner outside the disk flips at ~window radius + c  ->  bleed(rp) pins the mapping.
bg = 128 + rng.normal(0, 5, (H, W))
disk = np.zeros((H, W))
yy, xx = np.mgrid[0:H, 0:W]
mask = (yy-192)**2 + (xx-256)**2 <= 40**2
fA = bg.copy(); fA[mask] = 128 + rng.normal(0, 30, int(mask.sum()))
fB = bg.copy()
frames2 = [np.repeat(fA[..., None], 3, 2), np.repeat(fB[..., None], 3, 2)]
d2 = os.path.join(ROOT, 'probeP2'); os.makedirs(d2, exist_ok=True)
for i, f in enumerate(frames2): save_rgb(f, os.path.join(d2, 'p2_%d.jpg' % i))

# ---------------- P3: channel aggregation ----------------
# f0: R-channel noise sigma60 (G,B flat).  f1: G-channel noise sigma50.
# SUM measure: 3600 > 2500 -> f0.   LUMA(0.299,0.587,0.114): 322 < 861 -> f1.
f0 = np.full((H, W, 3), 128.0); f0[..., 0] = 128 + rng.normal(0, 60, (H, W))
f1 = np.full((H, W, 3), 128.0); f1[..., 1] = 128 + rng.normal(0, 50, (H, W))
frames3 = [f0, f1]
d3 = os.path.join(ROOT, 'probeP3'); os.makedirs(d3, exist_ok=True)
for i, f in enumerate(frames3): save_rgb(f, os.path.join(d3, 'p3_%d.jpg' % i))

# ---------------- run Helicon ----------------
def hf(stack_dir, tag, rp=3, sp=3, dmap=True):
    before = set(glob.glob(OUT + '\\*_dmap.tiff'))
    out = OUT + '\\' + tag + '.tif'
    p = subprocess.run([HF, '-silent', '-save:' + out, '-mp:1', '-rp:%d' % rp,
                        '-sp:%d' % sp, '-j:100'] + (['-dmap'] if dmap else []) + [stack_dir],
                       capture_output=True, timeout=600)
    dm = None
    if dmap:
        t0 = time.time()
        while time.time() - t0 < 25:
            new = set(glob.glob(OUT + '\\*_dmap.tiff')) - before
            if new:
                g = sorted(new)[0]
                s1 = os.path.getsize(g); time.sleep(0.3); s2 = os.path.getsize(g)
                if s1 == s2:
                    dst = os.path.join(RES, 'dmap_' + tag + '.tif')
                    os.replace(g, dst); dm = dst
                    break
            time.sleep(0.3)
    return out, dm, p.returncode

if __name__ == '__main__':
    o, dm, rc = hf(d1, 'probeP1_r3')
    print('P1', rc, dm)
    for rp in (1, 3, 8):
        o, dm, rc = hf(d2, 'probeP2_r%d' % rp)
        print('P2 rp=%d' % rp, rc, dm)
    o, dm, rc = hf(d3, 'probeP3_r3')
    print('P3', rc, dm)
