"""Value probes #2: alignment topology (crop oracle), auto-sort rule,
Radius/Smoothing semantics, Bl attempt."""
import sys, os, subprocess, glob, time
sys.path.insert(0, os.path.dirname(__file__))
import numpy as np
from PIL import Image
from probe_measure import ROOT, OUT, RES, HF, gauss1d, shift_bilinear, save_rgb

rng = np.random.default_rng(77)

def make_scene():
    """3 disks (sigma30 noise, r=40) on sigma5 background, 300x500."""
    H, W = 300, 500
    bg = 128 + rng.normal(0, 5, (H, W))
    return H, W, bg

def disk_frame(bg, centers, sharp_set):
    """frame with sharp disks at `centers`; disks in sharp_set get noise, others blurred."""
    H, W = bg.shape
    img = bg.copy()
    yy, xx = np.mgrid[0:H, 0:W]
    for i, cx in enumerate(centers):
        m = (yy - H//2)**2 + (xx - cx)**2 <= 40**2
        noise = rng.normal(0, 30, (H, W)) if i in sharp_set else np.zeros((H, W))
        img[m] = 128 + (noise[m] if i in sharp_set else 0)
    return img

# ---------- alignment topology probes ----------
H, W = 240, 400
tex = 128 + rng.normal(0, 22, (H, W))
texb = gauss1d(tex, 2.5)

# C1: f1 shifted +12, f2 shifted -12 (opposite) -> reference crop=13, chained crop=25
f0 = np.repeat(tex[..., None], 3, 2)
f1 = np.roll(np.repeat(tex[..., None], 3, 2), 12, axis=1)
f2 = np.roll(np.repeat(tex[..., None], 3, 2), -12, axis=1)
dc = os.path.join(ROOT, 'probeC1'); os.makedirs(dc, exist_ok=True)
for i, f in enumerate([f0, f1, f2]): save_rgb(f, os.path.join(dc, 'c1_%d.jpg' % i))

# C2: rotation 1.5 deg (does aligner handle rotation?)
def rotate_about_center(a, deg):
    im = Image.fromarray(np.clip(a, 0, 255).astype(np.uint8))
    return np.asarray(im.rotate(deg, resample=Image.BICUBIC, fillcolor=128), dtype=np.float64)
f1r = np.repeat(rotate_about_center(tex, 1.5)[..., None], 3, 2)
f0g = np.repeat(tex[..., None], 3, 2)
f2g = np.repeat(gauss1d(tex, 2.5)[..., None], 3, 2)
dc = os.path.join(ROOT, 'probeC2'); os.makedirs(dc, exist_ok=True)
for i, f in enumerate([f0g, f1r, f2g]): save_rgb(f, os.path.join(dc, 'c2_%d.jpg' % i))

# C3: big shift 40px (default limit probe)
f1s = np.roll(np.repeat(tex[..., None], 3, 2), 40, axis=1)
dc = os.path.join(ROOT, 'probeC3'); os.makedirs(dc, exist_ok=True)
for i, f in enumerate([f0, f1s]): save_rgb(f, os.path.join(dc, 'c3_%d.jpg' % i))

# ---------- auto-sort probes ----------
centers = [100, 250, 400]
Hs, Ws, bg = make_scene()
def stack_frames(order):
    """frames: disk i sharp in frame i (file order = order of centers)."""
    out = []
    for k in order:
        img = bg.copy()
        yy, xx = np.mgrid[0:Hs, 0:Ws]
        for i, cx in enumerate(centers):
            m = (yy - Hs//2)**2 + (xx - cx)**2 <= 40**2
            if i == k:
                img[m] = 128 + rng.normal(0, 30, int(m.sum()))
            else:
                img[m] = bg[m]
        out.append(np.repeat(img[..., None], 3, 2))
    return out
for tag, order in [('probeS_F', [0, 1, 2]), ('probeS_R', [2, 1, 0]), ('probeS_M', [1, 0, 2])]:
    d = os.path.join(ROOT, tag); os.makedirs(d, exist_ok=True)
    for i, f in enumerate(stack_frames(order)): save_rgb(f, os.path.join(d, 's_%d.jpg' % i))

# ---------- Bl attempt (tiny stack, short timeout) ----------
d = os.path.join(ROOT, 'probeBl'); os.makedirs(d, exist_ok=True)
save_rgb(np.repeat(tex[..., None], 3, 2), os.path.join(d, 'bl_0.jpg'))
save_rgb(np.repeat(texb[..., None], 3, 2), os.path.join(d, 'bl_1.jpg'))

def run(stack_dir, tag, mp=1, rp=3, sp=0, dmap=True, timeout=120):
    before = set(glob.glob(OUT + '\\*_dmap.tiff'))
    out = OUT + '\\' + tag + '.tif'
    try:
        p = subprocess.run([HF, '-silent', '-save:' + out, '-mp:%d' % mp, '-rp:%d' % rp,
                            '-sp:%d' % sp, '-j:100'] + (['-dmap'] if dmap else []) + [stack_dir],
                           capture_output=True, timeout=timeout)
        rc = p.returncode
    except subprocess.TimeoutExpired:
        rc = 'TIMEOUT'
    dm = None
    if dmap and rc == 0:
        t0 = time.time()
        while time.time() - t0 < 15:
            new = set(glob.glob(OUT + '\\*_dmap.tiff')) - before
            if new:
                g = sorted(new)[0]
                s1 = os.path.getsize(g); time.sleep(0.3)
                if s1 == os.path.getsize(g):
                    dm = os.path.join(RES, 'dmap_' + tag + '.tif')
                    os.replace(g, dm); break
            time.sleep(0.3)
    shape = None
    if os.path.exists(out):
        shape = Image.open(out).size
    print('%-14s rc=%-7s out=%s dmap=%s' % (tag, rc, shape, 'Y' if dm else '-'))
    return out, dm, rc

if __name__ == '__main__':
    for tag, d in [('probeC1', os.path.join(ROOT, 'probeC1')),
                   ('probeC2', os.path.join(ROOT, 'probeC2')),
                   ('probeC3', os.path.join(ROOT, 'probeC3')),
                   ('probeS_F', os.path.join(ROOT, 'probeS_F')),
                   ('probeS_R', os.path.join(ROOT, 'probeS_R')),
                   ('probeS_M', os.path.join(ROOT, 'probeS_M'))]:
        run(d, tag)
    run(os.path.join(ROOT, 'probeBl'), 'probeBl', mp=4, timeout=90)
    # cleanup stray HF
    os.system('taskkill /F /IM _HeliconFocus.exe >nul 2>&1')
