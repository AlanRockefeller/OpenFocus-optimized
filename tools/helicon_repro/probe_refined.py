"""Refined probes: P1b pre-blur discriminator (corrected variance levels),
P3b channel aggregation with polarity anchor regions."""
import sys, os, subprocess, glob, time
sys.path.insert(0, os.path.dirname(__file__))
import numpy as np
from PIL import Image
from probe_measure import (ROOT, OUT, RES, HF, gauss1d, shift_bilinear, save_rgb, hf)

H, W = 384, 512

# ---------------- P1b: pre-blur discriminator (corrected) ----------------
# raw measure:          sharp(484) > shifted0.5(242) > blur11(145)  -> sharp wins S2
# preblurred (>=sig1):  sharp->39,   shifted0.5->19,   blur11->132  -> BLUR wins S2
rng = np.random.default_rng(21)
T = 128 + rng.normal(0, 22, (H, W))
Tb = gauss1d(T, 11.0)
f_sharp = np.repeat(T[..., None], 3, 2)
f_blur = np.repeat(Tb[..., None], 3, 2)
frames = [f_sharp, f_blur]
for x0, x1, d in [(100, 200, 0.25), (200, 300, 0.5), (300, 400, 0.75)]:
    img = np.repeat(Tb[..., None], 3, 2)
    img[:, x0:x1] = shift_bilinear(T, d)[:, x0:x1, None]
    frames.append(img)
d1b = os.path.join(ROOT, 'probeP1b'); os.makedirs(d1b, exist_ok=True)
for i, f in enumerate(frames):
    save_rgb(f, os.path.join(d1b, 'p1b_%d.jpg' % i))

# ---------------- P3b: channel aggregation + polarity anchor ----------------
# A (x0-170):   f0 R60+G60  vs f1 G50   -> SUM:f0  LUMA:f0   (anchor)
# B (x170-340): f0 R60      vs f1 G50   -> SUM:f0  LUMA:f1   (discriminator)
# C (x340-512): f0 R100     vs f1 G10   -> SUM:f0  LUMA:f0   (double anchor)
rng = np.random.default_rng(31)
f0 = np.full((H, W, 3), 128.0); f1 = np.full((H, W, 3), 128.0)
f0[:, :170, 0] += rng.normal(0, 60, (H, 170)); f0[:, :170, 1] += rng.normal(0, 60, (H, 170))
f1[:, :170, 1] += rng.normal(0, 50, (H, 170))
f0[:, 170:340, 0] += rng.normal(0, 60, (H, 170))
f1[:, 170:340, 1] += rng.normal(0, 50, (H, 170))
f0[:, 340:, 0] += rng.normal(0, 100, (H, 172))
f1[:, 340:, 1] += rng.normal(0, 10, (H, 172))
d3b = os.path.join(ROOT, 'probeP3b'); os.makedirs(d3b, exist_ok=True)
save_rgb(f0, os.path.join(d3b, 'p3b_0.jpg')); save_rgb(f1, os.path.join(d3b, 'p3b_1.jpg'))

if __name__ == '__main__':
    for d, tag in [(d1b, 'probeP1b'), (d3b, 'probeP3b')]:
        o, dm, rc = hf(d, tag, rp=3, sp=0)
        print(tag, 'rc=%d' % rc, dm)
