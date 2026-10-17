"""Pin the default alignment shift limit: sweep horizontal shifts, read crop."""
import sys, os, subprocess, glob, time
sys.path.insert(0, os.path.dirname(__file__))
import numpy as np
from PIL import Image
from probe_measure import ROOT, OUT, RES, HF, gauss1d, save_rgb

rng = np.random.default_rng(123)
H, W = 240, 400
tex = 128 + rng.normal(0, 22, (H, W))
f0 = np.repeat(tex[..., None], 3, 2)
f2 = np.repeat(gauss1d(tex, 2.5)[..., None], 3, 2)   # decoy 3rd frame (sharp=f0)

print('%-6s %-14s %-10s %-8s %s' % ('shift', 'out shape', 'crop/side', 'hf(out)', 'aligned?'))
for sh in [0, 4, 8, 12, 16, 20, 24, 32, 40]:
    d = os.path.join(ROOT, 'probeLim%d' % sh); os.makedirs(d, exist_ok=True)
    f1 = np.roll(np.repeat(tex[..., None], 3, 2), sh, axis=1)
    for i, f in enumerate([f0, f1, f2]):
        save_rgb(f, os.path.join(d, 'l_%d.jpg' % i))
    out = OUT + r'\lim%d.tif' % sh
    if os.path.exists(out): os.remove(out)
    p = subprocess.run([HF, '-silent', '-save:' + out, '-mp:1', '-rp:3', '-sp:0', '-j:100', d],
                       capture_output=True, timeout=180)
    a = np.asarray(Image.open(out).convert('RGB'), dtype=np.float64)
    cw = (W - a.shape[1]) // 2; ch = (H - a.shape[0]) // 2
    g = a.mean(axis=2)
    k = np.exp(-0.5*(np.arange(-6,7)/2.0)**2); k /= k.sum()
    hp = g - np.array([[sum(k[i]*g[i:i+H if i+H<=len(g) else len(g), j:j+W]) for j in range(W)] for i in range(0,0)] or [0]) if False else None
    # simple high-frequency energy
    dxy = np.abs(np.diff(g, axis=1)).mean() + np.abs(np.diff(g, axis=0)).mean()
    # aligned? compare against f0 cropped to output size (center)
    y0 = (H - a.shape[0])//2; x0 = (W - a.shape[1])//2
    ref = f0[y0:y0+a.shape[0], x0:x0+a.shape[1]]
    err0 = np.abs(a - ref).mean()
    print('%-6d %-14s %dx%d      %.2f     |out-f0|=%.2f' %
          (sh, str(a.shape), ch, cw, dxy, err0))
    os.system('taskkill /F /IM _HeliconFocus.exe >nul 2>&1')
