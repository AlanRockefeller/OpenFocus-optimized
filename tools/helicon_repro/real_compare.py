"""Full real-data comparison: reproduction vs Helicon on Mobile Depth stacks."""
import sys, os, time, glob
sys.path.insert(0, os.path.dirname(__file__))
import numpy as np
from PIL import Image
from helicon_repro import smooth_depth, estimate_translation, apply_shift

ROOT = r'F:\Working\helicon-rea\tests'
OUT = ROOT + r'\out'
RES = os.path.join(os.path.dirname(__file__), 'results')
DATA = r'E:\ImagefusionDatasets\StackMFF_jpg\Mobile Depth\image stack'

def load_rgb(p): return np.asarray(Image.open(p).convert('RGB'), dtype=np.float64)
def interior(a, m=4): return a[m:-m, m:-m]

def subpix_shift(a, dy, dx):
    def bilin(dy0, dx0):
        return np.roll(np.roll(a, int(dy0), 0), int(dx0), 1)  # placeholder (unused path)
    import math
    y0, x0 = int(math.floor(dy)), int(math.floor(dx))
    fy, fx = dy - y0, dx - x0
    t = lambda yy, xx: np.roll(np.roll(a, yy, 0), xx, 1)
    return ((1-fy)*(1-fx)*t(y0, x0) + fy*(1-fx)*t(y0+1, x0)
            + (1-fy)*fx*t(y0, x0+1) + fy*fx*t(y0+1, x0+1))

def estimate_translation_subpix(ref, mov, max_frac=0.25):
    h, w = ref.shape
    win = np.outer(np.hanning(h), np.hanning(w))
    a = ref * win; b = mov * win
    a = (a - a.mean()) / (a.std() + 1e-9)
    b = (b - b.mean()) / (b.std() + 1e-9)
    corr = np.fft.irfft2(np.fft.rfft2(a) * np.conj(np.fft.rfft2(b)), s=ref.shape)
    py, px = np.unravel_index(np.argmax(corr), corr.shape)
    def para(c0, cm, cp):
        denom = (cm - 2*c0 + cp)
        return 0.5*(cm - cp)/denom if abs(denom) > 1e-12 else 0.0
    fy = fy_ = 0.0; fx = fx_ = 0.0
    if 0 < py < h-1:
        f = para(corr[py, px], corr[py-1, px], corr[py+1, px])
        if abs(f) <= 1: fy_ = f
    if 0 < px < w-1:
        f = para(corr[py, px], corr[py, px-1], corr[py, px+1])
        if abs(f) <= 1: fx_ = f
    dy, dx = py, px
    if dy > h//2: dy -= h
    if dx > w//2: dx -= w
    lim = int(max_frac * max(h, w))
    dy = int(np.clip(dy, -lim, lim)); dx = int(np.clip(dx, -lim, lim))
    return dy + fy_, dx + fx_

def box_blur(a, r):
    if r <= 0: return a.astype(np.float64)
    k = 2*r+1
    p = np.pad(a, ((r,r),(0,0)), mode='edge'); c = np.cumsum(p, 0)
    a = (c[k-1:] - np.concatenate([np.zeros((1,)+a.shape[1:]), c[:-k]]))/k
    p = np.pad(a, ((0,0),(r,r)), mode='edge'); c = np.cumsum(p, 1)
    return (c[:, k-1:] - np.concatenate([np.zeros((a.shape[0],1)), c[:, :-k]], 1))/k

def gauss1d(a, sigma):
    if sigma <= 0: return a.astype(np.float64)
    r = max(1, int(3*sigma))
    k = np.exp(-0.5*(np.arange(-r,r+1)/sigma)**2); k /= k.sum()
    pad0 = ((r, r), (0, 0)) + ((0, 0),) * (a.ndim - 2)
    a = sum(k[i]*np.pad(a, pad0, mode='edge')[i:i+a.shape[0]] for i in range(2*r+1))
    pad1 = ((0, 0), (r, r)) + ((0, 0),) * (a.ndim - 2)
    return sum(k[i]*np.pad(a, pad1, mode='edge')[:, i:i+a.shape[1]] for i in range(2*r+1))

def base_correct(out, last, sigma=32.0):
    return out + gauss1d(last, sigma) - gauss1d(out, sigma)

def align_stack_subpix(rgb_stack, margin=1):
    gray = rgb_stack.mean(axis=3)
    ref = gray[0]
    out = [rgb_stack[0]]; shifts = [(0.0, 0.0)]
    for k in range(1, len(rgb_stack)):
        dy, dx = estimate_translation_subpix(ref, gray[k])
        if abs(dy) < 0.05 and abs(dx) < 0.05:
            out.append(rgb_stack[k]); shifts.append((0.0, 0.0))
        else:
            moved = np.stack([subpix_shift(rgb_stack[k, ..., c], dy, dx) for c in range(3)], axis=2)
            out.append(moved); shifts.append((dy, dx))
    ys = int(max(abs(s[0]) for s in shifts)) + margin
    xs = int(max(abs(s[1]) for s in shifts)) + margin
    h, w = gray.shape[1:]
    sl = (slice(ys, h-ys), slice(xs, w-xs))
    return np.stack([f[sl] for f in out])

def repro_B_color(frames, radius=3, sp=3, base_sigma=32.0):
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
    d = smooth_depth(idx.astype(float), n, smoothing=1 if sp else 0, feather=1.0 if sp else None)
    sel = np.clip(np.rint(d), 0, n-1).astype(int)
    H, W = sel.shape
    yy, xx = np.meshgrid(np.arange(H), np.arange(W), indexing='ij')
    S = np.stack([crop_arr(f, (H, W)) for f in al])
    out = S[sel, yy, xx]
    return base_correct(out, S[-1], base_sigma)

def crop_arr(a, shape):
    H, W = shape
    return a[:H, :W]

def metrics(mine, ref, m=4):
    hh = min(mine.shape[0], ref.shape[0]); ww = min(mine.shape[1], ref.shape[1])
    a = mine[(mine.shape[0]-hh)//2:(mine.shape[0]-hh)//2+hh, (mine.shape[1]-ww)//2:(mine.shape[1]-ww)//2+ww]
    b = ref[(ref.shape[0]-hh)//2:(ref.shape[0]-hh)//2+hh, (ref.shape[1]-ww)//2:(ref.shape[1]-ww)//2+ww]
    a, b = interior(a, m), interior(b, m)
    e = float(np.abs(a - b).mean()); rng = float(b.max())
    return e, e/max(rng, 1)*100, float((np.abs(a-b) <= 6).mean())*100, float((np.abs(a-b) <= 12).mean())*100

if __name__ == '__main__':
    stacks = sys.argv[1:] or ['kitchen', 'balls', 'keyboard']
    print('%-10s %6s %7s %7s %7s %7s' % ('stack', 'radius', 'MAE', 'MAE%', '<=6', '<=12'))
    for stack in stacks:
        files = sorted(glob.glob(os.path.join(DATA, stack, '*.jpg')))
        frames = np.stack([load_rgb(f) for f in files])
        ref = np.asarray(Image.open(OUT + r'\real_%s_r3_s3.tif' % stack).convert('RGB'), dtype=np.float64)
        best = None
        for radius in [1, 2, 3, 4]:
            t0 = time.time()
            mine = repro_B_color(frames, radius=radius, sp=3)
            e, ep, p6, p12 = metrics(mine, ref)
            print('%-10s r=%d    %6.2f %6.2f%% %6.1f%% %6.1f%%  (%.0fs)'
                  % (stack, radius, e, ep, p6, p12, time.time()-t0))
            if best is None or e < best[0]:
                best = (e, radius)
        print('  -> best radius for %s: %d (MAE %.2f)' % (stack, best[1], best[0]))
