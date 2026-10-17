"""Reproduction of Helicon Focus 8.1 core algorithms (alignment + methods A/B/C).

Evidence-driven reimplementation; validated against the original binary's CLI output.
All functions work on float64 grayscale arrays, 16-bit range (0..65535).
"""
import numpy as np


# ---------------------------------------------------------------- primitives
def box_blur(a, r):
    """separable box (mean) filter along axes 0 and 1, radius r, edge-padded; N-dim safe"""
    if r <= 0:
        return a.astype(np.float64)
    k = 2 * r + 1
    pad = ((r, r), (0, 0)) + ((0, 0),) * (a.ndim - 2)
    p = np.pad(a, pad, mode='edge')
    c = np.cumsum(p, axis=0)
    a = (c[k - 1:] - np.concatenate([np.zeros((1,) + a.shape[1:]), c[:-k]])) / k
    pad = ((0, 0), (r, r)) + ((0, 0),) * (a.ndim - 2)
    p = np.pad(a, pad, mode='edge')
    c = np.cumsum(p, axis=1)
    return (c[:, k - 1:] - np.concatenate([np.zeros((a.shape[0], 1) + a.shape[2:]), c[:, :-k]], axis=1)) / k


def gauss_blur(a, sigma):
    if sigma <= 0:
        return a.astype(np.float64)
    r = max(1, int(3 * sigma))
    k = np.exp(-0.5 * (np.arange(-r, r + 1) / sigma) ** 2)
    k /= k.sum()
    pad = ((r, r), (0, 0)) + ((0, 0),) * (a.ndim - 2)
    a = sum(k[i] * np.pad(a, pad, mode='edge')[i:i + a.shape[0]] for i in range(2 * r + 1))
    pad = ((0, 0), (r, r)) + ((0, 0),) * (a.ndim - 2)
    return sum(k[i] * np.pad(a, pad, mode='edge')[:, i:i + a.shape[1]] for i in range(2 * r + 1))


# ---------------------------------------------------------------- focus measures
def grad_sq(I):
    gx = np.zeros_like(I); gy = np.zeros_like(I)
    gx[:, 1:-1] = I[:, 2:] - I[:, :-2]
    gy[1:-1, :] = I[2:, :] - I[:-2, :]
    return gx * gx + gy * gy


def grad_abs(I):
    gx = np.abs(np.diff(I, axis=1, prepend=I[:, :1]))
    gy = np.abs(np.diff(I, axis=0, prepend=I[:1, :]))
    return gx + gy


def lap_sq(I):
    L = -4.0 * I.copy()
    L[:, 1:] += I[:, :-1]; L[:, :-1] += I[:, 1:]
    L[1:, :] += I[:-1, :]; L[:-1, :] += I[1:, :]
    return L * L


def lap_abs(I):
    L = -4.0 * I.copy()
    L[:, 1:] += I[:, :-1]; L[:, :-1] += I[:, 1:]
    L[1:, :] += I[:-1, :]; L[:-1, :] += I[1:, :]
    return np.abs(L)


def modified_laplacian(I):
    dxx = np.abs(np.diff(I, axis=1, prepend=I[:, :1])) + np.abs(np.diff(I, axis=1, append=I[:, -1:]))
    dyy = np.abs(np.diff(I, axis=0, prepend=I[:1, :])) + np.abs(np.diff(I, axis=0, append=I[-1:, :]))
    return dxx + dyy


def dog(I, s1=1.0, s2=2.0):
    return np.abs(gauss_blur(I, s1) - gauss_blur(I, s2))


def sobel_sq(I):
    kx = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], np.float64)
    ky = kx.T
    p = np.pad(I, 1, mode='edge')
    gx = sum(kx[i, j] * p[i:i + I.shape[0], j:j + I.shape[1]] for i in range(3) for j in range(3))
    gy = sum(ky[i, j] * p[i:i + I.shape[0], j:j + I.shape[1]] for i in range(3) for j in range(3))
    return gx * gx + gy * gy


def _make_dog(s1, s2):
    def f(I):
        return dog(I, s1, s2)
    f.__name__ = f'dog{s1}{s2}'
    return f


def _make_var(r):
    """windowed local variance with box window of radius r (the suspected Method B measure)"""
    def f(I):
        m = box_blur(I, r)
        return box_blur((I - m) ** 2, r)
    f.__name__ = f'var{r}'
    return f


MEASURES = {
    'grad_sq': grad_sq,
    'grad_abs': grad_abs,
    'lap_sq': lap_sq,
    'lap_abs': lap_abs,
    'mod_lap': modified_laplacian,
    'dog': dog,
    'sobel_sq': sobel_sq,
    'dog12': _make_dog(1.0, 2.0),
    'dog13': _make_dog(1.0, 3.0),
    'dog24': _make_dog(2.0, 4.0),
    'dog35': _make_dog(3.0, 5.0),
    'var1': _make_var(1),
    'var2': _make_var(2),
    'var3': _make_var(3),
    'var4': _make_var(4),
    'var6': _make_var(6),
    'var8': _make_var(8),
}


def window_energy(resp, radius, agg='boxmean', pre_sigma=0.0):
    if pre_sigma:
        resp = gauss_blur(resp, pre_sigma)
    if agg == 'boxmean':
        return box_blur(resp, radius)
    return resp


# ---------------------------------------------------------------- alignment
def estimate_translation(ref, mov, max_dy=0.25, max_dx=0.25):
    """Robust FFT cross-correlation of standardized images.
    Returns integer (dy, dx) to apply to `mov` to align to `ref`."""
    h, w = ref.shape
    win = np.outer(np.hanning(h), np.hanning(w))
    a = ref * win; b = mov * win
    a = (a - a.mean()) / (a.std() + 1e-9)
    b = (b - b.mean()) / (b.std() + 1e-9)
    corr = np.fft.irfft2(np.fft.rfft2(a) * np.conj(np.fft.rfft2(b)), s=ref.shape)
    peak = np.unravel_index(np.argmax(corr), corr.shape)
    dy, dx = peak
    if dy > h // 2: dy -= h
    if dx > w // 2: dx -= w
    limy, limx = int(max_dy * max(h, w)), int(max_dx * max(h, w))
    dy = int(np.clip(dy, -limy, limy)); dx = int(np.clip(dx, -limx, limx))
    return dy, dx


def apply_shift(img, dy, dx):
    out = np.zeros_like(img)
    h, w = img.shape
    ys0, ys1 = max(0, dy), min(h, h + dy)
    xs0, xs1 = max(0, dx), min(w, w + dx)
    out[ys0:ys1, xs0:xs1] = img[ys0 - dy:ys1 - dy, xs0 - dx:xs1 - dx]
    return out


def align_stack(stack, max_dy_pct=0.25, max_dx_pct=0.25, margin=2):
    """align frames 1..n to frame 0; returns (aligned stack, common crop)"""
    ref = stack[0]
    aligned = [ref]
    shifts = [(0, 0)]
    for f in stack[1:]:
        dy, dx = estimate_translation(ref, f, max_dy_pct, max_dx_pct)
        aligned.append(apply_shift(f, dy, dx))
        shifts.append((dy, dx))
    ys = max(abs(s[0]) for s in shifts) + margin
    xs = max(abs(s[1]) for s in shifts) + margin
    h, w = ref.shape
    crop = (slice(ys, h - ys), slice(xs, w - xs))
    return np.stack([a[crop] for a in aligned]), crop, shifts


# ---------------------------------------------------------------- Method A
def method_a(stack, radius=3, smoothing=3, measure='var3', pre_sigma=0.0, prior_eps=1.0, base='last', base_sigma=8.0):
    """contrast-weighted average; observed: base (low-freq) comes from the LAST frame."""
    energy = [window_energy(MEASURES[measure](f), radius, pre_sigma=pre_sigma) for f in stack]
    if smoothing:
        energy = [gauss_blur(e, float(smoothing)) for e in energy]   # per-frame weight maps
    W = np.stack(energy)
    S = np.stack(stack)
    num = (W * S).sum(axis=0) + prior_eps * stack[-1]
    den = W.sum(axis=0) + prior_eps
    out = num / den
    if base == 'last':
        out = base_correct(out, stack[-1], base_sigma)
    return out


# ---------------------------------------------------------------- Method B
def method_b_depth(stack, radius=1, measure='grad_sq', pre_sigma=0.0, agg='boxmean'):
    E = np.stack([window_energy(MEASURES[measure](f), radius, agg=agg, pre_sigma=pre_sigma)
                  for f in stack])
    return E.argmax(axis=0), E


def smooth_depth(idx, n_frames, smoothing=0, feather=None):
    """smooth the index map in continuous depth space, then quantize to 1/32 units."""
    d = idx.astype(np.float64)
    if smoothing:
        # blur index map only where neighbors exist (simple full blur is close enough
        # because depth values are per-pixel and transitions are what matter)
        d = box_blur(d, smoothing)
    if feather:
        d = gauss_blur(d, float(feather))
    q = np.floor(d * 32 / max(1, n_frames - 1) + 0.5) / 32.0 * (n_frames - 1)
    return q


def base_correct(out, last, sigma=8.0):
    """Replace the low-frequency base of `out` with that of the LAST frame
    (observed Helicon behaviour: detail comes from fusion, base = last frame)."""
    return out + gauss_blur(last, sigma) - gauss_blur(out, sigma)


def method_b_composite(stack, depth_cont, transition=0.0, base='last', base_sigma=8.0):
    n = len(stack)
    S = np.stack(stack)
    if transition <= 0:
        sel = np.clip(np.rint(depth_cont), 0, n - 1).astype(int)
        yy, xx = np.meshgrid(np.arange(S.shape[1]), np.arange(S.shape[2]), indexing='ij')
        out = S[sel, yy, xx]
    else:
        W = np.clip(1.0 - np.abs(np.arange(n)[:, None, None] - depth_cont[None]) / transition, 0, None)
        tot = W.sum(axis=0)
        out = (W * S).sum(axis=0)
        out = np.where(tot > 0, out / np.maximum(tot, 1e-9), S[-1])
    if base == 'last':
        out = base_correct(out, stack[-1], base_sigma)
    return out


# ---------------------------------------------------------------- Method C
def _pyramid(I, levels):
    g = [I.astype(np.float64)]
    for _ in range(levels - 1):
        b = gauss_blur(g[-1], 1.0)
        g.append(b[::2, ::2])
    lap = [g[k] - np.repeat(np.repeat(g[k + 1], 2, axis=0), 2, axis=1)[:g[k].shape[0], :g[k].shape[1]]
           for k in range(levels - 1)]
    lap.append(g[-1])
    return lap


def _expand(lap, shape):
    up = np.repeat(np.repeat(lap, 2, axis=0), 2, axis=1)[:shape[0], :shape[1]]
    return gauss_blur(up, 1.0)


def method_c(stack, levels=6, wsigma=1.0, mode='softnorm', base='last'):
    """Laplacian-pyramid fusion: per-band energy weights, soft normalized combine.
    Coarsest (base) band is taken from the LAST frame (observed Helicon behaviour)."""
    n = len(stack)
    pyrs = [_pyramid(f, levels) for f in stack]
    out_pyr = []
    for k in range(levels):
        band = np.stack([p[k] for p in pyrs])
        if base == 'last' and k == levels - 1:
            out_pyr.append(band[-1].copy())       # base band: last frame only
            continue
        W = np.abs(band)
        if wsigma:
            W = gauss_blur(W, wsigma)
        if mode == 'softnorm':
            tot = W.sum(axis=0)
            mixed = (W * band).sum(axis=0) / np.maximum(tot, 1e-9)
            flat = tot <= 1e-9
            mixed[flat] = band[-1][flat]          # zero-energy fallback: last frame
        else:  # 'maxpick'
            idx = W.argmax(axis=0)
            yy, xx = np.meshgrid(np.arange(band.shape[1]), np.arange(band.shape[2]), indexing='ij')
            mixed = band[idx, yy, xx]
        out_pyr.append(mixed)
    # reconstruct
    img = out_pyr[-1]
    for k in range(levels - 2, -1, -1):
        img = _expand(img, out_pyr[k].shape) + out_pyr[k]
    return img
