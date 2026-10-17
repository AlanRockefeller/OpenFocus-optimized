"""Collect fresh Helicon reference renders (with distinguishable dmap names), then
fit reproduction parameters and report convergence vs the original binary."""
import numpy as np, subprocess, glob, os, re, sys, time, json
from PIL import Image

sys.path.insert(0, os.path.dirname(__file__))
from helicon_repro import (MEASURES, window_energy, method_a, method_b_depth, smooth_depth,
                           method_b_composite, method_c, gauss_blur, box_blur,
                           align_stack, estimate_translation, apply_shift)

ROOT = r'F:\Working\helicon-rea\tests'
OUT = ROOT + r'\out'
HF = r'D:\Helicon Focus 8\_HeliconFocus.exe'
RES = os.path.join(os.path.dirname(__file__), 'results')
os.makedirs(RES, exist_ok=True)

def load(p): return np.asarray(Image.open(p), dtype=np.float64)
def save16(a, p): Image.fromarray(np.clip(a, 0, 65535).astype(np.uint16)).save(p)

def hf_run(tag, args, want_dmap=False):
    before = set(glob.glob(OUT + '\\*_dmap.tiff'))
    p = subprocess.run([HF, '-silent', '-save:' + OUT + '\\' + tag] + args,
                       capture_output=True, timeout=600)
    time.sleep(0.6)
    dmap = None
    if want_dmap:
        new = set(glob.glob(OUT + '\\*_dmap.tiff')) - before
        for g in new:
            dst = os.path.join(RES, 'dmap_' + tag)
            os.replace(g, dst)
            dmap = dst
    return p.returncode, OUT + '\\' + tag, dmap

# ---------------- phase 1: collect references ----------------
print('== collecting Helicon references ==')
refs = {}
jobs = [
    ('refB_r3_s0.tif', ['-mp:1', '-rp:3', '-sp:0', '-dmap'], True, 'stackB'),
    ('refB_r3_s3.tif', ['-mp:1', '-rp:3', '-sp:3', '-dmap'], True, 'stackB'),
    ('refM3_B.tif',    ['-mp:3', '-rp:3', '-sp:3'], False, 'stackB'),   # is mp:3 == mp:2 ?
    ('refA_r1_s0.tif', ['-mp:0', '-rp:1', '-sp:0', '-dmap'], True, 'stackB'),
    ('refC_r1_s0.tif', ['-mp:3', '-rp:1', '-sp:0', '-dmap'], True, 'stackB'),
    ('refC_stackC.tif', ['-mp:3', '-rp:3', '-sp:3'], False, 'stackC'),   # method C on stackC
    ('refC_stackA.tif', ['-mp:3', '-rp:3', '-sp:3'], False, 'stackA'),   # method C on stackA
    ('refB_stackC.tif', ['-mp:1', '-rp:3', '-sp:3'], False, 'stackC'),   # method B on stackC
]
for tag, args, dm, stack in jobs:
    rc, out, dmap = hf_run(tag, args + [ROOT + '\\' + stack], want_dmap=dm)
    refs[tag] = (out, dmap)
    print(f'  {tag} rc={rc} dmap={"Y" if dmap else "-"}')

# ---------------- helpers ----------------
def interior(a, m=6):
    return a[m:-m, m:-m]

def mae(a, b):  return float(np.abs(a - b).mean())
def p95(a, b):  return float(np.percentile(np.abs(a - b), 95))
def within(a, b, t=256):  # within 1 8-bit level
    return float((np.abs(a - b) <= t).mean())

def best_shift_mae(mine, ref, rng_=4):
    """MAE after finding the best integer shift of `mine` vs `ref` (registration-free compare)"""
    best = (1e18, 0, 0)
    for dy in range(-rng_, rng_ + 1):
        for dx in range(-rng_, rng_ + 1):
            m = np.roll(np.roll(mine, dy, axis=0), dx, axis=1)
            e = float(np.abs(m[rng_:-rng_, rng_:-rng_] - ref[rng_:-rng_, rng_:-rng_]).mean())
            if e < best[0]:
                best = (e, dy, dx)
    return best

def report(name, mine, ref):
    m, r = interior(mine), interior(ref)
    e = mae(m, r)
    bs, dy, dx = best_shift_mae(m, r)
    print(f'  {name:34s} MAE={e:7.2f}  best-shift MAE={bs:6.2f}@(dy={dy},dx={dx})  P95={p95(m,r):8.2f}  within±1lev={within(m,r)*100:5.1f}%')
    return e

# ---------------- phase 2: fit Method B measure against Helicon depth map ----------------
print('\n== Method B: measure family fit vs Helicon depth map (stackB, rp=1, sp=0) ==')
src = [load(ROOT + rf'\stackB\b_{i}.tif') for i in range(3)]
dmap = load(RES + r'\dmap_refB_r3_s0.tif')        # radius 3 map: widest applicability
dmap1 = load(RES + r'\dmap_refB_r3_s0.tif')
n = 3
heli_idx = np.clip((n - 1) - np.rint(dmap / (65535.0 / (n - 1))).astype(int), 0, n - 1)   # decode polarity
# confident pixels only (plateau values)
plateau = (dmap == 65534) | (dmap == 32767) | (dmap == 0)

def my_dmap_idx(measure, radius, pre_sigma, agg):
    r_eff = 0 if measure.startswith('var') else radius   # var measures window internally
    E = np.stack([window_energy(MEASURES[measure](f), r_eff, agg=agg, pre_sigma=pre_sigma) for f in src])
    idx = E.argmax(axis=0)
    # crop my index map to dmap geometry (center crop)
    dh, dw = dmap.shape
    sh, sw = idx.shape
    ys, xs = (sh - dh) // 2, (sw - dw) // 2
    return idx[ys:ys + dh, xs:xs + dw]

best = []
for measure in MEASURES:
    radii = [0] if measure.startswith('var') else [1, 2, 3, 4, 6]
    for radius in radii:
        for pre_sigma in [0.0]:
            for agg in ['boxmean', 'none']:
                try:
                    mi = my_dmap_idx(measure, radius, pre_sigma, agg)
                except Exception:
                    continue
                agree = float((mi[plateau] == heli_idx[plateau]).mean())
                best.append((agree, measure, radius, pre_sigma, agg))
best.sort(reverse=True)
for a, m_, r_, s_, g_ in best[:10]:
    print(f'  agree={a*100:5.1f}%  measure={m_:10s} r={r_} sigma={s_} agg={g_}')

agreeB, mB, rB, sB, gB = best[0]
print(f'\n  chosen: {mB} r={rB} sigma={sB} agg={gB}  (plateau agreement {agreeB*100:.1f}%)')

# per-region breakdown of the winner
regions = {'R1': (slice(14,114), slice(14,104)), 'R2': (slice(14,114), slice(142,242)),
           'R3': (slice(14,114), slice(270,370)), 'R4': (slice(14,114), slice(398,498))}
mi = my_dmap_idx(mB, rB, sB, gB)
for k, (sy, sx) in regions.items():
    sub_p = plateau[sy, sx]
    if sub_p.sum():
        a = float((mi[sy, sx][sub_p] == heli_idx[sy, sx][sub_p]).mean())
        print(f'  region {k}: agreement {a*100:5.1f}%  helicon-hist={np.bincount(heli_idx[sy,sx][sub_p].astype(int), minlength=3)} mine={np.bincount(mi[sy,sx][sub_p].astype(int), minlength=3)}')

# ---------------- phase 3: Method B composite fitting ----------------
print('\n== Method B: composite vs Helicon output ==')
ref_out = load(OUT + r'\refB_r3_s0.tif')
idx, E = method_b_depth(src, radius=rB, measure=mB, pre_sigma=sB, agg=gB)
dh, dw = ref_out.shape
ys, xs = (512 - dw) // 2, 0
def crop_to(a, shape):
    hh, ww = shape
    y0, x0 = (a.shape[0]-hh)//2, (a.shape[1]-ww)//2
    return a[y0:y0+hh, x0:x0+ww]
results = []
for smoothing, feather in [(0, None), (0, 1.0), (1, None), (1, 1.0), (2, None)]:
    d = smooth_depth(crop_to(idx.astype(float), ref_out.shape), n, smoothing=smoothing, feather=feather)
    for t in [0.0, 0.5, 1.0, 2.0]:
        comp = method_b_composite([crop_to(f, ref_out.shape) for f in src], d, transition=t)
        results.append((mae(interior(comp), interior(ref_out)), smoothing, feather, t))
results.sort()
for m_, s_, f_, t_ in results[:6]:
    print(f'  MAE={m_:7.2f}  depth_smoothing={s_} feather={f_} transition={t_}')
mB2, sB2, fB2, tB2 = results[0]
d_final = smooth_depth(crop_to(idx.astype(float), ref_out.shape), n, smoothing=sB2, feather=fB2)
comp_final = method_b_composite([crop_to(f, ref_out.shape) for f in src], d_final, transition=tB2)
report('Method B reproduction', comp_final, ref_out)

# validate on the sp=5 reference (feathering/transition should grow)
ref5 = load(OUT + r'\refB_r3_s3.tif')   # actually r3 s3; compare with matched smoothing
d5 = smooth_depth(crop_to(idx.astype(float), ref5.shape), n, smoothing=sB2, feather=fB2)
comp5 = method_b_composite([crop_to(f, ref5.shape) for f in src], d5, transition=tB2)
report('Method B (r3,s3) reproduction', comp5, ref5)

# ---------------- phase 4: Method A fitting ----------------
print('\n== Method A: weight formula fit ==')
refA_B = load(OUT + r'\B_m0_r3_s3.tif')      # method A on stackB (r3 s3)
refA_A = load(OUT + r'\A_m0.tif')            # method A on stackA (r3 s3)
srcA = [load(ROOT + r'\stackA\a_0.tif'), load(ROOT + r'\stackA\a_1.tif')]
resA = []
for measure in ['var1', 'var2', 'var3', 'var4', 'var6', 'grad_sq', 'dog13']:
    for pre_sigma in [0.0, 1.0]:
        try:
            out = method_a([crop_to(f, refA_B.shape) for f in src], radius=3, smoothing=3,
                           measure=measure, pre_sigma=pre_sigma)
            resA.append((mae(interior(out), interior(refA_B)), measure, pre_sigma))
        except Exception as e:
            print('   A-fit fail', measure, repr(e)[:80])
resA.sort()
for m_, me_, s_ in resA[:5]:
    print(f'  stackB MAE={m_:7.2f}  measure={me_:10s} sigma={s_}')
if not resA:
    raise SystemExit('no Method A candidate succeeded')
mA, meA, sA = resA[0]
outA1 = method_a([crop_to(f, refA_A.shape) for f in srcA], radius=3, smoothing=3, measure=meA, pre_sigma=sA)
report(f'Method A stackA ({meA})', outA1, refA_A)
outA2 = method_a([crop_to(f, refA_B.shape) for f in src], radius=3, smoothing=3, measure=meA, pre_sigma=sA)
report(f'Method A stackB ({meA})', outA2, refA_B)

# flat-tie check
tie = method_a([np.full((64, 64), 100.0), np.full((64, 64), 150.0)], radius=3, smoothing=3,
               measure=meA, pre_sigma=sA)
print(f'  flat-tie output = {tie.mean():.2f} (Helicon: 150.00)')

# ---------------- phase 5: Method C fitting ----------------
print('\n== Method C: pyramid parameter fit ==')
mp3 = load(OUT + r'\refM3_B.tif'); mp2 = load(OUT + r'\B_m2_r3_s3.tif')
if mp3.shape == mp2.shape:
    print(f'  mp:3 vs mp:2 outputs identical? MAE={mae(interior(mp3), interior(mp2)):.3f}')
refC1 = load(OUT + r'\refC_r1_s0.tif')
refC2 = load(OUT + r'\B_m2_r3_s3.tif')      # mp:2 output on stackB (legacy-B or C variant)
resC = []
for levels in [3, 4, 5, 6, 7, 8]:
    for ws in [0.0, 1.0, 2.0, 3.0]:
        for mode in ['softnorm', 'maxpick']:
            try:
                out = method_c([crop_to(f, refC1.shape) for f in src], levels=levels, wsigma=ws, mode=mode)
                resC.append((mae(interior(out), interior(refC1)), levels, ws, mode))
            except Exception:
                pass
resC.sort()
for m_, l_, w_, mo_ in resC[:8]:
    print(f'  MAE={m_:7.2f}  levels={l_} wsigma={w_} mode={mo_}')
mC, lC, wC, moC = resC[0]
outC2 = method_c([crop_to(f, refC2.shape) for f in src], levels=lC, wsigma=wC, mode=moC)
report(f'Method C validate (L={lC},{moC})', outC2, refC2)
cstack = [load(ROOT + r'\stackC\c_0.tif'), load(ROOT + r'\stackC\c_1.tif')]
outC3 = method_c([crop_to(f, load(OUT + r'\refC_stackC.tif').shape) for f in cstack],
                 levels=lC, wsigma=wC, mode=moC)
report('Method C stackC', outC3, load(OUT + r'\refC_stackC.tif'))
outA3 = method_c([crop_to(f, load(OUT + r'\refC_stackA.tif').shape) for f in srcA],
                 levels=lC, wsigma=wC, mode=moC)
report('Method C stackA', outA3, load(OUT + r'\refC_stackA.tif'))

# Method B cross-validation on stackC
refBC = load(OUT + r'\refB_stackC.tif')
idxC, _ = method_b_depth(cstack, radius=2, measure=mB, pre_sigma=0.0)
dC = smooth_depth(crop_to(idxC.astype(float), refBC.shape), 2, smoothing=sB2, feather=fB2)
compC = method_b_composite([crop_to(f, refBC.shape) for f in cstack], dC, transition=tB2)
report('Method B stackC (cross-val)', compC, refBC)

# ---------------- phase 6: alignment reproduction ----------------
print('\n== Alignment reproduction (stackAlign3, shift dx=+12) ==')
al_src = [load(ROOT + r'\stackAlign3\p_0.tif'), load(ROOT + r'\stackAlign3\p_1.tif')]
al, crop, shifts = align_stack(al_src, margin=1)
print(f'  estimated shifts: {shifts}  crop={crop}  my output shape={al.shape[1:]}')
ref_al = load(OUT + r'\AL3_def.tif')
print(f'  Helicon output shape={ref_al.shape}')
idx_al, _ = method_b_depth(list(al), radius=2, measure=mB, pre_sigma=0.0)
comp_al = method_b_composite(list(al), idx_al.astype(float), transition=0.0)
hh = min(comp_al.shape[0], ref_al.shape[0]); ww = min(comp_al.shape[1], ref_al.shape[1])
cc = lambda a: a[(a.shape[0]-hh)//2:(a.shape[0]-hh)//2+hh, (a.shape[1]-ww)//2:(a.shape[1]-ww)//2+ww]
d1 = np.abs(interior(cc(comp_al)) - interior(cc(ref_al))).mean()
inlev = float((np.abs(interior(cc(comp_al)) - interior(cc(ref_al))) <= 256).mean())
print(f'  aligned-pipeline MAE vs Helicon AL3_def = {d1:.2f} (16-bit units)  within±1lev={inlev*100:.1f}%')

print('\nsummary: B-measure=%s r=%d sigma=%s agg=%s | B composite sm=%s feather=%s t=%s | '
      'A measure=%s sigma=%s | C levels=%d ws=%.1f mode=%s' % (mB, rB, sB, gB, sB2, fB2, tB2, meA, sA, lC, wC, moC))

# save diff images for the report
def save_diff(mine, ref, path):
    m, r = interior(mine), interior(ref)
    h, w = m.shape
    canvas = np.zeros((h, w * 3 + 8), np.float64)
    canvas[:, :w] = m; canvas[:, w + 4:2 * w + 4] = r
    d = np.abs(m - r)
    dd = np.clip(d / max(1e-6, np.percentile(d, 99)) * 255, 0, 255)
    canvas[:, 2 * w + 8:] = dd
    Image.fromarray(np.clip(canvas / 256, 0, 255).astype(np.uint8)).save(path)

save_diff(comp_final, ref_out, RES + r'\diff_B.png')
save_diff(outA2, refA_B, RES + r'\diff_A.png')
save_diff(outC2, refC2, RES + r'\diff_C.png')
print('diff images saved to', RES)
