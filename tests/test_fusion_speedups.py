"""The performance rewrites must not change what the algorithms compute.

Each test pins an optimized path to a compact copy of the implementation it
replaced: DTCWT and the guided filter are required to stay bit-identical,
the StackMFF-V4 inference optimization to agree on all but exact near-ties.
"""
import numpy as np
import pytest

cv2 = pytest.importorskip("cv2")


def _textured_stack(n=4, size=(70, 90), seed=0):
    """Frames sharp in different vertical bands, like a real focus stack."""
    rng = np.random.default_rng(seed)
    h, w = size
    scene = rng.integers(0, 256, (h, w, 3)).astype(np.float32)
    frames = []
    for k in range(n):
        blurred = cv2.GaussianBlur(scene, (0, 0), 3)
        frame = blurred.copy()
        lo, hi = k * w // n, (k + 1) * w // n
        frame[:, lo:hi] = scene[:, lo:hi]
        frames.append(np.clip(frame, 0, 255).astype(np.uint8))
    return frames


# --------------------------------------------------------------------------
# DTCWT
# --------------------------------------------------------------------------

def _reference_dtcwt(images, N=4):
    """The pre-optimization DTCWT fusion (scipy rule, all pyramids held)."""
    import dtcwt
    from scipy.ndimage import maximum_filter, convolve

    def fuse_pair(c1, c2):
        footprint = np.ones((3, 3, 1), dtype=bool)
        A1 = maximum_filter(np.abs(c1), footprint=footprint, mode='reflect')
        A2 = maximum_filter(np.abs(c2), footprint=footprint, mode='reflect')
        count = convolve((A1 > A2).astype(np.float32),
                         np.ones((3, 3, 1), dtype=np.float32),
                         mode='constant', cval=0.0)
        return np.where(count > 4.5, c1, c2)

    rgb = [cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
           for img in images]
    transform = dtcwt.Transform2d()
    channels = []
    for c in range(3):
        ts = [transform.forward(img[:, :, c], nlevels=N) for img in rgb]
        lowpass = np.mean(np.stack([t.lowpass for t in ts], axis=0), axis=0)
        highs = []
        for level in range(N):
            fused = ts[0].highpasses[level]
            for t in ts[1:]:
                fused = fuse_pair(fused, t.highpasses[level])
            highs.append(fused)
        channels.append(transform.inverse(dtcwt.Pyramid(lowpass, tuple(highs))))
    out = np.clip(np.stack(channels, axis=-1) * 255.0, 0, 255).astype(np.uint8)
    return cv2.cvtColor(out, cv2.COLOR_RGB2BGR)


@pytest.mark.parametrize("thread_count", [1, 3])
def test_dtcwt_is_bit_identical_to_reference(thread_count):
    pytest.importorskip("dtcwt")
    pytest.importorskip("scipy")
    from fusion_methods.dtcwt import _dtcwt_impl

    frames = _textured_stack(n=5)
    fused = _dtcwt_impl(frames, None, 4, False, thread_count=thread_count)
    assert np.array_equal(fused, _reference_dtcwt(frames))


def test_dtcwt_blocked_filters_are_bit_identical():
    pytest.importorskip("dtcwt")
    import fusion_methods.dtcwt as mod
    import dtcwt.numpy.lowlevel as lowlevel
    from dtcwt.coeffs import biort, qshift

    h0o, _, _, _ = biort('near_sym_a')
    h0a, h0b = qshift('qshift_a')[:2]
    rng = np.random.default_rng(1)
    # 64 rows -> 2048 columns per 1 MB block; 5000 columns spans 3 blocks.
    X = rng.standard_normal((64, 5000))
    for name, args in (("colfilter", (h0o,)), ("coldfilt", (h0b, h0a)),
                       ("colifilt", (h0b, h0a))):
        original = getattr(lowlevel, name)
        blocked = mod._column_blocked(original)
        assert np.array_equal(blocked(X, *args), original(X, *args)), name
        # Transposed (row-filter) views take the same path.
        assert np.array_equal(blocked(X.T[:, :48], *args),
                              original(X.T[:, :48], *args)), name


# --------------------------------------------------------------------------
# Guided filter
# --------------------------------------------------------------------------

def _reference_gff(images, kernel_size=31):
    """The pre-optimization guided-filter fusion (whole stack in memory)."""
    def gf(I, p, r, eps):
        k = (2 * r + 1, 2 * r + 1)
        mean_I = cv2.boxFilter(I, cv2.CV_32F, k)
        mean_p = cv2.boxFilter(p, cv2.CV_32F, k)
        mean_Ip = cv2.boxFilter(I * p, cv2.CV_32F, k)
        cov_Ip = mean_Ip - mean_I * mean_p
        mean_II = cv2.boxFilter(I * I, cv2.CV_32F, k)
        var_I = mean_II - mean_I * mean_I
        a = cov_Ip / (var_I + eps)
        b = mean_p - a * mean_I
        return cv2.boxFilter(a, cv2.CV_32F, k) * I + cv2.boxFilter(b, cv2.CV_32F, k)

    flt = [img.astype(np.float32) / 255.0 for img in images]
    bases = [cv2.blur(f, (kernel_size, kernel_size), borderType=cv2.BORDER_REFLECT)
             for f in flt]
    sal = []
    for f in flt:
        lap = np.abs(cv2.Laplacian(np.sum(f, axis=2), cv2.CV_32F, ksize=1,
                                   borderType=cv2.BORDER_REFLECT))
        sal.append(cv2.GaussianBlur(lap, (0, 0), sigmaX=5, sigmaY=5,
                                    borderType=cv2.BORDER_REFLECT))
    idx = np.argmax(np.stack(sal, axis=0), axis=0)
    h, w = idx.shape
    bn = np.zeros((h, w, 3), np.float32)
    bd = np.zeros((h, w), np.float32)
    dn = np.zeros((h, w, 3), np.float32)
    dd = np.zeros((h, w), np.float32)
    for k, f in enumerate(flt):
        gray = cv2.cvtColor(f, cv2.COLOR_BGR2GRAY)
        mask = (idx == k).astype(np.float32)
        wb = gf(gray, mask, 45, 0.3)
        wd = gf(gray, mask, 7, 10e-6)
        bn += bases[k] * wb[:, :, None]
        bd += wb
        dn += (f - bases[k]) * wd[:, :, None]
        dd += wd
    bd[bd < 1e-6] = 1e-6
    dd[dd < 1e-6] = 1e-6
    fused = bn / bd[:, :, None] + dn / dd[:, :, None]
    return np.clip(fused * 255, 0, 255).astype(np.uint8)


@pytest.mark.parametrize("thread_count", [1, 4])
def test_guided_filter_is_bit_identical_to_reference(thread_count):
    from fusion_methods.gff import gff_impl

    frames = _textured_stack(n=6)
    # A frame that never wins any pixel is skipped by the new code; its
    # weights were exactly zero before, so the output must not move.
    frames.append(cv2.GaussianBlur(frames[0], (0, 0), 8))
    # A near-duplicate frame creates exact saliency ties, which np.argmax
    # resolves to the earliest frame; the streaming argmax must agree.
    twin = frames[1].copy()
    twin[:8, :8] = 0
    frames.append(twin)
    fused = gff_impl(frames, None, kernel_size=31, thread_count=thread_count)
    assert np.array_equal(fused, _reference_gff(frames))


# --------------------------------------------------------------------------
# StackMFF-V4 inference optimization
# --------------------------------------------------------------------------

def test_stackmffv4_inference_optimization_preserves_focus_map():
    torch = pytest.importorskip("torch")
    pytest.importorskip("torchvision")
    import os
    from core.models.stackmffv4_network import StackMFF_V4
    from fusion_methods.stackmffv4 import _optimize_for_inference

    weights = os.path.join(os.path.dirname(os.path.dirname(
        os.path.abspath(__file__))), "weights", "stackmffv4.pth")
    if not os.path.isfile(weights):
        pytest.skip("model weights not present")
    state = torch.load(weights, map_location="cpu", weights_only=True)
    state = {k.replace('module.', ''): v for k, v in state.items()}

    def load():
        model = StackMFF_V4()
        model.load_state_dict(state)
        return model.eval()

    reference = load()
    optimized = _optimize_for_inference(load(), torch.device("cpu"))
    assert not any(isinstance(m, torch.nn.BatchNorm2d) for m in optimized.modules())

    frames = _textured_stack(n=4, size=(128, 256))
    x = torch.from_numpy(np.stack(
        [cv2.cvtColor(f, cv2.COLOR_BGR2GRAY) for f in frames]
    ).astype(np.float32) / 255.0)[None]
    with torch.inference_mode():
        _, ref_idx = reference(x)
        _, opt_idx = optimized(x)
    agreement = (ref_idx == opt_idx).float().mean().item()
    assert agreement >= 0.999, f"focus maps agree on only {agreement:.4%}"
