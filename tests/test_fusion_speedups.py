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
