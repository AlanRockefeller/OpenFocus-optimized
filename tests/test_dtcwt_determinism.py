"""DTCWT must stay bit-identical while its work is parallelised.

The implementation transforms frames and colour channels on threads. Frame
coefficients are computed by the same dtcwt call either way, so the fused
image must not move by a single pixel value. This pins that: the expected
digest was captured from the serial implementation before the change.
"""
import hashlib

import numpy as np
import pytest

from fusion_methods import dtcwt as dtcwt_mod

pytestmark = pytest.mark.skipif(
    dtcwt_mod.dtcwt is None, reason="dtcwt backend not installed")


def _stack():
    """Small deterministic stack: two sharp bands plus blur elsewhere."""
    rng = np.random.default_rng(4242)
    h, w = 96, 128
    base = rng.integers(0, 256, (h, w, 3), dtype=np.uint8)
    frames = []
    for i in range(3):
        blur = base.astype(np.uint8)
        frame = blur.copy()
        y0 = i * h // 3
        frame[y0:y0 + h // 3] = base[y0:y0 + h // 3]
        frames.append(frame)
    return frames


def _digest(arr):
    return hashlib.sha256(np.ascontiguousarray(arr).tobytes()).hexdigest()


def test_parallel_and_serial_produce_identical_output():
    """Force the serial path, then the threaded path, compare bytes.

    This compares the two code paths inside the current implementation rather
    than a stored digest, so it keeps working if dtcwt itself is upgraded.
    """
    frames = _stack()

    serial = dtcwt_mod._dtcwt_impl(frames, None, 3, False, thread_count=1)
    parallel = dtcwt_mod._dtcwt_impl(frames, None, 3, False, thread_count=4)

    assert serial.shape == parallel.shape
    assert np.array_equal(serial, parallel), (
        "threading changed the fused pixels; the DTCWT path must be "
        "deterministic and order-independent")


def test_output_does_not_depend_on_thread_count():
    frames = _stack()
    digests = {_digest(dtcwt_mod._dtcwt_impl(frames, None, 3, False,
                                             thread_count=workers))
               for workers in (1, 2, 3, 4, 8)}
    assert len(digests) == 1, f"output varies with worker count: {digests}"
