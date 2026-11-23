"""An ROI that reaches past the image edge renders only the part inside it."""

from types import SimpleNamespace

import numpy as np
import pytest
from PyQt6.QtCore import QRectF

from core.render_context import _clamp_roi
from core.workers import RenderWorker
from utils.validators import crop_roi

SHAPE = (400, 600, 3)
CASES = [
    ((-100, 10, 300, 50), (0, 10, 200, 50)),
    ((10, -40, 50, 100), (10, 0, 50, 60)),
    ((550, 380, 100, 50), (550, 380, 50, 20)),
    ((-10, -10, 700, 500), (0, 0, 600, 400)),
    ((20, 30, 40, 50), (20, 30, 40, 50)),
]
OUTSIDE = [(650, 10, 100, 50), (10, 400, 50, 50), (-200, 10, 100, 50),
           (10, -60, 50, 50), (10, 10, 0, 50)]


def _worker_rect(rect):
    return RenderWorker._normalize_roi_rect(SimpleNamespace(roi_rect=QRectF(*rect)), SHAPE)


@pytest.mark.parametrize("rect, expected", CASES)
def test_render_paste_and_provenance_use_the_same_region(rect, expected):
    assert _worker_rect(rect) == expected
    assert _clamp_roi(rect, SHAPE[:2]) == expected
    image = np.zeros(SHAPE, np.uint8)
    assert crop_roi([image], rect)[0][0].shape == (expected[3], expected[2], 3)


@pytest.mark.parametrize("rect", OUTSIDE)
def test_region_outside_the_image_is_an_error_not_a_strip(rect):
    with pytest.raises(ValueError):
        _worker_rect(rect)
    with pytest.raises(ValueError):
        _clamp_roi(rect, SHAPE[:2])
    with pytest.raises(ValueError):
        crop_roi([np.zeros(SHAPE, np.uint8)], rect)
