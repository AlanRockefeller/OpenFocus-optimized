"""Tile grids give every pixel weight, and fuse each tile once."""
import numpy as np
import pytest

from core import multi_focus_fusion as mff
from core.source_frames import resolve_frame
from core.multi_focus_fusion import (MultiFocusFusion, _ai_effective_tiling,
                                     _tile_axis_starts, is_ai_fusion_available)

V1_WEIGHTS = 'weights/stackmff_final.onnx'
V3_WEIGHTS = 'weights/stackmff_v3_final.onnx'


def _listed_starts(length, block, overlap, grid=1):
    """Tile origins as listed before repeated origins were removed."""
    grid = max(1, int(grid))
    if block < grid or (overlap and block - grid < 2 and length > block):
        grid = 1
    step = max(1, block - overlap)
    if grid > 1:
        step = max(grid, (step // grid) * grid)
    last = ((length - block + grid - 1) // grid) * grid
    return [min(start, last) for start in range(0, length, step)]


@pytest.mark.parametrize('length, expected', [
    (1600, [0, 576]),
    (2400, [0, 768, 1376]),
    (4000, [0, 768, 1536, 2304, 2976]),
    (3000, [0, 768, 1536, 1976]),
    (6000, [0, 768, 1536, 2304, 3072, 3840, 4608, 4976]),
])
def test_tile_origins_are_listed_once(length, expected):
    assert _tile_axis_starts(length, 1024, 256) == expected


def test_tile_origins_keep_every_distinct_origin_in_order():
    for length in (1, 37, 64, 65, 129, 200, 1100, 1600, 2049, 4000):
        for block in (16, 64, 65, 128, 1024):
            block = min(block, length)
            for overlap in (0, 1, 2, 3, 32, block // 2):
                for grid in (1, 64):
                    starts = _tile_axis_starts(length, block, overlap, grid)
                    assert starts == sorted(set(_listed_starts(length, block, overlap, grid)))


def _stack(h, w, count=3):
    rng = np.random.default_rng(11)
    return [np.clip(180 + rng.integers(-30, 30, (h, w, 3)), 0, 255).astype(np.uint8)
            for _ in range(count)]


def test_cpu_tiled_fusion_fuses_each_tile_once(monkeypatch):
    fusion = MultiFocusFusion('guided_filter')
    calls = []
    original = MultiFocusFusion._fuse_guided_filter

    def counting(self, *args, **kwargs):
        calls.append(1)
        return original(self, *args, **kwargs)

    monkeypatch.setattr(MultiFocusFusion, '_fuse_guided_filter', counting)
    fusion._fuse_tiled(_stack(40, 60), algorithm='guided_filter', block_size=24,
                       overlap=12, thread_count=1, **fusion.fusion_parameters)
    xs, ys = _tile_axis_starts(60, 24, 12), _tile_axis_starts(40, 24, 12)
    assert (xs, ys) == ([0, 12, 24, 36], [0, 12, 16])
    assert len(calls) == len(xs) * len(ys) == 12


def _record_weights(monkeypatch, h, w):
    total = np.zeros((h, w), np.float64)
    original = MultiFocusFusion._compute_tile_weights

    def recording(self, x0, y0, x1, y1, fw, fh, width, height, overlap):
        weights = original(self, x0, y0, x1, y1, fw, fh, width, height, overlap)
        total[y0:y0 + fh, x0:x0 + fw] += weights
        return weights

    monkeypatch.setattr(MultiFocusFusion, '_compute_tile_weights', recording)
    return total


@pytest.mark.parametrize('h, w, block', [(37, 53, 20), (129, 130, 64), (66, 131, 65)])
@pytest.mark.parametrize('overlap', [0, 1, 2, 3])
def test_cpu_tile_grid_gives_every_pixel_weight(monkeypatch, h, w, block, overlap):
    total = _record_weights(monkeypatch, h, w)
    fusion = MultiFocusFusion('guided_filter')
    out = fusion._fuse_tiled(_stack(h, w), algorithm='guided_filter', block_size=block,
                             overlap=overlap, thread_count=1, **fusion.fusion_parameters)
    assert (total > 0).all()
    assert out.min() > 0


def test_cpu_tile_overlap_of_one_renders_like_no_overlap():
    fusion = MultiFocusFusion('guided_filter')
    frames = _stack(37, 53)
    renders = [fusion._fuse_tiled(frames, algorithm='guided_filter', block_size=20,
                                  overlap=overlap, thread_count=1, **fusion.fusion_parameters)
               for overlap in (0, 1)]
    np.testing.assert_array_equal(renders[1], renders[0])


def _neural(monkeypatch, frames, block, overlap, *, algorithm='stackmffv3', mode='soft',
            min_tile_edge=None, predict=None):
    h, w = frames[0].shape[:2]
    fusion = MultiFocusFusion(algorithm, use_gpu=False, exposure_match=False)
    shapes = []

    def first_frame(_fusion, crops, _resize, **_kwargs):
        shapes.append(crops[0].shape[:2])
        # Streamed tiles hand the runner lazy crops; a runner reads their pixels.
        return resolve_frame(crops[0])

    runner, resolve = fusion.descriptor.runner, mff.resolve_runner
    monkeypatch.setattr(mff, 'resolve_runner', lambda name: (
        predict or first_frame) if name == runner else resolve(name))
    extra = {} if min_tile_edge is None else {'min_tile_edge': min_tile_edge}
    out = fusion._fuse_tiled_neural(
        frames, None, h, w, 3, block, overlap, label='test',
        default_weights=V3_WEIGHTS, mode_normalizer=lambda value: value, mode=mode, **extra)
    return out, shapes


@pytest.mark.parametrize('size, block', [(129, 65), (200, 65), (129, 64)])
@pytest.mark.parametrize('overlap', [0, 1, 2, 3])
def test_neural_tile_grid_gives_every_pixel_weight(monkeypatch, size, block, overlap):
    frames = [np.full((size, size, 3), value, np.uint8) for value in (100, 200)]
    total = _record_weights(monkeypatch, size, size)
    out, _shapes = _neural(monkeypatch, frames, block, overlap)
    assert (total > 0).all()
    assert (out == 100).all()


def test_v1_tile_grid_has_no_one_pixel_edge_tile(monkeypatch):
    frames = [np.full((129, 100, 3), value, np.uint8) for value in (100, 200)]
    out, shapes = _neural(monkeypatch, frames, 64, 0, algorithm='stackmff', min_tile_edge=2)
    assert min(min(shape) for shape in shapes) >= 2
    assert (out == 100).all()
    _block, _batch, coords = _ai_effective_tiling(2, 64, 129, 100, 0, 1, min_tile_edge=2)
    assert sorted({y for _x, y, _x1, _y1 in coords}) == [0, 64, 65]
    assert sorted({x for x, _y, _x1, _y1 in coords}) == [0, 64]


def test_moved_edge_tile_soft_blend_and_hard_pick(monkeypatch):
    rng = np.random.default_rng(5)
    a, b = (rng.integers(0, 256, (129, 100, 3)).astype(np.uint8) for _ in range(2))
    frames = [a, b]

    def render(mode):
        # Tiles in raster order: (0,0) (64,0) (0,64) (64,64) (0,65) (64,65).
        # The two moved tiles on the last row pick the other frame.
        choices = iter([0, 1, 0, 1, 1, 0])

        def predict(_fusion, crops, _resize, **_kwargs):
            return resolve_frame(crops[next(choices)])

        out, shapes = _neural(monkeypatch, frames, 64, 0, mode=mode, min_tile_edge=2,
                              predict=predict)
        assert shapes == []
        return out

    soft, hard = render('soft'), render('hard')
    expected = np.empty_like(a)
    expected[:, :64], expected[:, 64:] = a[:, :64], b[:, 64:]
    expected[128:, :64], expected[128:, 64:] = b[128:, :64], a[128:, 64:]
    # Equal weights: hard pick keeps the tile that covered a pixel first.
    np.testing.assert_array_equal(hard, expected)
    # The soft blend averages the two tiles on rows 65..127.
    mean = np.rint((a.astype(np.float32) + b) / 2).astype(np.uint8)
    expected[65:128] = mean[65:128]
    np.testing.assert_array_equal(soft, expected)


@pytest.mark.parametrize('size', [64, 128, 129, 130, 193, 320])
@pytest.mark.parametrize('block', [64, 65, 128])
@pytest.mark.parametrize('overlap', [0, 2, 32])
def test_minimum_edge_keeps_grids_without_a_thin_tile(size, block, overlap):
    default = _ai_effective_tiling(3, block, size, size, overlap, 1)
    widened = _ai_effective_tiling(3, block, size, size, overlap, 1, min_tile_edge=2)
    edges = [min(x1 - x0, y1 - y0) for x0, y0, x1, y1 in default[2]]
    if min(edges) >= 2:
        assert widened == default
    else:
        assert min(min(x1 - x0, y1 - y0) for x0, y0, x1, y1 in widened[2]) >= 2
    covered = np.zeros((size, size), bool)
    for x0, y0, x1, y1 in widened[2]:
        assert x1 - x0 <= widened[0] and y1 - y0 <= widened[0]
        covered[y0:y1, x0:x1] = True
    assert covered.all()


def test_v3_grid_keeps_its_aligned_edge_tile():
    _block, _batch, coords = _ai_effective_tiling(3, 64, 129, 100, 0, 1)
    assert sorted({y for _x, y, _x1, _y1 in coords}) == [0, 64, 128]


def test_v2_tile_overlap_of_one_renders_like_no_overlap(monkeypatch):
    from types import SimpleNamespace
    from core.frame_storage import scratch_directory
    from fusion_methods import neural_runtime, stackmff_v2_tiled as tiled

    def logits(planes, *_args, **_kwargs):
        out = np.zeros_like(planes)
        out[:, 1] = 3.0
        out[:, 2] = planes[:, 0] * 6.0
        return out

    monkeypatch.setattr(neural_runtime, 'logits', logits)
    plan = SimpleNamespace(native_threads=1)
    monkeypatch.setattr(tiled, '_plan', lambda *a, **k: (64, 'unused.onnx', 'CPU', plan))
    ramp = np.linspace(0, 255, 128, dtype=np.float32).round().astype(np.uint8)
    first = np.repeat(np.repeat(ramp[None, :, None], 128, 0), 3, 2)
    frames = [first, np.full_like(first, 80), np.full_like(first, 160)]
    grids = []
    results = {}
    with scratch_directory():
        for overlap in (0, 1):
            source = tiled._Source(frames, None, None, None)
            depth, index = tiled._tiled_depth_under_lease(
                source, 'unused.onnx', False, tile_overlap=overlap,
                tile_grid_callback=lambda *grid: grids.append(grid))
            results[overlap] = (depth.copy(), index.copy())
    np.testing.assert_array_equal(results[1][0], results[0][0])
    np.testing.assert_array_equal(results[1][1], results[0][1])
    assert grids == [(64, 0, 4), (64, 0, 4)]


@pytest.mark.skipif(not is_ai_fusion_available(), reason="ONNX Runtime not installed")
def test_v1_renders_a_dimension_one_pixel_past_the_tile_grid():
    rng = np.random.default_rng(3)
    base = rng.integers(40, 220, (129, 100, 3)).astype(np.uint8)
    frames = [base, np.clip(base.astype(np.int16) + 12, 0, 255).astype(np.uint8),
              np.clip(base.astype(np.int16) - 12, 0, 255).astype(np.uint8)]
    fusion = MultiFocusFusion(algorithm='stackmff', use_gpu=False, tile_enabled=True,
                              tile_block_size=64, tile_overlap=0, tile_threshold=32)
    out = fusion.fuse(input_source=frames, img_resize=None, model_path=V1_WEIGHTS,
                      thread_count=1)
    assert out.shape == base.shape
