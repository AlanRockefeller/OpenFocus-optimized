"""The retained depth map's alignment mask is measured while fusion resolves frames."""
import cv2
import numpy as np
import pytest

from core import frame_resolution_cache, render_context, workers
from core.registration import RegistrationCancelled
from core.source_frames import ProcessedFrame, SourceFrame, observe_full_resolutions, scan_source


@pytest.fixture(autouse=True)
def _application(qapp, monkeypatch, tmp_path):
    from core import frame_storage
    monkeypatch.setattr(frame_storage, '_storage_directory', str(tmp_path))
    return qapp


def _stack(folder):
    """Five shifted 16-bit frames of one scene whose black parts reach two borders."""
    folder.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(7)
    scene = cv2.GaussianBlur(rng.random((160, 200)).astype(np.float32), (0, 0), 3)
    scene = (scene - scene.min()) / (scene.max() - scene.min())
    scene[:, :40] = 0
    scene[-45:, :] = 0
    frames = []
    for k in range(5):
        crop = scene[10 + 2 * k: 130 + 2 * k, 12 + 3 * k: 162 + 3 * k]
        pixels = (np.stack([crop, crop * 0.9, crop * 0.8], axis=2) * (40000 + 3000 * k)).astype(np.uint16)
        path = folder / f'frame{k}.tif'
        assert cv2.imwrite(str(path), pixels)
        frames.append(scan_source(str(path))[0])
    return frames


def _render(frames, *, collect=True):
    """Render B + ECC + exposure matching; return the mask and the mask step's resolves."""
    resolved = []
    real_mask = render_context.render_alignment_mask
    real_resolve = render_context.resolve_frame
    real_collect = render_context.AlignmentMaskCollector.__call__

    def counting(frame, **options):
        if isinstance(frame, (ProcessedFrame, SourceFrame)):
            resolved.append(frame)
        return real_resolve(frame, **options)

    def tracking_mask(*args, **kwargs):
        render_context.resolve_frame = counting
        try:
            return real_mask(*args, **kwargs)
        finally:
            render_context.resolve_frame = real_resolve
    render_context.render_alignment_mask = tracking_mask
    if not collect:
        render_context.AlignmentMaskCollector.__call__ = lambda self, frame, pixels: None
    try:
        worker = workers.RenderWorker(frames, algorithm='openfocus_b', alignment='ecc',
                                      fusion_parameters={'radius': 8, 'smoothing': 4.0, 'feather': 3},
                                      exposure_match=True, use_gpu=False, thread_count=1)
        results, errors = [], []
        worker.finished_signal.connect(lambda *args: results.append(args))
        worker.error_signal.connect(errors.append)
        worker.run()
    finally:
        render_context.render_alignment_mask = real_mask
        render_context.AlignmentMaskCollector.__call__ = real_collect
    assert not errors and results
    return worker.rendered_depth['valid_mask'], resolved


@pytest.mark.parametrize('cache', [True, False])
def test_fusion_supplies_the_alignment_mask_without_resolving_frames_again(
        tmp_path, monkeypatch, cache):
    if not cache:
        monkeypatch.setattr(frame_resolution_cache.ScopedCache, '_plan', lambda self: None)
    reference, before = _render(_stack(tmp_path / 'a'), collect=False)
    mask, after = _render(_stack(tmp_path / 'b'))
    # The post-fusion pass resolved every frame; fusion now supplies all of them.
    assert len({id(frame) for frame in before}) == 5
    assert after == []
    assert mask is not None and not mask.all() and mask.any()
    np.testing.assert_array_equal(mask, reference)


def test_a_failing_measurement_leaves_that_frame_to_the_post_fusion_pass(tmp_path, monkeypatch):
    reference, _ = _render(_stack(tmp_path / 'a'), collect=False)
    real_call = render_context.AlignmentMaskCollector.__call__
    target = []

    def failing_for_one_frame(self, frame, pixels):
        if not target:
            target.append(frame)
        # Every measurement of the first observed frame fails inside the collector.
        return real_call(self, frame, None if frame is target[0] else pixels)
    monkeypatch.setattr(render_context.AlignmentMaskCollector, '__call__', failing_for_one_frame)
    mask, after = _render(_stack(tmp_path / 'b'))
    assert after == target
    np.testing.assert_array_equal(mask, reference)


def test_observer_sees_only_listed_frames_while_registered():
    pixels = np.arange(12, dtype=np.uint16).reshape(3, 4)
    listed = ProcessedFrame(pixels, lambda image: image + 1, pixels.shape, pixels.dtype)
    other = ProcessedFrame(pixels, lambda image: image + 2, pixels.shape, pixels.dtype)
    seen = []
    with observe_full_resolutions([listed, pixels], lambda frame, image: seen.append((frame, image.copy()))):
        listed.full_resolution()
        other.full_resolution()
    listed.full_resolution()
    assert len(seen) == 1 and seen[0][0] is listed
    np.testing.assert_array_equal(seen[0][1], pixels + 1)


def test_collector_lets_cancellation_reach_the_resolving_pass():
    pixels = np.zeros((4, 4, 3), np.uint16)
    frame = ProcessedFrame(pixels, lambda image: image, pixels.shape, pixels.dtype)
    collector = render_context.AlignmentMaskCollector([frame], should_cancel=lambda: True)
    with observe_full_resolutions([frame], collector), pytest.raises(RegistrationCancelled):
        frame.full_resolution()
    assert collector.seen == set()
