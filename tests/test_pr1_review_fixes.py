"""Regression coverage for the actionable findings on optimized PR #1."""

from types import SimpleNamespace

import numpy as np
import pytest


def test_loader_calibration_tracks_decode_scale(tmp_path, monkeypatch):
    cv2 = pytest.importorskip("cv2")
    from core.image_loader import ImageStackLoader
    from utils import scalebar

    for i in range(2):
        assert cv2.imwrite(
            str(tmp_path / f"f{i}.png"),
            np.full((20, 30, 3), i, np.uint8),
        )
    monkeypatch.setattr(scalebar, "detect_px_size_um", lambda _path: 0.4)

    loader = ImageStackLoader()
    ok, _message, images, _names = loader.load_from_folder(
        str(tmp_path), scale_factor=0.5)

    assert ok and images[0].shape[:2] == (10, 15)
    assert loader.px_size_um == pytest.approx(0.8)


def test_detected_calibration_tracks_later_resize():
    from utils.scalebar import effective_px_um

    window = SimpleNamespace(
        source_px_um=0.8,
        scale_um_per_px_manual=0.0,
        base_scale_factor=0.5,
        current_scale_factor=0.25,
    )
    assert effective_px_um(window) == pytest.approx(1.6)

    window.scale_um_per_px_manual = 0.65
    assert effective_px_um(window) == pytest.approx(0.65)


class _FakeFusion:
    created = []
    fuse_kwargs = []

    def __init__(self, **kwargs):
        self.created.append(kwargs)

    def fuse(self, input_source, **kwargs):
        self.fuse_kwargs.append(kwargs)
        return np.asarray(input_source[0]).copy()


def _batch_worker(settings, **kwargs):
    pytest.importorskip("PyQt6")
    from core.workers import BatchWorker

    return BatchWorker([], "same", "", settings, use_gpu=False, **kwargs)


def test_single_folder_batch_uses_nested_order_settings_and_calibration(
        tmp_path, monkeypatch):
    import core.workers as workers
    from utils import scalebar

    _FakeFusion.created.clear()
    _FakeFusion.fuse_kwargs.clear()
    monkeypatch.setattr(workers, "MultiFocusFusion", _FakeFusion)
    monkeypatch.setattr(scalebar, "detect_px_size_um", lambda _path: 0.5)

    settings = {
        "fusion_method": "openfocus_b",
        "fusion_params": {
            "kernel_size": 7,
            "base_frame": "last",
            "sort_mode": "desc",
        },
        "format": "png",
        "scale_bar": {"position": "bottom-right", "color": "auto"},
        "input_scale_factor": 0.5,
    }
    worker = _batch_worker(settings, import_mode="single_folder")
    saved_cfg = []
    monkeypatch.setattr(
        worker, "_save_image",
        lambda _path, _image, _extension, scale_cfg=None:
            saved_cfg.append(scale_cfg) or True,
    )
    frames = [
        (str(tmp_path / f"f{i}.tif"), np.full((8, 8, 3), i, np.uint8), None)
        for i in range(2)
    ]

    worker._process_single_stack(frames, str(tmp_path), "stack", 0)

    assert _FakeFusion.created[-1]["base_frame"] == "last"
    assert _FakeFusion.created[-1]["sort_mode"] == "desc"
    assert saved_cfg[-1]["px_um"] == pytest.approx(1.0)


def test_multi_folder_batch_uses_nested_order_settings_without_mutating_them(
        tmp_path, monkeypatch):
    import core.workers as workers

    _FakeFusion.created.clear()
    _FakeFusion.fuse_kwargs.clear()
    monkeypatch.setattr(workers, "MultiFocusFusion", _FakeFusion)
    settings = {
        "fusion_method": "openfocus_b",
        "fusion_params": {
            "kernel_size": 7,
            "base_frame": "last",
            "sort_mode": "desc",
        },
        "reg_methods": [],
        "format": "png",
    }
    worker = _batch_worker(settings)
    frames = [np.zeros((8, 8, 3), np.uint8) for _ in range(2)]
    monkeypatch.setattr(
        worker.image_loader, "load_from_folder",
        lambda _path: (True, "ok", frames, ["f0.png", "f1.png"]),
    )
    monkeypatch.setattr(worker, "save_fusion_result", lambda *_args: True)

    worker.process_single_folder(str(tmp_path))

    assert _FakeFusion.created[-1]["base_frame"] == "last"
    assert _FakeFusion.created[-1]["sort_mode"] == "desc"
    assert "base_frame" not in _FakeFusion.fuse_kwargs[-1]
    assert "sort_mode" not in _FakeFusion.fuse_kwargs[-1]
    assert settings["fusion_params"]["base_frame"] == "last"
    assert settings["fusion_params"]["sort_mode"] == "desc"


def test_registered_stack_reports_when_every_write_fails(tmp_path, monkeypatch):
    settings = {"format": "png"}
    worker = _batch_worker(settings)
    monkeypatch.setattr(worker, "_save_image", lambda *_args, **_kwargs: False)
    images = [np.zeros((8, 8, 3), np.uint8) for _ in range(2)]

    assert worker.save_registered_stack(
        str(tmp_path), images, ["f0.png", "f1.png"]) is False


def test_cli_depth_frames_follow_reverse_openfocus_order():
    from core.cli import _depth_frames_for

    frames = [object(), object(), object()]
    assert _depth_frames_for(frames, "openfocus_b", "desc") == frames[::-1]
    assert _depth_frames_for(frames, "openfocus_c", "asc") == frames
    assert _depth_frames_for(frames, "dct", "desc") == frames


def test_cli_openfocus_depth_map_uses_fusion_kernel_default(tmp_path, monkeypatch):
    import core.depth_map as depth_map
    import utils.colormap as colormap
    import utils.image_utils as image_utils
    from core.cli import _save_depth

    seen = {}

    def fake_index(_frames, _method, **kwargs):
        seen.update(kwargs)
        return np.zeros((4, 4), np.float32)

    monkeypatch.setattr(depth_map, "compute_focus_index", fake_index)
    monkeypatch.setattr(
        colormap, "colorize", lambda index, _name: np.zeros(index.shape + (3,), np.uint8))
    monkeypatch.setattr(image_utils, "imwrite_auto", lambda *_args, **_kwargs: True)
    args = SimpleNamespace(
        _depth_frames=[np.zeros((4, 4, 3), np.uint8) for _ in range(2)],
        depth_map=True,
        method="openfocus_b",
        kernel=None,
        cpu=True,
        depth_colormap="gray",
    )

    _save_depth(args, str(tmp_path / "fused.png"))

    assert seen["kernel_size"] == 11


def test_dtcwt_transform_queue_is_bounded_by_available_memory(monkeypatch):
    import fusion_methods.dtcwt as dtcwt_impl

    monkeypatch.setattr(dtcwt_impl, "_available_memory_bytes", lambda: 1 << 30)
    workers, lookahead = dtcwt_impl._memory_bounded_transform_plan(
        4000, 6000, 4, 8)

    assert (workers, lookahead) == (1, 1)

    workers, lookahead = dtcwt_impl._memory_bounded_transform_plan(
        1000, 1000, 4, 8)
    estimate = dtcwt_impl._estimate_pyramid_bytes(1000, 1000, 4)
    assert workers > 1
    assert lookahead * estimate <= (1 << 30) * 0.25
