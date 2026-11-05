"""Regression tests for the review findings on PR #2 (upstream v1.67 merge)."""
import json

import numpy as np
import pytest

pytest.importorskip("PyQt6")


# --------------------------------------------------------------------------
# Per-method brightness options survive a restart and a project round trip
# --------------------------------------------------------------------------

class _Slider:
    def __init__(self):
        self._value = 31

    def value(self):
        return self._value

    def setValue(self, value):
        self._value = value


def _brightness_window(memory=None):
    """The parts of MainWindow the settings store and project file use."""
    from main import OpenFocus

    w = type("W", (), {"_BRIGHTNESS_DEFAULTS": OpenFocus._BRIGHTNESS_DEFAULTS,
                       "brightness_option_for": OpenFocus.brightness_option_for})()
    w._brightness_by_method = dict(memory or {})
    w._fusion = "guided_filter"
    w.current_fusion_method = lambda: w._fusion
    w.set_fusion_method = lambda code: setattr(w, "_fusion", code)
    w.current_alignment_method = lambda: "none"
    w.set_alignment_method = lambda code: None
    w.slider_smooth = _Slider()
    w.chk_quick_preview = None
    w.raw_images = []
    w.image_filenames = []
    w.recent_files = []
    w.recent_projects = []
    return w


def test_brightness_memory_json_round_trip_drops_malformed_entries():
    from utils.settings_store import (
        brightness_memory_from_json, brightness_memory_to_json)

    memory = {("exposure_match", "guided_filter"): False,
              ("base_correct", "openfocus_b"): False}
    assert brightness_memory_from_json(
        brightness_memory_to_json(memory)) == memory
    assert brightness_memory_from_json("not json") == {}
    assert brightness_memory_from_json('{"exposure_match": {"dct": "yes"},'
                                       ' "unknown": {"dct": true}}') == {}


def test_per_method_brightness_survives_a_restart(qapp):
    from utils.settings_store import (
        BRIGHTNESS_BY_METHOD_KEY, get_settings, load_window_settings,
        reset_settings_singleton, save_window_settings)

    reset_settings_singleton()
    try:
        saved = _brightness_window({("exposure_match", "guided_filter"): False})
        save_window_settings(saved)

        reset_settings_singleton()
        restored = _brightness_window()
        load_window_settings(restored)
        assert restored.brightness_option_for("exposure_match") is False, (
            "turning exposure matching off must not be undone by a restart")
        # Methods the user never touched keep their defaults.
        assert restored.brightness_option_for(
            "exposure_match", "openfocus_a") is True
    finally:
        # The sandboxed settings are shared by the session: don't leave this
        # choice behind for windows created by later tests.
        get_settings().remove(BRIGHTNESS_BY_METHOD_KEY)
        reset_settings_singleton()


def test_project_restores_per_method_brightness():
    from utils import project_file

    saved = _brightness_window({("exposure_match", "guided_filter"): False,
                                ("base_correct", "openfocus_b"): False})
    # The global flag is stale (another method was toggled last); the project
    # must record what the current method renders with.
    saved.exposure_match = True
    state = json.loads(json.dumps(project_file._collect_state(saved)))
    assert state["settings"]["exposure_match"] is False

    restored = _brightness_window()
    project_file._apply_after_load(restored, state)
    assert restored._brightness_by_method == saved._brightness_by_method


def test_old_project_flags_apply_to_its_own_method():
    from utils import project_file

    restored = _brightness_window()
    project_file._apply_after_load(restored, {"settings": {
        "fusion_method": "openfocus_b",
        "exposure_match": False, "base_frame_correct": False}})
    assert restored.brightness_option_for(
        "exposure_match", "openfocus_b") is False
    assert restored.brightness_option_for(
        "base_correct", "openfocus_b") is False
    assert restored.brightness_option_for(
        "exposure_match", "openfocus_a") is True


# --------------------------------------------------------------------------
# StackMFF V3 depth maps pad their input once, with replicated edges
# --------------------------------------------------------------------------

def test_depth_map_input_is_padded_once_with_replicated_edges(monkeypatch):
    torch = pytest.importorskip("torch")
    from fusion_methods import stackmffv3

    seen = {}

    class FakeModel:
        def forward_probs(self, batch, sigma=0.0):
            seen["batch"] = batch.clone()
            n, h, w = batch.shape[1:]
            return torch.full((1, n, h, w), 1.0 / n)

    monkeypatch.setattr(stackmffv3, "_get_model_and_device",
                        lambda path, use_gpu: (FakeModel(), torch.device("cpu")))
    frames = [np.full((70, 100, 3), 200, np.uint8) for _ in range(3)]
    probs = stackmffv3.model_probabilities(frames, "unused.pth", False)

    batch = seen["batch"]
    assert batch.shape[-2:] == (128, 128), "padded to the next multiple of 64"
    assert torch.allclose(batch, torch.full_like(batch, 200 / 255.0)), (
        "the padded band must replicate the edge, not be black")
    assert probs.shape == (3, 70, 100)
