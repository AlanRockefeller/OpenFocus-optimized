"""Files written through a temporary file get the mode a plain open() gives.

Exports, projects and templates are staged in a ``mkstemp`` file, which is
owner-only. Publishing that file must not leave the user's result unreadable
to the rest of the system; private files must keep the owner-only mode.
"""
import json
import os
import stat

import numpy as np
import pytest

pytest.importorskip("PyQt6")
cv2 = pytest.importorskip("cv2")

pytestmark = pytest.mark.skipif(os.name == "nt", reason="POSIX permission bits")


def _mode(path):
    return stat.S_IMODE(os.stat(path).st_mode)


@pytest.fixture
def umask_022(monkeypatch):
    previous = os.umask(0o022)
    try:
        from utils import file_modes
    except ImportError:
        file_modes = None
    if file_modes is not None:
        monkeypatch.setattr(file_modes, "_UMASK", 0o022)
    yield
    os.umask(previous)


def _image():
    return np.arange(16 * 24 * 3, dtype=np.uint8).reshape(16, 24, 3)


@pytest.mark.parametrize("ext", [".png", ".jpg", ".tif"])
def test_published_export_is_readable_by_others(tmp_path, umask_022, ext):
    from utils.export_paths import publish_unique
    from utils.image_utils import imwrite_auto
    image = _image()
    first = publish_unique(str(tmp_path / f"result{ext}"),
                           lambda staging: imwrite_auto(staging, image))
    second = publish_unique(str(tmp_path / f"result{ext}"),
                            lambda staging: imwrite_auto(staging, image))
    assert os.path.basename(second) == f"result_2{ext}"
    assert _mode(first) == 0o644
    assert _mode(second) == 0o644


@pytest.mark.parametrize("ext", [".png", ".jpg", ".tif"])
def test_direct_image_write_uses_default_mode(tmp_path, umask_022, ext):
    from utils.image_utils import imwrite_auto
    path = tmp_path / f"direct{ext}"
    assert imwrite_auto(str(path), _image())
    assert _mode(path) == 0o644


def test_streamed_tiff_write_uses_default_mode(tmp_path, umask_022):
    from utils.image_utils import imwrite_auto
    path = tmp_path / "streamed.tif"
    assert imwrite_auto(str(path), _image(), exif=b"")
    assert _mode(path) == 0o644


def test_overwritten_image_keeps_its_mode(tmp_path, umask_022):
    from utils.image_utils import imwrite_auto
    path = tmp_path / "shared.png"
    assert cv2.imwrite(str(path), _image())
    os.chmod(path, 0o640)
    assert imwrite_auto(str(path), _image())
    assert _mode(path) == 0o640


def test_default_mode_follows_the_umask(tmp_path, monkeypatch):
    from utils import file_modes
    from utils.image_utils import imwrite_auto
    monkeypatch.setattr(file_modes, "_UMASK", 0o077)
    path = tmp_path / "private.png"
    assert imwrite_auto(str(path), _image())
    assert _mode(path) == 0o600


def test_startup_umask_reading_does_not_change_it():
    from utils import file_modes
    current = os.umask(0o027)
    try:
        assert file_modes._read_umask() == 0o027
        assert os.umask(0o027) == 0o027
    finally:
        os.umask(current)


def test_template_and_json_writes_keep_or_default_the_mode(tmp_path, umask_022):
    from utils.project_file import atomic_json_write
    fresh = tmp_path / "new.oftemplate"
    atomic_json_write(str(fresh), {"x": 1})
    assert _mode(fresh) == 0o644
    shared = tmp_path / "shared.oftemplate"
    shared.write_text("{}")
    os.chmod(shared, 0o664)
    atomic_json_write(str(shared), {"x": 2})
    assert _mode(shared) == 0o664
    assert json.loads(shared.read_text()) == {"x": 2}


def _project_window(tmp_path):
    from tests.test_project_file import _window
    paths = []
    for index in range(2):
        path = tmp_path / f"frame{index}.png"
        assert cv2.imwrite(str(path), np.full((8, 8, 3), 40 * index, np.uint8))
        paths.append(str(path))
    return _window(paths, str(tmp_path))


def test_saved_project_uses_default_mode_and_keeps_an_existing_mode(tmp_path, umask_022):
    from utils.project_file import save_project
    window = _project_window(tmp_path)
    path = str(tmp_path / "stack.ofproj")
    ok, error = save_project(window, path)
    assert ok, error
    assert _mode(path) == 0o644
    ok, error = save_project(window, path)
    assert ok, error
    assert _mode(path) == 0o644
    os.chmod(path, 0o660)
    ok, error = save_project(window, path)
    assert ok, error
    assert _mode(path) == 0o660


def _pathless_window(tmp_path, monkeypatch):
    from tests.test_project_file import _window
    from utils import frame_pixels
    monkeypatch.setattr(frame_pixels, "run_pixel_worker",
                        lambda parent, title, operation: operation())
    window = _window([], str(tmp_path))
    window.raw_images = [np.full((12, 18, 3), 93, np.uint8)]
    window.image_filenames = ["memory"]
    return window


def test_saved_project_sidecars_use_default_mode(tmp_path, monkeypatch, umask_022):
    from utils.project_file import save_project
    window = _pathless_window(tmp_path, monkeypatch)
    path = str(tmp_path / "memory.ofproj")
    ok, error = save_project(window, path)
    assert ok, error
    sidecars = [os.path.join(root, name)
                for root, _dirs, names in os.walk(path + ".sources") for name in names]
    assert sidecars
    assert _mode(path) == 0o644
    assert all(_mode(sidecar) == 0o644 for sidecar in sidecars)
    assert all(_mode(os.path.dirname(sidecar)) == 0o755 for sidecar in sidecars)


def test_recovery_snapshot_and_sidecars_stay_private(tmp_path, monkeypatch, umask_022):
    from utils import recovery
    window = _pathless_window(tmp_path, monkeypatch)
    monkeypatch.setattr(recovery, "recovery_dir", lambda: str(tmp_path / "recovery"))
    assert recovery.write_snapshot(window)
    session = recovery.session_path()
    sidecars = [os.path.join(root, name)
                for root, _dirs, names in os.walk(session + ".sources") for name in names]
    assert sidecars
    assert _mode(session) == 0o600
    assert all(_mode(sidecar) == 0o600 for sidecar in sidecars)
    # Rewriting an existing snapshot keeps it private as well.
    assert recovery.write_snapshot(window)
    assert _mode(session) == 0o600


def test_account_session_file_stays_private(tmp_path, umask_022):
    from tests.test_account_client import session
    from utils import account_client
    store = account_client.TokenStore(tmp_path / "private" / "session")
    store.save(session())
    assert _mode(store.path) == 0o600


def test_aligned_frames_use_default_mode(tmp_path, umask_022):
    from core.registration import _save_registration
    out = tmp_path / "aligned"
    _save_registration([_image(), _image()], str(out), ["a.png", "b.tif"], None)
    assert sorted(os.listdir(out)) == ["a.png", "b.tif"]
    assert all(_mode(out / name) == 0o644 for name in ("a.png", "b.tif"))


def test_cli_text_reports_use_default_mode(tmp_path, umask_022):
    from core.cli import _replace_text_file
    path = tmp_path / "summary.json"
    _replace_text_file(str(path), "{}\n", ".openfocus_summary_")
    assert _mode(path) == 0o644


def test_flip_review_records_use_default_mode(tmp_path, umask_022):
    from core.flip_review import atomic_write_json
    path = tmp_path / "review.json"
    atomic_write_json(str(path), {"items": []})
    assert _mode(path) == 0o644


def test_feedback_package_uses_default_mode(tmp_path, umask_022):
    from utils.logging_setup import export_feedback_package
    path = export_feedback_package(tmp_path / "feedback.zip", "note", include_all_logs=False)
    assert _mode(path) == 0o644


def test_animated_gif_uses_default_mode(tmp_path, umask_022):
    from core.workers import GifSaverWorker
    path = tmp_path / "stack.gif"
    outcomes = []
    worker = GifSaverWorker([_image(), _image()], str(path), 0.1, None, "raw")
    worker.finished_signal.connect(lambda *args: outcomes.append(args))
    worker.run()
    assert outcomes == [(True, str(path))]
    assert _mode(path) == 0o644
