"""The update-restore marker never outlives the snapshot it was written for."""

from types import SimpleNamespace

import pytest

from utils import recovery


@pytest.fixture
def slot(tmp_path, monkeypatch):
    monkeypatch.setattr(recovery, "recovery_dir", lambda: str(tmp_path))
    return tmp_path


def _write_snapshot_as(slot, monkeypatch, content):
    def write(window):
        (slot / recovery.SESSION_FILE).write_text(content, encoding="utf-8")
        return True
    monkeypatch.setattr(recovery, "write_snapshot", write)


def test_empty_workspace_leaves_no_marker(slot):
    (slot / recovery.PENDING_RESTORE_FILE).write_text("", encoding="utf-8")
    (slot / recovery.SESSION_FILE).write_text("older session", encoding="utf-8")
    assert recovery.mark_pending_restore(SimpleNamespace(project_path="p.ofproj", raw_images=[]))
    assert not (slot / recovery.SESSION_FILE).exists()
    assert recovery.pending_restore() is None
    assert not (slot / recovery.PENDING_RESTORE_FILE).exists()


def test_marker_without_a_snapshot_is_dropped(slot):
    (slot / recovery.PENDING_RESTORE_FILE).write_text("", encoding="utf-8")
    assert recovery.pending_restore() is None
    assert not (slot / recovery.PENDING_RESTORE_FILE).exists()
    # A later session's snapshot is not restored by the stale marker.
    (slot / recovery.SESSION_FILE).write_text("later session", encoding="utf-8")
    assert recovery.pending_restore() is None


def test_marker_with_its_snapshot_restores(slot, monkeypatch):
    _write_snapshot_as(slot, monkeypatch, "update session")
    assert recovery.mark_pending_restore(
        SimpleNamespace(project_path="saved.ofproj", raw_images=[object()]))
    assert recovery.pending_restore() == "saved.ofproj"
    assert recovery.pending_restore() == "saved.ofproj"
