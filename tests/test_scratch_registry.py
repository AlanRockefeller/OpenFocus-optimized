"""Scratch stranded by crashed processes is removed; live scratch never is."""
import os
import subprocess
import sys
import textwrap
import time

import numpy as np
import pytest

from core import frame_storage, scratch_registry

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


@pytest.fixture
def registry(tmp_path, monkeypatch):
    directory = tmp_path / "registry"
    monkeypatch.setattr(scratch_registry, "_directory_override", directory)
    monkeypatch.setattr(scratch_registry, "_handle", None)
    monkeypatch.setattr(scratch_registry, "_handle_path", None)
    monkeypatch.setattr(scratch_registry, "_disabled", False)
    scratch = tmp_path / "scratch"
    scratch.mkdir()
    # Legacy sweeping looks at the system temp folder; point it at the test's.
    monkeypatch.setattr(scratch_registry.tempfile, "gettempdir", lambda: str(scratch))
    yield directory, scratch
    scratch_registry._close_at_exit()


def sweep(*extra):
    return scratch_registry._real_sweep_abandoned(extra)


def child(registry_dir, body, *, wait=False):
    """Run a process that records scratch, then crashes, waits or exits."""
    code = textwrap.dedent(f"""
        import os, sys, time
        sys.path.insert(0, {ROOT!r})
        from core import scratch_registry
        scratch_registry._directory_override = {str(registry_dir)!r}
    """) + textwrap.dedent(body)
    if wait:
        code += "\nprint('ready', flush=True)\nsys.stdin.readline()\nos._exit(0)\n"
    process = subprocess.Popen([sys.executable, "-c", code], cwd=ROOT, text=True,
                               stdin=subprocess.PIPE, stdout=subprocess.PIPE)
    if wait:
        assert process.stdout.readline().strip() == "ready"
        return process
    process.communicate(timeout=60)
    return process


def make(path, size=4096):
    path.write_bytes(b"\0" * size)
    return path


def test_crashed_process_scratch_is_removed(registry):
    directory, scratch = registry
    frame = scratch / "openfocus-frame-crash.bin"
    folder = scratch / "openfocus-abcd"
    child(directory, f"""
        import pathlib
        folder = pathlib.Path({str(folder)!r}); folder.mkdir()
        scratch_registry.register(folder)
        (folder / "openfocus-frame-inner.bin").write_bytes(b"x" * 100)
        frame = pathlib.Path({str(frame)!r}); frame.write_bytes(b"x" * 5000)
        scratch_registry.register(frame)
        os._exit(-7)   # no finalizers or atexit, like SIGBUS
    """)
    assert frame.exists() and folder.exists()
    assert len(list(directory.glob("*.scratch"))) == 1

    removed, freed = sweep()

    assert not frame.exists() and not folder.exists()
    assert removed == 2 and freed >= 5100
    assert not list(directory.glob("*.scratch"))


def test_running_process_scratch_is_kept(registry):
    directory, scratch = registry
    frame = scratch / "openfocus-frame-live.bin"
    process = child(directory, f"""
        import pathlib
        path = pathlib.Path({str(frame)!r}); path.write_bytes(b"x" * 10)
        scratch_registry.register(path)
    """, wait=True)
    try:
        assert sweep() == (0, 0)
        assert frame.exists()
    finally:
        process.stdin.write("\n")
        process.stdin.close()
        process.wait(timeout=60)
    sweep()
    assert not frame.exists()


def test_clean_exit_leaves_no_registry(registry):
    directory, scratch = registry
    frame = scratch / "openfocus-frame-clean.bin"
    child(directory, f"""
        import pathlib
        path = pathlib.Path({str(frame)!r}); path.write_bytes(b"x")
        scratch_registry.register(path)
        path.unlink()
    """)
    assert not list(directory.glob("*.scratch"))


def test_registry_cannot_remove_unexpected_paths(registry, tmp_path):
    directory, scratch = registry
    precious = make(tmp_path / "my-photo.tif")
    target = make(tmp_path / "openfocus-frame-target.bin")
    link = scratch / "openfocus-frame-link.bin"
    link.symlink_to(target)
    child(directory, f"""
        for path in ({str(precious)!r}, {str(link)!r}, "relative/openfocus-frame-x.bin"):
            scratch_registry.register(path)
        os._exit(1)
    """)
    sweep()
    assert precious.exists() and target.exists() and link.is_symlink()


def test_frame_storage_records_scratch(registry):
    directory, scratch = registry
    frame_storage.configure_storage_directory(str(scratch))
    try:
        mapped = frame_storage.create_frame((8, 8), np.uint16)
        stored = frame_storage.store_frame(np.ones((4, 4), np.uint8), prefer_ram=False)
        lines = scratch_registry._handle_path.read_text().splitlines()
        assert os.path.abspath(mapped.filename) in lines
        assert os.path.abspath(stored.filename) in lines
        # A running process's mapped frames survive a concurrent sweep.
        assert sweep() == (0, 0)
        assert os.path.exists(mapped.filename)
        del mapped, stored
    finally:
        frame_storage.configure_storage_directory(None)


@pytest.mark.skipif(not sys.platform.startswith("linux"), reason="uses /proc")
def test_legacy_frames_removed_only_when_no_process_uses_them(registry):
    directory, scratch = registry
    stale = make(scratch / "openfocus-frame-old.bin")
    mapped = make(scratch / "openfocus-frame-mapped.bin")
    other = make(scratch / "unrelated-frame.bin")
    process = subprocess.Popen(
        [sys.executable, "-c", textwrap.dedent(f"""
            import mmap, sys
            handle = open({str(mapped)!r}, "r+b")
            view = mmap.mmap(handle.fileno(), 0); handle.close()
            print("ready", flush=True); sys.stdin.readline()
        """)], text=True, stdin=subprocess.PIPE, stdout=subprocess.PIPE)
    try:
        assert process.stdout.readline().strip() == "ready"
        sweep()
        assert not stale.exists()
        assert mapped.exists() and other.exists()
    finally:
        process.communicate("\n", timeout=60)


def test_legacy_previews_need_a_reboot(registry, monkeypatch):
    directory, scratch = registry
    boot = time.time() - 3600
    monkeypatch.setattr(scratch_registry, "_boot_time", lambda: boot)
    recent = make(scratch / "openfocus-preview-recent")
    old = make(scratch / "openfocus-preview-old")
    old_frame = make(scratch / "openfocus-frame-before-boot.bin")
    os.utime(old, (boot - 600, boot - 600))
    os.utime(old_frame, (boot - 600, boot - 600))
    monkeypatch.setattr(scratch_registry, "_open_by_any_process", lambda: None)

    sweep()

    assert recent.exists()   # might belong to an older build that is running
    assert not old.exists() and not old_frame.exists()


def test_legacy_sweep_skips_paths_recorded_by_a_live_process(registry, monkeypatch):
    directory, scratch = registry
    frame = scratch / "openfocus-frame-recorded.bin"
    process = child(directory, f"""
        import pathlib
        path = pathlib.Path({str(frame)!r}); path.write_bytes(b"x")
        scratch_registry.register(path)
    """, wait=True)
    try:
        # Not open anywhere and older than boot, but its owner is alive.
        monkeypatch.setattr(scratch_registry, "_open_by_any_process", lambda: set())
        monkeypatch.setattr(scratch_registry, "_boot_time", lambda: time.time() + 3600)
        sweep()
        assert frame.exists()
    finally:
        process.communicate("\n", timeout=60)


def test_unavailable_registry_never_blocks_processing(registry, tmp_path, monkeypatch):
    blocker = tmp_path / "blocked"
    blocker.write_text("not a folder")
    monkeypatch.setattr(scratch_registry, "_directory_override", blocker / "registry")
    scratch_registry.register(tmp_path / "openfocus-frame-x.bin")
    assert scratch_registry._disabled
    assert sweep() == (0, 0)


def test_boot_time_is_in_the_past():
    boot = scratch_registry._boot_time()
    if boot is not None:
        assert 0 < boot < time.time()
