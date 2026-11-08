"""Render subprocess isolation and frame-transfer regressions."""
import os
from pathlib import Path

import numpy as np

from core import frame_storage
from core.render_process import RenderProcessWorker
from utils.error_messages import processing_error


def test_render_runs_in_subprocess_and_returns_owned_pixels(qapp, tmp_path, monkeypatch):
    monkeypatch.setattr(frame_storage, '_storage_directory', None)
    monkeypatch.setattr(frame_storage, '_storage_parent', None)
    frame_storage.configure_storage_directory(tmp_path)
    first = np.full((32, 40, 3), 80, np.uint8)
    second = first.copy()
    second[10:20, 10:20] = 180
    worker = RenderProcessWorker(
        [first, second], algorithm='openfocus_a', alignment='none',
        fusion_parameters={'kernel_size': 11}, tile_enabled=False,
        thread_count=1, use_gpu=False,
    )
    results, errors = [], []
    worker.finished_signal.connect(lambda *args: results.append(args))
    worker.error_signal.connect(errors.append)

    worker.start()
    child_pid = worker._process.pid
    assert child_pid != os.getpid()
    assert worker.wait(30000)

    assert not errors and len(results) == 1
    assert results[0][1].shape == first.shape
    assert results[0][1].dtype == np.uint8
    assert Path(results[0][1].base.filename).parent == Path(frame_storage.storage_directory())


def test_native_crash_message_keeps_ui_alive_and_names_signal(qapp):
    worker = RenderProcessWorker([], algorithm=None, need_fusion=False)
    errors = []
    worker.error_signal.connect(errors.append)

    class CrashedProcess:
        exitcode = -11

        @staticmethod
        def is_alive():
            return False

        @staticmethod
        def join(timeout=0):
            pass

    worker._process = CrashedProcess()
    worker._running = True
    worker._poll()

    assert not worker.isRunning() and len(errors) == 1
    message = errors[0]
    assert 'signal 11' in message
    assert 'window is still safe' in message
    assert 'ONNX Runtime' in message


def test_cryptic_partial_write_gets_units_and_bit_depth_explanation():
    message = processing_error(OSError('61261920 requested and 5128192 written'))
    assert '61,261,920 array elements' in message
    assert '16-bit' in message and 'bytes' in message
