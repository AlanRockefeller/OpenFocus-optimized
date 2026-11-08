"""Crash-isolated render worker with a Qt-compatible controller interface."""
from __future__ import annotations

import multiprocessing
import os
import signal
import time
from typing import Any

import numpy as np
from PyQt6.QtCore import QObject, QTimer, pyqtSignal

from core.frame_storage import (adopt_storage_directory, describe_frame,
                                is_file_backed_frame, open_frame, storage_directory,
                                store_frame)
from core.algorithms import resolve_fusion_parameters
from utils.error_messages import processing_error


def _decode_input(value):
    if isinstance(value, dict) and value.get("__openfocus_frame__"):
        return open_frame(value["description"], writable=False)
    if isinstance(value, dict) and value.get("__openfocus_rect__"):
        from PyQt6.QtCore import QRectF
        return QRectF(*value["values"])
    if isinstance(value, dict):
        return {key: _decode_input(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_decode_input(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_decode_input(item) for item in value)
    return value


def _encode_output(frame):
    if not is_file_backed_frame(frame):
        frame = store_frame(np.ascontiguousarray(frame))
    return describe_frame(frame, transfer=True)


def _render_process_main(connection, cancelled, run_directory, args, kwargs):
    """Subprocess entry point; never imports or touches the main window."""
    try:
        adopt_storage_directory(run_directory)
        args = _decode_input(args)
        kwargs = _decode_input(kwargs)
        from PyQt6.QtCore import Qt
        from core.registration import RegistrationCancelled
        from core.workers import RenderWorker

        worker = RenderWorker(*args, **kwargs)

        def send(message):
            try:
                connection.send(message)
            except (BrokenPipeError, EOFError, OSError):
                import logging
                logging.getLogger("openfocus").exception(
                    "render subprocess could not send %s", message[0])
                worker.is_cancelled = True

        last_progress = [0.0]

        def progress(done, total):
            now = time.monotonic()
            if done == total or now - last_progress[0] >= 0.1:
                last_progress[0] = now
                send(("progress", int(done), int(total)))

        worker.stage_signal.connect(
            lambda stage: send(("stage", str(stage))), Qt.ConnectionType.DirectConnection)
        worker.progress_signal.connect(progress, Qt.ConnectionType.DirectConnection)

        import threading
        threading.Thread(
            target=lambda: (cancelled.wait(), setattr(worker, "is_cancelled", True)),
            name="render-cancel", daemon=True,
        ).start()

        send(("stage", "waiting"))
        if cancelled.is_set():
            raise RegistrationCancelled()
        send(("stage", "prepare"))
        result = worker._run_admitted()
        if cancelled.is_set():
            raise RegistrationCancelled()
        processed, fused, registered, align_s, fusion_s, device = result
        payload = {
            "processed": [_encode_output(frame) for frame in processed],
            "fused": _encode_output(fused) if fused is not None else None,
            "registered": bool(registered),
            "alignment_seconds": float(align_s),
            "fusion_seconds": float(fusion_s),
            "device": str(device),
            "alignment_reused": bool(worker.alignment_reused),
            "alignment_computed": bool(worker.alignment_computed),
            "alignment_report": worker.alignment_report,
            "alignment_report_provenance": worker.alignment_report_provenance,
        }
        send(("result", payload))
    except BaseException as exc:
        import logging
        logging.getLogger("openfocus").exception("render subprocess failed")
        try:
            from core.registration import RegistrationCancelled
            cancelled_error = isinstance(exc, RegistrationCancelled) or cancelled.is_set()
        except Exception:
            cancelled_error = cancelled.is_set()
        try:
            connection.send((
                "error",
                "CANCELLED: render cancelled" if cancelled_error else processing_error(exc),
            ))
        except (BrokenPipeError, EOFError, OSError):
            pass
    finally:
        try:
            connection.close()
        except OSError:
            pass


class RenderProcessWorker(QObject):
    """Run ``RenderWorker`` in a child process and expose its familiar signals."""

    process_isolated = True

    finished_signal = pyqtSignal(object, object, bool, float, float, str)
    progress_signal = pyqtSignal(int, int)
    stage_signal = pyqtSignal(str)
    error_signal = pyqtSignal(str)
    finished = pyqtSignal()

    def __init__(self, *args, **kwargs):
        super().__init__()
        self._args = list(args)
        self._kwargs = dict(kwargs)
        disk_cache = self._kwargs.get("disk_cache")
        if disk_cache is not None:
            self._kwargs["disk_cache"] = dict(disk_cache)
            self._kwargs["disk_cache"].pop("is_current", None)
        roi = self._kwargs.get("roi_rect")
        if roi is not None:
            self._kwargs["roi_rect"] = {
                "__openfocus_rect__": True,
                "values": (float(roi.x()), float(roi.y()),
                           float(roi.width()), float(roi.height())),
            }
        self._input_leases = []
        for index in range(min(2, len(self._args))):
            if self._args[index] is not None:
                self._args[index] = self._encode_frames(self._args[index])
        self.algorithm = kwargs.get("algorithm")
        self.fusion_parameters = (resolve_fusion_parameters(
            self.algorithm, kwargs.get("fusion_parameters")) if self.algorithm else {})
        self.alignment_reused = False
        self.alignment_computed = False
        self.alignment_report = [] if kwargs.get("collect_alignment_report") else None
        self.alignment_report_provenance = None
        self._process = None
        self._connection = None
        self._cancel_event = None
        self._running = False
        self._terminal = False
        self._is_cancelled = False
        self._timer = QTimer(self)
        self._timer.setInterval(25)
        self._timer.timeout.connect(self._poll)

    def _encode_frames(self, frames):
        encoded = []
        for frame in frames:
            if isinstance(frame, np.ndarray):
                if not is_file_backed_frame(frame):
                    frame = store_frame(np.ascontiguousarray(frame))
                    self._input_leases.append(frame)
                encoded.append({"__openfocus_frame__": True,
                                "description": describe_frame(frame)})
            else:
                encoded.append(frame)
        return encoded

    @property
    def is_cancelled(self):
        return self._is_cancelled

    @is_cancelled.setter
    def is_cancelled(self, value):
        self._is_cancelled = bool(value)
        if self._is_cancelled and self._cancel_event is not None:
            self._cancel_event.set()

    def start(self):
        if self._running:
            return
        context = multiprocessing.get_context("spawn")
        parent, child = context.Pipe(duplex=False)
        self._connection = parent
        self._cancel_event = context.Event()
        self._process = context.Process(
            target=_render_process_main,
            args=(child, self._cancel_event, storage_directory(),
                  tuple(self._args), self._kwargs),
            name="OpenFocus render worker", daemon=True,
        )
        self._running = True
        try:
            self._process.start()
        except BaseException:
            self._running = False
            child.close()
            parent.close()
            raise
        child.close()
        self._timer.start()

    def cancel(self):
        self.is_cancelled = True

    def requestInterruption(self):
        self.cancel()

    def quit(self):
        self.cancel()

    def isRunning(self):
        return self._running

    def _crash_message(self, exit_code):
        if os.name == "nt" and exit_code is not None and exit_code < -128:
            # multiprocessing exposes signed Windows NTSTATUS values. Hex is
            # what crash reports and driver documentation use (for example,
            # 0xC0000005 is an access violation).
            reason = f"native exception code 0x{exit_code & 0xffffffff:08X}"
        elif exit_code is not None and exit_code < 0:
            number = -exit_code
            try:
                label = signal.strsignal(number)
            except (ValueError, AttributeError):
                label = "native process failure"
            reason = f"signal {number} ({label})"
        else:
            reason = f"exit code {exit_code}" if exit_code is not None else "an unknown exit code"
        return (
            f"The render worker process stopped unexpectedly with {reason}. "
            "The OpenFocus window is still safe. This usually indicates a native NumPy, "
            "OpenCV, ONNX Runtime, or graphics-driver failure, or an operating-system "
            "out-of-memory termination. Try the render again with GPU acceleration "
            "disabled or a smaller tile/batch size."
        )

    def _decode_result(self, payload):
        output_paths = {item["path"] for item in payload["processed"]}
        if payload["fused"] is not None:
            output_paths.add(payload["fused"]["path"])
        for lease in self._input_leases:
            description = describe_frame(lease)
            if description["path"] in output_paths:
                describe_frame(lease, transfer=True)
        processed = [open_frame(item, owned=True, writable=True)
                     for item in payload["processed"]]
        fused = (open_frame(payload["fused"], owned=True, writable=True)
                 if payload["fused"] is not None else None)
        self.alignment_reused = payload["alignment_reused"]
        self.alignment_computed = payload["alignment_computed"]
        self.alignment_report = payload["alignment_report"]
        self.alignment_report_provenance = payload["alignment_report_provenance"]
        return (processed, fused, payload["registered"],
                payload["alignment_seconds"], payload["fusion_seconds"], payload["device"])

    def _handle(self, message):
        kind, *values = message
        if kind == "stage":
            self.stage_signal.emit(values[0])
        elif kind == "progress":
            self.progress_signal.emit(values[0], values[1])
        elif kind == "result":
            self._complete(result=self._decode_result(values[0]))
        elif kind == "error":
            self._complete(error=values[0])

    def _poll(self):
        connection = self._connection
        if connection is not None:
            while connection.poll():
                try:
                    message = connection.recv()
                except (EOFError, OSError):
                    break
                try:
                    self._handle(message)
                except Exception as exc:
                    self._complete(error=(
                        "The render worker completed, but OpenFocus could not read its result: "
                        + processing_error(exc)))
                if self._terminal:
                    break
        if not self._terminal and self._process is not None and not self._process.is_alive():
            self._process.join(timeout=0)
            # On Windows and spawn-based POSIX workers, process termination can
            # become visible a few milliseconds before the final pipe packet.
            try:
                if connection is not None and connection.poll(0.1):
                    self._handle(connection.recv())
            except (EOFError, OSError):
                pass
            if self._terminal:
                return
            self._complete(error=("CANCELLED: render cancelled" if self.is_cancelled
                                  else self._crash_message(self._process.exitcode)))

    def _complete(self, *, result=None, error=None):
        if self._terminal:
            return
        self._terminal = True
        self._running = False
        self._timer.stop()
        if self._process is not None:
            self._process.join(timeout=0.2)
        if self._connection is not None:
            self._connection.close()
            self._connection = None
        if error is None:
            self.finished_signal.emit(*result)
        else:
            self.error_signal.emit(str(error))
        self.finished.emit()

    def wait(self, timeout_ms=30000):
        deadline = time.monotonic() + max(0, timeout_ms) / 1000.0
        while self._running and time.monotonic() < deadline:
            self._poll()
            if self._running:
                time.sleep(0.01)
        return not self._running

    def terminate(self):
        if self._process is not None and self._process.is_alive():
            self._process.terminate()

