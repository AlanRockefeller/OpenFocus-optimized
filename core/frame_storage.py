"""Disk-backed frame storage whose lifetime follows numpy views."""
import atexit
import errno
import math
import os
from pathlib import Path
import re
import shutil
import tempfile
import time
import weakref

import numpy as np

_storage_parent = None
_storage_directory = None
_RUN_PREFIX = "openfocus-run-"
_RUN_PATTERN = re.compile(r"^openfocus-run-(\d+)$")
_OWNER_LOCK_SUFFIX = ".owner.lock"
_LEGACY_PREFIXES = ("openfocus-frame-", "openfocus-preview-")
# New builds never create loose files in the parent. The single-instance guard
# runs before window/storage initialization, so every legacy loose file found
# here was abandoned by an older session and can be removed immediately.
_LEGACY_MIN_AGE_SECONDS = 0
_owned_run_directories = set()
_owned_run_locks = {}
_frame_finalizers = {}


class TemporaryStorageError(OSError):
    """A temporary frame could not be admitted or completely written."""


def _size_text(size):
    size = max(0, int(size))
    if size < 1024:
        return f"{size} bytes"
    if size < 1024 ** 2:
        return f"{size} bytes ({size / 1024:.1f} KiB)"
    if size < 1024 ** 3:
        return f"{size} bytes ({size / 1024 ** 2:.1f} MiB)"
    return f"{size} bytes ({size / 1024 ** 3:.2f} GiB)"


format_bytes = _size_text


def _size_unit(size):
    """Human unit without replacing the exact byte count used by tests/logs."""
    size = max(0, int(size))
    if size < 1024:
        return None
    if size < 1024 ** 2:
        return f"{size / 1024:.1f} KiB"
    if size < 1024 ** 3:
        return f"{size / 1024 ** 2:.1f} MiB"
    return f"{size / 1024 ** 3:.2f} GiB"


def _storage_error(directory, requested, *, written=None, cause=None):
    """Build an actionable error without hiding the byte counts from logs."""
    try:
        free = shutil.disk_usage(directory).free
        free_text = _size_text(free)
    except OSError:
        free_text = "unknown"
    if written is None:
        operation = f"OpenFocus could not allocate {_size_text(requested)}"
    else:
        operation = (f"OpenFocus could not write a complete temporary frame: "
                     f"{_size_text(requested)} requested, {_size_text(written)} written")
    detail = f" Underlying storage error: {cause}." if cause else ""
    return TemporaryStorageError(
        f"{operation} in {directory}. The temporary disk is full, its quota was reached, "
        f"or it cannot accept a file of this size (free space reported: {free_text}).{detail} "
        "Free temporary disk space or choose a larger Temporary directory in Advanced Settings."
    )


def storage_directory():
    """Return the directory currently used for new temporary data."""
    if _storage_directory is None:
        configure_storage_directory("")
    return _storage_directory


def storage_parent_directory():
    """Return the configured parent, without the process-specific component."""
    return _storage_parent if _storage_parent is not None else tempfile.gettempdir()


def _pid_is_alive(pid):
    if pid == os.getpid():
        return True
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    except OSError as exc:
        return exc.errno not in (errno.ESRCH, errno.EINVAL)
    return True


def _try_owner_lock(directory, *, create=False):
    """Return a held directory-owner lock, or None when another process owns it.

    File locks cross PID-namespace boundaries, unlike ``kill(pid, 0)``.  The
    handle must remain open for as long as ownership is claimed.
    """
    directory = os.path.abspath(os.fspath(directory))
    lock_path = os.path.join(
        os.path.dirname(directory), f".{os.path.basename(directory)}{_OWNER_LOCK_SUFFIX}")
    try:
        handle = open(lock_path, "a+b" if create else "r+b")
    except OSError:
        return None
    try:
        if os.name == "nt":
            import msvcrt
            handle.seek(0, os.SEEK_END)
            if handle.tell() == 0:
                handle.write(b"\0")
                handle.flush()
            handle.seek(0)
            msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
        else:
            import fcntl
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        return handle
    except (OSError, ImportError):
        handle.close()
        return None


def cleanup_stale_storage(parent, *, now=None):
    """Remove abandoned run directories and old pre-run-directory files."""
    parent = Path(parent)
    now = time.time() if now is None else float(now)
    removed = []
    try:
        entries = list(parent.iterdir())
    except OSError:
        return removed
    for entry in entries:
        match = _RUN_PATTERN.match(entry.name)
        if match and entry.is_dir():
            lock_path = Path(
                os.path.dirname(entry), f".{entry.name}{_OWNER_LOCK_SUFFIX}")
            if lock_path.is_file():
                lock = _try_owner_lock(entry)
                if lock is None:
                    continue
                try:
                    lock.close()
                    shutil.rmtree(entry)
                    lock_path.unlink(missing_ok=True)
                    removed.append(str(entry))
                except OSError:
                    pass
            # A lockless directory came from an older OpenFocus build.  A PID
            # absent from this namespace may still be alive in another one, so
            # there is no safe automatic proof that this directory is stale.
            continue
        if entry.is_file() and entry.name.startswith(_LEGACY_PREFIXES):
            try:
                if now - entry.stat().st_mtime >= _LEGACY_MIN_AGE_SECONDS:
                    entry.unlink()
                    removed.append(str(entry))
            except OSError:
                pass
    return removed


def _cleanup_owned_storage():
    for path in tuple(_owned_run_directories):
        lock = _owned_run_locks.pop(path, None)
        if lock is not None:
            try:
                lock.close()
            except OSError:
                pass
        try:
            shutil.rmtree(path)
        except FileNotFoundError:
            pass
        except OSError:
            # A crash or a still-closing mapping is handled on next startup.
            pass
        try:
            lock_path = os.path.join(
                os.path.dirname(path), f".{os.path.basename(path)}{_OWNER_LOCK_SUFFIX}")
            os.unlink(lock_path)
        except FileNotFoundError:
            pass
        except OSError:
            pass


atexit.register(_cleanup_owned_storage)


def configure_storage_directory(path):
    """Select a writable parent and create this process's isolated run directory."""
    global _storage_parent, _storage_directory
    parent = (tempfile.gettempdir() if path is None or os.fspath(path) in ('', b'')
              else os.path.abspath(os.fsdecode(path)))
    fd, probe = tempfile.mkstemp(prefix='openfocus-probe-', dir=parent)
    try:
        os.close(fd)
    finally:
        os.unlink(probe)
    default_parent = os.path.abspath(tempfile.gettempdir())
    if os.path.abspath(parent) != default_parent:
        cleanup_stale_storage(default_parent)
    cleanup_stale_storage(parent)
    directory = os.path.join(parent, f"{_RUN_PREFIX}{os.getpid()}")
    os.makedirs(directory, exist_ok=True)
    if directory not in _owned_run_locks:
        lock = _try_owner_lock(directory, create=True)
        if lock is None:
            raise TemporaryStorageError(
                f"OpenFocus temporary directory is already owned by another process: {directory}"
            )
        _owned_run_locks[directory] = lock
    _storage_parent = parent
    _storage_directory = directory
    _owned_run_directories.add(directory)


def adopt_storage_directory(path):
    """Use an existing run directory in a render subprocess without owning it."""
    global _storage_parent, _storage_directory
    directory = os.path.abspath(os.fsdecode(path))
    if not os.path.isdir(directory):
        raise OSError(f"Temporary run directory no longer exists: {directory}")
    _storage_parent = os.path.dirname(directory)
    _storage_directory = directory


def _remove(path):
    try:
        os.unlink(path)
    except OSError:
        # Another mapping can still hold the file on Windows. Its finalizer,
        # or next-startup run-directory cleanup, gets another chance.
        pass


def _release(mapping, path):
    mapping.close()
    _remove(path)


def _release_registered(mapping, path):
    _release(mapping, path)
    finalizers = [item for item in _frame_finalizers.get(path, ()) if item.alive]
    if finalizers:
        _frame_finalizers[path] = finalizers
    else:
        _frame_finalizers.pop(path, None)


def _own_frame(mapped, path):
    path = os.path.abspath(os.fsdecode(path))
    finalizer = weakref.finalize(mapped, _release_registered, mapped._mmap, path)
    _frame_finalizers.setdefault(path, []).append(finalizer)
    return finalizer


def _mapped_base(image):
    current = image
    mapped = current if isinstance(current, np.memmap) else None
    seen = set()
    while getattr(current, "base", None) is not None and id(current) not in seen:
        seen.add(id(current))
        current = current.base
        if isinstance(current, np.memmap):
            mapped = current
    return mapped


def is_file_backed_frame(image):
    mapped = _mapped_base(image)
    return mapped is not None and getattr(mapped, "filename", None) is not None


def describe_frame(image, *, transfer=False):
    """Describe an array mapping for zero-copy transfer to another process."""
    mapped = _mapped_base(image)
    if mapped is None or not getattr(mapped, "filename", None):
        raise ValueError("Frame is not backed by a shareable file")
    offset = (int(image.__array_interface__["data"][0])
              - int(mapped.__array_interface__["data"][0])
              + int(getattr(mapped, "offset", 0)))
    if transfer:
        path = os.path.abspath(os.fsdecode(mapped.filename))
        for finalizer in _frame_finalizers.pop(path, []):
            finalizer.detach()
        # Compatibility with mappings created before the ownership registry.
        for reference in weakref.getweakrefs(mapped):
            if isinstance(reference, weakref.finalize):
                reference.detach()
    return {
        "path": os.path.abspath(os.fsdecode(mapped.filename)),
        "shape": tuple(int(value) for value in image.shape),
        "dtype": image.dtype.str,
        "strides": tuple(int(value) for value in image.strides),
        "offset": offset,
    }


def open_frame(description, *, owned=False, writable=False):
    """Open a frame descriptor received from another process."""
    path = os.path.abspath(os.fsdecode(description["path"]))
    backing = np.memmap(path, dtype=np.uint8, mode="r+" if writable else "r")
    image = np.ndarray(
        tuple(description["shape"]), dtype=np.dtype(description["dtype"]),
        buffer=backing, offset=int(description.get("offset", 0)),
        strides=tuple(description["strides"]),
    )
    if owned:
        _own_frame(backing, path)
    return image

def require_disk_space(size, directory=None):
    """Check a point-in-time estimate, retaining space for the system.

    Other processes can consume space after this check; it is not a reservation.
    """
    directory = storage_directory() if directory is None else directory
    usage = shutil.disk_usage(directory)
    reserve = max(1024 ** 3, min(usage.total // 20, 4 * 1024 ** 3))
    if size > max(0, usage.free - reserve):
        reserve_unit = _size_unit(reserve)
        reserve_detail = f" ({reserve_unit})" if reserve_unit else ""
        free_unit = _size_unit(usage.free)
        free_detail = f" ({free_unit})" if free_unit else ""
        raise TemporaryStorageError(
            f"Insufficient temporary disk space in {directory}: need {_size_text(size)}, "
            f"with {reserve} bytes reserved{reserve_detail} "
            f"(free: {usage.free} bytes{free_detail}). "
            "Free temporary disk space or choose a larger Temporary directory in Advanced Settings."
        )


def _allocate_file(fd, size, directory):
    """Reserve real disk blocks when supported, before pixels are produced.

    Merely extending a file can create a sparse mapping. Such a mapping passes
    the free-space check and then fails much later when dirty pages are flushed.
    ``posix_fallocate`` makes admission atomic at the filesystem layer and also
    prevents concurrent frame writers from all spending the same free bytes.
    """
    size = int(size)
    try:
        if hasattr(os, "posix_fallocate"):
            try:
                os.posix_fallocate(fd, 0, size)
                return
            except OSError as exc:
                if exc.errno not in (errno.EINVAL, errno.ENOSYS, errno.EOPNOTSUPP):
                    raise
        os.ftruncate(fd, size)
    except OSError as exc:
        raise _storage_error(directory, size, cause=exc) from exc


def _write_frame(image, stream):
    np.asarray(image).tofile(stream)


def _create_temporary_file(directory, size):
    try:
        return tempfile.mkstemp(prefix="openfocus-frame-", suffix=".bin", dir=directory)
    except OSError as exc:
        raise _storage_error(directory, size, cause=exc) from exc



def create_frame(shape, dtype, *, directory=None):
    """Create an empty temporary mapping owned by its NumPy views."""
    directory = storage_directory() if directory is None else directory
    size = math.prod(shape) * np.dtype(dtype).itemsize
    require_disk_space(size, directory)
    fd, path = _create_temporary_file(directory, size)
    try:
        _allocate_file(fd, size, directory)
        os.close(fd)
        fd = -1
        mapped = np.memmap(path, dtype=dtype, mode="r+", shape=shape)
        _own_frame(mapped, path)
        return mapped
    except TemporaryStorageError:
        if fd >= 0:
            os.close(fd)
        _remove(path)
        raise
    except OSError as exc:
        if fd >= 0:
            os.close(fd)
        _remove(path)
        raise _storage_error(directory, size, cause=exc) from exc
    except BaseException:
        if fd >= 0:
            os.close(fd)
        _remove(path)
        raise


def store_frame(image):
    """Copy one frame to a temporary mapping, preserving dtype and shape.

    The mapping, including any derived views, owns the file. It is removed
    after the last view is released, including on cancellation or exceptions.
    """
    directory = storage_directory()
    require_disk_space(image.nbytes, directory)
    fd, path = _create_temporary_file(directory, image.nbytes)
    try:
        # Write through the file descriptor before mapping it. Filling a writable
        # mapping faults every source-stack page into this process even though
        # the caller only needs one frame/strip at a time afterward.
        _allocate_file(fd, image.nbytes, directory)
        with os.fdopen(fd, "r+b") as stream:
            fd = -1
            try:
                _write_frame(image, stream)
                stream.flush()
            except OSError as exc:
                try:
                    written = stream.tell()
                except OSError:
                    written = os.path.getsize(path)
                raise _storage_error(directory, image.nbytes, written=written,
                                     cause=exc) from exc
        mapped = np.memmap(path, dtype=image.dtype, mode="r+", shape=image.shape)
        _own_frame(mapped, path)
        return mapped
    except TemporaryStorageError:
        if fd >= 0:
            os.close(fd)
        _remove(path)
        raise
    except OSError as exc:
        if fd >= 0:
            os.close(fd)
        _remove(path)
        raise _storage_error(directory, image.nbytes, cause=exc) from exc
    except BaseException:
        if fd >= 0:
            os.close(fd)
        _remove(path)
        raise
