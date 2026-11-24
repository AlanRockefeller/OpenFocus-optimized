"""Permission bits for files written through a temporary file.

``tempfile.mkstemp`` creates files readable only by their owner, and
``os.replace``/``os.link`` keep that mode. A file the user saves should
instead get the mode a plain ``open()`` would have given it: the existing
destination's mode when it is overwritten, otherwise ``0o666`` less the
umask. Private files (account data, crash-recovery snapshots, caches) keep
the owner-only mode and do not call these helpers.
"""
import os
import stat


def _read_umask():
    """Return the process umask without changing it where the OS allows."""
    if os.name == "nt":
        return 0
    try:
        with open("/proc/self/status", encoding="ascii") as status:
            for line in status:
                if line.startswith("Umask:"):
                    return int(line.split()[1], 8)
    except (OSError, ValueError, IndexError):
        pass
    # No /proc (macOS, BSD): read it by setting and restoring it. This runs
    # once, when the module is first imported during startup.
    mask = os.umask(0o022)
    os.umask(mask)
    return mask


_UMASK = _read_umask()


def default_file_mode():
    """Mode of a new file created by ``open()`` under the startup umask."""
    return 0o666 & ~_UMASK


def _chmod(path, mode):
    try:
        os.chmod(path, mode)
    except OSError:
        # Some file systems (for example FAT or a Windows volume mounted
        # without permission support) refuse chmod; keep the file as written.
        pass


def apply_default_mode(path):
    """Give a staged file that publishes under a new name the default mode."""
    if os.name != "nt":
        _chmod(path, default_file_mode())


def apply_destination_mode(path, destination):
    """Give a staged file the mode overwriting ``destination`` would keep.

    Call this before replacing ``destination`` with ``path``. An existing
    regular file keeps its permission bits; otherwise the default mode applies.
    """
    if os.name == "nt":
        return
    try:
        info = os.stat(destination)
    except OSError:
        info = None
    if info is not None and stat.S_ISREG(info.st_mode):
        mode = stat.S_IMODE(info.st_mode) & 0o777
    else:
        mode = default_file_mode()
    _chmod(path, mode)


def apply_default_directory_mode(path):
    """Give a directory made by ``mkdtemp`` the mode ``os.makedirs`` gives."""
    if os.name != "nt":
        _chmod(path, 0o777 & ~_UMASK)
