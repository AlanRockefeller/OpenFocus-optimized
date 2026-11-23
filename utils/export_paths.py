"""Collision-safe names for image and stack exports."""
import os
import re
import errno
import logging
import tempfile

from utils.file_modes import apply_default_mode


def _rename_noreplace(source, destination):
    """Use the platform's atomic no-replace rename when links are unavailable."""
    import ctypes
    import sys
    libc = ctypes.CDLL(None, use_errno=True)
    if sys.platform.startswith('linux') and hasattr(libc, 'renameat2'):
        fn = libc.renameat2
        args = (-100, os.fsencode(source), -100, os.fsencode(destination), 1)
    elif sys.platform == 'darwin' and hasattr(libc, 'renamex_np'):
        fn = libc.renamex_np
        args = (os.fsencode(source), os.fsencode(destination), 4)
    else:
        raise OSError(errno.ENOTSUP, 'Atomic no-replace rename is unavailable')
    if fn(*args) != 0:
        code = ctypes.get_errno()
        raise OSError(code, os.strerror(code), destination)


def publish_unique(path, writer, *, start=2, should_cancel=None):
    """Encode once, then atomically publish to the first unoccupied name."""
    path = os.fspath(path)
    directory = os.path.dirname(os.path.abspath(path))
    stem, extension = os.path.splitext(path)
    fd, staging = tempfile.mkstemp(prefix='.openfocus-staging-', suffix=extension,
                                   dir=directory)
    os.close(fd)
    published = False
    try:
        if writer(staging) is False:
            raise OSError('Image encoding or writing failed')
        # mkstemp made the staging file owner-only; a new export gets the
        # mode a plain open() would have given it.
        apply_default_mode(staging)
        if should_cancel and should_cancel():
            raise InterruptedError('Export cancelled')
        number = start
        candidate = path
        while True:
            if should_cancel and should_cancel():
                raise InterruptedError('Export cancelled')
            try:
                if os.name == 'nt':
                    os.rename(staging, candidate)
                    staging = None
                else:
                    try:
                        os.link(staging, candidate)
                    except OSError as exc:
                        if exc.errno not in (errno.EPERM, errno.ENOTSUP, errno.EOPNOTSUPP,
                                             errno.ENOSYS, errno.EXDEV):
                            raise
                        _rename_noreplace(staging, candidate)
                        staging = None
                published = True
                return candidate
            except FileExistsError:
                candidate = f'{stem}_{number}{extension}'
                number += 1
    finally:
        if staging is not None:
            try:
                os.unlink(staging)
            except FileNotFoundError:
                pass
            except OSError:
                if not published:
                    raise
                logging.getLogger(__name__).warning(
                    'Published export but could not remove staging file %s', staging,
                    exc_info=True)


# Suffixes this app appends when it writes an aligned copy of a frame. A
# source folder that once received them (older builds wrote `_registered`
# files beside the originals) must not feed them back into the next stack:
# 9 frames silently became 36, and every later batch fused duplicates.
_ALIGNED_SUFFIX = re.compile(r'^(?P<stem>.+?)(?:_registered)+(?:_(?P<n>\d+))?$',
                             re.IGNORECASE)


def valid_output_prefix(prefix: str) -> bool:
    """Accept a short portable prefix, or an empty prefix for the default."""
    return (isinstance(prefix, str) and len(prefix) <= 40
            and all(char.isalnum() or char in '_-' for char in prefix))


def prefix_output_name(prefix: str, name: str) -> str:
    """Apply only valid saved prefixes; damaged settings keep the default name."""
    return f"{prefix}_{name}" if prefix and valid_output_prefix(prefix) else name


def unique_path(path: str) -> str:
    """Preserve every existing target, including dangling links."""
    stem, extension = os.path.splitext(path)
    candidate = path
    number = 2
    while os.path.lexists(candidate):
        candidate = f"{stem}_{number}{extension}"
        number += 1
    return candidate


def create_stack_directory(destination: str, name: str) -> str:
    """Atomically reserve one new directory per stack."""
    name = re.sub(r'[\\/:*?"<>|]', '_', os.path.basename(name)).strip('. ') or 'Stack'
    os.makedirs(destination, exist_ok=True)
    number = 1
    while True:
        candidate = os.path.join(destination, name if number == 1 else f"{name}_{number}")
        try:
            os.mkdir(candidate)
            return candidate
        except FileExistsError:
            number += 1


def frame_export_name(filename: str, index: int, extension: str = '') -> str:
    """Keep original stems/page labels; an explicit format changes only the suffix."""
    name = os.path.basename(filename.replace('\\', '/')) if filename else f'frame_{index + 1:04d}.png'
    name = re.sub(r'[<>:"/\\|?*]', '_', name).strip('. ')
    stem, suffix = os.path.splitext(name)
    suffix = ('.' + extension.lstrip('.').lower()) if extension else suffix
    return (stem or f'frame_{index + 1:04d}') + (suffix or '.png')


def derived_aligned_source(name: str) -> str:
    """Stem of the original a `*_registered*` file was derived from, else ''.

    `01_registered.jpg`, `01_registered_2.jpg` (collision suffix) and
    `01_registered_registered.jpg` (a copy of a copy) all map to `01`.
    """
    stem = os.path.splitext(os.path.basename(name.replace('\\', '/')))[0]
    match = _ALIGNED_SUFFIX.match(stem)
    return match.group('stem') if match else ''


def batch_output_dir(source_folder: str, output_type: str, output_path) -> str:
    """Folder a batch task writes its results into; it is not created here."""
    if output_type == 'subfolder':
        root = os.path.realpath(source_folder)
        output_dir = os.path.realpath(os.path.join(root, output_path))
        if os.path.isabs(output_path) or os.path.commonpath((root, output_dir)) != root:
            raise ValueError('Output subfolder must be inside the source folder')
        return output_dir
    if output_type == 'same':
        return source_folder
    return output_path


def batch_result_among_inputs(folders, stacks, output_type, output_path, extension,
                              exclude_paths=()) -> str:
    """Return a fused result the batch would write that is already one of its frames.

    A result written into a source folder (the "same folder" option, or an
    output folder that is also a source) is listed with that folder's frames
    on the next run, so it would be fused into the new result. `folders` are
    the folders of a multiple-folder batch; `stacks` holds the frame paths of
    each split stack instead. Only exact paths are compared. Returns '' when
    no planned result is an input.
    """
    from utils.path_identity import same_path
    extension = str(extension or 'png').lstrip('.').lower()
    exclude = [path for path in exclude_paths if path]
    planned, inputs = [], []
    if stacks is not None:
        inputs = [path for stack in stacks for path in stack if path]
        source = os.path.dirname(stacks[0][0]) if stacks and stacks[0] else ''
        try:
            output_dir = batch_output_dir(source, output_type, output_path)
        except ValueError:
            return ''
        if not source or not output_dir:
            return ''
        name = os.path.basename(source)
        planned = [os.path.join(output_dir, f'{name}_{index + 1:03d}.{extension}')
                   for index in range(len(stacks))]
    else:
        from core.image_loader import ImageStackLoader
        loader = ImageStackLoader()
        for folder in folders:
            if not os.path.isdir(folder):
                continue
            inputs += [path for _name, path in loader.folder_image_files(folder, exclude)]
            try:
                output_dir = batch_output_dir(folder, output_type, output_path)
            except ValueError:
                continue
            if not output_dir:
                continue
            name = os.path.basename(os.path.normpath(folder))
            planned.append(os.path.join(output_dir, f'{name}.{extension}'))
    for result in planned:
        if os.path.isfile(result) and any(same_path(result, path) for path in inputs):
            return result
    return ''
