"""Turn low-level processing failures into actionable user-facing text."""
import errno
import re


_PARTIAL_WRITE = re.compile(r"^(\d+) requested and (\d+) written$")


def processing_error(error):
    """Return a useful message while retaining the original diagnostic."""
    message = str(error).strip()
    match = _PARTIAL_WRITE.match(message)
    if match:
        requested, written = (int(value) for value in match.groups())
        return (
            "A temporary array write was incomplete: "
            f"{requested:,} array elements requested and {written:,} written. "
            "The element size depends on the image bit depth (for 16-bit data, multiply by "
            "2 to obtain bytes). The temporary disk is probably full or its quota was reached."
        )
    if isinstance(error, MemoryError) and not message:
        return "The render worker ran out of memory (RAM); no additional details were provided."
    if isinstance(error, OSError) and getattr(error, "errno", None) in (errno.ENOSPC, errno.EDQUOT):
        reason = "disk space" if error.errno == errno.ENOSPC else "storage quota"
        return f"The render worker ran out of temporary {reason}: {message}"
    return message or f"{type(error).__name__} occurred without an error message"

