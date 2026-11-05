"""EXIF orientation must be applied exactly once, and exports must not keep
the source orientation tag (upright pixels + "rotate 90°" tag = rotated twice
in every viewer).

Regression: OpenCV >= 4.5 already honours the tag in imdecode/imread; the
loader then applied it again, so portrait phone shots loaded sideways.
"""
import numpy as np
import pytest
from PIL import Image, ImageOps


def _blocks(h=48, w=72):
    arr = np.zeros((h, w, 3), np.uint8)
    arr[: h // 2, : w // 2] = (220, 20, 20)     # red, top-left
    arr[h // 2 :, w // 2 :] = (20, 220, 20)     # green, bottom-right
    arr[: h // 3, w // 2 :] = (20, 20, 220)     # blue, top-right
    arr[h // 2 :, : w // 3] = (230, 230, 230)   # white, bottom-left
    return arr


def _write_jpeg(path, orientation, rgb):
    Image.fromarray(rgb).save(path, quality=100)
    im = Image.open(path)
    ex = im.getexif()
    ex[274] = orientation
    im.save(path, exif=ex, quality=100)


@pytest.mark.parametrize("orientation", range(1, 9))
def test_decode_image_applies_orientation_exactly_once(tmp_path, orientation):
    path = str(tmp_path / f"o{orientation}.jpg")
    _write_jpeg(path, orientation, _blocks())

    with Image.open(path) as im:
        # RGB, as viewers show it
        expected = np.array(ImageOps.exif_transpose(im))

    from core.image_loader import decode_image
    got = decode_image(path)
    assert got is not None
    assert got.shape == expected.shape
    # JPEG decoders differ by a few levels; orientation errors are massive.
    diff = np.abs(
        got[:, :, ::-1].astype(np.int16) - expected.astype(np.int16))
    assert float(diff.mean()) <= 3.0, f"orientation {orientation} decoded wrong"


def test_export_neutralizes_orientation_tag(tmp_path):
    src = str(tmp_path / "portrait.jpg")
    _write_jpeg(src, 6, _blocks())

    from core.image_loader import decode_image, _read_exif_bytes
    from utils.image_utils import imwrite_auto

    upright = decode_image(src)
    with Image.open(src) as im:
        expected = np.array(ImageOps.exif_transpose(im))
    assert upright.shape == expected.shape

    out = str(tmp_path / "export.png")
    assert imwrite_auto(out, upright, exif=_read_exif_bytes(src))

    with Image.open(out) as im:
        assert im.getexif().get(274, 1) == 1
        shown = np.array(im.convert("RGB"))
    # A viewer honouring the (now neutral) tag must show exactly the upright
    # pixels the application produced.
    assert np.array_equal(shown, upright[:, :, ::-1])


def test_png_orientation_applied_once(tmp_path):
    """PNG eXIf orientation must also be applied by our single explicit pass;
    cv2 would otherwise honour it (we disable that) and rotate twice."""
    path = str(tmp_path / "o6.png")
    rgb = _blocks()
    Image.fromarray(rgb).save(path)
    im = Image.open(path)
    ex = im.getexif()
    ex[274] = 6
    im.save(path, exif=ex)

    with Image.open(path) as im:
        expected = np.array(ImageOps.exif_transpose(im))

    from core.image_loader import decode_image
    got = decode_image(path)
    assert got.shape == expected.shape
    assert np.array_equal(got[:, :, ::-1], expected)
