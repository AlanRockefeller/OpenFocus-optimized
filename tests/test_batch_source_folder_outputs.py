"""Batch results written into a source folder, and the metadata they carry."""
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from PIL import Image

from core.cli import run_cli
from core.image_loader import ImageStackLoader
from core.workers import BatchWorker


def _write_frame(path, value, make):
    image = Image.fromarray(np.full((16, 24, 3), value, np.uint8))
    exif = image.getexif()
    exif[271] = make  # Make
    image.save(path, exif=exif)


def make_stack(folder, makes=('CameraOne', 'CameraTwo'), suffix='.jpg'):
    folder.mkdir(parents=True)
    for index, make in enumerate(makes):
        _write_frame(folder / f'img{index + 1}{suffix}', 60 + 40 * index, make)
    return folder


def read_make(path):
    with Image.open(path) as image:
        return image.getexif().get(271)


@pytest.mark.parametrize('batch', [False, True])
def test_cli_output_keeps_the_first_frame_exif(tmp_path, monkeypatch, batch):
    # A fresh CLI process starts without the single-stack source EXIF; an
    # earlier in-process run must not supply it.
    monkeypatch.setattr('utils.image_utils.SOURCE_EXIF', b'')
    source = make_stack(tmp_path / 'stack')
    if batch:
        destination, output = ['--output-dir', str(tmp_path / 'out')], tmp_path / 'out' / 'stack.png'
    else:
        destination, output = ['--output', str(tmp_path / 'out.png')], tmp_path / 'out.png'
    assert run_cli(['--input', str(source), *destination,
                    '--method', 'guided_filter', '--experimental', '--align', 'none']) == 0
    assert read_make(output) == 'CameraOne'
