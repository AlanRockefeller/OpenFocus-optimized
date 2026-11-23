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


def run_batch(folders, output_type, output_path=None, *, method='guided_filter',
              save_aligned=False, image_format='png', **options):
    settings = {'format': image_format, 'fusion_method': method, 'reg_methods': [],
                'save_aligned': save_aligned}
    settings.update(options.pop('settings', {}))
    worker = BatchWorker([str(path) for path in folders], output_type, output_path, settings,
                         use_gpu=False, thread_count=1, **options)
    result = {}
    worker.finished.connect(result.update)
    worker.error.connect(lambda message: result.update(error=message))
    worker.run()
    return worker, result


def test_batch_aligned_frames_keep_their_own_exif(qapp, tmp_path):
    source = make_stack(tmp_path / 'stack')
    _worker, result = run_batch([source], 'custom', str(tmp_path / 'out'),
                                save_aligned=True, image_format='jpg')
    assert result.get('counts', {}).get('success') == 1, result
    assert read_make(tmp_path / 'out' / 'stack.jpg') == 'CameraOne'
    [aligned] = result['aligned_stacks']
    assert [read_make(path) for path in aligned['files']] == ['CameraOne', 'CameraTwo']


def test_split_batch_aligned_frames_keep_their_own_exif(qapp, tmp_path):
    source = make_stack(tmp_path / 'stack')
    loader = ImageStackLoader()
    ok, _message, entries, _names = loader.load_images_with_timestamps(str(source))
    assert ok
    _worker, result = run_batch([], 'custom', str(tmp_path / 'out'), method='none',
                                image_format='jpg', import_mode='single_folder',
                                single_folder_images_with_times=entries,
                                settings={'frozen_stacks': loader.split_by_count(entries, 5)})
    assert result.get('counts', {}).get('success') == 1, result
    [aligned] = result['aligned_stacks']
    assert [read_make(path) for path in aligned['files']] == ['CameraOne', 'CameraTwo']


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
