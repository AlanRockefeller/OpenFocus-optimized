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


def test_same_folder_rerun_refuses_the_earlier_result(qapp, tmp_path):
    source = make_stack(tmp_path / 'stack')
    _worker, first = run_batch([source], 'same')
    assert first.get('counts', {}).get('success') == 1, first
    assert (source / 'stack.png').is_file()

    _worker, second = run_batch([source], 'same')
    assert 'error' in second, second
    assert str(source / 'stack.png') in second['error']
    assert sorted(path.name for path in source.iterdir()) == ['img1.jpg', 'img2.jpg', 'stack.png']


def test_split_same_folder_rerun_refuses_the_earlier_result(qapp, tmp_path):
    source = make_stack(tmp_path / 'stack')

    def split_run():
        loader = ImageStackLoader()
        ok, _message, entries, _names = loader.load_images_with_timestamps(str(source))
        assert ok
        return run_batch([], 'same', import_mode='single_folder',
                         single_folder_images_with_times=entries,
                         settings={'frozen_stacks': loader.split_by_count(entries, 5)})

    _worker, first = split_run()
    assert first.get('counts', {}).get('success') == 1, first
    assert (source / 'stack_001.png').is_file()
    _worker, second = split_run()
    assert 'error' in second, second
    assert str(source / 'stack_001.png') in second['error']
    assert not (source / 'stack_001_2.png').exists()


def test_result_written_into_a_later_source_folder_is_not_fused_there(qapp, tmp_path):
    first = make_stack(tmp_path / 'first')
    second = make_stack(tmp_path / 'second')
    worker = BatchWorker([str(first), str(second)], 'custom', str(second),
                         {'format': 'png', 'fusion_method': 'guided_filter', 'reg_methods': []},
                         use_gpu=False, thread_count=1)
    loaded = []
    original = worker.image_loader.load_from_folder

    def recording(folder, *args, **kwargs):
        outcome = original(folder, *args, **kwargs)
        loaded.append(sorted(outcome[3]))
        return outcome

    worker.image_loader.load_from_folder = recording
    result = {}
    worker.finished.connect(result.update)
    worker.error.connect(lambda message: result.update(error=message))
    worker.run()
    assert result.get('counts', {}).get('success') == 2, result
    assert loaded == [['img1.jpg', 'img2.jpg'], ['img1.jpg', 'img2.jpg']]
    assert sorted(path.name for path in second.iterdir()) == [
        'first.png', 'img1.jpg', 'img2.jpg', 'second.png']


def _dialog_stub(folders=(), stacks=(), method='guided_filter', image_format='PNG'):
    from dialogs.batch import BatchProcessingDialog
    stub = SimpleNamespace(
        folder_paths=[str(path) for path in folders],
        single_folder_stacks=[list(stack) for stack in stacks],
        single_folder_folder_path=str(Path(stacks[0][0][0]).parent) if stacks else '',
        fusion_combo=SimpleNamespace(currentData=lambda: method),
        format_combo=SimpleNamespace(currentText=lambda: image_format),
        sensor_cleanup=SimpleNamespace(path=SimpleNamespace(text=lambda: '')),
        get_import_mode=lambda: 'single_folder' if stacks else 'multiple_folders')
    for name in ('_source_output_folders', '_existing_result_input'):
        setattr(stub, name, getattr(BatchProcessingDialog, name).__get__(stub))
    return stub


@pytest.mark.parametrize('output_type', ['same', 'custom'])
def test_batch_dialog_refuses_a_result_that_is_already_an_input(qapp, tmp_path, output_type):
    from dialogs.batch import BatchProcessingDialog
    source = make_stack(tmp_path / 'stack')
    output_path = str(source) if output_type == 'custom' else None
    stub = _dialog_stub([source])
    assert BatchProcessingDialog._validate_output_target(stub, output_type, output_path) is None
    _write_frame(source / 'stack.png', 10, 'Earlier')
    problem = BatchProcessingDialog._validate_output_target(stub, output_type, output_path)
    assert problem and str(source / 'stack.png') in problem
    # Without fusion only aligned-frame folders are written; a JPG plan does
    # not reach the PNG either.
    assert BatchProcessingDialog._validate_output_target(
        _dialog_stub([source], method='none'), output_type, output_path) is None
    assert BatchProcessingDialog._validate_output_target(
        _dialog_stub([source], image_format='JPG'), output_type, output_path) is None


def test_batch_dialog_refuses_an_earlier_split_result(qapp, tmp_path):
    from dialogs.batch import BatchProcessingDialog
    source = make_stack(tmp_path / 'stack')
    _write_frame(source / 'stack_001.png', 10, 'Earlier')
    loader = ImageStackLoader()
    ok, _message, entries, _names = loader.load_images_with_timestamps(str(source))
    assert ok and len(entries) == 3
    stub = _dialog_stub(stacks=loader.split_by_count(entries, 5))
    problem = BatchProcessingDialog._validate_output_target(stub, 'same', None)
    assert problem and str(source / 'stack_001.png') in problem
    assert BatchProcessingDialog._validate_output_target(
        stub, 'subfolder', 'results') is None


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
