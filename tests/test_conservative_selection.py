"""Deterministic fusion-policy tests, independent of ONNX availability."""
import cv2
import numpy as np
import pytest

from fusion_methods.conservative_selection import selection_rows
from fusion_methods.stackmffv3 import _blend_tile


def select(frames, probs=None, **kwargs):
    h, w = frames[0].shape[:2]
    if probs is None:
        probs = np.empty((len(frames), h, w), np.float32)
        probs[0] = .01
        probs[1:] = .99 / (len(frames) - 1)
    maps = {}
    for lo, hi, selected, planes in selection_rows(probs, frames, **kwargs):
        for name, plane in planes.items():
            maps.setdefault(name, np.empty((h, w), plane.dtype))[lo:hi] = plane
    return maps


def scene(h=65, w=79):
    subject = np.full((h, w), 100, np.uint8)
    subject[15:50, 20:55] = 180  # smooth cap interior
    subject[8:57, 60] = 240  # one-pixel hair
    for x in range(24, 52, 4):
        subject[35:48, x] = 60  # gills
    return [cv2.GaussianBlur(subject, (21, 21), 5), subject]


@pytest.mark.parametrize('fallback', ['first', 'last'])
@pytest.mark.parametrize('textured', [False, True])
def test_blurry_background_stays_one_source(fallback, textured):
    yy, xx = np.indices((65, 79))
    base = (90 + 5 * np.sin(xx / 15) * np.cos(yy / 17)).astype(np.uint8) if textured else np.full((65, 79), 90, np.uint8)
    frames = [base, base + 30]
    maps = select(frames, fallback=fallback)
    assert not maps['accepted'].any()  # despite 99% confidence
    out = _blend_tile(np.stack([np.full(base.shape, .01), np.full(base.shape, .99)]), frames, 'conservative', fallback=fallback)
    np.testing.assert_array_equal(out[..., 0], frames[0 if fallback == 'first' else 1])


def test_edges_hair_and_nearby_smooth_interior_survive():
    maps = select(scene())
    assert maps['accepted'][12:53, 60].mean() > .9
    assert maps['accepted'][15:19, 23:52].mean() > .8
    assert maps['accepted'][36:47, 24:51].mean() > .9
    assert not maps['accepted'][:4, :8].any()


def test_decisive_detailed_subject_does_not_need_sharpness_advantage():
    """The fallback is a background guard, not the default for the subject."""
    base = np.full((65, 79), 90, np.uint8)
    yy, xx = np.indices(base.shape)
    subject = (((xx // 3 + yy // 3) % 2) * 80 + 100).astype(np.uint8)
    first = base.copy()
    last = base.copy()
    first[16:50, 20:59] = subject[16:50, 20:59]
    # Similar local contrast means the old AND gate rejected this decisive
    # source choice even though Hard Pick selected it confidently.
    last[16:50, 20:59] = np.clip(
        subject[16:50, 20:59].astype(np.int16) + 8, 0, 255).astype(np.uint8)
    maps = select([first, last], sharpness_advantage=10)
    assert maps['accepted'][20:46, 24:55].mean() > .95
    assert maps['confidence_accepted'][20:46, 24:55].mean() > .95
    assert not maps['accepted'][:8, :8].any()


def test_coherent_hard_subject_survives_weak_numeric_evidence():
    frames = scene()
    probs = np.empty((2, *frames[0].shape), np.float32)
    probs[0] = .49
    probs[1] = .51
    maps = select(frames, probs, confidence=.9, sharpness_advantage=10)
    assert maps['structure_accepted'][16:49, 21:54].mean() > .8
    assert maps['accepted'][16:49, 21:54].mean() > .8
    assert not maps['accepted'][:4, :8].any()


def test_thresholds_are_monotone_and_fallback_predictable():
    frames = scene()
    previous = np.ones(frames[0].shape, bool)
    for confidence in (0, .3, .6, .9, 1):
        current = select(frames, confidence=confidence)['accepted']
        assert not (current & ~previous).any()
        previous = current
    previous = np.ones(frames[0].shape, bool)
    for advantage in (0, .5, 1, 3, 10):
        current = select(frames, sharpness_advantage=advantage)['accepted']
        assert not (current & ~previous).any()
        previous = current
    assert not select(frames, fallback='last')['accepted'].any()


def test_relative_confidence_and_runner_up_margin():
    frames = scene()
    equal = np.full((2, *frames[0].shape), .5, np.float32)
    assert not select(frames, equal)['accepted'].any()
    many = [frames[0]] * 9 + [frames[1]]
    probs = np.full((10, *frames[0].shape), .04, np.float32)
    probs[-1] = .64  # normalized probability lift ~.892, sufficient margin
    assert select(many, probs, confidence=.89)['confidence_accepted'].any()
    assert not select(many, probs, confidence=.90)['confidence_accepted'].any()


def test_isolated_predictions_removed_but_line_retained():
    frames = scene()
    probs = np.zeros((2, *frames[0].shape), np.float32)
    probs[0] = 1
    probs[:, 30, 25] = [.01, .99]
    probs[:, 12:53, 60] = np.array([.01, .99])[:, None]
    maps = select(frames, probs)
    assert not maps['accepted'][30, 25]
    assert maps['accepted'][15:50, 60].all()


@pytest.mark.parametrize('dtype', [np.uint8, np.uint16])
@pytest.mark.parametrize('channels', [None, 1, 3])
def test_source_provenance_noncontiguous_odd_dimensions(dtype, channels):
    from utils.image_utils import quantize_fuse_output
    frames = scene()
    if dtype == np.uint16:
        frames = [f.astype(dtype) * 251 + 101 for f in frames]
    if channels:
        frames = [np.repeat(f[..., None], channels, axis=2) for f in frames]
    frames = [np.repeat(f, 2, axis=1)[:, ::2] for f in frames]
    probs = np.stack([np.full((65, 79), .01), np.full((65, 79), .99)])
    out = _blend_tile(probs, frames, 'conservative')
    if channels is None:
        out = out[..., 0]
    out = quantize_fuse_output(out, dtype == np.uint16)
    assert np.logical_or.reduce([out == f for f in frames]).all()
    if dtype == np.uint16:
        assert np.any(out % 257)


@pytest.mark.parametrize('polarity', [-1, 1])
def test_halo_band_is_fallback_when_no_detail_is_observed(polarity):
    # Known background, with a spurious dark/light defocus halo in alternative.
    yy, xx = np.indices((65, 79))
    truth = np.full((65, 79), 120, np.uint8)
    spill = polarity * 40 * np.exp(-((xx - 40) / 12) ** 2)
    frames = [truth, np.clip(truth + spill, 0, 255).astype(np.uint8)]
    probs = np.stack([np.full(truth.shape, .01), np.full(truth.shape, .99)])
    result = _blend_tile(probs, frames, 'conservative')[..., 0]
    assert np.abs(result - truth).mean() < np.abs(frames[1].astype(float) - truth).mean()
    # If every frame contains spill, selection cannot recover missing pixels.
    result = _blend_tile(probs, [frames[1], frames[1]], 'conservative')[..., 0]
    np.testing.assert_array_equal(result, frames[1])


def test_row_halos_match_one_piece(monkeypatch):
    import fusion_methods.conservative_selection as policy
    frames = scene(517, 79)
    a = select(frames)
    monkeypatch.setattr(policy, 'ROW_SIZE', 1024)
    b = select(frames)
    for key in a:
        np.testing.assert_array_equal(a[key], b[key])


def test_cancel_during_scoring():
    from core.registration import RegistrationCancelled
    calls = 0
    def cancel():
        nonlocal calls
        calls += 1
        return calls > 4
    with pytest.raises(RegistrationCancelled):
        select(scene(), should_cancel=cancel)


@pytest.mark.parametrize('key,value', [('fallback', 'auto'), ('confidence', -1), ('confidence', float('nan')), ('confidence', True), ('sharpness_advantage', 11), ('sharpness_advantage', '0.5')])
def test_validation(key, value):
    with pytest.raises(ValueError):
        select(scene(), **{key: value})


def test_parameter_panel_and_persistence(qapp):
    from core.algorithms import get_fusion
    from ui.algorithm_parameters import AlgorithmParameterPanel
    from utils.settings_store import parameter_memory_from_json, parameter_memory_to_json
    panel = AlgorithmParameterPanel()
    panel.set_parameters(get_fusion('stackmffv3').parameters)
    assert panel.editors['fallback'].isHidden()
    panel.editors['mode'].setCurrentIndex(2)
    assert not panel.editors['fallback'].isHidden()
    panel.editors['fallback'].setCurrentIndex(1)
    values = panel.values()
    saved = {'stackmffv3': values}
    assert parameter_memory_from_json(parameter_memory_to_json(saved)) == saved
    panel.editors['mode'].setCurrentIndex(0)
    assert panel.editors['fallback'].isHidden()
    assert panel.values()['fallback'] == 'last'


def test_cli_and_config_validation():
    from core.cli import build_parser, _resolve_algorithm_arguments
    args = build_parser().parse_args(['--input', 'unused', '--ai-mode', 'conservative', '--param', 'fallback="last"', '--param', 'confidence=0.8'])
    _resolve_algorithm_arguments(args)
    assert args._fusion_parameters['fallback'] == 'last'
    assert args._fusion_parameters['confidence'] == .8
    args.fusion_parameters = {'sharpness_advantage': 1.2}
    _resolve_algorithm_arguments(args)
    assert args._fusion_parameters['sharpness_advantage'] == 1.2
    args.depth_map = True
    with pytest.raises(ValueError, match='focus-depth'):
        _resolve_algorithm_arguments(args)


def test_tile_ownership_and_diagnostics_are_deterministic(monkeypatch):
    pytest.importorskip('onnxruntime')
    import core.multi_focus_fusion as mff
    frames = [np.full((71, 109, 3), value, np.uint8) for value in (19, 237)]
    fusion = mff.MultiFocusFusion('stackmffv3', use_gpu=False, exposure_match=False)
    def render():
        choices = iter([0, 1] * 100)
        provenance = np.full((71,109), -1, np.int32)
        def diagnostics(row, stop, maps, x=0, mask=None):
            np.copyto(provenance[row:stop, x:x+maps['selected'].shape[1]], maps['selected'], where=mask)
        def predict(_fusion, crops, _resize, **kwargs):
            index = next(choices)
            shape = crops[0].shape[:2]
            kwargs['diagnostics'](0, shape[0], {'selected': np.full(shape,index,np.int32)})
            return crops[index]
        monkeypatch.setattr(mff, 'resolve_runner', lambda _: predict)
        out = fusion._fuse_tiled_neural(frames, None, [], 71,109,3,48,0,
            prepare=lambda *a,**k:(None,'CPU',None), label='test',
            default_weights='weights/stackmff_v3_final.onnx', mode_normalizer=lambda x:x,
            mode='conservative', diagnostics=diagnostics)
        assert np.isin(out, (19,237)).all()
        np.testing.assert_array_equal(out[...,0], np.where(provenance == 0,19,237))
        assert np.all(provenance >= 0)
        return out
    np.testing.assert_array_equal(render(),render())


def test_tiled_and_whole_fixed_probabilities_match(monkeypatch):
    pytest.importorskip('onnxruntime')
    from fusion_methods import stackmffv3 as ai
    from core.multi_focus_fusion import MultiFocusFusion
    def probabilities(batch, *args, **kwargs):
        p = np.empty_like(batch)
        p[:,0] = .01
        p[:,1] = .99
        return p
    monkeypatch.setattr(ai, '_probabilities', probabilities)
    frames = [np.repeat(f[...,None],3,axis=2) for f in scene(97,113)]
    outputs = []
    for tiled in (False, True):
        fusion = MultiFocusFusion('stackmffv3', use_gpu=False, exposure_match=False,
            fusion_parameters={'mode':'conservative'}, tile_enabled=tiled,
            tile_block_size=64, tile_overlap=0, tile_threshold=1)
        outputs.append(fusion.fuse(frames, thread_count=1))
    np.testing.assert_array_equal(*outputs)


def test_real_onnx_conservative_resource_and_depth_contract(monkeypatch):
    pytest.importorskip('onnxruntime')
    from core import resource_policy
    from core.depth_map import compute_focus_index
    from fusion_methods.stackmffv3 import _stackmffv3_batch_impl
    frames = scene()
    out = _stackmffv3_batch_impl([frames], 'weights/stackmff_v3_final.onnx', False,
                               'conservative', thread_count=1)[0]
    assert out.dtype == np.uint8
    assert np.logical_or.reduce([out == f for f in frames]).all()
    with pytest.raises(ValueError, match='focus-depth'):
        compute_focus_index(frames, 'stackmffv3', parameters={'mode':'conservative'})
    monkeypatch.setattr(resource_policy, 'probe_resources', lambda *a, **k:
        resource_policy.ResourceSnapshot(available_ram=1024,total_ram=8*1024**3))
    with pytest.raises(resource_policy.ResourceLimitError):
        _stackmffv3_batch_impl([frames], 'weights/stackmff_v3_final.onnx', False,'conservative')


@pytest.mark.parametrize('tiled', [False, True])
def test_public_last_fallback_diagnostics_match_uint16_output(tiled):
    pytest.importorskip('onnxruntime')
    from core.multi_focus_fusion import MultiFocusFusion
    from utils.image_utils import quantize_fuse_output
    frames = [np.repeat((f.astype(np.uint16)*251+101)[...,None],3,axis=2) for f in scene()]
    selected = np.full(frames[0].shape[:2], -1, np.int32)
    def diagnostics(row, stop, maps, x=0, mask=None):
        target = selected[row:stop,x:x+maps['selected'].shape[1]]
        np.copyto(target,maps['selected'],where=True if mask is None else mask)
    fusion = MultiFocusFusion('stackmffv3',use_gpu=False,exposure_match=False,
        fusion_parameters={'mode':'conservative','fallback':'last','confidence':1.,
                           'sharpness_advantage':10.},
        tile_enabled=tiled,tile_block_size=64,tile_overlap=0,tile_threshold=1)
    out = quantize_fuse_output(fusion.fuse(frames,thread_count=1,diagnostics=diagnostics),True)
    np.testing.assert_array_equal(out,frames[-1])
    assert (selected == 1).all()


def test_batch_config_end_to_end_and_invalid_range(tmp_path):
    import json
    import subprocess
    import sys
    pytest.importorskip('onnxruntime')
    stack = tmp_path / 'stack'
    stack.mkdir()
    for i, level in enumerate((12001, 40003)):
        assert cv2.imwrite(str(stack / f'{i}.tif'), np.full((17,19,3),level,np.uint16))
    config = tmp_path / 'config.json'
    settings = {'stack': {'fusion_parameters': {'mode':'conservative','fallback':'last','confidence':.4,'sharpness_advantage':.25}}}
    config.write_text(json.dumps(settings))
    command = [sys.executable,'main.py','--input',str(stack),'--output-dir',str(tmp_path/'out'),
               '--config',str(config),'--cpu','--align','none','--no-exposure-match','--threads','1']
    result = subprocess.run(command,capture_output=True,text=True,timeout=60)
    assert result.returncode == 0, result.stdout + result.stderr
    output = cv2.imread(str(tmp_path/'out'/'stack.png'),cv2.IMREAD_UNCHANGED)
    assert output.dtype == np.uint16
    assert (output == 40003).all()
    settings['stack']['fusion_parameters']['confidence'] = 1.1
    config.write_text(json.dumps(settings))
    result = subprocess.run(command,capture_output=True,text=True,timeout=60)
    assert result.returncode == 2
    assert 'confidence' in result.stderr


@pytest.mark.parametrize('count', [2, 32, 200])
def test_streamed_top_two_diagnostics_match_reference(count):
    shape = (17, 19)
    probs = np.random.default_rng(83).random((count,17,38),dtype=np.float32)[:,:,::2]
    probs /= probs.sum(axis=0)
    # Include ties to verify stable winner/runner-up handling.
    probs[:,0,0] = 1/count
    frames = [np.full(shape,100,np.uint8)] * count
    maps = select(frames,probs)
    ordered = np.sort(probs,axis=0)
    np.testing.assert_array_equal(maps['top1'],ordered[-1])
    np.testing.assert_array_equal(maps['margin'],ordered[-1]-ordered[-2])
