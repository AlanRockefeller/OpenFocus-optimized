"""Two StackMFF V3 tiles in flight on the CPU: same pixels, admitted memory, clean stops."""
from dataclasses import replace
import threading
import time

import numpy as np
import pytest

from conftest import make_two_focus_stack
import core.multi_focus_fusion as mff
from core import resource_policy as policy
from core.multi_focus_fusion import MultiFocusFusion, _ai_effective_tiling, is_ai_fusion_available
from core.registration import RegistrationCancelled

WEIGHTS = "weights/stackmff_v3_final.onnx"
GIB = 1024 ** 3
needs_runtime = pytest.mark.skipif(not is_ai_fusion_available(),
                                   reason="ONNX Runtime not installed")


def _force(monkeypatch, tiles):
    """Plan a 2-thread budget with ``tiles`` in flight, whatever this machine offers."""
    planner = policy.plan_neural_tiles
    requested = []

    def planned(*args, **kwargs):
        requested.append(kwargs.get('in_flight', 1))
        block, plan = planner(*args, **kwargs)
        return block, replace(plan, workers=tiles, max_in_flight=tiles,
                              native_threads=2 // tiles)

    monkeypatch.setattr(policy, 'plan_neural_tiles', planned)
    return requested


def _no_tile_threads():
    return not [thread for thread in threading.enumerate()
                if thread.name.startswith('neural-tile')]


# ---- pixels ---------------------------------------------------------------

def _stack(dtype):
    first, second = make_two_focus_stack()
    frames = [first, second, ((first.astype(np.uint16) + second) // 2).astype(np.uint8)]
    if dtype is np.uint16:
        # Low bits that an 8-bit round trip would lose.
        frames = [frame.astype(np.uint16) * 257 + index for index, frame in enumerate(frames)]
    return frames


def _render(monkeypatch, tiles, mode, focus, dtype, exposure_match):
    """One tiled render; returns pixels, focus maps and how its runners ran."""
    from fusion_methods import stackmffv3 as ai

    _force(monkeypatch, tiles)
    lock = threading.Lock()
    runs = {'threads': [], 'intra_op': set(), 'active': 0, 'peak': 0}
    original = ai._stackmffv3_batch_impl

    def spy(tiles_list, *args, **kwargs):
        with lock:
            runs['threads'].append(threading.get_ident())
            runs['intra_op'].add(kwargs['tile_plan'].native_threads)
            runs['active'] += 1
            runs['peak'] = max(runs['peak'], runs['active'])
        try:
            return original(tiles_list, *args, **kwargs)
        finally:
            with lock:
                runs['active'] -= 1

    monkeypatch.setattr(ai, '_stackmffv3_batch_impl', spy)
    fusion = MultiFocusFusion(
        algorithm='stackmffv3', use_gpu=False, fusion_parameters={'mode': mode},
        exposure_match=exposure_match, tile_enabled=True, tile_block_size=64,
        tile_overlap=16, tile_threshold=32)
    maps, progress = [], []
    options = {}
    if focus:
        options['focus_callback'] = lambda positions, scale: maps.append(
            (np.array(positions), scale))
    image = fusion.fuse(input_source=_stack(dtype), img_resize=None, model_path=WEIGHTS,
                        thread_count=2,
                        progress_callback=lambda done, total: progress.append((done, total)),
                        **options)
    return image, maps, progress, runs


@needs_runtime
@pytest.mark.parametrize('mode, focus, dtype, exposure_match', [
    ('soft', False, np.uint8, False), ('soft', True, np.uint8, False),
    ('hard', False, np.uint8, False), ('hard', True, np.uint8, False),
    ('soft', True, np.uint16, True), ('hard', False, np.uint16, True)])
def test_two_tiles_in_flight_give_the_pixels_of_one(monkeypatch, mode, focus, dtype,
                                                    exposure_match):
    """The shipped arrangement (one tile, two threads) against two tiles, one thread each."""
    tiles = len(_ai_effective_tiling(3, 64, 120, 160, 16, 1)[2])
    assert tiles == 9, 'the grid should end on a tile with nothing beside it'
    case = (mode, focus, dtype, exposure_match)
    with monkeypatch.context() as patch:
        one, one_maps, one_progress, one_runs = _render(patch, 1, *case)
    with monkeypatch.context() as patch:
        two, two_maps, two_progress, two_runs = _render(patch, 2, *case)
    main = threading.get_ident()
    assert one_runs['threads'] == [main] * tiles
    assert (one_runs['intra_op'], one_runs['peak']) == ({2}, 1)
    assert len(two_runs['threads']) == tiles and main not in two_runs['threads']
    assert len(set(two_runs['threads'])) <= 2
    assert two_runs['intra_op'] == {1} and two_runs['peak'] <= 2

    assert two.dtype == one.dtype
    np.testing.assert_array_equal(two, one)
    assert len(two_maps) == len(one_maps) == int(focus)
    for (positions, scale), (expected, expected_scale) in zip(two_maps, one_maps):
        assert positions.dtype == expected.dtype and scale == expected_scale
        np.testing.assert_array_equal(positions, expected)
    assert two_progress == one_progress == [(done, tiles) for done in range(tiles + 1)]
    assert _no_tile_threads()


# ---- scheduling, with a stand-in runner -----------------------------------

GRID = _ai_effective_tiling(2, 12, 16, 24, 4, 1)[2]
ORIGINS = {(x0, y0): index for index, (x0, y0, _x1, _y1) in enumerate(GRID)}
# The first frame numbers its pixels, so a runner can tell which tile it was given.
FRAMES = [np.repeat(np.arange(16 * 24, dtype=np.uint16).reshape(16, 24, 1), 3, axis=2),
          np.full((16, 24, 3), 16, np.uint16)]


def _tile_index(crops):
    y0, x0 = divmod(int(crops[0][0, 0, 0]), 24)
    return ORIGINS[x0, y0]


def _tiled(runner, monkeypatch, *, frames=FRAMES, algorithm='stackmffv3', **options):
    """Run the tiled scheduler over the six-tile grid with ``runner`` as the fusion."""
    fusion = MultiFocusFusion(algorithm, use_gpu=False, exposure_match=False)
    height, width = frames[0].shape[:2]
    with monkeypatch.context() as patch:
        # Constructing a fusion resolves its own entry points, so patch after it.
        patch.setattr(mff, 'resolve_runner', lambda _name: runner)
        return fusion._fuse_tiled_neural(
            frames, None, height, width, 3, 12, 4, label='test', default_weights=WEIGHTS,
            mode_normalizer=lambda value: value, mode='soft', **options)


def test_two_runners_overlap_and_a_third_waits_for_the_first_result(monkeypatch):
    assert len(GRID) == 6
    _force(monkeypatch, 2)
    together = threading.Barrier(2, timeout=30)
    lock = threading.Lock()
    started, progress, active = {}, [], [0, 0]

    def runner(_fusion, crops, _resize, **_kwargs):
        index = _tile_index(crops)
        with lock:
            started[index] = progress[-1][0]
            active[0] += 1
            active[1] = max(active)
        if index < 2:
            # Passes only while tiles 0 and 1 are both inside their runners.
            together.wait()
        time.sleep(0.01)
        with lock:
            active[0] -= 1
        return crops[1]

    out = _tiled(runner, monkeypatch,
                 progress_callback=lambda done, total: progress.append((done, total)))
    assert out.shape == (16, 24, 3) and np.all(out == 16)
    assert active[1] == 2
    assert sorted(started) == list(range(6))
    # Tile k starts only after tile k - 2 was accumulated and reported.
    assert all(done >= index - 1 for index, done in started.items())
    assert progress == [(done, 6) for done in range(7)]
    assert _no_tile_threads()


def test_only_v3_asks_for_a_second_tile(monkeypatch):
    def runner(_fusion, crops, _resize, **_kwargs):
        return crops[1]

    asked = _force(monkeypatch, 1)
    _tiled(runner, monkeypatch)
    _tiled(runner, monkeypatch, algorithm='stackmff')
    assert asked == [2, 1]


def test_a_lone_tile_runs_on_the_caller_with_the_whole_budget(monkeypatch):
    _force(monkeypatch, 2)
    seen = []

    def runner(_fusion, crops, _resize, *, tile_plan, **_kwargs):
        seen.append((threading.get_ident(), tile_plan.max_in_flight, tile_plan.native_threads))
        return crops[1]

    _tiled(runner, monkeypatch, frames=[frame[:12, :12] for frame in FRAMES])
    assert seen == [(threading.get_ident(), 1, 2)]


def test_cancelling_stops_both_tiles_and_leaves_no_thread(monkeypatch):
    _force(monkeypatch, 2)
    cancelled = threading.Event()
    running = threading.Semaphore(0)
    stopped = []

    def runner(_fusion, _crops, _resize, *, should_cancel, **_kwargs):
        running.release()
        while not should_cancel():
            time.sleep(0.002)
        stopped.append(threading.get_ident())
        raise RegistrationCancelled()

    def cancel_once_both_run():
        both = running.acquire(timeout=30) and running.acquire(timeout=30)
        cancelled.set()
        assert both, 'two runners never ran at once'

    trigger = threading.Thread(target=cancel_once_both_run, daemon=True)
    trigger.start()
    progress = []
    with pytest.raises(RegistrationCancelled):
        _tiled(runner, monkeypatch, should_cancel=cancelled.is_set,
               progress_callback=lambda done, total: progress.append(done))
    trigger.join(timeout=30)
    assert len(set(stopped)) == 2 and threading.get_ident() not in stopped
    assert progress == [0]
    assert _no_tile_threads()


def test_a_failing_tile_surfaces_after_the_tiles_before_it(monkeypatch):
    _force(monkeypatch, 2)
    progress = []

    def runner(_fusion, crops, _resize, **_kwargs):
        if _tile_index(crops) == 1:
            raise ValueError('tile 1 failed')
        time.sleep(0.05)
        return crops[1]

    with pytest.raises(ValueError, match='tile 1 failed'):
        _tiled(runner, monkeypatch,
               progress_callback=lambda done, total: progress.append(done))
    # Tile 0 was accumulated and reported first, as it is one tile at a time.
    assert progress == [0, 1]
    assert _no_tile_threads()


# ---- shared inference state -----------------------------------------------

@needs_runtime
def test_two_callers_build_each_session_once_and_report_one_device():
    from fusion_methods import neural_runtime as runtime
    from utils import performance_profile as pp
    from utils.weights import weights_path

    model = weights_path('stackmff_v3_final.onnx')
    planes = np.random.default_rng(5).random((1, 3, 64, 64), dtype=np.float32)
    together = threading.Barrier(2, timeout=30)
    results = [None, None]

    def call(slot):
        together.wait()
        results[slot] = runtime.logits(planes, model, 'CPU', 1)

    profile = pp.PerformanceProfile()
    with runtime.inference_job(), pp.active(profile):
        expected = runtime.logits(planes, model, 'CPU', 2)
        runtime._clear_sessions()
        callers = [threading.Thread(target=call, args=(slot,)) for slot in range(2)]
        for caller in callers:
            caller.start()
        for caller in callers:
            caller.join()
        assert runtime.device_diagnostics()['actual_devices'] == ('CPU',)
        assert runtime.actual_device_report() == 'CPU'
    for result in results:
        np.testing.assert_array_equal(result, expected)
    one_thread = [key for key in profile.report('completed')['sessions']['keys']
                  if key['threads'] == 1]
    assert [(key['created'], key['hits'], key['runs']) for key in one_thread] == [(1, 1, 2)]


# ---- admission ------------------------------------------------------------

def _plan(available=48 * GIB, **options):
    from fusion_methods.neural_runtime import estimate_inference_bytes
    snapshot = policy.ResourceSnapshot(available_ram=available, total_ram=64 * GIB,
                                       physical_cores=8, logical_cpus=16)
    return policy.plan_neural_tiles(15, 5216, 3908, 3, 1024, 'CPU', estimate_inference_bytes,
                                    snapshot=snapshot, **options)


def _second_tile_bytes():
    from fusion_methods.neural_runtime import estimate_inference_bytes
    return sum(policy._neural_tile_terms(15, 3, 1024, 'CPU', estimate_inference_bytes))


def test_a_second_tile_is_admitted_with_its_own_working_set():
    block, one = _plan(user_threads=2)
    assert (block, one.workers, one.max_in_flight, one.native_threads) == (1024, 1, 1, 2)
    assert one.reasons == ('neural tile 1024px, CPU',)

    same_block, two = _plan(user_threads=2, in_flight=2)
    assert same_block == block
    assert (two.workers, two.max_in_flight, two.native_threads) == (2, 2, 1)
    assert two.required_ram == one.required_ram + _second_tile_bytes()
    assert two.required_ram <= two.ram_budget == one.ram_budget


@pytest.mark.parametrize('threads, expected', [(1, (1, 1)), (2, (2, 1)), (3, (1, 3)),
                                               (4, (2, 2)), (6, (2, 3))])
def test_the_thread_budget_is_split_only_when_it_divides_evenly(threads, expected):
    _block, plan = _plan(user_threads=threads, in_flight=2)
    assert (plan.max_in_flight, plan.native_threads) == expected
    assert plan.max_in_flight * plan.native_threads == threads


def test_a_second_tile_that_does_not_fit_keeps_one_tile_at_the_same_size():
    _block, one = _plan(user_threads=2)
    reserve = 48 * GIB - one.ram_budget
    # Room for one tile and half of another.
    available = reserve + one.required_ram + _second_tile_bytes() // 2
    block, tight = _plan(available, user_threads=2, in_flight=2)
    assert block == 1024
    assert (tight.workers, tight.max_in_flight, tight.native_threads) == (1, 1, 2)
    assert tight.required_ram == one.required_ram <= tight.ram_budget
    assert tight.ram_budget < one.required_ram + _second_tile_bytes()
    assert 'one tile in flight' in tight.reasons[-1]


def test_an_accelerator_plan_keeps_one_tile_in_flight():
    from fusion_methods.neural_runtime import estimate_inference_bytes
    snapshot = policy.ResourceSnapshot(
        available_ram=48 * GIB, total_ram=64 * GIB, physical_cores=8, logical_cpus=16,
        device_kind='CUDA', free_vram=24 * GIB, total_vram=24 * GIB)
    _block, plan = policy.plan_neural_tiles(
        15, 5216, 3908, 3, 512, 'CUDA', estimate_inference_bytes, snapshot=snapshot,
        user_threads=2, in_flight=2)
    assert plan.device_kind == 'CUDA'
    assert (plan.workers, plan.max_in_flight, plan.native_threads) == (1, 1, 2)
