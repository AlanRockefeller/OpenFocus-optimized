"""Small-stack contract for the render-scoped frame resolution cache."""
import os
import threading
from contextlib import contextmanager

import numpy as np
import pytest

from core import frame_resolution_cache as cache_module
from core import frame_storage
from core.source_frames import (SOURCE_CACHE, process_frame, scan_source,
                                transform_frame)


def _source(tmp_path, name='a.png', value=32):
    import cv2
    path = tmp_path / name
    cv2.imwrite(str(path), np.full((8, 10, 3), value, np.uint8))
    return scan_source(str(path))[0]


@contextmanager
def _bound(frames):
    from core.frame_storage import scratch_directory
    from core.source_frames import select_frame_scratch
    with scratch_directory():
        select_frame_scratch(frames)
        yield


def _processed(source, counter):
    def processor(image):
        counter.append(1)
        return np.array(image, copy=True)
    return process_frame(source, processor)


def test_source_frame_decodes_once_per_scope(tmp_path, monkeypatch):
    frame = _source(tmp_path)
    calls = []
    from core import image_loader
    real = image_loader.decode_image

    def counted(*args, **kwargs):
        calls.append(args)
        return real(*args, **kwargs)

    monkeypatch.setattr(image_loader, 'decode_image', counted)
    SOURCE_CACHE.clear()
    with cache_module.render_scope([frame]) as cache:
        with _bound([frame]):
            first = frame.full_resolution()
            SOURCE_CACHE.clear()      # the disk cache must be the only reuse
            second = frame.full_resolution()
    assert len(calls) == 1
    assert np.array_equal(np.asarray(first), np.asarray(second))
    assert second.flags.writeable is False
    assert cache_module.current() is None


def test_processed_frame_runs_processor_once(tmp_path):
    counter = []
    processed = _processed(_source(tmp_path), counter)
    SOURCE_CACHE.clear()
    with cache_module.render_scope([processed]):
        first = processed.full_resolution()
        second = processed.full_resolution()
    assert len(counter) == 1
    assert np.array_equal(np.asarray(first), np.asarray(second))


def test_cache_is_visible_to_producer_threads(tmp_path):
    counter = []
    processed = _processed(_source(tmp_path), counter)
    with cache_module.render_scope([processed]):
        results = []

        def work():
            results.append(processed.full_resolution())

        thread = threading.Thread(target=work)
        thread.start()
        thread.join()
        main = processed.full_resolution()
    assert len(counter) == 1
    assert np.array_equal(np.asarray(main), np.asarray(results[0]))


def test_cache_returns_read_only_views(tmp_path):
    counter = []
    processed = _processed(_source(tmp_path), counter)
    with cache_module.render_scope([processed]):
        image = processed.full_resolution()
        assert image.flags.writeable is False
        with pytest.raises(ValueError):
            image[0, 0] = 0


def test_source_frame_keeps_ram_fallback_when_disk_cache_is_disabled(tmp_path, monkeypatch):
    frame = _source(tmp_path)
    calls = []
    from core import image_loader
    real = image_loader.decode_image

    def counted(*args, **kwargs):
        calls.append(args)
        return real(*args, **kwargs)

    monkeypatch.setattr(image_loader, 'decode_image', counted)
    monkeypatch.setattr(cache_module, '_frame_need', lambda frames: 1 << 60)
    SOURCE_CACHE.clear()
    with cache_module.render_scope([frame]) as cache:
        with _bound([frame]):
            frame.full_resolution()
            frame.full_resolution()
    assert cache.enabled is False
    assert len(calls) == 1


def test_source_frame_keeps_ram_fallback_when_cache_write_fails(tmp_path, monkeypatch):
    frame = _source(tmp_path)
    calls = []
    from core import image_loader
    real = image_loader.decode_image

    def counted(*args, **kwargs):
        calls.append(args)
        return real(*args, **kwargs)

    def refuse(*_args, **_kwargs):
        raise OSError('scratch unavailable')

    monkeypatch.setattr(image_loader, 'decode_image', counted)
    monkeypatch.setattr(frame_storage, 'store_frame', refuse)
    SOURCE_CACHE.clear()
    with cache_module.render_scope([frame]) as cache:
        with _bound([frame]):
            frame.full_resolution()
            assert cache.degraded is True
            frame.full_resolution()
    assert len(calls) == 1


def test_programming_error_is_not_swallowed(tmp_path, monkeypatch):
    frame = _source(tmp_path)

    def broken(*_args, **_kwargs):
        raise ValueError('implementation error')

    monkeypatch.setattr(frame_storage, 'store_frame', broken)
    SOURCE_CACHE.clear()
    with cache_module.render_scope([frame]) as cache:
        with _bound([frame]):
            with pytest.raises(ValueError):
                frame.full_resolution()
            assert cache.degraded is False


def test_put_admission_failure_degrades_without_raising(tmp_path, monkeypatch):
    frame = _source(tmp_path)
    calls = []
    from core import image_loader
    real_decode = image_loader.decode_image

    def counted(*args, **kwargs):
        calls.append(args)
        return real_decode(*args, **kwargs)

    real_admit = frame_storage.require_disk_space
    state = {'calls': 0}

    def flaky(size, directory=None):
        state['calls'] += 1
        if state['calls'] >= 2:
            raise frame_storage.ScratchSpaceError(directory, size, 0)
        return real_admit(size, directory)

    monkeypatch.setattr(image_loader, 'decode_image', counted)
    monkeypatch.setattr(frame_storage, 'require_disk_space', flaky)
    SOURCE_CACHE.clear()
    with cache_module.render_scope([frame]) as cache:
        with _bound([frame]):
            frame.full_resolution()      # plan admits, put admission fails
            assert cache.degraded is True
            frame.full_resolution()      # RAM fallback, no second decode
    assert len(calls) == 1


def test_low_available_space_disables_cache_but_allows_pipeline_staging(tmp_path, monkeypatch):
    from types import SimpleNamespace
    from core.frame_storage import is_mapped_frame
    from core.source_frames import materialize_frames

    source = _source(tmp_path)
    limit = 128 * 1024
    usage = SimpleNamespace(total=10 ** 12, used=10 ** 12 - limit, free=limit)

    def fake_space(directory):
        return usage, 0, limit

    monkeypatch.setattr(frame_storage, '_space', fake_space)
    SOURCE_CACHE.clear()
    with cache_module.render_scope([source]) as cache:
        with _bound([source]):
            # The chosen volume only has room for the pipeline, not for the
            # cache's mandatory reserve: the cache must disable itself.
            pixels = source.full_resolution()
            assert pixels.shape == source.shape
            assert cache.enabled is False

            # Necessary pipeline staging underneath that volume still fits;
            # the cache has not consumed it.
            derived = _processed(source, [])
            staged = materialize_frames([derived], prefer_ram=False)
            assert len(staged) == 1
            assert is_mapped_frame(staged[0])
    assert cache.enabled is False
    assert cache.closed is True


def test_derived_frame_grows_budget_beyond_initial_need(tmp_path):
    from core.frame_storage import scratch_directory
    from core.source_frames import select_frame_scratch
    source = _source(tmp_path)
    counter = []
    with cache_module.render_scope([source]) as cache:
        with scratch_directory():
            select_frame_scratch([source])
            source.full_resolution()
            inner = cache._cache
            initial_budget = inner.budget
            derived = _processed(source, counter)   # created after planning
            derived.full_resolution()
            assert cache.enabled
            assert cache.degraded is False
            assert inner.budget > initial_budget
            assert inner._used > initial_budget
            derived.full_resolution()
    assert len(counter) == 1


def test_source_fingerprint_change_rejects_cached_copy(tmp_path):
    frame = _source(tmp_path)
    with cache_module.render_scope([frame]) as cache:
        with _bound([frame]):
            frame.full_resolution()
            assert cache.enabled
            # Explicit nanosecond stamp: os.utime(path, None) can round to the
            # same coarse timestamp on Windows and leave the fingerprint
            # unchanged.
            stamp = int(frame.fingerprint[1]) + 10_000_000_000
            os.utime(frame.path, ns=(stamp, stamp))
            with pytest.raises(OSError):
                cache.get(frame)


def test_cache_refuses_when_storage_volume_changes(tmp_path, monkeypatch):
    frame = _source(tmp_path)
    with cache_module.render_scope([frame]) as cache:
        with _bound([frame]):
            frame.full_resolution()
            assert cache.enabled
            other = tmp_path / "other"
            other.mkdir()
            monkeypatch.setattr(frame_storage, "storage_directory",
                                lambda: str(other))
            derived = _processed(frame, [])
            derived.full_resolution()
            assert cache.degraded is True


def test_scope_disables_when_budget_cannot_cover_working_set(tmp_path, monkeypatch):
    counter = []
    processed = _processed(_source(tmp_path), counter)
    monkeypatch.setattr(cache_module, '_frame_need', lambda frames: 1 << 60)
    with cache_module.render_scope([processed]) as cache:
        processed.full_resolution()
        assert cache.enabled is False
        processed.full_resolution()
    assert len(counter) == 2


def test_store_failure_degrades_without_raising(tmp_path, monkeypatch):
    counter = []
    processed = _processed(_source(tmp_path), counter)

    def refuse(*_args, **_kwargs):
        raise OSError('scratch unavailable')

    monkeypatch.setattr(frame_storage, 'store_frame', refuse)
    with cache_module.render_scope([processed]) as cache:
        processed.full_resolution()
        assert cache.degraded is True
        processed.full_resolution()
    assert len(counter) == 2


def test_scope_closes_and_clears_entries(tmp_path):
    counter = []
    processed = _processed(_source(tmp_path), counter)
    with cache_module.render_scope([processed]) as cache:
        processed.full_resolution()
        inner = cache._cache
        assert inner is not None and inner._entries
    assert cache.closed is True
    assert inner.closed is True
    assert not inner._entries
    assert cache_module.current() is None


def test_transform_frame_is_cached_by_identity(tmp_path):
    source = _source(tmp_path)
    transformed = transform_frame(source, 'flip', code=1)
    counter = []
    processed = _processed(transformed, counter)
    with cache_module.render_scope([processed]):
        first = processed.full_resolution()
        second = processed.full_resolution()
    assert len(counter) == 1
    assert np.array_equal(np.asarray(first), np.asarray(second))


def _count_hashes(monkeypatch):
    from core import openfocus_alignment as alignment
    hashed = []
    real = alignment._hash_pixels

    def counted(image, should_cancel):
        hashed.append(image)
        return real(image, should_cancel)

    monkeypatch.setattr(alignment, '_hash_pixels', counted)
    return hashed


def test_cached_pixels_are_hashed_once_per_scope(tmp_path, monkeypatch):
    import hashlib
    from core.openfocus_alignment import _pixel_digest
    frame = _source(tmp_path)
    hashed = _count_hashes(monkeypatch)
    with cache_module.render_scope([frame]):
        with _bound([frame]):
            first = frame.full_resolution()
            digest = _pixel_digest(first, None)
            SOURCE_CACHE.clear()
            second = frame.full_resolution()      # another view of the stored entry
            assert second is not first
            assert _pixel_digest(second, None) == digest
            assert _pixel_digest(first, None) == digest
    assert len(hashed) == 1
    assert digest == hashlib.sha256(np.full((8, 10, 3), 32, np.uint8).tobytes()).hexdigest()


def test_other_arrays_are_hashed_every_time(tmp_path, monkeypatch):
    import cv2
    from core.openfocus_alignment import _pixel_digest
    path = tmp_path / 'gradient.png'
    pixels = np.arange(8 * 10 * 3, dtype=np.uint8).reshape(8, 10, 3)
    cv2.imwrite(str(path), pixels)
    frame = scan_source(str(path))[0]
    hashed = _count_hashes(monkeypatch)
    with cache_module.render_scope([frame]):
        with _bound([frame]):
            cached = frame.full_resolution()
            digest = _pixel_digest(cached, None)
            assert len(hashed) == 1
            # Same shape and memory, but not the stored layout or extent.
            others = [np.array(cached), cached[::-1], cached[:, ::-1], cached[1:],
                      cached[:, :, :1], pixels]
            for count, other in enumerate(others, 1):
                assert _pixel_digest(other, None) == _pixel_digest(other, None)
                assert len(hashed) == 1 + 2 * count
            assert _pixel_digest(cached[::-1], None) != digest
            assert _pixel_digest(np.array(cached), None) == digest


def test_digest_is_not_remembered_outside_its_scope(tmp_path, monkeypatch):
    from core.openfocus_alignment import _pixel_digest
    frame = _source(tmp_path)
    hashed = _count_hashes(monkeypatch)
    image = np.full((8, 10, 3), 32, np.uint8)
    assert _pixel_digest(image, None) == _pixel_digest(image, None)
    assert len(hashed) == 2                        # no scope: nothing is remembered
    with cache_module.render_scope([frame]) as cache:
        # Before the job binds a scratch volume the scope stays dormant, and a
        # digest request must not plan it.
        assert _pixel_digest(image, None) == _pixel_digest(image, None)
        assert len(hashed) == 4 and cache.enabled is False
        with _bound([frame]):
            cached = frame.full_resolution()
            digest = _pixel_digest(cached, None)
            assert _pixel_digest(cached, None) == digest
            assert len(hashed) == 5
            inner = cache._cache
    assert not inner._digests
    assert _pixel_digest(cached, None) == digest   # closed scope: hashed again
    assert len(hashed) == 6


def test_cancelled_hash_is_not_remembered(tmp_path, monkeypatch):
    from core.openfocus_alignment import _pixel_digest
    from core.registration import RegistrationCancelled
    frame = _source(tmp_path)
    hashed = _count_hashes(monkeypatch)
    with cache_module.render_scope([frame]):
        with _bound([frame]):
            cached = frame.full_resolution()
            checks = iter((False, False, True))
            with pytest.raises(RegistrationCancelled):
                _pixel_digest(cached, lambda: next(checks))
            digest = _pixel_digest(cached, None)
            assert _pixel_digest(cached, None) == digest
    assert len(hashed) == 2                        # the cancelled attempt and one full hash


def test_remembered_digest_still_rejects_a_mismatched_fingerprint(tmp_path, monkeypatch):
    from core.openfocus_alignment import SourceFingerprint, _pixel_digest
    frame = _source(tmp_path)
    hashed = _count_hashes(monkeypatch)
    with cache_module.render_scope([frame]):
        with _bound([frame]):
            cached = frame.full_resolution()
            digest = _pixel_digest(cached, None)
            SourceFingerprint(cached.shape, cached.dtype.str, digest).validate_pixels(cached)
            with pytest.raises(ValueError, match='source pixels changed'):
                SourceFingerprint(cached.shape, cached.dtype.str, '0' * 64).validate_pixels(cached)
    assert len(hashed) == 1


def test_cached_tile_does_not_need_room_for_another_full_decode(tmp_path, monkeypatch):
    from core import resource_policy
    from core.source_frames import resolve_region
    frame = _source(tmp_path)
    with cache_module.render_scope([frame]), _bound([frame]):
        pixels = frame.full_resolution()
        def only_tile_fits(size, *args, **kwargs):
            if size > 12:
                raise resource_policy.ResourceLimitError('Only the tile fits')
        monkeypatch.setattr(resource_policy, 'require_memory', only_tile_fits)
        tile = resolve_region(frame, 1, 2, 2, 2)
        np.testing.assert_array_equal(tile, pixels[2:4, 1:3])
        assert not np.shares_memory(tile, pixels)


def test_cached_materialized_snapshot_retains_original_identity(tmp_path):
    from core.source_frames import materialize_for_repeated_access, crop_frame
    source = _source(tmp_path)
    snapshot = materialize_for_repeated_access([source], required=True)[0]
    crop = crop_frame(snapshot, 0, 0, 4, 4)
    # Staging a crop of prepared pixels must carry the original fingerprint
    # through a second snapshot, including after a cache hit.
    second = materialize_for_repeated_access([crop], required=True)[0]
    with cache_module.render_scope([snapshot, crop]):
        first = snapshot.full_resolution()
        assert np.shares_memory(first, snapshot.source)
        assert not first.flags.writeable
        np.testing.assert_array_equal(first, snapshot.full_resolution())
        np.testing.assert_array_equal(second.full_resolution(), first[:4, :4])
        crop.full_resolution()
        stamp = source.fingerprint[1] + 10_000_000_000
        os.utime(source.path, ns=(stamp, stamp))
        for frame in (snapshot, crop, second):
            with pytest.raises(OSError):
                frame.full_resolution()


def test_cache_discards_unpublished_store_on_cancellation(monkeypatch):
    from core.registration import RegistrationCancelled
    cancelled = []
    def store(image, **kwargs):
        cancelled.append(True)
        return image.copy()
    monkeypatch.setattr(frame_storage, 'store_frame', store)
    cache = cache_module.ResolutionCache(1000)
    with pytest.raises(RegistrationCancelled):
        cache.put(object(), np.zeros((2, 2), np.uint8), should_cancel=lambda: bool(cancelled))
    assert cache._used == 0 and not cache._entries


def _stores(monkeypatch, tmp_path, before=None):
    """Bind scratch to tmp_path and record the thread of every frame store."""
    monkeypatch.setattr(frame_storage, '_storage_directory', str(tmp_path))
    threads = []
    real = frame_storage.store_frame

    def store(image, **kwargs):
        threads.append(threading.current_thread().name)
        if before is not None:
            before()
        return real(image, **kwargs)
    monkeypatch.setattr(frame_storage, 'store_frame', store)
    return threads


def _pixels(seed=0):
    return np.arange(seed, seed + 48, dtype=np.uint16).reshape(4, 4, 3)


def _released(cache):
    """Wait until the frame last returned from memory has been released."""
    import gc
    import time
    deadline = time.monotonic() + 5
    while cache._lent() is not None and time.monotonic() < deadline:
        gc.collect()
        time.sleep(0.01)
    return cache._lent() is None


def test_write_behind_returns_the_callers_pixels_and_stores_a_copy(tmp_path, monkeypatch):
    threads = _stores(monkeypatch, tmp_path)
    cache = cache_module.ResolutionCache(1 << 20)
    key, image = object(), _pixels()
    lent = cache.put(key, image, behind=True)
    assert np.shares_memory(lent, image) and lent.flags.writeable is False
    stored = cache.get(key)                  # waits for the writer
    assert frame_storage.is_mapped_frame(stored) and not np.shares_memory(stored, image)
    assert stored.flags.writeable is False
    np.testing.assert_array_equal(stored, image)
    assert threads == ['openfocus-frame-cache_0']
    assert cache._used == image.nbytes
    # Putting the same frame again returns the stored copy, as before.
    assert np.shares_memory(cache.put(key, image, behind=True), stored)
    assert len(threads) == 1
    cache.close()


def test_write_behind_lends_one_frame_at_a_time(tmp_path, monkeypatch):
    threads = _stores(monkeypatch, tmp_path)
    cache = cache_module.ResolutionCache(1 << 20)
    first_image, second_image, third_image = _pixels(), _pixels(100), _pixels(200)
    first = cache.put('first', first_image, behind=True)
    # The first frame is still in its reader's hands: store this one now.
    second = cache.put('second', second_image, behind=True)
    assert np.shares_memory(first, first_image)
    assert frame_storage.is_mapped_frame(second)
    assert threading.current_thread().name in threads
    del first, first_image
    assert _released(cache)
    third = cache.put('third', third_image, behind=True)
    assert np.shares_memory(third, third_image)
    for key, image in (('first', _pixels()), ('second', second_image), ('third', third_image)):
        np.testing.assert_array_equal(cache.get(key), image)
    assert cache._used == 3 * third_image.nbytes
    cache.close()


def test_write_behind_does_not_remember_a_digest_until_the_copy_is_stored(tmp_path, monkeypatch):
    _stores(monkeypatch, tmp_path)
    cache = cache_module.ResolutionCache(1 << 20)
    hashes = []

    def compute():
        hashes.append(1)
        return 'digest'
    lent = cache.put('frame', _pixels(), behind=True)
    assert cache.pixel_digest(lent, compute) == cache.pixel_digest(lent, compute) == 'digest'
    assert len(hashes) == 2                  # memory outside the cache: hashed each time
    stored = cache.get('frame')
    assert cache.pixel_digest(stored, compute) == cache.pixel_digest(stored, compute) == 'digest'
    assert len(hashes) == 3
    cache.close()


def test_write_behind_store_failure_degrades_without_raising(tmp_path, monkeypatch):
    counter = []
    source = _source(tmp_path)

    def processor(image):
        counter.append(threading.current_thread().name)
        return np.array(image, copy=True)
    processed = process_frame(source, processor, write_behind=True)
    real = frame_storage.store_frame

    def refuse(image, **kwargs):
        if threading.current_thread().name.startswith('openfocus-frame-cache'):
            raise OSError('scratch unavailable')
        return real(image, **kwargs)

    monkeypatch.setattr(frame_storage, 'store_frame', refuse)
    with cache_module.render_scope([processed]) as cache:
        first = processed.full_resolution()          # served from memory
        assert first.flags.writeable is False and not frame_storage.is_mapped_frame(first)
        second = processed.full_resolution()         # the copy never arrived: recomputed
        assert cache.degraded is True
        assert cache.get(processed) is None
        np.testing.assert_array_equal(first, second)
        # The source stored before the failure is still served.
        assert cache.get(source) is not None
    assert len(counter) == 2


def test_write_behind_programming_error_is_not_swallowed(tmp_path, monkeypatch):
    _stores(monkeypatch, tmp_path, before=lambda: (_ for _ in ()).throw(TypeError('bug')))
    cache = cache_module.ResolutionCache(1 << 20)
    cache.put('frame', _pixels(), behind=True)
    with pytest.raises(TypeError, match='bug'):
        cache.get('frame')
    cache.close()


def test_closing_waits_for_a_write_in_flight_and_removes_its_file(tmp_path, monkeypatch):
    import gc
    started, release = threading.Event(), threading.Event()

    def hold():
        started.set()
        assert release.wait(10)
    _stores(monkeypatch, tmp_path, before=hold)
    cache = cache_module.ResolutionCache(1 << 20)
    lent = cache.put('frame', _pixels(), behind=True)
    assert started.wait(10)
    closer = threading.Thread(target=cache.close)
    closer.start()
    closer.join(0.2)
    assert closer.is_alive()                 # close() waits for the writer
    assert cache.get('frame') is None and cache.put('other', _pixels()) is None
    release.set()
    closer.join(10)
    assert not closer.is_alive()
    gc.collect()
    assert not list(tmp_path.glob('openfocus-frame-*'))
    assert not [thread for thread in threading.enumerate()
                if thread.name.startswith('openfocus-frame-cache')]
    np.testing.assert_array_equal(lent, _pixels())      # the reader keeps its pixels


def test_write_behind_cancellation_leaves_no_file(tmp_path, monkeypatch):
    import gc
    from core.registration import RegistrationCancelled
    threads = _stores(monkeypatch, tmp_path)
    cache = cache_module.ResolutionCache(1 << 20)
    checks = []

    def cancel():
        checks.append(1)
        return len(checks) > 2               # after the frame is handed to the writer
    with pytest.raises(RegistrationCancelled):
        cache.put('frame', _pixels(), behind=True, should_cancel=cancel)
    cache.close()
    gc.collect()
    assert len(threads) <= 1
    assert not list(tmp_path.glob('openfocus-frame-*'))


def test_put_shares_pixels_the_cache_already_holds(tmp_path, monkeypatch):
    threads = _stores(monkeypatch, tmp_path)
    cache = cache_module.ResolutionCache(1 << 20)
    image = _pixels()
    stored = cache.put('source', image)
    # A processor returned its cached input unchanged, as a gain of one does.
    for behind in (False, True):
        shared = cache.put(('same', behind), stored, behind=behind)
        assert np.shares_memory(shared, stored) and shared.flags.writeable is False
        assert np.shares_memory(cache.get(('same', behind)), stored)
    assert len(threads) == 1 and cache._used == image.nbytes
    # Equal pixels elsewhere in memory, or part of a stored frame, are other frames.
    copy = cache.put('copy', np.array(stored))
    part = cache.put('part', stored[1:])
    assert not np.shares_memory(copy, stored) and not np.shares_memory(part, stored)
    assert len(threads) == 3
    assert cache._used == 2 * image.nbytes + stored[1:].nbytes
    cache.close()


def test_write_behind_keeps_an_owned_scratch_directory_until_the_copy_is_stored(
        tmp_path, monkeypatch):
    import gc
    started, release = threading.Event(), threading.Event()

    def hold():
        started.set()
        assert release.wait(10)
    _stores(monkeypatch, tmp_path, before=hold)
    cache = cache_module.ResolutionCache(1 << 20)
    owned = tmp_path / 'owned'
    owned.mkdir()
    # A resolving call with no job scope around it owns its scratch directory
    # and removes it on return, unless a mapping or a pending write holds it.
    with frame_storage.scratch_directory(separate=True) as state:
        state.directory, state.owned = str(owned), True
        lent = cache.put('frame', _pixels(), behind=True)
        assert started.wait(10)
    assert owned.exists()
    release.set()
    stored = cache.get('frame')
    assert cache.degraded is False
    assert os.path.samefile(os.path.dirname(frame_storage._mapped_base(stored).filename), owned)
    np.testing.assert_array_equal(stored, lent)
    cache.close()
    del stored
    gc.collect()
    assert not owned.exists()                # gone with its last mapping


def test_write_behind_from_a_producer_thread_without_a_scratch_scope(tmp_path):
    counter = []

    def processor(image):
        counter.append(1)
        return np.array(image, copy=True)
    processed = process_frame(_source(tmp_path), processor, write_behind=True)
    with cache_module.render_scope([processed]) as cache:
        results = []

        def work():
            results.append(processed.full_resolution())

        thread = threading.Thread(target=work)
        thread.start()
        thread.join()
        main = processed.full_resolution()
        assert cache.degraded is False
        assert frame_storage.is_mapped_frame(main)
    assert len(counter) == 1
    assert np.array_equal(np.asarray(main), np.asarray(results[0]))
