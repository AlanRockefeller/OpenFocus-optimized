"""Regression tests for the implemented OpenFocus-derived features:
OpenFocus A/B fusion methods, base correction (brightness drift), and the
OpenFocus fast alignment (bounded phase correlation)."""
import numpy as np
import cv2
import pytest

from tests.conftest import make_two_focus_stack, laplacian_sharpness

pytest.importorskip("cv2")


class TestOpenFocusAB:
    @pytest.mark.parametrize("method", ["openfocus_a", "openfocus_b",
                                        "openfocus_c"])
    def test_both_halves_sharpen(self, method):
        from core.multi_focus_fusion import MultiFocusFusion
        frames = make_two_focus_stack()
        fusion = MultiFocusFusion(algorithm=method, use_gpu=False)
        out = fusion.fuse(input_source=frames, img_resize=None, kernel_size=7)
        assert out.shape[:2] == frames[0].shape[:2]
        assert float(out.mean()) > 1.0
        # fused must beat both inputs overall (sharp half from each)
        fused = laplacian_sharpness(out)
        assert fused >= max(laplacian_sharpness(f) for f in frames) * 0.9

    def test_b_is_a_hard_pick(self):
        """Method B picks whole pixels from one frame: output equals a blend
        of exact source pixels (left half from f0, right from f1)."""
        from fusion_methods.openfocus_ab import openfocus_b_impl
        frames = make_two_focus_stack()
        out = openfocus_b_impl(frames, None, 7)
        left = cv2.cvtColor(out[:, :80], cv2.COLOR_BGR2GRAY).astype(int)
        right = cv2.cvtColor(out[:, 80:], cv2.COLOR_BGR2GRAY).astype(int)
        l0 = cv2.cvtColor(frames[0][:, :80], cv2.COLOR_BGR2GRAY).astype(int)
        r1 = cv2.cvtColor(frames[1][:, 80:], cv2.COLOR_BGR2GRAY).astype(int)
        assert np.abs(left - l0).mean() < 1.0
        assert np.abs(right - r1).mean() < 1.0

    def test_b_tie_picks_first_frame(self):
        """Identical frames: np.argmax tie-break is the first frame."""
        from fusion_methods.openfocus_ab import openfocus_b_impl
        frame = make_two_focus_stack()[0]
        out = openfocus_b_impl([frame, frame.copy()], None, 7)
        assert np.array_equal(out, frame)

    def test_b_deep_stack_over_256_frames(self):
        """Regression (v1.40): depth indices must survive >256-frame stacks.

        v1.39 cast the depth map to uint8 before the median filter, so a
        260-frame stack clipped index 259 down to 255 and silently picked
        the wrong (blurred) frame. Float32 median keeps every index.
        """
        from fusion_methods.openfocus_ab import openfocus_b_impl
        rng = np.random.default_rng(7)
        base = rng.uniform(40, 60, (48, 64)).astype(np.float32)
        base = cv2.GaussianBlur(base, (0, 0), 3.0)  # blurred everywhere
        frames = [cv2.cvtColor(np.clip(base + i * 0.01, 0, 255)
                               .astype(np.uint8), cv2.COLOR_GRAY2BGR)
                  for i in range(259)]
        # last frame is the only sharp one -> argmax must reach index 259
        sharp = np.clip(base + rng.normal(0, 12, base.shape), 0, 255)
        sharp = cv2.GaussianBlur(sharp, (0, 0), 0.5).astype(np.uint8)
        frames.append(cv2.cvtColor(sharp, cv2.COLOR_GRAY2BGR))
        out = openfocus_b_impl(frames, None, 7)
        # interior (away from the boundary transition) must equal the sharp
        # frame's pixels, not the clipped-to-255 blurred one
        interior = out[8:-8, 8:-8].astype(int)
        sharp_i = frames[-1][8:-8, 8:-8].astype(int)
        clipped_i = frames[255][8:-8, 8:-8].astype(int)
        assert np.abs(interior - sharp_i).mean() < np.abs(interior - clipped_i).mean()
        assert np.abs(interior - sharp_i).mean() < 1.0

    def test_a_flat_region_degrades_to_last_frame(self):
        """The +1 last-frame prior: zero weights (flat) -> exact last frame."""
        from fusion_methods.openfocus_ab import openfocus_a_impl
        flat0 = np.full((60, 80, 3), 100, np.uint8)
        flat1 = np.full((60, 80, 3), 150, np.uint8)
        out = openfocus_a_impl([flat0, flat1], None, 7)
        assert np.array_equal(out, flat1)

    def test_two_frames_directory_input(self, tmp_path):
        """The methods also accept a directory (lazy path)."""
        from fusion_methods.openfocus_ab import openfocus_b_impl
        for i, frame in enumerate(make_two_focus_stack((60, 80))):
            cv2.imwrite(str(tmp_path / f"f{i}.png"), frame)
        out = openfocus_b_impl(str(tmp_path), None, 7)
        assert out.shape[:2] == (60, 80)

    @pytest.mark.parametrize("method", ["openfocus_a", "openfocus_b",
                                        "openfocus_c"])
    def test_tiled_matches_shape(self, method):
        from core.multi_focus_fusion import MultiFocusFusion
        frames = make_two_focus_stack((120, 160))
        fusion = MultiFocusFusion(algorithm=method, use_gpu=False,
                                  tile_enabled=True, tile_threshold=64)
        out = fusion.fuse(input_source=frames, img_resize=None, kernel_size=7)
        assert out.shape[:2] == (120, 160)

    def test_supported_and_cli_names(self):
        from core.multi_focus_fusion import MultiFocusFusion
        assert "openfocus_a" in MultiFocusFusion.SUPPORTED_ALGORITHMS
        assert "openfocus_b" in MultiFocusFusion.SUPPORTED_ALGORITHMS
        assert "openfocus_c" in MultiFocusFusion.SUPPORTED_ALGORITHMS
        from core.cli import VALID_METHODS, VALID_ALIGN
        assert "openfocus_a" in VALID_METHODS
        assert "openfocus_c" in VALID_METHODS
        assert "openfocus" in VALID_ALIGN


class TestOpenFocusC:
    """Method C specifics: pyramid structure and base-band behaviour."""

    def test_identity_reconstruction_is_exact(self):
        """Decompose + rebuild without mixing must return the frame bit-exact
        (the pyramid is mathematically invertible)."""
        from fusion_methods.openfocus_ab import (_pyramid_gauss,
                                                 _pyramid_expand)
        frame = make_two_focus_stack((120, 160))[0].astype(np.float32)
        levels = 5
        g = _pyramid_gauss(frame, levels)
        bands = [g[k] - _pyramid_expand(g[k + 1], g[k].shape[:2])
                 for k in range(levels - 1)]
        bands.append(g[-1])
        img = bands[-1]
        for k in range(levels - 2, -1, -1):
            img = _pyramid_expand(img, bands[k].shape[:2]) + bands[k]
        assert np.abs(img - frame).max() < 1e-3

    def test_base_band_takes_last_frame_lowfreq(self):
        """The coarsest band comes from the last frame: a brightness-drifted
        stack's output low-frequency must match the LAST frame."""
        from fusion_methods.openfocus_ab import openfocus_c_impl
        base = make_two_focus_stack((120, 160))
        drifted = [base[0].astype(np.float32) * 0.6,
                   base[1].astype(np.float32)]
        out = openfocus_c_impl(drifted, None, 7)
        lp = cv2.GaussianBlur(out.astype(np.float32), (0, 0), 32).mean()
        target = cv2.GaussianBlur(drifted[1], (0, 0), 32).mean()
        assert abs(lp - target) < 3.0

    def test_two_identical_frames_roundtrip(self):
        from fusion_methods.openfocus_ab import openfocus_c_impl
        frame = make_two_focus_stack((90, 120))[0]
        out = openfocus_c_impl([frame, frame.copy()], None, 7)
        assert np.abs(out.astype(int) - frame.astype(int)).mean() < 1.0

    def test_small_images_do_not_crash(self):
        from fusion_methods.openfocus_ab import openfocus_c_impl
        frames = make_two_focus_stack((48, 64))
        out = openfocus_c_impl(list(frames), None, 7)
        assert out.shape[:2] == (48, 64)
        assert float(out.mean()) > 1.0

    def test_pyramid_boosts_local_contrast(self):
        """implemented OpenFocus behaviour: C raises high-frequency
        energy above both inputs (the documented 'contrast and glare' side
        effect of per-band decisions)."""
        from fusion_methods.openfocus_ab import openfocus_c_impl
        frames = make_two_focus_stack((120, 160))
        out = openfocus_c_impl(list(frames), None, 7)
        assert laplacian_sharpness(out) > max(laplacian_sharpness(f)
                                              for f in frames)


class TestBaseCorrection:
    def _drifted_stack(self):
        base = make_two_focus_stack((120, 160))
        return [base[0].astype(np.float32) * 0.6,
                base[1].astype(np.float32)]

    def test_default_off(self):
        from core.multi_focus_fusion import MultiFocusFusion
        f = MultiFocusFusion(algorithm="openfocus_b", use_gpu=False)
        assert f.base_frame == "none"

    def test_last_matches_last_frame_lowfreq(self):
        from core.multi_focus_fusion import MultiFocusFusion
        stack = self._drifted_stack()
        lp = lambda x: cv2.GaussianBlur(
            np.asarray(x, np.float32), (0, 0), 32).mean()
        out = MultiFocusFusion(algorithm="openfocus_b", use_gpu=False,
                               base_frame="last").fuse(
            input_source=list(stack), img_resize=None, kernel_size=7)
        assert abs(lp(out) - lp(stack[1])) < 3.0

    def test_first_matches_first_frame_lowfreq(self):
        from core.multi_focus_fusion import MultiFocusFusion
        stack = self._drifted_stack()
        lp = lambda x: cv2.GaussianBlur(
            np.asarray(x, np.float32), (0, 0), 32).mean()
        out = MultiFocusFusion(algorithm="openfocus_b", use_gpu=False,
                               base_frame="first").fuse(
            input_source=list(stack), img_resize=None, kernel_size=7)
        assert abs(lp(out) - lp(stack[0])) < 4.0

    def test_none_leaves_blend_untouched(self):
        from core.multi_focus_fusion import MultiFocusFusion
        stack = self._drifted_stack()
        lp = lambda x: cv2.GaussianBlur(
            np.asarray(x, np.float32), (0, 0), 32).mean()
        out = MultiFocusFusion(algorithm="openfocus_b", use_gpu=False,
                               base_frame="none").fuse(
            input_source=list(stack), img_resize=None, kernel_size=7)
        # no correction: the blend sits between the two frames
        assert lp(stack[0]) + 10 < lp(out) < lp(stack[1]) - 10

    def test_works_for_every_algorithm(self):
        """The base correction is a post-step at the unified exit: classical
        and tiled paths must both honour it."""
        from core.multi_focus_fusion import MultiFocusFusion
        stack = self._drifted_stack()
        lp = lambda x: cv2.GaussianBlur(
            np.asarray(x, np.float32), (0, 0), 32).mean()
        for algo in ("guided_filter", "gfgfgf"):
            out = MultiFocusFusion(algorithm=algo, use_gpu=False,
                                   base_frame="last").fuse(
                input_source=list(stack), img_resize=None, kernel_size=7)
            assert abs(lp(out) - lp(stack[1])) < 4.0, algo


class TestOpenFocusAlignment:
    def _shifted_stack(self):
        base = make_two_focus_stack((120, 160))[0]
        # frame 1 = base shifted by (+6, -4): np.roll semantics
        f1 = np.roll(np.roll(base, -4, axis=0), 6, axis=1)
        return [base, f1]

    def test_recovers_known_translation(self):
        from core.registration import ImageRegistration
        reg = ImageRegistration(method="openfocus")
        out = reg.process(self._shifted_stack(), output_path=None,
                          thread_count=2)
        assert out[0].shape == out[1].shape
        # interior MAE must drop well below the unaligned residual
        m = np.abs(out[1].astype(int)[4:-4, 8:-8]
                   - out[0].astype(int)[4:-4, 8:-8]).mean()
        un = np.abs(self._shifted_stack()[1].astype(int)
                    - self._shifted_stack()[0].astype(int)).mean()
        assert m < un * 0.35

    def test_equalizes_brightness(self):
        from core.registration import ImageRegistration
        base, f1 = self._shifted_stack()
        dark = (f1.astype(np.float32) * 0.6).astype(np.uint8)
        reg = ImageRegistration(method="openfocus")
        out = reg.process([base, dark], output_path=None, thread_count=2)
        assert abs(float(out[1].mean()) - float(base.mean())) < 12.0

    def test_registered_in_supported_methods(self):
        from core.registration import ImageRegistration
        assert "openfocus" in ImageRegistration.SUPPORTED_METHODS

    def test_no_shift_on_identical_frames(self):
        from core.registration import ImageRegistration
        frame = make_two_focus_stack((80, 100))[0]
        reg = ImageRegistration(method="openfocus")
        out = reg.process([frame, frame.copy()], output_path=None,
                          thread_count=2)
        # identical frames: no crop beyond the tiny margin
        assert out[0].shape == out[1].shape
        assert np.array_equal(out[0], out[1])
