"""Flip review viewer: view geometry, flipping, keys, ratings, pins (offscreen)."""
import json
import os
import time

import cv2
import numpy as np
import pytest

pytest.importorskip("PyQt6")

from PyQt6.QtCore import QPoint, QPointF, Qt  # noqa: E402
from PyQt6.QtTest import QTest  # noqa: E402

from core import flip_review as model  # noqa: E402
from widgets.flip_review import ViewState, FlipReviewViewer, MAX_ZOOM  # noqa: E402

W, H = 160, 120


def wait_for(qapp, condition, timeout=20.0):
    deadline = time.monotonic() + timeout
    while not condition():
        if time.monotonic() > deadline:
            raise AssertionError("timed out")
        qapp.processEvents()
        time.sleep(0.002)
    qapp.processEvents()


def _images(tmp_path, count=3, size=(W, H)):
    """16-bit LZW TIFF, 8-bit PNG and grey JPEG of identical dimensions."""
    rng = np.random.default_rng(7)
    paths = []
    for index in range(count):
        kind = index % 3
        if kind == 0:
            pixels = rng.integers(0, 65535, (size[1], size[0], 3), dtype=np.uint16)
            path = tmp_path / f"c{index}.tif"
            assert cv2.imwrite(str(path), pixels, [cv2.IMWRITE_TIFF_COMPRESSION, 5])
        elif kind == 1:
            pixels = rng.integers(0, 255, (size[1], size[0], 3), dtype=np.uint8)
            path = tmp_path / f"c{index}.png"
            assert cv2.imwrite(str(path), pixels)
        else:
            pixels = rng.integers(0, 255, (size[1], size[0]), dtype=np.uint8)
            path = tmp_path / f"c{index}.jpg"
            assert cv2.imwrite(str(path), pixels)
        paths.append(str(path))
    return paths


def _viewer(qapp, tmp_path, paths, **options):
    review_set = model.resolve_inputs(paths)
    viewer = FlipReviewViewer(review_set, str(tmp_path / "review.json"), **options)
    viewer.resize(900, 600)
    viewer.show()
    wait_for(qapp, lambda: (viewer.record is not None or viewer.error is not None
                            or viewer.mismatch is not None)
             and not any(entry.loading for entry in viewer._entries))
    return viewer


@pytest.fixture
def viewer(qapp, tmp_path):
    widget = _viewer(qapp, tmp_path, _images(tmp_path))
    yield widget
    widget.shutdown()
    widget.deleteLater()


# -- view geometry ----------------------------------------------------------

def test_zoom_steps_are_exact_and_clamped():
    view = ViewState((3908, 5216), (1000, 800))
    assert view.fit and view.scale == pytest.approx(800 / 5216)
    seen = [view.scale]
    for _ in range(30):
        view.zoom_in()
        seen.append(view.scale)
    for exact in (1.0, 2.0, 4.0, 8.0):
        assert exact in seen  # exact float, not merely close
    assert view.scale == MAX_ZOOM
    view.zoom_in()
    assert view.scale == MAX_ZOOM
    for _ in range(30):
        view.zoom_out()
    assert view.fit and view.scale == pytest.approx(800 / 5216)
    view.zoom_to(100)
    assert view.scale == MAX_ZOOM
    view.zoom_to(0.0001)
    assert view.fit


def test_small_image_can_still_be_seen_at_100_percent():
    view = ViewState((426, 570), (1400, 900))
    assert view.fit_scale() > 1.0 and view.min_scale() == 1.0
    view.zoom_out()
    assert view.scale == 1.5
    view.zoom_out()
    assert view.scale == 1.0 and not view.fit
    view.zoom_out()
    assert view.scale == 1.0
    view.zoom_in()
    view.zoom_in()
    assert view.fit  # the fit scale is a step of its own
    view.set_fit()
    view.toggle_actual()
    assert view.scale == 1.0


@pytest.mark.parametrize("anchor", [(100.0, 80.0), (731.0, 422.0), (500.0, 400.0)])
def test_wheel_zoom_keeps_the_pixel_under_the_cursor(anchor):
    view = ViewState((3908, 5216), (1000, 800))
    view.zoom_to(0.5, anchor)
    before = view.view_to_image(*anchor, exact=True)
    for _ in range(5):
        view.zoom_in(anchor)
        assert view.view_to_image(*anchor, exact=True) == pytest.approx(before, abs=1e-6)
        # Drawing snaps the origin to whole screen pixels: half a pixel at most.
        shown = view.view_to_image(*anchor)
        assert abs(shown[0] - before[0]) * view.scale <= 0.5 + 1e-6
        assert abs(shown[1] - before[1]) * view.scale <= 0.5 + 1e-6


def test_pan_is_bounded_to_the_image():
    view = ViewState((400, 300), (200, 100))
    view.zoom_to(1.0)
    view.pan(10_000, 10_000)
    assert view.view_to_image(0, 0) == (0, 0)
    view.pan(-10_000, -10_000)
    assert view.view_to_image(200, 100) == (400, 300)


# -- viewer -----------------------------------------------------------------

def test_loads_mixed_depths_and_flips_with_wrap_around(viewer):
    assert viewer.error is None and viewer.record is not None
    assert [viewer.current_index()] == [0]
    canvas = viewer.canvas
    QTest.keyClick(canvas, Qt.Key.Key_Left)
    assert viewer.current_index() == 2
    QTest.keyClick(canvas, Qt.Key.Key_Right)
    assert viewer.current_index() == 0
    QTest.keyClick(canvas, Qt.Key.Key_Right)
    QTest.keyClick(canvas, Qt.Key.Key_Right)
    assert viewer.current_index() == 2
    QTest.keyClick(canvas, Qt.Key.Key_Right)
    assert viewer.current_index() == 0
    # Space toggles back to the previously viewed candidate.
    QTest.keyClick(canvas, Qt.Key.Key_Space)
    assert viewer.current_index() == 2
    QTest.keyClick(canvas, Qt.Key.Key_Space)
    assert viewer.current_index() == 0
    QTest.keyClick(canvas, Qt.Key.Key_2, Qt.KeyboardModifier.ControlModifier)
    assert viewer.current_index() == 1
    QTest.keyClick(canvas, Qt.Key.Key_9, Qt.KeyboardModifier.ControlModifier)
    assert viewer.current_index() == 1


def test_flip_keeps_the_shared_view(viewer):
    view = viewer.canvas.view
    view.zoom_to(4.0)
    view.center_on(50, 40)
    before = (view.scale, view.cx, view.cy)
    viewer.next()
    viewer.next()
    assert (view.scale, view.cx, view.cy) == before
    assert viewer.canvas.levels[0].factor == 1


def test_16bit_display_matches_the_application_conversion(viewer, tmp_path):
    from utils.image_utils import to_display_uint8
    source = cv2.imread(str(tmp_path / "c0.tif"), cv2.IMREAD_UNCHANGED)
    assert source.dtype == np.uint16
    np.testing.assert_array_equal(viewer._entries[0].levels[0].array, to_display_uint8(source))
    assert viewer._entries[2].levels[0].array.ndim == 2  # grey stays one channel


def test_pixels_are_unsmoothed_from_100_percent(viewer):
    viewer.canvas.labels_visible = False
    view = viewer.canvas.view
    view.zoom_to(8.0)
    view.center_on(80, 60)
    viewer._view_changed()
    image = viewer.canvas.grab().toImage()
    ox, oy = view.origin()
    source = viewer.canvas.levels[0].array
    for ix, iy in ((70, 55), (75, 58), (80, 60)):
        b, g, r = (int(v) for v in source[iy, ix])
        for dx in range(8):
            for dy in range(8):
                color = image.pixelColor(int(ox + ix * 8 + dx), int(oy + iy * 8 + dy))
                assert (color.red(), color.green(), color.blue()) == (r, g, b)


def test_rating_keys_best_and_autosave(viewer, qapp, tmp_path):
    canvas = viewer.canvas
    cid = viewer.current_cid
    QTest.keyClick(canvas, Qt.Key.Key_3)
    assert viewer.record.rating(cid) == 3
    assert [b.isChecked() for b in viewer.panel.rating_buttons] == [False, False, True, False, False]
    QTest.keyClick(canvas, Qt.Key.Key_3)
    assert viewer.record.rating(cid) is None
    QTest.keyClick(canvas, Qt.Key.Key_5)
    QTest.keyClick(canvas, Qt.Key.Key_B)
    assert viewer.record.best == cid
    QTest.keyClick(canvas, Qt.Key.Key_Right)
    QTest.keyClick(canvas, Qt.Key.Key_B)
    assert viewer.record.best == viewer.current_cid  # one best per set
    wait_for(qapp, lambda: not viewer.record.dirty, timeout=5)
    saved = json.loads((tmp_path / "review.json").read_text(encoding="utf-8"))
    ratings = {c["id"]: c["rating"] for c in saved["candidates"]}
    assert ratings[cid] == 5 and saved["best"] == viewer.current_cid
    assert viewer.panel.rating_buttons[4].isEnabled()


def test_zoom_fullscreen_and_label_keys(viewer):
    canvas = viewer.canvas
    view = canvas.view
    QTest.keyClick(canvas, Qt.Key.Key_Z)
    assert view.scale == 1.0
    QTest.keyClick(canvas, Qt.Key.Key_Plus)
    assert view.scale == 1.5
    QTest.keyClick(canvas, Qt.Key.Key_Minus)
    assert view.scale == 1.0
    QTest.keyClick(canvas, Qt.Key.Key_0)
    assert view.fit
    QTest.keyClick(canvas, Qt.Key.Key_H)
    assert not canvas.labels_visible
    QTest.keyClick(canvas, Qt.Key.Key_H)
    assert canvas.labels_visible and canvas.hud_lines[0] == "c0.tif"
    assert canvas.hud_lines[1].startswith("1 / 3")
    QTest.keyClick(canvas, Qt.Key.Key_F)
    assert viewer.window().isFullScreen()
    QTest.keyClick(canvas, Qt.Key.Key_Escape)
    assert not viewer.window().isFullScreen()


def test_wheel_zoom_is_anchored_at_the_cursor(viewer, qapp):
    from PyQt6.QtGui import QWheelEvent
    canvas = viewer.canvas
    position = QPointF(200, 150)
    before = canvas.view.view_to_image(position.x(), position.y())
    event = QWheelEvent(position, canvas.mapToGlobal(position), QPoint(0, 0), QPoint(0, 120),
                        Qt.MouseButton.NoButton, Qt.KeyboardModifier.NoModifier,
                        Qt.ScrollPhase.NoScrollPhase, False)
    qapp.sendEvent(canvas, event)
    after = canvas.view.view_to_image(position.x(), position.y())
    assert canvas.view.scale > canvas.view.fit_scale()
    assert abs(after[0] - before[0]) * canvas.view.scale <= 0.5 + 1e-6


def test_pins_are_placed_listed_and_jumped_to(viewer, qapp):
    canvas = viewer.canvas
    view = canvas.view
    view.zoom_to(2.0)
    view.center_on(80, 60)
    viewer._view_changed()
    ox, oy = view.origin()
    target = QPoint(int(ox + 70 * 2 + 1), int(oy + 50 * 2 + 1))
    QTest.mouseClick(canvas, Qt.MouseButton.LeftButton, Qt.KeyboardModifier.ControlModifier, target)
    pins = viewer.record.pins
    assert len(pins) == 1 and (pins[0]["x"], pins[0]["y"]) == (70, 50)
    assert pins[0]["candidate"] == viewer.current_cid
    QTest.keyClicks(viewer.panel.pin_edit, "halo here")
    assert viewer.record.pin(pins[0]["id"])["text"] == "halo here"
    assert viewer.panel.pin_list.count() == 1
    assert "halo here" in viewer.panel.pin_list.item(0).text()
    # P then click also places a pin.
    viewer.focus_canvas()
    QTest.keyClick(canvas, Qt.Key.Key_P)
    assert canvas.pin_mode
    QTest.mouseClick(canvas, Qt.MouseButton.LeftButton, Qt.KeyboardModifier.NoModifier,
                     QPoint(int(ox + 100 * 2), int(oy + 70 * 2)))
    assert not canvas.pin_mode and len(viewer.record.pins) == 2
    # Jump from another candidate and fit: shows the pin's candidate at 100 %.
    first_pin = pins[0]["id"]
    viewer.next()
    view.zoom_to(8.0)
    view.center_on(150, 110)
    viewer.select_pin(first_pin, jump=True)
    assert viewer.current_cid == pins[0]["candidate"]
    assert view.scale == 8.0
    assert view.image_to_view(70.5, 50.5) == pytest.approx((view.view_w / 2, view.view_h / 2), abs=1)
    viewer.remove_selected_pin()
    assert len(viewer.record.pins) == 1


def test_text_fields_keep_their_keys(viewer):
    viewer.panel.comment_edit.setFocus()
    QTest.keyClicks(viewer.panel.comment_edit, "4 B soft")
    assert viewer.record.comment(viewer.current_cid) == "4 B soft"
    assert viewer.record.rating(viewer.current_cid) is None
    assert viewer.record.best is None
    viewer.activateWindow()
    viewer.panel.comment_edit.setFocus()
    QTest.keyClick(viewer.panel.comment_edit, Qt.Key.Key_Escape)
    assert viewer.canvas.hasFocus()
    QTest.keyClick(viewer.canvas, Qt.Key.Key_Right)
    assert viewer.current_index() == 1


def test_blind_mode_hides_labels_and_rates_true_candidates(qapp, tmp_path):
    paths = _images(tmp_path, 4)
    viewer = _viewer(qapp, tmp_path, paths, blind=True)
    try:
        assert viewer.record.blind_active
        labels = [viewer._label(cid) for cid in viewer.order]
        assert labels == [f"Candidate {letter}" for letter in "ABCD"]
        assert viewer.canvas.hud_lines[0] == "Candidate A"
        assert all(os.path.basename(p) not in viewer.panel.candidate_list.item(i).text()
                   for i, p in enumerate(paths))
        viewer.next()
        rated = viewer.current_cid
        QTest.keyClick(viewer.canvas, Qt.Key.Key_4)
        viewer.reveal()
        assert not viewer.record.blind_active
        assert viewer.current_cid == rated  # still on the same picture
        assert viewer.record.rating(rated) == 4
        assert viewer.canvas.hud_lines[0] == viewer.record.label(rated)
    finally:
        viewer.shutdown()


def test_mismatching_record_is_not_applied(qapp, tmp_path):
    paths = _images(tmp_path, 2)
    viewer = _viewer(qapp, tmp_path, paths)
    QTest.keyClick(viewer.canvas, Qt.Key.Key_5)
    viewer.shutdown()
    pixels = np.zeros((H, W, 3), np.uint8)
    cv2.imwrite(paths[1], pixels)  # a different picture with the same name
    messages = []
    review_set = model.resolve_inputs(paths)
    second = FlipReviewViewer(review_set, str(tmp_path / "review.json"), autostart=False)
    second.recordMismatch.connect(messages.append)
    second.start()
    try:
        wait_for(qapp, lambda: second.mismatch is not None)
        assert second.record is None
        assert "c1.png has changed" in messages[0]
        assert not second.panel.current_box.isEnabled()
        QTest.keyClick(second.canvas, Qt.Key.Key_3)  # ignored, nothing saved
        saved = json.loads((tmp_path / "review.json").read_text(encoding="utf-8"))
        assert saved["candidates"][0]["rating"] == 5
        second.start_new_record()
        assert second.record is not None and second.record.rating(second.current_cid) is None
        assert any(".mismatch-" in name for name in os.listdir(tmp_path))
    finally:
        second.shutdown()


def test_dimension_mismatch_names_files(qapp, tmp_path):
    paths = _images(tmp_path, 2)
    cv2.imwrite(paths[1], np.zeros((H + 1, W, 3), np.uint8))
    viewer = _viewer(qapp, tmp_path, paths)
    try:
        assert viewer.error and "c0.tif" in viewer.error and "c1.png" in viewer.error
        assert viewer.record is None
        assert viewer.error in viewer.canvas.message
    finally:
        viewer.shutdown()


def test_single_image_opens_without_flipping(qapp, tmp_path):
    viewer = _viewer(qapp, tmp_path, _images(tmp_path, 1))
    try:
        QTest.keyClick(viewer.canvas, Qt.Key.Key_Right)
        QTest.keyClick(viewer.canvas, Qt.Key.Key_Space)
        assert viewer.current_index() == 0 and viewer.record is not None
        assert viewer.canvas.hud_lines[1].startswith("1 / 1")
    finally:
        viewer.shutdown()


def test_memory_budget_keeps_previews_and_reloads(qapp, tmp_path):
    paths = _images(tmp_path, 6, size=(1600, 1200))
    viewer = _viewer(qapp, tmp_path, paths, memory_budget=8 << 20)
    try:
        full = viewer.prepared_candidates()
        assert 0 in full and len(full) < 6
        assert all(entry.levels for entry in viewer._entries)  # previews for everyone
        viewer.show_position(3)
        wait_for(qapp, lambda: viewer._entries[viewer.current_index()].has_full())
        assert len(viewer.prepared_candidates()) < 6
    finally:
        viewer.shutdown()


def test_difference_shows_amplified_absolute_difference(viewer, qapp, tmp_path):
    viewer.show_position(1)  # 8-bit PNG
    viewer.set_reference()   # reference = c1.png
    viewer.show_position(2)  # grey JPEG
    QTest.keyClick(viewer.canvas, Qt.Key.Key_D)
    wait_for(qapp, lambda: not viewer._diff_pending)
    shown = viewer.canvas.levels[0].array
    first = cv2.cvtColor(cv2.imread(str(tmp_path / "c2.jpg"), cv2.IMREAD_GRAYSCALE),
                         cv2.COLOR_GRAY2BGR).astype(int)
    second = cv2.imread(str(tmp_path / "c1.png")).astype(int)
    np.testing.assert_array_equal(shown, np.clip(np.abs(first - second) * 4, 0, 255))
    assert "difference" in viewer.canvas.hud_badges[0]
    viewer.show_position(1)
    assert "reference" in viewer.canvas.hud_badges[0]
    QTest.keyClick(viewer.canvas, Qt.Key.Key_D)
    assert not viewer.canvas.hud_badges


def test_export_from_viewer_has_no_paths(viewer, tmp_path):
    viewer.rate(2)
    out = viewer.export(str(tmp_path / "share.json"))
    text = open(out, encoding="utf-8").read()
    assert str(tmp_path) not in text and json.loads(text)["candidates"][0]["rating"] == 2


def test_jumping_to_a_pin_from_fit_shows_it_at_100_percent(qapp, tmp_path):
    viewer = _viewer(qapp, tmp_path, _images(tmp_path, 2, size=(1600, 1200)))
    try:
        view = viewer.canvas.view
        assert view.fit and view.scale < 1.0
        pid = viewer.record.add_pin(1000, 700, viewer.record.set_ids[1], "seam")
        viewer.select_pin(pid, jump=True)
        assert viewer.current_index() == 1 and view.scale == 1.0
        assert view.image_to_view(1000.5, 700.5) == pytest.approx(
            (view.view_w / 2, view.view_h / 2), abs=0.5)
        assert viewer.canvas.selected_pin == pid
    finally:
        viewer.shutdown()


_RATIO_SCRIPT = r'''
import sys, time
import cv2, numpy as np
from PyQt6.QtCore import QPoint, QPointF, Qt
from PyQt6.QtGui import QWheelEvent
from PyQt6.QtTest import QTest
from PyQt6.QtWidgets import QApplication
app = QApplication([])
from core import flip_review as model
from widgets.flip_review import FlipReviewViewer
folder, expected = sys.argv[1], float(sys.argv[2])
rng = np.random.default_rng(3)
paths = []
for index in range(2):
    path = f"{folder}/r{index}.png"
    cv2.imwrite(path, rng.integers(0, 255, (300, 400, 3), dtype=np.uint8))
    paths.append(path)
viewer = FlipReviewViewer(model.resolve_inputs(paths), f"{folder}/review.json")
viewer.resize(1000, 700)
viewer.show()
deadline = time.monotonic() + 30
while viewer.record is None or any(e.loading for e in viewer._entries):
    assert time.monotonic() < deadline and viewer.error is None
    app.processEvents()
canvas, view = viewer.canvas, viewer.canvas.view
ratio = canvas.devicePixelRatioF()
assert ratio == expected, ratio
assert (view.view_w, view.view_h) == (round(canvas.width() * ratio), round(canvas.height() * ratio))
# Fit is computed against the device size of the canvas.
assert view.fit and abs(view.scale - min(view.view_w / 400, view.view_h / 300)) < 1e-9
source = viewer.canvas.levels[0].array

def check(scale):
    view.zoom_to(scale)
    view.center_on(200, 150)
    viewer._view_changed()
    canvas.labels_visible = False
    image = canvas.grab().toImage()
    assert (image.width(), image.height()) == (view.view_w, view.view_h)
    ox, oy = view.origin()
    block = int(scale)
    for iy in range(140, 160, 3):
        for ix in range(190, 210, 3):
            b, g, r = (int(v) for v in source[iy, ix])
            for dy in range(block):
                for dx in range(block):
                    c = image.pixelColor(int(ox + ix * block + dx), int(oy + iy * block + dy))
                    assert (c.red(), c.green(), c.blue()) == (r, g, b), (scale, ix, iy, dx, dy)
check(1.0)   # one image pixel per device pixel
check(8.0)   # exact 8x8 device blocks
# A click lands on the image pixel under it.
ox, oy = view.origin()
device = (ox + 203 * 8 + 4, oy + 151 * 8 + 4)
QTest.mouseClick(canvas, Qt.MouseButton.LeftButton, Qt.KeyboardModifier.ControlModifier,
                 QPoint(round(device[0] / ratio), round(device[1] / ratio)))
pin = viewer.record.pins[-1]
assert (pin["x"], pin["y"]) == (203, 151), pin
# The wheel keeps the image point under the cursor.
view.zoom_to(4.0)
view.center_on(200, 150)
viewer._view_changed()
logical = QPointF(canvas.width() / 2 - 31.0, canvas.height() / 2 + 17.0)
before = view.view_to_image(logical.x() * ratio, logical.y() * ratio, exact=True)
event = QWheelEvent(logical, canvas.mapToGlobal(logical), QPoint(0, 0), QPoint(0, 120),
                    Qt.MouseButton.NoButton, Qt.KeyboardModifier.NoModifier,
                    Qt.ScrollPhase.NoScrollPhase, False)
app.sendEvent(canvas, event)
after = view.view_to_image(logical.x() * ratio, logical.y() * ratio, exact=True)
assert view.scale == 6.0 and max(abs(a - b) for a, b in zip(after, before)) < 1e-6
viewer.shutdown()
print("RATIO-OK", ratio)
'''


@pytest.mark.parametrize("ratio", ["2", "1.5"])
def test_zoom_is_exact_in_device_pixels(tmp_path, ratio):
    import subprocess
    import sys
    script = tmp_path / "ratio_check.py"
    script.write_text(_RATIO_SCRIPT, encoding="utf-8")
    env = dict(os.environ, QT_QPA_PLATFORM="offscreen", QT_SCALE_FACTOR=ratio,
               PYTHONPATH=os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    env.pop("QT_QPA_PLATFORMTHEME", None)
    result = subprocess.run([sys.executable, str(script), str(tmp_path), ratio], env=env,
                            capture_output=True, text=True, timeout=120)
    assert result.returncode == 0, result.stdout + result.stderr
    assert f"RATIO-OK {float(ratio)}" in result.stdout


def test_each_candidate_has_a_distinct_colour(viewer):
    colours = [viewer.candidate_color(cid) for cid in viewer.order]
    assert len(set(colours)) == 3 and None not in colours
    assert viewer.canvas.hud_color == colours[0]
    viewer.next()
    assert viewer.canvas.hud_color == colours[1]
    assert not viewer.panel.candidate_list.item(1).icon().isNull()


def test_blind_colours_follow_letters_not_pictures(qapp, tmp_path):
    from utils.design_tokens import CATEGORICAL
    paths = _images(tmp_path, 4)
    viewer = _viewer(qapp, tmp_path, paths)
    try:
        for seed in range(3):
            viewer.record.stop_blind()
            viewer.record.start_blind(__import__("random").Random(seed))
            viewer._reorder(viewer.current_cid)
            letters = {viewer._label(cid)[-1]: viewer.candidate_color(cid) for cid in viewer.order}
            # A is always the first colour, whichever picture A is.
            assert letters == {letter: CATEGORICAL[i] for i, letter in enumerate("ABCD")}
        before = {cid: viewer.candidate_color(cid) for cid in viewer.order}
        viewer.reveal()
        assert {cid: viewer.candidate_color(cid) for cid in viewer.order} == before
    finally:
        viewer.shutdown()


def test_fullscreen_hides_the_panel_and_restores_it(viewer):
    canvas, panel = viewer.canvas, viewer.panel
    assert not panel.isHidden()
    QTest.keyClick(canvas, Qt.Key.Key_F)
    assert viewer.window().isFullScreen() and panel.isHidden()
    QTest.keyClick(canvas, Qt.Key.Key_N)
    assert not panel.isHidden()
    QTest.keyClick(canvas, Qt.Key.Key_N)
    QTest.keyClick(canvas, Qt.Key.Key_Escape)
    assert not viewer.window().isFullScreen() and not panel.isHidden()
    # A panel hidden before fullscreen stays hidden afterwards.
    QTest.keyClick(canvas, Qt.Key.Key_N)
    QTest.keyClick(canvas, Qt.Key.Key_F11)
    QTest.keyClick(canvas, Qt.Key.Key_F11)
    assert not viewer.window().isFullScreen() and panel.isHidden()
    # Leaving fullscreen through the window (not our keys) also restores it.
    QTest.keyClick(canvas, Qt.Key.Key_N)
    QTest.keyClick(canvas, Qt.Key.Key_F)
    viewer.window().showNormal()
    assert not panel.isHidden()
