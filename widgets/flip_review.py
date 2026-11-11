"""Flip review viewer: one shared view, many candidates, scores and notes.

Embeddable component: ``FlipReviewViewer(review_set, record_path, parent=...)``
needs no main window. The data side (review sets, records, export) lives in
``core.flip_review`` without Qt; user-visible strings live in
``widgets.flip_review_text``.

Display buffers
---------------
Each candidate is decoded once with ``core.image_loader.decode_image`` and
converted to 8-bit like the rest of the application (``to_display_uint8``:
16-bit values divided by 257). A pyramid of 2x area-averaged levels is built
next to the full-resolution buffer. Flipping only swaps which prepared
``QImage`` the canvas paints; nothing is decoded or converted on a flip.

The small pyramid levels (1/4 and smaller) of every candidate stay in memory.
Full and half resolution levels are kept within a byte budget, prioritised by
the current candidate, the previously viewed one (for A/B toggling), the
neighbours and the difference reference. Candidates evicted from the budget
show their preview level immediately and reload in the background.
"""
from __future__ import annotations

import os
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import List, Optional

import cv2
import numpy as np
from PyQt6.QtCore import QObject, QPointF, QRect, QRectF, Qt, QTimer, pyqtSignal
from PyQt6.QtGui import QColor, QFont, QImage, QPainter, QPainterPath, QPen
from PyQt6.QtWidgets import (QCheckBox, QComboBox, QFormLayout, QFrame, QGroupBox,
                             QHBoxLayout, QLabel, QLineEdit, QListWidget,
                             QListWidgetItem, QPlainTextEdit, QPushButton,
                             QScrollArea, QSizePolicy, QToolButton, QVBoxLayout,
                             QWidget)

from core import flip_review as model
from widgets.flip_review_text import TEXT, text

# Zoom ladder used by keys and the wheel, so 100/200/400/800 % are exact.
ZOOM_STEPS = (0.125, 1 / 6, 0.25, 1 / 3, 0.5, 2 / 3, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, 8.0)
MAX_ZOOM = 8.0
DIFF_GAINS = (4, 16, 64)
PREVIEW_FACTOR = 4          # levels at 1/4 and smaller are always kept
MIN_LEVEL_SIDE = 384        # stop the pyramid below this longest side
DEFAULT_BUDGET_CAP = 2 << 30
_EPS = 1e-9


# ---------------------------------------------------------------------------
# View geometry (no Qt types, unit-testable)
# ---------------------------------------------------------------------------

class ViewState:
    """Shared zoom/pan of all candidates.

    ``scale`` is screen pixels per image pixel; ``center`` is the image point
    at the middle of the viewport. In fit mode the scale follows the viewport.
    """

    def __init__(self, image_size=(1, 1), viewport=(1, 1)):
        self.image_w, self.image_h = (max(1, int(v)) for v in image_size)
        self.view_w, self.view_h = (max(1, int(v)) for v in viewport)
        self.fit = True
        self.scale = self.fit_scale()
        self.cx, self.cy = self.image_w / 2, self.image_h / 2

    # -- limits --
    def fit_scale(self) -> float:
        return min(MAX_ZOOM, self.view_w / self.image_w, self.view_h / self.image_h)

    def min_scale(self) -> float:
        return min(self.fit_scale(), 1.0)

    def steps(self) -> List[float]:
        """Zoom ladder from the minimum up to 800 %, including the fit scale."""
        low, fit = self.min_scale(), self.fit_scale()
        steps = [low] + [s for s in ZOOM_STEPS if s > low + _EPS]
        if all(abs(fit - s) > _EPS for s in steps):
            steps.append(fit)
        return sorted(steps)

    # -- updates --
    def set_image_size(self, width, height) -> None:
        self.image_w, self.image_h = max(1, int(width)), max(1, int(height))
        self.set_fit()

    def set_viewport(self, width, height) -> None:
        self.view_w, self.view_h = max(1, int(width)), max(1, int(height))
        if self.fit:
            self.set_fit()
        else:
            self.scale = max(self.min_scale(), self.scale)
            self._clamp()

    def set_fit(self) -> None:
        self.fit = True
        self.scale = self.fit_scale()
        self.cx, self.cy = self.image_w / 2, self.image_h / 2

    def zoom_to(self, scale: float, anchor=None) -> None:
        """Zoom keeping the image point under ``anchor`` (view coords) fixed."""
        scale = max(self.min_scale(), min(MAX_ZOOM, float(scale)))
        if anchor is None:
            anchor = (self.view_w / 2, self.view_h / 2)
        # The unsnapped mapping keeps repeated zooms free of drift.
        ix, iy = self.view_to_image(*anchor, exact=True)
        self.scale = scale
        self.fit = abs(scale - self.fit_scale()) < _EPS
        if self.fit:
            self.set_fit()
            return
        self.cx = ix - (anchor[0] - self.view_w / 2) / scale
        self.cy = iy - (anchor[1] - self.view_h / 2) / scale
        self._clamp()

    def zoom_in(self, anchor=None) -> None:
        for step in self.steps():
            if step > self.scale + _EPS:
                self.zoom_to(step, anchor)
                return

    def zoom_out(self, anchor=None) -> None:
        lower = [step for step in self.steps() if step < self.scale - _EPS]
        if lower:
            self.zoom_to(lower[-1], anchor)

    def toggle_actual(self, anchor=None) -> None:
        """Fit <-> 100 %."""
        if abs(self.scale - 1.0) < _EPS:
            self.set_fit()
        else:
            self.zoom_to(1.0, anchor)

    def pan(self, dx: float, dy: float) -> None:
        self.cx -= dx / self.scale
        self.cy -= dy / self.scale
        if not self.fit or self._overflows():
            self.fit = False
        self._clamp()

    def center_on(self, x: float, y: float) -> None:
        self.cx, self.cy = float(x), float(y)
        self.fit = False
        self._clamp()

    def _overflows(self) -> bool:
        return (self.image_w * self.scale > self.view_w + _EPS
                or self.image_h * self.scale > self.view_h + _EPS)

    def _clamp(self) -> None:
        for axis in ("x", "y"):
            size = self.image_w if axis == "x" else self.image_h
            view = self.view_w if axis == "x" else self.view_h
            half = view / 2 / self.scale
            value = self.cx if axis == "x" else self.cy
            value = size / 2 if size * self.scale <= view else max(half, min(size - half, value))
            if axis == "x":
                self.cx = value
            else:
                self.cy = value

    # -- mapping --
    def origin(self, exact: bool = False):
        """View position of image pixel (0, 0); whole pixels from 100 % up."""
        ox = self.view_w / 2 - self.cx * self.scale
        oy = self.view_h / 2 - self.cy * self.scale
        if self.scale >= 1.0 - _EPS and not exact:
            ox, oy = round(ox), round(oy)
        return ox, oy

    def image_to_view(self, x: float, y: float):
        ox, oy = self.origin()
        return ox + x * self.scale, oy + y * self.scale

    def view_to_image(self, x: float, y: float, exact: bool = False):
        ox, oy = self.origin(exact)
        return (x - ox) / self.scale, (y - oy) / self.scale

    def visible_image_rect(self):
        """(x0, y0, x1, y1) of the image area on screen, clipped to the image."""
        x0, y0 = self.view_to_image(0, 0)
        x1, y1 = self.view_to_image(self.view_w, self.view_h)
        return (max(0.0, x0), max(0.0, y0), min(float(self.image_w), x1),
                min(float(self.image_h), y1))

    def percent(self) -> float:
        return self.scale * 100.0


# ---------------------------------------------------------------------------
# Display buffers
# ---------------------------------------------------------------------------

@dataclass
class _Level:
    factor: int
    array: np.ndarray
    image: QImage

    @property
    def nbytes(self) -> int:
        return int(self.array.nbytes)


def _max_half(image: np.ndarray) -> np.ndarray:
    """2x reduction keeping the largest sample of each 2x2 block."""
    height, width = image.shape[:2]
    even = image[:height - height % 2, :width - width % 2]
    return np.ascontiguousarray(np.maximum(
        np.maximum(even[0::2, 0::2], even[0::2, 1::2]),
        np.maximum(even[1::2, 0::2], even[1::2, 1::2])))


def display_levels(image: np.ndarray, *, keep_peaks: bool = False) -> List[tuple]:
    """Decoded pixels -> [(factor, uint8 array)] from full size down.

    ``keep_peaks`` reduces by the 2x2 maximum instead of the area mean, so a
    one-pixel difference stays visible when the view is zoomed out.
    """
    from utils.image_utils import composite_transparent_for_display, to_display_uint8
    if image.ndim == 3 and image.shape[2] == 4:
        image = composite_transparent_for_display(image)
    display = to_display_uint8(image)
    if display.ndim == 3 and display.shape[2] == 1:
        display = display[:, :, 0]
    display = np.ascontiguousarray(display)
    levels = [(1, display)]
    factor, current = 1, display
    while max(current.shape[:2]) // 2 >= MIN_LEVEL_SIDE and factor < 16:
        height, width = current.shape[:2]
        current = (_max_half(current) if keep_peaks else
                   cv2.resize(current, (max(1, width // 2), max(1, height // 2)),
                              interpolation=cv2.INTER_AREA))
        factor *= 2
        levels.append((factor, np.ascontiguousarray(current)))
    return levels


def _qimage(array: np.ndarray) -> QImage:
    height, width = array.shape[:2]
    if array.ndim == 2:
        return QImage(array.data, width, height, array.strides[0],
                      QImage.Format.Format_Grayscale8)
    return QImage(array.data, width, height, array.strides[0], QImage.Format.Format_BGR888)


def _make_levels(raw_levels) -> List[_Level]:
    return [_Level(factor, array, _qimage(array)) for factor, array in raw_levels]


def decode_for_display(path: str):
    """Worker-thread job: decode one candidate into display levels."""
    from core.image_loader import decode_image
    image = decode_image(path, preserve_grayscale=True, allow_transparent_png=True)
    if image is None:
        raise ValueError("unsupported or unreadable image")
    height, width = image.shape[:2]
    return width, height, display_levels(image)


def difference_levels(path: str, reference: str, gain: int):
    """Amplified absolute difference of the decoded (full bit depth) samples."""
    from core.image_loader import decode_image
    first = decode_image(path, preserve_grayscale=True, allow_transparent_png=True)
    second = decode_image(reference, preserve_grayscale=True, allow_transparent_png=True)
    if first is None or second is None:
        raise ValueError("unsupported or unreadable image")
    if first.ndim != second.ndim:
        if first.ndim == 2:
            first = cv2.cvtColor(first, cv2.COLOR_GRAY2BGR)
        else:
            second = cv2.cvtColor(second, cv2.COLOR_GRAY2BGR)
    if first.shape[:2] != second.shape[:2]:
        raise ValueError("pixel dimensions differ")
    first = first[..., :3] if first.ndim == 3 else first
    second = second[..., :3] if second.ndim == 3 else second
    scale = 1.0
    if first.dtype != second.dtype:
        first = first.astype(np.float32) / np.iinfo(first.dtype).max
        second = second.astype(np.float32) / np.iinfo(second.dtype).max
        scale = 255.0
    elif first.dtype == np.uint16:
        scale = 1.0 / 257.0
    delta = cv2.absdiff(first, second)
    del first, second
    out = cv2.convertScaleAbs(delta, alpha=gain * scale).reshape(delta.shape)
    return display_levels(out, keep_peaks=True)


class _Bridge(QObject):
    """Delivers worker results to the GUI thread (queued connection)."""
    finished = pyqtSignal(object)


class _Entry:
    def __init__(self):
        self.levels: List[_Level] = []
        self.size = None          # (width, height) of decoded pixels
        self.sha256 = None
        self.loading = False
        self.error = None

    def has_full(self) -> bool:
        return bool(self.levels) and self.levels[0].factor == 1

    def resident_bytes(self) -> int:
        return sum(level.nbytes for level in self.levels if level.factor < PREVIEW_FACTOR)

    def drop_full(self) -> None:
        keep = [level for level in self.levels if level.factor >= PREVIEW_FACTOR]
        self.levels = keep or self.levels[-1:]


def default_memory_budget() -> int:
    """A quarter of the currently available RAM, capped at 2 GiB."""
    try:
        from utils.memory import available_memory_bytes
        available = available_memory_bytes()
    except Exception:
        available = 0
    if available <= 0:
        return 1 << 30
    return int(max(256 << 20, min(DEFAULT_BUDGET_CAP, available // 4)))


# ---------------------------------------------------------------------------
# Canvas
# ---------------------------------------------------------------------------

def _overlay_palette():
    """Overlay chips sit on image content, so they use the dark tokens."""
    from utils.design_tokens import palette
    return palette("dark")


def _theme_palette():
    import sys
    from utils.design_tokens import palette
    styles = sys.modules.get("ui.styles")
    return palette(getattr(styles, "CURRENT_THEME", "dark") if styles else "dark")


class FlipCanvas(QWidget):
    """Paints the current candidate through the shared ViewState."""

    pinPlaced = pyqtSignal(int, int)
    pinClicked = pyqtSignal(str)

    PIN_RADIUS = 9

    def __init__(self, viewer: "FlipReviewViewer"):
        super().__init__(viewer)
        self._viewer = viewer
        self.view = ViewState()
        self.levels: List[_Level] = []
        self.image_size = None
        self.hud_lines: List[str] = []
        self.hud_badges: List[str] = []
        self.status = ""
        self.message = ""
        self.pins: List[dict] = []
        self.current_cid = None
        self.selected_pin = None
        self.labels_visible = True
        self.help_visible = False
        self.pin_mode = False
        self._drag_from = None
        self._press_at = None
        self._wheel_rest = 0
        self.setMouseTracking(True)
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
        self.setMinimumSize(200, 150)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.setCursor(Qt.CursorShape.OpenHandCursor)
        self.setProperty("controlId", "flipReviewCanvas")
        self.setAccessibleName("flip review image")

    # -- state --
    def set_image_size(self, size) -> None:
        if size != self.image_size:
            self.image_size = size
            self.view.set_image_size(*size)
            self.view.set_viewport(self.width(), self.height())

    def set_levels(self, levels: List[_Level]) -> None:
        self.levels = levels
        self.update()

    def resizeEvent(self, event):  # noqa: N802
        self.view.set_viewport(self.width(), self.height())
        super().resizeEvent(event)

    # -- painting --
    def _choose_level(self) -> Optional[_Level]:
        if not self.levels:
            return None
        wanted = 1.0 / max(self.view.scale, _EPS)
        best = self.levels[-1]
        for level in self.levels:
            # Largest reduction that still has at least one source pixel
            # per screen pixel.
            if level.factor <= wanted + _EPS:
                best = level
        if best.factor > wanted + _EPS:
            best = self.levels[0]
        return best

    def paintEvent(self, event):  # noqa: N802
        painter = QPainter(self)
        theme = _theme_palette()
        painter.fillRect(self.rect(), QColor(theme["canvas"]))
        level = self._choose_level()
        if level is not None and self.image_size is not None:
            self._paint_image(painter, level)
            if self.labels_visible:
                self._paint_pins(painter)
        elif self.message:
            painter.setPen(QColor(theme["textMuted"]))
            painter.drawText(self.rect().adjusted(24, 24, -24, -24),
                             Qt.AlignmentFlag.AlignCenter | Qt.TextFlag.TextWordWrap,
                             self.message)
        overlay = _overlay_palette()
        if self.labels_visible and self.hud_lines:
            self._paint_hud(painter, overlay)
        status = text("pin_mode") if self.pin_mode else self.status
        if status:
            self._paint_status(painter, overlay, status)
        if self.help_visible:
            self._paint_help(painter, overlay)
        painter.end()

    def _paint_image(self, painter: QPainter, level: _Level) -> None:
        view = self.view
        width, height = self.image_size
        x0, y0, x1, y1 = view.visible_image_rect()
        if x1 <= x0 or y1 <= y0:
            return
        sx = level.image.width() / width
        sy = level.image.height() / height
        # Integer source pixels of this level covering the visible area.
        lx0, ly0 = int(np.floor(x0 * sx)), int(np.floor(y0 * sy))
        lx1 = min(level.image.width(), int(np.ceil(x1 * sx)))
        ly1 = min(level.image.height(), int(np.ceil(y1 * sy)))
        source = QRectF(lx0, ly0, lx1 - lx0, ly1 - ly0)
        tx0, ty0 = view.image_to_view(lx0 / sx, ly0 / sy)
        tx1, ty1 = view.image_to_view(lx1 / sx, ly1 / sy)
        target = QRectF(tx0, ty0, tx1 - tx0, ty1 - ty0)
        # Below 100 % smooth from the nearest pyramid level; at 100 % and
        # above every image pixel is drawn as a sharp block.
        smooth = view.scale < 1.0 - _EPS or level.factor > 1
        painter.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform, smooth)
        painter.drawImage(target, level.image, source)

    def _paint_pins(self, painter: QPainter) -> None:
        accent = QColor(_overlay_palette()["accent"])
        font = QFont(self.font())
        font.setPixelSize(11)
        font.setBold(True)
        painter.setFont(font)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        for number, pin in enumerate(self.pins, 1):
            x, y = self.view.image_to_view(pin["x"] + 0.5, pin["y"] + 0.5)
            if not (-20 <= x <= self.width() + 20 and -20 <= y <= self.height() + 20):
                continue
            center = QPointF(x, y)
            own = pin.get("candidate") == self.current_cid
            selected = pin["id"] == self.selected_pin
            radius = self.PIN_RADIUS + (3 if selected else 0)
            painter.setPen(QPen(QColor(0, 0, 0, 220), 4))
            painter.setBrush(Qt.BrushStyle.NoBrush)
            painter.drawEllipse(center, radius, radius)
            painter.setPen(QPen(QColor(255, 255, 255) if not selected else accent, 2))
            painter.setBrush(QColor(accent.red(), accent.green(), accent.blue(),
                                    230 if own else 90))
            painter.drawEllipse(center, radius, radius)
            # The marked pixel itself stays uncovered at high zoom.
            gap = max(2.0, self.view.scale / 2 + 1)
            painter.setPen(QPen(QColor(255, 255, 255), 1))
            for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                painter.drawLine(QPointF(x + dx * gap, y + dy * gap),
                                 QPointF(x + dx * (gap + 4), y + dy * (gap + 4)))
            painter.setPen(QColor(255, 255, 255))
            painter.drawText(QRectF(x + radius + 2, y - radius - 12, 40, 14),
                             Qt.AlignmentFlag.AlignLeft, str(number))
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, False)

    def _chip(self, painter, rect: QRectF, overlay) -> None:
        path = QPainterPath()
        path.addRoundedRect(rect, 6, 6)
        background = QColor(overlay["bg"])
        background.setAlpha(205)
        painter.fillPath(path, background)
        painter.setBrush(Qt.BrushStyle.NoBrush)
        painter.setPen(QPen(QColor(255, 255, 255, 60), 1))
        painter.drawPath(path)

    def _paint_hud(self, painter: QPainter, overlay) -> None:
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        title_font = QFont(self.font())
        title_font.setPixelSize(17)
        title_font.setBold(True)
        body_font = QFont(self.font())
        body_font.setPixelSize(14)
        lines = [(self.hud_lines[0], title_font)] + [(line, body_font) for line in self.hud_lines[1:]]
        lines += [(badge, body_font) for badge in self.hud_badges]
        widths, heights = [], []
        for line, font in lines:
            painter.setFont(font)
            metrics = painter.fontMetrics()
            widths.append(metrics.horizontalAdvance(line))
            heights.append(metrics.height())
        max_width = max(40, self.width() - 40)
        box = QRectF(12, 12, min(max_width, max(widths) + 32), sum(heights) + 14)
        self._chip(painter, box, overlay)
        y = box.top() + 7
        for (line, font), height in zip(lines, heights):
            painter.setFont(font)
            painter.setPen(QColor(overlay["text"]) if font is title_font
                           else QColor(overlay["textMuted"] if line not in self.hud_badges
                                       else overlay["warning"]))
            elided = painter.fontMetrics().elidedText(line, Qt.TextElideMode.ElideMiddle,
                                                      int(box.width() - 20))
            painter.drawText(QRectF(box.left() + 12, y, box.width() - 20, height),
                             Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter, elided)
            y += height
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, False)

    def _paint_status(self, painter, overlay, message: str) -> None:
        font = QFont(self.font())
        font.setPixelSize(13)
        painter.setFont(font)
        flags = Qt.AlignmentFlag.AlignLeft | Qt.TextFlag.TextWordWrap
        width = max(40, self.width() - 48)
        bounds = painter.fontMetrics().boundingRect(QRect(0, 0, width, 2000), int(flags), message)
        box = QRectF(12, self.height() - bounds.height() - 26,
                     bounds.width() + 24, bounds.height() + 14)
        self._chip(painter, box, overlay)
        painter.setPen(QColor(overlay["text"]))
        painter.drawText(box.adjusted(12, 7, -12, -7), flags, message)

    def _paint_help(self, painter, overlay) -> None:
        font = QFont(self.font())
        font.setPixelSize(14)
        painter.setFont(font)
        message = text("help_overlay")
        bounds = painter.fontMetrics().boundingRect(QRect(0, 0, 2000, 2000),
                                                    int(Qt.AlignmentFlag.AlignLeft), message)
        box = QRectF((self.width() - bounds.width()) / 2 - 18,
                     (self.height() - bounds.height()) / 2 - 14,
                     bounds.width() + 36, bounds.height() + 28)
        self._chip(painter, box, overlay)
        painter.setPen(QColor(overlay["text"]))
        painter.drawText(box.adjusted(18, 14, -18, -14), int(Qt.AlignmentFlag.AlignLeft), message)

    # -- interaction --
    def pin_at(self, pos: QPointF) -> Optional[str]:
        if not self.labels_visible:
            return None
        for pin in reversed(self.pins):
            x, y = self.view.image_to_view(pin["x"] + 0.5, pin["y"] + 0.5)
            if (x - pos.x()) ** 2 + (y - pos.y()) ** 2 <= (self.PIN_RADIUS + 3) ** 2:
                return pin["id"]
        return None

    def image_pixel(self, pos: QPointF):
        if self.image_size is None:
            return None
        x, y = self.view.view_to_image(pos.x(), pos.y())
        if 0 <= x < self.image_size[0] and 0 <= y < self.image_size[1]:
            return int(x), int(y)
        return None

    def wheelEvent(self, event):  # noqa: N802
        if self.image_size is None:
            return
        self._wheel_rest += event.angleDelta().y()
        anchor = (event.position().x(), event.position().y())
        while abs(self._wheel_rest) >= 120:
            if self._wheel_rest > 0:
                self.view.zoom_in(anchor)
                self._wheel_rest -= 120
            else:
                self.view.zoom_out(anchor)
                self._wheel_rest += 120
        self._viewer._view_changed()
        event.accept()

    def mousePressEvent(self, event):  # noqa: N802
        if event.button() == Qt.MouseButton.LeftButton:
            self.setFocus()
            self._press_at = QPointF(event.position())
            pin = self.pin_at(event.position())
            if pin is not None and not self.pin_mode:
                self.pinClicked.emit(pin)
                self._press_at = None
            elif not self.pin_mode and not event.modifiers() & Qt.KeyboardModifier.ControlModifier:
                self._drag_from = QPointF(event.position())
                self.setCursor(Qt.CursorShape.ClosedHandCursor)
            event.accept()
            return
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event):  # noqa: N802
        if self._drag_from is not None:
            delta = event.position() - self._drag_from
            self._drag_from = QPointF(event.position())
            self.view.pan(delta.x(), delta.y())
            self._viewer._view_changed()
        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event):  # noqa: N802
        if event.button() == Qt.MouseButton.LeftButton:
            place = (self._press_at is not None
                     and (self.pin_mode or event.modifiers() & Qt.KeyboardModifier.ControlModifier)
                     and (event.position() - self._press_at).manhattanLength() < 6)
            self._drag_from = None
            self._press_at = None
            self.setCursor(Qt.CursorShape.CrossCursor if self.pin_mode
                           else Qt.CursorShape.OpenHandCursor)
            if place:
                pixel = self.image_pixel(event.position())
                if pixel is not None:
                    self.pinPlaced.emit(*pixel)
            event.accept()
            return
        super().mouseReleaseEvent(event)

    def mouseDoubleClickEvent(self, event):  # noqa: N802
        if event.button() == Qt.MouseButton.LeftButton and not self.pin_mode:
            self.view.toggle_actual((event.position().x(), event.position().y()))
            self._viewer._view_changed()
            event.accept()
            return
        super().mouseDoubleClickEvent(event)

    def keyPressEvent(self, event):  # noqa: N802
        if not self._viewer.handle_key(event):
            super().keyPressEvent(event)


# ---------------------------------------------------------------------------
# Notes panel
# ---------------------------------------------------------------------------

class _Section(QWidget):
    """A header button that shows or hides its body (collapsed by default)."""

    def __init__(self, title: str, body: QWidget, parent=None, *, expanded=False, tip=""):
        super().__init__(parent)
        self.button = QToolButton(self)
        self.button.setText(title)
        self.button.setCheckable(True)
        self.button.setChecked(expanded)
        self.button.setToolButtonStyle(Qt.ToolButtonStyle.ToolButtonTextBesideIcon)
        self.button.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self.button.setToolTip(tip)
        self.body = body
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self.button)
        layout.addWidget(body)
        self.button.toggled.connect(self._toggle)
        self._toggle(expanded)

    def _toggle(self, expanded):
        self.button.setArrowType(Qt.ArrowType.DownArrow if expanded else Qt.ArrowType.RightArrow)
        self.body.setVisible(expanded)


class ReviewPanel(QScrollArea):
    """Side panel for ratings and notes. It only edits the viewer's record."""

    def __init__(self, viewer: "FlipReviewViewer"):
        super().__init__(viewer)
        self._viewer = viewer
        self._updating = False
        self.setWidgetResizable(True)
        self.setFrameShape(QFrame.Shape.NoFrame)
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.setMinimumWidth(280)
        self.setMaximumWidth(380)
        self.setProperty("controlId", "flipReviewPanel")
        body = QWidget()
        layout = QVBoxLayout(body)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(8)

        self.mismatch_box = QGroupBox(self)
        mismatch_layout = QVBoxLayout(self.mismatch_box)
        self.mismatch_label = QLabel()
        self.mismatch_label.setWordWrap(True)
        self.mismatch_label.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        self.new_record_button = QPushButton(text("new_record"))
        self.new_record_button.setObjectName("flipReviewNewRecord")
        self.new_record_button.clicked.connect(viewer.start_new_record)
        mismatch_layout.addWidget(self.mismatch_label)
        mismatch_layout.addWidget(self.new_record_button)
        self.mismatch_box.hide()
        layout.addWidget(self.mismatch_box)

        candidates_box = QGroupBox(text("panel_candidates"))
        candidates_layout = QVBoxLayout(candidates_box)
        self.candidate_list = QListWidget()
        self.candidate_list.setObjectName("flipReviewCandidates")
        self.candidate_list.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self.candidate_list.setMaximumHeight(150)
        self.candidate_list.itemClicked.connect(
            lambda item: viewer.show_position(self.candidate_list.row(item)))
        candidates_layout.addWidget(self.candidate_list)
        layout.addWidget(candidates_box)

        self.current_box = QGroupBox(text("panel_current"))
        current = QVBoxLayout(self.current_box)
        rating_row = QHBoxLayout()
        rating_row.addWidget(QLabel(text("rating")))
        self.rating_buttons = []
        theme = _theme_palette()
        # The application sheet has no checked state for buttons.
        checked_style = (f"QToolButton:checked {{ background-color: {theme['accent']}; "
                         f"color: {theme['textOnAccent']}; border-color: {theme['accent']}; }}")
        for value in range(model.RATING_MIN, model.RATING_MAX + 1):
            button = QToolButton()
            button.setText(str(value))
            button.setCheckable(True)
            button.setFocusPolicy(Qt.FocusPolicy.NoFocus)
            button.setToolTip(text("rating_tip"))
            button.setObjectName(f"flipReviewRating{value}")
            button.setStyleSheet(checked_style)
            button.clicked.connect(lambda _checked, v=value: viewer.rate(v, toggle=True))
            rating_row.addWidget(button)
            self.rating_buttons.append(button)
        rating_row.addStretch(1)
        current.addLayout(rating_row)
        self.best_box = QCheckBox(text("best_checkbox"))
        self.best_box.setObjectName("flipReviewBest")
        self.best_box.setToolTip(text("best_tip"))
        self.best_box.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self.best_box.clicked.connect(lambda _checked: viewer.toggle_best())
        current.addWidget(self.best_box)
        self.comment_edit = QPlainTextEdit()
        self.comment_edit.setObjectName("flipReviewComment")
        self.comment_edit.setPlaceholderText(text("comment_placeholder"))
        self.comment_edit.setFixedHeight(70)
        self.comment_edit.textChanged.connect(self._comment_changed)
        current.addWidget(self.comment_edit)

        criteria_body = QWidget()
        criteria_form = QFormLayout(criteria_body)
        criteria_form.setContentsMargins(0, 0, 0, 0)
        self.criteria_combos = {}
        for name in viewer.criteria:
            combo = QComboBox()
            combo.setObjectName(f"flipReviewCriterion_{name}")
            combo.addItem(text("criterion_none"), None)
            for value in range(model.RATING_MIN, model.RATING_MAX + 1):
                combo.addItem(str(value), value)
            combo.setFocusPolicy(Qt.FocusPolicy.NoFocus)
            combo.currentIndexChanged.connect(
                lambda _index, n=name, c=combo: self._criterion_changed(n, c))
            criteria_form.addRow(TEXT.get(f"criterion_{name}", name), combo)
            self.criteria_combos[name] = combo
        self.criteria_section = _Section(text("criteria_toggle"), criteria_body,
                                         tip=text("criteria_tip"))
        current.addWidget(self.criteria_section)
        layout.addWidget(self.current_box)

        pins_box = QGroupBox(text("pins"))
        pins_layout = QVBoxLayout(pins_box)
        self.pin_list = QListWidget()
        self.pin_list.setObjectName("flipReviewPins")
        self.pin_list.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self.pin_list.setMaximumHeight(130)
        self.pin_list.itemClicked.connect(
            lambda item: viewer.select_pin(item.data(Qt.ItemDataRole.UserRole), jump=True))
        pins_layout.addWidget(self.pin_list)
        self.pin_edit = QLineEdit()
        self.pin_edit.setObjectName("flipReviewPinText")
        self.pin_edit.setPlaceholderText(text("pin_placeholder"))
        self.pin_edit.textEdited.connect(self._pin_text_changed)
        self.pin_edit.returnPressed.connect(viewer.focus_canvas)
        pins_layout.addWidget(self.pin_edit)
        pin_buttons = QHBoxLayout()
        self.pin_add_button = QPushButton(text("pin_add"))
        self.pin_add_button.setObjectName("flipReviewPinAdd")
        self.pin_add_button.setCheckable(True)
        self.pin_add_button.setToolTip(text("pin_add_tip"))
        self.pin_add_button.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self.pin_add_button.toggled.connect(viewer.set_pin_mode)
        self.pin_remove_button = QPushButton(text("pin_remove"))
        self.pin_remove_button.setObjectName("flipReviewPinRemove")
        self.pin_remove_button.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self.pin_remove_button.clicked.connect(viewer.remove_selected_pin)
        pin_buttons.addWidget(self.pin_add_button)
        pin_buttons.addWidget(self.pin_remove_button)
        pins_layout.addLayout(pin_buttons)
        layout.addWidget(pins_box)

        set_box = QGroupBox(text("set_comment"))
        set_layout = QVBoxLayout(set_box)
        self.set_comment_edit = QPlainTextEdit()
        self.set_comment_edit.setObjectName("flipReviewSetComment")
        self.set_comment_edit.setPlaceholderText(text("set_comment_placeholder"))
        self.set_comment_edit.setFixedHeight(70)
        self.set_comment_edit.textChanged.connect(self._set_comment_changed)
        set_layout.addWidget(self.set_comment_edit)
        reviewer_row = QFormLayout()
        self.reviewer_edit = QLineEdit()
        self.reviewer_edit.setObjectName("flipReviewReviewer")
        self.reviewer_edit.setPlaceholderText(text("reviewer_placeholder"))
        self.reviewer_edit.textEdited.connect(self._reviewer_changed)
        self.reviewer_edit.returnPressed.connect(viewer.focus_canvas)
        reviewer_row.addRow(text("reviewer"), self.reviewer_edit)
        set_layout.addLayout(reviewer_row)
        blind_row = QHBoxLayout()
        self.blind_box = QCheckBox(text("blind"))
        self.blind_box.setObjectName("flipReviewBlind")
        self.blind_box.setToolTip(text("blind_tip"))
        self.blind_box.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self.blind_box.clicked.connect(viewer.set_blind)
        self.reveal_button = QPushButton(text("reveal"))
        self.reveal_button.setObjectName("flipReviewReveal")
        self.reveal_button.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self.reveal_button.clicked.connect(self._reveal_clicked)
        self._reveal_armed = False
        blind_row.addWidget(self.blind_box)
        blind_row.addWidget(self.reveal_button)
        blind_row.addStretch(1)
        set_layout.addLayout(blind_row)
        layout.addWidget(set_box)

        export_box = QGroupBox(text("export_group"))
        export_row = QVBoxLayout(export_box)
        self.export_paths_box = QCheckBox(text("export_paths"))
        self.export_paths_box.setObjectName("flipReviewExportPaths")
        self.export_paths_box.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self.export_reviewer_box = QCheckBox(text("export_reviewer"))
        self.export_reviewer_box.setObjectName("flipReviewExportReviewer")
        self.export_reviewer_box.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self.export_button = QPushButton(text("export"))
        self.export_button.setObjectName("flipReviewExport")
        self.export_button.setToolTip(text("export_tip"))
        self.export_button.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self.export_button.clicked.connect(viewer.export_with_dialog)
        export_row.addWidget(self.export_paths_box)
        export_row.addWidget(self.export_reviewer_box)
        export_row.addWidget(self.export_button)
        layout.addWidget(export_box)

        self.status_label = QLabel()
        self.status_label.setObjectName("flipReviewStatus")
        self.status_label.setWordWrap(True)
        self.status_label.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        layout.addWidget(self.status_label)
        self.path_label = QLabel()
        self.path_label.setWordWrap(True)
        self.path_label.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        layout.addWidget(self.path_label)
        layout.addStretch(1)
        self.setWidget(body)
        self._editors = [self.current_box, pins_box, set_box, export_box]
        self._reveal_timer = QTimer(self)
        self._reveal_timer.setSingleShot(True)
        self._reveal_timer.timeout.connect(self._disarm_reveal)

    # -- record -> widgets --
    def refresh(self) -> None:
        viewer = self._viewer
        record = viewer.record
        self._updating = True
        try:
            enabled = record is not None
            for widget in self._editors:
                widget.setEnabled(enabled)
            self.blind_box.setEnabled(enabled)
            self.reveal_button.setEnabled(enabled and record.blind_active)
            self.candidate_list.clear()
            for position, cid in enumerate(viewer.order):
                self.candidate_list.addItem(QListWidgetItem(viewer.candidate_line(cid)))
            if viewer.order:
                self.candidate_list.setCurrentRow(viewer.position)
            if not enabled:
                return
            cid = viewer.current_cid
            rating = record.rating(cid)
            for button in self.rating_buttons:
                button.setChecked(rating is not None and int(button.text()) == rating)
            self.best_box.setChecked(record.best == cid)
            if self.comment_edit.toPlainText() != record.comment(cid):
                self.comment_edit.setPlainText(record.comment(cid))
            for name, combo in self.criteria_combos.items():
                index = combo.findData(record.criterion(cid, name))
                combo.setCurrentIndex(max(0, index))
            if self.set_comment_edit.toPlainText() != record.overall_comment:
                self.set_comment_edit.setPlainText(record.overall_comment)
            if not self.reviewer_edit.hasFocus():
                self.reviewer_edit.setText(record.reviewer)
            self.blind_box.setChecked(record.blind_used)
            self.blind_box.setEnabled(not record.blind_used or record.blind_active)
            self.reveal_button.setText(text("revealed") if record.blind_used and not
                                       record.blind_active else text("reveal"))
            self._refresh_pins()
        finally:
            self._updating = False

    def _refresh_pins(self) -> None:
        viewer = self._viewer
        selected = viewer.selected_pin
        self.pin_list.clear()
        for pin in viewer.record.pins:
            candidate = (viewer.record.display_label(pin["candidate"])
                         if pin["candidate"] else text("pin_set"))
            item = QListWidgetItem(text("pin_item", x=pin["x"], y=pin["y"],
                                        candidate=candidate, text=pin["text"]))
            item.setData(Qt.ItemDataRole.UserRole, pin["id"])
            self.pin_list.addItem(item)
            if pin["id"] == selected:
                self.pin_list.setCurrentItem(item)
        self.pin_edit.setEnabled(selected is not None)
        self.pin_remove_button.setEnabled(selected is not None)
        if selected is not None and not self.pin_edit.hasFocus():
            self.pin_edit.setText(viewer.record.pin(selected)["text"])
        elif selected is None:
            self.pin_edit.clear()
        self.pin_add_button.blockSignals(True)
        self.pin_add_button.setChecked(viewer.canvas.pin_mode)
        self.pin_add_button.blockSignals(False)

    def show_mismatch(self, message: Optional[str]) -> None:
        self.mismatch_box.setVisible(bool(message))
        self.mismatch_label.setText(message or "")

    # -- widgets -> record --
    def _comment_changed(self):
        if not self._updating and self._viewer.record is not None:
            self._viewer.record.set_comment(self._viewer.current_cid,
                                            self.comment_edit.toPlainText())
            self._viewer._record_changed(refresh_panel=False)

    def _set_comment_changed(self):
        if not self._updating and self._viewer.record is not None:
            self._viewer.record.set_overall_comment(self.set_comment_edit.toPlainText())
            self._viewer._record_changed(refresh_panel=False)

    def _reviewer_changed(self, value):
        if not self._updating and self._viewer.record is not None:
            self._viewer.record.set_reviewer(value)
            self._viewer._record_changed(refresh_panel=False)

    def _criterion_changed(self, name, combo):
        if not self._updating and self._viewer.record is not None:
            self._viewer.record.set_criterion(self._viewer.current_cid, name, combo.currentData())
            self._viewer._record_changed()

    def _pin_text_changed(self, value):
        viewer = self._viewer
        if not self._updating and viewer.record is not None and viewer.selected_pin:
            viewer.record.update_pin(viewer.selected_pin, text=value)
            viewer._record_changed(refresh_panel=False)
            item = self.pin_list.currentItem()
            if item is not None:
                pin = viewer.record.pin(viewer.selected_pin)
                candidate = (viewer.record.display_label(pin["candidate"])
                             if pin["candidate"] else text("pin_set"))
                item.setText(text("pin_item", x=pin["x"], y=pin["y"],
                                  candidate=candidate, text=pin["text"]))

    def _reveal_clicked(self):
        # Two clicks instead of a modal confirmation: revealing cannot be undone.
        if not self._reveal_armed:
            self._reveal_armed = True
            self.reveal_button.setText(text("reveal_confirm"))
            self._reveal_timer.start(4000)
            return
        self._disarm_reveal()
        self._viewer.reveal()

    def _disarm_reveal(self):
        self._reveal_armed = False
        self._reveal_timer.stop()
        record = self._viewer.record
        self.reveal_button.setText(text("revealed") if record is not None and record.blind_used
                                   and not record.blind_active else text("reveal"))


# ---------------------------------------------------------------------------
# Viewer component
# ---------------------------------------------------------------------------

class FlipReviewViewer(QWidget):
    """Shared-view flip comparison with ratings, notes and an auto-saved record."""

    loaded = pyqtSignal()
    loadFailed = pyqtSignal(str)
    recordMismatch = pyqtSignal(str)
    saved = pyqtSignal(str)
    saveFailed = pyqtSignal(str)
    currentChanged = pyqtSignal(int)

    SAVE_DELAY_MS = 400

    def __init__(self, review_set: model.ReviewSet, record_path: Optional[str] = None,
                 parent: Optional[QWidget] = None, *, memory_budget: Optional[int] = None,
                 blind: bool = False, reviewer: Optional[str] = None,
                 criteria=model.CRITERIA, workers: int = 2, autostart: bool = True):
        super().__init__(parent)
        self.review_set = review_set
        self.record_path = os.path.abspath(record_path or review_set.default_record_path())
        self.criteria = list(criteria)
        self.memory_budget = int(memory_budget) if memory_budget else default_memory_budget()
        self.record: Optional[model.ReviewRecord] = None
        self.mismatch: Optional[model.RecordMismatch] = None
        self.error: Optional[str] = None
        self.identities: List[Optional[model.CandidateIdentity]] = [None] * len(review_set.candidates)
        self._entries = [_Entry() for _ in review_set.candidates]
        self._ids = [f"#{i}" for i in range(len(review_set.candidates))]
        self.order: List[str] = list(self._ids)  # candidate ids in viewing order
        self.position = 0
        self._partner: Optional[str] = None
        self.selected_pin: Optional[str] = None
        self._blind_request = bool(blind)
        # Labels stay neutral until the record is bound when blind review is
        # requested or the existing record is still blind.
        self._conceal = self._blind_request or self._record_is_blind()
        self._record_problem = False
        self._reviewer_request = reviewer
        self.diff_enabled = False
        self.diff_gain = DIFF_GAINS[0]
        self._diff_cache = {}
        self._diff_pending = set()
        self._closed = False
        self._executor = ThreadPoolExecutor(max_workers=max(1, workers),
                                            thread_name_prefix="flip-review")
        self._bridge = _Bridge(self)
        self._bridge.finished.connect(self._on_result)
        self.load_started = None
        self.load_seconds = None
        self._fullscreen_restore_maximized = False

        self.setProperty("controlId", "flipReviewViewer")
        self.canvas = FlipCanvas(self)
        self.panel = ReviewPanel(self)
        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)
        layout.addWidget(self.canvas, 1)
        layout.addWidget(self.panel)
        self.canvas.pinPlaced.connect(self.add_pin)
        self.canvas.pinClicked.connect(lambda pid: self.select_pin(pid, jump=False))
        self._save_timer = QTimer(self)
        self._save_timer.setSingleShot(True)
        self._save_timer.timeout.connect(self.flush)
        self.canvas.message = text("loading", done=0, total=len(self._entries))
        self.panel.path_label.setText(text("record_path", path=self.record_path))
        self.panel.refresh()
        self.setFocusProxy(self.canvas)
        if autostart:
            self.start()

    # -- identifiers --
    @property
    def current_cid(self) -> str:
        return self.order[self.position]

    def _index_of(self, cid: str) -> int:
        return self._ids.index(cid)

    def current_index(self) -> int:
        """Review-set index of the candidate on screen."""
        return self._index_of(self.current_cid)

    def candidate_line(self, cid: str) -> str:
        label = self._label(cid)
        if self.record is None:
            return label
        rating = self.record.rating(cid)
        parts = [label, "★" * rating if rating else "-"]
        if self.record.best == cid:
            parts.append(text("best_marker"))
        if self.record.reference == cid:
            parts.append(text("reference_tag"))
        return "   ".join(parts)

    def _record_is_blind(self) -> bool:
        try:
            return model.ReviewRecord.read(self.record_path).blind_active
        except (OSError, ValueError):
            return False

    def _label(self, cid: str) -> str:
        if self.record is None and self._conceal:
            return text("blind_letter", letter="?")
        if self.record is not None:
            if self.record.blind_active:
                return text("blind_letter", letter=self.record.display_label(cid))
            return self.record.label(cid)
        return self.review_set.candidates[self._index_of(cid)].label

    # -- loading --
    def start(self) -> None:
        if self.load_started is not None:
            return
        self.load_started = time.perf_counter()
        for index in range(len(self._entries)):
            self._request_decode(index)
            self._submit(("hash", index), model.file_sha256,
                         self.review_set.candidates[index].path)

    def _submit(self, key, function, *args) -> None:
        if self._closed:
            return
        future = self._executor.submit(function, *args)

        def done(result, key=key):
            try:
                self._bridge.finished.emit((key, result))
            except RuntimeError:  # viewer already deleted
                pass
        future.add_done_callback(done)

    def _request_decode(self, index: int) -> None:
        entry = self._entries[index]
        if entry.loading or entry.has_full() or entry.error:
            return
        entry.loading = True
        self._submit(("decode", index), decode_for_display, self.review_set.candidates[index].path)

    def _on_result(self, payload) -> None:
        if self._closed:
            return
        (kind, *key), future = payload
        try:
            result = future.result()
            failure = None
        except Exception as exc:  # reported in the viewer, not raised
            result, failure = None, exc
        if kind == "diff":
            self._diff_finished(tuple(key), result, failure)
            return
        index = key[0]
        entry = self._entries[index]
        name = os.path.basename(self.review_set.candidates[index].path)
        if failure is not None:
            entry.loading = False
            entry.error = str(failure)
            self._fail(text("load_failed", name=name, error=failure))
            return
        if kind == "hash":
            entry.sha256 = result
        else:
            width, height, raw_levels = result
            entry.loading = False
            entry.size = (width, height)
            entry.levels = _make_levels(raw_levels)
        if entry.size is not None and entry.sha256 is not None and self.identities[index] is None:
            path = self.review_set.candidates[index].path
            self.identities[index] = model.identify(path, *entry.size, sha256=entry.sha256)
            if all(identity is not None for identity in self.identities):
                self._all_identified()
        if kind == "decode":
            self._enforce_budget()
            if self.error is None:
                self._show_current()

    def _fail(self, message: str) -> None:
        if self.error is not None:
            return
        self.error = message
        self.canvas.levels = []
        self.canvas.message = message
        self.canvas.hud_lines = []
        self.panel.status_label.setText(message)
        self.canvas.update()
        self.loadFailed.emit(message)

    def _all_identified(self) -> None:
        try:
            model.check_dimensions([(i.name, i.width, i.height) for i in self.identities])
        except model.DimensionMismatch as exc:
            self._fail(str(exc))
            return
        self.load_seconds = time.perf_counter() - self.load_started
        self._open_record()
        self.loaded.emit()

    def _open_record(self, record=None) -> None:
        try:
            record = record or model.ReviewRecord.open(self.record_path, self.review_set,
                                                       self.identities)
        except model.RecordMismatch as exc:
            self.mismatch = exc
            message = text("mismatch", path=self.record_path,
                           problems="\n".join(f"- {p}" for p in exc.problems))
            self.panel.show_mismatch(message)
            self.panel.status_label.setText(text("not_saving"))
            self._record_problem = True
            self.recordMismatch.emit(message)
            self.panel.refresh()
            self._show_current()
            return
        except model.ReviewError as exc:
            self.mismatch = None
            self.panel.show_mismatch(str(exc))
            self.panel.status_label.setText(text("not_saving"))
            self._record_problem = True
            self.recordMismatch.emit(str(exc))
            self.panel.refresh()
            self._show_current()
            return
        current = self.current_cid if self.order else None
        self.mismatch = None
        self.panel.show_mismatch(None)
        self._record_problem = False
        self._conceal = False
        self._partner = None
        self.record = record
        self._ids = record.set_ids
        index = int(current[1:]) if current and current.startswith("#") else 0
        if self._reviewer_request is not None:
            record.set_reviewer(self._reviewer_request)
        if self._blind_request and not record.blind_used:
            record.start_blind()
        self._blind_request = False
        self.order = record.order()
        self.position = self.order.index(self._ids[index]) if not record.blind_active else 0
        self.panel.refresh()
        self._show_current()
        if record.dirty:
            self._schedule_save()

    def start_new_record(self) -> None:
        """Keep the mismatching file as a backup and start a fresh record."""
        if self.record is not None or None in self.identities:
            return
        try:
            record = model.start_new_record(self.record_path, self.review_set, self.identities)
        except OSError as exc:
            self.panel.status_label.setText(text("save_failed", error=exc))
            return
        self._open_record(record)
        record.dirty = True
        self._schedule_save()

    # -- memory budget --
    def _priority(self) -> List[int]:
        """Review-set indices, most needed first."""
        if not self.order:
            return []
        wanted = [self.current_cid]
        if self._partner:
            wanted.append(self._partner)
        count = len(self.order)
        for step in range(1, count):
            wanted.append(self.order[(self.position + step) % count])
            wanted.append(self.order[(self.position - step) % count])
            if step == 1 and self.diff_enabled and self.record and self.record.reference:
                wanted.append(self.record.reference)
        seen = []
        for cid in wanted:
            index = self._index_of(cid)
            if index not in seen:
                seen.append(index)
        return seen

    def _enforce_budget(self) -> None:
        used = 0
        keep = set()
        for index in self._priority():
            entry = self._entries[index]
            cost = entry.resident_bytes() if entry.has_full() else self._estimate(index)
            # The current candidate is always admitted.
            if not keep or used + cost <= self.memory_budget:
                keep.add(index)
                used += cost
        for index, entry in enumerate(self._entries):
            if index in keep:
                self._request_decode(index)
            elif entry.has_full() and not entry.loading:
                entry.drop_full()

    def _estimate(self, index: int) -> int:
        sizes = [e for e in self._entries if e.size is not None]
        reference = self._entries[index] if self._entries[index].size else (sizes[0] if sizes else None)
        if reference is None:
            return 0
        width, height = reference.size
        channels = 1 if reference.levels and reference.levels[-1].array.ndim == 2 else 3
        return int(width * height * channels * 1.25)

    def resident_bytes(self) -> int:
        """Bytes held by prepared display buffers (all levels)."""
        total = sum(level.nbytes for entry in self._entries for level in entry.levels)
        total += sum(level.nbytes for levels in self._diff_cache.values() for level in levels)
        return total

    def prepared_candidates(self) -> List[int]:
        return [index for index, entry in enumerate(self._entries) if entry.has_full()]

    # -- showing --
    def _show_current(self) -> None:
        if self.error is not None or not self.order:
            return
        cid = self.current_cid
        index = self._index_of(cid)
        entry = self._entries[index]
        if entry.size is not None:
            self.canvas.set_image_size(entry.size)
        levels = entry.levels
        badges = []
        if self.diff_enabled and self.record is not None:
            reference = self.record.reference or self.order[0]
            if reference == cid:
                badges.append(text("diff_same"))
            else:
                key = (cid, reference, self.diff_gain)
                if key in self._diff_cache:
                    levels = self._diff_cache[key]
                    badges.append(text("diff_badge", gain=self.diff_gain,
                                       reference=self._label(reference)))
                else:
                    badges.append(text("diff_computing"))
                    self._request_diff(key)
        self.canvas.current_cid = cid
        self.canvas.levels = levels
        if levels:
            self.canvas.message = ""
        else:
            done = sum(1 for e in self._entries if e.levels)
            self.canvas.message = text("loading", done=done, total=len(self._entries))
        self.canvas.hud_badges = badges
        self.canvas.pins = self.record.pins if self.record is not None else []
        self.canvas.selected_pin = self.selected_pin
        self._update_hud()
        if self._record_problem:
            self.canvas.status = text("not_saving")
        elif (levels and levels[0].factor > 1
              and self.canvas.view.scale > 1.0 / levels[0].factor + _EPS):
            self.canvas.status = text("loading_full")
        else:
            self.canvas.status = ""
        self.canvas.update()

    def _update_hud(self) -> None:
        if not self.order:
            return
        cid = self.current_cid
        view = self.canvas.view
        percent = view.percent()
        shown = f"{percent:.0f}" if percent >= 10 else f"{percent:.1f}"
        zoom = text("zoom_fit", percent=shown) if view.fit else text("zoom_percent", percent=shown)
        details = [text("position", index=self.position + 1, total=len(self.order)), zoom]
        if self.record is not None:
            rating = self.record.rating(cid)
            if rating:
                details.append("★" * rating + "☆" * (model.RATING_MAX - rating))
            if self.record.best == cid:
                details.append(text("best_marker"))
        self.canvas.hud_lines = [self._label(cid), "   ".join(details)]

    def _view_changed(self) -> None:
        self._update_hud()
        self._show_current()

    def show_position(self, position: int) -> None:
        if not self.order:
            return
        position %= len(self.order)
        if position != self.position:
            self._partner = self.current_cid
            self.position = position
        self._after_flip()

    def show_candidate(self, cid: str) -> None:
        self.show_position(self.order.index(cid))

    def _after_flip(self) -> None:
        self._show_current()
        self._enforce_budget()
        self.panel.refresh()
        self.currentChanged.emit(self.current_index())

    def next(self) -> None:
        self.show_position(self.position + 1)

    def previous(self) -> None:
        self.show_position(self.position - 1)

    def toggle_partner(self) -> None:
        """Jump back to the previously viewed candidate (A/B toggling)."""
        if self._partner is not None and self._partner in self.order:
            self.show_candidate(self._partner)

    def focus_canvas(self) -> None:
        self.canvas.setFocus()

    # -- record edits --
    def _record_changed(self, refresh_panel: bool = True) -> None:
        if refresh_panel:
            self.panel.refresh()
        else:
            # Keep the candidate list in step without resetting editors.
            for row, cid in enumerate(self.order):
                item = self.panel.candidate_list.item(row)
                if item is not None:
                    item.setText(self.candidate_line(cid))
        self._update_hud()
        self.canvas.pins = self.record.pins if self.record is not None else []
        self.canvas.selected_pin = self.selected_pin
        self.canvas.update()
        self._schedule_save()

    def _schedule_save(self) -> None:
        if self.record is not None and self.record.dirty:
            self._save_timer.start(self.SAVE_DELAY_MS)

    def flush(self) -> bool:
        """Save now if anything changed; returns False on a write failure."""
        self._save_timer.stop()
        if self.record is None or not self.record.dirty:
            return True
        try:
            path = self.record.save()
        except (OSError, ValueError) as exc:
            message = text("save_failed", error=exc)
            self.panel.status_label.setText(message)
            self.saveFailed.emit(message)
            return False
        self.panel.status_label.setText(text("saved", time=time.strftime("%H:%M:%S")))
        self.saved.emit(path)
        return True

    def rate(self, value: Optional[int], *, toggle: bool = False) -> None:
        if self.record is None:
            return
        cid = self.current_cid
        if toggle and self.record.rating(cid) == value:
            value = None
        self.record.set_rating(cid, value)
        self._record_changed()

    def toggle_best(self) -> None:
        if self.record is not None:
            self.record.toggle_best(self.current_cid)
            self._record_changed()

    def set_reference(self) -> None:
        if self.record is not None:
            self.record.set_reference(self.current_cid)
            self._record_changed()
            self._show_current()

    def add_pin(self, x: int, y: int) -> Optional[str]:
        if self.record is None:
            return None
        pid = self.record.add_pin(x, y, self.current_cid)
        self.set_pin_mode(False)
        self.selected_pin = pid
        self._record_changed()
        if not self.panel.isVisible():
            self.panel.show()
        self.panel.pin_edit.setFocus()
        return pid

    def select_pin(self, pid: Optional[str], *, jump: bool = False) -> None:
        self.selected_pin = pid
        if pid is not None and jump and self.record is not None:
            pin = self.record.pin(pid)
            if pin["candidate"] in self.order and pin["candidate"] != self.current_cid:
                self._partner = self.current_cid
                self.position = self.order.index(pin["candidate"])
                self.currentChanged.emit(self.current_index())
            view = self.canvas.view
            if view.scale < 1.0:
                view.zoom_to(1.0)
            view.center_on(pin["x"] + 0.5, pin["y"] + 0.5)
            self._enforce_budget()
        self.panel.refresh()
        self._show_current()

    def remove_selected_pin(self) -> None:
        if self.record is not None and self.selected_pin is not None:
            self.record.remove_pin(self.selected_pin)
            self.selected_pin = None
            self._record_changed()

    def set_pin_mode(self, enabled: bool) -> None:
        enabled = bool(enabled) and self.record is not None
        self.canvas.pin_mode = enabled
        self.canvas.setCursor(Qt.CursorShape.CrossCursor if enabled
                              else Qt.CursorShape.OpenHandCursor)
        self.panel.pin_add_button.blockSignals(True)
        self.panel.pin_add_button.setChecked(enabled)
        self.panel.pin_add_button.blockSignals(False)
        self.canvas.update()

    def set_blind(self, enabled: bool) -> None:
        if self.record is None:
            return
        current = self.current_cid
        if enabled and not self.record.blind_used:
            self.record.start_blind()
        elif not enabled and self.record.blind_active:
            self.record.stop_blind()
        self._reorder(current)

    def reveal(self) -> None:
        if self.record is not None and self.record.blind_active:
            current = self.current_cid
            self.record.reveal()
            self._reorder(current)

    def _reorder(self, current: str) -> None:
        self.order = self.record.order()
        self.position = self.order.index(current) if not self.record.blind_active else 0
        self._partner = None
        self._record_changed()
        self._show_current()
        self.currentChanged.emit(self.current_index())

    # -- difference --
    def toggle_difference(self) -> None:
        self.diff_enabled = not self.diff_enabled
        if not self.diff_enabled:
            self._diff_cache.clear()
        self._enforce_budget()
        self._show_current()

    def cycle_difference_gain(self) -> None:
        self.diff_gain = DIFF_GAINS[(DIFF_GAINS.index(self.diff_gain) + 1) % len(DIFF_GAINS)]
        if self.diff_enabled:
            self._show_current()

    def _request_diff(self, key) -> None:
        if key in self._diff_pending:
            return
        self._diff_pending.add(key)
        cid, reference, gain = key
        self._submit(("diff", *key), difference_levels,
                     self.review_set.candidates[self._index_of(cid)].path,
                     self.review_set.candidates[self._index_of(reference)].path, gain)

    def _diff_finished(self, key, result, failure) -> None:
        self._diff_pending.discard(key)
        if failure is not None:
            self.panel.status_label.setText(str(failure))
            return
        if not self.diff_enabled:
            return
        self._diff_cache[key] = _make_levels(result)
        # Two differences at most: the current one and the A/B partner's.
        while len(self._diff_cache) > 2:
            keep = {(self.current_cid, self.record.reference if self.record else None,
                     self.diff_gain)}
            oldest = next(k for k in self._diff_cache if k not in keep)
            del self._diff_cache[oldest]
        self._show_current()

    # -- display toggles --
    def toggle_labels(self) -> None:
        self.canvas.labels_visible = not self.canvas.labels_visible
        self.canvas.update()

    def toggle_help(self) -> None:
        self.canvas.help_visible = not self.canvas.help_visible
        self.canvas.update()

    def toggle_panel(self) -> None:
        self.panel.setVisible(not self.panel.isVisible())

    def toggle_fullscreen(self) -> None:
        window = self.window()
        if window.isFullScreen():
            self.leave_fullscreen()
            return
        self._fullscreen_restore_maximized = window.isMaximized()
        window.showFullScreen()

    def leave_fullscreen(self) -> bool:
        window = self.window()
        if not window.isFullScreen():
            return False
        if self._fullscreen_restore_maximized:
            window.showMaximized()
        else:
            window.showNormal()
        return True

    # -- keys --
    def handle_key(self, event) -> bool:
        key = event.key()
        modifiers = event.modifiers() & ~Qt.KeyboardModifier.KeypadModifier
        ctrl = bool(modifiers & Qt.KeyboardModifier.ControlModifier)
        shift = bool(modifiers & Qt.KeyboardModifier.ShiftModifier)
        view = self.canvas.view
        digits = {getattr(Qt.Key, f"Key_{n}"): n for n in range(10)}
        if key in digits and ctrl:
            number = digits[key]
            if 1 <= number <= len(self.order):
                self.show_position(number - 1)
            return True
        if ctrl or modifiers & Qt.KeyboardModifier.AltModifier:
            return False
        if key == Qt.Key.Key_Right:
            self.next()
        elif key == Qt.Key.Key_Left:
            self.previous()
        elif key == Qt.Key.Key_Space:
            self.toggle_partner()
        elif key in digits and model.RATING_MIN <= digits[key] <= model.RATING_MAX:
            self.rate(digits[key], toggle=True)
        elif key == Qt.Key.Key_0:
            view.set_fit()
            self._view_changed()
        elif key in (Qt.Key.Key_Plus, Qt.Key.Key_Equal):
            view.zoom_in()
            self._view_changed()
        elif key in (Qt.Key.Key_Minus, Qt.Key.Key_Underscore):
            view.zoom_out()
            self._view_changed()
        elif key == Qt.Key.Key_Z:
            view.toggle_actual()
            self._view_changed()
        elif key in (Qt.Key.Key_F, Qt.Key.Key_F11):
            self.toggle_fullscreen()
        elif key == Qt.Key.Key_Escape:
            if not self.canvas.hasFocus():
                self.focus_canvas()  # leave a text field
            elif self.canvas.help_visible:
                self.toggle_help()
            elif self.canvas.pin_mode:
                self.set_pin_mode(False)
            else:
                return self.leave_fullscreen()
        elif key == Qt.Key.Key_H:
            self.toggle_labels()
        elif key == Qt.Key.Key_B:
            self.toggle_best()
        elif key == Qt.Key.Key_D:
            if shift:
                self.cycle_difference_gain()
            else:
                self.toggle_difference()
        elif key == Qt.Key.Key_R:
            self.set_reference()
        elif key == Qt.Key.Key_P:
            self.set_pin_mode(not self.canvas.pin_mode)
        elif key == Qt.Key.Key_N:
            self.toggle_panel()
        elif key == Qt.Key.Key_C:
            if self.record is not None:
                self.panel.show()
                self.panel.comment_edit.setFocus()
        elif key in (Qt.Key.Key_Question, Qt.Key.Key_F1):
            self.toggle_help()
        else:
            return False
        return True

    def keyPressEvent(self, event):  # noqa: N802
        if not self.handle_key(event):
            super().keyPressEvent(event)

    # -- export --
    def export(self, path: str, *, include_paths: bool = False,
               include_reviewer: bool = False) -> str:
        if self.record is None:
            raise model.ReviewError(text("not_saving"))
        self.flush()
        return self.record.export(path, include_paths=include_paths,
                                  include_reviewer=include_reviewer)

    def export_with_dialog(self) -> None:
        if self.record is None:
            return
        from utils.file_dialogs import FileDialog
        stem = os.path.splitext(os.path.basename(self.record_path))[0]
        suggestion = os.path.join(os.path.dirname(self.record_path), f"{stem}-export.json")
        path, _selected = FileDialog.getSaveFileName(self, text("export_title"), suggestion,
                                                     text("export_filter"))
        if not path:
            return
        try:
            self.export(path, include_paths=self.panel.export_paths_box.isChecked(),
                        include_reviewer=self.panel.export_reviewer_box.isChecked())
        except (OSError, ValueError) as exc:
            self.panel.status_label.setText(text("export_failed", error=exc))
            return
        self.panel.status_label.setText(text("export_done", path=path))

    # -- lifetime --
    def shutdown(self) -> None:
        """Save pending changes and stop background work; safe to call twice."""
        if self._closed:
            return
        self.flush()
        self._closed = True
        self._executor.shutdown(wait=False, cancel_futures=True)

    def closeEvent(self, event):  # noqa: N802
        self.shutdown()
        super().closeEvent(event)

    def hideEvent(self, event):  # noqa: N802
        self.flush()
        super().hideEvent(event)
