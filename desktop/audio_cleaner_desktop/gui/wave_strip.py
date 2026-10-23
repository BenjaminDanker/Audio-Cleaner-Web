"""The strip under the player in the per-part editor: ruler with the suggestion pins, the waveform, the playhead, the selected part with
its two handles, and (with "Show all layers") one row per layer with its parts. Click to move the playhead, drag across the waveform or a
row to choose a part, drag a handle to change it. Nothing here knows about audio or jobs: it draws an `editor_model.EditModel`."""
from __future__ import annotations

import math

from PySide6.QtCore import QPointF, QRectF, Qt, Signal
from PySide6.QtGui import QBrush, QColor, QFont, QPainter, QPainterPath, QPen
from PySide6.QtWidgets import QSizePolicy, QWidget

from ..editor_model import LAYER_NAMES, LAYER_ORDER, EditModel, clock

LAYER_COLORS = {"noise": "#3b5bdb", "rebuild": "#0f9b8e", "echo": "#c2417b", "leveling": "#b7791f"}
RULER_H, WAVE_H, BAND_H, ROW_H = 30, 56, 8, 26
HANDLE_R = 7  # pixels around a handle that grab it
DRAG_PX = 5  # a press that moves less than this is a click


class WaveStrip(QWidget):
    seekRequested = Signal(float)
    selectionDragged = Signal(float, float)  # the selection while the mouse drags (seconds)
    pinClicked = Signal(str)

    def __init__(self, model: EditModel, peaks: list[int] | None = None, parent: QWidget | None = None):
        super().__init__(parent)
        self.model = model
        self.peaks = list(peaks or [])
        self.playhead = 0.0
        self.rows = False
        self.hear = {k: True for k in LAYER_ORDER}
        self.missing: dict[str, list[tuple[float, float]]] = {}
        self._drag: tuple | None = None  # ("seek"|"select"|"h0"|"h1", press x, press t)
        self.setMouseTracking(True)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        self.setAccessibleName("Waveform of the recording")
        self.setToolTip("Click to move the playhead. Drag across the waveform to choose a part; drag its handles to change it.")
        self.setMinimumWidth(240)
        self._resize()
        model.on_change(self.update)

    # -- geometry ---------------------------------------------------------------------------------------------------------
    def _layers(self) -> list[str]:
        return [k for k in LAYER_ORDER if k != "echo" or self.model.echo_available]

    def _resize(self) -> None:
        h = RULER_H + WAVE_H + (len(self._layers()) * ROW_H + 6 if self.rows else BAND_H + 4)
        self.setFixedHeight(h)

    def set_rows(self, on: bool) -> None:
        self.rows = bool(on)
        self._resize()
        self.update()

    def set_peaks(self, peaks: list[int]) -> None:
        self.peaks = list(peaks)
        self.update()

    def set_playhead(self, t: float) -> None:
        self.playhead = max(0.0, min(self.model.seconds, t))
        self.update()

    def set_hear(self, hear: dict) -> None:
        self.hear = dict(hear)
        self.update()

    def set_missing(self, missing: dict) -> None:
        self.missing = missing
        self.update()

    def x_of(self, t: float) -> float:
        return t / max(self.model.seconds, 1e-9) * (self.width() - 1)

    def time_at(self, x: float) -> float:
        return max(0.0, min(self.model.seconds, x / max(self.width() - 1, 1) * self.model.seconds))

    def pin_positions(self) -> list[tuple[str, float]]:
        return [(s.id, (s.start + s.end) / 2) for s in self.model.open_suggestions()]

    def pin_at(self, x: float, y: float) -> str | None:
        if y > RULER_H:
            return None
        for sid, t in self.pin_positions():
            if abs(self.x_of(t) - x) <= 9:
                return sid
        return None

    def handle_at(self, x: float, y: float) -> str | None:
        sel = self.model.selection
        if sel is None or y <= RULER_H:
            return None
        for name, t in (("h0", sel[0]), ("h1", sel[1])):
            if abs(self.x_of(t) - x) <= HANDLE_R:
                return name
        return None

    def row_at(self, y: float) -> str | None:
        """The layer whose row is at height y (rows view)."""
        if not self.rows or y < RULER_H + WAVE_H:
            return None
        i = int((y - RULER_H - WAVE_H - 3) // ROW_H)
        layers = self._layers()
        return layers[i] if 0 <= i < len(layers) else None

    # -- mouse ------------------------------------------------------------------------------------------------------------
    def mousePressEvent(self, event) -> None:  # noqa: N802
        if event.button() != Qt.MouseButton.LeftButton:
            return
        pos = event.position()
        x, y = pos.x(), pos.y()
        pin = self.pin_at(x, y)
        if pin is not None:
            self.pinClicked.emit(pin)
            return
        if y <= RULER_H:
            self._drag = ("seek", x, self.time_at(x))
            self.seekRequested.emit(self.time_at(x))
            return
        handle = self.handle_at(x, y)
        if handle:
            self._drag = (handle, x, self.time_at(x))
            return
        self._drag = ("press", x, self.time_at(x))

    def mouseMoveEvent(self, event) -> None:  # noqa: N802
        pos = event.position()
        if self._drag is None:
            self.setCursor(Qt.CursorShape.SizeHorCursor if self.handle_at(pos.x(), pos.y()) else Qt.CursorShape.ArrowCursor)
            return
        kind, x0, t0 = self._drag
        t = self.time_at(pos.x())
        if kind == "seek":
            self.seekRequested.emit(t)
        elif kind in ("h0", "h1") and self.model.selection is not None:
            a, b = self.model.selection
            self.selectionDragged.emit(min(t, b - 1.0), b) if kind == "h0" else self.selectionDragged.emit(a, max(t, a + 1.0))
        elif kind in ("press", "select") and abs(pos.x() - x0) >= DRAG_PX:
            self._drag = ("select", x0, t0)
            self.selectionDragged.emit(min(t0, t), max(t0, t))

    def mouseReleaseEvent(self, event) -> None:  # noqa: N802
        if self._drag is not None and self._drag[0] == "press":
            self.seekRequested.emit(self._drag[2])  # a click: move the playhead
        self._drag = None

    # -- painting ---------------------------------------------------------------------------------------------------------
    def paintEvent(self, event) -> None:  # noqa: N802
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        pal = self.palette()
        text = pal.windowText().color()
        muted = QColor(text)
        muted.setAlphaF(0.55)
        w = self.width()
        # the waveform
        wave = QRectF(0, RULER_H, w, WAVE_H)
        shade = QColor(text)
        shade.setAlphaF(0.06)
        p.fillRect(wave, shade)
        if self.peaks:
            path = QPainterPath()
            mid = wave.center().y()
            n = len(self.peaks)
            for i, v in enumerate(self.peaks):
                x = i / max(n - 1, 1) * (w - 1)
                h = v / 100.0 * (WAVE_H / 2 - 3)
                path.moveTo(x, mid - h)
                path.lineTo(x, mid + h)
            pen = QPen(muted, max(1.0, w / max(n, 1)))
            p.setPen(pen)
            p.drawPath(path)
        # the ruler
        p.setPen(QPen(muted, 1))
        font = QFont(self.font())
        font.setPointSizeF(max(7.0, font.pointSizeF() * 0.8))
        p.setFont(font)
        total = self.model.seconds
        step = next((s for s in (1, 2, 5, 10, 15, 30, 60, 120, 300, 600, 900, 1800, 3600) if s / max(total, 1e-9) * w >= 70), 3600)
        t = 0.0
        while t <= total + 1e-6:
            x = self.x_of(t)
            p.drawLine(QPointF(x, RULER_H - 4), QPointF(x, RULER_H))
            label = clock(t)
            tw = p.fontMetrics().horizontalAdvance(label)
            p.drawText(QPointF(min(max(0.0, x - tw / 2), w - tw), RULER_H - 7), label)
            t += step
        # the parts: a band under the waveform, or a row per layer
        y0 = RULER_H + WAVE_H
        if self.rows:
            for i, layer in enumerate(self._layers()):
                top = y0 + 3 + i * ROW_H
                color = QColor(LAYER_COLORS[layer])
                row_bg = QColor(text)
                row_bg.setAlphaF(0.04)
                p.fillRect(QRectF(0, top, w, ROW_H - 2), row_bg)
                for r in self.model.runs(layer):
                    self._block(p, r, top, ROW_H - 2, color, layer, label=True)
                p.setPen(QPen(muted, 1))
                if not self.hear.get(layer, True):
                    p.drawText(QPointF(6, top + ROW_H - 9), "not heard")
        else:
            for layer in LAYER_ORDER:
                for r in self.model.runs(layer):
                    self._block(p, r, y0 + 2, BAND_H, QColor(LAYER_COLORS[layer]), layer, label=False, slim=True)
        # the pins of the suggestions
        spark = QColor("#7c4dff")
        for sid, tm in self.pin_positions():
            x = self.x_of(tm)
            focus = sid == self.model.focus
            path = QPainterPath()
            r = 7 if focus else 5.5
            cy = 8
            for k in range(8):  # a four-pointed spark
                ang = k * math.pi / 4
                rad = r if k % 2 == 0 else r * 0.35
                pt = QPointF(x + rad * math.sin(ang), cy - rad * math.cos(ang))
                path.moveTo(pt) if k == 0 else path.lineTo(pt)
            path.closeSubpath()
            p.setBrush(QBrush(spark))
            p.setPen(Qt.PenStyle.NoPen)
            p.drawPath(path)
        # the selected part
        sel = self.model.selection
        if sel is not None:
            a, b = self.x_of(sel[0]), self.x_of(sel[1])
            tint = QColor("#3b5bdb")
            tint.setAlphaF(0.16)
            p.fillRect(QRectF(a, RULER_H, b - a, self.height() - RULER_H), tint)
            p.setPen(QPen(QColor("#3b5bdb"), 2))
            for x in (a, b):
                p.drawLine(QPointF(x, RULER_H), QPointF(x, self.height()))
                p.setBrush(QBrush(QColor("#3b5bdb")))
                p.drawRoundedRect(QRectF(x - 4, RULER_H + 6, 8, 20), 3, 3)
            label = f"{clock(sel[0])}–{clock(sel[1])}"
            lw = p.fontMetrics().horizontalAdvance(label) + 10
            lx = min(max(0.0, (a + b) / 2 - lw / 2), w - lw)
            p.setBrush(QBrush(QColor("#3b5bdb")))
            p.setPen(Qt.PenStyle.NoPen)
            p.drawRoundedRect(QRectF(lx, RULER_H + WAVE_H - 17, lw, 14), 6, 6)
            p.setPen(QColor("white"))
            p.drawText(QPointF(lx + 5, RULER_H + WAVE_H - 6), label)
        # the playhead
        px = self.x_of(self.playhead)
        p.setPen(QPen(text, 2))
        p.drawLine(QPointF(px, RULER_H - 2), QPointF(px, self.height()))
        p.end()

    def _block(self, p: QPainter, r: dict, top: float, height: float, color: QColor, layer: str, *, label: bool, slim: bool = False) -> None:
        x0, x1 = self.x_of(r["start"]), self.x_of(r["end"])
        rect = QRectF(x0, top, max(2.0, x1 - x0), height)
        fill = QColor(color)
        fill.setAlphaF(0.55 if not slim else 0.85)
        if not self.hear.get(layer, True):
            fill.setAlphaF(0.2)
        p.setBrush(QBrush(fill))
        p.setPen(QPen(color, 1.5, Qt.PenStyle.DashLine if r["pending"] else Qt.PenStyle.SolidLine))
        p.drawRoundedRect(rect, 3, 3)
        if label and rect.width() > 40:
            text = r["label"] or LAYER_NAMES[layer]
            p.setPen(self.palette().windowText().color())
            p.drawText(rect.adjusted(5, 0, -2, 0), int(Qt.AlignmentFlag.AlignVCenter | Qt.AlignmentFlag.AlignLeft), text)
