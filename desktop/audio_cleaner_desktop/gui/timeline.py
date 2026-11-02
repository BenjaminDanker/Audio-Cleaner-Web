"""The timeline of the editor screen (the web's Timeline.jsx): the ruler with the suggestions' pins, the waveform with the handles of the chosen
part and the playhead, and a row for each of Voice, Echo removal (a file with a second audio track), Leveling and Subtitles, each as sections that
say what is done there; a last row, Music, is not there yet. Drag across the waveform or a row to choose a part, click a section to choose it,
click the ruler or the waveform to move the playhead. Zoom makes it wider than the window (it scrolls).

This draws a `doc.Doc` (its model, waveform and state) and tells its owner what the person did through signals; it edits nothing itself.
"""
from __future__ import annotations

from PySide6.QtCore import QPointF, QRectF, QSize, Qt, Signal
from PySide6.QtGui import QColor, QFont, QFontMetrics, QPainter, QPainterPath, QPen
from PySide6.QtWidgets import QFrame, QGraphicsOpacityEffect, QHBoxLayout, QScrollArea, QSizePolicy, QToolButton, QVBoxLayout, QWidget

from ..editor_model import clock
from . import icons
from .copy import loudness_name
from .studio_widgets import app_palette, brand, is_dark, lane_color, tint

RULER_H, WAVE_H, LANE_H, GAP = 30, 92, 30, 6
HANDLE_W = 14
DRAG_PX = 5
NICE = (1, 2, 5, 10, 15, 20, 30, 60, 120, 300, 600, 900, 1200, 1800, 3600, 7200)
LABEL_W, LABEL_W_HEAR = 84, 112
LANE_NAMES = {"voice": "Voice", "echo": "Echo removal", "level": "Leveling", "subs": "Subtitles", "music": "Music"}
HEARABLE = ("voice", "echo", "level")


def nice(x: float) -> int:
    return next((n for n in NICE if n >= x), 7200)


def split_at(sections: list[dict], cuts: list[float]) -> list[dict]:
    """The sections, divided where the person split the file (each piece can then be chosen by itself)."""
    if not cuts:
        return sections
    out = []
    for s in sections:
        edges = [s["start"], *[c for c in sorted(cuts) if s["start"] + 1e-6 < c < s["end"] - 1e-6], s["end"]]
        for lo, hi in zip(edges, edges[1:]):
            out.append({**s, "start": lo, "end": hi})
    return out


def lanes_of(doc, subs_label: str) -> list[tuple[str, list[dict]]]:
    """[(lane, sections)] for the file: Voice, Echo removal (when the file has a second track), Leveling, Subtitles (the whole file), Music."""
    m = doc.model
    whole = [{"start": 0.0, "end": doc.seconds, "on": bool(subs_label), "label": subs_label or "Off", "pending": False, "settings": None}]
    voice = [{**s, "on": s["mode"] != "off", "label": s["label"] or "Off"} for s in m.voice_sections()]
    out = [("voice", split_at(voice, doc.cuts))]
    if m.echo_available:
        out.append(("echo", split_at([{**s, "label": "Echo removal" if s["on"] else "Off"} for s in m.sections("echo")], doc.cuts)))
    level = []
    for s in m.sections("leveling"):
        name = loudness_name((s["settings"] or {}).get("loudness_lufs")) if s["on"] else ""
        level.append({**s, "label": name or "Off"})
    out.append(("level", split_at(level, doc.cuts)))
    out.append(("subs", whole))
    out.append(("music", []))
    return out


class Plot(QWidget):
    """The drawn part: the ruler, the waveform and the rows. Its width is the width of the whole recording (zoom makes it wider)."""

    seekRequested = Signal(float)
    selectionDragged = Signal(float, float)  # seconds, while the mouse drags a part or a handle
    sectionClicked = Signal(float, float)  # the start and end of the section that was clicked
    pinClicked = Signal(str)

    def __init__(self, parent: QWidget | None = None):
        super().__init__(parent)
        self.doc = None
        self.playhead = 0.0
        self.progress: float | None = None  # while a run goes: how far it is (0..1)
        self.hearing_text = ""
        self.subs_label = ""
        self.hear_on = {"voice": True, "echo": True, "level": True}
        self._drag: tuple | None = None
        self.setMouseTracking(True)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        self.setAccessibleName("Timeline")
        self.setToolTip("Click to move the playhead. Drag across the waveform or a row to choose a part; drag its handles to change it.")

    # -- geometry -------------------------------------------------------------------------------------------------------------
    def lanes(self) -> list[tuple[str, list[dict]]]:
        return lanes_of(self.doc, self.subs_label) if self.doc is not None and self.doc.model is not None else []

    def total_height(self) -> int:
        return RULER_H + WAVE_H + GAP + max(1, len(self.lanes()) or 4) * (LANE_H + GAP)

    def lane_top(self, index: int) -> int:
        return RULER_H + WAVE_H + GAP + index * (LANE_H + GAP)

    def sizeHint(self) -> QSize:  # noqa: N802
        return QSize(600, self.total_height())

    def refresh(self) -> None:
        self.setFixedHeight(self.total_height())
        self.update()

    @property
    def seconds(self) -> float:
        return max(self.doc.seconds, 1e-9) if self.doc is not None else 1.0

    def x_of(self, t: float) -> float:
        return t / self.seconds * (self.width() - 1)

    def time_at(self, x: float) -> float:
        return max(0.0, min(self.seconds, x / max(self.width() - 1, 1) * self.seconds))

    def _selection(self) -> tuple[float, float] | None:
        m = self.doc.model if self.doc is not None else None
        return m.selection if m is not None else None

    def _pin_positions(self) -> list[tuple[str, float, str]]:
        m = self.doc.model
        if m is None or not self.doc.show_suggestions or self.progress is not None:
            return []
        return [(s.id, (s.start + s.end) / 2, s.state) for s in m.suggestions if s.state != "dismissed"]

    def pin_at(self, x: float, y: float) -> str | None:
        if y > RULER_H:
            return None
        for sid, t, _ in self._pin_positions():
            if abs(self.x_of(t) - x) <= 12:
                return sid
        return None

    def handle_at(self, x: float, y: float) -> str | None:
        if self.doc is None or self.doc.model is None or not (RULER_H <= y < RULER_H + WAVE_H):
            return None
        a, b = self.doc.selection()
        left, right = self._handle_rects(a, b)
        if left[0] - 2 <= x <= left[1] + 2:
            return "h0"
        if right[0] - 2 <= x <= right[1] + 2:
            return "h1"
        return None

    def _handle_rects(self, a: float, b: float) -> tuple[tuple[float, float], tuple[float, float]]:
        """The x range of each handle: the start's sits left of the start, the end's right of the end, and both stay inside the timeline."""
        xa, xb = self.x_of(a), self.x_of(b)
        end_a = max(xa, HANDLE_W)
        start_b = min(xb, self.width() - HANDLE_W)
        return (end_a - HANDLE_W, end_a), (start_b, start_b + HANDLE_W)

    def lane_at(self, y: float) -> int | None:
        i = int((y - (RULER_H + WAVE_H + GAP)) // (LANE_H + GAP))
        lanes = self.lanes()
        if y >= RULER_H + WAVE_H + GAP and 0 <= i < len(lanes) and (y - self.lane_top(i)) <= LANE_H:
            return i
        return None

    # -- mouse ----------------------------------------------------------------------------------------------------------------
    def mousePressEvent(self, event) -> None:  # noqa: N802
        if event.button() != Qt.MouseButton.LeftButton or self.doc is None or self.doc.model is None:
            return
        x, y = event.position().x(), event.position().y()
        pin = self.pin_at(x, y)
        if pin is not None:
            self.pinClicked.emit(pin)
            return
        t = self.time_at(x)
        if y <= RULER_H:
            self._drag = ("seek", x, t)
            self.seekRequested.emit(t)
            return
        handle = self.handle_at(x, y)
        if handle:
            self._drag = (handle, x, t)
            return
        lane = self.lane_at(y)
        if y >= RULER_H + WAVE_H and lane is None:
            return
        self._drag = ("lane" if lane is not None else "wave", x, t, lane)

    def mouseMoveEvent(self, event) -> None:  # noqa: N802
        x, y = event.position().x(), event.position().y()
        if self._drag is None:
            self.setCursor(Qt.CursorShape.SizeHorCursor if self.handle_at(x, y) else Qt.CursorShape.CrossCursor if y > RULER_H else Qt.CursorShape.ArrowCursor)
            return
        kind = self._drag[0]
        t = self.time_at(x)
        if kind == "seek":
            self.seekRequested.emit(t)
        elif kind in ("h0", "h1"):
            a, b = self.doc.selection()
            if kind == "h0":
                self.selectionDragged.emit(min(t, b - 1.0), b)
            else:
                self.selectionDragged.emit(a, max(t, a + 1.0))
        elif kind in ("wave", "lane", "select") and abs(x - self._drag[1]) >= DRAG_PX:
            self._drag = ("select", self._drag[1], self._drag[2], self._drag[3] if len(self._drag) > 3 else None)
            t0 = self._drag[2]
            self.selectionDragged.emit(min(t0, t), max(t0, t))

    def mouseReleaseEvent(self, event) -> None:  # noqa: N802
        d, self._drag = self._drag, None
        if d is None:
            return
        if d[0] == "wave":
            self.seekRequested.emit(d[2])  # a click on the waveform: move the playhead
        elif d[0] == "lane":
            lane = d[3]
            for s in self.lanes()[lane][1]:
                if s["start"] <= d[2] < s["end"] or (d[2] >= self.seconds and s["end"] >= self.seconds):
                    if self.lanes()[lane][0] != "music":
                        self.sectionClicked.emit(s["start"], s["end"])
                    break

    # -- painting -------------------------------------------------------------------------------------------------------------
    def paintEvent(self, event) -> None:  # noqa: N802
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        doc = self.doc
        if doc is None or doc.model is None:
            p.end()
            return
        dark = is_dark(self)
        pal = app_palette()
        text = pal.windowText().color()
        muted = QColor(text)
        muted.setAlphaF(0.55)
        surface2 = QColor(text)
        surface2.setAlphaF(0.06)
        accent = brand(dark)
        w, T = self.width(), doc.seconds
        wave = QRectF(0, RULER_H, w, WAVE_H)
        # the waveform
        p.setPen(Qt.PenStyle.NoPen)
        p.setBrush(surface2)
        p.drawRoundedRect(wave, 8, 8)
        mid = wave.center().y()
        mx = WAVE_H / 2 - 3

        def bars(values, color: QColor, alpha: float = 1.0) -> None:
            n = max(40, int(w / 3.2))
            step = w / n
            N = len(values)
            path = QPainterPath()
            for i in range(n):
                a = i * N // n
                b = max(a + 1, (i + 1) * N // n)
                v = max(values[a:b]) / 100.0 if a < N else 0.0
                y = max(1.5, v * mx)
                x = (i + 0.5) * step
                path.moveTo(x, mid - y)
                path.lineTo(x, mid + y)
            c = QColor(color)
            c.setAlphaF(alpha)
            pen = QPen(c, 2)
            pen.setCapStyle(Qt.PenCapStyle.RoundCap)
            p.setPen(pen)
            p.setBrush(Qt.BrushStyle.NoBrush)
            p.drawPath(path)

        peaks = doc.peaks
        if peaks:
            bars(peaks, muted, 0.45 if doc.heard else 1.0)
        elif doc.wave in ("none", "failed", "reading"):
            bars([4] * 80, muted, 0.5)
        if doc.heard and self.hear_on.get("voice", True) and self.progress is None:
            bars(doc.heard, accent)
        # the bands of the chosen part (on the waveform and every row)
        sel = self._selection()
        a, b = (sel if sel is not None else (0.0, T))
        xa, xb = self.x_of(a), self.x_of(b)
        whole = sel is None or (a <= 0 and b >= T)
        bottom = self.total_height()
        band = QColor(accent)
        band.setAlphaF(0.05 if whole else 0.14)
        p.setPen(Qt.PenStyle.NoPen)
        p.setBrush(band)
        p.drawRect(QRectF(xa, RULER_H, max(1.0, xb - xa), bottom - RULER_H))
        if not whole:
            p.setPen(QPen(accent, 1.5))
            p.drawLine(QPointF(xa, RULER_H), QPointF(xa, bottom))
            p.drawLine(QPointF(xb, RULER_H), QPointF(xb, bottom))
        # the ruler
        font = QFont(self.font())
        font.setPointSizeF(max(7.0, font.pointSizeF() * 0.78))
        p.setFont(font)
        fm = QFontMetrics(font)
        step = nice(T / (max(6, round(w / 110)) if w >= 560 else 3))
        tick = nice(step / (3 if w >= 560 else 6))
        p.setPen(QPen(muted, 1))
        t = 0.0
        while t <= T + 1e-6:
            x = self.x_of(t)
            p.drawLine(QPointF(x, RULER_H - 5), QPointF(x, RULER_H))
            t += tick
        t = 0.0
        last_right = -1e9
        while t <= T + 1e-6:
            if not (t != T and T - t < step * 0.4):
                label = clock(t)
                tw = fm.horizontalAdvance(label)
                x = self.x_of(t)
                lx = 0.0 if t == 0 else (x - tw if t >= T - 1e-6 else x - tw / 2)
                if lx > last_right + 6:
                    p.drawText(QPointF(lx, RULER_H - 9), label)
                    last_right = lx + tw
            t += step
        # suggestions: dashed boxes on the waveform, pins on the ruler
        sugg = lane_color("sugg", dark)
        for s in doc.model.suggestions:
            if s.state == "open" and doc.show_suggestions and self.progress is None:
                box = QColor(sugg)
                box.setAlphaF(0.14)
                p.setBrush(box)
                p.setPen(QPen(sugg, 1.5, Qt.PenStyle.DashLine))
                p.drawRoundedRect(QRectF(self.x_of(s.start), RULER_H, max(2.0, self.x_of(s.end) - self.x_of(s.start)), WAVE_H), 4, 4)
        for sid, tm, state in self._pin_positions():
            x = self.x_of(tm)
            applied = state == "applied"
            col = QColor("#12794f") if applied else sugg
            p.setPen(Qt.PenStyle.NoPen)
            p.setBrush(col)
            p.drawEllipse(QPointF(x, 12), 10, 10)
            p.setPen(QPen(QColor("white"), 1.8))
            if applied:
                p.drawPolyline([QPointF(x - 4, 12), QPointF(x - 1, 15), QPointF(x + 5, 9)])
            else:
                p.drawLine(QPointF(x, 6), QPointF(x, 18))
                p.drawLine(QPointF(x - 6, 12), QPointF(x + 6, 12))
        # the tag on the waveform
        tag = "Reading the sound of your file…" if doc.wave == "reading" and not peaks else self.hearing_text
        if tag:
            tw = QFontMetrics(font).horizontalAdvance(tag) + 16
            p.setPen(QPen(muted, 1))
            bg = QColor(pal.base().color())
            bg.setAlphaF(0.88)
            p.setBrush(bg)
            p.drawRoundedRect(QRectF(6, RULER_H + 5, tw, 18), 9, 9)
            p.setPen(text)
            p.drawText(QRectF(6, RULER_H + 5, tw, 18), int(Qt.AlignmentFlag.AlignCenter), tag)
        # the rows
        lanes = self.lanes()
        for i, (lane, sections) in enumerate(lanes):
            top = self.lane_top(i)
            track = QRectF(0, top, w, LANE_H)
            p.setPen(QPen(QColor(text.red(), text.green(), text.blue(), 40), 1))
            p.setBrush(surface2)
            p.drawRoundedRect(track.adjusted(0.5, 0.5, -0.5, -0.5), 7, 7)
            if lane == "music":
                self._draw_na(p, track, font, muted, text, pal)
                continue
            for s in sections:
                self._section(p, lane, s, top, w, font, dark, text, muted)
        # the part that is being made
        if self.progress is not None:
            px = self.x_of(self.progress * T)
            tintc = QColor(accent)
            tintc.setAlphaF(0.12)
            p.setPen(Qt.PenStyle.NoPen)
            p.setBrush(tintc)
            p.drawRect(QRectF(0, RULER_H, px, WAVE_H))
            p.setPen(QPen(accent, 2))
            p.drawLine(QPointF(px, RULER_H), QPointF(px, RULER_H + WAVE_H))
        # the handles
        if self.progress is None:
            (l0, _), (r0, _) = self._handle_rects(a, b)
            self._handle(p, l0, accent)
            self._handle(p, r0, accent)
        # the playhead
        px = self.x_of(self.playhead)
        p.setPen(QPen(text, 2))
        p.drawLine(QPointF(px, RULER_H - 2), QPointF(px, bottom))
        p.end()

    def _handle(self, p: QPainter, x: float, accent: QColor) -> None:
        r = QRectF(x, RULER_H + WAVE_H / 2 - 16, HANDLE_W, 32)
        p.setPen(QPen(app_palette().base().color(), 2))
        p.setBrush(accent)
        p.drawRoundedRect(r, 6, 6)
        p.setPen(QPen(QColor(255, 255, 255, 180), 1.5))
        cx = r.center().x()
        p.drawLine(QPointF(cx - 2, r.top() + 9), QPointF(cx - 2, r.bottom() - 9))
        p.drawLine(QPointF(cx + 2, r.top() + 9), QPointF(cx + 2, r.bottom() - 9))

    def _section(self, p: QPainter, lane: str, s: dict, top: float, w: float, font: QFont, dark: bool, text: QColor, muted: QColor) -> None:
        x0, x1 = self.x_of(s["start"]), self.x_of(s["end"])
        rect = QRectF(x0 + 1, top + 2, max(2.0, x1 - x0 - 2), LANE_H - 4)
        key = lane if lane != "voice" or s.get("mode") != "rebuild" else "rebuild"
        color = lane_color(key, dark)
        if not s["on"]:
            p.setPen(QPen(QColor(text.red(), text.green(), text.blue(), 60), 1, Qt.PenStyle.DashLine))
            p.setBrush(Qt.BrushStyle.NoBrush)
            p.drawRoundedRect(rect, 5, 5)
            p.setPen(muted)
        else:
            fill = QColor(color)
            fill.setAlphaF(0.30 if lane != "subs" else 0.26)
            p.setBrush(fill)
            outline = QColor(color)
            outline.setAlphaF(0.75)
            p.setPen(QPen(outline, 1.2, Qt.PenStyle.DashLine if s.get("pending") else Qt.PenStyle.SolidLine))
            p.drawRoundedRect(rect, 5, 5)
            p.setPen(text)
        bold = QFont(font)
        bold.setBold(True)
        p.setFont(bold)
        label = s["label"]
        fm = QFontMetrics(bold)
        if fm.horizontalAdvance(label) + 14 <= rect.width():
            p.drawText(rect.adjusted(7, 0, -3, 0), int(Qt.AlignmentFlag.AlignVCenter | Qt.AlignmentFlag.AlignLeft), label)
        p.setFont(font)

    def _draw_na(self, p: QPainter, track: QRectF, font: QFont, muted: QColor, text: QColor, pal) -> None:
        p.setPen(QPen(QColor(text.red(), text.green(), text.blue(), 28), 1))
        x = track.left() + 11
        while x < track.right():
            p.drawLine(QPointF(x, track.top() + 1), QPointF(x, track.bottom() - 1))
            x += 12
        label = "Coming"
        fm = QFontMetrics(font)
        tw = fm.horizontalAdvance(label) + 16
        r = QRectF(track.center().x() - tw / 2, track.top() + 6, tw, LANE_H - 12)
        p.setPen(Qt.PenStyle.NoPen)
        p.setBrush(pal.base().color())
        p.drawRoundedRect(r, 4, 4)
        p.setPen(muted)
        p.setFont(font)
        p.drawText(r, int(Qt.AlignmentFlag.AlignCenter), label)


class LaneLabels(QWidget):
    """The names of the rows, in a column that stays when the timeline scrolls; after a run each row of what can be heard has a speaker switch."""

    hearToggled = Signal(str, bool)

    def __init__(self, plot: Plot, parent: QWidget | None = None):
        super().__init__(parent)
        self._plot = plot
        self._buttons: dict[str, QToolButton] = {}
        self._present: set[str] = set()
        self._hear = False
        self.setFixedWidth(LABEL_W)
        self.setSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Fixed)

    def set_hear(self, on: bool, enabled: bool = True, state: dict | None = None) -> None:
        self._hear = on
        self.setFixedWidth(LABEL_W_HEAR if on else LABEL_W)
        for lane in HEARABLE:
            b = self._buttons.get(lane)
            if b is not None:
                b.setVisible(on and lane in self._present)
                b.setEnabled(enabled)
                if state is not None:
                    b.blockSignals(True)
                    b.setChecked(bool(state.get(lane, True)))
                    b.blockSignals(False)
                    self._icon(b, lane)
        self.update()

    def _icon(self, b: QToolButton, lane: str) -> None:
        dark = is_dark(self)
        col = lane_color(lane, dark) if b.isChecked() else app_palette().windowText().color()
        b.setIcon(icons.icon("speaker" if b.isChecked() else "speaker-off", col, 14))

    def refresh(self) -> None:
        lanes = self._plot.lanes()
        self.setFixedHeight(self._plot.total_height())
        names = [lane for lane, _ in lanes]
        self._present = set(names)
        for lane in HEARABLE:
            b = self._buttons.get(lane)
            if b is None:
                b = QToolButton(self)
                b.setCheckable(True)
                b.setChecked(True)
                b.setAutoRaise(True)
                b.setFixedSize(24, 24)
                b.setIconSize(QSize(14, 14))
                b.setCursor(Qt.CursorShape.PointingHandCursor)
                b.setAccessibleName(f"Hear {LANE_NAMES[lane]}")
                b.setToolTip(f"Hear the file with {LANE_NAMES[lane]} or without it. The waveform shows the difference.")
                b.toggled.connect(lambda on, k=lane, btn=b: (self._icon(btn, k), self.hearToggled.emit(k, on)))
                self._buttons[lane] = b
                self._icon(b, lane)
            if lane in names:
                b.move(2, self._plot.lane_top(names.index(lane)) + (LANE_H - 24) // 2)
                b.setVisible(self._hear)
            else:
                b.hide()
        self.update()

    def paintEvent(self, _event) -> None:  # noqa: N802
        p = QPainter(self)
        muted = QColor(app_palette().windowText().color())
        muted.setAlphaF(0.7)
        font = QFont(self.font())
        font.setBold(True)
        font.setPointSizeF(max(7.5, font.pointSizeF() * 0.88))
        p.setFont(font)
        for i, (lane, _) in enumerate(self._plot.lanes()):
            top = self._plot.lane_top(i)
            left = 30 if self._hear and lane in HEARABLE else 4
            p.setPen(muted if lane != "music" else QColor(muted.red(), muted.green(), muted.blue(), 110))
            p.drawText(QRectF(left, top, self.width() - left, LANE_H), int(Qt.AlignmentFlag.AlignVCenter | Qt.AlignmentFlag.AlignLeft), LANE_NAMES[lane])
        p.end()


class TimelineView(QWidget):
    """The labels, and the plot in a scroll area (zoom makes the plot wider than the window)."""

    seekRequested = Signal(float)
    selectionDragged = Signal(float, float)
    sectionClicked = Signal(float, float)
    pinClicked = Signal(str)
    hearToggled = Signal(str, bool)

    def __init__(self, parent: QWidget | None = None):
        super().__init__(parent)
        self.zoom = 1
        self.plot = Plot()
        self.labels = LaneLabels(self.plot)
        self.scroll = QScrollArea()
        self.scroll.setFrameShape(QFrame.Shape.NoFrame)
        self.scroll.setWidgetResizable(False)
        self.scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        self.scroll.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.scroll.setWidget(self.plot)
        self.scroll.viewport().setAutoFillBackground(False)
        lay = QHBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(10)
        lay.addWidget(self.labels, 0, Qt.AlignmentFlag.AlignTop)
        lay.addWidget(self.scroll, 1)
        self.plot.seekRequested.connect(self.seekRequested)
        self.plot.selectionDragged.connect(self.selectionDragged)
        self.plot.sectionClicked.connect(self.sectionClicked)
        self.plot.pinClicked.connect(self.pinClicked)
        self.labels.hearToggled.connect(self.hearToggled)

    def set_doc(self, doc) -> None:
        self.plot.doc = doc
        self.plot.playhead = 0.0
        self.set_zoom(1)
        self.refresh()

    def set_zoom(self, zoom: int) -> None:
        self.zoom = max(1, min(8, int(zoom)))
        self._fit()
        self.ensure_playhead_visible()

    def _fit(self) -> None:
        vw = max(self.scroll.viewport().width(), 200)
        self.plot.setFixedWidth(vw * self.zoom)
        self.plot.refresh()
        self.scroll.setFixedHeight(self.plot.total_height() + (self.scroll.horizontalScrollBar().sizeHint().height() if self.zoom > 1 else 0))

    def refresh(self) -> None:
        self._fit()
        self.labels.refresh()
        self.plot.update()

    def resizeEvent(self, event) -> None:  # noqa: N802
        super().resizeEvent(event)
        self._fit()

    def set_playhead(self, seconds: float) -> None:
        self.plot.playhead = max(0.0, seconds)
        self.plot.update()

    def ensure_playhead_visible(self) -> None:
        if self.zoom <= 1:
            self.scroll.horizontalScrollBar().setValue(0)
            return
        x = int(self.plot.x_of(self.plot.playhead))
        self.scroll.ensureVisible(x, 0, 60, 0)

    def minimumSizeHint(self) -> QSize:  # noqa: N802
        return QSize(260, self.plot.total_height())


class GhostRows(QWidget):
    """The rows before a file is chosen: the names with grey bars (the editor is already there, only greyed)."""

    def __init__(self, parent: QWidget | None = None):
        super().__init__(parent)
        lay = QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(GAP)
        for name in ("Voice", "Leveling", "Subtitles", "Music"):
            row = QHBoxLayout()
            row.setSpacing(10)
            label = QLabel_(name)
            bar = QFrame()
            bar.setObjectName("GhostBar")
            bar.setFixedHeight(LANE_H)
            row.addWidget(label)
            row.addWidget(bar, 1)
            lay.addLayout(row)
        self.setEnabled(False)
        fade = QGraphicsOpacityEffect(self)
        fade.setOpacity(0.55)
        self.setGraphicsEffect(fade)


def QLabel_(text: str):  # noqa: N802
    from PySide6.QtWidgets import QLabel

    label = QLabel(text)
    label.setObjectName("LaneName")
    label.setFixedWidth(LABEL_W)
    return label


__all__ = ["TimelineView", "Plot", "GhostRows", "lanes_of", "split_at", "HEARABLE", "LANE_NAMES", "tint"]
