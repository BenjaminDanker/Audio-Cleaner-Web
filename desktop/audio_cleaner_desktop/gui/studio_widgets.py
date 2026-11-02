"""The small pieces the editor screen is made of (the web's Cleanup page: panes, the colour of each row, switches, pills, soft segmented buttons)."""
from __future__ import annotations

from PySide6.QtCore import QRectF, QSize, Qt, Signal
from PySide6.QtGui import QColor, QPainter
from PySide6.QtWidgets import QAbstractButton, QButtonGroup, QFrame, QHBoxLayout, QLabel, QPushButton, QSizePolicy, QVBoxLayout, QWidget

from .theme import set_property
from .widgets import Segmented

#: The colour of each row, as on the web (light, dark): Voice is the brand colour, Voice rebuild the accent.
LANE_COLORS = {
    "voice": ("#3b5bdb", "#4c68f0"), "rebuild": ("#0b8f8a", "#2bb5af"), "echo": ("#d9480f", "#ff922b"),
    "level": ("#b87500", "#f0b84a"), "subs": ("#c2255c", "#f06595"), "sugg": ("#7c4dff", "#b79bff"), "auto": ("#0b8f8a", "#2bb5af"),
}


def app_palette():
    """The application's palette. (A widget's own palette is changed by the style sheet: a transparent background makes its Base brush transparent.)"""
    from PySide6.QtWidgets import QApplication

    return QApplication.palette()


def is_dark(_widget=None) -> bool:
    pal = app_palette()
    return pal.color(pal.ColorRole.Window).lightness() < 128


def lane_color(lane: str, dark: bool = False) -> QColor:
    light, night = LANE_COLORS[lane]
    return QColor(night if dark else light)


def brand(dark: bool = False) -> QColor:
    return QColor("#4c68f0" if dark else "#3b5bdb")


def tint(color: QColor, alpha: float) -> str:
    return f"rgba({color.red()}, {color.green()}, {color.blue()}, {alpha})"


class Switch(QAbstractButton):
    """An on/off switch (38 x 22, a white knob), as the Automatic switch on the web."""

    def __init__(self, parent: QWidget | None = None, lane: str = "auto"):
        super().__init__(parent)
        self.setCheckable(True)
        self._lane = lane
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        self.setFixedSize(38, 22)
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)

    def sizeHint(self) -> QSize:  # noqa: N802
        return QSize(38, 22)

    def paintEvent(self, _event) -> None:  # noqa: N802
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        dark = is_dark(self)
        muted = QColor(app_palette().windowText().color())
        muted.setAlphaF(0.30)
        track = lane_color(self._lane, dark) if self.isChecked() else muted
        if not self.isEnabled():
            track.setAlphaF(0.4)
        p.setPen(Qt.PenStyle.NoPen)
        p.setBrush(track)
        p.drawRoundedRect(QRectF(0, 0, 38, 22), 11, 11)
        p.setBrush(QColor("white"))
        x = 19 if self.isChecked() else 3
        p.drawEllipse(QRectF(x, 3, 16, 16))
        if self.hasFocus():
            p.setBrush(Qt.BrushStyle.NoBrush)
            p.setPen(brand(dark))
            p.drawRoundedRect(QRectF(0.5, 0.5, 37, 21), 10.5, 10.5)
        p.end()


class SoftSegmented(Segmented):
    """Mutually exclusive buttons in a soft tray; the one that is on is lifted and underlined in the colour of its row."""

    def __init__(self, options: list[tuple[str, str]], lane: str = "voice", parent: QWidget | None = None):
        super().__init__(options, parent)
        self.setObjectName("SoftSeg")
        self.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, True)
        lay = self.layout()
        lay.setContentsMargins(3, 3, 3, 3)
        lay.setSpacing(2)
        for b in self._buttons.values():
            b.setProperty("soft", True)
            b.setProperty("lane", lane)
            b.setProperty("pos", "mid")
            b.setFocusPolicy(Qt.FocusPolicy.StrongFocus)

    def set_tooltips(self, tips: dict[str, str]) -> None:
        for key, text in tips.items():
            if key in self._buttons:
                self._buttons[key].setToolTip(text)

    def set_disabled_why(self, key: str, disabled: bool, why: str = "", tip: str = "") -> None:
        b = self._buttons[key]
        b.setEnabled(not disabled)
        b.setToolTip(why if disabled and why else tip)


class Pills(QWidget):
    """A row of small pill buttons of which one is on (the subtitle file types), or none while the row is switched off."""

    changed = Signal(str)

    def __init__(self, options: list[tuple[str, str]], parent: QWidget | None = None):
        super().__init__(parent)
        lay = QHBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(6)
        self._group = QButtonGroup(self)
        self._group.setExclusive(True)
        self._buttons: dict[str, QPushButton] = {}
        for key, label in options:
            b = QPushButton(label)
            b.setObjectName("Pill")
            b.setCheckable(True)
            b.setCursor(Qt.CursorShape.PointingHandCursor)
            b.toggled.connect(lambda on, k=key: on and self.changed.emit(k))
            self._group.addButton(b)
            self._buttons[key] = b
            lay.addWidget(b)
        lay.addStretch(1)

    def value(self) -> str:
        return next((k for k, b in self._buttons.items() if b.isChecked()), "")

    def set_value(self, key: str) -> None:
        was = self.blockSignals(True)
        self._group.setExclusive(False)  # (an exclusive group cannot be left with none on)
        for k, b in self._buttons.items():
            b.setChecked(k == key)
        self._group.setExclusive(True)
        self.blockSignals(was)

    def set_pill(self, key: str, *, visible: bool = True, enabled: bool = True, tip: str = "") -> None:
        b = self._buttons[key]
        b.setVisible(visible)
        b.setEnabled(enabled)
        b.setToolTip(tip)

    def set_all_enabled(self, on: bool) -> None:
        for b in self._buttons.values():
            b.setEnabled(on)


class Pane(QFrame):
    """A white pane with a rounded border (the editor, the output, the settings)."""

    def __init__(self, parent: QWidget | None = None, *, name: str = "Pane"):
        super().__init__(parent)
        self.setObjectName(name)


class Card(QFrame):
    """One of the settings cards: a coloured edge on the left, a coloured square and the title (Voice, Leveling, Subtitles)."""

    def __init__(self, lane: str, title: str, parent: QWidget | None = None):
        super().__init__(parent)
        self.setObjectName("Ctl")
        self.setProperty("lane", lane)
        self.lane = lane
        self.body = QVBoxLayout(self)
        self.body.setContentsMargins(12, 8, 10, 10)
        self.body.setSpacing(8)
        head = QHBoxLayout()
        head.setSpacing(8)
        self._mark = QLabel()
        self._mark.setFixedSize(10, 10)
        self._mark.setObjectName("Mark")
        self._mark.setProperty("lane", lane)
        self.title = QLabel(title)
        self.title.setObjectName("CardHead")
        self.tag = QLabel("Automatic")
        self.tag.setObjectName("AutoTag")
        self.tag.hide()
        self.mixed = QLabel("Mixed")
        self.mixed.setObjectName("MixedTag")
        self.mixed.hide()
        head.addWidget(self._mark)
        head.addWidget(self.title)
        head.addWidget(self.tag)
        head.addWidget(self.mixed)
        head.addStretch(1)
        self.head = head
        self.body.addLayout(head)

    def set_auto(self, on: bool) -> None:
        self.tag.setVisible(on)

    def set_mixed(self, on: bool) -> None:
        self.mixed.setVisible(on)


def small_label(text: str = "", *, wrap: bool = True, muted: bool = True) -> QLabel:
    label = QLabel(text)
    label.setObjectName("Small" if muted else "Plain")
    label.setWordWrap(wrap)
    return label


def link_button(text: str, tip: str = "") -> QPushButton:
    b = QPushButton(text)
    b.setObjectName("Link")
    b.setCursor(Qt.CursorShape.PointingHandCursor)
    b.setFlat(True)
    if tip:
        b.setToolTip(tip)
    return b


def spacer_policy(widget: QWidget, h: QSizePolicy.Policy, v: QSizePolicy.Policy) -> None:
    widget.setSizePolicy(h, v)


__all__ = [
    "LANE_COLORS", "Switch", "SoftSegmented", "Pills", "Pane", "Card", "brand", "lane_color", "tint", "is_dark", "small_label", "link_button",
    "set_property",
]
