"""Small reusable widgets."""
from __future__ import annotations

from PySide6.QtCore import QEvent, QRect, QSize, Qt, Signal
from PySide6.QtGui import QKeyEvent
from PySide6.QtWidgets import (
    QButtonGroup, QCheckBox, QFrame, QHBoxLayout, QLabel, QProxyStyle, QPushButton, QSizePolicy, QSlider, QStyle,
    QStyleOptionButton, QStylePainter, QToolButton, QWidget,
)

from . import icons
from .theme import set_property


def format_time(seconds: float) -> str:
    s = max(0, int(round(seconds)))
    h, rem = divmod(s, 3600)
    m, s = divmod(rem, 60)
    return f"{h}:{m:02d}:{s:02d}" if h else f"{m}:{s:02d}"


class ElidedLabel(QLabel):
    """A single-line label that shortens its text with an ellipsis instead of growing."""

    def __init__(self, text: str = "", elide: Qt.TextElideMode = Qt.TextElideMode.ElideRight, parent: QWidget | None = None):
        super().__init__(parent)
        self._full = ""
        self._elide = elide
        self.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Preferred)
        self.setText(text)

    def setText(self, text: str) -> None:  # noqa: N802 - Qt override
        self._full = text
        self._refresh()

    def text(self) -> str:
        return self._full

    def minimumSizeHint(self) -> QSize:  # noqa: N802
        return QSize(0, super().minimumSizeHint().height())

    def resizeEvent(self, event) -> None:  # noqa: N802
        super().resizeEvent(event)
        self._refresh()

    def _refresh(self) -> None:
        shown = self.fontMetrics().elidedText(self._full, self._elide, max(self.width(), 1))
        super().setText(shown)
        self.setToolTip(self._full if shown != self._full else "")


class IconButton(QToolButton):
    """A flat button showing one of the painted icons, recoloured when the theme changes."""

    def __init__(self, icon_name: str, tooltip: str, size: int = 18, parent: QWidget | None = None):
        super().__init__(parent)
        self._icon_name = icon_name
        self._size = size
        self.setIconSize(QSize(size, size))
        self.setToolTip(tooltip)
        self.setAccessibleName(tooltip)
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        self.set_icon(icon_name)

    def set_icon(self, name: str) -> None:
        self._icon_name = name
        self.setIcon(icons.icon(name, self.palette().buttonText().color(), self._size))

    def changeEvent(self, event: QEvent) -> None:  # noqa: N802
        if event.type() == QEvent.Type.PaletteChange:
            self.set_icon(self._icon_name)
        super().changeEvent(event)


class WrapCheck(QCheckBox):
    """A check box whose text wraps onto more lines instead of widening its column. The box sits at the first line."""

    def __init__(self, text: str = "", parent: QWidget | None = None):
        super().__init__(parent)
        self._label = QLabel(text, self)
        self._label.setWordWrap(True)
        self._label.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents)
        policy = self.sizePolicy()
        policy.setHeightForWidth(True)
        self.setSizePolicy(policy)
        if text:
            self.setAccessibleName(text)

    def setText(self, text: str) -> None:  # noqa: N802 - Qt override
        self._label.setText(text)
        self.setAccessibleName(text)
        self.updateGeometry()

    def text(self) -> str:
        return self._label.text()

    def _indent(self) -> int:
        style = self.style()
        return style.pixelMetric(QStyle.PixelMetric.PM_IndicatorWidth) + style.pixelMetric(QStyle.PixelMetric.PM_CheckBoxLabelSpacing) + 2

    def hasHeightForWidth(self) -> bool:  # noqa: N802
        return True

    def heightForWidth(self, width: int) -> int:  # noqa: N802
        return max(self.style().pixelMetric(QStyle.PixelMetric.PM_IndicatorHeight), self._label.heightForWidth(max(1, width - self._indent()))) + 2

    def sizeHint(self) -> QSize:  # noqa: N802
        return QSize(self._indent() + self._label.sizeHint().width(), self.heightForWidth(self.width() or 240))

    def minimumSizeHint(self) -> QSize:  # noqa: N802
        return QSize(self._indent() + 120, self.heightForWidth(self._indent() + 120))

    def resizeEvent(self, event) -> None:  # noqa: N802
        super().resizeEvent(event)
        indent = self._indent()
        self._label.setGeometry(indent, 0, max(1, self.width() - indent), self._label.heightForWidth(max(1, self.width() - indent)))

    def changeEvent(self, event) -> None:  # noqa: N802
        super().changeEvent(event)
        if event.type() == QEvent.Type.EnabledChange:
            self._label.setEnabled(self.isEnabled())

    def paintEvent(self, event) -> None:  # noqa: N802
        option = QStyleOptionButton()
        self.initStyleOption(option)
        option.text = ""
        option.rect = QRect(0, 0, self.width(), self.style().pixelMetric(QStyle.PixelMetric.PM_IndicatorHeight) + 2)
        QStylePainter(self).drawControl(QStyle.ControlElement.CE_CheckBox, option)


class _Segment(QPushButton):
    def __init__(self, text: str, owner: "Segmented", parent: QWidget):
        super().__init__(text, parent)
        self._owner = owner
        self.setObjectName("Segment")
        self.setCheckable(True)
        self.setCursor(Qt.CursorShape.PointingHandCursor)

    def keyPressEvent(self, event: QKeyEvent) -> None:  # noqa: N802
        step = {Qt.Key.Key_Left: -1, Qt.Key.Key_Right: 1}.get(Qt.Key(event.key()))
        if step is None:
            super().keyPressEvent(event)
        else:
            self._owner.step(self, step)


class Segmented(QWidget):
    """A row of mutually exclusive buttons; arrow keys move the selection."""

    changed = Signal(str)

    def __init__(self, options: list[tuple[str, str]], parent: QWidget | None = None):
        super().__init__(parent)
        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)
        self._group = QButtonGroup(self)
        self._group.setExclusive(True)
        self._buttons: dict[str, _Segment] = {}
        for i, (key, label) in enumerate(options):
            button = _Segment(label, self, self)
            button.setProperty("pos", "first" if i == 0 else "last" if i == len(options) - 1 else "mid")
            self._group.addButton(button)
            self._buttons[key] = button
            layout.addWidget(button, 1)
            button.toggled.connect(lambda on, k=key: on and self.changed.emit(k))

    def value(self) -> str:
        return next(k for k, b in self._buttons.items() if b.isChecked())

    def set_value(self, key: str) -> None:
        self._buttons[key].setChecked(True)

    def set_enabled_key(self, key: str, enabled: bool) -> None:
        self._buttons[key].setEnabled(enabled)

    def step(self, current: _Segment, delta: int) -> None:
        enabled = [b for b in self._buttons.values() if b.isEnabled()]
        target = enabled[(enabled.index(current) + delta) % len(enabled)]
        target.setChecked(True)
        target.setFocus()


class Banner(QFrame):
    """A dismissible message strip with an optional action button."""

    action = Signal()

    def __init__(self, parent: QWidget | None = None):
        super().__init__(parent)
        self.setObjectName("Banner")
        layout = QHBoxLayout(self)
        layout.setContentsMargins(14, 8, 8, 8)
        self._label = QLabel()
        self._label.setWordWrap(True)
        self._label.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        self._button = QPushButton()
        self._button.clicked.connect(self.action)
        self._close = IconButton("close", "Dismiss")
        self._close.clicked.connect(self.hide)
        layout.addWidget(self._label, 1)
        layout.addWidget(self._button)
        layout.addWidget(self._close)
        self.hide()

    def show_message(self, text: str, kind: str = "info", action_text: str = "") -> None:
        self._label.setText(text)
        self._button.setText(action_text)
        self._button.setVisible(bool(action_text))
        set_property(self, "kind", kind)
        self.show()


class _JumpToClickStyle(QProxyStyle):
    def styleHint(self, hint, option=None, widget=None, returnData=None):  # noqa: N802
        if hint == QStyle.StyleHint.SH_Slider_AbsoluteSetButtons:
            return int(Qt.MouseButton.LeftButton.value)
        return super().styleHint(hint, option, widget, returnData)


class SeekSlider(QSlider):
    """A slider whose groove jumps to the clicked position instead of paging."""

    def __init__(self, parent: QWidget | None = None):
        super().__init__(Qt.Orientation.Horizontal, parent)
        self._jump_style = _JumpToClickStyle("Fusion")
        self.setStyle(self._jump_style)
