"""Readable tooltips. Qt draws a plain-text tooltip on one line however long it is, and its rich-text
tooltips wrap at an arbitrary narrow width. Every tooltip in the app is shown by TipPopup instead: wrapped
at a comfortable width, keeping the structure of the text. Tooltip texts stay plain text: a blank line
starts a new paragraph, a line starting with "\u2022 " is a list item (wrapped lines stay indented under it).
The "?" buttons show the same popup on click and keep it up until the next click or key."""
from __future__ import annotations

import html

from PySide6.QtCore import QEvent, QObject, QPoint, QRect, Qt, QTimer
from PySide6.QtGui import QCursor, QGuiApplication
from PySide6.QtWidgets import QApplication, QLabel, QWidget

MAX_WIDTH = 360  # px of text; about 55 characters a line
_MARGIN_H, _MARGIN_V = 11, 8
_BULLET = "\u2022 "


def to_html(text: str) -> str:
    """Plain tooltip text as simple rich text: paragraphs, and list items with a hanging indent."""
    blocks = []
    for para in text.strip().split("\n\n"):
        lines = para.split("\n")
        parts, plain = [], []

        def flush() -> None:
            if plain:
                parts.append(f'<p style="margin:0">{"<br>".join(html.escape(x) for x in plain)}</p>')
                plain.clear()

        for line in lines:
            if line.startswith(_BULLET):
                flush()
                parts.append(
                    '<table cellspacing="0" cellpadding="0"><tr><td width="14" valign="top">\u2022</td>'
                    f"<td>{html.escape(line[len(_BULLET):])}</td></tr></table>"
                )
            else:
                plain.append(line)
        flush()
        blocks.append("".join(parts))
    return '<p style="margin:0; font-size:5pt">&nbsp;</p>'.join(blocks)  # a small gap between paragraphs


class TipPopup(QLabel):
    def __init__(self) -> None:
        super().__init__(None, Qt.WindowType.ToolTip | Qt.WindowType.FramelessWindowHint)
        self.setObjectName("Tip")
        self.setTextFormat(Qt.TextFormat.RichText)
        self.setWordWrap(True)
        self.setAttribute(Qt.WidgetAttribute.WA_ShowWithoutActivating)
        self.setContentsMargins(_MARGIN_H, _MARGIN_V, _MARGIN_H, _MARGIN_V)
        self.owner: QWidget | None = None
        self.pinned = False
        self._timer = QTimer(self, singleShot=True, timeout=self.dismiss)

    def show_text(self, text: str, at: QPoint, owner: QWidget | None, pinned: bool = False) -> None:
        self.owner, self.pinned = owner, pinned
        self.setText(to_html(text))
        fm = self.fontMetrics()
        natural = max((fm.horizontalAdvance(line) for line in text.splitlines()), default=0) + 2
        width = min(natural, MAX_WIDTH) + 2 * _MARGIN_H
        self.setFixedWidth(width)
        self.setFixedHeight(self.heightForWidth(width))
        self.move(self._place(at))
        self.show()
        self.raise_()
        # Long enough to read: Qt's own default is 10 s; longer texts and pinned help stay up longer.
        self._timer.start(60_000 if pinned else max(10_000, 60 * len(text)))

    def dismiss(self) -> None:
        self._timer.stop()
        self.owner, self.pinned = None, False
        self.hide()

    def _place(self, at: QPoint) -> QPoint:
        screen = QGuiApplication.screenAt(at) or QGuiApplication.primaryScreen()
        area: QRect = screen.availableGeometry()
        x, y = at.x() + 12, at.y() + 20
        if x + self.width() > area.right():
            x = max(area.left(), area.right() - self.width())
        if y + self.height() > area.bottom():
            y = max(area.top(), at.y() - self.height() - 8)
        return QPoint(x, y)


class _TipFilter(QObject):
    def __init__(self, popup: TipPopup, parent: QObject):
        super().__init__(parent)
        self.popup = popup

    def eventFilter(self, obj: QObject, event: QEvent) -> bool:  # noqa: N802
        try:
            return self._filter(obj, event)
        except RuntimeError:  # the popup is already gone (application shutting down)
            return False

    def _filter(self, obj: QObject, event: QEvent) -> bool:
        kind = event.type()
        popup = self.popup
        if kind == QEvent.Type.ToolTip and isinstance(obj, QWidget):
            text = obj.toolTip()
            if not text:
                return False  # let Qt try the parent widget
            if not (popup.pinned and popup.isVisible()):
                popup.show_text(text, event.globalPos(), obj)
            return True
        if popup.isVisible():
            if kind in (QEvent.Type.MouseButtonPress, QEvent.Type.KeyPress, QEvent.Type.Wheel,
                        QEvent.Type.ApplicationDeactivate):
                if not (kind == QEvent.Type.MouseButtonPress and obj is popup.owner and popup.pinned):
                    popup.dismiss()
            elif kind in (QEvent.Type.Leave, QEvent.Type.Hide) and obj is popup.owner and not popup.pinned:
                popup.dismiss()
        return False


_popup: TipPopup | None = None


def install(app: QApplication) -> TipPopup:
    """Route every tooltip in `app` through the wrapped popup."""
    global _popup
    if _popup is None:
        _popup = TipPopup()
        app.installEventFilter(_TipFilter(_popup, app))
    return _popup


def is_showing() -> bool:
    """True while a help text or tooltip is on screen (Esc closes that, and nothing else)."""
    return _popup is not None and _popup.isVisible()


def pin(text: str, owner: QWidget) -> None:
    """Show `text` next to `owner` until the next click or key (the "?" buttons)."""
    popup = _popup or install(QApplication.instance())
    if popup.isVisible() and popup.owner is owner and popup.pinned:
        popup.dismiss()
        return
    at = owner.mapToGlobal(QPoint(0, owner.height())) if owner.isVisible() else QCursor.pos()
    popup.show_text(text, at, owner, pinned=True)
