"""Small vector icons drawn with QPainter, so they follow the theme and stay sharp at any scale."""
from __future__ import annotations

from collections.abc import Callable

from PySide6.QtCore import QPointF, QRectF, QSize, Qt
from PySide6.QtGui import QBrush, QColor, QIcon, QPainter, QPainterPath, QPen, QPixmap, QPolygonF

# Everything is drawn on a 24 x 24 grid.
_GRID = 24
_SCALES = (1, 2, 3)

Drawer = Callable[[QPainter, QColor], None]


def _stroke(p: QPainter, color: QColor, width: float = 2.0) -> None:
    p.setPen(QPen(color, width, Qt.PenStyle.SolidLine, Qt.PenCapStyle.RoundCap, Qt.PenJoinStyle.RoundJoin))
    p.setBrush(Qt.BrushStyle.NoBrush)


def _fill(p: QPainter, color: QColor) -> None:
    p.setPen(Qt.PenStyle.NoPen)
    p.setBrush(QBrush(color))


def _play(p: QPainter, c: QColor) -> None:
    _fill(p, c)
    p.drawPolygon(QPolygonF([QPointF(7, 4.5), QPointF(19.5, 12), QPointF(7, 19.5)]))


def _pause(p: QPainter, c: QColor) -> None:
    _fill(p, c)
    p.drawRoundedRect(QRectF(6, 5, 4, 14), 1.2, 1.2)
    p.drawRoundedRect(QRectF(14, 5, 4, 14), 1.2, 1.2)


def _folder(p: QPainter, c: QColor) -> None:
    _stroke(p, c, 1.8)
    path = QPainterPath()
    path.moveTo(3.5, 7)
    path.quadTo(3.5, 5.5, 5, 5.5)
    path.lineTo(9.2, 5.5)
    path.lineTo(11.4, 8)
    path.lineTo(19, 8)
    path.quadTo(20.5, 8, 20.5, 9.5)
    path.lineTo(20.5, 17.5)
    path.quadTo(20.5, 19, 19, 19)
    path.lineTo(5, 19)
    path.quadTo(3.5, 19, 3.5, 17.5)
    path.closeSubpath()
    p.drawPath(path)


def _close(p: QPainter, c: QColor) -> None:
    _stroke(p, c, 2.0)
    p.drawLine(QPointF(7, 7), QPointF(17, 17))
    p.drawLine(QPointF(17, 7), QPointF(7, 17))


def _plus(p: QPainter, c: QColor) -> None:
    _stroke(p, c, 2.2)
    p.drawLine(QPointF(12, 5.5), QPointF(12, 18.5))
    p.drawLine(QPointF(5.5, 12), QPointF(18.5, 12))


def _upload(p: QPainter, c: QColor) -> None:
    _stroke(p, c, 1.8)
    p.drawLine(QPointF(12, 15.5), QPointF(12, 4.5))
    p.drawPolyline(QPolygonF([QPointF(7.5, 9), QPointF(12, 4.5), QPointF(16.5, 9)]))
    p.drawPolyline(QPolygonF([QPointF(4.5, 14.5), QPointF(4.5, 19.5), QPointF(19.5, 19.5), QPointF(19.5, 14.5)]))


def _waveform(p: QPainter, c: QColor) -> None:
    _fill(p, c)
    heights = (5, 10, 16, 8, 20, 12, 6, 14, 4)
    for i, h in enumerate(heights):
        p.drawRoundedRect(QRectF(1.2 + i * 2.4, 12 - h / 2, 1.4, h), 0.7, 0.7)


def _info(p: QPainter, c: QColor) -> None:
    _stroke(p, c, 1.7)
    p.drawEllipse(QRectF(3.5, 3.5, 17, 17))
    _fill(p, c)
    p.drawEllipse(QRectF(11, 7.2, 2, 2))
    p.drawRoundedRect(QRectF(11, 10.8, 2, 6.2), 1, 1)


def _up(p: QPainter, c: QColor) -> None:
    _stroke(p, c, 2.0)
    p.drawPolyline(QPolygonF([QPointF(6.5, 14.5), QPointF(12, 9), QPointF(17.5, 14.5)]))


def _down(p: QPainter, c: QColor) -> None:
    _stroke(p, c, 2.0)
    p.drawPolyline(QPolygonF([QPointF(6.5, 9.5), QPointF(12, 15), QPointF(17.5, 9.5)]))


def _trash(p: QPainter, c: QColor) -> None:
    _stroke(p, c, 1.8)
    p.drawLine(QPointF(4.5, 7), QPointF(19.5, 7))
    p.drawLine(QPointF(9.5, 7), QPointF(9.5, 4.8))
    p.drawLine(QPointF(9.5, 4.8), QPointF(14.5, 4.8))
    p.drawLine(QPointF(14.5, 4.8), QPointF(14.5, 7))
    p.drawPolyline(QPolygonF([QPointF(6.5, 7), QPointF(7.5, 19.5), QPointF(16.5, 19.5), QPointF(17.5, 7)]))


_DRAWERS: dict[str, Drawer] = {
    "play": _play, "pause": _pause, "folder": _folder, "close": _close, "plus": _plus,
    "upload": _upload, "waveform": _waveform, "info": _info, "up": _up, "down": _down, "trash": _trash,
}


def pixmap(name: str, color: QColor, size: int, dpr: float = 1.0) -> QPixmap:
    pm = QPixmap(QSize(round(size * dpr), round(size * dpr)))
    pm.setDevicePixelRatio(dpr)
    pm.fill(Qt.GlobalColor.transparent)
    p = QPainter(pm)
    p.setRenderHint(QPainter.RenderHint.Antialiasing)
    p.scale(size / _GRID, size / _GRID)
    _DRAWERS[name](p, color)
    p.end()
    return pm


def icon(name: str, color: QColor, size: int = 20) -> QIcon:
    result = QIcon()
    for scale in _SCALES:
        result.addPixmap(pixmap(name, color, size, scale))
    return result
