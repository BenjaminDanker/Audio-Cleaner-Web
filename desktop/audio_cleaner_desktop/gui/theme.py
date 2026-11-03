"""Colours and the application style sheet.

The base look is Qt's Fusion style in the system palette; the style sheet only adds
the card layout and the accent colour, and is rebuilt
when the system switches between light and dark.
"""
from __future__ import annotations

from dataclasses import dataclass

from PySide6.QtCore import QEvent, QObject
from PySide6.QtGui import QColor, QPalette
from PySide6.QtWidgets import QApplication, QWidget


@dataclass(frozen=True)
class Theme:
    dark: bool
    window: QColor
    surface: QColor
    text: QColor
    muted: QColor
    border: QColor
    hover: QColor
    brand: str
    brand_hover: str
    brand_soft: str
    success: str
    danger: str
    danger_soft: str
    warn: str
    warn_soft: str
    base_pt: float


def mix(a: QColor, b: QColor, t: float) -> QColor:
    """`a` blended towards `b` by `t` (0..1)."""
    return QColor(
        round(a.red() + (b.red() - a.red()) * t),
        round(a.green() + (b.green() - a.green()) * t),
        round(a.blue() + (b.blue() - a.blue()) * t),
    )


def current_theme(palette: QPalette, base_pt: float) -> Theme:
    window = palette.color(QPalette.ColorRole.Window)
    base = palette.color(QPalette.ColorRole.Base)
    text = palette.color(QPalette.ColorRole.WindowText)
    dark = window.lightness() < 128
    return Theme(
        dark=dark,
        window=window,
        surface=base,
        text=text,
        muted=mix(text, window, 0.38),
        border=mix(window, text, 0.16),
        hover=mix(base, text, 0.07),
        brand="#4c68f0" if dark else "#3b5bdb",
        brand_hover="#6279f5" if dark else "#2f49b8",
        brand_soft="#1e2850" if dark else "#e6ebff",
        success="#4fd29a" if dark else "#12794f",
        danger="#ff7b83" if dark else "#b9242b",
        danger_soft="#431a1d" if dark else "#fde5e6",
        warn="#f0b84a" if dark else "#8a5a00",
        warn_soft="#3a2d0e" if dark else "#fff1cf",
        base_pt=base_pt,
    )


def stylesheet(t: Theme) -> str:
    return f"""
QMainWindow, QDialog {{ background: {t.window.name()}; }}
QLabel {{ background: transparent; }}
QToolTip {{ padding: 4px 6px; }}
QLabel#Tip {{ background: {t.surface.name()}; color: {t.text.name()}; border: 1px solid {t.border.name()}; border-radius: 6px; }}

QFrame#Card {{ background: {t.surface.name()}; border: 1px solid {t.border.name()}; border-radius: 12px; }}
QFrame#Card[drop="true"] {{ border: 2px dashed {t.brand}; background: {t.brand_soft}; }}
QFrame#DropZone {{ background: transparent; border: 2px dashed {t.border.name()}; border-radius: 10px; }}
QFrame#DropZone[drop="true"] {{ border-color: {t.brand}; }}

QLabel#CardTitle {{ font-weight: 600; font-size: {t.base_pt * 1.05:.1f}pt; }}
QLabel#Heading {{ font-weight: 600; font-size: {t.base_pt * 1.35:.1f}pt; }}
QLabel#Muted, QLabel[role="muted"] {{ color: {t.muted.name()}; }}
QLabel#Small {{ color: {t.muted.name()}; font-size: {t.base_pt * 0.9:.1f}pt; }}
QLabel[state="done"] {{ color: {t.success}; font-weight: 600; }}
QLabel[state="failed"] {{ color: {t.danger}; font-weight: 600; }}
QLabel[state="cleaning"] {{ color: {t.brand}; font-weight: 600; }}
QLabel[state="warn"] {{ color: {t.warn}; }}
QLabel#Badge {{
    background: {t.brand_soft}; color: {t.brand}; border-radius: 6px;
    font-weight: 700; font-size: {t.base_pt * 0.78:.1f}pt; padding: 0 2px;
}}
QLabel#Badge[kind="video"] {{ background: {t.brand}; color: white; }}

QFrame#Sugg {{ background: {t.surface.name()}; border: 1px solid {t.border.name()}; border-radius: 10px; }}
QFrame#Sugg[focus="true"] {{ border: 2px solid #7c4dff; background: {"#241a3f" if t.dark else "#f3effe"}; }}

QFrame#Row {{ background: transparent; border-radius: 10px; }}
QFrame#Row:hover {{ background: {t.hover.name()}; }}
QWidget#RowHolder {{ background: transparent; }}
QLabel#Small[state="failed"] {{ font-weight: 400; }}

QListWidget {{ background: transparent; border: none; outline: none; }}
QListWidget::item {{ border-radius: 8px; margin: 1px 0; }}
QListWidget::item:hover {{ background: {t.hover.name()}; }}
QListWidget::item:selected {{ background: {t.brand_soft}; }}

QPushButton {{
    background: {t.surface.name()}; color: {t.text.name()};
    border: 1px solid {t.border.name()}; border-radius: 8px; padding: 6px 14px;
}}
QPushButton:hover {{ background: {t.hover.name()}; }}
QPushButton:pressed {{ background: {t.border.name()}; }}
QPushButton:focus {{ border: 2px solid {t.brand}; padding: 5px 13px; }}
QPushButton:disabled {{ color: {t.muted.name()}; background: transparent; }}

QPushButton#Primary {{ background: {t.brand}; color: white; border: 1px solid {t.brand}; font-weight: 600; padding: 8px 22px; }}
QPushButton#Primary:hover {{ background: {t.brand_hover}; border-color: {t.brand_hover}; }}
QPushButton#Primary:focus {{ border: 2px solid {t.text.name()}; padding: 7px 21px; }}
QPushButton#Primary:disabled {{ background: {t.border.name()}; border-color: {t.border.name()}; color: {t.muted.name()}; }}
QPushButton#Primary[mode="cancel"] {{ background: {t.danger}; border-color: {t.danger}; color: {"#1a0b0c" if t.dark else "white"}; }}

QPushButton#Flat, QToolButton {{ background: transparent; border: none; border-radius: 6px; padding: 5px 8px; }}
QPushButton#Flat:hover, QToolButton:hover {{ background: {t.hover.name()}; }}
QPushButton#Flat[on="true"] {{ background: {t.brand_soft}; color: {t.brand}; font-weight: 600; }}
QPushButton#Flat:focus, QToolButton:focus {{ background: {t.hover.name()}; border: 1px solid {t.brand}; }}
QToolButton {{ padding: 4px; }}
QPushButton#Link {{ background: transparent; border: none; color: {t.brand}; padding: 2px 4px; text-decoration: underline; }}
QPushButton#Link:hover {{ color: {t.brand_hover}; background: transparent; }}
QPushButton#Link:focus {{ border: 1px solid {t.brand}; padding: 1px 3px; }}

QPushButton#Segment {{
    background: {t.surface.name()}; border: 1px solid {t.border.name()}; border-radius: 0; padding: 6px 6px; margin-left: -1px;
}}
QPushButton#Segment[pos="first"] {{ border-top-left-radius: 8px; border-bottom-left-radius: 8px; margin-left: 0; }}
QPushButton#Segment[pos="last"] {{ border-top-right-radius: 8px; border-bottom-right-radius: 8px; }}
QPushButton#Segment:checked {{ background: {t.brand}; border-color: {t.brand}; color: white; }}
QPushButton#Segment:checked:hover {{ background: {t.brand_hover}; }}
QPushButton#Segment:focus {{ border: 2px solid {t.text.name()}; padding: 5px 5px; }}
QPushButton#Segment:disabled {{ color: {t.muted.name()}; background: transparent; }}
QPushButton#Segment:checked:disabled {{ background: {t.border.name()}; border-color: {t.border.name()}; }}

QLineEdit {{
    background: {t.surface.name()}; color: {t.text.name()}; border: 1px solid {t.border.name()}; border-radius: 8px;
    padding: 6px 10px; selection-background-color: {t.brand}; selection-color: white;
}}
QLineEdit:focus {{ border: 2px solid {t.brand}; padding: 5px 9px; }}

QLineEdit#NameBox {{ border-top-right-radius: 0px; border-bottom-right-radius: 0px; padding: 3px 8px; }}
QLineEdit#NameBox:focus {{ padding: 2px 7px; }}
QLineEdit#NameBox[invalid="true"] {{ border-color: {t.danger}; }}
QLineEdit#NameBox:disabled {{ color: {t.muted.name()}; background: transparent; }}
QComboBox#FormatBox {{ border-top-left-radius: 0px; border-bottom-left-radius: 0px; border-left: 0px; padding: 3px 8px; min-height: 1.2em; }}
QComboBox#FormatBox:focus {{ padding: 2px 7px; }}

QComboBox {{
    background: {t.surface.name()}; border: 1px solid {t.border.name()}; border-radius: 8px; padding: 5px 10px; min-height: 1.3em;
}}
QComboBox:hover {{ background: {t.hover.name()}; }}
QComboBox:focus {{ border: 2px solid {t.brand}; padding: 4px 9px; }}
QComboBox:disabled {{ color: {t.muted.name()}; }}
QComboBox::drop-down {{ border: none; width: 22px; }}
QComboBox QAbstractItemView {{ background: {t.surface.name()}; border: 1px solid {t.border.name()}; selection-background-color: {t.brand}; selection-color: white; outline: none; }}

QProgressBar {{ background: {t.border.name()}; border: none; border-radius: 3px; max-height: 6px; min-height: 6px; text-align: center; }}
QProgressBar::chunk {{ background: {t.brand}; border-radius: 3px; }}

QSlider::groove:horizontal {{ background: {t.border.name()}; height: 4px; border-radius: 2px; }}
QSlider::sub-page:horizontal {{ background: {t.brand}; border-radius: 2px; }}
QSlider::handle:horizontal {{ background: {t.brand}; width: 14px; height: 14px; margin: -5px 0; border-radius: 7px; }}
QSlider::handle:horizontal:hover {{ background: {t.brand_hover}; }}
QSlider::handle:horizontal:focus {{ border: 2px solid {t.text.name()}; width: 10px; height: 10px; }}
QSlider::handle:horizontal:disabled {{ background: {t.muted.name()}; }}
QSlider::sub-page:horizontal:disabled {{ background: {t.muted.name()}; }}

QFrame#Banner {{ border-radius: 10px; border: 1px solid {t.brand}; background: {t.brand_soft}; }}
QFrame#Banner[kind="error"] {{ border-color: {t.danger}; background: {t.danger_soft}; }}
QFrame#Banner[kind="warn"] {{ border-color: {t.warn}; background: {t.warn_soft}; }}

QFrame#Preview {{ background: {"#05070b" if t.dark else "#10131a"}; border-radius: 8px; border: none; }}
QFrame#Preview QLabel {{ color: #aab3c5; }}

QTabWidget::pane {{ border: none; border-top: 1px solid {t.border.name()}; }}
QTabBar {{ background: transparent; }}
QTabBar::tab {{ background: transparent; color: {t.muted.name()}; padding: 8px 18px; margin-right: 6px; border: none; border-bottom: 2px solid transparent; }}
QTabBar::tab:hover {{ color: {t.text.name()}; }}
QTabBar::tab:selected {{ color: {t.text.name()}; font-weight: 600; border-bottom: 2px solid {t.brand}; }}
QTabBar::tab:focus {{ background: {t.hover.name()}; }}
QScrollArea, QScrollArea > QWidget > QWidget {{ background: transparent; }}
QPlainTextEdit, QListWidget#Licenses {{ background: {t.surface.name()}; border: 1px solid {t.border.name()}; border-radius: 8px; }}
""" + studio_stylesheet(t)


def _rgba(hex_color: str, alpha: float) -> str:
    c = QColor(hex_color)
    return f"rgba({c.red()}, {c.green()}, {c.blue()}, {alpha})"


_LANES = {  # (light, dark): the colour of each row of the editor screen, as on the web
    "voice": ("#3b5bdb", "#4c68f0"), "rebuild": ("#0b8f8a", "#2bb5af"), "echo": ("#d9480f", "#ff922b"), "level": ("#b87500", "#f0b84a"),
    "subs": ("#c2255c", "#f06595"), "sugg": ("#7c4dff", "#b79bff"),
}


def studio_stylesheet(t: Theme) -> str:
    """The editor screen: panes, the settings cards, the soft segmented buttons, pills, the jobs column (the web's studio.css)."""
    lane = {k: v[1 if t.dark else 0] for k, v in _LANES.items()}
    surface, border, muted, text, hover = t.surface.name(), t.border.name(), t.muted.name(), t.text.name(), t.hover.name()
    rules = [f"""
QWidget#Page, QWidget#Studio, QWidget#InspBody {{ background: transparent; }}
QFrame#Pane {{ background: {surface}; border: 1px solid {border}; border-radius: 14px; }}
QFrame#Poster {{ background: {hover}; border: 1px solid {border}; border-radius: 9px; }}
QFrame#Poster[video="true"] {{ background: #0b0f19; border: none; }}
QFrame#Poster QLabel#PosterText {{ color: {muted}; font-size: {t.base_pt * 1.1:.1f}pt; background: transparent; }}
QFrame#Picture {{ background: transparent; border: none; }}
QLabel#TimeBadge {{ background: rgba(0, 0, 0, 160); color: white; border-radius: 6px; padding: 1px 7px; font-size: {t.base_pt * 0.82:.1f}pt; }}
QLabel#FileName {{ font-weight: 700; font-size: {t.base_pt * 1.4:.1f}pt; }}
QLabel#AutoLabel {{ font-weight: 650; font-size: {t.base_pt * 1.05:.1f}pt; }}
QLabel#TimeText {{ color: {muted}; }}
QLabel#FieldLabel {{ font-weight: 700; }}
QLabel#LaneName {{ color: {muted}; font-weight: 700; }}
QLabel#Plain {{ color: {text}; }}
QLabel#InspHead {{ font-weight: 700; font-size: {t.base_pt * 1.2:.1f}pt; }}
QLabel#ColHead {{ font-weight: 700; font-size: {t.base_pt * 1.3:.1f}pt; }}
QLabel#DlgH3 {{ font-weight: 700; font-size: {t.base_pt * 1.15:.1f}pt; margin-top: 8px; }}
QLabel#DlgH4 {{ font-weight: 650; }}
QFrame#GhostBar {{ background: {hover}; border: 1px solid {border}; border-radius: 7px; }}
QFrame#DropArea {{ background: transparent; border: 2px dashed {border}; border-radius: 14px; }}
QFrame#DropArea:hover, QFrame#DropArea[over="true"] {{ border-color: {t.brand}; background: {_rgba(t.brand, 0.06)}; }}
QToolButton#InfoDot {{ border: 1.5px solid {t.brand}; border-radius: 13px; color: {t.brand}; font-weight: 800; background: transparent; padding: 0; }}
QToolButton#InfoDot:hover {{ background: {t.brand_soft}; }}
QToolButton#MenuDots {{ font-size: {t.base_pt * 1.4:.1f}pt; font-weight: 700; padding: 0 8px; }}
QToolButton#MenuDots::menu-indicator {{ image: none; }}

QPushButton#Tool {{ background: {surface}; border: 1px solid {border}; border-radius: 9px; padding: 6px 10px; font-weight: 650; }}
QPushButton#Tool:hover {{ background: {hover}; }}
QPushButton#Tool:checked {{ background: {t.brand_soft}; border-color: {t.brand}; }}
QPushButton#Tool:disabled {{ color: {muted}; background: transparent; }}
QPushButton#PlayButton {{ background: {t.brand}; border: 1px solid {t.brand}; border-radius: 9px; }}
QPushButton#PlayButton:hover {{ background: {t.brand_hover}; }}
QPushButton#PlayButton:disabled {{ background: {border}; border-color: {border}; }}
QPushButton#Chip {{ background: {_rgba(lane['sugg'], 0.12)}; border: 1px solid {_rgba(lane['sugg'], 0.5)}; border-radius: 12px; padding: 2px 12px; font-weight: 700; }}
QPushButton#DangerOutline {{ background: transparent; border: 1px solid {t.danger}; color: {t.danger}; }}
QPushButton#DangerOutline:hover {{ background: {t.danger_soft}; }}
QPushButton#Pill {{ border: 1px solid {border}; border-radius: 12px; padding: 3px 12px; font-weight: 700; color: {muted}; background: transparent; }}
QPushButton#Pill:checked {{ border-color: {lane['subs']}; color: {text}; background: {_rgba(lane['subs'], 0.14)}; }}
QPushButton#Pill:disabled {{ color: {muted}; border-style: dashed; }}
QPushButton#Primary[state="processing"] {{ background: {t.brand}; }}

QFrame#Ctl {{ border: 1px solid {border}; border-radius: 12px; }}
QFrame#SgCard {{ background: {_rgba(lane['sugg'], 0.10)}; border: 1px solid {_rgba(lane['sugg'], 0.45)}; border-radius: 10px; }}
QLabel#Mark {{ border-radius: 3px; }}
QLabel#CardHead {{ font-weight: 700; font-size: {t.base_pt * 1.0:.1f}pt; }}
QLabel#AutoTag {{ color: {muted}; border: 1px solid {border}; border-radius: 9px; padding: 0 7px; font-size: {t.base_pt * 0.78:.1f}pt; font-weight: 600; }}
QLabel#MixedTag {{ color: {t.warn}; background: {t.warn_soft}; border-radius: 9px; padding: 0 7px; font-size: {t.base_pt * 0.78:.1f}pt; font-weight: 700; }}
QFrame#VOpt {{ background: {surface}; border: 1px solid {border}; border-radius: 10px; }}
QRadioButton#VRadio {{ font-weight: 700; background: transparent; spacing: 8px; }}
QRadioButton#VRadio::indicator {{ width: 12px; height: 12px; border-radius: 8px; border: 2px solid {muted}; background: transparent; }}
QRadioButton#VRadio::indicator:checked {{ border: 2px solid {t.brand}; background: qradialgradient(cx:0.5, cy:0.5, radius:0.5, fx:0.5, fy:0.5, stop:0 {t.brand}, stop:0.45 {t.brand}, stop:0.5 transparent, stop:1 transparent); }}
QWidget#SoftSeg {{ background: {hover}; border: 1px solid {border}; border-radius: 10px; }}
QPushButton#Segment[soft="true"] {{ background: transparent; border: none; border-radius: 7px; color: {muted}; font-weight: 650; padding: 6px 4px; margin: 0; }}
QPushButton#Segment[soft="true"]:hover {{ color: {text}; background: transparent; }}
QPushButton#Segment[soft="true"]:checked {{ background: {surface}; color: {text}; border-bottom: 3px solid {t.brand}; }}
QPushButton#Segment[soft="true"]:disabled {{ color: {muted}; }}
QPushButton#Segment[soft="true"]:focus {{ border: 1px solid {t.brand}; padding: 5px 3px; }}

QFrame#Popover {{ background: {surface}; border: 1px solid {border}; border-radius: 12px; }}
QFrame#JobCard {{ background: {surface}; border: 1px solid {border}; border-radius: 12px; }}
QFrame#JobCard:hover {{ border-color: {_rgba(t.brand, 0.6)}; }}
QFrame#JobCard[current="true"] {{ border: 2px solid {t.brand}; }}
QLabel#JobName {{ font-weight: 650; }}
QFrame#JobsCol {{ background: transparent; }}
"""]
    voice_edge = lane
    for key, color in lane.items():
        if key in ("voice", "echo", "level", "subs"):
            tint = 0.10 if key == "voice" else 0.05
            rules.append(f"""
QFrame#Ctl[lane="{key}"] {{ border-left: 4px solid {color}; background: {_rgba(color, tint)}; }}
QLabel#Mark[lane="{key}"] {{ background: {color}; }}
QPushButton#Segment[soft="true"][lane="{key}"]:checked {{ border-bottom: 3px solid {color}; }}
QFrame#VOpt[mode="noise"][on="true"] {{ border-color: {lane['voice']}; background: {_rgba(lane['voice'], 0.08)}; }}
QFrame#VOpt[mode="rebuild"][on="true"] {{ border-color: {lane['rebuild']}; background: {_rgba(lane['rebuild'], 0.08)}; }}
QFrame#VOpt[mode="off"][on="true"] {{ border-color: {muted}; }}
""")
    del voice_edge
    return "\n".join(rules)


def set_property(widget: QWidget, name: str, value: object) -> None:
    """Change a dynamic property that style sheet selectors depend on and restyle."""
    if widget.property(name) == value:
        return
    widget.setProperty(name, value)
    style = widget.style()
    style.unpolish(widget)
    style.polish(widget)
    widget.update()


class ThemeWatcher(QObject):
    """Rebuilds the style sheet when the application palette flips between light and dark."""

    def __init__(self, app: QApplication):
        super().__init__(app)
        self._app = app
        self._applied: Theme | None = None
        app.installEventFilter(self)
        self.apply()

    @property
    def theme(self) -> Theme:
        assert self._applied is not None
        return self._applied

    def apply(self) -> None:
        theme = current_theme(self._app.palette(), self._app.font().pointSizeF())
        if theme == self._applied:
            return
        self._applied = theme
        self._app.setStyleSheet(stylesheet(theme))
        # Checkboxes, radio buttons and selections use the app's blue, not the system accent colour
        # (on Windows that is whatever the user picked, e.g. orange).
        palette = self._app.palette()
        for group in (QPalette.ColorGroup.Active, QPalette.ColorGroup.Inactive):
            palette.setColor(group, QPalette.ColorRole.Accent, QColor(theme.brand))
            palette.setColor(group, QPalette.ColorRole.Highlight, QColor(theme.brand))
            palette.setColor(group, QPalette.ColorRole.HighlightedText, QColor("#ffffff"))
        self._app.setPalette(palette)

    def eventFilter(self, obj: QObject, event: QEvent) -> bool:
        if event.type() in (QEvent.Type.ApplicationPaletteChange, QEvent.Type.ApplicationFontChange) and obj is self._app:
            self.apply()
        return False
