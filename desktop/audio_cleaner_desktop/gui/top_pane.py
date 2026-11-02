"""The top of the editor (the web's TopPane.jsx): the picture (or the poster of a sound), the file's name and facts with "Different file", the
Automatic switch with what it did and the suggestions, and the tools: play, split at the playhead, undo and redo, zoom, select all."""
from __future__ import annotations

from PySide6.QtCore import QSize, Qt, Signal
from PySide6.QtGui import QColor
from PySide6.QtWidgets import QFrame, QHBoxLayout, QLabel, QPushButton, QVBoxLayout, QWidget

from . import copy, icons
from .items import format_size
from .studio_widgets import Switch, link_button, small_label
from .widgets import ElidedLabel, Segmented

PICTURE = (176, 99)


def tool_button(icon: str, tip: str, text: str = "") -> QPushButton:
    b = QPushButton(text)
    b.setObjectName("Tool")
    b.setToolTip(tip)
    b.setAccessibleName(tip if not text else text)
    b.setCursor(Qt.CursorShape.PointingHandCursor)
    b.setProperty("icon_name", icon)
    b.setIconSize(QSize(16, 16))
    return b


class TopPane(QWidget):
    differentFile = Signal()
    autoToggled = Signal(bool)
    autoUndo = Signal()
    autoAgain = Signal()
    findMoments = Signal()
    cancelFind = Signal()
    nextSuggestion = Signal()
    play = Signal()
    split = Signal()
    undo = Signal()
    redo = Signal()
    zoomIn = Signal()
    zoomOut = Signal()
    selectAll = Signal()
    listeningChanged = Signal(str)

    def __init__(self, parent: QWidget | None = None):
        super().__init__(parent)
        self.doc = None
        self.picture_slot = QFrame()
        self.picture_slot.setObjectName("Poster")
        self.picture_slot.setFixedSize(*PICTURE)
        self.picture_lay = QVBoxLayout(self.picture_slot)
        self.picture_lay.setContentsMargins(0, 0, 0, 0)
        self.poster = QLabel("Audio")
        self.poster.setObjectName("PosterText")
        self.poster.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.picture_lay.addWidget(self.poster)
        self.time_badge = QLabel("0:00", self.picture_slot)
        self.time_badge.setObjectName("TimeBadge")
        self.time_badge.hide()
        self.preview = None

        info = QVBoxLayout()
        info.setSpacing(8)
        self.name = ElidedLabel()
        self.name.setObjectName("FileName")
        self.name.setAccessibleName("File name")
        meta = QHBoxLayout()
        meta.setSpacing(4)
        meta.setContentsMargins(0, 0, 0, 0)
        self.meta = QLabel()
        self.meta.setObjectName("Small")
        self.different = link_button("Different file", "Take this file off the editor and choose another one (a file that was not started stays in your jobs)")
        self.different.clicked.connect(self.differentFile)
        meta.addWidget(self.meta)
        meta.addWidget(self.different)
        meta.addStretch(1)
        info.addWidget(self.name)
        info.addLayout(meta)
        auto = QHBoxLayout()
        auto.setSpacing(10)
        self.switch = Switch(lane="auto")
        self.switch.setAccessibleName("Automatic")
        self.switch.clicked.connect(self.autoToggled)
        self.auto_label = QLabel("Automatic")
        self.auto_label.setObjectName("AutoLabel")
        self.chip = self._make_chip()
        auto.addWidget(self.switch)
        auto.addWidget(self.auto_label)
        auto.addWidget(self.chip)
        auto.addStretch(1)
        self.auto_box = QWidget()
        self.auto_box.setLayout(auto)
        auto.setContentsMargins(0, 0, 0, 0)
        info.addWidget(self.auto_box)
        # (a finished job has no Automatic: what can be listened to is there instead)
        self.which = Segmented([("original", "Original"), ("cleaned", "Cleaned")])
        self.which.setToolTip("Switch between the recording as it was and the cleaned version; the position stays")
        self.which.changed.connect(self.listeningChanged)
        self.hear_box = QWidget()
        hl = QHBoxLayout(self.hear_box)
        hl.setContentsMargins(0, 0, 0, 0)
        hl.setSpacing(10)
        hear_label = QLabel("Listen to")
        hear_label.setObjectName("AutoLabel")
        hl.addWidget(hear_label)
        hl.addWidget(self.which)
        self.chip2 = self._make_chip()  # (the same count, beside what can be listened to, for a job)
        hl.addWidget(self.chip2)
        hl.addStretch(1)
        self.hear_box.hide()
        info.addWidget(self.hear_box)
        self.auto_line = QWidget()
        al = QHBoxLayout(self.auto_line)
        al.setContentsMargins(0, 0, 0, 0)
        al.setSpacing(8)
        self.auto_text = small_label(wrap=True)
        self.auto_text.setObjectName("Small")
        self.auto_action = link_button("")
        self.auto_action.clicked.connect(self._auto_action)
        al.addWidget(self.auto_text, 1)
        al.addWidget(self.auto_action)
        info.addWidget(self.auto_line)
        self.why = link_button("Why these choices")
        self.why.clicked.connect(self._toggle_why)
        self.why_text = small_label()
        self.why_text.hide()
        info.addWidget(self.why, 0, Qt.AlignmentFlag.AlignLeft)
        info.addWidget(self.why_text)
        self.sg_line = QWidget()
        sl = QHBoxLayout(self.sg_line)
        sl.setContentsMargins(0, 0, 0, 0)
        sl.setSpacing(8)
        self.sg_text = small_label()
        self.sg_action = link_button("Cancel")
        self.sg_action.clicked.connect(self._sg_action)
        sl.addWidget(self.sg_text, 1)
        sl.addWidget(self.sg_action)
        info.addWidget(self.sg_line)
        info.addStretch(1)

        tools = QHBoxLayout()
        tools.setSpacing(8)
        self.play_button = QPushButton()
        self.play_button.setObjectName("PlayButton")
        self.play_button.setFixedSize(42, 34)
        self.play_button.setAccessibleName("Play")
        self.play_button.setCursor(Qt.CursorShape.PointingHandCursor)
        self.play_button.clicked.connect(self.play)
        self.time = QLabel("0:00 / 0:00")
        self.time.setObjectName("TimeText")
        self.split_button = tool_button("split", "Split at the playhead: the part around it can then be chosen as two", "Split at playhead")
        self.undo_button = tool_button("undo", "Undo")
        self.redo_button = tool_button("redo", "Redo")
        self.zoom_out_button = tool_button("zoom-out", "Zoom out")
        self.zoom_in_button = tool_button("zoom-in", "Zoom in")
        self.all_button = tool_button("select-all", "Choose the whole file", "Select all")
        self.all_button.setCheckable(True)
        for b, sig in ((self.split_button, self.split), (self.undo_button, self.undo), (self.redo_button, self.redo),
                       (self.zoom_out_button, self.zoomOut), (self.zoom_in_button, self.zoomIn), (self.all_button, self.selectAll)):
            b.clicked.connect(sig)
        tools.addWidget(self.play_button)
        tools.addWidget(self.time)
        tools.addWidget(self.split_button)
        tools.addWidget(self.undo_button)
        tools.addWidget(self.redo_button)
        tools.addWidget(self.zoom_out_button)
        tools.addWidget(self.zoom_in_button)
        tools.addWidget(self.all_button)
        tools.addStretch(1)

        grid = QVBoxLayout(self)
        grid.setContentsMargins(0, 0, 0, 0)
        grid.setSpacing(12)
        head = QHBoxLayout()
        head.setSpacing(16)
        head.addWidget(self.picture_slot, 0, Qt.AlignmentFlag.AlignTop)
        head.addLayout(info, 1)
        grid.addLayout(head)
        self.tools_box = QWidget()
        self.tools_box.setLayout(tools)
        tools.setContentsMargins(0, 0, 0, 0)
        grid.addWidget(self.tools_box)
        self._set_icons()

    def _make_chip(self) -> QPushButton:
        chip = QPushButton()
        chip.setObjectName("Chip")
        chip.setCursor(Qt.CursorShape.PointingHandCursor)
        chip.clicked.connect(self.nextSuggestion)
        chip.hide()
        return chip

    def _set_icons(self) -> None:
        color = self.palette().buttonText().color()
        for b in (self.split_button, self.undo_button, self.redo_button, self.zoom_out_button, self.zoom_in_button, self.all_button):
            b.setIcon(icons.icon(b.property("icon_name"), color, 16))
        self.set_playing(False)

    def changeEvent(self, event) -> None:  # noqa: N802
        super().changeEvent(event)
        if event.type().name == "PaletteChange" and hasattr(self, "all_button"):
            self._set_icons()

    # ------------------------------------------------------------------------------------------------------ the picture
    def attach_preview(self, preview) -> None:
        """The pane that shows the picture and plays (created by the screen when first needed)."""
        self.preview = preview
        preview.setParent(self.picture_slot)
        self.picture_lay.addWidget(preview)
        preview.hide()
        self.time_badge.raise_()

    def set_picture(self, has_video: bool, *, show_player: bool) -> None:
        self.poster.setText("Video" if has_video else "Audio")
        show = bool(has_video and show_player and self.preview is not None)
        self.poster.setVisible(not show)
        if self.preview is not None:
            self.preview.setVisible(show)
        self.picture_slot.setProperty("video", bool(has_video))
        self.time_badge.setVisible(show)
        self.time_badge.move(7, PICTURE[1] - 26)
        self.time_badge.raise_()

    # ------------------------------------------------------------------------------------------------------ what it says
    def set_doc(self, doc) -> None:
        self.doc = doc
        self.why_text.hide()
        self.refresh()

    def set_playing(self, playing: bool) -> None:
        self.play_button.setIcon(icons.icon("pause" if playing else "play", QColor("white"), 20))
        self.play_button.setAccessibleName("Pause" if playing else "Play")

    def set_time(self, position: float, total: float) -> None:
        self.time.setText(f"{copy.say_clock(position)} / {copy.say_clock(total)}")
        self.time_badge.setText(copy.say_clock(position))
        self.time_badge.adjustSize()

    def refresh(self) -> None:
        doc = self.doc
        if doc is None:
            return
        self.name.setText(doc.name)
        f = doc.facts
        if doc.error:
            self.meta.setText(doc.error + " ·")
        elif f is None:
            self.meta.setText("Reading the file… ·")
        else:
            bits = [copy.say_clock(f.duration), doc.kind]
            if f.size:
                bits.append(format_size(f.size))
            self.meta.setText(" · ".join(bits) + " ·")
        self.set_picture(bool(f and f.has_video), show_player=True)
        m = doc.model
        locked = bool(getattr(doc, "locked", False))
        self.different.setEnabled(True)
        self.switch.setChecked(doc.auto)
        self.switch.setEnabled(doc.has_audio and not locked and not doc.is_job)
        self.auto_label.setEnabled(self.switch.isEnabled())
        self.switch.setToolTip("Listens to a little of your file and picks the settings it needs. You can still change anything.")
        pend = len(m.open_suggestions()) if (m is not None and doc.show_suggestions) else 0
        for chip in (self.chip, self.chip2):
            chip.setVisible(pend > 0)
            chip.setText(f"{pend}  ✦")
            chip.setAccessibleName(f"{pend} suggestions. Show the next one")
        # Automatic's line
        text, action = "", ""
        if doc.auto and doc.listening:
            text = "Listening to a little of your file to pick the best settings…"
        elif doc.auto and doc.applied:
            if doc.auto_custom:
                text, action = "Set automatically, then changed by you.", "Use them again"
            else:
                text, action = "Set automatically for this file.", "Undo"
        self.auto_line.setVisible(bool(text))
        self.auto_text.setText(text)
        self.auto_action.setText(action)
        self.auto_action.setVisible(bool(action) and not locked)
        self.auto_action.setProperty("act", "again" if doc.auto_custom else "undo")
        show_why = bool(doc.auto and doc.applied and doc.reasons)
        self.why.setVisible(show_why)
        self.why_text.setText("\n".join(doc.reasons))
        if not show_why:
            self.why_text.hide()
        # the suggestions' line
        line, act = "", ""
        if doc.sg == "running":
            line, act = f"Listening to your file for moments worth a look… {round(doc.sg_fraction * 100)}%", "Cancel"
        elif doc.sg == "canceled":
            line = "Stopped looking. You can still choose parts yourself."
        elif doc.sg == "error":
            line = "Couldn’t look for suggestions in this file. You can still choose parts yourself."
        elif doc.sg == "done" and doc.sg_note:
            line = doc.sg_note
        elif doc.can_offer_find:
            line, act = "Moments worth a look are not looked for by themselves in this file.", "Look for them"
        self.sg_line.setVisible(bool(line))
        self.sg_text.setText(line)
        self.sg_action.setText(act)
        self.sg_action.setVisible(bool(act))
        # the tools
        self.play_button.setEnabled(doc.has_audio)
        self.split_button.setEnabled(doc.has_audio and not locked)
        self.undo_button.setEnabled(bool(m is not None and m.can_undo) and not locked)
        self.redo_button.setEnabled(bool(m is not None and m.can_redo) and not locked)
        self.all_button.setEnabled(doc.has_audio)
        self.all_button.setChecked(bool(m is not None and m.selection is None))
        # what can be listened to
        listen = bool(doc.is_job and (doc._result_file() is not None or doc.session is not None))
        self.hear_box.setVisible(listen)
        self.auto_box.setVisible(not listen)
        self.auto_line.setVisible(self.auto_line.isVisibleTo(self) and not listen)
        if listen and self.preview is not None:
            self.which.blockSignals(True)
            self.which.set_value(self.preview.listening)
            self.which.blockSignals(False)

    def resizeEvent(self, event) -> None:  # noqa: N802
        super().resizeEvent(event)
        w = self.width()
        for b, long, short in ((self.split_button, "Split at playhead", "Split"), (self.all_button, "Select all", "All")):
            b.setText("" if w < 480 else short if w < 620 else long)

    def _auto_action(self) -> None:
        (self.autoAgain if self.auto_action.property("act") == "again" else self.autoUndo).emit()

    def _sg_action(self) -> None:
        (self.cancelFind if self.sg_action.text() == "Cancel" else self.findMoments).emit()

    def _toggle_why(self) -> None:
        self.why_text.setVisible(not self.why_text.isVisible())
