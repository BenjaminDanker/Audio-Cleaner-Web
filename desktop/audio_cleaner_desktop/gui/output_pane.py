"""The output pane under the editor (the web's OutputPane): the name of the result with its file type, the video size, where it is saved, and what
the run saves by itself. Typing another ending changes the type; picking a type changes the ending."""
from __future__ import annotations

from pathlib import Path

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import QComboBox, QHBoxLayout, QLabel, QLineEdit, QVBoxLayout, QWidget

from .. import engine, labels
from . import copy
from .studio_widgets import Pane, link_button, small_label
from .theme import set_property
from .widgets import ElidedLabel


class OutputPane(Pane):
    nameEdited = Signal(str)
    endingPicked = Signal(str)
    nameDone = Signal()
    sizePicked = Signal(str)
    changeFolder = Signal()
    nextToOriginal = Signal()

    def __init__(self, parent: QWidget | None = None):
        super().__init__(parent, name="Pane")
        self.setProperty("outp", True)
        self.doc = None
        self._building = False
        self._endings: list[str] = []
        lay = QVBoxLayout(self)
        lay.setContentsMargins(16, 14, 16, 14)
        lay.setSpacing(10)
        self.empty = small_label(copy.OUTPUT_EMPTY)
        lay.addWidget(self.empty)
        self.fields = QWidget()
        f = QVBoxLayout(self.fields)
        f.setContentsMargins(0, 0, 0, 0)
        f.setSpacing(6)
        title = QLabel("Name of the result")
        title.setObjectName("FieldLabel")
        f.addWidget(title)
        line = QHBoxLayout()
        line.setSpacing(-1)  # the two borders overlap into one
        self.name = QLineEdit()
        self.name.setObjectName("NameBox")
        self.name.setAccessibleName("Name of the result")
        self.name.setToolTip("What the result is called. Change the ending (for example .mp3) to save it as something else, or pick a type in the list.")
        self.format = QComboBox()
        self.format.setObjectName("FormatBox")
        self.format.setAccessibleName("File type")
        self.format.setMinimumContentsLength(9)
        self.format.setSizeAdjustPolicy(QComboBox.SizeAdjustPolicy.AdjustToMinimumContentsLengthWithIcon)
        self.format.setToolTip("The types this file can be saved as. A video can become another kind of video, or just its sound. A sound can become a video "
                               "with a plain black picture, which can carry subtitles.")
        line.addWidget(self.name, 1)
        line.addWidget(self.format)
        f.addLayout(line)
        self.hint = small_label(copy.NAME_TIP)
        f.addWidget(self.hint)
        self.size_box = QWidget()
        s = QVBoxLayout(self.size_box)
        s.setContentsMargins(0, 4, 0, 0)
        s.setSpacing(6)
        size_title = QLabel("Video size")
        size_title.setObjectName("FieldLabel")
        self.size = QComboBox()
        self.size.setAccessibleName("Video size")
        for key, label in labels.VIDEO_SIZE_LABELS.items():
            self.size.addItem(label, key)
        self.size_hint = small_label()
        s.addWidget(size_title)
        s.addWidget(self.size)
        s.addWidget(self.size_hint)
        f.addWidget(self.size_box)
        where = QHBoxLayout()
        where.setSpacing(8)
        self.where = ElidedLabel()
        self.where.setObjectName("Small")
        self.change = link_button("Change…", "Pick the folder the cleaned files are saved in")
        self.next_to = link_button("Next to the original", "Save the cleaned files next to each original again")
        where.addWidget(self.where, 1)
        where.addWidget(self.change)
        where.addWidget(self.next_to)
        f.addLayout(where)
        self.will = small_label()
        self.will.setObjectName("Plain")
        f.addWidget(self.will)
        lay.addWidget(self.fields)
        lay.addStretch(1)
        self.name.textEdited.connect(lambda t: self.nameEdited.emit(t))
        self.name.editingFinished.connect(self.nameDone)
        self.format.activated.connect(lambda i: self.format.itemData(i) and self.endingPicked.emit(self.format.itemData(i)))
        self.size.activated.connect(lambda i: self.sizePicked.emit(self.size.itemData(i)))
        self.change.clicked.connect(self.changeFolder)
        self.next_to.clicked.connect(self.nextToOriginal)
        self.set_empty()

    def set_empty(self) -> None:
        self.doc = None
        self.empty.setObjectName("Small")
        self.empty.show()
        self.fields.hide()
        self.setEnabled(False)

    def set_doc(self, doc) -> None:
        self.doc = doc
        self.empty.hide()
        self.fields.show()
        self.refresh()

    def refresh(self, out_dir: Path | None = None, files_text: str = "", problem: str = "") -> None:
        doc = self.doc
        if doc is None:
            return
        self.setEnabled(not getattr(doc, "locked", False))
        facts = doc.facts
        source = facts.source() if facts is not None else None
        self._building = True
        try:
            self.name.setEnabled(source is not None)
            self.format.setEnabled(source is not None)
            if source is not None:
                endings = engine.available_endings(source)
                if endings != self._endings:
                    self._endings = endings
                    self.format.clear()
                    group = None
                    for ending in endings:
                        kind = "audio" if ending in engine.AUDIO_ENDINGS else "video"
                        if group is not None and kind != group:
                            self.format.insertSeparator(self.format.count())
                        group = kind
                        label = engine.ENDING_NAMES[ending]
                        if kind == "audio":
                            label += " (sound only)" if source.has_video else ""
                        elif not source.has_video:
                            label += " (plain black picture)"
                        self.format.addItem(label, ending)
                if self.name.text() != doc.out_name:  # (a changed text would move the cursor: only when it differs)
                    self.name.setText(doc.out_name)
                self.format.setCurrentIndex(self.format.findData(engine.split_name(doc.out_name)[1]))
                picture = engine.split_name(doc.out_name)[1] in (*engine.VIDEO_ENDINGS, ".m4v")  # (a sound made into a video has a plain black one)
                self.size_box.setVisible(bool(picture))
                self.size.setCurrentIndex(max(0, self.size.findData(doc.video_size)))
                self.size_hint.setText("The picture is copied as it is whenever the new file type can hold it." if doc.video_size == "same" else
                                       "The picture is made again at this size on this computer, so it takes a little longer; a size bigger than the "
                                       "original only stretches the picture.")
            else:
                self.size_box.hide()
            bad = problem or (engine.check_output_name(doc.out_name, source)[1] if source is not None and doc.out_name else "")
            self.hint.setText(bad or copy.NAME_TIP)
            set_property(self.hint, "state", "failed" if bad else "")
            set_property(self.name, "invalid", bool(bad))
            self.where.setText(f"Saved in {out_dir}" if out_dir is not None else "Saved next to the original")
            self.next_to.setVisible(out_dir is not None)
            self.will.setText(files_text)
            self.will.setVisible(bool(files_text))
            self.will.setTextFormat(Qt.TextFormat.RichText)
        finally:
            self._building = False
