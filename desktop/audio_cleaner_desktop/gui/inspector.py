"""The settings column of the editor screen (the web's Inspector.jsx): the heading says which part the settings are for ("Whole file · 0:11", or
the part that is chosen), then the cards Voice (Off, Noise removal with its strength and remover, Voice rebuild), Echo removal (a file with a second
audio track), Leveling and Subtitles. Before a file is chosen the same column is there with greyed cards.

It reads and edits the `doc.Doc`'s model for the chosen part; the screen redraws from the model's signals."""
from __future__ import annotations

from PySide6.QtCore import QSize, Qt, Signal
from PySide6.QtWidgets import (
    QGraphicsOpacityEffect, QButtonGroup, QComboBox, QFrame, QHBoxLayout, QLabel, QPushButton, QRadioButton, QSizePolicy, QToolButton, QVBoxLayout, QWidget,
)

from .. import editor_job, engine
from . import copy
from .speech_model import SpeechModelDownloader, SpeechModelPanel
from .studio_widgets import Card, Pills, SoftSegmented, Switch, link_button, small_label
from .theme import set_property

STRENGTHS = (("gentle", "Gentle"), ("balanced", "Balanced"), ("strong", "Strong"), ("max", "Max"))
STRENGTH_TIPS = {"gentle": "s.1", "balanced": "s.2", "strong": "s.3", "max": "s.4"}
REMOVERS = (("auto", "Auto"), ("light", "Light"), ("heavy", "Heavy"))
LEVELS = (("off", "Off"), ("podcast", "Podcast"), ("youtube", "YouTube"), ("broadcast", "Broadcast"), ("keep", "Even"))
SUBTITLE_FILES = (("srt", "SRT"), ("vtt", "VTT"), ("txt", "TXT"), ("embed", "Embed"))
WIDTH = 356


class InfoButton(QToolButton):
    """The (i) that opens "All the options"."""

    def __init__(self, parent: QWidget | None = None):
        super().__init__(parent)
        self.setObjectName("InfoDot")
        self.setText("i")
        self.setFixedSize(26, 26)
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        self.setAccessibleName("Compare all the options")
        self.setToolTip(copy.TIPS["info"])


class _VoiceOption(QFrame):
    """One row of the Voice card: a radio button, its name and a line about it; the strength and remover open under Noise removal."""

    def __init__(self, key: str, name: str, line: str, group: QButtonGroup, parent: QWidget | None = None):
        super().__init__(parent)
        self.key = key
        self.setObjectName("VOpt")
        self.setProperty("mode", key)
        lay = QVBoxLayout(self)
        lay.setContentsMargins(10, 6, 10, 6)
        lay.setSpacing(2)
        self.radio = QRadioButton(name)
        self.radio.setObjectName("VRadio")
        self.radio.setAccessibleName(name)
        self.radio.setToolTip(copy.TIPS[f"v.{key}"])
        group.addButton(self.radio)
        lay.addWidget(self.radio)
        self.line = small_label(line)
        self.line.setContentsMargins(22, 0, 0, 0)
        self.line.setVisible(bool(line))
        lay.addWidget(self.line)
        self.extra = QWidget()
        self.extra_lay = QVBoxLayout(self.extra)
        self.extra_lay.setContentsMargins(22, 4, 0, 4)
        self.extra_lay.setSpacing(6)
        self.extra.hide()
        lay.addWidget(self.extra)

    def set_on(self, on: bool) -> None:
        set_property(self, "on", on)

    def mouseReleaseEvent(self, event) -> None:  # noqa: N802
        """A click anywhere on the row picks it, not only on the button or its name (the line under it, the space around them)."""
        if event.button() == Qt.MouseButton.LeftButton and self.radio.isEnabled() and not self.radio.isChecked() and self.rect().contains(event.position().toPoint()):
            self.radio.click()
        super().mouseReleaseEvent(event)


class Inspector(QFrame):
    infoRequested = Signal()
    suggestionApplied = Signal(str)
    suggestionDismissed = Signal(str)
    selectionRequested = Signal(float, float)  # a suggestion's part was asked for

    def __init__(self, speech: SpeechModelDownloader, voice: SpeechModelDownloader, parent: QWidget | None = None):
        super().__init__(parent)
        self.setObjectName("Pane")
        self.setMinimumWidth(WIDTH)
        self.setMaximumWidth(WIDTH)
        self.setSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Preferred)
        self.doc = None
        self._building = False
        self._empty = True
        self._last_subtitles = "srt"
        self.speech, self.voice = speech, voice
        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(0)
        head = QHBoxLayout()
        head.setContentsMargins(16, 14, 16, 10)
        self.heading = QLabel("Settings")
        self.heading.setObjectName("InspHead")
        self.heading.setWordWrap(True)
        self.info = InfoButton()
        self.info.clicked.connect(self.infoRequested)
        head.addWidget(self.heading, 1)
        head.addWidget(self.info, 0, Qt.AlignmentFlag.AlignTop)
        outer.addLayout(head)
        self.body = QWidget()
        self.body.setObjectName("InspBody")
        self.body_lay = QVBoxLayout(self.body)
        self.body_lay.setContentsMargins(16, 0, 16, 14)
        self.body_lay.setSpacing(8)
        outer.addWidget(self.body)
        outer.addStretch(1)
        self._build_ghost()
        self._build_cards()
        self.set_empty()

    # ------------------------------------------------------------------------------------------------------ construction
    def _build_ghost(self) -> None:
        self.ghost = QWidget()
        lay = QVBoxLayout(self.ghost)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(8)
        for lane, title in (("voice", "Voice"), ("level", "Leveling"), ("subs", "Subtitles")):
            card = Card(lane, title)
            card.body.addWidget(small_label(copy.CARD_LINES[{"voice": "voice", "level": "level", "subs": "subs"}[lane]]))
            lay.addWidget(card)
        self.ghost.setEnabled(False)
        fade = QGraphicsOpacityEffect(self.ghost)
        fade.setOpacity(0.55)
        self.ghost.setGraphicsEffect(fade)
        self.body_lay.addWidget(self.ghost)

    def _build_cards(self) -> None:
        self.cards = QWidget()
        lay = QVBoxLayout(self.cards)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(8)
        # the suggestion for this part
        self.sg_card = QFrame()
        self.sg_card.setObjectName("SgCard")
        sg = QHBoxLayout(self.sg_card)
        sg.setContentsMargins(10, 6, 6, 6)
        self.sg_text = QLabel()
        self.sg_text.setWordWrap(True)
        self.sg_apply = QPushButton("Apply")
        self.sg_apply.setObjectName("Primary")
        self.sg_apply.setStyleSheet("QPushButton { padding: 4px 12px; }")
        self.sg_x = QToolButton()
        self.sg_x.setText("×")
        self.sg_x.setToolTip("Dismiss")
        self.sg_x.setAccessibleName("Dismiss")
        sg.addWidget(self.sg_text, 1)
        sg.addWidget(self.sg_apply)
        sg.addWidget(self.sg_x)
        self.sg_apply.clicked.connect(lambda: self._sg_id and self.suggestionApplied.emit(self._sg_id))
        self.sg_x.clicked.connect(lambda: self._sg_id and self.suggestionDismissed.emit(self._sg_id))
        self._sg_id = ""
        lay.addWidget(self.sg_card)
        # for a part
        self.part_row = QHBoxLayout()
        self.only = link_button("Leave the rest as it is", "Take every setting off everything outside this part: only this part is processed.")
        self.only.clicked.connect(self._leave_rest)
        self.revert = link_button("Undo changes in this part", "Put this part back the way the saved file has it.")
        self.revert.clicked.connect(self._revert)
        self.part_row.addWidget(self.only)
        self.part_row.addWidget(self.revert)
        self.part_row.addStretch(1)
        self.part_box = QWidget()
        self.part_box.setLayout(self.part_row)
        self.part_row.setContentsMargins(0, 0, 0, 0)
        lay.addWidget(self.part_box)
        # Voice
        self.voice_card = Card("voice", "Voice")
        self._modes = QButtonGroup(self)
        self._modes.setExclusive(True)
        self.options: dict[str, _VoiceOption] = {}
        for key, name, line in copy.VOICE_MODES:
            opt = _VoiceOption(key, name, line, self._modes)
            opt.radio.clicked.connect(lambda _=False, k=key: self._voice(k))
            self.options[key] = opt
            self.voice_card.body.addWidget(opt)
        self.strength = SoftSegmented(list(STRENGTHS), "voice")
        self.strength.set_tooltips({k: copy.TIPS[t] for k, t in STRENGTH_TIPS.items()})
        self.strength.changed.connect(self._strength)
        self.strength_hint = small_label()
        self.remover = SoftSegmented(list(REMOVERS), "voice")
        self.remover.set_tooltips({"auto": copy.TIPS["r.0"], "light": copy.TIPS["r.1"], "heavy": copy.TIPS["r.2"]})
        self.remover.changed.connect(self._remover)
        self.remover_hint = small_label()
        noise = self.options["noise"].extra_lay
        noise.addWidget(small_label("Strength"))
        noise.addWidget(self.strength)
        noise.addWidget(self.strength_hint)
        noise.addWidget(small_label("Remover"))
        noise.addWidget(self.remover)
        noise.addWidget(self.remover_hint)
        self.rebuild_panel = SpeechModelPanel(self.voice, need="Voice rebuild needs an extra model", what="the model that rebuilds the voice")
        self.options["rebuild"].extra_lay.addWidget(self.rebuild_panel)
        self.which = link_button("Which one should I pick?", "How Noise removal and Voice rebuild differ")
        self.which.clicked.connect(self.infoRequested)
        self.voice_card.body.addWidget(self.which, 0, Qt.AlignmentFlag.AlignLeft)
        self.voice.changed.connect(self._refresh_models)
        lay.addWidget(self.voice_card)
        # Echo removal
        self.echo_card = Card("echo", "Echo removal")
        self.echo_card.body.removeItem(self.echo_card.head)
        row = QHBoxLayout()
        row.addLayout(self.echo_card.head, 1)
        self.echo = Switch(lane="echo")
        self.echo.setAccessibleName("Echo removal")
        self.echo.setToolTip(copy.TIPS["echo"])
        self.echo.clicked.connect(self._echo)
        row.addWidget(self.echo)
        self.echo_card.body.addLayout(row)
        lay.addWidget(self.echo_card)
        # Leveling
        self.level_card = Card("level", "Leveling")
        self.level_card.title.setToolTip(copy.TIPS["level"])
        self.level = SoftSegmented(list(LEVELS), "level")
        self.level.set_tooltips({k: copy.TIPS[f"l.{i}"] for i, (k, _) in enumerate(LEVELS)})
        self.level.changed.connect(self._level)
        self.level_card.body.addWidget(self.level)
        lay.addWidget(self.level_card)
        # Subtitles
        self.subs_card = Card("subs", "Subtitles")
        self.subs_card.title.setToolTip(copy.TIPS["subs"])
        self.subs = SoftSegmented([("off", "Off"), ("on", "On")], "subs")
        self.subs.set_tooltips({"off": copy.TIPS["q.0"], "on": copy.TIPS["q.1"]})
        self.subs.changed.connect(self._subs)
        self.language = QComboBox()
        self.language.setAccessibleName("Language")
        self.language.setToolTip(copy.TIPS["lang"])
        self.language.addItem("Detect automatically", "auto")
        for code, (name, _) in engine.LANGUAGES.items():
            self.language.addItem(name, code)
        self.language.activated.connect(lambda i: self.doc and self.doc.set_language(self.language.itemData(i)))
        self.language_row = QWidget()
        lr = QHBoxLayout(self.language_row)
        lr.setContentsMargins(0, 0, 0, 0)
        label = small_label("Language", wrap=False)
        label.setFixedWidth(66)
        lr.addWidget(label)
        lr.addWidget(self.language, 1)
        self.files = Pills(list(SUBTITLE_FILES))
        self.files.changed.connect(self._file_kind)
        self.files_row = QWidget()
        fr = QVBoxLayout(self.files_row)  # (the label above the file types: beside it the four did not fit and the last was cut off)
        fr.setContentsMargins(0, 0, 0, 0)
        fr.setSpacing(4)
        fr.addWidget(small_label("Files", wrap=False))
        fr.addWidget(self.files)
        self.subs_note = small_label()
        self.speech_panel = SpeechModelPanel(self.speech)
        self.speech.changed.connect(self._refresh_models)
        for w in (self.subs, self.language_row, self.files_row, self.subs_note, self.speech_panel):
            self.subs_card.body.addWidget(w)
        lay.addWidget(self.subs_card)
        self.body_lay.addWidget(self.cards)

    # ------------------------------------------------------------------------------------------------------ the doc
    def set_empty(self) -> None:
        self._empty = True
        self.doc = None
        self.heading.setText("Settings")
        self.ghost.show()
        self.cards.hide()

    def set_doc(self, doc) -> None:
        self.doc = doc
        self._empty = False
        self.ghost.hide()
        self.cards.show()
        self.refresh()

    def _range(self) -> tuple[float, float]:
        return self.doc.selection()

    def refresh(self) -> None:
        doc = self.doc
        if doc is None or doc.model is None:
            self.heading.setText("Settings")
            self.cards.setEnabled(False)
            return
        m = doc.model
        a, b = self._range()
        whole = a <= 0 and b >= doc.seconds
        editable = not self._locked()
        self._building = True
        try:
            self.cards.setEnabled(editable)
            self.heading.setText(copy.selection_title(None if whole else (a, b), doc.seconds))
            # a suggestion for exactly this part
            sg = next((s for s in m.suggestions if s.state == "open" and doc.show_suggestions and abs(s.start - a) < 0.6 and abs(s.end - b) < 0.6), None)
            self._sg_id = sg.id if sg is not None else ""
            self.sg_card.setVisible(sg is not None)
            if sg is not None:
                self.sg_text.setText(f"<b>{sg.what}</b> · {sg.ask}")
                self.sg_apply.setText("Apply")
            show_revert = doc.is_job and doc.changed_sound
            self.only.setVisible(not whole)
            self.revert.setVisible(show_revert)
            self.part_box.setVisible(not whole or show_revert)
            # Voice
            mode = m.voice_mode_of(a, b)
            self._modes.setExclusive(False)
            for key, opt in self.options.items():
                opt.radio.setChecked(mode == key)
                opt.set_on(mode == key)
            self._modes.setExclusive(True)
            self.voice_card.set_mixed(mode == "mixed")
            self.voice_card.set_auto(doc.auto_matches("noise", "rebuild"))
            noise_on = mode == "noise"
            self.options["noise"].extra.setVisible(noise_on)
            if noise_on:
                key = m.strength_of(a, b)
                self.strength.set_value(key) if key in dict(STRENGTHS) else self.strength.clear()
                self.strength_hint.setText(copy.TIPS[STRENGTH_TIPS[key]].split(": ", 1)[1] if key in STRENGTH_TIPS else
                                           "This part has different strengths." if key == "mixed" else "")
                rem = m.remover_of(a, b)
                self.remover.set_value(rem) if rem in dict(REMOVERS) else self.remover.clear()
                self.remover_hint.setText({"auto": copy.TIPS["r.0"], "light": copy.TIPS["r.1"], "heavy": copy.TIPS["r.2"]}.get(rem, "").split(": ", 1)[-1]
                                          if rem in dict(REMOVERS) else "This part has different removers.")
            rebuild_on = mode == "rebuild"
            self.options["rebuild"].extra.setVisible(rebuild_on or not self.voice.ready and bool(self.voice.running or self.voice.last_error))
            self.rebuild_panel.setVisible(rebuild_on and not self.voice.ready or self.voice.running or bool(self.voice.last_error))
            # Echo removal
            self.echo_card.setVisible(m.echo_available)
            on = m.switch_of("echo", a, b)
            self.echo.setChecked(on is True)
            self.echo_card.set_mixed(on is None)
            self.echo_card.set_auto(doc.auto_matches("echo"))
            # Leveling
            lv = m.level_of(a, b)
            if lv == "on":
                self.level.set_value(editor_job.loudness_key(m.loudness()))
            elif lv == "off":
                self.level.set_value("off")
            else:
                self.level.clear()
            self.level_card.set_mixed(lv == "mixed")
            self.level_card.set_auto(doc.auto_matches("leveling"))
            # Subtitles
            kind = doc.subtitles
            if kind != "off":
                self._last_subtitles = kind
            self.subs.set_value("off" if kind == "off" else "on")
            self.language.setCurrentIndex(max(0, self.language.findData(doc.language)))
            self.language_row.setEnabled(kind != "off")
            self.files.set_value(kind if kind != "off" else "")
            self.files.set_all_enabled(kind != "off")
            video_out = self._video_out(doc)
            self.files.set_pill("embed", visible=bool(doc.facts is not None and (doc.facts.has_video or video_out)), enabled=kind != "off" and video_out,
                                tip=copy.TIPS["sf.embed"] if video_out else copy.TIPS["sf.embed.no"])
            for k in ("srt", "vtt", "txt"):
                self.files.set_pill(k, enabled=kind != "off", tip=copy.TIPS[f"sf.{k}"])
            self.subs_note.setText("Subtitles are for the whole file." if not whole and kind != "off" else "")
            self.subs_note.setVisible(bool(self.subs_note.text()))
            self._refresh_models()
        finally:
            self._building = False

    @staticmethod
    def _video_out(doc) -> bool:
        return engine.split_name(doc.out_name)[1] in engine._EMBED_ENDINGS if doc.out_name else bool(doc.facts is not None and doc.facts.has_video)

    def _refresh_models(self) -> None:
        doc = self.doc
        wants = doc is not None and doc.subtitles != "off"
        dl = self.speech
        self.speech_panel.setVisible(bool(dl.running or dl.last_error or (wants and not dl.ready)))
        self.speech_panel.refresh()
        self.rebuild_panel.refresh()
        if doc is not None and doc.model is not None:
            a, b = self._range()
            rebuild_on = doc.model.voice_mode_of(a, b) == "rebuild"
            vd = self.voice
            self.rebuild_panel.setVisible(bool(vd.running or vd.last_error or (rebuild_on and not vd.ready)))
            self.options["rebuild"].extra.setVisible(self.rebuild_panel.isVisibleTo(self.options["rebuild"].extra) or self.rebuild_panel.isVisible() or
                                                     bool(vd.running or vd.last_error or (rebuild_on and not vd.ready)))

    def _locked(self) -> bool:
        return self.doc is not None and getattr(self.doc, "locked", False)

    # ------------------------------------------------------------------------------------------------------ edits
    def _model(self):
        return self.doc.model if self.doc is not None and not self._building and not self._locked() else None

    def _voice(self, key: str) -> None:
        m = self._model()
        if m is not None:
            m.set_voice(key, *self._range())

    def _strength(self, key: str) -> None:
        m = self._model()
        if m is not None:
            m.set_strength(key, *self._range())

    def _remover(self, key: str) -> None:
        m = self._model()
        if m is not None:
            m.set_remover(key, *self._range())

    def _echo(self, on: bool) -> None:
        m = self._model()
        if m is not None:
            m.set_layer("echo", on, *self._range())

    def _level(self, key: str) -> None:
        m = self._model()
        if m is not None:
            m.set_level(engine.LOUDNESS_TARGETS[key] if key != "off" else None, key != "off", *self._range())

    def _subs(self, key: str) -> None:
        if self._building or self.doc is None or self._locked():
            return
        self.doc.set_subtitles(self._last_subtitles if key == "on" else "off")

    def _file_kind(self, key: str) -> None:
        if self._building or self.doc is None or self._locked():
            return
        self.doc.set_subtitles(key)

    def _leave_rest(self) -> None:
        m = self._model()
        if m is not None:
            m.leave_the_rest(*self._range())

    def _revert(self) -> None:
        m = self._model()
        if m is not None:
            m.undo_changes(*self._range())

    def sizeHint(self) -> QSize:  # noqa: N802
        return QSize(WIDTH, super().sizeHint().height())
