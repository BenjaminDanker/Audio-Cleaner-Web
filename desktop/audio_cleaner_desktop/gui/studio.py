"""The editor screen (the web's Cleanup page and its Studio): always the same screen. Before a file is chosen the layout is already there (the place
to drop a file where the picture and the timeline go, greyed settings, Start off); choosing a file fills it in, in the same window.

    left     the editor (the picture, the file's name and facts, Automatic, the tools, the timeline with its rows), and under it the output
    right    the settings for the part that is chosen: Voice, Leveling, Subtitles
    under    the start row: Details or Your files, or the progress, and then the button

The screen shows one `doc.Doc` (a file that has not been started, one that is running or a finished job). The window decides what Start does."""
from __future__ import annotations

import logging
from collections.abc import Callable
from pathlib import Path

from PySide6.QtCore import QEvent, Qt, QTimer, Signal
from PySide6.QtWidgets import QBoxLayout, QFrame, QLabel, QStackedWidget, QVBoxLayout, QWidget

from .. import copies, engine
from . import copy, icons
from .doc import Doc
from .inspector import Inspector
from .job_preview import JobPreview
from .output_pane import OutputPane
from .speech_model import SpeechModelDownloader
from .start_bar import Popover, StartBar, file_row
from .studio_widgets import Pane, link_button, small_label
from .timeline import GhostRows, TimelineView
from .top_pane import TopPane
from .widgets import Banner

log = logging.getLogger(__name__)



class DropZone(QFrame):
    """The place to drop a file, in the editor's main pane: a click opens the file chooser, a folder can be added too."""

    chooseFiles = Signal()
    chooseFolder = Signal()

    def __init__(self, parent: QWidget | None = None):
        super().__init__(parent)
        self.setObjectName("DropArea")
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        self.setMinimumHeight(170)
        lay = QVBoxLayout(self)
        lay.setContentsMargins(24, 22, 24, 18)
        lay.setSpacing(6)
        lay.addStretch(1)
        self.art = QLabel()
        self.art.setAlignment(Qt.AlignmentFlag.AlignCenter)
        title = QLabel(copy.DROP_TITLE)
        title.setObjectName("Heading")
        title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        hint = small_label(copy.DROP_HINT)
        hint.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.folder = link_button("Add a folder…", "Add every video and audio file inside a folder, including its sub-folders")
        self.folder.clicked.connect(self.chooseFolder)
        lay.addWidget(self.art)
        lay.addWidget(title)
        lay.addWidget(hint)
        lay.addWidget(self.folder, 0, Qt.AlignmentFlag.AlignHCenter)
        lay.addStretch(1)
        self._refresh_art()

    def _refresh_art(self) -> None:
        self.art.setPixmap(icons.pixmap("upload", self.palette().placeholderText().color(), 40, self.devicePixelRatioF()))

    def mouseReleaseEvent(self, event) -> None:  # noqa: N802
        if event.button() == Qt.MouseButton.LeftButton and self.rect().contains(event.position().toPoint()):
            self.chooseFiles.emit()
        super().mouseReleaseEvent(event)

    def changeEvent(self, event) -> None:  # noqa: N802
        super().changeEvent(event)
        if event.type() == QEvent.Type.PaletteChange and hasattr(self, "art"):
            self._refresh_art()


class Studio(QWidget):
    chooseFiles = Signal()
    chooseFolder = Signal()
    differentFile = Signal()
    primaryClicked = Signal()
    startAllClicked = Signal()
    shareClicked = Signal()
    cancelClicked = Signal()
    deleteRequested = Signal()
    openResult = Signal()
    changeFolder = Signal()
    nextToOriginal = Signal()
    infoRequested = Signal()
    outputChanged = Signal()  # the name, the type or the size of the result was changed
    findOriginal = Signal()
    subtitlesOnly = Signal()

    def __init__(self, prefs, speech: SpeechModelDownloader, voice: SpeechModelDownloader, parent: QWidget | None = None):
        super().__init__(parent)
        self.setObjectName("Studio")
        self.prefs = prefs
        self.doc: Doc | None = None
        self.problems: Callable[[Doc], list[str]] = lambda doc: []
        # (files not started yet: all of them, and the others this file's settings can go to)
        self.waiting: Callable[[Doc | None], tuple[int, int]] = lambda doc: (0, 0)
        self.out_dir: Callable[[], Path | None] = lambda: None
        self.preview: JobPreview | None = None
        self._refresh_timer = QTimer(self, singleShot=True, interval=0, timeout=self._refresh_all)
        self._live_timer = QTimer(self, singleShot=True, interval=120, timeout=self._model_settled)
        self._progress_timer = QTimer(self, singleShot=True, interval=100, timeout=self._progress_tick)  # (how far a run is: drawn at most ten times a second)
        self._player_key = None
        self._hear = {"voice": True, "echo": True, "level": True}
        self._playhead = 0.0
        self._playing = False
        self._stream_for = None

        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(14)
        self.notes = QVBoxLayout()
        self.notes.setSpacing(8)
        outer.addLayout(self.notes)

        self.ed = QBoxLayout(QBoxLayout.Direction.LeftToRight)
        self.ed.setSpacing(14)
        left = QVBoxLayout()
        left.setSpacing(14)
        self.main = Pane(name="Pane")
        main_lay = QVBoxLayout(self.main)
        main_lay.setContentsMargins(16, 14, 16, 16)
        self.stack = QStackedWidget()
        main_lay.addWidget(self.stack)
        self.stack.addWidget(self._build_empty())
        self.stack.addWidget(self._build_editor())
        left.addWidget(self.main)
        self.output = OutputPane()
        left.addWidget(self.output, 1)  # it stretches: the left column ends level with the settings
        self.inspector = Inspector(speech, voice)
        self.ed.addLayout(left, 1)
        self.ed.addWidget(self.inspector)  # (it stretches too: both columns end at the same line)
        outer.addLayout(self.ed)
        self.start = StartBar()
        outer.addWidget(self.start)
        self._wire()
        self.set_doc(None)

    # ------------------------------------------------------------------------------------------------------ construction
    def _build_empty(self) -> QWidget:
        page = QWidget()
        lay = QVBoxLayout(page)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(10)
        title = QLabel(copy.EMPTY_TITLE)
        title.setObjectName("FileName")
        line = small_label(copy.EMPTY_LINE)
        self.drop = DropZone()
        self.drop.chooseFiles.connect(self.chooseFiles)
        self.drop.chooseFolder.connect(self.chooseFolder)
        self.problem = Banner()
        self.ghost = GhostRows()
        lay.addWidget(title)
        lay.addWidget(line)
        lay.addWidget(self.drop)
        lay.addWidget(self.problem)
        lay.addSpacing(6)
        lay.addWidget(self.ghost)
        return page

    def _build_editor(self) -> QWidget:
        page = QWidget()
        lay = QVBoxLayout(page)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(16)
        self.top = TopPane()
        self.timeline = TimelineView()
        lay.addWidget(self.top)
        lay.addWidget(self.timeline)
        return page

    def _wire(self) -> None:
        t = self.top
        t.differentFile.connect(self.differentFile)
        t.autoToggled.connect(lambda on: self.doc and self.doc.set_auto(on))
        t.autoUndo.connect(lambda: self.doc and self.doc.undo_auto())
        t.autoAgain.connect(lambda: self.doc and self.doc.use_auto_again())
        t.findMoments.connect(lambda: self.doc and self.doc.find())
        t.cancelFind.connect(lambda: self.doc and self.doc.cancel_find())
        t.nextSuggestion.connect(self._next_suggestion)
        t.play.connect(self.toggle_play)
        t.split.connect(self.split)
        t.undo.connect(lambda: self.doc and self.doc.model.undo_edit())
        t.redo.connect(lambda: self.doc and self.doc.model.redo_edit())
        t.zoomIn.connect(lambda: self._zoom(2))
        t.zoomOut.connect(lambda: self._zoom(0.5))
        t.selectAll.connect(self.select_all)
        t.listeningChanged.connect(self._listening_changed)
        tl = self.timeline
        tl.seekRequested.connect(self.seek)
        tl.selectionDragged.connect(self._dragged)
        tl.sectionClicked.connect(self._section)
        tl.pinClicked.connect(self._pin)
        tl.hearToggled.connect(self._hear_toggled)
        i = self.inspector
        i.infoRequested.connect(self.infoRequested)
        i.suggestionApplied.connect(lambda sid: self.doc and (self.doc.model.apply(sid), self.doc.keep_suggestions()))
        i.suggestionDismissed.connect(lambda sid: self.doc and (self.doc.model.dismiss(sid), self.doc.keep_suggestions()))
        o = self.output
        o.nameEdited.connect(self._name_typed)
        o.endingPicked.connect(self._ending_picked)
        o.nameDone.connect(self._refresh_all)
        o.sizePicked.connect(lambda k: self.doc and self.doc.set_video_size(k))
        o.changeFolder.connect(self.changeFolder)
        o.nextToOriginal.connect(self.nextToOriginal)
        s = self.start
        s.startClicked.connect(self.primaryClicked)
        s.startAllClicked.connect(self.startAllClicked)
        s.shareClicked.connect(self.shareClicked)
        s.cancelClicked.connect(self.cancelClicked)
        s.set_fillers(self._fill_details, self._fill_files)

    # ------------------------------------------------------------------------------------------------------ the doc
    def set_doc(self, doc: Doc | None) -> None:
        """Show this file (or the empty screen) in place."""
        old = self.doc
        if old is not None and old is not doc:
            for sig, slot in ((old.changed, self._schedule), (old.modelChanged, self._model_edited), (old.sessionReady, self._session_ready),
                              (old.progressed, self._progressed)):
                try:
                    sig.disconnect(slot)
                except (RuntimeError, TypeError):
                    pass
        self.doc = doc
        self._playhead = 0.0
        self.start.close_popover()
        if doc is None:
            self.stack.setCurrentIndex(0)
            self.top.doc = None
            self.timeline.plot.doc = None
            self.inspector.set_empty()
            self.output.set_empty()
            self._stop_player()
            self._refresh_start()
            return
        if old is not doc:
            doc.changed.connect(self._schedule)
            doc.modelChanged.connect(self._model_edited)
            doc.sessionReady.connect(self._session_ready)
            doc.progressed.connect(self._progressed)
        self.stack.setCurrentIndex(1)
        self.top.set_doc(doc)
        self.timeline.set_doc(doc)
        self.inspector.set_doc(doc)
        self.output.set_doc(doc)
        self._player_key = None
        self._refresh_all()

    def show_problem(self, text: str) -> None:
        """A file that could not be taken in (on the empty screen)."""
        if text:
            self.problem.show_message(text, "error")
        else:
            self.problem.hide()

    def _schedule(self) -> None:
        self._refresh_timer.start()

    def _progressed(self) -> None:
        if not self._progress_timer.isActive():
            self._progress_timer.start()

    def _progress_tick(self) -> None:
        """A run moved on: only the bar, the line and the marker on the timeline change."""
        doc = self.doc
        if doc is None:
            return
        self.timeline.plot.progress = doc.run_fraction if doc.locked and not doc.saving else None
        self.timeline.plot.update()
        self._refresh_start()

    def _model_edited(self) -> None:
        self._schedule()
        self._live_timer.start()

    def _model_settled(self) -> None:
        """The draft was edited and the edits stopped for a moment: a finished job's live sound follows it, and what saving would take is worked out."""
        doc = self.doc
        if doc is None or doc.model is None:
            return
        doc.refresh_session()
        self._refresh_timeline_state()
        self._refresh_start()

    def _refresh_all(self) -> None:
        doc = self.doc
        if doc is None:
            return
        self.top.refresh()
        self.timeline.refresh()
        self.inspector.refresh()
        self._refresh_output()
        self._refresh_timeline_state()
        self._refresh_start()
        self._refresh_notes()
        self._sync_player_sources()
        if not self._playing:
            self.top.set_time(self._playhead, doc.seconds)

    def _refresh_timeline_state(self) -> None:
        doc = self.doc
        if doc is None:
            return
        plot = self.timeline.plot
        plot.progress = doc.run_fraction if doc.locked and not doc.saving else None
        plot.subs_label = {"srt": "SRT", "vtt": "VTT", "txt": "TXT", "embed": "Embed"}.get(doc.subtitles, "")
        live = doc.stage == "done" and self.preview is not None and self.preview.live
        self.timeline.labels.set_hear(doc.stage == "done", enabled=live, state=self._hear)
        plot.hear_on = dict(self._hear)
        plot.hearing_text = self._hearing_text(doc) if doc.stage == "done" else ""
        plot.update()

    def _hearing_text(self, doc: Doc) -> str:
        lanes = [("voice", "Voice"), ("echo", "Echo removal"), ("level", "Leveling")]
        if not doc.echo_available:
            lanes = [x for x in lanes if x[0] != "echo"]
        off = [name for key, name in lanes if not self._hear.get(key, True)]
        if not off:
            return "Hearing: everything"
        if len(off) == len(lanes):
            return "Hearing: the original"
        return "Hearing: without " + " and ".join(off)

    def _refresh_output(self) -> None:
        doc = self.doc
        if doc is None:
            return
        self.output.refresh(self.out_dir(), self._files_text(doc), "")

    def _files_text(self, doc: Doc) -> str:
        """"When it's done, these are saved by themselves: ..."""
        if doc.stage == "processing":
            return ""
        names = []
        if doc.out_name:
            names.append(doc.out_name + (" (with the subtitles inside)" if doc.subtitles == "embed" else ""))
        kind = {"srt": "SRT", "vtt": "VTT", "txt": "TXT"}.get(doc.subtitles)
        if kind:
            names.append(f"the subtitles ({kind})")
        if not names:
            return ""
        return f"When it’s done, these are saved by themselves: <b>{' and '.join(names)}</b>."

    def _refresh_notes(self) -> None:
        doc = self.doc
        while self.notes.count():
            item = self.notes.takeAt(0)
            if item.widget() is not None:
                item.widget().hide()
                item.widget().deleteLater()
        if doc is None:
            return
        for text, kind in self.note_lines(doc):
            b = Banner()
            self.notes.addWidget(b)  # (in the screen first: shown without a parent it flashed as a small window of its own)
            find = doc.job is not None and not doc.original.ok and getattr(doc.original, "state", "") == "missing" and "original" in text.lower()
            b.show_message(text, kind, action_text="Find the original…" if find else "")
            b.action.connect(self.findOriginal)
            b._close.hide()

    def note_lines(self, doc: Doc) -> list[tuple[str, str]]:
        lines: list[tuple[str, str]] = []
        if doc.error:
            lines.append((doc.error, "error"))
        if doc.note:
            lines.append((doc.note, doc.note_kind))
        if doc.stage != "processing":
            lines.extend((p, "warn") for p in self.problems(doc))
        if doc.is_job and doc.job is not None and not doc.original.ok and not any("original" in t.lower() for t, _ in lines):
            lines.append((f"{doc.original.text} Changing a moment needs the original.", "warn"))
        if doc.session_state == "failed" and doc.session_note:
            lines.append((doc.session_note + " You can still change the settings and save; you just can't hear the changes while it plays.", "warn"))
        return lines

    # ------------------------------------------------------------------------------------------------------ the start row
    def _refresh_start(self) -> None:
        doc, s = self.doc, self.start
        waiting, others = self.waiting(doc)
        s.set_many(doc is not None and doc.model is not None and doc.stage != "processing" and others > 0, waiting)
        if doc is None or doc.model is None:
            s.set_mode("empty")
            s.set_button("Start", False)
            return
        problems = self.problems(doc)
        if doc.stage == "processing":
            s.set_mode("progress", text=doc.run_text, fraction=doc.run_fraction, cancel="Cancel job")
            pct = f" · {round((doc.run_fraction or 0) * 100)}%" if doc.run_fraction is not None else ""
            s.set_button(("Saving" if doc.saving else "Processing") + pct, False, state="processing")
        elif doc.stage == "done":
            s.set_mode("files")
            changed = doc.changed_sound or doc.changed_output
            if changed:
                quote = doc.quote() if doc.changed_sound else None
                about = f" · {_say_time(quote.seconds)}" if quote is not None and not quote.nothing else ""
                s.set_button("Save changes" + about, not problems, "; ".join(problems), "save")
            else:
                has_file = bool(self.result_files(doc))
                s.set_button("Open the file" if has_file else "Open the file", has_file, "" if has_file else "The file has been moved or deleted.", "open")
        else:
            s.set_mode("details")
            again = doc.item is not None and doc.item.state in ("failed", "cancelled")
            total = doc.estimate()[1]  # (the website says what a run costs here; on this computer it is free, so how long it takes)
            s.set_button(("Try again" if again else "Start") + (f" · {_say_time(total)}" if total > 0 else ""), not problems, "; ".join(problems), "start")

    def _fill_details(self, pop: Popover) -> None:
        doc = self.doc
        if doc is None:
            return
        quote = doc.quote() if doc.stage == "done" and doc.changed_sound else None
        rows, total = doc.estimate()
        if quote is not None:
            names = {"light": "Noise removal", "heavy": "Noise removal", "rebuild": "Voice rebuild", "echo": "Echo removal"}
            what = sorted({names[k] for k in (quote.steps or {}) if k in names})
            if quote.nothing:
                pop.row("Put the sound together from what is already made", "right away" if quote.seconds < 10 else _say_time(quote.seconds))
            else:
                pop.row("Make what changed: " + (", ".join(what) or "the new parts"), _say_time(quote.seconds), bold=False)
            pop.note("Only what changed is made again; the rest comes from what the job kept.")
            return
        if not rows:
            pop.row("Nothing switched on", "")
        for label, secs in rows:
            pop.row(label, _say_time(secs))
        if rows:
            pop.row("Total", _say_time(total), bold=True)
        pop.note("This computer does the work, so it is free and nothing is uploaded. Each line is the part of the file it covers; the time depends on "
                 "your computer.")

    def _fill_files(self, pop: Popover) -> None:
        doc = self.doc
        if doc is None:
            return
        files = self.result_files(doc)
        if not files:
            pop.row("Nothing was saved by this job.", "")
        for path in files:
            file_row(pop, path)
        if doc.entry is not None and doc.entry.message:
            pop.note(doc.entry.message)
        if doc.job is not None:
            from .items import format_size

            pop.note(f"Saved copies on this computer: {format_size(doc.job.bytes())}. They let this job be changed and saved again; they stay until you "
                     "delete the job.")
        from PySide6.QtWidgets import QPushButton

        if doc.job is not None and doc.job.has(copies.TRANSCRIPT):
            subs = QPushButton("Save only the subtitles")
            subs.setToolTip("Write the saved transcript as a subtitle file next to the last cleaned file. Needs neither the original nor a model.")
            subs.setEnabled(doc.subtitles in ("srt", "vtt", "txt"))
            subs.clicked.connect(lambda: (pop.hide(), self.subtitlesOnly.emit()))
            pop.lay.addWidget(subs, 0, Qt.AlignmentFlag.AlignLeft)
        delete = QPushButton("Delete this job")
        delete.setObjectName("DangerOutline")
        delete.setToolTip("Delete this job: its saved copies and its place in the list. Your original and the cleaned files are not touched.")
        delete.clicked.connect(lambda: (pop.hide(), self.deleteRequested.emit()))
        pop.lay.addWidget(delete, 0, Qt.AlignmentFlag.AlignLeft)

    # ------------------------------------------------------------------------------------------------------ the edits on the screen
    def _usual_name(self, doc: Doc) -> str:
        return engine.default_output_name(doc.path, doc.facts.source(), self.prefs.job_options()) if doc.facts is not None else ""

    def _name_typed(self, text: str) -> None:
        """The box holds the name; the ending is the type picked in the list beside it. An ending typed into the box ("talk.mp3") becomes the type
        as it is typed (one this file can't be saved as stays, and the hint says so); not when the box is left, which moved the screen under a
        click on Start."""
        doc = self.doc
        if doc is not None:
            if engine.split_name(text)[1] in engine.ENDING_NAMES:
                name = text
            else:  # the type in the list (one the file can't be saved as was typed earlier: the usual one)
                ok = engine.available_endings(doc.facts.source()) if doc.facts is not None else engine.ENDING_NAMES
                ending = engine.split_name(doc.out_name)[1] if doc.out_name else ""
                if ending not in ok:
                    ending = engine.split_name(self._usual_name(doc))[1]
                name = text + ending
            doc.set_out_name(name, custom=name.strip() != self._usual_name(doc) or doc.is_job)
            self._after_output()

    def _ending_picked(self, ending: str) -> None:
        doc = self.doc
        if doc is None or doc.facts is None:
            return
        name = engine.with_ending(doc.out_name or self._usual_name(doc), ending)
        doc.set_out_name(name, custom=name != self._usual_name(doc) or doc.is_job)
        self._after_output()

    def _after_output(self) -> None:
        doc = self.doc
        if doc is not None and doc.subtitles == "embed" and engine.split_name(doc.out_name)[1] not in engine._EMBED_ENDINGS:
            doc.set_subtitles("srt")  # only a video file can hold them
        self.outputChanged.emit()

    def _next_suggestion(self) -> None:
        doc = self.doc
        if doc is None or doc.model is None:
            return
        open_ = doc.model.open_suggestions()
        if not open_:
            return
        sel = doc.model.selection
        idx = next((i for i, s in enumerate(open_) if sel is not None and abs(s.start - sel[0]) < 0.6), -1)
        s = open_[(idx + 1) % len(open_)]
        self.seek(s.start)
        doc.model.show(s.id)

    def _pin(self, sid: str) -> None:
        doc = self.doc
        s = doc.model.suggestion(sid) if doc is not None and doc.model is not None else None
        if s is not None:
            self.seek(s.start)
            doc.model.show(sid)

    def _dragged(self, a: float, b: float) -> None:
        doc = self.doc
        if doc is None or doc.model is None or doc.locked:
            return
        doc.model.focus = None
        doc.model.select(float(round(a)), float(round(b)))

    def _section(self, a: float, b: float) -> None:
        doc = self.doc
        if doc is None or doc.model is None or doc.locked:
            return
        doc.model.focus = None
        if a <= 0 and b >= doc.seconds:
            doc.model.clear_selection()
        else:
            doc.model.select(a, b)

    def select_all(self) -> None:
        if self.doc is not None and self.doc.model is not None:
            self.doc.model.focus = None
            self.doc.model.clear_selection()

    def split(self) -> None:
        """Split at the playhead: the parts around it can then be chosen as two."""
        doc = self.doc
        if doc is None or doc.model is None or doc.locked:
            return
        t = float(round(self._playhead))
        if not 0 < t < doc.seconds:
            return
        if t not in doc.cuts:
            doc.cuts.append(t)
            doc.cuts.sort()
        from .timeline import split_at

        pieces = split_at([{**s, "on": True} for s in doc.model.voice_sections()], doc.cuts)
        piece = next((p for p in pieces if p["start"] <= t < p["end"]), None)
        if piece is not None:
            doc.model.select(piece["start"], piece["end"])
        self._schedule()

    def _zoom(self, factor: float) -> None:
        self.timeline.set_zoom(int(self.timeline.zoom * factor) if factor > 1 else max(1, int(self.timeline.zoom * factor)))
        self.top.zoom_out_button.setEnabled(self.timeline.zoom > 1)
        self.top.zoom_in_button.setEnabled(self.timeline.zoom < 8)

    # ------------------------------------------------------------------------------------------------------ the player
    def _ensure_preview(self) -> JobPreview | None:
        if self.preview is None:
            self.preview = JobPreview(compact=True)
            self.top.attach_preview(self.preview)
            self.preview.positionMoved.connect(self._position)
            self.preview.stateChanged.connect(self._state)
            self.preview.durationChanged.connect(lambda ms: None)
        return self.preview

    def _stop_player(self) -> None:
        if self.preview is not None:
            self.preview.stop()
            self.preview.use_stream(None)
        self._playing = False
        self.top.set_playing(False)
        self._stream_for = None

    def _sync_player_sources(self) -> None:
        """Give the player the files it needs once they are known (the original of a file; the original and the result of a job), and the live sound
        of a finished job once it is ready."""
        doc = self.doc
        if doc is None or doc.facts is None:
            return
        original = doc.path if doc.path.exists() else None
        cleaned = next(iter(self.result_files_audio(doc)), None) if doc.is_job else None
        key = (id(doc), str(original), str(cleaned), doc.entry.id if doc.entry is not None else 0, doc.stage == "done")
        if key == self._player_key:
            self._sync_stream()
            return
        self._player_key = key
        preview = self._ensure_preview()
        if original is None and cleaned is None:
            preview.stop()
            return
        subtitle = next((p for p in self.result_files(doc) if p.suffix.lower() in (".srt", ".vtt")), None) if doc.is_job else None
        preview.show_files(doc.entry.id if doc.entry is not None else doc.item.id if doc.item is not None else 0, original or cleaned, cleaned,
                           doc.facts.has_video, doc.name, subtitle)
        self._stream_for = None
        self._sync_stream()
        self.top.set_picture(doc.facts.has_video, show_player=True)

    def _sync_stream(self) -> None:
        doc = self.doc
        if doc is None or self.preview is None:
            return
        if doc.session is not None and doc.session.mix is not None and self._stream_for is not doc.session:
            self.preview.use_stream(doc.session.mix)
            self._stream_for = doc.session
            
            self._refresh_timeline_state()

    def result_files(self, doc: Doc) -> list[Path]:
        return doc.result_paths()

    def result_files_audio(self, doc: Doc) -> list[Path]:
        return [p for p in doc.result_paths() if p.suffix.lower() not in (".srt", ".vtt", ".txt") and p.exists()]

    def _session_ready(self, session) -> None:
        self._sync_stream()
        self._hear = {"voice": True, "echo": True, "level": True}
        self._refresh_timeline_state()

    def toggle_play(self) -> None:
        if self.preview is not None and self.preview.available:
            self.preview.toggle()

    def seek(self, seconds: float) -> None:
        doc = self.doc
        if doc is None:
            return
        seconds = max(0.0, min(doc.seconds, seconds))
        self._playhead = seconds
        self.timeline.set_playhead(seconds)
        self.top.set_time(seconds, doc.seconds)
        if self.preview is not None and self.preview.available:
            self.preview.seek_ms(int(seconds * 1000))

    def _position(self, ms: int) -> None:
        doc = self.doc
        if doc is None:
            return
        self._playhead = ms / 1000.0
        self.timeline.set_playhead(self._playhead)
        self.top.set_time(self._playhead, doc.seconds)
        if self._playing and self.timeline.zoom > 1:
            self.timeline.ensure_playhead_visible()

    def _state(self, playing: bool) -> None:
        self._playing = playing
        self.top.set_playing(playing)

    def _hear_toggled(self, lane: str, on: bool) -> None:
        self._hear[lane] = on
        doc = self.doc
        mix = doc.session.mix if doc is not None and doc.session is not None else None
        if mix is not None:
            for layer in {"voice": ("noise", "rebuild"), "echo": ("echo",), "level": ("leveling",)}[lane]:
                mix.set_hear(layer, on)
        self._refresh_timeline_state()

    def _listening_changed(self, which: str) -> None:
        if self.preview is not None:
            self.preview.set_listening(which)

    def release_player(self) -> None:
        """Let go of the files the player has open (Windows will not move or delete a file that is open); they are loaded again when they are needed."""
        if self.preview is not None:
            self.preview.release()
            self._playing = False
            self.top.set_playing(False)
        self._player_key = None

    def shutdown(self) -> None:
        """The window closes: the sound stops and the picture lets go of the file (Windows locks open files)."""
        if self.preview is not None:
            self.preview.stop()
            self.preview.release()

    # ------------------------------------------------------------------------------------------------------ window events
    def set_narrow(self, narrow: bool) -> None:
        """Narrow (the window tells how much room the screen has): the settings go under the editor, as wide as it. (Not decided from this widget's own
        width: the wide layout cannot get that narrow, so it could never find out.)"""
        want = QBoxLayout.Direction.TopToBottom if narrow else QBoxLayout.Direction.LeftToRight
        if self.ed.direction() != want:
            self.ed.setDirection(want)
            self.inspector.setMinimumWidth(0 if narrow else 356)
            self.inspector.setMaximumWidth(16777215 if narrow else 356)
            self.updateGeometry()

    def resizeEvent(self, event) -> None:  # noqa: N802
        super().resizeEvent(event)
        self.timeline.refresh() if self.doc is not None else None


def _say_time(seconds: float) -> str:
    from .. import editor_job

    return editor_job.say_time(seconds)
