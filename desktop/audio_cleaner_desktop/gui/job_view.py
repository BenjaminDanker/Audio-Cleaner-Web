"""The job view: one job of the list opened. Its results, its saved copies and the disk they take, the player with the strength
slider (the sound changes while it plays), and "Change and save again" (only the later steps run again, from the saved copies)."""
from __future__ import annotations

import logging
import threading
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from PySide6.QtCore import QThread, Qt, Signal
from PySide6.QtWidgets import (
    QComboBox, QFileDialog, QFrame, QHBoxLayout, QLabel, QLineEdit, QMessageBox, QProgressBar, QPushButton, QScrollArea, QSlider,
    QVBoxLayout, QWidget,
)

from acw_common.ffmpeg import MediaError

from .. import blend, copies, editor_session, engine, labels, redo, resources
from . import shell
from .history import DONE, Entry, HistoryStore
from .job_preview import JobPreview
from .queue_view import format_size
from .theme import set_property
from .widgets import Banner, ElidedLabel, IconButton, WrapCheck, format_time

log = logging.getLogger(__name__)

SLIDER_STEPS = blend.SLIDER_STEPS


@dataclass
class JobEnv:
    """What the job view asks of the rest of the app (and what tests replace)."""

    voice_ready: Callable[[], bool] = lambda: resources.voice_model_dir() is not None
    speech_ready: Callable[[], bool] = resources.speech_ready
    noise_ready: Callable[[], bool] = resources.noise_ready
    run: Callable = redo.run
    prepare: Callable = redo.prepare_stream
    trash: Callable[[Path], bool] = shell.move_to_trash
    scratch: Callable[[], Path | None] = lambda: None  # where the live sound is prepared; None: next to the saved copies
    live: Callable[[], bool] = lambda: True  # prepare the sound for live listening (off where there is no sound output)
    open_session: Callable = editor_session.Session.open  # a job that keeps blocks, opened in the editor (editor_session.py)


class _Worker(QThread):
    """Runs `fn(cancel, progress)` off the window's thread."""

    done = Signal(object)
    failed = Signal(str)
    stopped = Signal()
    progress = Signal(float)

    def __init__(self, fn: Callable, parent=None):
        super().__init__(parent)
        self._fn = fn
        self.cancel = threading.Event()

    def run(self) -> None:
        try:
            result = self._fn(self.cancel, self.progress.emit)
        except engine.Cancelled:
            self.stopped.emit()
        except MediaError as e:
            self.failed.emit(str(e))
        except Exception as e:  # noqa: BLE001 - one bad job must not take the app down
            log.exception("the job view's background work failed")
            self.failed.emit("Disk is full." if getattr(e, "errno", None) == 28 else "Something went wrong; see the log.")
        else:
            self.done.emit(result)


def _source_of(job: copies.JobCopies) -> engine.Source:
    m = job.media
    return engine.Source(bool(m.get("has_video")), m.get("ext") or job.original_path.suffix.lower(), m.get("video_codec"),
                         float(m.get("duration") or 0.0), int(m.get("channels") or 2))


def _card(title: str) -> tuple[QFrame, QVBoxLayout]:
    card = QFrame()
    card.setObjectName("Card")
    layout = QVBoxLayout(card)
    layout.setContentsMargins(18, 14, 18, 16)
    layout.setSpacing(8)
    heading = QLabel(title)
    heading.setObjectName("CardTitle")
    layout.addWidget(heading)
    return card, layout


def _small(text: str = "", wrap: bool = True) -> QLabel:
    label = QLabel(text)
    label.setObjectName("Small")
    label.setWordWrap(wrap)
    return label


class JobView(QWidget):
    closed = Signal()  # "Back to the jobs"
    deleteRequested = Signal(int)
    changed = Signal()  # the entry (its results, its copies) is different now: the list should be drawn again

    def __init__(self, store: HistoryStore, env: JobEnv | None = None, parent: QWidget | None = None):
        super().__init__(parent)
        self._store = store
        self._env = env or JobEnv()
        self.entry: Entry | None = None
        self.job: copies.JobCopies | None = None
        self._check = copies.OriginalCheck("missing")
        self._prepared: redo.Prepared | None = None
        self._prepare_thread: _Worker | None = None
        self._save_thread: _Worker | None = None
        self._generation = 0
        self._db = 100.0
        self._building = False
        self.preview: JobPreview | None = None
        self._player_in_editor = False  # the player sits in the editor (a job that keeps blocks), not in the Listen card

        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 16, 0, 0)
        outer.setSpacing(10)
        top = QHBoxLayout()
        self._back = QPushButton("← Jobs")
        self._back.setToolTip("Back to the list of jobs")
        self._back.clicked.connect(self.closed)
        self._title = ElidedLabel()
        self._title.setObjectName("Heading")
        self._status = QLabel()
        self._delete = IconButton("trash", "Delete this job: its saved copies and its place in the list. Your original and the cleaned "
                                           "files are not touched.")
        self._delete.setFixedSize(34, 34)
        self._delete.clicked.connect(lambda: self.entry and self.deleteRequested.emit(self.entry.id))
        top.addWidget(self._back)
        top.addWidget(self._title, 1)
        top.addWidget(self._status)
        top.addWidget(self._delete)
        outer.addLayout(top)
        self._meta = _small()
        outer.addWidget(self._meta)
        self.banner = Banner()
        outer.addWidget(self.banner)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.Shape.NoFrame)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        holder = QWidget()
        holder.setObjectName("RowHolder")
        self._body = QVBoxLayout(holder)
        self._body.setContentsMargins(0, 0, 8, 8)
        self._body.setSpacing(12)
        scroll.setWidget(holder)
        outer.addWidget(scroll, 1)

        self._build_results()
        self._build_copies()
        self._build_listen()
        from .job_editor import JobEditor  # (it imports this module's helpers)

        self._ed = JobEditor(self)
        self._body.addWidget(self._ed.card)
        self._build_change()
        self._body.addStretch(1)

    # -- construction -------------------------------------------------------------------------------------------------

    def _build_results(self) -> None:
        card, layout = _card("Cleaned files")
        self._results_layout = QVBoxLayout()
        self._results_layout.setSpacing(4)
        layout.addLayout(self._results_layout)
        self._results_note = _small()
        layout.addWidget(self._results_note)
        self._body.addWidget(card)

    def _build_copies(self) -> None:
        card, layout = _card("Saved copies")
        self._copies_text = QLabel()
        self._copies_text.setWordWrap(True)
        self._copies_text.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        self._disk = _small()
        self._original_text = QLabel()
        self._original_text.setWordWrap(True)
        self._original_text.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        self._find = QPushButton("Find the original...")
        self._find.setToolTip("Point at the original if it was moved or renamed. It is only accepted if it is the same file.")
        self._find.clicked.connect(self._find_original)
        layout.addWidget(self._copies_text)
        layout.addWidget(self._disk)
        layout.addWidget(self._original_text)
        row = QHBoxLayout()
        row.addWidget(self._find)
        row.addStretch(1)
        layout.addLayout(row)
        self._body.addWidget(card)

    def _build_listen(self) -> None:
        self._listen_card, layout = _card("Listen")
        self._listen_layout = layout
        self._player_slot = QVBoxLayout()
        layout.addLayout(self._player_slot)
        head = QHBoxLayout()
        self._strength_label = QLabel()
        head.addWidget(self._strength_label, 1)
        layout.addLayout(head)
        self._slider = QSlider(Qt.Orientation.Horizontal)
        self._slider.setRange(0, SLIDER_STEPS)
        self._slider.setTickPosition(QSlider.TickPosition.TicksBelow)
        self._slider.setTickInterval(SLIDER_STEPS // 3)
        self._slider.setAccessibleName("Strength")
        self._slider.valueChanged.connect(self._slider_moved)
        layout.addWidget(self._slider)
        names = QHBoxLayout()
        names.setSpacing(6)
        self._name_buttons: dict[str, QPushButton] = {}
        for key, label, db in blend.NAMES:
            button = QPushButton(f"{label}")
            button.setObjectName("Flat")
            button.setToolTip(f"{label} ({db:.0f} dB)")
            button.clicked.connect(lambda _=False, d=db: self.set_strength(d))
            names.addWidget(button)
            self._name_buttons[key] = button
        layout.addLayout(names)
        self._prep_bar = QProgressBar()
        self._prep_bar.setRange(0, 0)
        self._prep_bar.setTextVisible(False)
        self._prep_bar.setMaximumHeight(8)
        layout.addWidget(self._prep_bar)
        self._slide_note = _small()
        layout.addWidget(self._slide_note)
        self._body.addWidget(self._listen_card)

    def _build_change(self) -> None:
        self._change_card, layout = _card("Change and save again")
        self._change_title = layout.itemAt(0).widget()
        self._change_intro = intro = _small("Pick what to change and save it as a new file next to the old one. Only the steps that come after "
                                            "the saved copies run again, so this takes seconds.")
        layout.addWidget(intro)
        self._rebuild = WrapCheck("Rebuild the voice")
        self._rebuild.setToolTip("Rebuild the voice instead of removing the noise: for echo, a phone-quality sound or a muffled voice. "
                                 "It has no strength.")
        self._rebuild.toggled.connect(self._choices_changed)
        self._leveling = WrapCheck("Leveling (even out the volume and set the loudness)")
        self._leveling.toggled.connect(self._leveling_toggled)
        self._loudness = QComboBox()
        for key, label in labels.LOUDNESS_LABELS.items():
            self._loudness.addItem(label, key)
        self._loudness.currentIndexChanged.connect(self._choices_changed)
        layout.addWidget(self._rebuild)
        layout.addWidget(self._leveling)
        layout.addWidget(self._loudness)
        name_row = QHBoxLayout()
        name_row.setSpacing(0)
        self._name = QLineEdit()
        self._name.setObjectName("NameBox")
        self._name.setPlaceholderText("Name of the new file")
        self._name.setAccessibleName("Name of the new file")
        self._name.textChanged.connect(self._choices_changed)
        self._ending = QComboBox()
        self._ending.setObjectName("FormatBox")
        self._ending.setToolTip("The type of the new file")
        self._ending.activated.connect(self._ending_picked)
        name_row.addWidget(self._name, 1)
        name_row.addWidget(self._ending)
        layout.addLayout(name_row)
        self._subtitles = QComboBox()
        for key, label in labels.SUBTITLE_LABELS.items():
            self._subtitles.addItem(label, key)
        self._subtitles.setToolTip("Subtitles from what was said. A transcript that is saved is used again, nothing is listened to twice.")
        self._subtitles.currentIndexChanged.connect(self._choices_changed)
        layout.addWidget(self._subtitles)
        self._plan_text = QLabel()
        self._plan_text.setWordWrap(True)
        self._plan_text.setObjectName("Muted")
        layout.addWidget(self._plan_text)
        self._save_bar = QProgressBar()
        self._save_bar.setRange(0, 1000)
        self._save_bar.setTextVisible(False)
        self._save_bar.hide()
        layout.addWidget(self._save_bar)
        buttons = QHBoxLayout()
        self._save = QPushButton("Save as a new file")
        self._save.setObjectName("Primary")
        self._save.clicked.connect(self.save)
        self._stop = QPushButton("Stop")
        self._stop.clicked.connect(self._stop_save)
        self._stop.hide()
        self._save_subs = QPushButton("Save only the subtitles")
        self._save_subs.setToolTip("Write the saved transcript as a subtitle file next to the last cleaned file. Needs neither the "
                                   "original nor a model.")
        self._save_subs.clicked.connect(self.save_subtitles_only)
        buttons.addWidget(self._save_subs)
        buttons.addStretch(1)
        buttons.addWidget(self._stop)
        buttons.addWidget(self._save)
        layout.addLayout(buttons)
        self._body.addWidget(self._change_card)

    # -- showing a job -------------------------------------------------------------------------------------------------

    @property
    def busy(self) -> bool:
        """A change is being saved."""
        return self._save_thread is not None

    def show_entry(self, entry_id: int) -> bool:
        """Open the job with this id. False when there is no such job."""
        entry = self._store.get(entry_id)
        if entry is None:
            return False
        self.unload()
        self.entry = entry
        self.job = None
        if entry.job_dir:
            folder = copies.job_dir(self._store.jobs_root, entry.job_dir)
            job = copies.JobCopies.load(folder) if folder is not None else None
            self.job = job if job is not None and job else None
        self._check = self.job.check_original() if self.job else copies.OriginalCheck("missing")
        self.banner.hide()
        self._fill()
        if not self._ed.show():
            self._start_preparing()
        return True

    def _fill(self) -> None:
        entry, job = self.entry, self.job
        self._building = True
        try:
            self._title.setText(entry.name)
            ok = entry.status == DONE
            self._status.setText("Done" if ok else "Failed")
            set_property(self._status, "state", "done" if ok else "failed")
            self._fill_meta()
            self._fill_results()
            self._fill_copies()
            self._fill_choices()
        finally:
            self._building = False
        self._refresh_all()

    def _fill_meta(self) -> None:
        entry = self.entry
        ok = entry.status == DONE
        when = datetime.fromtimestamp(entry.finished_at).strftime("%d %b %Y, %H:%M")
        meta = f"{when}  \u00b7  {format_time(entry.duration)}"
        if ok:
            meta += f"  \u00b7  took {_spent(entry.seconds_taken)}  \u00b7  {labels.describe_settings(entry.settings, entry.has_video)}"
        else:
            meta += f"  \u00b7  {entry.message}"
        self._meta.setText(meta)

    def _result_paths(self) -> list[tuple[Path, str]]:
        """(file, when) of the results of this job, newest first."""
        if self.job is not None and self.job.saves:
            out = []
            for save in reversed(self.job.saves):
                when = datetime.fromtimestamp(save.get("at", 0)).strftime("%d %b, %H:%M")
                out += [(Path(f), when) for f in save.get("files", [])]
            return out
        when = datetime.fromtimestamp(self.entry.finished_at).strftime("%d %b, %H:%M")
        return [(Path(p), when) for p in self.entry.outputs]

    def _fill_results(self) -> None:
        while self._results_layout.count():
            item = self._results_layout.takeAt(0)
            if item.widget() is not None:
                item.widget().hide()
                item.widget().deleteLater()
            elif item.layout() is not None:
                _clear(item.layout())
        paths = self._result_paths()
        for path, when in paths:
            row = QHBoxLayout()
            exists = path.exists()
            name = ElidedLabel(path.name if exists else f"{path.name}  (moved or deleted)")
            name.setToolTip(str(path))
            if not exists:
                name.setObjectName("Muted")
            when_label = _small(when, wrap=False)
            row.addWidget(name, 1)
            row.addWidget(when_label)
            if exists:
                open_btn = IconButton("play", "Open it with your default player")
                open_btn.clicked.connect(lambda _=False, p=path: shell.open_path(p))
                reveal = IconButton("folder", "Show it in its folder")
                reveal.clicked.connect(lambda _=False, p=path: shell.reveal_in_folder(p))
                for b in (open_btn, reveal):
                    b.setFixedSize(30, 30)
                    row.addWidget(b)
            self._results_layout.addLayout(row)
        self._results_note.setVisible(not paths)
        self._results_note.setText("No cleaned file was saved by this job." if not paths else "")

    def _fill_copies(self) -> None:
        job = self.job
        if job is None:
            self._copies_text.setText("This job has no saved copies. It was cleaned before Audio Cleaner kept them, or they were "
                                      "removed. You can still listen to it and delete it, but it can't be changed and saved again; "
                                      "clean the original again to get that.")
            self._disk.setText("")
            self._original_text.setText("")
            self._find.hide()
            return
        lines = [f"• {copies.KIND_NAMES[k]}: {format_size(n)}" for k, n in job.sizes()]
        self._copies_text.setText("These stay on this computer until you delete the job:\n" + "\n".join(lines))
        self._disk.setText(f"Disk used by this job: {format_size(job.bytes())}")
        if self._check.ok:
            self._original_text.setText(f"The original is where it was and unchanged: {job.original_path}")
        else:
            self._original_text.setText(self._check.text)
        self._find.setVisible(self._check.state == "missing")

    def _fill_choices(self) -> None:
        job = self.job
        if job is None:
            return
        wanted = redo.wanted_from(job)
        source = _source_of(job)
        self._db = wanted.atten_lim_db
        self._slider.setValue(round(blend.pos_of(self._db) * SLIDER_STEPS / 3))
        self._rebuild.setChecked(wanted.rebuild_voice)
        self._leveling.setChecked(wanted.leveling)
        self._loudness.setCurrentIndex(max(0, self._loudness.findData(wanted.loudness)))
        self._subtitles.setCurrentIndex(max(0, self._subtitles.findData(wanted.subtitles)))
        self._ending.clear()
        for ending in engine.available_endings(source):
            self._ending.addItem(engine.ENDING_NAMES.get(ending, ending), ending)
        self._name.setText(redo.suggest_name(job, source))
        self._sync_ending()

    # -- the choices ------------------------------------------------------------------------------------------------

    def set_strength(self, db: float) -> None:
        """The strength to hear and to save (the slider moves to it)."""
        self._building = True
        try:
            self._slider.setValue(round(blend.pos_of(db) * SLIDER_STEPS / 3))
        finally:
            self._building = False
        self._set_db(blend.snap(db))

    def _slider_moved(self, value: int) -> None:
        if self._building:
            return
        self._set_db(blend.snap(blend.db_at(value * 3 / SLIDER_STEPS)))

    def _set_db(self, db: float) -> None:
        self._db = float(db)
        if self.preview is not None:
            self.preview.set_strength(self._db)
        self._refresh_all()

    def _leveling_toggled(self, on: bool) -> None:
        self._loudness.setEnabled(on)
        if self.preview is not None:
            self.preview.set_exact(not on)
        self._choices_changed()

    def _choices_changed(self, *_args) -> None:
        if not self._building:
            self._sync_ending()
            self._refresh_all()

    def _ending_picked(self, index: int) -> None:
        ending = self._ending.itemData(index)
        if ending and self.job is not None:
            self._name.setText(engine.with_ending(self._name.text() or redo.suggest_name(self.job, _source_of(self.job)), ending))

    def _sync_ending(self) -> None:
        """The type box shows the ending the name has."""
        ending = engine.split_name(self._name.text())[1]
        i = self._ending.findData(ending)
        self._ending.blockSignals(True)
        self._ending.setCurrentIndex(i if i >= 0 else -1)
        self._ending.blockSignals(False)

    def wanted(self) -> redo.Wanted:
        return redo.Wanted(
            atten_lim_db=self._db, rebuild_voice=self._rebuild.isChecked(), leveling=self._leveling.isChecked(),
            loudness=self._loudness.currentData() or engine.DEFAULT_LOUDNESS, subtitles=self._subtitles.currentData() or "off",
            language=(self.job.made.get("language") if self.job and self.job.made.get("language") in engine.LANGUAGES else "auto"),
            video_size=(self.job.made.get("video_size") if self.job else "same") or "same", output_name=self._name.text().strip(),
            timeline=self._ed.timeline_json() if self._ed.active else None,
        )

    def plan(self) -> redo.Plan | None:
        if self.job is None:
            return None
        return redo.plan(self.job, self.wanted(), self._check, source=_source_of(self.job), voice_model=self._env.voice_ready(),
                         speech_model=self._env.speech_ready(), noise_models=self._env.noise_ready())

    def _slide_problem(self) -> str:
        """Why the strength can't be changed for this job ("" when it can)."""
        job = self.job
        if job is None:
            return "This job was cleaned before Audio Cleaner kept copies, so its strength can't be changed."
        if not self._check.ok:
            return f"{self._check.text} The strength can't be changed without it."
        if not redo.can_slide(job):
            if job.has(copies.REBUILT) and not job.has(copies.NOISE):
                return ("This job rebuilt the voice, which has no strength. Turn off Rebuild the voice below to remove the noise "
                        "instead (that runs the noise removal once).")
            return "The saved noise removal of this job is missing, so its strength can't be changed."
        return ""

    def _refresh_all(self) -> None:
        if self.entry is None:
            return
        job = self.job
        edit = self._ed.active  # the job keeps blocks: its sound is changed in the editor
        self._listen_card.setVisible(not edit)
        self._change_card.setVisible(job is not None)
        for w in (self._rebuild, self._leveling, self._loudness, self._save, self._change_intro):
            w.setVisible(not edit)
        self._change_title.setText("Save as a new file" if edit else "Change and save again")
        problem = self._slide_problem()
        rebuild_on = job is not None and self._rebuild.isChecked()
        can_slide = not problem and not rebuild_on
        self._slider.setEnabled(can_slide and not self.busy)
        for b in self._name_buttons.values():
            b.setEnabled(can_slide and not self.busy)
        self._strength_label.setText(
            f"Strength: <b>{blend.strength_name(self._db)}</b> ({self._db:.0f} dB)" if can_slide
            else "Strength: not used" if rebuild_on else "Strength: can't be changed for this job")
        for key, b in self._name_buttons.items():
            b.setProperty("on", abs(blend.db_of_key(key) - self._db) < 0.5 and can_slide)
            b.style().unpolish(b)
            b.style().polish(b)
        if self._prepare_thread is not None:
            self._slide_note.setText("Getting the sound ready to try other strengths...")
        elif rebuild_on:
            self._slide_note.setText("Rebuilding the voice has no strength. What you hear here is the noise removal; the rebuilt voice "
                                     "can be heard once it is saved.")
        elif problem:
            self._slide_note.setText(problem)
        elif self.preview is not None and self.preview.live:
            exact = not self._leveling.isChecked()
            self._slide_note.setText("What you hear is what you save." if exact else
                                     "This is the noise removal at an even loudness. Leveling is added when you save.")
        else:
            self._slide_note.setText(self._live_note)
        self._prep_bar.setVisible(self._prepare_thread is not None)
        if job is None:
            return
        p = self.plan()
        text = "\n".join(["• " + s for s in p.steps] + ["", *p.problems]) if p.problems else "\n".join("• " + s for s in p.steps)
        self._plan_text.setText(("What will happen:\n" + text) if p.steps else text)
        busy = self.busy
        for w in (self._rebuild, self._leveling, self._name, self._ending, self._subtitles):
            w.setEnabled(not busy)
        self._loudness.setEnabled(self._leveling.isChecked() and not busy)
        self._save.setEnabled(p.ok and not busy)
        self._save.setToolTip("; ".join(p.problems))
        can_text = job.has(copies.TRANSCRIPT) and (self._subtitles.currentData() or "off") in redo.TEXT_KINDS and not busy
        self._save_subs.setVisible(job.has(copies.TRANSCRIPT))
        self._save_subs.setEnabled(can_text)
        self._back.setEnabled(not busy)
        self._delete.setEnabled(not busy)
        if edit:
            self._ed.refresh_bar()

    _live_note = ""

    # -- the player and the live sound ---------------------------------------------------------------------------------

    def _ensure_preview(self) -> JobPreview:
        if self.preview is None:
            self.preview = JobPreview()
            self._player_slot.addWidget(self.preview)
        return self.preview

    def _show_player(self, stream=None) -> None:
        """The player for this job: the original against the saved result, and with `stream` the live blend."""
        entry = self.entry
        original = self.job.original_path if self.job is not None else Path(entry.source)
        results = [p for p, _ in self._result_paths() if p.exists()]
        cleaned = next((p for p in results if p.suffix.lower() not in (".srt", ".vtt", ".txt")), None)
        if not original.exists() or (cleaned is None and stream is None):
            if self.preview is not None:
                self.preview.stop()
                self.preview.hide()
            return
        preview = self._ensure_preview()
        subtitle = next((p for p in results if p.suffix.lower() in (".srt", ".vtt")), None)
        preview.show_job(entry.id, original, cleaned, entry.has_video, entry.name, subtitle, stream)
        preview.set_strength(self._db)
        preview.show()

    def _start_preparing(self) -> None:
        self._live_note = ""
        job = self.job
        self._generation += 1
        if job is None or self._slide_problem():
            self._show_player()
            self._refresh_all()
            return
        self._show_player()
        if not self._env.live():
            self._live_note = "Hear the original and the saved result with the buttons above. Drag the slider to pick a strength to save."
            self._refresh_all()
            return
        if self.preview is not None and not self.preview.available:
            self._live_note = ("Listening is not available on this computer (no sound support), but you can still pick a strength and "
                               "save it.")
            self._refresh_all()
            return
        generation = self._generation
        env = self._env
        exact = not self._leveling.isChecked()
        scratch = env.scratch()
        worker = _Worker(lambda cancel, progress: env.prepare(job, cancel=cancel, scratch_parent=scratch, exact=exact), self)
        worker.done.connect(lambda prepared, g=generation: self._prepared_ready(prepared, g))
        worker.failed.connect(lambda message, g=generation: self._prepare_failed(message, g))
        worker.stopped.connect(lambda g=generation: self._prepare_over(g))
        self._prepare_thread = worker
        worker.start()
        self._refresh_all()

    def _prepare_over(self, generation: int) -> None:
        thread, self._prepare_thread = self._prepare_thread, None
        if thread is not None:
            thread.wait()
            thread.deleteLater()
        if generation == self._generation:
            self._refresh_all()

    def _prepared_ready(self, prepared: redo.Prepared, generation: int) -> None:
        self._prepare_over(generation)
        if generation != self._generation or self.entry is None:
            prepared.close()
            return
        self._prepared = prepared
        self._show_player(prepared.stream)
        if self.preview is not None:
            self.preview.set_exact(not self._leveling.isChecked())
            self.preview.set_strength(self._db)
        self._refresh_all()

    def _prepare_failed(self, message: str, generation: int) -> None:
        self._prepare_over(generation)
        if generation == self._generation:
            self._live_note = f"{message} You can still pick a strength and save it."
            self._refresh_all()

    def unload(self) -> None:
        """Let go of the job: stop the sound, stop getting it ready, delete the scratch files of the live sound."""
        self._generation += 1
        if self.preview is not None:
            self.preview.stop()
            self.preview.use_stream(None)
        thread = self._prepare_thread
        if thread is not None:
            thread.cancel.set()
            thread.wait()
            self._prepare_thread = None
            thread.deleteLater()
        prepared, self._prepared = self._prepared, None
        if prepared is not None:
            prepared.close()
        self._ed.unload()

    def shutdown(self, timeout_ms: int | None = None) -> bool:
        """The window is closing: stop a save that is running (waiting at most `timeout_ms` for it), then unload. Returns whether
        everything ended."""
        ended = True
        thread = self._save_thread
        if thread is not None:
            thread.cancel.set()
            ended = thread.wait(timeout_ms) if timeout_ms is not None else thread.wait()
            if ended:
                self._save_thread = None
        self.unload()
        return ended

    # -- saving ------------------------------------------------------------------------------------------------------------

    def ask_replace(self, name: str) -> str:
        """A file with the chosen name is next to the old result: "replace" (the old one goes to the bin), "new" (another name) or
        "cancel"."""
        box = QMessageBox(QMessageBox.Icon.Question, "A file with that name exists",
                          f"{name} already exists. Replace it with the new file?\n\nThe old one goes to the "
                          f"{shell.TRASH_NAME}, so you can bring it back.", QMessageBox.StandardButton.NoButton, self)
        replace = box.addButton("Replace it", QMessageBox.ButtonRole.DestructiveRole)
        new = box.addButton("Keep both", QMessageBox.ButtonRole.AcceptRole)
        box.addButton(QMessageBox.StandardButton.Cancel)
        box.exec()
        clicked = box.clickedButton()
        return "replace" if clicked is replace else "new" if clicked is new else "cancel"

    def save(self) -> None:
        """Make the new result from the saved copies with the choices as they are."""
        job = self.job
        if job is None or self.busy:
            return
        p = self.plan()
        if p is None or not p.ok:
            return
        wanted = self.wanted()
        folder = redo.result_dir(job)
        target = folder / wanted.output_name
        replace = None
        if target.exists():
            answer = self.ask_replace(wanted.output_name)
            if answer == "cancel":
                return
            replace = target if answer == "replace" else None
        env = self._env
        worker = _Worker(lambda cancel, progress: env.run(job, wanted, progress=progress, cancel=cancel, replace=replace, trash=env.trash), self)
        worker.progress.connect(lambda f: (self._save_bar.setValue(int(f * 1000)), self._ed.busy(True, f)))
        worker.done.connect(self._saved)
        worker.failed.connect(self._save_failed)
        worker.stopped.connect(lambda: self._save_over("Stopped. Nothing was saved."))
        self._save_thread = worker
        self._save_bar.setValue(0)
        self._save_bar.show()
        self._stop.show()
        self._save.setText("Saving...")
        self.banner.hide()
        self._ed.busy(True, 0.0)
        self._refresh_all()
        worker.start()

    def _finish_thread(self) -> None:
        thread, self._save_thread = self._save_thread, None
        if thread is not None:
            thread.wait()
            thread.deleteLater()
        self._save_bar.hide()
        self._stop.hide()
        self._save.setText("Save as a new file")
        self._ed.busy(False)

    def _stop_save(self) -> None:
        if self._save_thread is not None:
            self._save_thread.cancel.set()
            self._stop.setEnabled(False)

    def _save_over(self, text: str, kind: str = "info") -> None:
        self._finish_thread()
        self._stop.setEnabled(True)
        self.banner.show_message(text, kind)
        self._refresh_all()

    def _save_failed(self, message: str) -> None:
        self._save_over(message, "error")

    def _saved(self, result: engine.JobResult) -> None:
        self._finish_thread()
        self._stop.setEnabled(True)
        entry, job = self.entry, self.job
        # The job file was changed by the thread: read it again.
        fresh = copies.JobCopies.load(job.dir)
        self.job = fresh if fresh is not None else job
        made = result.made or {}
        paths = [str(p) for p in result.paths]
        settings = {**entry.settings, "leveling": made.get("leveling", True), "loudness": made.get("loudness", "podcast"),
                    "subtitles": made.get("subtitles", "off"), "rebuild_voice": bool(made.get("rebuilt")),
                    "strength_db": made.get("atten_lim_db")}
        settings.pop("picked_strength", None)
        settings.pop("picked_rebuild", None)
        if made.get("parts"):
            settings["parts"] = True
        else:
            settings.pop("parts", None)
        self._store.set_job_result(entry.id, paths, settings, self.job.dir.name, self.job.bytes())
        self.entry = self._store.get(entry.id)
        self._check = self.job.check_original()
        self.banner.show_message(f"Saved {result.output.name}" + (f" ({result.notice})" if result.notice else "") + ".", "info")
        self._building = True
        try:
            self._fill_results()
            self._fill_copies()
            self._name.setText(redo.suggest_name(self.job, _source_of(self.job)))
            self._sync_ending()
            self._fill_meta()
            self._status.setText("Done")
            set_property(self._status, "state", "done")
        finally:
            self._building = False
        self._ed.saved()
        self._refresh_all()
        self.changed.emit()
        if not self._ed.active and self._prepared is None and self._prepare_thread is None and not self._slide_problem():
            self._start_preparing()  # a step that ran now made the copy the slider needs

    def save_subtitles_only(self) -> None:
        job = self.job
        kind = self._subtitles.currentData()
        if job is None or kind not in redo.TEXT_KINDS or self.busy:
            return
        last = redo.last_result(job)
        folder = redo.result_dir(job)
        stem = engine.split_name(last.name)[0] if last is not None else job.original_path.stem + "-cleaned"
        try:
            path = redo.save_subtitles(job, kind, folder, stem)
        except (MediaError, OSError) as e:
            self.banner.show_message(str(e), "error")
            return
        job.add_save([path], {**job.made, "subtitles": kind})
        self._store.set_job_result(self.entry.id, [*self.entry.outputs, str(path)], self.entry.settings, job.dir.name, job.bytes())
        self.entry = self._store.get(self.entry.id)
        self.banner.show_message(f"Saved {path.name}.", "info")
        self._fill_results()
        self.changed.emit()

    def _choose_file(self) -> str:
        path, _ = QFileDialog.getOpenFileName(self, "Find the original file", str(self.job.original_path.parent) if self.job else "")
        return path

    def _find_original(self) -> None:
        if self.job is None:
            return
        path = self._choose_file()
        if not path:
            return
        check = self.job.relocate(Path(path))
        if not check.ok:
            self.banner.show_message(check.text, "warn")
            return
        self._store.set_source(self.entry.id, str(Path(path)))
        self.entry = self._store.get(self.entry.id)
        self._check = self.job.check_original()
        self.banner.show_message("Found it. This job can be changed and saved again.", "info")
        self._fill_copies()
        if not self._ed.show():
            self._start_preparing()
        self._refresh_all()


def _clear(layout) -> None:
    while layout.count():
        item = layout.takeAt(0)
        if item.widget() is not None:
            item.widget().hide()  # (until it is deleted it would still be drawn)
            item.widget().deleteLater()
        elif item.layout() is not None:
            _clear(item.layout())


def _spent(seconds: float) -> str:
    minutes = round(seconds / 60)
    if minutes < 60:
        return f"{minutes} min" if minutes else ("0 min" if seconds == 0 else "< 1 min")
    return f"{minutes // 60} h {minutes % 60:02d} min"
