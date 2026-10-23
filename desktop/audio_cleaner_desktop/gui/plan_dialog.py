"""Choose parts: the per-part editor in plan mode, opened for a file that is waiting to be cleaned. The file is read (for the waveform and the
suggestions), "Find moments to fix" listens for parts that other settings would suit, and the parts to process (or the settings of each part)
are chosen. Nothing is processed here and nothing can be heard but the recording as it is; "Use these parts" hands a timeline to the queue
(`timeline()`), which cleans the file with it (engine.process(..., timeline=...))."""
from __future__ import annotations

import logging
from collections.abc import Callable
from pathlib import Path

from PySide6.QtWidgets import QDialog, QHBoxLayout, QLabel, QProgressBar, QPushButton, QVBoxLayout

from acw_common.timeline import Timeline, TimelineError

from .. import editor_job, editor_session, engine, resources
from ..editor_model import EditModel, clock
from .job_preview import JobPreview
from .part_editor import PartEditor

log = logging.getLogger(__name__)


class PlanDialog(QDialog):
    def __init__(self, path: Path, name: str, has_video: bool, options: engine.JobOptions, *, state: dict | None = None,
                 scratch: Path | None = None, parent=None, open_plan: Callable = editor_session.PlanSession.open):  # noqa: ANN001
        super().__init__(parent)
        from .job_view import _Worker

        self.setWindowTitle(f"Choose parts: {name}")
        self.resize(1080, 760)
        self._path, self._name, self._has_video, self._options = Path(path), name, has_video, options
        self._state = state
        self.session: editor_session.PlanSession | None = None
        self.model: EditModel | None = None
        self.editor: PartEditor | None = None
        self.preview: JobPreview | None = None
        self._baseline: Timeline | None = None
        self._find_thread: _Worker | None = None
        self._open_thread: _Worker | None = None
        self._cancelled = False
        self._gen = 0

        self._layout = QVBoxLayout(self)
        self._intro = QLabel("Process the whole file as it is, or choose parts: give a part its own settings, or process only the parts you "
                             "choose.")
        self._intro.setObjectName("Muted")
        self._intro.setWordWrap(True)
        self._layout.addWidget(self._intro)
        self._status = QLabel("Reading the file...")
        self._bar = QProgressBar()
        self._bar.setRange(0, 0)
        self._bar.setTextVisible(False)
        self._layout.addWidget(self._status)
        self._layout.addWidget(self._bar)
        self._holder = QVBoxLayout()
        self._layout.addLayout(self._holder, 1)
        buttons = QHBoxLayout()
        self._whole = QPushButton("Use the whole file")
        self._whole.setToolTip("Forget the parts: the file is cleaned with the settings as they are")
        self._whole.clicked.connect(self._use_whole)
        self._cancel = QPushButton("Cancel")
        self._cancel.clicked.connect(self.reject)
        self._ok = QPushButton("Use these parts")
        self._ok.setObjectName("Primary")
        self._ok.setEnabled(False)
        self._ok.clicked.connect(self.accept)
        buttons.addWidget(self._whole)
        buttons.addStretch(1)
        buttons.addWidget(self._cancel)
        buttons.addWidget(self._ok)
        self._layout.addLayout(buttons)
        self.cleared = False  # "Use the whole file" was chosen

        worker = _Worker(lambda cancel, progress: self._read(open_plan, cancel, scratch), self)
        worker.done.connect(self._ready)
        worker.failed.connect(self._failed)
        worker.stopped.connect(lambda: None)
        self._open_thread = worker
        worker.start()

    def _read(self, open_plan, cancel, scratch):
        session = open_plan(self._path, cancel=cancel, scratch_parent=scratch, threads=2)
        try:
            baseline = session.baseline(self._options, cancel=cancel)
        except BaseException:
            session.close()
            raise
        return session, baseline

    def _end_open(self) -> None:
        thread, self._open_thread = self._open_thread, None
        if thread is not None:
            thread.wait()  # (the thread is a child of the dialog and goes with it: no deleteLater, the dialog may be gone first)
        self._bar.hide()

    def _failed(self, message: str) -> None:
        self._end_open()
        self._status.setText(message)

    def _ready(self, result) -> None:  # noqa: ANN001
        self._end_open()
        session, baseline = result
        if self._cancelled:
            session.close()
            return
        self.session, self._baseline = session, baseline
        self._status.hide()
        st = self._state or {}
        draft = baseline
        if st.get("draft"):
            try:
                draft = Timeline.from_json(st["draft"], seconds=session.seconds)
            except TimelineError:
                draft = baseline
        model = EditModel(session.seconds, draft, None, echo_available=session.echo_available, plan_mode=True,
                          loudness=editor_job.timeline_loudness(baseline))
        model.chosen = [tuple(c) for c in st.get("chosen", [])]
        model.only = bool(st.get("only")) and bool(model.chosen)
        if st.get("suggestions") is not None:
            model.set_suggestions(st["suggestions"])
        editor = PartEditor(model, session.peaks)
        editor.set_rebuild_available(resources.voice_model_dir() is not None)
        editor.set_finding(None, "", True)
        self.model, self.editor = model, editor
        self._holder.addWidget(editor)
        self.preview = JobPreview()
        editor.slot.addWidget(self.preview)
        self.preview.show_files(0, self._path, None, self._has_video, self._name)
        self.preview.positionMoved.connect(lambda ms: editor.set_playhead(ms / 1000.0))
        editor.seekRequested.connect(lambda t: self.preview.seek_ms(int(t * 1000)))
        editor.playPartRequested.connect(lambda a, b: self.preview.play_from(int(a * 1000)))
        editor.draftChanged.connect(self._draft_changed)
        editor.findRequested.connect(self.find)
        editor.stopRequested.connect(self._stop_find)
        self._ok.setEnabled(True)
        self._draft_changed()

    def _draft_changed(self) -> None:
        if self.model is None or self.session is None or self.editor is None:
            return
        try:
            quote = self.session.quote(self.model.effective())
        except Exception:  # noqa: BLE001
            log.warning("could not quote the parts", exc_info=True)
            quote = None
        self.editor.show_quote(quote)

    # -- suggestions ---------------------------------------------------------------------------------------------------------
    def find(self) -> None:
        from .job_view import _Worker

        if self.session is None or self.editor is None or self._find_thread is not None:
            return
        session, editor = self.session, self.editor
        editor.set_finding(0.0)
        worker = _Worker(lambda cancel, progress: session.suggest(cancel=cancel, progress=progress), self)
        worker.progress.connect(editor.set_finding)
        worker.done.connect(self._found)
        worker.failed.connect(lambda m: self._not_found(m))
        worker.stopped.connect(lambda: self._not_found("Stopped looking."))
        self._find_thread = worker
        worker.start()

    def _end_find(self) -> None:
        thread, self._find_thread = self._find_thread, None
        if thread is not None:
            thread.wait()

    def _found(self, found: dict) -> None:
        self._end_find()
        if self.model is not None and self.editor is not None:
            self.model.set_suggestions(found["items"])
            self.editor.set_finding(None, "" if found["items"] else "No moments found that need other settings.", True)

    def _not_found(self, message: str) -> None:
        self._end_find()
        if self.editor is not None:
            self.editor.set_finding(None, message, True)

    def _stop_find(self) -> None:
        if self._find_thread is not None:
            self._find_thread.cancel.set()

    # -- the result ----------------------------------------------------------------------------------------------------------
    def timeline(self) -> dict | None:
        """The timeline to clean the file with: None when nothing was chosen (the file is cleaned with its settings, as it is)."""
        if self.cleared or self.model is None:
            return None
        m = self.model
        if m.only and m.chosen:
            return m.effective().to_json()
        if self._baseline is not None and m.draft != self._baseline:
            return m.draft.to_json()
        return None

    def state(self) -> dict | None:
        """What the dialog shows, to open it again where it was left (None when nothing was chosen)."""
        if self.cleared or self.model is None or self.timeline() is None:
            return None
        m = self.model
        return {"draft": m.draft.to_json(), "chosen": [list(c) for c in m.chosen], "only": m.only, "suggestions": m.suggestions_json()}

    def summary(self) -> str:
        """One line for the file's row: what is processed."""
        m = self.model
        if self.timeline() is None or m is None:
            return ""
        if m.only and m.chosen:
            return f"only the chosen parts ({m.processed_seconds():.0f} s of {clock(m.seconds)})"
        return "some parts have their own settings"

    def _use_whole(self) -> None:
        self.cleared = True
        self.accept()

    def done(self, result: int) -> None:  # noqa: N802 - Qt override
        self._cancelled = True
        for name in ("_find_thread", "_open_thread"):
            thread = getattr(self, name)
            if thread is not None:
                thread.cancel.set()
                thread.wait()
                setattr(self, name, None)
        if self.preview is not None:
            self.preview.stop()
        session, self.session = self.session, None
        if session is not None:
            session.close()
        super().done(result)
