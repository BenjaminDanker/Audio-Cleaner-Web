"""The job view's "Fix a moment" part: opens a job's editor session on a worker thread (decoding the original, laying out what the block cache
has for listening), builds the editor around the player, keeps the quote and the suggestions up to date, and hands the changes to the job
view's Save. The job view owns one of these; this class never saves anything itself."""
from __future__ import annotations

import logging

from PySide6.QtWidgets import QProgressBar, QVBoxLayout

from .. import editor_job, editor_session
from ..editor_model import EditModel
from .part_editor import PartEditor

log = logging.getLogger(__name__)


class JobEditor:
    def __init__(self, view) -> None:  # noqa: ANN001 - the JobView
        from .job_view import _card, _small  # (job_view imports this module only when it is built)

        self.v = view
        self._orig_slot = view._player_slot
        self.session: editor_session.Session | None = None
        self.editor: PartEditor | None = None
        self.model: EditModel | None = None
        self._open_thread = None
        self._find_thread = None
        self._gen = 0
        self.quote: editor_job.Quote | None = None
        self.card, layout = _card("Fix a moment")
        self.card.setVisible(False)
        self._intro = _small("Pick a moment or a part of the recording and change just that: the noise reduction, Rebuild the voice, "
                             "Leveling. You hear the changes while it plays; only what is new needs processing.")
        layout.addWidget(self._intro)
        self._note = _small()
        layout.addWidget(self._note)
        self._bar = QProgressBar()
        self._bar.setRange(0, 0)
        self._bar.setTextVisible(False)
        self._bar.setMaximumHeight(8)
        self._bar.hide()
        layout.addWidget(self._bar)
        self.holder = QVBoxLayout()
        layout.addLayout(self.holder)

    # -- state ------------------------------------------------------------------------------------------------------------
    @property
    def active(self) -> bool:
        """The job's sound is changed here (every job with saved copies)."""
        return self.v.job is not None

    def show(self) -> None:
        """A job was opened: open it in the editor when its original is there."""
        job = self.v.job
        self.card.setVisible(job is not None)
        self._note.setText("")
        if job is None:
            return
        if not self.v._check.ok:
            self._note.setText(f"{self.v._check.text} Fixing a moment needs the original.")
            self.v._show_player()
            return
        self._open()

    def _open(self) -> None:
        from .job_view import _Worker

        v, job = self.v, self.v.job
        if job is None or self._open_thread is not None:
            return
        self._note.setText("Getting the sound ready to hear your changes..." if job.has_blocks else "Getting this job ready...")
        self._bar.show()
        self._gen += 1
        gen, env, scratch = self._gen, v._env, v._env.scratch()
        worker = _Worker(lambda cancel, progress: env.open_session(job, cancel=cancel, progress=progress, scratch_parent=scratch, threads=2), v)
        worker.done.connect(lambda session, g=gen: self._ready(session, g))
        worker.failed.connect(lambda message, g=gen: self._failed(message, g))
        worker.stopped.connect(lambda g=gen: self._finish_open(g))
        self._open_thread = worker
        worker.start()

    def _finish_open(self, gen: int) -> None:
        thread, self._open_thread = self._open_thread, None
        if thread is not None:
            thread.wait()
            thread.deleteLater()
        if gen == self._gen:
            self._bar.hide()

    def _failed(self, message: str, gen: int) -> None:
        self._finish_open(gen)
        if gen == self._gen:
            self._note.setText(message)
            self.v._show_player()
            self.v._refresh_all()

    def _ready(self, session: editor_session.Session, gen: int) -> None:
        self._finish_open(gen)
        v, job = self.v, self.v.job
        if gen != self._gen or job is None:
            session.close()
            return
        self.session = session
        self._note.setText("")
        seconds = session.seconds
        done = editor_job.current_timeline(job, seconds)
        model = EditModel(seconds, done, done, echo_available=session.echo_available, loudness=editor_job.timeline_loudness(done))
        stored = editor_job.stored_suggestions(job)
        model.set_suggestions(stored["items"] if stored else None)
        editor = PartEditor(model, job.editor.get("peaks") or [])
        self.model, self.editor = model, editor
        editor.set_rebuild_available(v._env.voice_ready() or bool(done.layers["rebuild"]))
        self.holder.addWidget(editor)
        v._player_slot = editor.slot
        v._player_in_editor = True
        v._show_player(session.mix)
        if v.preview is not None:
            v.preview.positionMoved.connect(lambda ms: editor.set_playhead(ms / 1000.0))
            editor.seekRequested.connect(lambda t: v.preview.seek_ms(int(t * 1000)))
            editor.playPartRequested.connect(lambda a, b: v.preview.play_from(int(a * 1000)))
        editor.hearChanged.connect(lambda layer, on: session.mix.set_hear(layer, on))
        editor.set_hear_available(v.preview is not None and v.preview.live)
        editor.draftChanged.connect(self.draft_changed)
        editor.suggestionsChanged.connect(self._keep_suggestions)
        editor.saveRequested.connect(v.save)
        editor.stopRequested.connect(self._stop)
        editor.findRequested.connect(self.find)
        v._refresh_all()
        self.draft_changed()
        if not stored:
            if seconds <= editor_job.SUGGEST_AUTO_SECONDS:
                self.find()
            else:
                editor.set_finding(None, "This is a long recording, so looking for moments that need other settings is left to you.", True)

    # -- the draft ---------------------------------------------------------------------------------------------------------
    def draft_changed(self) -> None:
        if self.model is None or self.session is None or self.editor is None:
            return
        try:
            self.session.refresh(self.model.draft)
            self.quote = self.session.quote(self.model.draft)
        except Exception:  # noqa: BLE001 - a quote or a refresh must never break editing
            log.warning("could not refresh the editor", exc_info=True)
            self.quote = None
        mix = self.session.mix
        self.editor.strip.set_missing({layer: mix.missing(layer) for layer in ("noise", "rebuild", "echo")} if mix else {})
        needs = mix is not None and any(mix.missing(layer) for layer in ("noise", "rebuild", "echo"))
        self.editor.set_heard_note("Parts that still need processing are not heard with their change yet: press Save to make them."
                                   if needs else "")
        self.refresh_bar()

    def refresh_bar(self) -> None:
        editor = self.editor
        if editor is None or self.model is None:
            return
        editor.show_quote(self.quote)
        p = self.v.plan()
        if p is not None and not p.ok:
            editor.save_button.setEnabled(False)
            editor.save_button.setToolTip("; ".join(p.problems))
        else:
            editor.save_button.setToolTip("")
            if self.v.busy:
                editor.save_button.setEnabled(False)

    def timeline_json(self) -> dict | None:
        return self.model.effective().to_json() if self.model is not None else None

    # -- saving -----------------------------------------------------------------------------------------------------------
    def busy(self, busy: bool, fraction: float | None = None) -> None:
        if self.editor is not None:
            self.editor.set_busy(busy, fraction)
            if not busy:
                self.refresh_bar()

    def saved(self) -> None:
        """A save finished: the job's latest result is what the draft says now, and the blocks it made can be heard."""
        job, model, session = self.v.job, self.model, self.session
        if job is None or model is None or session is None:
            return
        done = editor_job.current_timeline(job, session.seconds)
        model.done = done.copy()
        model.draft = done.copy()
        model.selection = None
        model._changed()
        self.draft_changed()

    # -- suggestions ---------------------------------------------------------------------------------------------------------
    def _keep_suggestions(self) -> None:
        if self.model is None or self.v.job is None:
            return
        stored = editor_job.stored_suggestions(self.v.job) or {}
        editor_job.keep_suggestions(self.v.job, {**stored, "items": self.model.suggestions_json()})
        self.draft_changed()

    def find(self) -> None:
        from .job_view import _Worker

        if self.session is None or self.editor is None or self._find_thread is not None:
            return
        session, editor, gen = self.session, self.editor, self._gen
        editor.set_finding(0.0)
        worker = _Worker(lambda cancel, progress: session.suggest(cancel=cancel, progress=progress), self.v)
        worker.progress.connect(lambda f: gen == self._gen and editor.set_finding(f))
        worker.done.connect(lambda found, g=gen: self._found(found, g))
        worker.failed.connect(lambda message, g=gen: self._not_found(message, g))
        worker.stopped.connect(lambda g=gen: self._not_found("Stopped looking.", g))
        self._find_thread = worker
        worker.start()

    def _end_find(self) -> None:
        thread, self._find_thread = self._find_thread, None
        if thread is not None:
            thread.wait()
            thread.deleteLater()

    def _found(self, found: dict, gen: int) -> None:
        self._end_find()
        if gen != self._gen or self.model is None or self.editor is None:
            return
        self.model.set_suggestions(found["items"])
        self.editor.set_finding(None, "" if found["items"] else "No moments found that need other settings.", True)

    def _not_found(self, message: str, gen: int) -> None:
        self._end_find()
        if gen == self._gen and self.editor is not None:
            self.editor.set_finding(None, message, True)

    def _stop(self) -> None:
        if self._find_thread is not None:
            self._find_thread.cancel.set()
        else:
            self.v._stop_save()

    # -- letting go -----------------------------------------------------------------------------------------------------------
    def unload(self) -> None:
        self._gen += 1
        for name in ("_find_thread", "_open_thread"):
            thread = getattr(self, name)
            if thread is not None:
                thread.cancel.set()
                thread.wait()
                thread.deleteLater()
                setattr(self, name, None)
        self._bar.hide()
        editor, self.editor, self.model = self.editor, None, None
        if editor is not None:
            editor.hide()
            self.holder.removeWidget(editor)
            editor.deleteLater()
        if self.v._player_in_editor:  # the player went with the editor
            self.v.preview = None
            self.v._player_slot = self._orig_slot
            self.v._player_in_editor = False
        session, self.session = self.session, None
        if session is not None:
            session.close()
        self.card.setVisible(False)
        self.quote = None
