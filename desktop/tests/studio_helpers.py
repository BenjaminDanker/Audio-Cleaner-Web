"""Stand-ins for what the editor screen reads in the background (the file's facts, its waveform, what Automatic picks, a finished job's live sound), so
the screen's own behaviour can be tested without decoding anything; the real ones are tested in test_quick_look.py and test_editor_job.py."""
from __future__ import annotations

import os
import threading
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from acw_common import reco_model
from acw_common.timeline import Timeline
from audio_cleaner_desktop import copies, editor_job, engine, live_mix, redo
from audio_cleaner_desktop.gui.history import DONE, Entry

SR = 48_000


def info(seconds: float = 11.0, *, channels: int = 1, tracks: int = 1, video: bool = False, ext: str = ".wav") -> SimpleNamespace:
    return SimpleNamespace(duration=seconds, has_video=video, audio_channels=channels, audio_tracks=tracks, ext=ext, video_codec="h264" if video else None,
                           video_width=1280 if video else 0, video_height=720 if video else 0)


def choices(strength: str = "balanced", noise: str = "light", rebuild: bool = False) -> reco_model.Choices:
    return reco_model.Choices(strength=strength, atten_lim_db=reco_model.STRENGTH_DB[strength], noise=noise, heavy=noise == "heavy", rebuild=rebuild,
                              rebuild_wanted=rebuild, rebuild_reason="echoey" if rebuild else None, leveling=True, no_speech=False, noise_db=-20.0,
                              reasons=("There is some noise, so it gets Balanced.",))


class Fakes:
    """What the background work of a Doc is replaced with: each can be made slow (to see the screen stay usable) and each says what it was asked."""

    def __init__(self, seconds: float = 11.0, **facts):
        self.facts = dict(facts)
        self.seconds = seconds
        self.peaks_delay = 0.0
        self.decide_delay = 0.0
        self.probe_delay = 0.0
        self.session_delay = 0.0
        self.decide_result = choices()
        self.calls: list[str] = []
        self.gate: threading.Event | None = None

    def probe(self, path, ff=None):
        self.calls.append("probe")
        time.sleep(self.probe_delay)
        return info(self.seconds, **self.facts)

    def peaks(self, ff, path, probed, *, partial=None, **_):
        self.calls.append("peaks")
        n = 200
        for step in range(1, 5):
            if self.peaks_delay:
                time.sleep(self.peaks_delay / 4)
            if partial is not None:
                partial([30, 80, 50, 100][: 1 + step % 4] * (n // 4) + [0] * (n - (n // 4) * (1 + step % 4)), step / 4)
        return [30, 80, 50, 100] * (n // 4), self.seconds

    def decide(self, path, probed, options, *, threads=2, cancel=None, voice_model=False):
        self.calls.append("decide")
        end = time.monotonic() + self.decide_delay
        while time.monotonic() < end:
            if cancel is not None and cancel.wait(0.01):
                from audio_cleaner_desktop import engine

                raise engine.Cancelled()
        return self.decide_result if options.decides else None

    def install(self, window) -> "Fakes":
        env = window.doc_env
        env.probe, env.peaks, env.decide, env.auto_suggest = self.probe, self.peaks, self.decide, False
        return self




N = 10 * SR


def make_folder(tmp_path):
    d = tmp_path / "media"
    d.mkdir()
    (d / "talk.wav").write_bytes(os.urandom(200_000))
    (d / "talk-cleaned.wav").write_bytes(b"first result")
    return SimpleNamespace(dir=d, original=d / "talk.wav", result=d / "talk-cleaned.wav")


def add_job(store, folder, *, name="talk.wav", blocks=True, legacy=False, rebuilt=False, transcript=False, suggestions=None, seconds=10.0, status=DONE,
            saves=True, finished_at=1000.0, settings=None, made=None):
    """A job as the engine leaves it: it keeps blocks (stand-in bytes), or the old whole copies (`legacy`: a noise copy, with `rebuilt` a rebuilt-voice
    copy instead), a transcript, or nothing; listed in `store`. Returns (the job, its entry's id)."""
    job = copies.JobCopies.create(copies.new_job_dir(store.jobs_root), folder.original, copies.fingerprint(folder.original),
                                  {"duration": seconds, "has_video": False, "ext": ".wav", "channels": 1, "audio_tracks": 1})
    job.made.update({"atten_lim_db": 24.0, "rebuilt": False, "leveling": True, "loudness": "podcast", "subtitles": "off", "language": "auto",
                     "noise": "light", **(made or {})})
    if blocks:
        (job.dir / copies.BLOCKS_DIR).mkdir()
        (job.dir / copies.BLOCKS_DIR / "a.flac").write_bytes(b"b" * 100)
        job.set_blocks(N, False)
        job.editor["peaks"] = [30, 80, 50, 100] * 100
        job.editor["facts"] = {"n": N, "channels": 1, "plan": [0], "spans": {"0": [[0, N]]}, "rebuild_mid": False, "reference": False}
        if suggestions is not None:
            job.editor["suggestions"] = {"seconds": seconds, "whole": {}, "items": suggestions}
    if legacy:
        job.data["samples"] = N
        if rebuilt:
            (job.dir / "rebuilt.flac").write_bytes(b"r" * 3000)
            job.put_audio(copies.REBUILT, {"file": "rebuilt.flac", "channels": [0], "scale": 1.0, "samples": N})
            job.made.update({"atten_lim_db": None, "rebuilt": True})
        else:
            (job.dir / "noise.flac").write_bytes(b"n" * 500)
            job.put_audio(copies.NOISE, {"file": "noise.flac", "channels": [0], "scale": 1.0, "samples": N}, plan=[0], profile="light")
    if transcript:
        from acw_common import captions

        job.put_transcript([captions.Segment(0.0, 1.0, "hello there", [captions.Word(0.0, 1.0, "hello there")])], "en")
    if saves:
        job.add_save([folder.result], dict(job.made))
    job.save()
    eid = store.add(Entry(name, str(folder.original), seconds, False, settings or {"strength": "balanced", "leveling": True, "loudness": "podcast"}, 3.0,
                          [str(folder.result)] if saves else [], status, "" if status == DONE else "Saving the cleaned audio failed.", finished_at=finished_at,
                          job_dir=job.dir.name, copies_bytes=job.bytes()))
    return job, eid


class FakeSession:
    """What editor_session.Session gives the job view, without decoding anything: a live mix over made-up layers, a quote that says what a
    heavy remover or Voice rebuild would cost, suggestions that are found when asked."""

    made: list["FakeSession"] = []

    def __init__(self, job, seconds=10.0):
        n = int(seconds * SR) if seconds <= 60 else N
        rng = np.random.default_rng(1)
        src = (0.1 * rng.standard_normal((n, 1))).astype(np.float32)
        layers = live_mix.Layers(src, [0])
        layers.array("light")[:] = src * 0.25
        layers.mark("light", 0, n)
        self.job, self.seconds, self.echo_available, self.closed = job, seconds, False, False
        self.refreshed: list[Timeline] = []
        self.gate: threading.Event | None = None
        self.found = [{"id": "s0", "start": 3.0, "end": 5.0, "kind": "clipping", "rebuild": True, "noise": None,
                       "reasons": ["sounds distorted here (clipped)"], "state": "open"}]
        self.mix = live_mix.LiveMix(layers, editor_job.current_timeline(job, seconds))
        FakeSession.made.append(self)

    def refresh(self, tl):
        self.refreshed.append(tl.copy())
        self.mix.set_timeline(tl)

    def quote(self, tl):
        heavy = any(p.settings["remover"] == "heavy" for p in tl.layers["noise"]) or bool(tl.layers["rebuild"])
        return editor_job.Quote(seconds=41.0, steps={"heavy": 1}, misses=1) if heavy else editor_job.Quote()

    def suggest(self, *, cancel=None, progress=None):
        if progress:
            progress(0.5)
        if self.gate is not None:
            for _ in range(3000):  # (until it is let go or stopped, at most 30 s: a failed test must not leave it spinning)
                if cancel.wait(0.01) or self.gate.is_set():
                    break
            if cancel.is_set():
                raise engine.Cancelled()
        found = {"seconds": self.seconds, "whole": {}, "items": self.found}
        editor_job.keep_suggestions(self.job, found)
        return found

    def close(self):
        self.closed = True
        self.mix.close()


SESSION_DELAY = [0.0]  # how long opening the live sound takes (a test that wants to see the screen before it)


def open_session(job, *, cancel=None, progress=None, scratch_parent=None, threads=2):
    if SESSION_DELAY[0]:
        end = time.monotonic() + SESSION_DELAY[0]
        while time.monotonic() < end:
            if cancel is not None and cancel.wait(0.01):
                raise engine.Cancelled()
            if progress:
                progress(0.5)
    if not job.has_blocks:  # an old job: its copies become blocks
        (job.dir / copies.BLOCKS_DIR).mkdir(exist_ok=True)
        (job.dir / copies.BLOCKS_DIR / "m.flac").write_bytes(b"m" * 50)
        job.set_blocks(N, False)
        job.drop(copies.NOISE)
        job.editor["peaks"] = [50] * 100
    return FakeSession(job, job.samples / SR if job.samples else 10.0)


class FakeRun:
    """redo.run, stood in for: writes the new file next to the old one and records what it was asked."""

    def __init__(self):
        self.calls = []
        self.gate = None
        self.fail = None

    def __call__(self, job, wanted, *, progress=None, cancel=None, replace=None, trash=None, out_dir=None):
        self.calls.append(SimpleNamespace(wanted=wanted, replace=replace, trash=trash))
        if progress:
            progress(0.5)
        if self.gate is not None:
            self.gate.wait(10)
        if cancel is not None and cancel.is_set():
            raise engine.Cancelled()
        if self.fail:
            raise engine.MediaError(self.fail)
        out = redo.result_dir(job) / wanted.output_name
        if out.exists() and replace is None:
            out = out.with_name(out.stem + " (2)" + out.suffix)
        out.write_bytes(b"new result")
        files = [out]
        subs = None
        if wanted.subtitles in redo.TEXT_KINDS:
            subs = out.with_suffix("." + wanted.subtitles)
            subs.write_text("1\n00:00:00,000 --> 00:00:01,000\nhello there\n")
            files.append(subs)
        seconds = editor_job.seconds_of(job)
        tl = Timeline.from_json(wanted.timeline, seconds=seconds) if wanted.timeline else editor_job.current_timeline(job, seconds)
        made = {**editor_job.made_from_timeline(tl, echo=False), "subtitles": wanted.subtitles, "language": "auto"}
        job.made.update(made)
        editor_job.remember(job, tl, {})
        job.add_save(files, made)
        return engine.JobResult(out, seconds, False, subs, made=made, kept_dir=job.dir)


def shut(w) -> None:
    w._closing = True
    w.queue.shutdown()
    w._show(None)
    for item in list(w._items.values()):
        if item.doc is not None:
            item.doc.dispose()
    w.history.close()
    w._pool.waitForDone()


class QuickRunner:
    """A run that finishes at once and writes its result next to the original (no model, no copies: a result a job opened later can only listen to)."""

    def __init__(self):
        self.calls = []
        self.gate: threading.Event | None = None

    def __call__(self, path, options, *, cancel, progress, out_dir=None, output_name=None, timeline=None, **_):
        self.calls.append({"options": options, "timeline": timeline, "output_name": output_name})
        progress(0.5)
        if self.gate is not None:
            for _ in range(3000):  # (at most 30 s, as above)
                if self.gate.wait(0.01):
                    break
                if cancel.is_set():
                    raise engine.Cancelled()
        out = Path(out_dir or path.parent) / (output_name or path.stem + "-cleaned.wav")
        out.write_bytes(b"cleaned")
        made = editor_job.made_from_timeline(Timeline.from_json(timeline, seconds=11.0), echo=False) if timeline else None
        return engine.JobResult(out, 11.0, False, made=made)




def windows_showing() -> list:
    from PySide6.QtWidgets import QApplication, QDialog, QMainWindow

    return [w for w in QApplication.topLevelWidgets() if w.isVisible() and isinstance(w, (QMainWindow, QDialog))]


class TopLevels:
    """Catches every top-level window that is shown while it is on: through the Show event (a window that comes and goes inside one event is caught
    too) and a poll every few milliseconds (the screen's own timers run in the same loop). `seen` keeps what each one was (class, name, size, the
    chain of its parents, its flags and text), in the order they showed."""

    def __init__(self, main):
        from PySide6.QtCore import QTimer
        from PySide6.QtWidgets import QApplication

        self.main, self.seen, self._open = main, [], {}
        self._app = QApplication.instance()
        self._timer = QTimer(main, interval=2)
        self._timer.timeout.connect(self.poll)

    def start(self) -> "TopLevels":
        self._app.installEventFilter(self_filter(self))
        self._timer.start()
        return self

    def stop(self) -> None:
        self._timer.stop()
        self._app.removeEventFilter(self._filter)

    def record(self, w) -> None:
        self._open[id(w)] = w
        chain, p = [], w.parent()
        while p is not None:
            chain.append(f"{type(p).__name__}({p.objectName()})")
            p = p.parent()
        self.seen.append({"widget": w, "cls": type(w).__name__, "name": w.objectName(), "size": (w.width(), w.height()), "parents": chain,
                          "flags": int(w.windowFlags().value), "text": self._text_of(w), "title": w.windowTitle()})

    @staticmethod
    def _text_of(w) -> str:
        """What it says: its own text, or that of the first label inside it (a banner's message)."""
        from PySide6.QtWidgets import QLabel

        if callable(getattr(w, "text", None)):
            return str(w.text())
        label = w.findChild(QLabel)
        return label.text() if label is not None else ""

    def poll(self) -> None:
        visible = {id(w): w for w in self._app.topLevelWidgets() if w.isVisible() and w.isWindow()}
        for key in [k for k in self._open if k not in visible]:
            del self._open[key]  # (hidden again: its next show counts again)
        for key, w in visible.items():
            if key not in self._open:
                self.record(w)

    def mark(self) -> int:
        return len(self.seen)

    def strangers(self, since: int = 0, allow: tuple = ()) -> list[str]:
        """What showed since `mark` that is neither the main window nor one of the classes in `allow`, described."""
        return [f"{r['cls']} name={r['name']!r} size={r['size']} parents={r['parents']} flags={r['flags']:#x} text={r['text'][:70]!r} title={r['title']!r}"
                for r in self.seen[since:] if r["widget"] is not self.main and r["cls"] not in allow]


def self_filter(watch):
    """The application-wide event filter of a TopLevels (kept on it, so it is not collected)."""
    from PySide6.QtCore import QEvent, QObject

    class _Filter(QObject):
        def eventFilter(self, obj, event):  # noqa: N802
            if event.type() == QEvent.Type.Show and obj.isWidgetType() and obj.isWindow():
                watch.record(obj)
            return False

    watch._filter = _Filter(watch.main)
    return watch._filter


def add_file(window, tmp_path, name="talk.wav", size=2000) -> Path:
    path = tmp_path / name
    path.write_bytes(b"x" * size)
    window.add_paths([path])
    return path


def ready(window) -> None:
    """Wait until the file in the editor is read, its waveform is there and Automatic has listened."""
    from PySide6.QtWidgets import QApplication

    from conftest import wait_until

    doc = window.doc
    assert wait_until(lambda: doc.model is not None and doc.wave == "ready" and not doc.listening, 15)
    QApplication.processEvents()


def settle(w) -> None:
    """The screen redraws a moment after something changed (and the live sound follows an edit after a short quiet)."""
    from PySide6.QtWidgets import QApplication

    from conftest import wait_until

    QApplication.processEvents()
    wait_until(lambda: not w.studio._refresh_timer.isActive() and not w.studio._live_timer.isActive(), 3)
    QApplication.processEvents()


def resized(window, width: int, height: int, until=lambda: True, timeout: float = 5.0) -> None:
    """Resize the window and wait until the screen has taken it over: the window's own size and whatever `until()` says the layout does with it.
    On some platforms (Windows) the new size reaches the widgets inside a few events later, not by the next processEvents."""
    from PySide6.QtWidgets import QApplication

    from conftest import wait_until

    window.resize(width, height)
    taken = wait_until(lambda: window.width() == width and window.height() == height and bool(until()), timeout)
    assert taken, f"the window did not take {width}x{height} (it is {window.width()}x{window.height()}) or the screen did not follow within {timeout} s"
    QApplication.processEvents()
