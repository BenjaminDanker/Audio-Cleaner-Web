"""The editor in the windows: the job view's "Fix a moment" (the player with its live sound, the strip, the panel, the bar, Save, suggestions that
are found, applied, undone and kept) and "Choose parts" before a file is cleaned (the same editor in plan mode, its timeline going to the
queue and to the engine). The engine, the models and the sound card are stood in for (fake_media.py); the real ones are in test_editor_job.py."""
from __future__ import annotations

import os
import threading
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from PySide6.QtCore import QSettings
from PySide6.QtWidgets import QApplication

from acw_common.timeline import Timeline
from audio_cleaner_desktop import copies, editor_job, editor_session, engine, live_mix, redo
from audio_cleaner_desktop.gui import job_preview, main_window, plan_dialog, player
from audio_cleaner_desktop.gui.history import DONE, Entry, HistoryStore
from audio_cleaner_desktop.gui.job_view import JobEnv, JobView
from audio_cleaner_desktop.gui.main_window import MainWindow
from audio_cleaner_desktop.gui.theme import ThemeWatcher

import fake_media
from conftest import wait_until
from fake_media import FakeSink

SR = 48_000
N = 10 * SR


@pytest.fixture
def media(monkeypatch):
    FakeSink.made.clear()
    monkeypatch.setattr(player, "_multimedia", fake_media.namespace())
    monkeypatch.setattr(job_preview.JobPreview, "sink_factory", fake_media.sink_factory)


@pytest.fixture
def store(tmp_path):
    s = HistoryStore(tmp_path / "history.db")
    yield s
    s.close()


@pytest.fixture
def folder(tmp_path):
    d = tmp_path / "media"
    d.mkdir()
    (d / "talk.wav").write_bytes(os.urandom(200_000))
    (d / "talk-cleaned.wav").write_bytes(b"first result")
    return SimpleNamespace(dir=d, original=d / "talk.wav", result=d / "talk-cleaned.wav")


def add_job(store, folder, *, blocks=True, legacy=False, suggestions=None, seconds=10.0):
    """A job as the engine leaves it: it keeps blocks (stand-in bytes), or an old noise copy (legacy), or nothing."""
    job = copies.JobCopies.create(copies.new_job_dir(store.jobs_root), folder.original, copies.fingerprint(folder.original),
                                  {"duration": seconds, "has_video": False, "ext": ".wav", "channels": 1, "audio_tracks": 1})
    job.made.update({"atten_lim_db": 24.0, "rebuilt": False, "leveling": True, "loudness": "podcast", "subtitles": "off", "language": "auto",
                     "noise": "light"})
    if blocks:
        (job.dir / copies.BLOCKS_DIR).mkdir()
        (job.dir / copies.BLOCKS_DIR / "a.flac").write_bytes(b"b" * 100)
        job.set_blocks(N, False)
        job.editor["peaks"] = [30, 80, 50, 100] * 100
        if suggestions is not None:
            job.editor["suggestions"] = {"seconds": seconds, "whole": {}, "items": suggestions}
    if legacy:
        job.data["samples"] = N
        (job.dir / "noise.flac").write_bytes(b"n" * 500)
        job.put_audio(copies.NOISE, {"file": "noise.flac", "channels": [0], "scale": 1.0, "samples": N}, plan=[0], profile="light")
    job.add_save([folder.result], dict(job.made))
    job.save()
    eid = store.add(Entry("talk.wav", str(folder.original), seconds, False, {"strength": "balanced", "leveling": True, "loudness": "podcast"}, 3.0,
                          [str(folder.result)], DONE, "", finished_at=1000.0, job_dir=job.dir.name, copies_bytes=job.bytes()))
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
            while not cancel.wait(0.01) and not self.gate.is_set():
                pass
            if cancel.is_set():
                raise engine.Cancelled()
        found = {"seconds": self.seconds, "whole": {}, "items": self.found}
        editor_job.keep_suggestions(self.job, found)
        return found

    def close(self):
        self.closed = True
        self.mix.close()


def open_session(job, *, cancel=None, progress=None, scratch_parent=None, threads=2):
    if not job.has_blocks:  # an old job: its copies become blocks
        (job.dir / copies.BLOCKS_DIR).mkdir(exist_ok=True)
        (job.dir / copies.BLOCKS_DIR / "m.flac").write_bytes(b"m" * 50)
        job.set_blocks(N, False)
        job.drop(copies.NOISE)
        job.editor["peaks"] = [50] * 100
    return FakeSession(job, job.samples / SR if job.samples else 10.0)


class FakeRun:
    def __init__(self):
        self.calls = []
        self.gate = None

    def __call__(self, job, wanted, *, progress=None, cancel=None, replace=None, trash=None, out_dir=None):
        self.calls.append(wanted)
        if progress:
            progress(0.5)
        if self.gate is not None:
            self.gate.wait(10)
        if cancel is not None and cancel.is_set():
            raise engine.Cancelled()
        out = redo.result_dir(job) / wanted.output_name
        if out.exists():
            out = out.with_name(out.stem + " (2)" + out.suffix)
        out.write_bytes(b"new result")
        tl = Timeline.from_json(wanted.timeline, seconds=job.samples / SR)
        made = {**editor_job.made_from_timeline(tl, echo=False), "subtitles": "off", "language": "auto"}
        job.made.update(made)
        editor_job.remember(job, tl, {})
        job.add_save([out], made)
        return engine.JobResult(out, 10.0, False, made=made, kept_dir=job.dir)


@pytest.fixture
def run():
    return FakeRun()


@pytest.fixture
def env(run):
    return JobEnv(voice_ready=lambda: True, speech_ready=lambda: True, noise_ready=lambda: True, run=run, trash=lambda p: False, open_session=open_session)


@pytest.fixture
def view(qapp, store, env, media):
    ThemeWatcher(qapp)
    FakeSession.made.clear()
    v = JobView(store, env)
    v.resize(1180, 1400)
    v.show()
    yield v
    v.shutdown()
    v.close()


def open_job(view, eid):
    assert view.show_entry(eid)
    QApplication.processEvents()  # (a widget added to a window that is showing is shown a moment later)
    return view._ed


def ready(view, eid):
    ed = open_job(view, eid)
    assert wait_until(lambda: ed.editor is not None, 10)
    QApplication.processEvents()
    return ed


# ---------------------------------------------------------------------------------------------------- the job view

def test_a_job_that_keeps_blocks_opens_with_the_editor_and_its_live_sound(view, store, folder):
    job, eid = add_job(store, folder)
    ed = ready(view, eid)
    session = FakeSession.made[-1]
    assert ed.card.isVisible() and not view._listen_card.isVisible() and view._player_in_editor
    # the old controls are gone: the editor is the one way to change the sound; the Save card has the name, the type and the subtitles only
    assert not any(hasattr(view, name) for name in ("_slider", "_name_buttons", "_strength_label", "_rebuild", "_leveling", "_loudness", "_slide_note"))
    assert not hasattr(ed, "_setup") and view._change_card.isVisible() and view._name.isVisible() and view._subtitles.isVisible()
    assert not view._save.isVisibleTo(view)  # once the editor is open its own Save button saves
    assert view.preview is not None and view.preview.live and view.preview._stream is session.mix and ed.editor.slot.indexOf(view.preview) >= 0
    m = ed.model
    assert m.done == editor_job.baseline_timeline(job, 10.0) and m.strength_of(0, 10) == "balanced" and not m.dirty
    assert ed.editor.strip.peaks[:4] == [30, 80, 50, 100] and ed.editor.bar_title.text() == "No changes yet" and not ed.editor.save_button.isEnabled()
    assert ed.editor.chips_row.isVisible() and not ed.editor._chips["echo"].isVisible()


def test_the_playhead_follows_the_player_and_the_strip_moves_it(view, store, folder):
    _, eid = add_job(store, folder)
    ed = ready(view, eid)
    pane = view.preview
    pane._player.setPosition(4500)
    assert ed.editor.strip.playhead == pytest.approx(4.5, abs=0.01)
    ed.editor.strip.seekRequested.emit(7.0)
    assert pane._player.position() == 7000 and ed.editor.strip.playhead == 7.0


def test_the_hear_chips_change_the_live_sound_and_not_the_timeline(view, store, folder):
    _, eid = add_job(store, folder)
    ed = ready(view, eid)
    mix = FakeSession.made[-1].mix
    ed.editor._chips["noise"].click()
    assert not mix.hear["noise"] and mix.hear["leveling"] and not ed.model.dirty
    gain = 10 ** (mix.level_gains_db()[0][2] / 20)  # (Leveling is still heard: the recording at its loudness)
    assert np.allclose(mix.read(SR, 50), mix.layers.src[SR:SR + 50] * gain, atol=1e-6)
    ed.editor._chips["noise"].click()
    ed.editor._chips["leveling"].click()
    assert mix.hear == {"echo": True, "noise": True, "rebuild": True, "leveling": False}
    share = 10 ** (-24.0 / 20)  # the job's strength (Balanced) is a blend of the remover's sound and the recording
    assert np.allclose(mix.read(SR, 50), (1 - share) * mix.layers.arrays["light"][SR:SR + 50] + share * mix.layers.src[SR:SR + 50], atol=1e-6)


def test_a_change_is_heard_at_once_quoted_in_the_bar_and_saved_as_a_new_file_from_the_timeline(view, store, folder, run):
    job, eid = add_job(store, folder)
    ed = ready(view, eid)
    session = FakeSession.made[-1]
    m, editor = ed.model, ed.editor
    m.select(3, 5)
    m.set_strength("max")
    assert wait_until(lambda: session.refreshed and editor.bar_title.text() == "Changes: nothing to process", 5)  # a strength is a blend of blocks
    assert editor.save_button.isEnabled() and session.refreshed[-1].layers["noise"][1].settings["strength_db"] == 100.0
    session.mix.set_hear("leveling", False)
    assert np.allclose(session.mix.read(4 * SR, 20), session.mix.layers.arrays["light"][4 * SR:4 * SR + 20], atol=1e-6)  # heard now
    m.set_remover("heavy")
    assert wait_until(lambda: editor.bar_title.text() == "Process changes · about 40 s", 5)
    assert "noise removal" in editor.bar_detail.text() or "Makes the parts" in editor.bar_detail.text()
    run.gate = threading.Event()
    editor.save_button.click()
    assert wait_until(lambda: view.busy and editor.progress.isVisible(), 5)
    assert editor.stop_button.isVisible() and not editor.panel.isEnabled() and editor.save_button.text() == "Saving..."
    run.gate.set()
    assert wait_until(lambda: not view.busy, 10)
    wanted = run.calls[-1]
    assert wanted.timeline["layers"]["noise"][1] == {"start": 3.0, "end": 5.0, "strength_db": 100.0, "remover": "heavy"}
    assert wanted.output_name == "talk-cleaned (2).wav" and (folder.dir / "talk-cleaned (2).wav").is_file()
    assert editor.panel.isEnabled() and not editor.progress.isVisible()
    assert not m.dirty and m.done == m.draft and ed.model.selection is None  # the result is the draft now
    assert copies.JobCopies.load(job.dir).editor["timeline"]["layers"]["noise"][1]["remover"] == "heavy" and editor.bar_title.text() == "No changes yet"
    assert [p for p, _ in view._result_paths()][0].name == "talk-cleaned (2).wav" and folder.result.read_bytes() == b"first result"


def test_save_can_be_stopped_and_nothing_is_saved(view, store, folder, run):
    _, eid = add_job(store, folder)
    ed = ready(view, eid)
    ed.model.select(3, 5)
    ed.model.set_strength("gentle")
    assert wait_until(lambda: ed.editor.save_button.isEnabled(), 5)
    run.gate = threading.Event()
    ed.editor.save_button.click()
    assert wait_until(lambda: view.busy, 5)
    ed.editor.stop_button.click()
    run.gate.set()
    assert wait_until(lambda: not view.busy, 10)
    assert not (folder.dir / "talk-cleaned (2).wav").exists() and ed.model.dirty and ed.editor.save_button.isEnabled()


def test_a_name_that_cannot_be_used_stops_the_save_and_says_why(view, store, folder):
    _, eid = add_job(store, folder)
    ed = ready(view, eid)
    ed.model.select(3, 5)
    ed.model.set_strength("gentle")
    assert wait_until(lambda: ed.editor.save_button.isEnabled(), 5)
    view._name.setText("bad/name.wav")
    assert wait_until(lambda: not ed.editor.save_button.isEnabled(), 5)
    assert ed.editor.save_button.toolTip() != ""


def test_suggestions_are_found_when_none_were_kept_applied_undone_dismissed_and_kept_with_the_job(view, store, folder):
    job, eid = add_job(store, folder)
    ed = ready(view, eid)
    assert wait_until(lambda: [s.id for s in ed.model.suggestions] == ["s0"], 10) and ed._find_thread is None
    job = copies.JobCopies.load(job.dir)
    assert editor_job.stored_suggestions(job)["items"][0]["id"] == "s0"  # kept with the job
    editor = ed.editor
    editor.set_playhead(1.0)
    assert editor.strip.pin_positions() == [("s0", 4.0)]
    ed.model.apply("s0")
    editor.suggestionsChanged.emit()
    assert ed.model.switch_of("rebuild", 3, 5) is True and ed.model.dirty
    assert wait_until(lambda: ed.quote is not None and not ed.quote.nothing, 5)
    assert copies.JobCopies.load(job.dir).editor["suggestions"]["items"][0]["state"] == "applied"
    ed.model.undo("s0")
    editor.suggestionsChanged.emit()
    ed.model.dismiss("s0")
    editor.suggestionsChanged.emit()
    assert copies.JobCopies.load(job.dir).editor["suggestions"]["items"][0]["state"] == "dismissed" and not editor.strip.pin_positions()
    # the next time the job is opened they are not looked for again, and the dismissed one stays dismissed
    view.unload()
    n = len(FakeSession.made)
    ed2 = ready(view, eid)
    assert len(FakeSession.made) == n + 1 and ed2._find_thread is None and [s.state for s in ed2.model.suggestions] == ["dismissed"]


def test_looking_for_suggestions_can_be_stopped_and_a_long_recording_waits_for_the_button(view, store, folder, monkeypatch):
    job, eid = add_job(store, folder)
    monkeypatch.setattr(open_session, "__defaults__", None, raising=False)
    original = FakeSession.__init__

    def slow(self, job, seconds=10.0):
        original(self, job, seconds)
        self.gate = threading.Event()

    monkeypatch.setattr(FakeSession, "__init__", slow)
    ed = ready(view, eid)
    assert wait_until(lambda: ed._find_thread is not None, 5)
    assert ed.editor._finding is not None
    ed.editor.stopRequested.emit()
    assert wait_until(lambda: ed._find_thread is None, 10) and ed.model.suggestions == [] and ed.editor._finding is None
    assert "Stopped" in ed.editor._find_text
    # a long recording: nothing is looked for by itself
    long_job, long_id = add_job(store, folder, seconds=editor_job.SUGGEST_AUTO_SECONDS + 60.0)
    long_job.data["samples"] = int((editor_job.SUGGEST_AUTO_SECONDS + 60.0) * SR)
    long_job.save()
    monkeypatch.setattr(FakeSession, "__init__", original)
    view.unload()
    ed = ready(view, long_id)
    assert ed._find_thread is None and "long recording" in ed.editor._find_text and ed.editor._can_find


def test_an_old_job_with_saved_copies_opens_in_the_editor_at_once_and_has_no_setup_button(view, store, folder):
    job, eid = add_job(store, folder, blocks=False, legacy=True)
    ed = ready(view, eid)
    assert not hasattr(ed, "_setup") and ed.card.isVisible() and view._player_in_editor and not view._listen_card.isVisible()
    job = copies.JobCopies.load(job.dir)
    assert job.has_blocks and not job.has(copies.NOISE)  # its copy became blocks on the way in (editor_job.make_ready)


def test_a_job_with_no_saved_copies_at_all_shows_its_player_in_the_listen_card_and_no_editor(view, store, folder):
    bare, bare_id = add_job(store, folder, blocks=False)  # (a folder with nothing in it is no saved copy)
    assert view.show_entry(bare_id) and view.job is None and view.preview is not None and view._listen_card.isVisible()
    eid = store.add(Entry("old.wav", str(folder.original), 5.0, False, {}, 1.0, [str(folder.result)], DONE, finished_at=5.0))
    assert view.show_entry(eid)
    QApplication.processEvents()
    ed = view._ed
    assert view.job is None and not ed.card.isVisible() and ed.editor is None and not FakeSession.made
    assert view.preview is not None and view._listen_card.isVisible() and not view._player_in_editor and not view.preview.live
    assert not view._change_card.isVisible()


def test_when_the_editor_cannot_open_the_save_card_keeps_its_own_button(view, store, folder):
    job, eid = add_job(store, folder)
    folder.original.write_bytes(b"something else entirely")
    ed = open_job(view, eid)
    assert ed.card.isVisible() and "Fixing a moment needs the original" in ed._note.text() and ed.editor is None and not FakeSession.made
    assert view._change_card.isVisible() and view._save.isVisibleTo(view) and not view._save.isEnabled()


def test_leaving_a_job_closes_its_session_and_gives_the_player_back(view, store, folder):
    _, eid = add_job(store, folder)
    other = store.add(Entry("old.wav", str(folder.original), 5.0, False, {}, 1.0, [str(folder.result)], DONE, finished_at=5.0))  # (no saved copies)
    ready(view, eid)
    session = FakeSession.made[-1]
    view.show_entry(other)
    assert session.closed and view._ed.editor is None and not view._player_in_editor and not view._ed.card.isVisible()
    assert view._listen_card.isVisible()
    view.show_entry(eid)
    assert wait_until(lambda: view._ed.editor is not None, 10)
    view.unload()
    assert FakeSession.made[-1].closed


# ---------------------------------------------------------------------------------------------------- choose parts

class FakePlan:
    """editor_session.PlanSession, without decoding: a file of ten seconds, suggestions when asked."""

    closed = False

    def __init__(self):
        self.seconds, self.peaks, self.echo_available = 10.0, [40, 90] * 100, False
        self.baselines = []

    def baseline(self, options, *, cancel=None):
        self.baselines.append(options)
        return editor_session.baseline_for(options, self.seconds, echo=False)

    def quote(self, tl):
        return editor_job.Quote(seconds=12.0, steps={"light": 1}, misses=1)

    def suggest(self, *, cancel=None, progress=None):
        if progress:
            progress(0.5)
        return {"seconds": 10.0, "whole": {}, "items": [{"id": "s0", "start": 2.0, "end": 4.0, "kind": "part", "rebuild": False,
                                                        "noise": {"strength_db": 40.0, "remover": "light"}, "reasons": ["is noisier"], "state": "open"}]}

    def close(self):
        self.closed = True


def open_plan(path, *, cancel=None, scratch_parent=None, threads=2):
    return FakePlan()


@pytest.fixture
def window(qapp, tmp_path, monkeypatch, media):
    calls = []

    def runner(path, options, *, cancel, progress, out_dir=None, threads=None, output_name=None, keep_dir=None, **extra):
        calls.append({"path": path, "options": options, **extra})
        out = Path(out_dir or path.parent) / (output_name or "x-cleaned.wav")
        out.write_bytes(b"result")
        return engine.JobResult(out, 10.0, False)

    monkeypatch.setattr(main_window.resources, "check", lambda: [])
    monkeypatch.setattr(engine, "probe", lambda path, ff=None: SimpleNamespace(
        duration=10.0, has_video=False, audio_channels=1, ext=path.suffix, video_codec=None, audio_tracks=1))
    ThemeWatcher(qapp)
    w = MainWindow([], settings=QSettings(str(tmp_path / "s.ini"), QSettings.Format.IniFormat), history_path=tmp_path / "h.db", runner=runner)
    w._open_plan = open_plan
    w.show()
    w.calls = calls
    yield w
    w.queue.shutdown()
    w.history_page.shutdown()
    w.history.close()
    w._pool.waitForDone()


def drive(monkeypatch, steps):
    """The dialog is modal: stand in for exec() with what the user does in it."""
    def fake_exec(self):
        assert wait_until(lambda: self.editor is not None, 10)
        steps(self)
        return 1

    monkeypatch.setattr(plan_dialog.PlanDialog, "exec", fake_exec)


def add_file(window, tmp_path, name="talk.wav"):
    (tmp_path / name).write_bytes(b"x")
    window.add_paths([tmp_path / name])
    item = next(i for i in window._items.values() if i.path.name == name)
    assert wait_until(lambda: item.probed, 10)
    return item


def test_a_waiting_file_has_a_choose_parts_button_and_the_chosen_parts_go_to_the_engine(window, tmp_path, monkeypatch):
    item = add_file(window, tmp_path)
    row = window.view.row(item.id)
    assert row._parts.isVisible() and row._parts.text() == "Choose parts..." and item.timeline is None

    def choose(dialog):
        assert dialog.windowTitle() == "Choose parts: talk.wav" and dialog.model.plan_mode and dialog.model.seconds == 10.0
        dialog.model.select(2, 5)
        dialog.model.set_strength("gentle")
        dialog.model.choose()
        dialog._draft_changed()
        assert dialog.editor.bar_title.text() == "Chosen parts · about 10 s"
        dialog.accept()

    drive(monkeypatch, choose)
    window.choose_parts(item.id)
    assert item.timeline is not None and item.parts_note == "only the chosen parts (3 s of 0:10)" and item.plan_state["only"] is True
    assert [(p["start"], p["end"], p["strength_db"]) for p in item.timeline["layers"]["noise"]] == [(2.0, 5.0, 12.0)]
    assert row._parts.text() == "Change the chosen parts..." and "only the chosen parts (3 s of 0:10)" in row._meta_text()
    window.start()
    assert wait_until(lambda: item.state == "done", 10)
    assert window.calls[-1]["timeline"] == item.timeline
    [entry] = window.history.search()
    assert entry.settings["parts"] is True


def test_the_dialog_opens_again_where_it_was_left_and_the_whole_file_can_be_chosen_again(window, tmp_path, monkeypatch):
    item = add_file(window, tmp_path)
    drive(monkeypatch, lambda d: (d.model.select(2, 5), d.model.choose(), d.accept()))
    window.choose_parts(item.id)
    assert item.timeline is not None
    seen = {}

    def look(dialog):
        seen["chosen"], seen["only"] = list(dialog.model.chosen), dialog.model.only
        dialog._use_whole()

    drive(monkeypatch, look)
    window.choose_parts(item.id)
    assert seen == {"chosen": [(2.0, 5.0)], "only": True} and item.timeline is None and item.plan_state is None and item.parts_note == ""


def test_a_cancelled_dialog_changes_nothing_and_a_file_in_line_cannot_be_given_parts(window, tmp_path, monkeypatch):
    item = add_file(window, tmp_path)
    drive(monkeypatch, lambda d: (d.model.select(2, 5), d.model.choose(), d.reject()))
    window.choose_parts(item.id)
    assert item.timeline is None
    window._queued.add(item.id)
    drive(monkeypatch, lambda d: pytest.fail("a file that is in line has no parts to choose"))
    window.choose_parts(item.id)


def test_settings_that_differ_per_part_are_passed_even_without_only_parts_and_suggestions_are_found_first(window, tmp_path, monkeypatch):
    item = add_file(window, tmp_path)

    def work(dialog):
        dialog.find()
        assert wait_until(lambda: dialog._find_thread is None and dialog.model.suggestions, 10)
        assert [s.id for s in dialog.model.open_suggestions()] == ["s0"] and dialog.timeline() is None  # (nothing is chosen yet)
        dialog.model.apply("s0")
        assert dialog.timeline()["layers"]["noise"][0]["strength_db"] == 100.0 or len(dialog.timeline()["layers"]["noise"]) == 3
        dialog.accept()

    drive(monkeypatch, work)
    window.choose_parts(item.id)
    noise = item.timeline["layers"]["noise"]
    assert [(p["start"], p["end"], p["strength_db"]) for p in noise] == [(0.0, 2.0, 100.0), (2.0, 4.0, 40.0), (4.0, 10.0, 100.0)]
    assert item.parts_note == "some parts have their own settings" and item.plan_state["suggestions"][0]["state"] == "applied"


def test_the_dialog_reads_the_file_with_the_settings_it_will_be_cleaned_with(qapp, tmp_path, media):
    plan = FakePlan()
    dialog = plan_dialog.PlanDialog(tmp_path / "talk.wav", "talk.wav", False, engine.JobOptions(strength="strong", noise="heavy", leveling=False),
                                    open_plan=lambda *a, **k: plan)
    assert wait_until(lambda: dialog.editor is not None, 10)
    tl = dialog.model.draft
    assert [(p.settings["strength_db"], p.settings["remover"]) for p in tl.layers["noise"]] == [(40.0, "heavy")] and not tl.layers["leveling"]
    assert not dialog.editor.chips_row.isVisible() and dialog.timeline() is None and dialog.state() is None
    dialog.reject()
    assert plan.closed


def test_a_file_that_cannot_be_read_says_so_in_the_dialog(qapp, tmp_path, media):
    def broken(*a, **k):
        raise engine.MediaError("This file was not found.")

    dialog = plan_dialog.PlanDialog(tmp_path / "x.wav", "x.wav", False, engine.JobOptions(), open_plan=broken)
    assert wait_until(lambda: dialog._open_thread is None, 10)
    assert dialog._status.text() == "This file was not found." and dialog.editor is None and not dialog._ok.isEnabled()
    dialog.reject()


def test_the_settings_say_which_one_to_pick_in_plain_words(qapp, tmp_path):
    from audio_cleaner_desktop import labels
    from audio_cleaner_desktop.gui.preferences import Preferences
    from audio_cleaner_desktop.gui.settings_panel import SettingsPanel

    panel = SettingsPanel(Preferences(QSettings(str(tmp_path / "p.ini"), QSettings.Format.IniFormat)))
    panel.show()
    assert panel._which.text() == "Which one should I pick?" and not panel._which_text.isVisible()
    panel._which.setChecked(True)
    assert panel._which_text.isVisible() and panel._which_text.text() == labels.WHICH_TEXT
    text = labels.WHICH_TEXT
    for phrase in ("keeps your voice exactly as recorded", "never changes a word", "laughs, shouts, whispers and singing stay as they were",
                   "any length", "game sounds", "fixes echo, a muffled or phone-quality voice and distortion", "can sound slightly too clean",
                   "keeps only speech", "may come out oddly", "can change a word, so listen before you publish", "slower", "many languages"):
        assert phrase.lower() in text.lower(), phrase
    assert "polish" not in text.lower() and "model" not in text.lower()
    assert "the voice was rebuilt because it sounded echoey" in labels.rebuild_reason_text("echoey") and "keeps only speech" in labels.rebuild_reason_text("echoey")
