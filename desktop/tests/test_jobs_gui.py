"""The Jobs tab (the old History page): the list (newest first, sorted, searched, paged), the job view opened from a row (saved copies and
the disk they take, the original's state, the editor it opens in, "Save as a new file"), deleting (copies and entry only), entries made
before the copies existed, and the window recording what a job kept. Model runs, the editor's session and the sound card are stood in for
(fake_media.py, and test_editor_gui.py's FakeSession); the real engine is in test_redo.py."""
from __future__ import annotations

import os
import threading
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from PySide6.QtCore import QSettings
from PySide6.QtWidgets import QApplication

from acw_common import captions
from acw_common.timeline import Timeline
from audio_cleaner_desktop import copies, editor_job, engine, redo
from audio_cleaner_desktop.gui import history_view, job_preview, main_window, player
from audio_cleaner_desktop.gui.history import DONE, FAILED, Entry, HistoryStore
from audio_cleaner_desktop.gui.job_view import JobEnv
from audio_cleaner_desktop.gui.main_window import MainWindow
from audio_cleaner_desktop.gui.theme import ThemeWatcher

import fake_media
from conftest import wait_until
from fake_media import FakeSink
from test_editor_gui import FakeSession, open_session

N = 96000  # two seconds


@pytest.fixture
def media(monkeypatch):
    """QtMultimedia and the sound card stood in for."""
    FakeSink.made.clear()
    monkeypatch.setattr(player, "_multimedia", fake_media.namespace())
    monkeypatch.setattr(job_preview.JobPreview, "sink_factory", fake_media.sink_factory)


class Folder:
    """A user's folder with an original and cleaned files in it."""

    def __init__(self, tmp_path):
        self.dir = tmp_path / "media"
        self.dir.mkdir()
        self.original = self.dir / "talk.wav"
        self.original.write_bytes(os.urandom(200_000))
        self.result = self.dir / "talk-cleaned.wav"
        self.result.write_bytes(b"first result")


@pytest.fixture
def folder(tmp_path):
    return Folder(tmp_path)


@pytest.fixture
def store(tmp_path):
    s = HistoryStore(tmp_path / "history.db")
    yield s
    s.close()


def add_job(store, folder, *, name="talk.wav", noise=True, rebuilt=False, transcript=False, made=None, status=DONE, finished_at=None,
            saves=True, settings=None):
    """A job as the engine leaves it (the copies are stand-in bytes: nothing here decodes them), listed."""
    job = copies.JobCopies.create(copies.new_job_dir(store.jobs_root), folder.original, copies.fingerprint(folder.original),
                                  {"duration": 2.0, "has_video": False, "ext": ".wav", "channels": 2, "audio_tracks": 1})
    job.data["samples"] = N
    if noise:
        (job.dir / "noise.flac").write_bytes(b"n" * 4000)
        job.put_audio(copies.NOISE, {"file": "noise.flac", "channels": [0, 1], "scale": 1.0, "samples": N}, plan=[0, 1], profile="light")
    if rebuilt:
        (job.dir / "rebuilt.flac").write_bytes(b"r" * 3000)
        job.put_audio(copies.REBUILT, {"file": "rebuilt.flac", "channels": [0, 1], "scale": 1.0, "samples": N})
    if transcript:
        job.put_transcript([captions.Segment(0.0, 1.0, "hello there", [captions.Word(0.0, 1.0, "hello there")])], "en")
    job.made.update({"atten_lim_db": None if rebuilt and not noise else 100.0, "rebuilt": rebuilt and not noise, "leveling": True,
                     "loudness": "podcast", "subtitles": "off", "language": "auto", "noise": "light", **(made or {})})
    if saves:
        job.add_save([folder.result], dict(job.made))
    job.save()
    entry_id = store.add(Entry(
        name, str(folder.original), 2.0, False, settings or {"strength": "max", "leveling": True, "loudness": "podcast"}, 3.0,
        [str(folder.result)] if saves else [], status, "" if status == DONE else "Saving the cleaned audio failed.",
        finished_at=finished_at or 1000.0, job_dir=job.dir.name, copies_bytes=job.bytes(),
    ))
    return job, entry_id


class FakeRun:
    """The redo, stood in for: writes the new file next to the old one and records what it was asked."""

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
        return engine.JobResult(out, 2.0, False, subs, made=made, kept_dir=job.dir)


@pytest.fixture
def run():
    return FakeRun()


@pytest.fixture
def env(run):
    return JobEnv(voice_ready=lambda: True, speech_ready=lambda: True, noise_ready=lambda: True, run=run, trash=lambda p: False,
                  open_session=open_session)


@pytest.fixture
def page(qapp, store, env, media):
    FakeSession.made.clear()
    ThemeWatcher(qapp)
    p = history_view.HistoryPage(store, env=env)
    p.confirm = lambda title, text: True
    p._stayed = lambda ids: None
    p.show()
    yield p
    p.shutdown()
    p.close()


def entry_ids(page):
    return [r.entry.id for r in page.findChildren(history_view._HistoryRow)]


def open_ready(page, entry_id):
    """Open a job and wait until its editor is up (the job view always opens it when the original is there)."""
    page.open_job(entry_id)
    view = page.view
    assert wait_until(lambda: view._ed.editor is not None, 10)
    QApplication.processEvents()
    return view


# ---------------------------------------------------------------------------------------------------- the list

def test_the_list_is_newest_first_by_default_and_can_be_sorted(page, store, folder):
    ids = [add_job(store, folder, name=n, finished_at=t)[1] for n, t in (("b.wav", 100.0), ("a.wav", 300.0), ("c.wav", 200.0))]
    page.refresh()
    assert entry_ids(page) == [ids[1], ids[2], ids[0]] and page.sort == "newest"
    names = lambda: [r.entry.name for r in page.findChildren(history_view._HistoryRow)]  # noqa: E731
    for key, expected in (("oldest", ["b.wav", "c.wav", "a.wav"]), ("name", ["a.wav", "b.wav", "c.wav"]), ("newest", ["a.wav", "c.wav", "b.wav"])):
        page._sort.setCurrentIndex(page._sort.findData(key))
        assert names() == expected


def test_the_list_is_searched_by_name_folder_and_settings_in_words(page, store, folder):
    add_job(store, folder, name="interview.wav", settings={"strength": "gentle", "leveling": True, "loudness": "podcast"}, finished_at=1.0)
    add_job(store, folder, name="lecture.wav", settings={"strength": "max", "leveling": False}, finished_at=2.0)
    for text, expected in (("interview", 1), ("media", 2), ("gentle", 1), ("leveled", 1), ("gentle podcast", 1), ("nothing", 0), ("", 2)):
        page._search.setText(text)
        page._search_changed()
        assert len(entry_ids(page)) == expected, text
    page._search.setText("nothing")
    page._search_changed()
    assert page._empty.text() == "Nothing matches your search."


def test_there_is_no_cap_the_list_is_paged_and_each_page_is_added_under_the_last(page, store, folder):
    for n in range(130):
        store.add(Entry(f"f{n}.wav", f"/a/f{n}.wav", 5.0, False, {}, 1.0, [], DONE, finished_at=1000.0 + n))
    page.refresh()
    assert len(entry_ids(page)) == 50 and page._count.text() == "Showing 50 of 130"
    first_row = page.findChildren(history_view._HistoryRow)[0]
    page._show_more()
    assert len(entry_ids(page)) == 100 and first_row in page.findChildren(history_view._HistoryRow)  # the first page was not redrawn
    page._show_more()
    assert len(entry_ids(page)) == 130 and not page._more.isVisible() and page._count.text() == "130 jobs"
    assert entry_ids(page)[0] == store.search(limit=1)[0].id and page._stats["files"].value.text() == "130"
    page.refresh()  # drawn again with as many as were shown
    assert len(entry_ids(page)) == 130


def test_the_page_shows_the_disk_of_each_job_and_the_total(page, store, folder):
    a, id_a = add_job(store, folder, finished_at=2.0)
    b, id_b = add_job(store, folder, finished_at=1.0)
    store.add(Entry("old.wav", "/a/old.wav", 5.0, False, {}, 1.0, [], DONE, finished_at=0.5))
    page.refresh()
    rows = {r.entry.id: r for r in page.findChildren(history_view._HistoryRow)}
    assert rows[id_a].disk.text() != "" and rows[id_b].disk.text() != "" and all(r.disk.text() == "" for i, r in rows.items() if i not in (id_a, id_b))
    assert page._stats["disk"].caption.text() == "Saved copies (2 jobs)"
    from audio_cleaner_desktop.gui.queue_view import format_size

    assert page._stats["disk"].value.text() == format_size(a.bytes() + b.bytes())


# ---------------------------------------------------------------------------------------------------- deleting

def test_deleting_a_job_removes_its_copies_and_entry_and_never_the_results_or_the_original(page, store, folder):
    job, eid = add_job(store, folder)
    other, other_id = add_job(store, folder, finished_at=2000.0)
    page.refresh()
    asked = []
    page.confirm = lambda title, text: (asked.append(text), False)[1]
    page.delete_entry(eid)
    assert job.dir.exists() and store.get(eid) is not None and "saved copies" in asked[0] and "not touched" in asked[0]  # declined: nothing happens
    page.confirm = lambda title, text: True
    page.delete_entry(eid)
    assert not job.dir.exists() and store.get(eid) is None and other.dir.exists()
    assert folder.original.exists() and folder.result.read_bytes() == b"first result"
    assert entry_ids(page) == [other_id]


def test_delete_all_removes_every_copy_and_entry_also_failed_and_old_ones(page, store, folder):
    a, _ = add_job(store, folder)
    b, _ = add_job(store, folder, status=FAILED, saves=False)
    store.add(Entry("old.wav", "/a/old.wav", 5.0, False, {}, 1.0, ["/a/old-cleaned.wav"], DONE))
    stray = copies.new_job_dir(store.jobs_root)  # copies nothing in the list points to
    (stray / "noise.flac").write_bytes(b"x")
    page.refresh()
    asked = []
    page.confirm = lambda title, text: (asked.append(text), True)[1]
    page._clear_all()
    assert "3 jobs" in asked[0] and "not touched" in asked[0]
    assert store.count() == 0 and not a.dir.exists() and not b.dir.exists() and not stray.exists()
    assert folder.original.exists() and folder.result.exists() and entry_ids(page) == []
    assert not page._clear.isEnabled()


def test_a_job_whose_copies_cannot_be_removed_stays_and_the_user_is_told(page, store, folder, monkeypatch):
    job, eid = add_job(store, folder)
    page.refresh()
    told = []
    page._stayed = told.append
    monkeypatch.setattr(copies, "delete_job_dir", lambda root, name: False)
    page.delete_entry(eid)
    assert told == [[eid]] and store.get(eid) is not None and job.dir.exists()


def test_deleting_the_job_that_is_open_closes_its_editor_and_stops_the_sound(page, store, folder):
    job, eid = add_job(store, folder)
    view = open_ready(page, eid)
    session = FakeSession.made[-1]
    assert page.viewing == eid and view._ed.session is session and view.preview.live
    page.delete_entry(eid)
    assert page.viewing is None and store.get(eid) is None and not job.dir.exists() and view._ed.session is None
    assert session.closed and view.preview is None  # the editor's session (its scratch files) and the sound went with it


# ---------------------------------------------------------------------------------------------------- old entries and failed jobs

def test_an_entry_made_before_the_copies_existed_still_shows_and_can_be_deleted(page, store, folder):
    eid = store.add(Entry("old.wav", str(folder.original), 5.0, False, {"strength": "gentle", "leveling": True}, 1.0, [str(folder.result)], DONE,
                          finished_at=5.0))
    page.refresh()
    assert entry_ids(page) == [eid]
    page.open_job(eid)
    view = page.view
    assert view.job is None and "no saved copies" in view._copies_text.text() and "can't be changed and saved again" in view._copies_text.text()
    assert not view._change_card.isVisible() and not view._ed.card.isVisible() and view._ed.editor is None  # no editor: nothing to start from
    assert view.preview is not None and view._listen_card.isVisible() and not view.preview.live  # it can still be listened to
    texts = [view._results_layout.itemAt(i).layout().itemAt(0).widget().text() for i in range(view._results_layout.count())]
    assert texts == [folder.result.name]
    page.delete_entry(eid)
    assert store.get(eid) is None and page.viewing is None and folder.result.exists()


def test_a_failed_job_with_copies_lists_them_and_can_still_be_saved(page, store, folder, run):
    job, eid = add_job(store, folder, status=FAILED, saves=False)
    page.refresh()
    view = open_ready(page, eid)
    assert view._status.text() == "Failed" and "Saving the cleaned audio failed." in view._meta.text()
    assert "No cleaned file was saved" in view._results_note.text() and view._change_card.isVisible() and view._ed.editor is not None
    view._name.setText("late.wav")
    assert view._save.isEnabled() and view._ed.editor.save_button.isEnabled() is not None
    view.save()
    assert wait_until(lambda: not view.busy)
    assert store.get(eid).status == DONE and store.get(eid).outputs == [str(folder.dir / "late.wav")]


def test_a_job_with_a_folder_that_has_no_usable_manifest_is_listed_like_an_old_one(page, store, folder):
    job, eid = add_job(store, folder)
    (job.dir / copies.MANIFEST).write_text("{broken")
    page.open_job(eid)
    assert page.view.job is None and not page.view._change_card.isVisible()


# ---------------------------------------------------------------------------------------------------- the job view

def test_the_job_view_lists_the_results_the_copies_and_the_disk_they_take(page, store, folder, env):
    env.open_session = lambda job, **kw: FakeSession(job, 2.0)  # (the editor opening in the background must not turn the copies into blocks here)
    job, eid = add_job(store, folder, transcript=True, rebuilt=True)
    page.open_job(eid)
    view = page.view
    assert page.viewing == eid and view._title.text() == "talk.wav" and view._status.text() == "Done"
    text = view._copies_text.text()
    assert "Noise removal at full strength" in text and "Rebuilt voice" in text and "Transcript" in text
    from audio_cleaner_desktop.gui.queue_view import format_size

    assert view._disk.text() == f"Disk used by this job: {format_size(job.bytes())}"
    assert "unchanged" in view._original_text.text() and not view._find.isVisible()
    assert folder.result.name in view._results_layout.itemAt(0).layout().itemAt(0).widget().text()
    assert "←" in view._back.text()
    page.view.closed.emit()
    assert page.viewing is None


def test_a_moved_original_makes_every_change_unavailable_and_it_can_be_found_again(page, store, folder):
    job, eid = add_job(store, folder)
    moved = folder.dir / "renamed.wav"
    folder.original.rename(moved)
    page.open_job(eid)
    view = page.view
    assert view._ed.editor is None and view._ed.session is None and not FakeSession.made  # the editor needs the original
    assert "not at" in view._ed._note.text() and "Fixing a moment needs the original" in view._ed._note.text()
    assert not view._save.isEnabled() and "needs the original" in view._save.toolTip() and view._save.isVisibleTo(view)
    assert "not at" in view._original_text.text() and view._find.isVisible()
    # pointing at the wrong file is refused
    wrong = folder.dir / "other.wav"
    wrong.write_bytes(os.urandom(200_000))
    view._choose_file = lambda: str(wrong)
    view._find_original()
    assert "not the same file" in view.banner._label.text() and not view._save.isEnabled()
    # the right one is accepted; the job can be changed again and its editor opens
    view._choose_file = lambda: str(moved)
    view._find_original()
    assert view._check.ok and store.get(eid).source == str(moved)
    assert wait_until(lambda: view._ed.editor is not None, 10)
    assert view._save.isEnabled() and not view._save.isVisibleTo(view)  # (the editor's own Save button saves now)


def test_a_changed_original_is_said_and_cannot_be_found_by_hand(page, store, folder):
    job, eid = add_job(store, folder)
    folder.original.write_bytes(b"something else entirely")
    page.open_job(eid)
    view = page.view
    assert view._check.state == "changed" and "changed" in view._original_text.text() and not view._find.isVisible()
    assert not view._save.isEnabled() and view._save.isVisibleTo(view)  # (no editor is open: the Save card keeps its own button)
    assert view._ed.editor is None and view._ed.session is None and "Fixing a moment needs the original" in view._ed._note.text()


def test_a_job_opens_in_the_editor_and_the_player_moves_into_it(page, store, folder):
    job, eid = add_job(store, folder)
    view = open_ready(page, eid)
    ed = view._ed
    assert view._player_in_editor and not view._listen_card.isVisible() and view.preview.live and ed.editor.slot.indexOf(view.preview) >= 0
    assert not view._save.isVisibleTo(view) and ed.editor.save_button.isVisibleTo(ed.editor)
    # none of the old controls: the strength slider, the strength buttons, Leveling, Loudness and Voice rebuild are the editor's now
    assert not any(hasattr(view, name) for name in ("_slider", "_name_buttons", "_strength_label", "_rebuild", "_leveling", "_loudness", "_slide_note"))
    assert not hasattr(ed, "_setup") and not hasattr(view, "_prepared") and not hasattr(view, "_prepare_thread")
    assert not hasattr(JobEnv(), "prepare") and not hasattr(JobEnv(), "live")


def test_an_old_job_with_a_saved_copy_opens_in_the_editor_at_once(page, store, folder):
    job, eid = add_job(store, folder, rebuilt=True)  # whole copies, no blocks: made before the block cache
    assert not job.has_blocks
    view = open_ready(page, eid)
    assert view._ed.editor is not None and not hasattr(view._ed, "_setup") and view._player_in_editor
    job = copies.JobCopies.load(job.dir)
    assert job.has_blocks and not job.has(copies.NOISE)  # (the open session made the job ready: editor_job.make_ready)


def test_a_job_that_rebuilt_the_voice_opens_with_the_rebuild_on_its_whole_length(page, store, folder):
    job, eid = add_job(store, folder, noise=False, rebuilt=True)
    view = open_ready(page, eid)
    done = view._ed.model.done
    assert done.layers["rebuild"] and not done.layers["noise"]
    assert view.plan() is not None and view._save.isEnabled()


def test_without_sound_support_the_editor_still_opens_and_the_change_can_be_saved(page, store, folder, run, monkeypatch):
    monkeypatch.setattr(player, "_multimedia", None)
    job, eid = add_job(store, folder)
    view = open_ready(page, eid)
    assert view.preview is None or not view.preview.live
    view._name.setText("quiet.wav")
    view.save()
    assert wait_until(lambda: not view.busy) and [c.wanted.output_name for c in run.calls] == ["quiet.wav"]


def test_voice_rebuild_without_its_model_cannot_be_saved(page, store, folder, env):
    env.voice_ready = lambda: False
    job, eid = add_job(store, folder, noise=False, rebuilt=True)  # its timeline rebuilds the voice and no rebuilt block is there
    view = open_ready(page, eid)
    assert not view._save.isEnabled() and "rebuilds the voice is not installed" in view._plan_text.text()
    assert not view._ed.editor.save_button.isEnabled()


def test_the_name_must_be_usable_and_the_type_box_follows_the_ending(page, store, folder):
    job, eid = add_job(store, folder)
    open_ready(page, eid)
    view = page.view
    assert view._name.text() == "talk-cleaned (2).wav" and view._ending.currentData() == ".wav"  # next to the old one, a free name
    view._ending.setCurrentIndex(view._ending.findData(".mp3"))
    view._ending_picked(view._ending.currentIndex())
    assert view._name.text() == "talk-cleaned (2).mp3"
    view._name.setText("bad/name.mp3")
    assert not view._save.isEnabled() and "can't contain" in view._plan_text.text()
    view._name.setText("noending")
    assert not view._save.isEnabled() and view._ending.currentIndex() == -1
    view._name.setText("")
    assert not view._save.isEnabled() and "Type a name" in view._plan_text.text()


# ---------------------------------------------------------------------------------------------------- change and save again

def test_save_runs_with_the_drafts_timeline_and_adds_a_new_result_next_to_the_old_one(page, store, folder, run):
    job, eid = add_job(store, folder)
    view = open_ready(page, eid)
    changed = []
    view.changed.connect(lambda: changed.append(1))
    ed = view._ed
    ed.model.select(0, ed.session.seconds)
    ed.model.set_strength("strong")  # the editor says what to do to which part; the Save card says how to save it
    view._name.setText("talk-v2.mp3")
    assert "talk-v2.mp3" in view._plan_text.text() and wait_until(lambda: ed.editor.save_button.isEnabled(), 5)
    ed.editor.save_button.click()
    assert view.busy and not view._back.isEnabled() and not view._delete.isEnabled() and view._stop.isVisibleTo(view)
    assert wait_until(lambda: not view.busy)
    [call] = run.calls
    assert call.wanted.output_name == "talk-v2.mp3" and call.wanted.subtitles == "off" and call.replace is None
    assert call.wanted.timeline["layers"]["noise"][0]["strength_db"] == 40.0  # Strong, from the draft
    new = folder.dir / "talk-v2.mp3"
    assert new.exists() and folder.result.read_bytes() == b"first result"  # the old result is as it was
    entry = store.get(eid)
    assert entry.outputs == [str(new)] and entry.settings["strength_db"] == 40.0 and entry.settings["leveling"] is True
    assert "Strong noise reduction" in entry.summary and entry.job_dir == job.dir.name
    assert "Saved talk-v2.mp3" in view.banner._label.text() and changed
    names = [view._results_layout.itemAt(i).layout().itemAt(0).widget().text() for i in range(view._results_layout.count())]
    assert names == ["talk-v2.mp3", "talk-cleaned.wav"]  # the new one first, the old one still listed
    assert view._name.text() == "talk-v2 (2).mp3" and view._save.isEnabled()  # the next name is free
    assert not ed.model.dirty and ed.model.done == ed.model.draft  # the result is the draft now
    # the list shows what it was made with now
    page.close_job()
    assert "Strong" in page.findChildren(history_view._HistoryRow)[0].findChildren(history_view.ElidedLabel)[1].text()


def test_a_name_that_is_taken_asks_and_replaces_only_when_the_user_says_so(page, store, folder, run):
    job, eid = add_job(store, folder)
    view = open_ready(page, eid)
    asked = []
    for answer in ("cancel", "new", "replace"):
        view.ask_replace = lambda name, a=answer: (asked.append(name), a)[1]
        view._name.setText("talk-cleaned.wav")  # taken: it is the first result
        view.save()
        assert wait_until(lambda: not view.busy)
    assert asked == ["talk-cleaned.wav"] * 3
    assert len(run.calls) == 2 and run.calls[0].replace is None and run.calls[1].replace == folder.result  # cancelled: nothing ran
    # a free name is never asked about
    view.ask_replace = lambda name: pytest.fail("asked")
    view._name.setText("free.wav")
    view.save()
    assert wait_until(lambda: not view.busy) and len(run.calls) == 3


def test_a_change_that_fails_says_why_and_changes_nothing(page, store, folder, run):
    job, eid = add_job(store, folder)
    view = open_ready(page, eid)
    run.fail = "The original no longer matches the saved copies."
    before = store.get(eid)
    view.save()
    assert wait_until(lambda: not view.busy)
    assert "no longer matches" in view.banner._label.text() and store.get(eid).outputs == before.outputs
    assert view._save.isEnabled() and not view._stop.isVisibleTo(view) and folder.result.exists()


def test_a_save_can_be_stopped(page, store, folder, run):
    job, eid = add_job(store, folder)
    view = open_ready(page, eid)
    run.gate = threading.Event()
    view.save()
    assert view.busy
    view._stop_save()
    run.gate.set()
    assert wait_until(lambda: not view.busy)
    assert "Stopped" in view.banner._label.text() and store.get(eid).outputs == [str(folder.result)] and len(job.saves) == 1


def test_a_job_cannot_be_deleted_or_left_while_a_change_is_being_saved(page, store, folder, run):
    job, eid = add_job(store, folder)
    view = open_ready(page, eid)
    run.gate = threading.Event()
    view.save()
    page.delete_entry(eid)
    assert store.get(eid) is not None and job.dir.exists()
    assert not view._back.isEnabled() and not view._delete.isEnabled()
    run.gate.set()
    assert wait_until(lambda: not view.busy)
    page.delete_entry(eid)
    assert store.get(eid) is None


def test_closing_the_window_stops_a_save_that_is_running(page, store, folder, run):
    job, eid = add_job(store, folder)
    view = open_ready(page, eid)
    run.gate = threading.Event()
    view.save()
    threading.Timer(0.2, run.gate.set).start()
    assert page.shutdown(5000) and not view.busy and view._ed.session is None


def test_saving_only_the_subtitles_needs_the_transcript_but_not_the_original(page, store, folder, run):
    job, eid = add_job(store, folder, transcript=True)
    folder.original.unlink()  # gone: every other change is unavailable
    page.open_job(eid)
    view = page.view
    assert not view._save.isEnabled() and view._save_subs.isVisibleTo(view) and not view._save_subs.isEnabled()  # no kind chosen yet
    view._subtitles.setCurrentIndex(view._subtitles.findData("srt"))
    assert view._save_subs.isEnabled()
    view.save_subtitles_only()
    srt = folder.dir / "talk-cleaned.srt"
    assert srt.exists() and "hello there" in srt.read_text() and run.calls == []
    assert str(srt) in store.get(eid).outputs and "Saved talk-cleaned.srt" in view.banner._label.text()


def test_without_a_transcript_there_is_no_button_for_subtitles_alone(page, store, folder):
    job, eid = add_job(store, folder)
    page.open_job(eid)
    assert not page.view._save_subs.isVisibleTo(page.view)


def test_subtitles_in_the_choices_are_part_of_the_save(page, store, folder, run):
    job, eid = add_job(store, folder, transcript=True)
    view = open_ready(page, eid)
    view._subtitles.setCurrentIndex(view._subtitles.findData("vtt"))
    assert "Listen to the original again" not in view._plan_text.text()  # the saved transcript is used
    view.save()
    assert wait_until(lambda: not view.busy)
    assert run.calls[0].wanted.subtitles == "vtt" and [Path(p).suffix for p in store.get(eid).outputs] == [".wav", ".vtt"]


# ---------------------------------------------------------------------------------------------------- the window

@pytest.fixture
def window(qapp, tmp_path, monkeypatch):
    made = {"keep_dirs": []}

    def runner(path, options, *, cancel, progress, out_dir=None, threads=None, output_name=None, keep_dir=None, **_):
        made["keep_dirs"].append(keep_dir)
        out = Path(out_dir or path.parent) / (output_name or "x-cleaned.wav")
        if path.name.startswith("stop"):
            while not cancel.wait(0.01):
                pass
            raise engine.Cancelled()
        if keep_dir is not None and not path.name.startswith(("nokeep", "hopeless")):
            Path(keep_dir).mkdir(parents=True, exist_ok=True)
            job = copies.JobCopies.create(Path(keep_dir), path, copies.fingerprint(path), {"duration": 5.0})
            (Path(keep_dir) / "noise.flac").write_bytes(b"n" * 777)
            job.put_audio(copies.NOISE, {"file": "noise.flac", "channels": [0], "scale": 1.0, "samples": 10}, plan=[0], profile="light")
        if path.name.startswith("fail"):
            raise engine.MediaError("Saving the cleaned audio failed.")
        if path.name.startswith("hopeless"):
            if keep_dir is not None:
                Path(keep_dir).mkdir(parents=True, exist_ok=True)  # an empty folder: it made nothing
            raise engine.MediaError("This file could not be read.")
        out.write_bytes(b"result")
        return engine.JobResult(out, 5.0, False, kept_dir=Path(keep_dir) if keep_dir is not None and not path.name.startswith("nokeep") else None)

    monkeypatch.setattr(main_window.resources, "check", lambda: [])
    monkeypatch.setattr(engine, "probe", lambda path, ff=None: SimpleNamespace(
        duration=5.0, has_video=False, audio_channels=1, ext=path.suffix, video_codec=None))
    ThemeWatcher(qapp)
    w = MainWindow([], settings=QSettings(str(tmp_path / "s.ini"), QSettings.Format.IniFormat), history_path=tmp_path / "h.db", runner=runner)
    w.show()
    w.made = made
    yield w
    w.queue.shutdown()
    w.history_page.shutdown()
    w.history.close()
    w._pool.waitForDone()


def clean(window, tmp_path, name):
    (tmp_path / name).write_bytes(b"x")
    window.add_paths([tmp_path / name])
    item = next(i for i in window._items.values() if i.path.name == name)
    window.start()
    from audio_cleaner_desktop.gui.queue_view import CANCELLED, DONE as D, FAILED as F

    assert wait_until(lambda: item.state in (D, F, CANCELLED), 20)
    return item


def test_the_window_lists_a_finished_job_with_its_copies_and_the_tab_is_called_jobs(window, tmp_path):
    assert window.tabs.tabText(1) == "Jobs"
    item = clean(window, tmp_path, "talk.wav")
    [entry] = window.history.search()
    assert entry.status == "done" and entry.job_dir == window.made["keep_dirs"][0].name and entry.copies_bytes > 777
    assert (window.history.jobs_root / entry.job_dir / copies.MANIFEST).exists() and window.history.copies_total().jobs == 1
    window.tabs.setCurrentIndex(1)
    assert window.history_page._stats["disk"].caption.text() == "Saved copies (1 job)"
    window.history_page.open_job(entry.id)
    assert window.history_page.viewing == entry.id


def test_a_job_that_kept_nothing_is_listed_without_copies(window, tmp_path):
    clean(window, tmp_path, "nokeep.wav")
    [entry] = window.history.search()
    assert entry.status == "done" and entry.job_dir == "" and not window.made["keep_dirs"][0].exists()


def test_a_failed_job_keeps_the_copies_it_made_and_one_that_made_none_keeps_no_folder(window, tmp_path):
    clean(window, tmp_path, "fail.wav")
    clean(window, tmp_path, "hopeless.wav")
    by_name = {e.name: e for e in window.history.search()}
    assert by_name["fail.wav"].status == "failed" and by_name["fail.wav"].job_dir != ""
    assert (window.history.jobs_root / by_name["fail.wav"].job_dir).is_dir()
    assert by_name["hopeless.wav"].status == "failed" and by_name["hopeless.wav"].job_dir == ""
    assert window.made["keep_dirs"][1].exists() is False
    assert [p.name for p in window.history.jobs_root.iterdir()] == [by_name["fail.wav"].job_dir]


def test_a_stopped_job_is_not_listed_and_leaves_no_folder(window, tmp_path):
    (tmp_path / "stop.wav").write_bytes(b"x")
    window.add_paths([tmp_path / "stop.wav"])
    item = next(iter(window._items.values()))
    window.start()
    assert wait_until(lambda: item.state == "running", 10)
    window.queue.cancel(item.id)
    assert wait_until(lambda: item.state == "cancelled", 10)
    assert window.history.count() == 0 and not (window.history.jobs_root.exists() and any(window.history.jobs_root.iterdir()))


def test_the_window_checks_the_saved_copies_at_start_up(window, tmp_path):
    clean(window, tmp_path, "talk.wav")
    [entry] = window.history.search()
    folder = window.history.jobs_root / entry.job_dir
    (folder / "extra.bin").write_bytes(b"z" * 1000)  # grew behind the app's back
    gone = copies.new_job_dir(window.history.jobs_root)
    old = 1.0
    for p in (gone,):
        os.utime(p, (old, old))
    window._sweep_leftovers()
    window._pool.waitForDone()
    assert window.history.get(entry.id).copies_bytes == copies.scan_bytes(folder) and not gone.exists()


def test_the_player_goes_back_to_the_files_when_another_job_without_copies_is_opened(page, store, folder):
    job, eid = add_job(store, folder)
    old = store.add(Entry("old.wav", str(folder.original), 5.0, False, {}, 1.0, [str(folder.result)], DONE, finished_at=5.0))
    view = open_ready(page, eid)
    session = FakeSession.made[-1]
    assert view.preview.live and view._player_in_editor
    page.close_job()
    page.open_job(old)
    assert session.closed and view._ed.session is None and not view._player_in_editor
    assert view.preview is not None and not view.preview.live and view.preview._player is view.preview._file and view._listen_card.isVisible()
    page.open_job(eid)  # and the editor, with the live sound, again for the one that has copies
    assert wait_until(lambda: view._ed.editor is not None and view.preview is not None and view.preview.live, 10)
    assert not view._listen_card.isVisible()
