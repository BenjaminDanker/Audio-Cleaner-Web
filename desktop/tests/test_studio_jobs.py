"""The jobs in the editor screen: the column (newest first, sorted, searched, paged, deleted), a finished job opened in the same editor (the parts it was
made with, the live sound, hearing a layer or the original, a change heard at once, quoted and saved as a new file, the original found again), the
window recording what a job kept, and a file that has not run finding moments worth a look. Model runs, the live sound and the sound card are stood in
for (studio_helpers.py, fake_media.py); the real engine is in test_redo.py and test_one_window.py's native test."""
from __future__ import annotations

import os
import threading
from pathlib import Path

import numpy as np
import pytest
from PySide6.QtWidgets import QApplication

from audio_cleaner_desktop import copies, editor_job, engine, resources
from audio_cleaner_desktop.gui import main_window
from audio_cleaner_desktop.gui.history import DONE, FAILED, Entry
from audio_cleaner_desktop.gui.items import format_size

import studio_helpers as sh
from conftest import wait_until
from studio_helpers import FakeSession, add_job, settle

SR = 48_000


@pytest.fixture
def folder(tmp_path):
    return sh.make_folder(tmp_path)


@pytest.fixture
def run():
    return sh.FakeRun()


@pytest.fixture
def jw(window, run, monkeypatch):
    """The window with the live sound, the redo and the models stood in for."""
    FakeSession.made.clear()
    sh.SESSION_DELAY[0] = 0.0
    window.doc_env.open_session = sh.open_session
    window._redo = run
    monkeypatch.setattr(resources, "voice_model_dir", lambda: Path("/models/voice"))
    monkeypatch.setattr(resources, "speech_ready", lambda: True)
    window._confirm = lambda title, text: True
    window._stayed = lambda ids: None
    return window


def entries(w) -> list[int]:
    return [key[1] for key in w.jobs._cards if key[0] == "entry"]


def names(w) -> list[str]:
    return [c.name.text() for k, c in w.jobs._cards.items() if k[0] == "entry"]


def open_job(w, eid, *, live=True):
    w.open_entry(eid)
    if live:
        assert wait_until(lambda: w.doc.session_state == "ready", 10)
    QApplication.processEvents()
    return w.doc


# ---------------------------------------------------------------------------------------------------- the column

def test_the_list_is_newest_first_by_default_and_can_be_sorted(jw, folder):
    for n, t in (("b.wav", 100.0), ("a.wav", 300.0), ("c.wav", 200.0)):
        add_job(jw.history, folder, name=n, finished_at=t)
    jw._draw_jobs()
    assert names(jw) == ["a.wav", "c.wav", "b.wav"] and jw.jobs.sort == "newest"
    for key, expected in (("oldest", ["b.wav", "c.wav", "a.wav"]), ("name", ["a.wav", "b.wav", "c.wav"]), ("newest", ["a.wav", "c.wav", "b.wav"])):
        jw.jobs._set_sort(key)
        assert names(jw) == expected


def test_the_list_is_searched_by_name_folder_and_settings_in_words(jw, folder):
    add_job(jw.history, folder, name="interview.wav", settings={"strength": "gentle", "leveling": True, "loudness": "podcast"}, finished_at=1.0)
    add_job(jw.history, folder, name="lecture.wav", settings={"strength": "max", "leveling": False}, finished_at=2.0)
    for text, expected in (("interview", 1), ("media", 2), ("gentle", 1), ("leveled", 1), ("gentle podcast", 1), ("nothing", 0), ("", 2)):
        jw.jobs.search.setText(text)
        jw.jobs._search_changed()
        assert len(entries(jw)) == expected, text
    jw.jobs.search.setText("nothing")
    jw.jobs._search_changed()
    assert jw.jobs.empty.isVisibleTo(jw.jobs) and jw.jobs.empty.text() == "Nothing matches your search."


def test_there_is_no_cap_the_list_is_paged_and_each_page_adds_to_the_last(jw):
    from audio_cleaner_desktop.gui.jobs_column import PAGE_SIZE

    for n in range(PAGE_SIZE * 3 + 5):
        jw.history.add(Entry(f"f{n}.wav", f"/a/f{n}.wav", 5.0, False, {}, 1.0, [], DONE, finished_at=1000.0 + n))
    jw._draw_jobs()
    total = PAGE_SIZE * 3 + 5
    assert len(entries(jw)) == PAGE_SIZE and jw.jobs.more.isVisibleTo(jw.jobs)
    jw.jobs._show_more()
    assert len(entries(jw)) == PAGE_SIZE * 2 and jw.jobs.more.isVisibleTo(jw.jobs)
    jw.jobs._show_more()
    assert len(entries(jw)) == PAGE_SIZE * 3 and jw.jobs.more.isVisibleTo(jw.jobs)
    jw.jobs._show_more()
    assert len(entries(jw)) == total and not jw.jobs.more.isVisibleTo(jw.jobs)
    assert entries(jw)[0] == jw.history.search(limit=1)[0].id and f"{total} files cleaned" in jw.jobs.totals.text()


def test_the_column_shows_what_was_cleaned_and_the_disk_of_the_saved_copies(jw, folder):
    a, _ = add_job(jw.history, folder, finished_at=2.0)
    b, _ = add_job(jw.history, folder, finished_at=1.0)
    jw.history.add(Entry("old.wav", "/a/old.wav", 5.0, False, {}, 1.0, [], DONE, finished_at=0.5))
    jw._draw_jobs()
    text = jw.jobs.totals.text()
    assert "3 files cleaned" in text and f"saved copies {format_size(a.bytes() + b.bytes())}" in text


def test_deleting_a_job_removes_its_copies_and_entry_and_never_the_results_or_the_original(jw, folder):
    job, eid = add_job(jw.history, folder)
    other, other_id = add_job(jw.history, folder, finished_at=2000.0)
    jw._draw_jobs()
    asked = []
    jw._confirm = lambda title, text: (asked.append(text), False)[1]
    jw.delete_entry(eid)
    assert job.dir.exists() and jw.history.get(eid) is not None and "saved copies" in asked[0] and "not touched" in asked[0]  # declined: nothing happens
    jw._confirm = lambda title, text: True
    jw.delete_entry(eid)
    assert not job.dir.exists() and jw.history.get(eid) is None and other.dir.exists()
    assert folder.original.exists() and folder.result.read_bytes() == b"first result" and entries(jw) == [other_id]


def test_delete_all_removes_every_copy_and_entry_also_failed_and_old_ones(jw, folder):
    a, _ = add_job(jw.history, folder)
    b, _ = add_job(jw.history, folder, status=FAILED, saves=False)
    jw.history.add(Entry("old.wav", "/a/old.wav", 5.0, False, {}, 1.0, ["/a/old-cleaned.wav"], DONE))
    stray = copies.new_job_dir(jw.history.jobs_root)  # copies nothing in the list points to
    (stray / "noise.flac").write_bytes(b"x")
    jw._draw_jobs()
    asked = []
    jw._confirm = lambda title, text: (asked.append(text), True)[1]
    jw.delete_all()
    assert "3 jobs" in asked[0] and "not touched" in asked[0]
    assert jw.history.count() == 0 and not a.dir.exists() and not b.dir.exists() and not stray.exists()
    assert folder.original.exists() and folder.result.exists() and entries(jw) == [] and not jw.jobs.act_delete_all.isEnabled()


def test_a_job_whose_copies_cannot_be_removed_stays_and_the_user_is_told(jw, folder, monkeypatch):
    job, eid = add_job(jw.history, folder)
    jw._draw_jobs()
    told = []
    jw._stayed = told.append
    monkeypatch.setattr(copies, "delete_job_dir", lambda root, name: False)
    jw.delete_entry(eid)
    assert told == [[eid]] and jw.history.get(eid) is not None and job.dir.exists()


def test_deleting_the_job_that_is_open_closes_its_live_sound_and_shows_the_empty_screen(jw, folder):
    job, eid = add_job(jw.history, folder)
    doc = open_job(jw, eid)
    session = FakeSession.made[-1]
    assert jw.studio.preview.live and doc.session is session
    jw.delete_entry(eid)
    assert jw.doc is None and jw.history.get(eid) is None and not job.dir.exists() and session.closed
    assert jw.studio.stack.currentIndex() == 0


# ---------------------------------------------------------------------------------------------------- a job in the editor

def test_a_job_that_keeps_blocks_opens_with_its_parts_and_its_live_sound(jw, folder):
    job, eid = add_job(jw.history, folder)
    doc = open_job(jw, eid)
    s = jw.studio
    session = FakeSession.made[-1]
    assert s.preview.live and s.preview._stream is session.mix and doc.model.done == editor_job.baseline_timeline(job, 10.0)
    assert doc.model.strength_of(0, 10) == "balanced" and not doc.changed_sound and doc.peaks[:4] == [30, 80, 50, 100]
    assert s.start.button.text() == "Open the file" and s.start.button.isEnabled() and s.top.hear_box.isVisible()
    # the timeline's rows show what the job was made with, and the speakers of the rows are on
    assert [(x["label"]) for x in dict(s.timeline.plot.lanes())["voice"]] == ["Balanced · Light"]
    assert s.timeline.labels._hear and s.timeline.labels._buttons["voice"].isChecked() and s.timeline.labels._buttons["voice"].isVisibleTo(s.timeline.labels)
    assert "Hearing: everything" == s.timeline.plot.hearing_text
    assert not s.timeline.labels._buttons["echo"].isVisibleTo(s.timeline.labels)  # (no second audio track)


def test_the_playhead_follows_the_player_and_the_timeline_moves_it(jw, folder):
    _, eid = add_job(jw.history, folder)
    open_job(jw, eid)
    s = jw.studio
    s.preview._player.setPosition(4500)
    s.preview._on_position(4500)
    assert s.timeline.plot.playhead == pytest.approx(4.5, abs=0.01) and s.top.time.text() == "0:05 / 0:10" or s.top.time.text() == "0:04 / 0:10"
    s.timeline.seekRequested.emit(7.0)
    assert s.preview._player.position() == 7000 and s.timeline.plot.playhead == 7.0 and s.top.time.text() == "0:07 / 0:10"


def test_the_speakers_of_the_rows_change_the_live_sound_and_not_the_timeline(jw, folder):
    _, eid = add_job(jw.history, folder)
    doc = open_job(jw, eid)
    s = jw.studio
    mix = FakeSession.made[-1].mix
    voice, level = s.timeline.labels._buttons["voice"], s.timeline.labels._buttons["level"]
    voice.click()
    assert not mix.hear["noise"] and not mix.hear["rebuild"] and mix.hear["leveling"] and not doc.changed_sound
    gain = 10 ** (mix.level_gains_db()[0][2] / 20)  # (Leveling is still heard: the recording at its loudness)
    assert np.allclose(mix.read(SR, 50), mix.layers.src[SR:SR + 50] * gain, atol=1e-6)
    assert s.timeline.plot.hearing_text == "Hearing: without Voice"
    voice.click()
    level.click()
    assert mix.hear == {"echo": True, "noise": True, "rebuild": True, "leveling": False} and s.timeline.plot.hearing_text == "Hearing: without Leveling"
    share = 10 ** (-24.0 / 20)  # the job's strength (Balanced) is a blend of the remover's sound and the recording
    assert np.allclose(mix.read(SR, 50), (1 - share) * mix.layers.arrays["light"][SR:SR + 50] + share * mix.layers.src[SR:SR + 50], atol=1e-6)
    voice.click()  # (Leveling is off already)
    assert s.timeline.plot.hearing_text == "Hearing: the original"


def test_the_original_and_the_cleaned_can_be_switched_and_the_position_stays(jw, folder):
    _, eid = add_job(jw.history, folder)
    open_job(jw, eid)
    s = jw.studio
    mix = FakeSession.made[-1].mix
    s.top.which._buttons["original"].click()
    assert mix.original and s.preview.listening == "original"
    s.top.which._buttons["cleaned"].click()
    assert not mix.original


def test_a_change_is_heard_at_once_quoted_on_the_button_and_saved_as_a_new_file_from_the_timeline(jw, folder, run):
    job, eid = add_job(jw.history, folder)
    doc = open_job(jw, eid)
    s = jw.studio
    session = FakeSession.made[-1]
    m = doc.model
    m.select(3, 5)
    m.set_strength("max")
    assert wait_until(lambda: session.refreshed, 5)  # (the live sound follows an edit after a short quiet)
    settle(jw)
    assert s.start.button.text() == "Save changes" and s.start.button.isEnabled() and session.refreshed[-1].layers["noise"][1].settings["strength_db"] == 100.0
    session.mix.set_hear("leveling", False)
    assert np.allclose(session.mix.read(4 * SR, 20), session.mix.layers.arrays["light"][4 * SR:4 * SR + 20], atol=1e-6)  # heard now
    m.set_remover("heavy")
    settle(jw)
    assert s.start.button.text() == "Save changes · about 40 s"  # what saving would take, from the blocks that are there
    run.gate = threading.Event()
    s.start.button.click()
    assert wait_until(lambda: jw._saving is not None and doc.stage == "processing", 5)
    settle(jw)
    assert s.start.mode == "progress" and not s.inspector.cards.isEnabled() and s.start.button.text().startswith("Saving") and doc.saving
    run.gate.set()
    assert wait_until(lambda: jw._saving is None, 10)
    settle(jw)
    wanted = run.calls[-1].wanted
    assert wanted.timeline["layers"]["noise"][1] == {"start": 3.0, "end": 5.0, "strength_db": 100.0, "remover": "heavy"}
    assert wanted.output_name == "talk-cleaned (2).wav" and (folder.dir / "talk-cleaned (2).wav").is_file()
    assert doc.stage == "done" and not doc.changed_sound and doc.model.done == doc.model.draft and doc.model.selection is None  # the result is the draft now
    assert copies.JobCopies.load(job.dir).editor["timeline"]["layers"]["noise"][1]["remover"] == "heavy"
    assert s.start.button.text() == "Open the file" and [p.name for p in doc.result_paths()][0] == "talk-cleaned (2).wav"
    assert folder.result.read_bytes() == b"first result" and "Saved talk-cleaned (2).wav" in doc.note
    entry = jw.history.get(eid)
    assert entry.outputs == [str(folder.dir / "talk-cleaned (2).wav")] and entry.settings["leveling"] is True and entry.job_dir == job.dir.name


def test_a_change_to_the_name_the_type_or_the_subtitles_is_saved_too(jw, folder, run):
    _, eid = add_job(jw.history, folder, transcript=True)
    doc = open_job(jw, eid)
    s = jw.studio
    s.output.endingPicked.emit(".mp3")
    settle(jw)
    assert doc.out_name == "talk-cleaned (2).mp3" and doc.changed_output and s.start.button.text() == "Save changes"
    s.inspector.subs._buttons["on"].click()
    settle(jw)
    s.inspector.files._buttons["vtt"].click()  # (on, as a VTT file)
    settle(jw)
    assert doc.subtitles == "vtt"
    s.start.button.click()
    assert wait_until(lambda: jw._saving is None and run.calls, 10)
    assert run.calls[0].wanted.subtitles == "vtt" and run.calls[0].wanted.output_name == "talk-cleaned (2).mp3"
    assert wait_until(lambda: doc.stage == "done", 5)
    assert [Path(p).suffix for p in jw.history.get(eid).outputs] == [".mp3", ".vtt"]


def test_save_can_be_stopped_and_nothing_is_saved(jw, folder, run):
    job, eid = add_job(jw.history, folder)
    doc = open_job(jw, eid)
    doc.model.select(3, 5)
    doc.model.set_strength("gentle")
    settle(jw)
    run.gate = threading.Event()
    jw.studio.start.button.click()
    assert wait_until(lambda: jw._saving is not None, 5)
    jw.cancel_current()
    run.gate.set()
    assert wait_until(lambda: jw._saving is None, 10)
    settle(jw)
    assert not (folder.dir / "talk-cleaned (2).wav").exists() and doc.changed_sound and jw.studio.start.button.isEnabled()
    assert "Stopped" in doc.note and len(job.saves) == 1


def test_a_name_that_cannot_be_used_stops_the_save_and_says_why(jw, folder):
    _, eid = add_job(jw.history, folder)
    doc = open_job(jw, eid)
    doc.model.select(3, 5)
    doc.model.set_strength("gentle")
    jw.studio.output.nameEdited.emit("bad/name.wav")
    settle(jw)
    s = jw.studio
    assert not s.start.button.isEnabled() and "can't contain" in s.output.hint.text() and s.output.name.property("invalid")
    jw.studio.output.nameEdited.emit("")
    settle(jw)
    assert not s.start.button.isEnabled() and any("Type a name" in t for t, _ in s.note_lines(doc))


def test_a_name_that_is_taken_asks_and_replaces_only_when_the_user_says_so(jw, folder, run):
    _, eid = add_job(jw.history, folder)
    doc = open_job(jw, eid)
    asked = []
    for answer in ("cancel", "new", "replace"):
        jw.ask_replace = lambda name, a=answer: (asked.append(name), a)[1]
        doc.set_out_name("talk-cleaned.wav")  # taken: it is the first result
        jw.save_changes(doc)
        assert wait_until(lambda: jw._saving is None, 10)
        settle(jw)
    assert asked == ["talk-cleaned.wav"] * 3
    assert len(run.calls) == 2 and run.calls[0].replace is None and run.calls[1].replace == folder.result  # cancelled: nothing ran
    jw.ask_replace = lambda name: pytest.fail("asked")  # a free name is never asked about
    doc.set_out_name("free.wav")
    jw.save_changes(doc)
    assert wait_until(lambda: jw._saving is None, 10) and len(run.calls) == 3


def test_a_change_that_fails_says_why_and_changes_nothing(jw, folder, run):
    job, eid = add_job(jw.history, folder)
    doc = open_job(jw, eid)
    doc.set_out_name("again.wav")
    run.fail = "The original no longer matches the saved copies."
    before = jw.history.get(eid)
    jw.save_changes(doc)
    assert wait_until(lambda: jw._saving is None, 10)
    settle(jw)
    assert "no longer matches" in doc.note and doc.note_kind == "error" and jw.history.get(eid).outputs == before.outputs and folder.result.exists()
    assert jw.studio.start.button.isEnabled() and doc.stage == "done"


def test_a_job_cannot_be_deleted_while_a_change_is_being_saved_and_closing_stops_the_save(jw, folder, run):
    job, eid = add_job(jw.history, folder)
    doc = open_job(jw, eid)
    doc.set_out_name("again.wav")
    run.gate = threading.Event()
    jw.save_changes(doc)
    assert wait_until(lambda: jw._saving is not None, 5)
    jw.delete_entry(eid)
    assert jw.history.get(eid) is not None and job.dir.exists()
    threading.Timer(0.2, run.gate.set).start()
    jw._closing = True
    jw._saving.cancel.set()
    assert jw._saving.wait(5000)
    run.gate.set()


def test_saving_only_the_subtitles_needs_the_transcript_but_not_the_original(jw, folder, run):
    job, eid = add_job(jw.history, folder, transcript=True)
    folder.original.unlink()  # gone: every other change is unavailable
    doc = open_job(jw, eid, live=False)
    settle(jw)
    s = jw.studio
    assert doc.session_state == "none" and not doc.original.ok
    doc.set_subtitles("srt")
    settle(jw)
    assert not s.start.button.isEnabled() and any("original" in t.lower() for t, _ in s.note_lines(doc))
    jw.save_subtitles_only(doc)
    srt = folder.dir / "talk-cleaned.srt"
    assert srt.exists() and "hello there" in srt.read_text() and run.calls == []
    assert str(srt) in jw.history.get(eid).outputs and "Saved talk-cleaned.srt" in doc.note


def test_a_moved_original_makes_every_change_unavailable_and_it_can_be_found_again(jw, folder, monkeypatch):
    job, eid = add_job(jw.history, folder)
    moved = folder.dir / "renamed.wav"
    folder.original.rename(moved)
    doc = open_job(jw, eid, live=False)
    settle(jw)
    s = jw.studio
    assert doc.session_state == "none" and not FakeSession.made  # the live sound needs the original
    texts = [t for t, _ in s.note_lines(doc)]
    assert any("not at" in t for t in texts)
    banner = s.notes.itemAt(0).widget()
    assert banner._button.text() == "Find the original…"
    doc.model.select(3, 5)
    doc.model.set_strength("gentle")
    settle(jw)
    assert not s.start.button.isEnabled()
    # pointing at the wrong file is refused
    wrong = folder.dir / "other.wav"
    wrong.write_bytes(os.urandom(200_000))
    monkeypatch.setattr(main_window.QFileDialog, "getOpenFileName", staticmethod(lambda *a, **k: (str(wrong), "")))
    jw.find_original()
    assert "not the same file" in doc.note and not doc.original.ok
    monkeypatch.setattr(main_window.QFileDialog, "getOpenFileName", staticmethod(lambda *a, **k: (str(moved), "")))
    jw.find_original()
    assert doc.original.ok and jw.history.get(eid).source == str(moved)
    assert wait_until(lambda: doc.session_state == "ready", 10)
    settle(jw)
    assert s.start.button.isEnabled()


def test_a_changed_original_is_said_and_cannot_be_found_by_hand(jw, folder):
    _, eid = add_job(jw.history, folder)
    folder.original.write_bytes(b"something else entirely")
    doc = open_job(jw, eid, live=False)
    settle(jw)
    assert doc.original.state == "changed" and doc.session_state == "none"
    banner = jw.studio.notes.itemAt(0).widget()
    assert "changed" in banner._label.text() and not banner._button.isVisibleTo(banner)


def test_an_old_job_with_saved_copies_opens_at_once_and_has_no_setup_button(jw, folder):
    job, eid = add_job(jw.history, folder, blocks=False, legacy=True)
    doc = open_job(jw, eid)
    assert doc.model is not None and jw.studio.preview.live
    job = copies.JobCopies.load(job.dir)
    assert job.has_blocks and not job.has(copies.NOISE)  # its copy became blocks on the way in (editor_job.make_ready)


def test_a_job_that_rebuilt_the_voice_opens_with_the_rebuild_on_its_whole_length(jw, folder):
    job, eid = add_job(jw.history, folder, blocks=False, legacy=True, rebuilt=True)
    doc = open_job(jw, eid)
    done = doc.model.done
    assert done.layers["rebuild"] and not done.layers["noise"] and doc.model.voice_mode_of(0, 10) == "rebuild"
    assert dict(jw.studio.timeline.plot.lanes())["voice"][0]["label"] == "Rebuild"


def test_without_sound_support_the_job_still_opens_and_the_change_can_be_saved(jw, folder, run, monkeypatch):
    from audio_cleaner_desktop.gui import player

    monkeypatch.setattr(player, "_multimedia", None)
    _, eid = add_job(jw.history, folder)
    doc = open_job(jw, eid)
    assert not jw.studio.preview.available and doc.model is not None
    doc.set_out_name("quiet.wav")
    jw.save_changes(doc)
    assert wait_until(lambda: jw._saving is None and run.calls, 10) and [c.wanted.output_name for c in run.calls] == ["quiet.wav"]


def test_voice_rebuild_without_its_model_cannot_be_saved(jw, folder, monkeypatch):
    monkeypatch.setattr(resources, "voice_model_dir", lambda: None)
    _, eid = add_job(jw.history, folder)
    doc = open_job(jw, eid)
    doc.model.select(3, 5)
    doc.model.set_voice("rebuild")
    settle(jw)
    s = jw.studio
    assert not s.start.button.isEnabled() and any("model" in t for t, _ in s.note_lines(doc))


def test_a_job_with_no_saved_copies_can_be_listened_to_and_deleted_but_not_changed(jw, folder):
    eid = jw.history.add(Entry("old.wav", str(folder.original), 5.0, False, {"strength": "gentle", "leveling": True}, 1.0, [str(folder.result)], DONE,
                               finished_at=5.0))
    jw._draw_jobs()
    doc = open_job(jw, eid, live=False)
    s = jw.studio
    assert doc.readonly and doc.locked and doc.session_state == "none" and not FakeSession.made
    assert not s.inspector.cards.isEnabled() and "no saved copies" in " ".join(t for t, _ in s.note_lines(doc)) and s.start.button.text() == "Open the file"
    assert [p.name for p in doc.result_paths()] == [folder.result.name] and s.preview is not None
    jw.delete_entry(eid)
    assert jw.history.get(eid) is None and jw.doc is None and folder.result.exists()


def test_a_job_with_a_folder_that_has_no_usable_manifest_is_listed_like_an_old_one(jw, folder):
    job, eid = add_job(jw.history, folder)
    (job.dir / copies.MANIFEST).write_text("{broken")
    doc = open_job(jw, eid, live=False)
    assert doc.readonly and doc.job is None


def test_a_failed_job_with_copies_says_why_lists_none_and_can_still_be_saved(jw, folder, run):
    job, eid = add_job(jw.history, folder, status=FAILED, saves=False)
    doc = open_job(jw, eid)
    assert "Saving the cleaned audio failed." in doc.note and doc.note_kind == "error" and not doc.result_paths() or doc.result_paths() == []
    doc.set_out_name("late.wav")
    settle(jw)
    jw.save_changes(doc)
    assert wait_until(lambda: jw._saving is None and run.calls, 10)
    assert jw.history.get(eid).status == DONE and jw.history.get(eid).outputs == [str(folder.dir / "late.wav")]


def test_leaving_a_job_closes_its_live_sound_and_another_file_takes_its_place(jw, folder):
    _, eid = add_job(jw.history, folder)
    doc = open_job(jw, eid)
    session = FakeSession.made[-1]
    other = jw.history.add(Entry("old.wav", str(folder.original), 5.0, False, {}, 1.0, [str(folder.result)], DONE, finished_at=5.0))  # (no saved copies)
    jw.open_entry(other)
    assert session.closed and doc.session is None and jw.doc is not doc and jw.doc.readonly
    jw.open_entry(eid)
    assert wait_until(lambda: jw.doc.session_state == "ready", 10)
    jw._show(None)
    assert FakeSession.made[-1].closed


# ---------------------------------------------------------------------------------------------------- suggestions

def test_suggestions_are_found_when_none_were_kept_applied_undone_dismissed_and_kept_with_the_job(jw, folder):
    jw.doc_env.auto_suggest = True
    job, eid = add_job(jw.history, folder)
    doc = open_job(jw, eid)  # (nothing was kept: they are looked for by themselves once the live sound is there)
    assert wait_until(lambda: [s.id for s in doc.model.suggestions] == ["s0"], 10) and "find" not in doc._workers
    job2 = copies.JobCopies.load(job.dir)
    assert editor_job.stored_suggestions(job2)["items"][0]["id"] == "s0"  # kept with the job
    settle(jw)
    s = jw.studio
    assert s.timeline.plot._pin_positions() == [("s0", 4.0, "open")] and s.top.chip2.isVisible() and s.top.chip2.text().startswith("1")
    s.inspector.suggestionApplied.emit("s0")
    assert doc.model.switch_of("rebuild", 3, 5) is True and doc.changed_sound
    assert copies.JobCopies.load(job.dir).editor["suggestions"]["items"][0]["state"] == "applied"
    doc.model.undo("s0")
    doc.keep_suggestions()
    s.inspector.suggestionDismissed.emit("s0")
    assert copies.JobCopies.load(job.dir).editor["suggestions"]["items"][0]["state"] == "dismissed" and not doc.model.open_suggestions()
    # the next time the job is opened they are not looked for again, and the dismissed one stays dismissed
    jw._show(None)
    n = len(FakeSession.made)
    again = open_job(jw, eid)
    assert len(FakeSession.made) == n + 1 and again.sg == "done" and [x.state for x in again.model.suggestions] == ["dismissed"]


def test_looking_for_suggestions_can_be_stopped_and_a_long_recording_waits_for_the_button(jw, folder, monkeypatch):
    job, eid = add_job(jw.history, folder)
    original = FakeSession.__init__

    def slow(self, job, seconds=10.0):
        original(self, job, seconds)
        self.gate = threading.Event()

    monkeypatch.setattr(FakeSession, "__init__", slow)
    doc = open_job(jw, eid)
    s = jw.studio
    assert doc.can_offer_find and s.top.sg_action.text() == "Look for them"  # (not looked for by themselves here)
    s.top.sg_action.click()
    assert wait_until(lambda: doc.sg == "running", 5)
    settle(jw)
    assert "Listening to your file for moments worth a look" in s.top.sg_text.text() and s.top.sg_action.text() == "Cancel"
    s.top.sg_action.click()
    assert wait_until(lambda: doc.sg == "canceled", 10)
    settle(jw)
    assert doc.model.suggestions == [] and "Stopped looking" in s.top.sg_text.text()
    # a long recording: nothing is looked for by itself, there is a button
    monkeypatch.setattr(FakeSession, "__init__", original)
    jw._show(None)
    long_job, long_id = add_job(jw.history, folder, seconds=editor_job.SUGGEST_AUTO_SECONDS + 60.0, name="long.wav")
    long_job.data["samples"] = int((editor_job.SUGGEST_AUTO_SECONDS + 60.0) * SR)
    long_job.editor["facts"]["n"] = long_job.data["samples"]
    long_job.save()
    jw.doc_env.auto_suggest = True
    jw.open_entry(long_id)
    longdoc = jw.doc
    assert wait_until(lambda: longdoc.session_state == "ready", 10)
    settle(jw)
    assert longdoc.sg == "idle" and longdoc.can_offer_find and s.top.sg_action.text() == "Look for them"


def test_a_file_that_has_not_run_finds_moments_worth_a_look_and_applying_one_goes_to_the_run(jw, tmp_path, runner):
    class Plan:
        closed = False

        def suggest(self, *, cancel=None, progress=None):
            if progress:
                progress(0.5)
            return {"seconds": 11.0, "whole": {}, "items": [{"id": "s0", "start": 2.0, "end": 4.0, "kind": "part", "rebuild": False,
                                                            "noise": {"strength_db": 40.0, "remover": "light"}, "reasons": ["is noisier"], "state": "open"}]}

        def close(self):
            Plan.closed = True

    jw.doc_env.open_plan = lambda path, **kw: Plan()
    jw.doc_env.auto_suggest = True
    sh.add_file(jw, tmp_path)
    doc = jw.doc
    assert wait_until(lambda: doc.model is not None and [s.id for s in doc.model.open_suggestions()] == ["s0"], 15)
    assert Plan.closed  # (all of the file was decoded for it and let go again)
    settle(jw)
    s = jw.studio
    assert s.top.chip.isVisible() and s.top.chip.text().startswith("1")
    s.top.chip.click()  # Show the next one: the part is chosen and the card for it is there
    settle(jw)
    assert doc.model.selection == (2.0, 4.0) and s.inspector.sg_card.isVisibleTo(s.inspector) and "is noisier" in s.inspector.sg_text.text()
    s.inspector.sg_apply.click()
    assert doc.model.strength_of(2, 4) == "strong" and not doc.untouched_auto
    s.start.button.click()
    assert wait_until(lambda: doc.stage == "done", 10)
    noise = runner.calls[0]["timeline"]["layers"]["noise"]
    assert [(p["start"], p["end"], p["strength_db"]) for p in noise][1] == (2.0, 4.0, 40.0)
    assert jw.history.search()[0].settings["parts"] is True


# ---------------------------------------------------------------------------------------------------- the window records what a job kept

@pytest.fixture
def keeper(window, tmp_path):
    """A runner that keeps copies the way the engine does (stand-in bytes), and fails, stops or keeps nothing by the file's name."""
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
            job = copies.JobCopies.create(Path(keep_dir), path, copies.fingerprint(path), {"duration": 11.0})
            (Path(keep_dir) / "noise.flac").write_bytes(b"n" * 777)
            job.put_audio(copies.NOISE, {"file": "noise.flac", "channels": [0], "scale": 1.0, "samples": 10}, plan=[0], profile="light")
        if path.name.startswith("fail"):
            raise engine.MediaError("Saving the cleaned audio failed.")
        if path.name.startswith("hopeless"):
            if keep_dir is not None:
                Path(keep_dir).mkdir(parents=True, exist_ok=True)  # an empty folder: it made nothing
            raise engine.MediaError("This file could not be read.")
        out.write_bytes(b"result")
        return engine.JobResult(out, 11.0, False, kept_dir=Path(keep_dir) if keep_dir is not None and not path.name.startswith("nokeep") else None)

    window.queue._runner = runner
    window.made = made
    window.doc_env.open_session = sh.open_session
    return window


def clean(w, tmp_path, name):
    sh.add_file(w, tmp_path, name)
    item = next(i for i in w._items.values() if i.path.name == name)
    assert wait_until(lambda: item.doc.model is not None, 5)
    w.start()
    assert wait_until(lambda: item.state in ("done", "failed", "cancelled"), 20)
    QApplication.processEvents()
    return item


def test_the_window_lists_a_finished_job_with_its_copies(keeper, tmp_path):
    w = keeper
    item = clean(w, tmp_path, "talk.wav")
    [entry] = w.history.search()
    assert entry.status == "done" and entry.job_dir == w.made["keep_dirs"][0].name and entry.copies_bytes > 777 and item.entry_id == entry.id
    assert (w.history.jobs_root / entry.job_dir / copies.MANIFEST).exists() and w.history.copies_total().jobs == 1
    assert list(w.jobs._cards) == [("entry", entry.id)] and "1 file cleaned" in w.jobs.totals.text()
    assert w.doc.entry.id == entry.id and w.doc.stage == "done"


def test_a_job_that_kept_nothing_is_listed_without_copies(keeper, tmp_path):
    w = keeper
    clean(w, tmp_path, "nokeep.wav")
    [entry] = w.history.search()
    assert entry.status == "done" and entry.job_dir == "" and not w.made["keep_dirs"][0].exists()
    assert w.doc.readonly and w.studio.start.button.text() == "Open the file"


def test_a_failed_job_keeps_the_copies_it_made_and_one_that_made_none_keeps_no_folder(keeper, tmp_path):
    w = keeper
    failed = clean(w, tmp_path, "fail.wav")
    clean(w, tmp_path, "hopeless.wav")
    by_name = {e.name: e for e in w.history.search()}
    assert by_name["fail.wav"].status == "failed" and by_name["fail.wav"].job_dir != ""
    assert (w.history.jobs_root / by_name["fail.wav"].job_dir).is_dir()
    assert by_name["hopeless.wav"].status == "failed" and by_name["hopeless.wav"].job_dir == "" and w.made["keep_dirs"][1].exists() is False
    assert [p.name for p in w.history.jobs_root.iterdir()] == [by_name["fail.wav"].job_dir]
    # the screen says why, and the file can be started again from the same editor
    assert failed.doc.stage == "ready" and failed.doc.note == "Saving the cleaned audio failed." and failed.doc.note_kind == "error"
    w.open_item(failed.id)
    assert w.studio.start.button.text() == "Try again" and w.studio.start.button.isEnabled()


def test_a_stopped_job_is_not_listed_and_leaves_no_folder(keeper, tmp_path):
    w = keeper
    sh.add_file(w, tmp_path, "stop.wav")
    item = next(iter(w._items.values()))
    assert wait_until(lambda: item.doc.model is not None, 5)
    w.start()
    assert wait_until(lambda: item.state == "running", 10)
    w.cancel_current()
    assert wait_until(lambda: item.state == "cancelled", 10)
    assert w.history.count() == 0 and not (w.history.jobs_root.exists() and any(w.history.jobs_root.iterdir()))
    assert item.doc.stage == "ready" and "Stopped" in item.doc.note and list(w.jobs._cards) == [("item", item.id)]


def test_the_window_checks_the_saved_copies_at_start_up(keeper, tmp_path):
    w = keeper
    clean(w, tmp_path, "talk.wav")
    [entry] = w.history.search()
    folder = w.history.jobs_root / entry.job_dir
    (folder / "extra.bin").write_bytes(b"z" * 1000)  # grew behind the app's back
    gone = copies.new_job_dir(w.history.jobs_root)
    os.utime(gone, (1.0, 1.0))
    w._sweep_leftovers()
    w._pool.waitForDone()
    assert w.history.get(entry.id).copies_bytes == copies.scan_bytes(folder) and not gone.exists()


def test_a_file_in_line_can_be_taken_out_again_and_several_start_together(keeper, tmp_path):
    w = keeper
    for name in ("one.wav", "two.wav"):
        (tmp_path / name).write_bytes(b"x" * 50)
    w.add_paths([tmp_path / "one.wav", tmp_path / "two.wav"])
    assert wait_until(lambda: all(i.doc.model is not None for i in w._items.values()), 5)
    w.start_all()
    assert wait_until(lambda: all(i.state == "done" for i in w._items.values()), 20)
    assert w.history.count() == 2 and (tmp_path / "one-cleaned.wav").exists() and (tmp_path / "two-cleaned.wav").exists()


def test_files_that_wait_can_be_moved_earlier_or_later_in_line(keeper, tmp_path):
    w = keeper
    for name in ("one.wav", "two.wav", "three.wav"):
        (tmp_path / name).write_bytes(b"x" * 50)
    w.add_paths([tmp_path / n for n in ("one.wav", "two.wav", "three.wav")])
    assert wait_until(lambda: all(i.doc.model is not None for i in w._items.values()), 5)
    assert [c.name.text() for c in w.jobs._cards.values()] == ["one.wav", "two.wav", "three.wav"]
    w._move(3, -1)
    w._move(1, 1)
    assert [c.name.text() for c in w.jobs._cards.values()] == ["three.wav", "one.wav", "two.wav"]
    w._move(3, -1)  # already first: nothing
    assert next(iter(w.jobs._cards.values())).name.text() == "three.wav"
    order = [i.path.name for i in w._items.values()]
    w.prefs.parallel = 1
    w._settings_changed()
    w.start_all()
    assert wait_until(lambda: all(i.state == "done" for i in w._items.values()), 20)
    started = [e.name for e in sorted(w.history.search(), key=lambda e: e.id)]
    assert started == order  # they were started in the order they were put in
