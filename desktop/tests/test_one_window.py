"""One window, one screen: the editor is there before a file is chosen (the place to drop one, greyed settings, Start off), choosing a file fills it in
place (no new window, no dialog), a run finishes on the same screen, a past job opens in it, and opening a long file or job never holds the window."""
from __future__ import annotations

import threading
import time

import pytest
from PySide6.QtCore import QPoint, QTimer
from PySide6.QtWidgets import QApplication

from audio_cleaner_desktop.gui import main_window

import studio_helpers as sh
from conftest import wait_until
from studio_helpers import add_file, ready, windows_showing


# ---------------------------------------------------------------------------------------------------- the empty screen

def test_the_empty_screen_already_has_the_whole_layout(window):
    s = window.studio
    assert s.stack.currentIndex() == 0 and window.doc is None
    # the place to drop a file is in the editor's main pane, where the picture and the timeline go; the rows are there, greyed
    assert s.drop.isVisible() and s.ghost.isVisible() and s.output.isVisible() and not s.output.fields.isVisible()
    assert s.inspector.ghost.isVisible() and not s.inspector.cards.isVisible() and s.inspector.heading.text() == "Settings"
    ghost_text = " ".join(label.text() for label in s.inspector.ghost.findChildren(type(s.inspector.heading)))
    for title in ("Voice", "Leveling", "Subtitles"):
        assert title in ghost_text
    # the (i) works already, Start does not
    assert s.inspector.info.isVisible() and s.inspector.info.isEnabled()
    assert s.start.button.text() == "Start" and not s.start.button.isEnabled()
    # the jobs are a column beside the editor, and the left column ends level with the settings column
    assert window.jobs.isVisible() and window.jobs.mapTo(window, QPoint(0, 0)).x() > s.mapTo(window, QPoint(0, 0)).x() + s.width() - 5
    out_bottom = s.output.mapTo(window, QPoint(0, s.output.height())).y()
    insp_bottom = s.inspector.mapTo(window, QPoint(0, s.inspector.height())).y()
    assert abs(out_bottom - insp_bottom) <= 2
    assert len(windows_showing()) == 1


def test_the_i_opens_all_the_options_with_good_to_know(qapp, monkeypatch):
    from PySide6.QtWidgets import QLabel

    from audio_cleaner_desktop.gui.info_dialog import InfoDialog

    dialog = InfoDialog()
    text = " ".join(label.text() for label in dialog.findChildren(QLabel))
    for heading in ("All the options", "Noise removal", "Echo removal", "Leveling", "Subtitles", "Good to know", "Voice: Noise removal or Voice rebuild?"):
        assert heading in text
    assert "Podcast" in text and "−16 LUFS" in text and "Nothing is uploaded" in text
    dialog.close()


def test_the_i_button_asks_for_the_options_dialog(window, monkeypatch):
    shown = []
    monkeypatch.setattr(main_window.InfoDialog, "exec", lambda self: shown.append(self.windowTitle()))
    window.studio.inspector.info.click()
    assert shown == ["All the options"]


# ---------------------------------------------------------------------------------------------------- choosing a file

def test_choosing_a_file_fills_the_screen_in_place(window, tmp_path):
    before = windows_showing()
    s = window.studio
    window.fakes.peaks_delay = 0.4
    add_file(window, tmp_path)
    # at once: the editor with the name, the settings and the Start button, in the same window, with no dialog
    assert s.stack.currentIndex() == 1 and s.top.name.text() == "talk.wav" and windows_showing() == before
    assert QApplication.activeModalWidget() is None
    assert wait_until(lambda: window.doc.model is not None, 5)
    QApplication.processEvents()
    assert s.inspector.cards.isVisible() and not s.inspector.ghost.isVisible() and s.inspector.heading.text() == "Whole file · 0:11"
    assert s.top.meta.text().startswith("0:11 · Audio") and s.top.different.isVisible()
    assert s.output.fields.isVisible() and s.output.name.text() == "talk-cleaned" and s.output.format.currentData() == ".wav"
    assert s.start.button.text().startswith("Start · about ") and s.start.button.isEnabled()  # (what is on, and about how long it takes here)
    lanes = [lane for lane, _ in s.timeline.plot.lanes()]
    assert lanes == ["voice", "level", "subs"]  # (Echo removal only for a file with a second audio track)
    # the waveform fills in while it is read
    assert wait_until(lambda: window.doc.wave == "reading" and window.doc.peaks, 5) or window.doc.wave == "ready"
    assert wait_until(lambda: window.doc.wave == "ready", 10)
    assert s.top.switch.isChecked()  # Automatic is on, as it was left
    assert wait_until(lambda: window.doc.applied, 10)
    assert "Set automatically for this file." in s.top.auto_text.text()
    # the file is in the jobs column as one that was not started; "Different file" takes it away and the empty screen is back
    assert list(window.jobs._cards) == [("item", 1)]
    s.top.different.click()
    assert s.stack.currentIndex() == 0 and window.doc is None and not window._items and windows_showing() == before


def test_a_second_audio_track_adds_the_echo_row_and_card(window, tmp_path):
    window.fakes.facts.update(tracks=2)
    add_file(window, tmp_path)
    ready(window)
    s = window.studio
    assert [lane for lane, _ in s.timeline.plot.lanes()] == ["voice", "echo", "level", "subs"]
    assert s.inspector.echo_card.isVisible()
    s.inspector.echo.click()
    assert window.doc.model.switch_of("echo", 0, 11) is True


def test_a_file_that_cannot_be_read_is_not_taken_in_and_says_why(window, tmp_path):
    def broken(path, ff=None):
        from acw_common.ffmpeg import MediaError

        raise MediaError("This file could not be read as audio or video.")

    window.doc_env.probe = broken
    add_file(window, tmp_path, "broken.wav")
    assert wait_until(lambda: not window._items, 5)
    QApplication.processEvents()
    assert window.studio.stack.currentIndex() == 0 and "broken.wav: This file could not be read" in window.studio.problem._label.text()
    assert window.studio.problem.isVisible()


def test_several_files_are_taken_in_the_first_is_shown_and_the_rest_wait_in_the_jobs(window, tmp_path):
    for name in ("a.wav", "b.wav", "c.wav"):
        (tmp_path / name).write_bytes(b"x" * 100)
    assert window.add_paths([tmp_path / "a.wav", tmp_path / "b.wav", tmp_path / "c.wav"]) == 3
    assert window.studio.top.name.text() == "a.wav" and len(window._items) == 3
    assert [c for c in window.jobs._cards] == [("item", 1), ("item", 2), ("item", 3)]
    window.open_item(2)  # opening another one shows it in the same editor
    assert window.studio.top.name.text() == "b.wav" and window.doc.item.id == 2
    assert window.add_paths([tmp_path / "b.wav"]) == 1 and len(window._items) == 3  # the same file is shown again, not added twice


# ---------------------------------------------------------------------------------------------------- a run

def test_a_run_finishes_on_the_same_screen(window, tmp_path, runner):
    add_file(window, tmp_path)
    ready(window)
    s = window.studio
    doc = window.doc
    runner.gate = threading.Event()
    before = windows_showing()
    s.start.button.click()
    assert wait_until(lambda: doc.stage == "processing" and doc.run_fraction == 0.5, 10)
    assert wait_until(lambda: s.timeline.plot.progress == 0.5 and s.start.mode == "progress", 5)  # (a run's progress is drawn at most ten times a second)
    QApplication.processEvents()
    # while it runs: the same screen, locked, with the progress where the details were
    assert s.start.mode == "progress" and s.start.progress_box.isVisible() and not s.start.button.isEnabled() and s.start.button.text().startswith("Processing")
    assert not s.inspector.cards.isEnabled() and not s.output.isEnabled() and s.timeline.plot.progress == 0.5
    assert window.jobs._cards[("item", 1)].status.text().startswith("Cleaning")
    assert windows_showing() == before
    runner.gate.set()
    assert wait_until(lambda: doc.stage == "done", 10)
    QApplication.processEvents()
    # the result: the same screen, the same window, the files, the button that opens the file
    assert window.studio is s and window.doc is doc and s.stack.currentIndex() == 1 and windows_showing() == before
    assert s.start.mode == "files" and s.start.button.text() == "Open the file" and s.start.button.isEnabled()
    assert (tmp_path / "talk-cleaned.wav").read_bytes() == b"cleaned" and doc.entry is not None and doc.entry.status == "done"
    assert wait_until(lambda: s.notes.count() > 0 and s.notes.itemAt(0).widget() is not None, 5)  # (the note comes with the screen's redraw after the run)
    assert "Saved as talk-cleaned.wav" in s.notes.itemAt(0).widget()._label.text()
    assert list(window.jobs._cards) == [("entry", doc.entry.id)]  # the finished job replaced the waiting one in the column
    assert window.jobs.current == ("entry", doc.entry.id)
    # Automatic was left untouched, so the run decided for itself (the whole file with its own settings): no timeline was handed over
    assert runner.calls[0]["timeline"] is None and runner.calls[0]["options"].strength == "auto"


def test_a_change_made_by_hand_goes_to_the_run_as_a_timeline_and_the_settings_are_remembered(window, tmp_path, runner):
    add_file(window, tmp_path)
    ready(window)
    s, doc = window.studio, window.doc
    s.top.switch.click()  # Automatic off: the settings are the ones chosen by hand
    assert not doc.auto and doc.model.strength_of(0, 11) == "max"
    s.inspector.strength._buttons["gentle"].click()
    s.inspector.level._buttons["youtube"].click()
    assert doc.model.strength_of(0, 11) == "gentle" and doc.model.loudness() == -14.0
    s.start.button.click()
    assert wait_until(lambda: doc.stage == "done", 10)
    tl = runner.calls[0]["timeline"]
    assert tl["layers"]["noise"][0]["strength_db"] == 12.0 and tl["layers"]["leveling"][0]["loudness_lufs"] == -14.0
    entry = window.history.get(doc.entry.id)
    assert "Gentle" in entry.summary and "youtube" in entry.summary.lower()  # told in words, as before
    assert "chosen parts" not in entry.summary  # (it is the whole file, not parts)
    # the next file starts with what was chosen
    add_file(window, tmp_path, "next.wav")
    assert wait_until(lambda: window.doc.model is not None, 5)
    assert window.doc.model.strength_of(0, 11) in ("gentle", "balanced", "max")
    assert window.prefs.strength == "gentle" and window.prefs.loudness == "youtube"


def test_a_part_chosen_on_the_timeline_has_its_own_settings(window, tmp_path, runner):
    add_file(window, tmp_path)
    ready(window)
    s, doc = window.studio, window.doc
    s.top.switch.click()
    s.timeline.selectionDragged.emit(3.2, 6.7)  # drag across the timeline: whole seconds
    QApplication.processEvents()
    assert doc.model.selection == (3.0, 7.0) and s.inspector.heading.text() == "0:03–0:07 · 4 s"
    s.inspector.options["rebuild"].radio.click()
    QApplication.processEvents()
    assert doc.model.voice_mode_of(3, 7) == "rebuild" and doc.model.voice_mode_of(0, 11) == "mixed" and not s.inspector.voice_card.mixed.isVisibleTo(s.inspector)
    voice_lane = dict(s.timeline.plot.lanes())["voice"]
    assert [(x["start"], x["end"], x["label"]) for x in voice_lane] == [(0.0, 3.0, "Max"), (3.0, 7.0, "Rebuild"), (7.0, 11.0, "Max")]
    s.timeline.sectionClicked.emit(0.0, 3.0)  # a click on a section chooses it
    assert doc.model.selection == (0.0, 3.0)
    s.top.all_button.click()  # Select all: the settings of the whole file say that they differ from part to part
    QApplication.processEvents()
    assert doc.model.selection is None and s.inspector.heading.text() == "Whole file · 0:11"
    assert s.inspector.voice_card.mixed.isVisibleTo(s.inspector)
    # undo and redo are the tools of the editor
    s.top.undo_button.click()
    assert doc.model.voice_mode_of(0, 11) == "noise"
    QApplication.processEvents()
    assert s.top.redo_button.isEnabled()
    s.top.redo_button.click()
    assert doc.model.voice_mode_of(3, 7) == "rebuild"


# ---------------------------------------------------------------------------------------------------- a job of the list

@pytest.fixture
def job_window(window, tmp_path, monkeypatch):
    sh.FakeSession.made.clear()
    sh.SESSION_DELAY[0] = 0.0
    window.doc_env.open_session = sh.open_session
    folder = sh.make_folder(tmp_path)
    return window, folder


def test_a_past_job_opens_in_the_same_window_with_what_it_kept(job_window):
    window, folder = job_window
    job, eid = sh.add_job(window.history, folder)
    window._draw_jobs()
    assert list(window.jobs._cards) == [("entry", eid)]
    before = windows_showing()
    window.jobs._cards[("entry", eid)].clicked.emit()
    s, doc = window.studio, window.doc
    # at once: the same screen with the job's name, its stored waveform and the parts it was made with (nothing was decoded for it)
    assert windows_showing() == before and s.stack.currentIndex() == 1 and s.top.name.text() == "talk.wav" and doc.entry.id == eid
    assert doc.peaks[:4] == [30, 80, 50, 100] and doc.model.strength_of(0, 10) == "balanced" and doc.stage == "done"
    assert s.inspector.heading.text() == "Whole file · 0:10" and s.start.button.text() == "Open the file"
    assert s.top.hear_box.isVisible() and not s.top.auto_box.isVisible()  # a job has no Automatic: what can be listened to is there
    # the live sound follows in the background, and the player plays the result file meanwhile
    assert wait_until(lambda: doc.session_state == "ready", 10)
    QApplication.processEvents()
    assert s.preview is not None and s.preview.live and s.top.hear_box.isVisible()
    assert [lane for lane, _ in s.timeline.plot.lanes()] == ["voice", "level", "subs"]
    assert window.jobs.current == ("entry", eid)
    # another file replaces it in the same editor, and the job's live sound (its scratch files) is closed
    session = sh.FakeSession.made[-1]
    (folder.dir / "other.wav").write_bytes(b"x" * 100)
    window.add_paths([folder.dir / "other.wav"])
    assert s.top.name.text() == "other.wav" and session.closed and windows_showing() == before


def test_opening_a_job_does_not_wait_for_the_live_sound(job_window):
    window, folder = job_window
    sh.SESSION_DELAY[0] = 2.0
    _, eid = sh.add_job(window.history, folder)
    began = time.monotonic()
    window.open_entry(eid)
    assert time.monotonic() - began < 1.0
    doc = window.doc
    assert doc.session_state == "opening" and doc.peaks and doc.model is not None and window.studio.top.name.text() == "talk.wav"
    # the parts can be changed and what saving would take is known (from the facts the job kept) before the sound is ready
    doc.model.select(3, 5)
    doc.model.set_strength("gentle")
    assert doc.changed_sound and doc.quote() is not None
    assert wait_until(lambda: doc.session_state == "ready", 10)


def test_a_long_file_shows_at_once_and_the_window_keeps_answering(window, tmp_path):
    fakes = window.fakes
    fakes.seconds = 3600.0
    fakes.probe_delay, fakes.peaks_delay, fakes.decide_delay = 0.3, 1.2, 1.2
    began = time.monotonic()
    path = add_file(window, tmp_path, "long.wav")
    assert time.monotonic() - began < 1.0  # nothing is read on the window's thread (the stand-in probe alone takes 0.3 s)
    assert window.studio.stack.currentIndex() == 1 and window.studio.top.name.text() == "long.wav"
    gaps, last = [], [time.monotonic()]

    def tick():
        now = time.monotonic()
        gaps.append(now - last[0])
        last[0] = now

    timer = QTimer()
    timer.setInterval(10)
    timer.timeout.connect(tick)
    timer.start()
    seen_reading = []
    assert wait_until(lambda: seen_reading.append(window.doc.wave) or (window.doc.wave == "ready" and window.doc.applied), 30)
    timer.stop()
    assert "reading" in seen_reading and max(gaps) < 1.0  # it kept answering while the sound, then Automatic, were worked out
    assert window.studio.inspector.heading.text() == "Whole file · 1:00:00" and path.exists()
    assert fakes.calls[0] == "probe" and set(fakes.calls) == {"probe", "peaks", "decide"}  # (the waveform and Automatic are read side by side)


@pytest.mark.native
def test_a_real_ten_minute_file_shows_its_waveform_in_seconds_and_the_window_keeps_answering(native_window, speech_wav, tmp_path):
    from conftest import bundled_ffmpeg, run_ffmpeg

    w = native_window
    long = tmp_path / "long.wav"
    run_ffmpeg(bundled_ffmpeg(), "-stream_loop", "-1", "-i", str(speech_wav), "-t", "600", str(long))
    gaps, last = [], [time.monotonic()]

    def tick():
        now = time.monotonic()
        gaps.append(now - last[0])
        last[0] = now

    timer = QTimer()
    timer.setInterval(10)
    timer.timeout.connect(tick)
    began = time.monotonic()
    w.add_paths([long])
    assert time.monotonic() - began < 1.5  # (the editor shows at once: nothing is read on the window's thread)
    timer.start()
    shown = w.studio.stack.currentIndex() == 1 and w.studio.top.name.text() == "long.wav"
    assert shown
    doc = w.doc
    assert wait_until(lambda: doc.wave == "ready", 90)
    wave_took = time.monotonic() - began
    assert wait_until(lambda: doc.applied and not doc.listening, 120)
    timer.stop()
    assert wave_took < 60 and max(gaps) < 1.5, (wave_took, max(gaps))  # (about 1 s and 0.1 s on a desktop machine; the margin is for CI runners)
    assert len(doc.peaks) >= 1000 and doc.facts.duration == pytest.approx(600.0, abs=1) and w.studio.inspector.heading.text() == "Whole file · 10:00"
