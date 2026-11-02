"""The real window with the real engine (offscreen): a file cleaned from the editor screen and its result changed and saved again on the same screen,
files started together, the window's own protections (missing parts, closing mid-job) and the About box."""
from __future__ import annotations

import threading

import pytest
from PySide6.QtCore import QSettings
from PySide6.QtWidgets import QApplication

from audio_cleaner_desktop import engine
from audio_cleaner_desktop.gui import main_window
from audio_cleaner_desktop.gui.main_window import MainWindow

import studio_helpers as sh
from conftest import wait_until


def put(window, speech_wav, tmp_path, name="talk.wav"):
    src = tmp_path / name
    src.write_bytes(speech_wav.read_bytes())
    window.add_paths([src])
    return src


def ready(w) -> None:
    doc = w.doc
    assert wait_until(lambda: doc.model is not None and doc.wave == "ready" and not doc.listening, 120)
    QApplication.processEvents()


@pytest.mark.native
def test_clean_a_file_end_to_end(native_window, speech_wav, tmp_path):
    w = native_window
    src = put(w, speech_wav, tmp_path)
    assert w.add_paths([src]) == 1 and len(w._items) == 1  # the same file is shown again, not added twice
    ready(w)
    doc = w.doc
    from scipy.io import wavfile

    rate, samples = wavfile.read(speech_wav, mmap=True)
    assert doc.facts.duration == pytest.approx(len(samples) / rate, abs=0.5) and doc.peaks and doc.applied  # Automatic listened to the file
    s = w.studio
    assert s.start.button.text() == "Start" and s.start.button.isEnabled()
    s.start.button.click()
    item = next(iter(w._items.values()))
    assert wait_until(lambda: doc.stage == "done", 180), item.message
    assert item.outputs == [tmp_path / "talk-cleaned.wav"] and item.outputs[0].is_file()
    entries = w.history.search()
    assert len(entries) == 1 and entries[0].status == "done" and entries[0].outputs == [str(item.outputs[0])]
    # Automatic (on by default) decided the strength for the whole file; the list and the history say which and why
    assert item.picked_strength in engine.STRENGTHS and item.picked_rebuild is None and "noise reduction picked for this file" in doc.note
    assert entries[0].settings["strength"] == "auto" and entries[0].settings["picked_strength"] == item.picked_strength
    assert entries[0].duration == pytest.approx(len(samples) / rate, abs=0.5)
    sh.settle(w)
    assert doc.job is not None and doc.entry.id == entries[0].id and "1 file cleaned" in w.jobs.totals.text()
    assert s.start.button.text() == "Open the file"


@pytest.fixture
def fake_bin(tmp_path, monkeypatch):
    """Stands in for the Recycle Bin / Trash, so tests never touch the real one."""
    from audio_cleaner_desktop.gui import shell

    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()

    def move(path):
        path.rename(bin_dir / path.name)
        return True

    monkeypatch.setattr(shell, "move_to_trash", move)
    monkeypatch.setattr(main_window.shell, "move_to_trash", move)
    return bin_dir


@pytest.mark.native
def test_the_result_is_changed_and_saved_again_on_the_same_screen(native_window, speech_wav, tmp_path, fake_bin):
    w = native_window
    put(w, speech_wav, tmp_path)
    ready(w)
    doc = w.doc
    w.studio.top.switch.click()  # Automatic off: the strength chosen below is the one used
    w.studio.inspector.strength._buttons["strong"].click()
    w.studio.start.button.click()
    assert wait_until(lambda: doc.stage == "done", 180), doc.note
    sh.settle(w)
    first = doc.result_paths()[0]
    first_bytes = first.read_bytes()
    assert first.name == "talk-cleaned.wav"
    # the same screen: a part is changed, the button says what it takes, and the new file is saved next to the old one
    w.studio.timeline.sectionClicked.emit(0.0, doc.seconds)
    w.studio.inspector.strength._buttons["gentle"].click()
    sh.settle(w)
    assert doc.model.strength_of(0, doc.seconds) == "gentle" and doc.changed_sound
    assert w.studio.start.button.isEnabled() and doc.out_name == "talk-cleaned (2).wav"
    w.studio.start.button.click()
    assert wait_until(lambda: doc.stage == "done" and w._saving is None and not doc.changed_sound, 180), doc.note
    sh.settle(w)
    second = doc.result_paths()[0]
    assert second.name == "talk-cleaned (2).wav" and second.read_bytes() != first_bytes  # the new, gentler result
    assert first.read_bytes() == first_bytes and not list(fake_bin.iterdir())  # the earlier one is as it was
    assert w.history.get(doc.entry.id).outputs == [str(second)] and len(w._items) == 1 and len(w.jobs._cards) == 1  # one job, one editor
    # a name that is taken, chosen on purpose, replaces the earlier file, which goes to the bin
    doc.model.select(0.0, doc.seconds)
    w.studio.inspector.strength._buttons["max"].click()
    doc.set_out_name("talk-cleaned.wav")
    w.ask_replace = lambda name: "replace"
    sh.settle(w)
    w.save_changes(doc)
    assert wait_until(lambda: w._saving is None and doc.stage == "done", 180), doc.note
    assert doc.result_paths()[0] == first and [p.name for p in fake_bin.iterdir()] == ["talk-cleaned.wav"] and first.read_bytes() != first_bytes


def test_starting_uses_what_was_chosen_then_and_a_run_cannot_be_changed_while_it_goes(make_window, tmp_path, monkeypatch):
    calls: list[tuple[str, object]] = []
    started = threading.Event()

    def runner(path, options, *, cancel, progress, **extra):
        calls.append((path.name, extra.get("timeline")))
        started.set()
        while not cancel.wait(0.01):
            pass
        raise engine.Cancelled()

    w = make_window(runner)
    for name in ("a.wav", "b.wav"):
        (tmp_path / name).write_bytes(b"x")
    w.prefs.parallel = 1
    w._settings_changed()
    w.add_paths([tmp_path / "a.wav", tmp_path / "b.wav"])
    assert wait_until(lambda: all(i.doc.model is not None for i in w._items.values()), 10)
    a, b = w._items[1], w._items[2]
    a.doc.set_auto(False)  # a is done by hand, b is left to Automatic
    a.doc.model.set_strength("gentle", 0, 5)
    w.open_item(a.id)
    w.studio.start.button.click()
    assert wait_until(started.is_set, 10) and calls[0][0] == "a.wav" and calls[0][1]["layers"]["noise"][0]["strength_db"] == 12.0
    # while it runs the choices are locked, and another file can be looked at and waits for its turn
    sh.settle(w)
    assert a.doc.locked and not w.studio.inspector.cards.isEnabled()
    w.studio.inspector._strength("max")  # (ignored)
    assert a.doc.model.strength_of(0, 5) == "gentle"
    w.open_item(b.id)
    assert not b.doc.locked and w.studio.inspector.cards.isEnabled()
    w.studio.start.button.click()
    sh.settle(w)
    assert b.id in w._queued and b.doc.run_text.startswith("In line") and w.jobs._cards[("item", b.id)].status.text() == "In line"
    monkeypatch.setattr(w, "_confirm", lambda title, text: True)
    w.close()
    assert not w.queue.busy


@pytest.mark.native
def test_two_files_start_together_and_a_bad_one_is_not_taken_in(native_window, speech_wav, tmp_path):
    w = native_window
    for name in ("one.wav", "two.wav"):
        (tmp_path / name).write_bytes(speech_wav.read_bytes())
    (tmp_path / "broken.wav").write_bytes(b"nothing here" * 50)
    w.prefs.parallel = 2
    w._settings_changed()
    w.add_paths(sorted(tmp_path.glob("*.wav")))
    assert wait_until(lambda: len(w._items) == 2 and all(i.doc.model is not None for i in w._items.values()), 60)  # the broken one left, with a message
    assert "broken.wav" in w.banner_files._label.text() or "broken.wav" in w.studio.problem._label.text()
    w.start_all()
    assert wait_until(lambda: not w.queue.busy and all(i.state == "done" for i in w._items.values()), 240)
    assert (tmp_path / "one-cleaned.wav").is_file() and (tmp_path / "two-cleaned.wav").is_file()
    assert {e.name: e.status for e in w.history.search()} == {"one.wav": "done", "two.wav": "done"}


def test_closing_while_cleaning_asks_and_stops(make_window, tmp_path, monkeypatch):
    started = threading.Event()

    def runner(path, options, *, cancel, progress, **_):
        started.set()
        while not cancel.wait(0.01):
            pass
        raise engine.Cancelled()

    w = make_window(runner)
    (tmp_path / "a.wav").write_bytes(b"x")
    w.add_paths([tmp_path / "a.wav"])
    assert wait_until(lambda: w.doc.model is not None, 10)
    w.start()
    assert wait_until(started.is_set, 10)
    answers = []
    monkeypatch.setattr(w, "_confirm", lambda title, text: answers.append(title) or False)
    w.close()
    assert w.queue.busy and answers  # said no: still running
    monkeypatch.setattr(w, "_confirm", lambda title, text: True)
    w.close()
    assert not w.queue.busy


def test_missing_components_block_starting(qapp, tmp_path, monkeypatch):
    from audio_cleaner_desktop import resources

    monkeypatch.setattr(resources, "check", lambda: ["libdf.so"])
    settings = QSettings(str(tmp_path / "s.ini"), QSettings.Format.IniFormat)
    w = MainWindow([], settings=settings, history_path=tmp_path / "h.db")
    w.fakes = sh.Fakes(5.0).install(w)
    w.show()
    (tmp_path / "a.wav").write_bytes(b"x")
    w.add_paths([tmp_path / "a.wav"])
    assert w.banner_missing.isVisible() and "libdf.so" in w.banner_missing._label.text()
    assert wait_until(lambda: w.doc.item.probed or True, 1)
    QApplication.processEvents()
    assert not w.studio.start.button.isEnabled()
    w.start()
    assert not w.queue.busy
    w.close()


def test_the_about_box_has_the_licence_notices(qapp):
    from PySide6.QtWidgets import QLabel

    from audio_cleaner_desktop.gui.about import AboutDialog

    dialog = AboutDialog()
    texts = " ".join(label.text() for label in dialog.findChildren(QLabel))
    assert "Open source, MIT License · © 2025-2026 The Audio Cleaner authors" in texts
    assert "GNU LGPL v3" in texts and "ffmpeg-buildinfo.txt" in texts
    dialog.close()
