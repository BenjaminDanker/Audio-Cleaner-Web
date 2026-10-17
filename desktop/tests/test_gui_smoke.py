"""The real window with the real engine (offscreen): a file cleaned from the window, the stop and the clean-again
behaviour, the automatic choice shown and kept, and the window's own protections (missing parts, closing mid-job)."""
from __future__ import annotations

from types import SimpleNamespace

import pytest
from PySide6.QtCore import QSettings

from audio_cleaner_desktop import engine
from audio_cleaner_desktop.gui.main_window import MainWindow
from audio_cleaner_desktop.gui.queue_view import DONE, FAILED, WAITING
from audio_cleaner_desktop.gui.theme import ThemeWatcher

from conftest import wait_until


@pytest.fixture
def window(qapp, tmp_path):
    ThemeWatcher(qapp)
    settings = QSettings(str(tmp_path / "settings.ini"), QSettings.Format.IniFormat)
    w = MainWindow([], settings=settings, history_path=tmp_path / "history.db")
    w.show()
    yield w
    w.queue.shutdown()
    w.history.close()
    w._pool.waitForDone()


@pytest.mark.native
def test_clean_a_file_end_to_end(window, speech_wav, tmp_path):
    src = tmp_path / "talk.wav"
    src.write_bytes(speech_wav.read_bytes())
    assert window.add_paths([src]) == 1
    assert window.add_paths([src]) == 0  # the same file is not queued twice
    assert window.primary.isEnabled() and window.primary.text() == "Clean 1 file"
    window.start()
    item = next(iter(window._items.values()))
    assert wait_until(lambda: item.state in (DONE, FAILED), timeout=120)
    assert item.state == DONE, item.message
    assert item.outputs == [tmp_path / "talk-cleaned.wav"] and item.outputs[0].is_file()
    assert window.status.text() == "1 file cleaned"
    entries = window.history.search()
    assert len(entries) == 1 and entries[0].status == "done" and entries[0].outputs == [str(item.outputs[0])]
    # Automatic best settings (on by default) picked the strength, and the list and the history say which
    assert item.picked_strength in engine.STRENGTHS and item.picked_rebuild is None
    assert "noise reduction picked for this file" in item.message
    assert entries[0].settings["strength"] == "auto" and entries[0].settings["picked_strength"] == item.picked_strength
    from scipy.io import wavfile

    rate, samples = wavfile.read(speech_wav, mmap=True)
    assert entries[0].duration == pytest.approx(len(samples) / rate, abs=0.5)
    assert window.history_page._stats["files"].value.text() == "0"  # not drawn while the tab is hidden
    window.tabs.setCurrentIndex(1)
    assert window.history_page._stats["files"].value.text() == "1"


@pytest.fixture
def fake_bin(tmp_path, monkeypatch):
    """Stands in for the Recycle Bin / Trash, so tests never touch the real one."""
    from audio_cleaner_desktop.gui import main_window, shell

    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()

    def move(path):
        path.rename(bin_dir / path.name)
        return True

    monkeypatch.setattr(shell, "move_to_trash", move)
    monkeypatch.setattr(main_window.shell, "move_to_trash", move)
    return bin_dir


@pytest.mark.native
def test_clean_again_keeps_the_row_and_the_name_and_takes_the_new_settings(window, speech_wav, tmp_path, fake_bin):
    src = tmp_path / "talk.wav"
    src.write_bytes(speech_wav.read_bytes())
    window.prefs.auto_settings = False  # the strength chosen below is the one used
    window.add_paths([src])
    window.start()
    item = next(iter(window._items.values()))
    assert wait_until(lambda: item.state == DONE, timeout=120), item.message
    first = item.outputs[0]
    first_bytes = first.read_bytes()

    window.settings_panel._strength.set_value("gentle")
    window.prefs.strength = "gentle"
    window.clean_again(item.id)
    assert wait_until(lambda: item.state in (DONE, FAILED), timeout=120)
    assert item.state == DONE, item.message
    assert len(window._items) == 1 and len(window.view) == 1  # same row
    assert item.outputs == [first] and first.name == "talk-cleaned.wav"  # same name, no "(2)"
    assert first.read_bytes() != first_bytes  # the new, gentler result
    assert [p.name for p in fake_bin.iterdir()] == ["talk-cleaned.wav"]  # the earlier one is in the bin
    assert not [p.name for p in tmp_path.iterdir() if "cleaned (" in p.name]  # no "(2)" copies left behind

    # Adding the same file again does not make a twin row either.
    assert window.add_paths([src]) == 1
    assert len(window.view) == 1 and item.state == WAITING


@pytest.mark.native
def test_clean_again_never_bins_a_result_that_was_changed(window, speech_wav, tmp_path, fake_bin, quick_clean):
    src = tmp_path / "talk.wav"
    src.write_bytes(speech_wav.read_bytes())
    window.add_paths([src])
    window.start()
    item = next(iter(window._items.values()))
    assert wait_until(lambda: item.state == DONE, timeout=120), item.message
    first = item.outputs[0]
    with open(first, "ab") as f:  # edited elsewhere since
        f.write(b"\0" * 10)
    window.clean_again(item.id)
    assert wait_until(lambda: item.state == DONE, timeout=120), item.message
    assert first.exists() and item.outputs[0].name == "talk-cleaned (2).wav"
    assert not any(fake_bin.iterdir())


def test_changed_settings_reach_waiting_files_and_a_running_one_can_restart_with_them(qapp, tmp_path, monkeypatch):
    import threading

    from audio_cleaner_desktop.gui import main_window
    from audio_cleaner_desktop.gui.queue_view import RUNNING

    calls: list[tuple[str, str]] = []
    started = threading.Event()

    def runner(path, options, *, cancel, progress, **_):
        calls.append((path.name, options.noise))
        started.set()
        while not cancel.wait(0.01):
            pass
        raise engine.Cancelled()

    monkeypatch.setattr(main_window.resources, "check", lambda: [])
    settings = QSettings(str(tmp_path / "s.ini"), QSettings.Format.IniFormat)
    w = MainWindow([], settings=settings, history_path=tmp_path / "h.db", runner=runner)
    w.show()
    for name in ("a.wav", "b.wav"):
        (tmp_path / name).write_bytes(b"x")
    monkeypatch.setattr(engine, "probe", lambda path, ff=None: SimpleNamespace(duration=5.0, has_video=False, audio_channels=1))
    w.prefs.parallel = 1
    w.prefs.auto_settings = False  # Background noise is chosen by hand
    w._settings_changed()
    w.add_paths([tmp_path / "a.wav", tmp_path / "b.wav"])
    assert wait_until(lambda: all(i.probed for i in w._items.values()), timeout=10)
    w.start()
    assert wait_until(started.is_set, timeout=10) and calls == [("a.wav", "auto")]
    a = next(i for i in w._items.values() if i.path.name == "a.wav")

    w.prefs.noise = "heavy"  # what the Background noise setting does
    w._settings_changed()
    assert [j.options.noise for j in w.queue._pending] == ["heavy"]  # b, still waiting, takes the new setting
    row = w.view.row(a.id)
    assert a.state == RUNNING and a.outdated and row._again.isVisibleTo(row)

    w.clean_again(a.id)  # the row's "Restart with the new settings"
    assert wait_until(lambda: len(calls) == 2, timeout=10)
    assert calls[1] == ("a.wav", "heavy")  # straight back in, before b, with the new setting
    monkeypatch.setattr(w, "_confirm", lambda title, text: True)
    w.close()


@pytest.mark.native
def test_two_files_in_parallel_and_a_bad_one(window, speech_wav, tmp_path):
    for name in ("one.wav", "two.wav"):
        (tmp_path / name).write_bytes(speech_wav.read_bytes())
    (tmp_path / "broken.wav").write_bytes(b"nothing here" * 50)
    window.prefs.parallel = 2
    window.add_paths(sorted(tmp_path.glob("*.wav")))
    window.start()
    assert wait_until(lambda: not window.queue.busy and all(i.state in (DONE, FAILED) for i in window._items.values()), 180)
    states = {i.path.name: i.state for i in window._items.values()}
    assert states == {"broken.wav": FAILED, "one.wav": DONE, "two.wav": DONE}
    assert (tmp_path / "one-cleaned.wav").is_file() and (tmp_path / "two-cleaned.wav").is_file()
    history = {e.name: e.status for e in window.history.search()}
    assert history == {"broken.wav": "failed", "one.wav": "done", "two.wav": "done"}
    window.clear_finished()
    assert len(window.view) == 0


def test_closing_while_cleaning_asks_and_stops(qapp, tmp_path, monkeypatch):
    import threading

    from audio_cleaner_desktop.gui import main_window

    started = threading.Event()

    def runner(path, options, *, cancel, progress, **_):
        started.set()
        while not cancel.wait(0.01):
            pass
        raise engine.Cancelled()

    monkeypatch.setattr(main_window.resources, "check", lambda: [])
    settings = QSettings(str(tmp_path / "s.ini"), QSettings.Format.IniFormat)
    w = MainWindow([], settings=settings, history_path=tmp_path / "h.db", runner=runner)
    w.show()
    (tmp_path / "a.wav").write_bytes(b"x")
    monkeypatch.setattr(engine, "probe", lambda path, ff=None: SimpleNamespace(duration=5.0, has_video=False, audio_channels=1))
    w.add_paths([tmp_path / "a.wav"])
    assert wait_until(lambda: next(iter(w._items.values())).probed, timeout=10)
    w.start()
    assert wait_until(started.is_set, timeout=10)
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
    w.show()
    (tmp_path / "a.wav").write_bytes(b"x")
    w.add_paths([tmp_path / "a.wav"])
    assert w.banner_missing.isVisible() and "libdf.so" in w.banner_missing._label.text()
    assert not w.primary.isEnabled()
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
