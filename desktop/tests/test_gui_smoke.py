"""The real window with the real engine (offscreen)."""
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


def test_window_opens_with_the_empty_state(window):
    assert window.windowTitle() == "Audio Cleaner"
    assert window.minimumWidth() >= 900 and window.minimumHeight() >= 600
    assert not window.primary.isEnabled()
    assert window.tabs.tabText(0) == "Clean" and window.tabs.tabText(1) == "History"
    assert not window.view.isVisible()


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
    assert entries[0].duration == pytest.approx(10.6, abs=0.5)
    assert window.history_page._stats["files"].value.text() == "1"


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


def test_rows_can_be_reordered_and_removed_before_starting(window, tmp_path, monkeypatch):
    monkeypatch.setattr(engine, "probe", lambda path, ff=None: (_ for _ in ()).throw(RuntimeError("not used")))
    window._missing = ["x"]  # keeps the background probe from running
    paths = []
    for name in ("a.wav", "b.wav", "c.wav"):
        (tmp_path / name).write_bytes(b"x")
        paths.append(tmp_path / name)
    window.add_paths(paths)
    ids = [r.item.id for r in window.view.rows()]
    window._move(ids[2], -1)
    assert [r.item.path.name for r in window.view.rows()] == ["a.wav", "c.wav", "b.wav"]
    window.remove_item(ids[0])
    assert [r.item.path.name for r in window.view.rows()] == ["c.wav", "b.wav"]
    assert all(r.item.state == WAITING for r in window.view.rows())


def test_dropping_a_folder_adds_the_media_inside(window, tmp_path):
    (tmp_path / "in" / "nested").mkdir(parents=True)
    (tmp_path / "in" / "a.mp3").write_bytes(b"x")
    (tmp_path / "in" / "nested" / "b.mp4").write_bytes(b"x")
    (tmp_path / "in" / "readme.txt").write_text("x")
    window._missing = ["x"]
    assert window.add_paths([tmp_path / "in"]) == 2


def test_about_dialog_has_the_notices(window):
    from PySide6.QtWidgets import QLabel

    from audio_cleaner_desktop.gui.about import AboutDialog

    dialog = AboutDialog(window)
    texts = " ".join(label.text() for label in dialog.findChildren(QLabel))
    assert "Free to use \u00b7 \u00a9 2026 Benjamin Danker" in texts
    assert "GNU LGPL v3" in texts and "ffmpeg-buildinfo.txt" in texts
    dialog.close()


def test_preview_works_without_multimedia(window, tmp_path):
    from audio_cleaner_desktop.gui.player import PreviewPane, load_multimedia

    pane = PreviewPane(window)
    pane.show_files(1, tmp_path / "a.wav", None, False, "a.wav")
    assert pane.available == (load_multimedia() is not None)
    pane.stop()
    pane.deleteLater()


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


def test_low_disk_space_shows_a_warning(window, tmp_path, monkeypatch):
    from collections import namedtuple

    from audio_cleaner_desktop.gui import main_window

    monkeypatch.setattr(main_window.shutil, "disk_usage", lambda p: namedtuple("U", "total used free")(10**10, 10**10 - 10**8, 10**8))
    window._missing = ["x"]
    (tmp_path / "long.wav").write_bytes(b"x")
    window.add_paths([tmp_path / "long.wav"])
    item = next(iter(window._items.values()))
    item.duration, item.channels = 2 * 3600.0, 2
    window._refresh_state()
    assert window.banner_disk.isVisible() and "enough free space" in window.banner_disk._label.text()
    window.remove_item(item.id)
    assert not window.banner_disk.isVisible()


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
