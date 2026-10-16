"""The name box under each file and the list of types joined to it: what it starts with, how the two follow each other,
what happens with a name that can't be used, and that the choice reaches the job."""
from __future__ import annotations

import threading
from types import SimpleNamespace

import pytest
from PySide6.QtCore import QSettings, Qt
from PySide6.QtTest import QTest

from audio_cleaner_desktop import engine
from audio_cleaner_desktop.gui import main_window
from audio_cleaner_desktop.gui.main_window import MainWindow
from audio_cleaner_desktop.gui.queue_view import DONE, FAILED, RUNNING, WAITING
from audio_cleaner_desktop.gui.theme import ThemeWatcher

from conftest import wait_until


def ready_window(qapp, tmp_path, monkeypatch, runner):
    """A window whose files are read (by a stand-in for the probe) and whose jobs run `runner`."""
    monkeypatch.setattr(main_window.resources, "check", lambda: [])
    monkeypatch.setattr(engine, "probe", lambda path, ff=None: SimpleNamespace(
        duration=5.0, has_video=False, audio_channels=1, ext=path.suffix, video_codec=None))
    return open_window(qapp, tmp_path, runner, missing=False)


def open_window(qapp, tmp_path, runner=engine.process, missing=True):
    ThemeWatcher(qapp)
    settings = QSettings(str(tmp_path / "settings.ini"), QSettings.Format.IniFormat)
    w = MainWindow([], settings=settings, history_path=tmp_path / "history.db", runner=runner)
    if missing:
        w._missing = ["x"]  # no file is read in the background: the rows start from what their ending says
    w.show()
    return w


def close_window(w):
    w.queue.shutdown()
    w.history.close()
    w._pool.waitForDone()


@pytest.fixture
def window(qapp, tmp_path):
    w = open_window(qapp, tmp_path)
    yield w
    close_window(w)


def add(window, tmp_path, *names):
    paths = []
    for name in names:
        (tmp_path / name).write_bytes(b"x")
        paths.append(tmp_path / name)
    window.add_paths(paths)
    items = [next(i for i in window._items.values() if i.path == p) for p in paths]
    return [window.view.row(i.id) for i in items]


def type_name(row, text):
    box = row._namebox
    box.setFocus()
    box.selectAll()
    if text:
        QTest.keyClicks(box, text)
    else:
        QTest.keyClick(box, Qt.Key.Key_Delete)


def pick(row, ending):
    index = row._format.findData(ending)
    assert index >= 0, f"{ending} is not in the list"
    row._format.setCurrentIndex(index)
    row._format.activated.emit(index)


def listed(row):
    return [row._format.itemData(i) for i in range(row._format.count()) if row._format.itemData(i)]


def test_the_box_starts_with_todays_result_name_and_the_list_has_only_what_the_file_can_become(window, tmp_path):
    song, clip = add(window, tmp_path, "song.mp3", "clip.mp4")
    assert song._namebox.text() == "song-cleaned.mp3" and song._format.currentData() == ".mp3"
    assert clip._namebox.text() == "clip-cleaned.mp4" and clip._format.currentData() == ".mp4"
    assert listed(song)[:6] == [".mp3", ".wav", ".flac", ".m4a", ".opus", ".ogg"]  # a sound: its own kind first
    assert listed(clip)[:5] == [".mp4", ".mov", ".mkv", ".flv", ".webm"]  # a video: the video types first
    assert ".avi" not in listed(clip) and ".wma" not in listed(song)
    assert song._format.currentText() == "MP3 audio" and clip._format.currentText() == "MP4 video"
    assert clip._format.itemText(clip._format.findData(".mp3")) == "MP3 audio"


def test_the_tip_about_the_box_is_there_while_files_are_in_the_list(window, tmp_path):
    assert not window.tip.isVisible()
    add(window, tmp_path, "song.mp3")
    assert window.tip.isVisible()
    assert window.tip.text() == "Tip: change the ending or pick a format to save as something else, e.g. .mp3 for just the sound."


def test_typing_an_ending_changes_the_list_and_picking_a_type_changes_the_ending(window, tmp_path):
    (row,) = add(window, tmp_path, "talk.wav")
    item = row.item
    type_name(row, "interview.mp3")
    assert row._format.currentData() == ".mp3" and item.out_name == "interview.mp3" and item.out_custom
    type_name(row, "interview.FLAC")  # capitals are fine
    assert row._format.currentData() == ".flac"
    type_name(row, "interview.mp")  # half an ending is no type: the list shows none
    assert row._format.currentIndex() == -1 and row._format.currentText() == ""
    pick(row, ".ogg")
    assert row._namebox.text() == "interview.mp.ogg"  # ".mp" is no ending we know, so it was part of the name
    type_name(row, "interview.mp3")
    pick(row, ".m4a")
    assert row._namebox.text() == "interview.m4a" and item.out_name == "interview.m4a"
    pick(row, ".mp4")  # a sound can become a video
    assert row._namebox.text() == "interview.mp4"


def test_picking_a_type_when_the_box_is_empty_starts_from_the_usual_name(window, tmp_path):
    (row,) = add(window, tmp_path, "talk.wav")
    type_name(row, "")
    assert row._namebox.text() == ""
    pick(row, ".opus")
    assert row._namebox.text() == "talk-cleaned.opus"
    assert row.item.out_custom  # not the type it started with: the choice is the user's now


def test_esc_in_the_name_box_only_leaves_the_box(window, tmp_path, monkeypatch):
    (row,) = add(window, tmp_path, "talk.wav")
    asked = []
    monkeypatch.setattr(window, "_confirm", lambda *a: asked.append(a) or True)
    window.queue.cancel_all = lambda: asked.append("stopped")
    row._namebox.setFocus()
    assert window.focusWidget() is row._namebox or row._namebox.hasFocus()
    window._escape()
    assert asked == [] and not row._namebox.hasFocus()


def test_a_name_that_is_the_usual_one_again_follows_the_settings_again(window, tmp_path):
    (row,) = add(window, tmp_path, "talk.wav")
    type_name(row, "mine.wav")
    assert row.item.out_custom
    type_name(row, "talk-cleaned.wav")
    assert not row.item.out_custom


def test_a_name_that_cannot_be_used_gets_a_short_red_reason_once_the_box_is_left(window, tmp_path):
    (row,) = add(window, tmp_path, "talk.wav")
    type_name(row, "talk-cleaned.xyz")
    assert not row._hint.isVisible()  # not while still typing
    row._namebox.clearFocus()
    row._namebox.editingFinished.emit()
    assert row._hint.isVisible() and "can't be saved as .xyz" in row._hint.text()
    assert row._namebox.property("invalid") is True and row._hint.property("state") == "failed"
    assert row._format.currentIndex() == -1
    type_name(row, "talk-cleaned.mp3")  # fixed: the reason goes away at once
    assert not row._hint.isVisible() and row._namebox.property("invalid") is False


@pytest.mark.parametrize("text", ["", "talk", "a/b.mp3", "con.mp3", ".mp3"])
def test_other_unusable_names_say_why(window, tmp_path, text):
    (row,) = add(window, tmp_path, "talk.wav")
    type_name(row, text)
    row._namebox.editingFinished.emit()
    assert row._hint.isVisible() and row._hint.text()


def test_starting_leaves_out_a_file_whose_name_is_unusable_and_says_so(qapp, tmp_path, monkeypatch):
    started: list[tuple[str, str | None]] = []
    gate = threading.Event()

    def runner(path, options, *, cancel, progress, output_name=None, **_):
        started.append((path.name, output_name))
        gate.wait(5)
        return engine.JobResult(path, 1.0, False)

    w = ready_window(qapp, tmp_path, monkeypatch, runner)
    try:
        good, bad = add(w, tmp_path, "good.wav", "bad.wav")
        assert wait_until(lambda: good.item.probed and bad.item.probed, timeout=10)
        type_name(bad, "oops")
        assert w.status.text() == "2 files ready, 1 can't start until its name is fixed"
        w.start()
        assert wait_until(lambda: started, timeout=10)
        assert started == [("good.wav", "good-cleaned.wav")]  # the box's name goes with the job
        assert bad.item.state == WAITING and bad._hint.isVisible()  # refused, with the reason on its row
        assert bad.item.id not in w.queue.pending_ids() + w.queue.running_ids()
        gate.set()
        assert wait_until(lambda: good.item.state == DONE, timeout=10)
        assert bad.item.state == WAITING
        # nothing to start at all: the queue is not left "active"
        w.start()
        assert not w.queue.active and not w.queue.busy
        type_name(bad, "bad-cleaned.wav")  # fixed: it can go
        assert not bad._hint.isVisible()
        w.start()
        assert wait_until(lambda: bad.item.state == DONE, timeout=10)
        assert started[1] == ("bad.wav", "bad-cleaned.wav")
    finally:
        gate.set()
        close_window(w)


def test_the_name_of_a_file_already_in_line_follows_the_box_and_a_bad_one_waits(qapp, tmp_path, monkeypatch):
    release = threading.Event()
    seen: list[tuple[str, str | None]] = []

    def runner(path, options, *, cancel, progress, output_name=None, **_):
        seen.append((path.name, output_name))
        release.wait(10)
        return engine.JobResult(path, 1.0, False)

    w = ready_window(qapp, tmp_path, monkeypatch, runner)
    try:
        w.prefs.parallel = 1
        w._settings_changed()
        a, b = add(w, tmp_path, "a.wav", "b.wav")
        assert wait_until(lambda: a.item.probed and b.item.probed, timeout=10)
        w.start()
        assert wait_until(lambda: seen, timeout=10)
        assert a.item.state == RUNNING and not a._namebox.isEnabled() and not a._format.isEnabled()  # busy: no editing
        assert b._namebox.isEnabled()
        type_name(b, "second.mp3")  # b is in line behind a
        assert [(j.id, j.output_name) for j in w.queue._pending] == [(b.item.id, "second.mp3")]
        type_name(b, "second.exe")  # not good: out of the line until it is
        assert w.queue.pending_ids() == [] and b.item.state == WAITING
        type_name(b, "second.flac")  # good again: back in line, with the new name
        assert [(j.id, j.output_name) for j in w.queue._pending] == [(b.item.id, "second.flac")]
        release.set()
        assert wait_until(lambda: b.item.state == DONE, timeout=10)
        assert seen == [("a.wav", "a-cleaned.wav"), ("b.wav", "second.flac")]
    finally:
        release.set()
        close_window(w)


def test_the_usual_name_follows_the_audio_file_type_setting_and_a_typed_name_does_not(window, tmp_path):
    plain, mine = add(window, tmp_path, "plain.wav", "mine.wav")
    type_name(mine, "mine.ogg")
    window.settings_panel._format.setCurrentIndex(window.settings_panel._format.findData("flac"))
    assert plain._namebox.text() == "plain-cleaned.flac" and plain._format.currentData() == ".flac"
    assert mine._namebox.text() == "mine.ogg"
    video, = add(window, tmp_path, "clip.mp4")
    assert video._namebox.text() == "clip-cleaned.mp4"  # a video is not touched by the audio file type


def test_the_list_of_types_follows_what_the_file_turns_out_to_be(qapp, tmp_path, monkeypatch):
    seen = {}

    def probe(path, ff=None):
        return SimpleNamespace(duration=5.0, has_video=path.suffix == ".mp4", audio_channels=2, ext=path.suffix,
                               video_codec="vp9" if path.suffix == ".mp4" else None)

    monkeypatch.setattr(engine, "probe", probe)
    w = open_window(qapp, tmp_path, missing=False)
    try:
        (row,) = add(w, tmp_path, "weird.mp4")
        assert not row._namebox.isEnabled()  # still being read
        assert wait_until(lambda: row.item.probed, timeout=10)
        assert row._namebox.isEnabled() and row._namebox.text() == "weird-cleaned.mp4"
        assert row.item.source == engine.Source(True, ".mp4", "vp9", 5.0, 2)
        assert listed(row)[:2] == [".mp4", ".mov"]
    finally:
        close_window(w)


def test_a_file_that_cannot_be_read_has_no_name_box(qapp, tmp_path, monkeypatch):
    from acw_common.ffmpeg import MediaError

    monkeypatch.setattr(engine, "probe", lambda path, ff=None: (_ for _ in ()).throw(MediaError("not media")))
    w = open_window(qapp, tmp_path, missing=False)
    try:
        (row,) = add(w, tmp_path, "broken.wav")
        assert wait_until(lambda: row.item.state == FAILED, timeout=10)
        assert not row._editor.isVisibleTo(row)
    finally:
        close_window(w)


# -- with the real engine ----------------------------------------------------------------------------------------


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
def test_the_chosen_name_and_type_are_what_gets_written(qapp, tmp_path, clip_wav, quick_clean):
    src = tmp_path / "in" / "talk.wav"
    src.parent.mkdir()
    src.write_bytes(clip_wav.read_bytes())
    w = open_window(qapp, tmp_path, missing=False)
    try:
        w.add_paths([src])
        row = w.view.rows()[0]
        assert wait_until(lambda: row.item.probed, timeout=30)
        type_name(row, "My talk.mp3")
        w.start()
        assert wait_until(lambda: row.item.state in (DONE, FAILED), timeout=60)
        assert row.item.state == DONE, row.item.message
        assert row.item.outputs == [src.parent / "My talk.mp3"] and (src.parent / "My talk.mp3").is_file()
        assert "Saved as My talk.mp3" in row._meta.text()
    finally:
        close_window(w)


@pytest.mark.native
def test_clean_again_with_a_changed_name_makes_a_new_file_and_with_the_same_name_replaces(qapp, tmp_path, clip_wav, quick_clean, fake_bin):
    src = tmp_path / "talk.wav"
    src.write_bytes(clip_wav.read_bytes())
    w = open_window(qapp, tmp_path, missing=False)
    try:
        w.add_paths([src])
        row = w.view.rows()[0]
        item = row.item
        w.start()
        assert wait_until(lambda: item.state == DONE, timeout=60), item.message
        first = tmp_path / "talk-cleaned.wav"
        assert item.outputs == [first]

        # another name: a new file next to the first, which stays where it is
        type_name(row, "episode 1.wav")
        w.clean_again(item.id)
        assert wait_until(lambda: item.state == DONE, timeout=60), item.message
        assert item.outputs == [tmp_path / "episode 1.wav"] and first.is_file()
        assert list(fake_bin.iterdir()) == []

        # the same name as the last time: the new result takes the place of the old one, which goes to the bin
        w.clean_again(item.id)
        assert wait_until(lambda: item.state == DONE, timeout=60), item.message
        assert item.outputs == [tmp_path / "episode 1.wav"]
        assert [p.name for p in fake_bin.iterdir()] == ["episode 1.wav"]
        assert not [p for p in tmp_path.iterdir() if "(2)" in p.name]

        # the type changed in the list: the file of the other type is a new file, too
        pick(row, ".flac")
        assert row._namebox.text() == "episode 1.flac"
        w.clean_again(item.id)
        assert wait_until(lambda: item.state == DONE, timeout=60), item.message
        assert item.outputs == [tmp_path / "episode 1.flac"] and (tmp_path / "episode 1.wav").is_file()
    finally:
        close_window(w)


@pytest.mark.native
def test_clean_again_will_not_start_with_an_unusable_name(qapp, tmp_path, clip_wav, quick_clean):
    src = tmp_path / "talk.wav"
    src.write_bytes(clip_wav.read_bytes())
    w = open_window(qapp, tmp_path, missing=False)
    try:
        w.add_paths([src])
        row = w.view.rows()[0]
        w.start()
        assert wait_until(lambda: row.item.state == DONE, timeout=60), row.item.message
        type_name(row, "no/good.wav")
        w.clean_again(row.item.id)
        assert row.item.state == WAITING and row._hint.isVisible() and not w.queue.busy and not w.queue.active
    finally:
        close_window(w)


@pytest.mark.native
def test_a_sound_can_become_a_video_from_the_window(qapp, tmp_path, clip_wav, quick_clean):
    from conftest import displayed_size

    src = tmp_path / "talk.wav"
    src.write_bytes(clip_wav.read_bytes())
    w = open_window(qapp, tmp_path, missing=False)
    try:
        w.add_paths([src])
        row = w.view.rows()[0]
        assert wait_until(lambda: row.item.probed, timeout=30)
        pick(row, ".mkv")
        w.start()
        assert wait_until(lambda: row.item.state in (DONE, FAILED), timeout=60)
        assert row.item.state == DONE, row.item.message
        assert displayed_size(row.item.outputs[0]) == (1280, 720)
    finally:
        close_window(w)
