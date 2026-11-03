"""The output pane under the editor: the name of the result with the list of types joined to it (how the two follow each other, what happens with a
name that can't be used, that the choice reaches the job and the file that is written), and the video size."""
from __future__ import annotations

import threading

import pytest
from PySide6.QtCore import Qt
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QApplication

from audio_cleaner_desktop import engine

from conftest import wait_until
from studio_helpers import add_file, settle


def type_name(w, text):
    box = w.studio.output.name
    assert wait_until(lambda: box.isEnabled(), 5), "the name box is still off"  # (it is enabled at the screen's next redraw)
    box.setFocus()
    box.selectAll()
    if text:
        QTest.keyClicks(box, text)
    else:
        QTest.keyClick(box, Qt.Key.Key_Delete)
    settle(w)


def leave_the_box(w):
    """Leave the name box the way a person does (the focus goes to the file type list): that is what moves an ending typed into the box to the list."""
    o = w.studio.output
    assert o.name.hasFocus(), "the name box has no focus to leave"
    o.format.setFocus()
    settle(w)
    assert not o.name.hasFocus()


def pick(w, ending):
    box = w.studio.output.format
    index = box.findData(ending)
    assert index >= 0, f"{ending} is not in the list"
    box.setCurrentIndex(index)
    box.activated.emit(index)
    settle(w)


def open_file(w, tmp_path, name="talk.wav"):
    add_file(w, tmp_path, name)
    assert wait_until(lambda: w.doc.model is not None, 5)
    settle(w)
    return w.doc


def test_the_box_holds_only_the_name_and_the_list_the_ending_and_typing_or_picking_keep_each_other_in_step(window, tmp_path):
    doc = open_file(window, tmp_path)
    o = window.studio.output
    assert o.name.text() == "talk-cleaned" and o.format.currentData() == ".wav" and doc.out_name == "talk-cleaned.wav"
    assert not o.hint.isVisibleTo(o)  # (no tip line: only a problem with the name is said here)
    type_name(window, "interview")  # typing keeps the type's ending
    assert o.name.text() == "interview" and o.format.currentData() == ".wav" and doc.out_name == "interview.wav"
    assert doc.out_custom and doc.item.out_name == "interview.wav"
    pick(window, ".mp3")  # picking changes the ending, not what the box shows
    assert o.name.text() == "interview" and doc.out_name == "interview.mp3" and doc.item.out_name == "interview.mp3"
    type_name(window, "talk.flac")  # an ending typed into the box does not change the type while it is typed...
    assert o.name.text() == "talk.flac" and o.format.currentData() == ".mp3" and doc.out_name == "talk.flac.mp3"
    leave_the_box(window)  # ...it moves to the list when the box is left
    assert o.name.text() == "talk" and o.format.currentData() == ".flac" and doc.out_name == "talk.flac"
    type_name(window, "interview.OGG")  # capitals are fine
    leave_the_box(window)
    assert o.name.text() == "interview" and o.format.currentData() == ".ogg" and doc.out_name == "interview.ogg"
    type_name(window, "v1.2")  # ".2" is no type: it is part of the name
    leave_the_box(window)
    assert o.name.text() == "v1.2" and o.format.currentData() == ".ogg" and doc.out_name == "v1.2.ogg"
    pick(window, ".wav")
    assert o.name.text() == "v1.2" and doc.out_name == "v1.2.wav"
    type_name(window, "talk.xyz")  # a type this app does not know stays in the name
    leave_the_box(window)
    assert o.name.text() == "talk.xyz" and o.format.currentData() == ".wav" and doc.out_name == "talk.xyz.wav"
    type_name(window, "interview")
    pick(window, ".m4a")
    assert o.name.text() == "interview" and doc.out_name == "interview.m4a"
    pick(window, ".mp4")  # a sound can become a video
    assert o.name.text() == "interview" and doc.out_name == "interview.mp4" and "(plain black picture)" in o.format.itemText(o.format.currentIndex())
    assert o.size_box.isVisibleTo(o)  # and then it has a picture size to choose
    type_name(window, "clip.mp3")  # a sound is also fine as the type of what was a video
    leave_the_box(window)
    assert doc.out_name == "clip.mp3" and not o.size_box.isVisibleTo(o)


def test_leaving_the_box_with_x_mp3_switches_the_type_to_mp3_and_the_box_shows_x(window, tmp_path):
    doc = open_file(window, tmp_path)
    o = window.studio.output
    type_name(window, "x.mp3")
    assert o.format.currentData() == ".wav" and doc.out_name == "x.mp3.wav"  # (still being typed: the type is as it was)
    leave_the_box(window)
    assert o.format.currentData() == ".mp3" and doc.out_name == "x.mp3" and o.name.text() == "x" and doc.item.out_name == "x.mp3"
    assert "MP3" in o.format.itemText(o.format.currentIndex()).upper()


def test_leaving_the_box_with_an_ending_that_the_file_cannot_become_leaves_the_type(window, tmp_path):
    window.fakes.facts.update(video=True, ext=".mp4")
    doc = open_file(window, tmp_path, "clip.mp4")
    o = window.studio.output
    assert o.name.text() == "clip-cleaned" and o.format.currentData() == ".mp4"
    type_name(window, "clip.m4v")  # (only an .m4v video can become .m4v)
    leave_the_box(window)
    assert o.format.currentData() == ".mp4" and doc.out_name == "clip.m4v.mp4" and o.name.text() == "clip.m4v"


def test_the_name_box_and_the_file_type_list_are_one_joined_control(window, tmp_path):
    open_file(window, tmp_path)
    o = window.studio.output
    n, f = o.name.geometry(), o.format.geometry()
    assert n.top() == f.top() and n.height() == f.height()  # as tall as each other
    assert n.right() + 1 == f.left()  # no gap between the two: the list begins where the box ends
    sheet = QApplication.instance().styleSheet()  # (and the list has no left border of its own, so the two share one line)
    assert "QComboBox#FormatBox" in sheet and "border-left: 0px" in sheet.split("QComboBox#FormatBox {", 1)[1].split("}", 1)[0]


def test_a_name_that_is_the_usual_one_again_follows_the_settings_again(window, tmp_path):
    doc = open_file(window, tmp_path)
    type_name(window, "mine")
    assert doc.out_custom
    type_name(window, "talk-cleaned")
    assert not doc.out_custom
    pick(window, ".mp3")  # another type than the usual one is a choice too
    assert doc.out_custom
    pick(window, ".wav")
    assert not doc.out_custom


def test_a_name_that_cannot_be_used_gets_a_short_reason_and_it_goes_when_fixed(window, tmp_path):
    open_file(window, tmp_path)
    o = window.studio.output
    type_name(window, "no/good")
    assert "can't contain" in o.hint.text() and o.hint.isVisibleTo(o) and o.name.property("invalid") is True
    assert not window.studio.start.button.isEnabled() and any("can't contain" in t for t, _ in window.studio.note_lines(window.doc))
    type_name(window, "")  # nothing typed: only the ending is left
    assert "Type a name" in o.hint.text() and not window.studio.start.button.isEnabled() and o.name.text() == ""
    type_name(window, "fine")  # fixed: the reason goes away at once
    assert o.hint.text() == "" and not o.hint.isVisibleTo(o) and o.name.property("invalid") is False and window.studio.start.button.isEnabled()


def test_a_file_whose_name_is_unusable_is_left_out_of_a_start_and_the_others_go(make_window, tmp_path):
    started: list[tuple[str, str | None]] = []
    gate = threading.Event()

    def runner(path, options, *, cancel, progress, output_name=None, **_):
        started.append((path.name, output_name))
        gate.wait(5)
        return engine.JobResult(path, 1.0, False)

    w = make_window(runner)
    for name in ("good.wav", "bad.wav"):
        (tmp_path / name).write_bytes(b"x")
    w.add_paths([tmp_path / "good.wav", tmp_path / "bad.wav"])
    assert wait_until(lambda: all(i.doc.model is not None for i in w._items.values()), 10)
    good, bad = w._items[1], w._items[2]
    w.open_item(bad.id)
    type_name(w, "oops/x")
    w.start_all()
    assert wait_until(lambda: started, timeout=10)
    assert started == [("good.wav", "good-cleaned.wav")]  # the box's name goes with the job
    assert bad.state == "waiting" and bad.id not in w._queued
    w.open_item(bad.id)
    settle(w)
    assert any("Type" in t or "ending" in t or "can't" in t for t, _ in w.studio.note_lines(bad.doc)) and not w.studio.start.button.isEnabled()
    gate.set()
    assert wait_until(lambda: good.state == "done", timeout=10)
    type_name(w, "bad-cleaned")  # fixed: it can go
    w.studio.start.button.click()
    assert wait_until(lambda: bad.state == "done", timeout=10)
    assert started[1] == ("bad.wav", "bad-cleaned.wav")


def test_a_file_in_line_or_running_cannot_be_renamed_and_keeps_the_name_it_had(make_window, tmp_path):
    release = threading.Event()
    seen: list[tuple[str, str | None]] = []

    def runner(path, options, *, cancel, progress, output_name=None, **_):
        seen.append((path.name, output_name))
        release.wait(10)
        return engine.JobResult(path, 1.0, False)

    w = make_window(runner)
    try:
        w.prefs.parallel = 1
        w._settings_changed()
        for name in ("a.wav", "b.wav"):
            (tmp_path / name).write_bytes(b"x")
        w.add_paths([tmp_path / "a.wav", tmp_path / "b.wav"])
        assert wait_until(lambda: all(i.doc.model is not None for i in w._items.values()), 10)
        a, b = w._items[1], w._items[2]
        w.open_item(b.id)
        type_name(w, "second")  # b is renamed, then started: it is in line behind a
        pick(w, ".mp3")
        w.start_all()
        assert wait_until(lambda: seen, timeout=10)
        w.open_item(a.id)
        settle(w)
        assert a.state == "running" and not w.studio.output.isEnabled()  # busy: no editing
        w.open_item(b.id)
        settle(w)
        assert b.id in w._queued and not w.studio.output.isEnabled()
        release.set()
        assert wait_until(lambda: b.state == "done", timeout=10)
        assert seen == [("a.wav", "a-cleaned.wav"), ("b.wav", "second.mp3")]
    finally:
        release.set()


def test_the_usual_name_follows_the_audio_file_type_setting_and_a_typed_name_does_not(window, tmp_path):
    plain = open_file(window, tmp_path, "plain.wav")
    mine_item = None
    (tmp_path / "mine.wav").write_bytes(b"x")
    window.add_paths([tmp_path / "mine.wav"])
    assert wait_until(lambda: window.doc.model is not None, 5)
    settle(window)
    assert wait_until(lambda: window.studio.output.name.isEnabled(), 5)
    mine = window.doc
    type_name(window, "mine")
    pick(window, ".ogg")
    window.show_settings()
    box = window._settings_dialog._format
    box.setCurrentIndex(box.findData("flac"))
    settle(window)
    assert plain.out_name == "plain-cleaned.flac" and mine.out_name == "mine.ogg" and mine_item is None
    window.open_item(plain.item.id)
    settle(window)
    assert window.studio.output.name.text() == "plain-cleaned" and window.studio.output.format.currentData() == ".flac"
    window.fakes.facts.update(video=True, ext=".mp4")
    (tmp_path / "clip.mp4").write_bytes(b"x")
    window.add_paths([tmp_path / "clip.mp4"])
    assert wait_until(lambda: window.doc.model is not None, 5)
    settle(window)
    assert window.studio.output.name.text() == "clip-cleaned" and window.studio.output.format.currentData() == ".mp4"  # a video is not touched by the audio file type


def test_the_video_size_is_there_only_for_a_picture_and_is_remembered(window, tmp_path):
    window.fakes.facts.update(video=True, ext=".mp4")
    doc = open_file(window, tmp_path, "clip.mp4")
    o = window.studio.output
    assert o.size_box.isVisibleTo(o) and o.size.currentData() == "same" and "copied as it is" in o.size_hint.text()
    o.sizePicked.emit("720")
    settle(window)
    assert doc.video_size == "720" and window.prefs.video_size == "720" and "made again at this size" in o.size_hint.text()
    pick(window, ".mp3")  # only its sound: no picture, no size
    assert not o.size_box.isVisibleTo(o)
    assert window._wanted(doc).video_size == "720" if doc.is_job else True


def test_the_pane_says_where_the_result_is_saved_and_what_it_will_save(window, tmp_path):
    doc = open_file(window, tmp_path)
    o = window.studio.output
    assert o.where.text() == "Saved next to the original" and not o.next_to.isVisibleTo(o)
    assert "<b>talk-cleaned.wav</b>" in o.will.text() and o.will.text().startswith("When it’s done, these are saved by themselves")
    doc.set_subtitles("srt")
    settle(window)
    assert "talk-cleaned.wav and the subtitles (SRT)" in o.will.text()
    window.prefs.output_folder, window.prefs.output_mode = str(tmp_path / "out"), "folder"
    window._settings_changed()
    settle(window)
    assert o.where.text() == f"Saved in {tmp_path / 'out'}" and o.next_to.isVisibleTo(o)
    o.next_to.click()
    settle(window)
    assert window.prefs.output_mode == "next" and o.where.text() == "Saved next to the original"


@pytest.mark.native
def test_the_chosen_name_and_type_are_what_gets_written(native_window, tmp_path, clip_wav, quick_clean):
    w = native_window
    src = tmp_path / "in" / "talk.wav"
    src.parent.mkdir()
    src.write_bytes(clip_wav.read_bytes())
    w.add_paths([src])
    assert wait_until(lambda: w.doc.model is not None, 30)
    settle(w)
    type_name(w, "My talk.mp3")  # the ending typed into the box is the type once the box is left
    leave_the_box(w)
    assert w.studio.output.name.text() == "My talk" and w.studio.output.format.currentData() == ".mp3"
    w.studio.start.button.click()
    item = next(iter(w._items.values()))
    assert wait_until(lambda: item.state in ("done", "failed"), timeout=60)
    assert item.state == "done", item.message
    assert item.outputs == [src.parent / "My talk.mp3"] and (src.parent / "My talk.mp3").is_file()


@pytest.mark.native
def test_saving_again_under_another_name_makes_a_new_file_and_under_the_same_name_replaces(native_window, tmp_path, clip_wav, monkeypatch):
    from audio_cleaner_desktop.gui import main_window, shell

    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()

    def move(path):
        path.rename(bin_dir / path.name)
        return True

    monkeypatch.setattr(shell, "move_to_trash", move)
    monkeypatch.setattr(main_window.shell, "move_to_trash", move)
    w = native_window
    src = tmp_path / "talk.wav"
    src.write_bytes(clip_wav.read_bytes())
    w.add_paths([src])
    assert wait_until(lambda: w.doc.model is not None, 30)
    settle(w)
    doc = w.doc
    w.studio.start.button.click()
    assert wait_until(lambda: doc.stage == "done", timeout=60), doc.note
    settle(w)
    first = tmp_path / "talk-cleaned.wav"
    assert doc.result_paths() == [first]

    def save(name_or_ending, *, ending=False):
        (pick if ending else type_name)(w, name_or_ending)
        assert doc.changed_output
        w.studio.start.button.click()
        assert wait_until(lambda: w._saving is None and doc.stage == "done" and not doc.changed_output, 60), doc.note
        settle(w)

    type_name(w, "no/good")  # an unusable name does not start anything
    assert not w.studio.start.button.isEnabled() and w._saving is None
    w.primary()
    assert w._saving is None and doc.stage == "done"
    save("episode 1")  # another name: a new file next to the first, which stays where it is
    assert doc.result_paths() == [tmp_path / "episode 1.wav"] and first.is_file() and list(bin_dir.iterdir()) == []
    w.ask_replace = lambda name: "replace"
    save("episode 1")  # the name of the last result, on purpose: the new one takes its place and the old one goes to the bin
    assert doc.result_paths() == [tmp_path / "episode 1.wav"] and [p.name for p in bin_dir.iterdir()] == ["episode 1.wav"]
    assert not [p for p in tmp_path.iterdir() if "(2)" in p.name]
    save(".flac", ending=True)  # another type: the file of the other type is a new file
    assert doc.result_paths() == [tmp_path / "episode 1 (2).flac"] and (tmp_path / "episode 1.wav").is_file()  # (the name had moved on to the next free one)
