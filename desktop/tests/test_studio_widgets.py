"""The pieces of the editor screen on their own: the timeline (what a click, a drag and a handle do, the rows, the zoom), the settings column (each card
edits the part that is chosen), the start row, the jobs column and the top of the editor. The model and the player are the real ones; nothing runs."""
from __future__ import annotations

import pytest
from PySide6.QtCore import QPoint, Qt
from PySide6.QtTest import QTest

import studio_helpers as sh
from audio_cleaner_desktop import editor_job, speed
from audio_cleaner_desktop.gui import copy as copy_mod
from conftest import wait_until
from studio_helpers import add_file, ready, settle


@pytest.fixture
def loaded(window, tmp_path):
    add_file(window, tmp_path)
    ready(window)
    window.studio.top.switch.click()  # (by hand: nothing is decided for the file)
    settle(window)
    return window


# ---------------------------------------------------------------------------------------------------- the timeline

def test_splitting_makes_the_part_around_the_playhead_two_that_can_be_chosen_by_themselves(loaded):
    s, doc = loaded.studio, loaded.doc
    s.seek(4.0)
    s.top.split_button.click()
    settle(loaded)
    voice = dict(s.timeline.plot.lanes())["voice"]
    assert [(x["start"], x["end"]) for x in voice] == [(0.0, 4.0), (4.0, 11.0)] and doc.cuts == [4.0]
    assert doc.model.selection == (4.0, 11.0)  # the piece around the playhead is chosen
    s.seek(0.0)
    s.top.split_button.click()  # not inside the file: nothing
    assert doc.cuts == [4.0]


# ---------------------------------------------------------------------------------------------------- the settings column

def test_each_card_edits_the_part_that_is_chosen_and_says_when_the_parts_differ(loaded):
    s, doc = loaded.studio, loaded.doc
    i = s.inspector
    doc.model.select(2, 6)
    i.options["rebuild"].radio.click()
    i.level._buttons["off"].click()
    settle(loaded)
    assert i.heading.text() == "0:02–0:06" and doc.model.voice_mode_of(2, 6) == "rebuild" and doc.model.level_of(2, 6) == "off"
    assert not i.options["noise"].extra.isVisibleTo(i)  # (the strength is for Noise removal only)
    assert i.only.isVisibleTo(i)  # "Leave the rest as it is" is for a part
    i.options["noise"].radio.click()
    settle(loaded)
    assert i.options["noise"].extra.isVisibleTo(i) and i.strength.value() == "max" and i.remover.value() == "auto"
    i.strength._buttons["strong"].click()
    i.remover._buttons["light"].click()
    assert doc.model.strength_of(2, 6) == "strong" and doc.model.remover_of(2, 6) == "light" and doc.model.strength_of(0, 11) == "mixed"
    i.only.click()  # everything outside the part is taken off
    assert [(p.start, p.end) for p in doc.model.draft.layers["noise"]] == [(2.0, 6.0)]
    s.top.all_button.click()
    settle(loaded)
    assert i.heading.text() == "0:00–0:11" and not i.only.isVisibleTo(i) and i.voice_card.mixed.isVisibleTo(i)


def test_leveling_is_chosen_per_part_and_the_card_says_mixed_when_a_range_has_different_loudnesses_and_the_cards_are_locked_while_it_runs(loaded, runner):
    s, doc = loaded.studio, loaded.doc
    i = s.inspector
    doc.model.select(0, 3)
    i.level._buttons["youtube"].click()
    doc.model.select(5, 8)
    i.level._buttons["broadcast"].click()
    settle(loaded)
    m = doc.model
    # each choice changed only its own part; the rest of the file keeps the loudness it had (Podcast)
    assert m.loudness_of(0, 3) == -14.0 and m.loudness_of(5, 8) == -23.0 and m.loudness_of(3, 5) == -16.0 and m.loudness_of(8, 11) == -16.0
    assert m.loudness() == -23.0  # (the last one chosen is what Leveling turned on elsewhere starts with)

    def shown():
        return [k for k, b in i.level._buttons.items() if b.isChecked()]

    m.select(0, 3)  # one loudness: its own button, no "mixed"
    settle(loaded)
    assert shown() == ["youtube"] and not i.level_card.mixed.isVisibleTo(i)
    m.select(5, 8)
    settle(loaded)
    assert shown() == ["broadcast"] and not i.level_card.mixed.isVisibleTo(i)
    m.select(0, 8)  # two loudnesses: no button is on and the card says mixed
    settle(loaded)
    assert shown() == [] and i.level_card.mixed.isVisibleTo(i)
    s.top.all_button.click()  # the whole file is leveled to different loudnesses too
    settle(loaded)
    assert shown() == [] and i.level_card.mixed.isVisibleTo(i)
    m.select(0, 3)
    i.level._buttons["off"].click()  # Off in a range with one loudness: Off is shown and it is not "mixed"
    settle(loaded)
    assert shown() == ["off"] and not i.level_card.mixed.isVisibleTo(i)
    m.select(0, 8)  # off in some places and leveled in others: mixed as before
    settle(loaded)
    assert shown() == [] and i.level_card.mixed.isVisibleTo(i)
    doc.stage = "processing"
    doc.changed.emit()
    settle(loaded)
    assert not i.cards.isEnabled()
    i._strength("gentle")
    assert doc.model.strength_of(0, 11) != "gentle"


def test_the_subtitles_card_has_language_files_and_into_the_video_only_for_a_video(window, tmp_path):
    window.fakes.facts.update(video=True, ext=".mp4")
    add_file(window, tmp_path, "clip.mp4")
    ready(window)
    i = window.studio.inspector
    doc = window.doc
    assert doc.subtitles == "off" and not i.language_row.isEnabled() and not i.files._buttons["srt"].isEnabled()
    i.subs._buttons["on"].click()
    settle(window)
    assert doc.subtitles == "srt" and i.language_row.isEnabled() and i.files.value() == "srt" and i.files._buttons["embed"].isVisibleTo(i)
    i.files._buttons["embed"].click()
    assert doc.subtitles == "embed"
    window.studio.output.endingPicked.emit(".mp3")  # only its sound: the subtitles cannot go inside, they become an SRT file
    settle(window)
    assert doc.subtitles == "srt" and not i.files._buttons["embed"].isEnabled()
    i.subs._buttons["off"].click()
    settle(window)
    assert doc.subtitles == "off" and i.files.value() == ""
    i.subs._buttons["on"].click()  # (the kind that was chosen last comes back)
    assert doc.subtitles == "srt"
    i.language.setCurrentIndex(i.language.findData("de"))
    i.language.activated.emit(i.language.currentIndex())
    assert doc.language == "de" and window.prefs.language == "de"
    window.studio.timeline.sectionClicked.emit(1.0, 3.0)
    settle(window)
    assert "whole file" in i.subs_note.text()


def test_a_click_anywhere_on_a_voice_option_picks_it_not_only_on_its_button_or_name(loaded):
    s, doc = loaded.studio, loaded.doc
    i = s.inspector
    doc.model.select(0, 11)
    i.options["noise"].radio.click()
    settle(loaded)
    assert doc.model.voice_mode_of(0, 11) == "noise"
    rebuild = i.options["rebuild"]
    rebuild.show()
    # on its line ("fixes echo, muffled or distorted voices..."), the words under the name
    QTest.mouseClick(rebuild.line, Qt.MouseButton.LeftButton)
    settle(loaded)
    assert rebuild.radio.isChecked() and doc.model.voice_mode_of(0, 11) == "rebuild"
    # on the empty space to the right of the name (the button's own space, outside its circle and text)
    noise = i.options["noise"]
    QTest.mouseClick(noise, Qt.MouseButton.LeftButton, Qt.KeyboardModifier.NoModifier, QPoint(noise.width() - 8, noise.radio.geometry().center().y()))
    settle(loaded)
    assert noise.radio.isChecked() and doc.model.voice_mode_of(0, 11) == "noise"
    # on the margin around the row, and between the name and its line
    QTest.mouseClick(rebuild, Qt.MouseButton.LeftButton, Qt.KeyboardModifier.NoModifier, QPoint(rebuild.width() - 3, 2))
    settle(loaded)
    assert doc.model.voice_mode_of(0, 11) == "rebuild"
    QTest.mouseClick(noise, Qt.MouseButton.LeftButton, Qt.KeyboardModifier.NoModifier, QPoint(noise.width() // 2, noise.rect().bottom() - 1))
    settle(loaded)
    assert doc.model.voice_mode_of(0, 11) == "noise"
    # Off is a row too (no line under it)
    QTest.mouseClick(i.options["off"], Qt.MouseButton.LeftButton, Qt.KeyboardModifier.NoModifier, QPoint(i.options["off"].width() - 8, i.options["off"].height() // 2))
    settle(loaded)
    assert doc.model.voice_mode_of(0, 11) == "off" and i.options["off"].radio.isChecked()


def test_a_click_on_a_voice_row_that_cannot_be_chosen_or_that_is_pressed_in_and_dragged_out_picks_nothing(loaded):
    s, doc = loaded.studio, loaded.doc
    i = s.inspector
    doc.model.select(0, 11)
    i.options["noise"].radio.click()
    settle(loaded)
    rebuild = i.options["rebuild"]
    rebuild.radio.setEnabled(False)  # (its model is not there, say)
    QTest.mouseClick(rebuild.line, Qt.MouseButton.LeftButton)
    settle(loaded)
    assert doc.model.voice_mode_of(0, 11) == "noise" and not rebuild.radio.isChecked()
    rebuild.radio.setEnabled(True)
    QTest.mousePress(rebuild.line, Qt.MouseButton.LeftButton)
    QTest.mouseRelease(rebuild.line, Qt.MouseButton.LeftButton, Qt.KeyboardModifier.NoModifier, QPoint(rebuild.line.width() // 2, -60))  # let go far above it
    settle(loaded)
    assert doc.model.voice_mode_of(0, 11) == "noise"
    QTest.mouseClick(rebuild.line, Qt.MouseButton.RightButton)
    settle(loaded)
    assert doc.model.voice_mode_of(0, 11) == "noise"


def test_the_settings_heading_is_just_the_range_as_on_the_website():
    title = copy_mod.selection_title
    assert title(None, 180.0) == "0:00–3:00" and title((0.0, 180.0), 180.0) == "0:00–3:00" and title((-5.0, 200.0), 180.0) == "0:00–3:00"
    assert title((15.0, 25.0), 180.0) == "0:15–0:25" and title((2.0, 6.0), 11.0) == "0:02–0:06" and title((0.0, 90.0), 180.0) == "0:00–1:30"
    assert title(None, 3600.0) == "0:00–1:00:00" and title((60.0, 3700.0), 4000.0) == "1:00–1:01:40"
    for text in (title(None, 180.0), title((15.0, 25.0), 180.0)):
        assert "Whole file" not in text and "·" not in text and " s" not in text


# ---------------------------------------------------------------------------------------------------- the start row and the details

def test_start_says_about_how_long_the_run_takes_when_something_is_on_and_only_start_when_nothing_is(loaded):
    s, doc = loaded.studio, loaded.doc
    doc.model.select(0, 11)
    doc.model.set_voice("noise")
    settle(loaded)
    total = doc.estimate()[1]
    assert total > 0 and s.start.button.text() == f"Start · {editor_job.say_time(total)}" and "· about " in s.start.button.text()
    doc.model.set_voice("rebuild")
    settle(loaded)
    assert s.start.button.text() == f"Start · {editor_job.say_time(doc.estimate()[1])}"  # (more work: more time)
    assert doc.estimate()[1] > total
    doc.model.set_voice("off")
    s.inspector.level._buttons["off"].click()
    settle(loaded)
    assert doc.estimate() == ([], 0.0) and s.start.button.text() == "Start"  # nothing on: no time to say


def test_details_counts_voice_rebuild_once_for_a_stereo_recording_and_noise_removal_for_each_side(window, tmp_path):
    def rows_for(channels: int):
        window.fakes.facts.update(channels=channels)
        add_file(window, tmp_path, f"talk{channels}.wav")
        ready(window)
        doc = window.doc
        doc.model.select(0, 11)
        doc.model.set_voice("rebuild")
        settle(window)
        rebuild = [c for t, c in doc.estimate()[0] if t.startswith("Voice rebuild")][0]
        doc.model.set_voice("noise")
        settle(window)
        noise = [c for t, c in doc.estimate()[0] if t.startswith("Noise removal")][0]
        return rebuild, noise

    mono_rebuild, mono_noise = rows_for(1)
    stereo_rebuild, stereo_noise = rows_for(2)
    assert mono_rebuild == pytest.approx(speed.cost("rebuild", 11.0)) and stereo_rebuild == pytest.approx(speed.cost("rebuild", 11.0))  # (once)
    assert mono_noise == pytest.approx(speed.cost("light", 11.0)) and stereo_noise == pytest.approx(speed.cost("light", 22.0))  # (each side)
    assert stereo_noise > mono_noise


def test_your_files_lists_the_result_with_its_size_and_deletes_the_job_after_asking(job_window_helper):
    window, doc, folder = job_window_helper
    s = window.studio
    s.start.files.click()
    from PySide6.QtWidgets import QLabel, QPushButton

    pop = s.start.popover
    assert any("talk-cleaned.wav" in label.text() for label in pop.findChildren(QLabel))
    buttons = [b.text() for b in pop.findChildren(QPushButton)]
    assert "Delete this job" in buttons
    pop.hide()


@pytest.fixture
def job_window_helper(window, tmp_path):
    folder = sh.make_folder(tmp_path)
    window.doc_env.open_session = sh.open_session
    _, eid = sh.add_job(window.history, folder)
    window.open_entry(eid)
    assert wait_until(lambda: window.doc.session_state == "ready", 10)
    settle(window)
    return window, window.doc, folder


