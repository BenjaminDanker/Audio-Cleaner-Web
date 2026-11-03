"""The pieces of the editor screen on their own: the timeline (what a click, a drag and a handle do, the rows, the zoom), the settings column (each card
edits the part that is chosen), the start row, the jobs column and the top of the editor. The model and the player are the real ones; nothing runs."""
from __future__ import annotations

import pytest
from PySide6.QtCore import QPoint, QPointF, Qt
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QApplication, QLabel

import studio_helpers as sh
from audio_cleaner_desktop import editor_job, speed
from audio_cleaner_desktop.gui import copy as copy_mod, timeline as tl_mod
from conftest import wait_until
from studio_helpers import add_file, ready, settle


def plot_of(window):
    plot = window.studio.timeline.plot
    plot.resize(plot.width(), plot.total_height())
    return plot


def press(plot, x, y):
    QTest.mousePress(plot, Qt.MouseButton.LeftButton, Qt.KeyboardModifier.NoModifier, QPoint(int(x), int(y)))


def move(plot, x, y):
    from PySide6.QtGui import QMouseEvent
    from PySide6.QtCore import QEvent

    event = QMouseEvent(QEvent.Type.MouseMove, QPointF(x, y), plot.mapToGlobal(QPointF(x, y)), Qt.MouseButton.NoButton, Qt.MouseButton.LeftButton,
                        Qt.KeyboardModifier.NoModifier)
    QApplication.sendEvent(plot, event)


def release(plot, x, y):
    QTest.mouseRelease(plot, Qt.MouseButton.LeftButton, Qt.KeyboardModifier.NoModifier, QPoint(int(x), int(y)))


@pytest.fixture
def loaded(window, tmp_path):
    add_file(window, tmp_path)
    ready(window)
    window.studio.top.switch.click()  # (by hand: nothing is decided for the file)
    settle(window)
    return window


# ---------------------------------------------------------------------------------------------------- the timeline

def test_a_click_on_the_ruler_or_the_waveform_moves_the_playhead_and_a_drag_chooses_a_part(loaded):
    plot = plot_of(loaded)
    seen, picked = [], []
    plot.seekRequested.connect(seen.append)
    plot.selectionDragged.connect(lambda a, b: picked.append((a, b)))
    w = plot.width()
    y_wave = tl_mod.RULER_H + 40
    press(plot, w * 0.5, 10)  # the ruler
    release(plot, w * 0.5, 10)
    assert seen and seen[-1] == pytest.approx(5.5, abs=0.1)
    press(plot, w * 0.25, y_wave)
    release(plot, w * 0.25, y_wave)  # a click on the waveform: the playhead, no part
    assert seen[-1] == pytest.approx(11 * 0.25, abs=0.1) and not picked
    press(plot, w * 0.2, y_wave)
    move(plot, w * 0.5, y_wave)  # a drag: a part
    release(plot, w * 0.5, y_wave)
    assert picked and picked[-1][0] == pytest.approx(2.2, abs=0.15) and picked[-1][1] == pytest.approx(5.5, abs=0.15)


def test_the_handles_of_the_chosen_part_are_dragged_and_stay_inside_the_timeline(loaded):
    doc = loaded.doc
    plot = plot_of(loaded)
    picked = []
    plot.selectionDragged.connect(lambda a, b: picked.append((a, b)))
    doc.model.select(3, 7)
    settle(loaded)
    (l0, l1), (r0, r1) = plot._handle_rects(3, 7)
    assert l1 == pytest.approx(plot.x_of(3), abs=1) and r0 == pytest.approx(plot.x_of(7), abs=1)  # the start's sits left of the start, the end's right
    assert plot.handle_at((l0 + l1) / 2, tl_mod.RULER_H + 40) == "h0" and plot.handle_at((r0 + r1) / 2, tl_mod.RULER_H + 40) == "h1"
    press(plot, (r0 + r1) / 2, tl_mod.RULER_H + 40)
    move(plot, plot.x_of(9), tl_mod.RULER_H + 40)
    release(plot, plot.x_of(9), tl_mod.RULER_H + 40)
    assert picked[-1][0] == 3 and picked[-1][1] == pytest.approx(9, abs=0.2)
    # a whole-file choice has its handles at the two edges, inside the timeline
    (l0, l1), (r0, r1) = plot._handle_rects(0, 11)
    assert l0 >= 0 and r1 <= plot.width() + 1e-6


def test_a_click_on_a_section_chooses_it_and_the_rows_say_what_each_part_does(loaded):
    doc = loaded.doc
    plot = plot_of(loaded)
    doc.model.select(3, 7)
    doc.model.set_voice("rebuild")
    doc.model.select(0, 11)
    doc.model.clear_selection()
    settle(loaded)
    lanes = dict(plot.lanes())
    assert [(x["start"], x["end"], x["label"]) for x in lanes["voice"]] == [(0.0, 3.0, "Max"), (3.0, 7.0, "Rebuild"), (7.0, 11.0, "Max")]
    got = []
    plot.sectionClicked.connect(lambda a, b: got.append((a, b)))
    y = plot.lane_top(0) + 10
    press(plot, plot.x_of(5), y)
    release(plot, plot.x_of(5), y)
    assert got == [(3.0, 7.0)]
    # there is no Music row any more
    assert [lane for lane, _ in plot.lanes()] == ["voice", "level", "subs"]


def test_there_is_no_music_row_in_the_timeline_its_names_or_the_rows_before_a_file(window, tmp_path):
    assert "music" not in tl_mod.LANE_NAMES and sorted(tl_mod.LANE_NAMES) == ["echo", "level", "subs", "voice"]
    ghost = [label.text() for label in window.studio.ghost.findChildren(QLabel)]
    assert ghost == ["Voice", "Leveling", "Subtitles"]  # (the empty screen's rows: nothing "coming")
    add_file(window, tmp_path)
    ready(window)
    plot = plot_of(window)
    assert [lane for lane, _ in plot.lanes()] == ["voice", "level", "subs"]
    assert plot.total_height() < plot.lane_top(2) + 2 * tl_mod.LANE_H  # (no room for a fourth row under Subtitles)
    names = " ".join(label.text() for label in window.studio.timeline.labels.findChildren(QLabel))
    assert "Music" not in names and "Coming" not in names


def test_the_timeline_row_for_subtitles_says_embed_for_a_video_that_gets_them_inside(window, tmp_path):
    window.fakes.facts.update(video=True, ext=".mp4")
    add_file(window, tmp_path, "clip.mp4")
    ready(window)
    i = window.studio.inspector
    i.subs._buttons["on"].click()
    settle(window)
    i.files._buttons["embed"].click()
    settle(window)
    assert window.doc.subtitles == "embed"
    subs = dict(plot_of(window).lanes())["subs"]
    assert [(x["label"], x["on"]) for x in subs] == [("Embed", True)]
    i.files._buttons["vtt"].click()
    settle(window)
    assert dict(plot_of(window).lanes())["subs"][0]["label"] == "VTT"


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


def test_the_zoom_makes_the_timeline_wider_than_the_window_and_scrolls(loaded):
    s = loaded.studio
    tl = s.timeline
    base = tl.plot.width()
    s.top.zoom_in_button.click()
    s.top.zoom_in_button.click()
    assert tl.zoom == 4 and tl.plot.width() == pytest.approx(base * 4, abs=4) and tl.scroll.horizontalScrollBar().maximum() > 0
    assert s.top.zoom_in_button.isEnabled()
    for _ in range(3):
        s.top.zoom_in_button.click()
    assert tl.zoom == 8 and not s.top.zoom_in_button.isEnabled()
    s.seek(10.0)
    tl.ensure_playhead_visible()
    assert tl.scroll.horizontalScrollBar().value() > 0
    for _ in range(4):
        s.top.zoom_out_button.click()
    assert tl.zoom == 1 and not s.top.zoom_out_button.isEnabled() and tl.scroll.horizontalScrollBar().value() == 0


def test_while_the_sound_is_read_the_timeline_says_so_and_while_it_runs_it_shows_how_far(window, tmp_path):
    window.fakes.peaks_delay = 0.8
    add_file(window, tmp_path)
    assert wait_until(lambda: window.doc.model is not None and window.doc.wave == "reading", 5)
    plot = plot_of(window)
    plot.update()
    assert window.doc.wave == "reading"
    assert wait_until(lambda: window.doc.wave == "ready", 10)


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


def test_leveling_calls_its_fifth_option_even_everywhere_and_it_still_means_the_loudness_as_recorded(loaded):
    s, doc = loaded.studio, loaded.doc
    i = s.inspector
    assert [b.text() for b in i.level._buttons.values()] == ["Off", "Podcast", "YouTube", "Broadcast", "Even"]
    assert "Even" in i.level._buttons["keep"].toolTip() and "Keep" not in i.level._buttons["keep"].toolTip() and i.level._buttons["keep"].toolTip().startswith("Even:")
    assert copy_mod.loudness_name(None) == "Even" and copy_mod.TIPS["l.4"].startswith("Even: ") and "Keep" not in copy_mod.TIPS["l.4"]
    assert ("Even", "Evens out the volume but keeps the overall loudness as recorded") in copy_mod.LEVELING_ROWS and all(n != "Keep" for n, _ in copy_mod.LEVELING_ROWS)
    doc.model.select(0, 11)
    i.level._buttons["keep"].click()  # (the same choice as before: no target, the loudness stays)
    settle(loaded)
    assert doc.model.loudness() is None and doc.model.level_of(0, 11) == "on"
    assert [x["label"] for x in dict(plot_of(loaded).lanes())["level"]] == ["Even"]
    from audio_cleaner_desktop.gui.info_dialog import InfoDialog

    dialog = InfoDialog()
    text = " ".join(label.text() for label in dialog.findChildren(QLabel))
    assert "Evens out the volume but keeps the overall loudness as recorded" in text and "Keep loudness" not in text
    dialog.close()
    assert not hasattr(copy_mod, "NAME_TIP")  # (the tip line under the name box is gone)


def test_the_subtitles_say_on_and_embed_and_the_files_row_has_its_label_above_the_file_types_so_none_is_cut_off(window, tmp_path):
    window.fakes.facts.update(video=True, ext=".mp4")
    add_file(window, tmp_path, "clip.mp4")
    ready(window)
    i = window.studio.inspector
    assert [b.text() for b in i.subs._buttons.values()] == ["Off", "On"]
    assert [b.text() for b in i.files._buttons.values()] == ["SRT", "VTT", "TXT", "Embed"]
    i.subs._buttons["on"].click()
    settle(window)
    row = i.files_row
    assert row.isVisibleTo(i)
    label = next(w for w in row.findChildren(QLabel) if w.text() == "Files")
    pills = list(i.files._buttons.values())
    wait_until(lambda: label.geometry().bottom() <= i.files.geometry().top() and pills[-1].geometry().right() <= i.files.width(), 5)  # (the row is laid out)
    assert label.geometry().bottom() <= i.files.geometry().top() and label.geometry().left() <= i.files.geometry().left()  # (above, not beside)
    assert all(b.isVisibleTo(i) for b in pills)
    assert pills[-1].geometry().right() <= i.files.width() and i.files.geometry().right() <= row.width()  # (every pill is inside the row)
    assert i.files.minimumSizeHint().width() <= row.width()
    assert all(b.width() >= b.fontMetrics().horizontalAdvance(b.text()) for b in pills)  # (and wide enough for its own word)


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


def test_details_and_your_files_sit_beside_the_start_button_and_their_popover_opens_right_aligned_above_the_link(loaded):
    s, doc = loaded.studio, loaded.doc
    bar = s.start
    lay = bar.layout()
    order = [("spacer" if lay.itemAt(n).spacerItem() is not None else lay.itemAt(n).widget()) for n in range(lay.count())]
    stretch = order.index("spacer")
    assert order.index(bar.details) > stretch and order.index(bar.files) > stretch and order.index(bar.button) > order.index(bar.files)  # (after the stretch)
    assert order.index(bar.progress_box) < stretch
    assert bar.details.isVisibleTo(bar) and not bar.files.isVisibleTo(bar)
    wait_until(lambda: bar.details.geometry().right() < bar.button.geometry().left() and bar.details.geometry().left() > bar.width() // 2, 5)  # (laid out)
    d, b = bar.details.geometry(), bar.button.geometry()
    assert d.right() < b.left() and b.left() - d.right() <= 30  # (right beside the button, not at the left of the bar)
    assert d.left() > bar.width() // 2
    bar.set_mode("files")  # Your files is in the same place
    settle(loaded)
    wait_until(lambda: bar.button.geometry().left() - bar.files.geometry().right() <= 30, 5)  # (laid out)
    assert bar.files.isVisibleTo(bar) and not bar.details.isVisibleTo(bar) and bar.button.geometry().left() - bar.files.geometry().right() <= 30


@pytest.mark.parametrize("which", ["details", "files"])
def test_the_popover_of_details_and_your_files_opens_above_the_link_with_its_right_side_at_the_links_right_side(loaded, which):
    bar = loaded.studio.start
    bar.set_fillers(lambda pop: pop.note("What will be done"), lambda pop: pop.note("The files this made"))  # (what is in it does not matter here)
    bar.set_mode(which)
    link_widget = getattr(bar, which)
    settle(loaded)
    link = link_widget.mapToGlobal(QPoint(0, 0))
    link_widget.click()
    pop = bar.popover
    assert pop.isVisible()
    wait_until(lambda: abs(pop.x() + pop.width() - (link.x() + link_widget.width())) <= 2, 3)  # (a popup window takes its place a few events after it shows, on some platforms)
    assert pop.x() + pop.width() == pytest.approx(link.x() + link_widget.width(), abs=2), (  # (its right side is the link's right side)
        f"{which}: the popover's right edge is at {pop.x() + pop.width()}, the link's at {link.x() + link_widget.width()} "
        f"(the popover is {pop.width()} wide, its size hint {pop.sizeHint().width()})")
    assert pop.y() + pop.height() <= link.y() and link.y() - (pop.y() + pop.height()) < 40  # (above the link, close to it)
    assert pop.x() >= bar.window().mapToGlobal(QPoint(0, 0)).x()
    pop.hide()


def test_details_lists_what_will_be_done_and_the_total_and_says_nothing_is_billed(loaded):
    s, doc = loaded.studio, loaded.doc
    doc.model.select(2, 6)
    doc.model.set_voice("rebuild")
    doc.set_subtitles("srt")
    settle(loaded)
    pop = s.start.popover
    s.start.details.click()
    texts = [label.text() for label in pop.findChildren(type(s.top.name))]
    from PySide6.QtWidgets import QLabel

    texts = [label.text() for label in pop.findChildren(QLabel)]
    joined = " | ".join(texts)
    assert "Noise removal · 0:07" in joined and "Voice rebuild · 0:04" in joined and "Subtitles · 0:11" in joined and "Total" in joined
    assert "free and nothing is uploaded" in joined
    pop.hide()


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


def test_the_jobs_column_cards_say_where_each_file_stands(window, tmp_path):
    for name in ("a.wav", "b.wav"):
        (tmp_path / name).write_bytes(b"x" * 100)
    window.add_paths([tmp_path / "a.wav", tmp_path / "b.wav"])
    assert wait_until(lambda: all(i.doc.model is not None for i in window._items.values()), 5)
    settle(window)
    cards = window.jobs._cards
    assert [c.status.text() for c in cards.values()] == ["Not started", "Not started"] and all("0:11" in c.meta.text() for c in cards.values())
    a = window._items[1]
    a.state, a.progress = "running", 0.4
    window.jobs.refresh()
    window.jobs.update_progress(a)
    card = window.jobs._cards[("item", 1)]
    assert card.status.text() == "Cleaning 40%" and card.bar.isVisibleTo(card) and not card.meta.isVisibleTo(card)
    a.state = "cancelled"
    window.jobs.refresh()
    assert window.jobs._cards[("item", 1)].status.text() == "Cancelled"
    menu = window.jobs
    assert menu.act_start_all.isEnabled() and "Start all" in menu.act_start_all.text()


def test_the_screen_is_two_columns_wide_and_one_column_narrow(window):
    s = window.studio
    from PySide6.QtWidgets import QBoxLayout

    wide, narrow = QBoxLayout.Direction.LeftToRight, QBoxLayout.Direction.TopToBottom

    def directions():
        return window.body.direction(), s.ed.direction()

    sh.resized(window, 1440, 900, lambda: directions() == (wide, wide))
    assert directions() == (wide, wide)
    sh.resized(window, 780, 900, lambda: directions() == (narrow, narrow) and window.jobs.maximumWidth() > 300 and s.inspector.maximumWidth() > 356)
    assert directions() == (narrow, narrow)
    assert window.jobs.maximumWidth() > 300 and s.inspector.maximumWidth() > 356  # (under the editor: the whole width)
    sh.resized(window, 1440, 900, lambda: directions() == (wide, wide) and window.jobs.maximumWidth() == 300)
    assert directions() == (wide, wide) and window.jobs.maximumWidth() == 300
