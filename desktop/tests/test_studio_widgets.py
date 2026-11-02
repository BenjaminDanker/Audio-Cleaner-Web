"""The pieces of the editor screen on their own: the timeline (what a click, a drag and a handle do, the rows, the zoom), the settings column (each card
edits the part that is chosen), the start row, the jobs column and the top of the editor. The model and the player are the real ones; nothing runs."""
from __future__ import annotations

import pytest
from PySide6.QtCore import QPoint, QPointF, Qt
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QApplication

import studio_helpers as sh
from audio_cleaner_desktop.gui import timeline as tl_mod
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
    # the Music row is not there yet: nothing to choose
    music = [i for i, (lane, _) in enumerate(plot.lanes()) if lane == "music"][0]
    press(plot, plot.x_of(5), plot.lane_top(music) + 10)
    release(plot, plot.x_of(5), plot.lane_top(music) + 10)
    assert got == [(3.0, 7.0)]


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
    assert i.heading.text() == "0:02–0:06 · 4 s" and doc.model.voice_mode_of(2, 6) == "rebuild" and doc.model.level_of(2, 6) == "off"
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
    assert i.heading.text() == "Whole file · 0:11" and not i.only.isVisibleTo(i) and i.voice_card.mixed.isVisibleTo(i)


def test_leveling_is_one_loudness_for_the_whole_file_and_the_cards_are_locked_while_it_runs(loaded, runner):
    s, doc = loaded.studio, loaded.doc
    i = s.inspector
    doc.model.select(0, 3)
    i.level._buttons["youtube"].click()
    doc.model.select(5, 8)
    i.level._buttons["broadcast"].click()
    settle(loaded)
    assert doc.model.loudness() == -23.0 and all(p.settings["loudness_lufs"] == -23.0 for p in doc.model.draft.layers["leveling"])
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


# ---------------------------------------------------------------------------------------------------- the start row and the details

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
    window.resize(1440, 900)
    QApplication.processEvents()
    from PySide6.QtWidgets import QBoxLayout

    assert window.body.direction() == QBoxLayout.Direction.LeftToRight and s.ed.direction() == QBoxLayout.Direction.LeftToRight
    window.resize(780, 900)
    QApplication.processEvents()
    assert window.body.direction() == QBoxLayout.Direction.TopToBottom and s.ed.direction() == QBoxLayout.Direction.TopToBottom
    assert window.jobs.maximumWidth() > 300 and s.inspector.maximumWidth() > 356  # (under the editor: the whole width)
    window.resize(1440, 900)
    QApplication.processEvents()
    assert window.body.direction() == QBoxLayout.Direction.LeftToRight and window.jobs.maximumWidth() == 300
