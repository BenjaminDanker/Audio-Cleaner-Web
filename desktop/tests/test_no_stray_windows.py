"""The owner saw a small window flash now and then. The main flows (open a file, change settings, choose a part, run, open a finished job, open Details
and Your files) must never show a top-level window other than the main window, and, for Details and Your files, their popover. Every top-level widget
that is shown is caught by the Show event and by a poll every few milliseconds (studio_helpers.TopLevels); a window that shows is described with its
class, name, size, parents, flags and text, so the one that flashes can be found."""
from __future__ import annotations

import threading

from PySide6.QtCore import QPoint, Qt
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QApplication

import studio_helpers as sh
from conftest import wait_until
from studio_helpers import TopLevels, add_file, ready, settle


FOUND: list[str] = []


def no_strangers(watch: TopLevels, since: int, step: str, allow: tuple = ()) -> None:
    """Notes what showed in `step` besides the main window (and the classes in `allow`); the test says them all at its end, so one flow shows every
    window it makes, not only the first."""
    FOUND.extend(f"{step}: {d}" for d in watch.strangers(since, allow))


def choose_part_by_dragging(w, a: float, b: float) -> None:
    plot = w.studio.timeline.plot
    plot.resize(plot.width(), plot.total_height())
    y = 30 + 40  # (the waveform)
    QTest.mousePress(plot, Qt.MouseButton.LeftButton, Qt.KeyboardModifier.NoModifier, QPoint(int(plot.x_of(a)), y))
    QApplication.processEvents()
    from PySide6.QtCore import QEvent, QPointF
    from PySide6.QtGui import QMouseEvent

    for fraction in (0.25, 0.5, 0.75, 1.0):
        x = plot.x_of(a + (b - a) * fraction)
        QApplication.sendEvent(plot, QMouseEvent(QEvent.Type.MouseMove, QPointF(x, y), plot.mapToGlobal(QPointF(x, y)), Qt.MouseButton.NoButton,
                                                 Qt.MouseButton.LeftButton, Qt.KeyboardModifier.NoModifier))
        QApplication.processEvents()
    QTest.mouseRelease(plot, Qt.MouseButton.LeftButton, Qt.KeyboardModifier.NoModifier, QPoint(int(plot.x_of(b)), y))
    settle(w)


def change_settings(w) -> None:
    i = w.studio.inspector
    if w.studio.top.switch.isChecked():
        w.studio.top.switch.click()  # (Automatic off: by hand)
        settle(w)
    for key in ("rebuild", "noise", "off", "noise"):
        i.options[key].radio.click()
        QApplication.processEvents()
    for key in ("gentle", "strong", "max"):
        i.strength._buttons[key].click()
        QApplication.processEvents()
    for key in ("light", "heavy", "auto"):
        i.remover._buttons[key].click()
        QApplication.processEvents()
    for key in ("podcast", "youtube", "broadcast", "keep", "off", "podcast"):
        i.level._buttons[key].click()
        QApplication.processEvents()
    i.subs._buttons["on"].click()
    settle(w)
    for key in ("vtt", "txt", "srt"):
        i.files._buttons[key].click()
        QApplication.processEvents()
    i.language.setCurrentIndex(i.language.findData("de"))
    i.language.activated.emit(i.language.currentIndex())
    i.subs._buttons["off"].click()
    settle(w)
    # the name and the file type
    o = w.studio.output
    assert wait_until(lambda: o.name.isEnabled(), 5)
    o.name.setFocus()
    o.name.selectAll()
    QTest.keyClicks(o.name, "my talk.mp3")
    o.format.setFocus()  # (leaving the box moves the ending to the list)
    settle(w)
    index = o.format.findData(".wav")
    o.format.setCurrentIndex(index)
    o.format.activated.emit(index)
    settle(w)


def pick_a_part(w) -> None:
    choose_part_by_dragging(w, 2.0, 6.0)
    w.studio.timeline.sectionClicked.emit(1.0, 3.0)
    settle(w)
    w.studio.top.split_button.click()
    settle(w)
    w.studio.top.all_button.click()
    settle(w)


def test_the_main_flows_show_no_window_but_the_main_one_and_the_popover_of_details_and_your_files(window, tmp_path, runner):
    FOUND.clear()
    s = window.studio
    watch = TopLevels(window).start()
    try:
        # open a file
        mark = watch.mark()
        add_file(window, tmp_path)
        ready(window)
        no_strangers(watch, mark, "opening a file")
        # change settings
        mark = watch.mark()
        change_settings(window)
        no_strangers(watch, mark, "changing settings")
        # choose a part
        mark = watch.mark()
        pick_a_part(window)
        no_strangers(watch, mark, "choosing a part")
        # Details
        mark = watch.mark()
        s.inspector.level._buttons["podcast"].click()
        settle(window)
        s.start.details.click()
        assert s.start.popover.isVisible()
        QApplication.processEvents()
        s.start.popover.hide()
        QApplication.processEvents()
        no_strangers(watch, mark, "opening Details", allow=("Popover",))
        assert [r["cls"] for r in watch.seen[mark:] if r["widget"] is not window] == ["Popover"]  # (it did show, and it is the only one)
        # run
        mark = watch.mark()
        runner.gate = threading.Event()
        doc = window.doc
        s.start.button.click()
        assert wait_until(lambda: doc.stage == "processing" and doc.run_fraction == 0.5, 10)
        assert wait_until(lambda: s.start.mode == "progress", 5)
        runner.gate.set()
        assert wait_until(lambda: doc.stage == "done", 10)
        settle(window)
        no_strangers(watch, mark, "running a file")
        # Your files (after the run)
        mark = watch.mark()
        assert s.start.mode == "files"
        s.start.files.click()
        QApplication.processEvents()
        s.start.popover.hide()
        QApplication.processEvents()
        no_strangers(watch, mark, "opening Your files", allow=("Popover",))
        # open a finished job (one from the history) and its Your files
        mark = watch.mark()
        (tmp_path / "old").mkdir()
        folder = sh.make_folder(tmp_path / "old")
        window.doc_env.open_session = sh.open_session
        _, eid = sh.add_job(window.history, folder)
        window.open_entry(eid)
        assert wait_until(lambda: window.doc.session_state == "ready", 10)
        settle(window)
        choose_part_by_dragging(window, 1.0, 4.0)
        s.inspector.level._buttons["youtube"].click()
        settle(window)
        no_strangers(watch, mark, "opening a finished job")
        mark = watch.mark()
        s.start.files.click() if s.start.files.isVisibleTo(s.start) else s.start.details.click()
        QApplication.processEvents()
        s.start.popover.hide()
        QApplication.processEvents()
        no_strangers(watch, mark, "opening Details or Your files of a finished job", allow=("Popover",))
        # the empty screen again, and another file
        mark = watch.mark()
        window.open_item(next(iter(window._items)))
        settle(window)
        s.top.different.click()
        settle(window)
        add_file(window, tmp_path, "other.wav")
        ready(window)
        no_strangers(watch, mark, "going back and opening another file")
    finally:
        watch.stop()
    assert not FOUND, "a window other than the main one (and the popover of Details or Your files) was shown:\n  " + "\n  ".join(FOUND)
