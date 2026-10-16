"""Every option of the settings panel (and the window's size, position and tab) is still there after the app is
closed and opened again: the real window on an INI-backed QSettings, driven through its widgets."""
from __future__ import annotations

import pytest
from PySide6.QtCore import QSettings
from PySide6.QtWidgets import QFileDialog

from audio_cleaner_desktop import resources
from audio_cleaner_desktop.gui.main_window import MainWindow
from audio_cleaner_desktop.gui.theme import ThemeWatcher


def _open(tmp_path):
    settings = QSettings(str(tmp_path / "settings.ini"), QSettings.Format.IniFormat)
    w = MainWindow([], settings=settings, history_path=tmp_path / "history.db")
    w.show()
    return w


def _close(w):
    w.close()  # closeEvent saves the geometry and tab and shuts the queue down
    w._pool.waitForDone()
    w.deleteLater()


def _pick(box, key):
    box.setCurrentIndex(box.findData(key))
    assert box.currentData() == key


def _click(check, on):
    if check.isChecked() != on:
        check.click()
    assert check.isChecked() == on


def _shown(panel):
    """Everything the panel shows, read back from its widgets."""
    return {
        "strength": panel._strength.value(), "polish": panel._polish.isChecked(),
        "loudness": panel._loudness.currentData(), "subtitles": panel._subtitles.currentData(),
        "language": panel._language.currentData(), "noise": panel._noise.currentData(),
        "echo": panel._echo.isChecked(), "highs": panel._highs.isChecked(), "rebuild": panel._rebuild.isChecked(),
        "format": panel._format.currentData(), "video_size": panel._video_size.currentData(),
        "parallel": panel._parallel.currentData(),
        "where": panel._where.value(), "folder": panel._folder.text(),
    }


def _change_everything(panel, folder, monkeypatch):
    panel._strength._buttons["gentle"].click()  # while rebuild is still off: strength is enabled
    _pick(panel._noise, "heavy")
    _click(panel._polish, False)  # greys the loudness list out
    _pick(panel._loudness, "broadcast")  # still stored while greyed out
    _pick(panel._subtitles, "vtt")
    _pick(panel._language, "de")
    _click(panel._echo, True)
    _click(panel._highs, False)
    _pick(panel._format, "flac")
    _pick(panel._video_size, "720")
    _pick(panel._parallel, 3)
    monkeypatch.setattr(QFileDialog, "getExistingDirectory", lambda *a, **k: str(folder))
    panel._where._buttons["folder"].click()
    _click(panel._rebuild, True)  # greys out strength and noise


EXPECTED = {
    "strength": "gentle", "polish": False, "loudness": "broadcast", "subtitles": "vtt", "language": "de",
    "noise": "heavy", "echo": True, "highs": False, "rebuild": True, "format": "flac", "video_size": "720", "parallel": 3,
    "where": "folder",
}


@pytest.mark.parametrize("models_missing", [False, True], ids=["models-as-installed", "no-models"])
def test_every_setting_survives_closing_and_reopening(qapp, tmp_path, monkeypatch, models_missing):
    ThemeWatcher(qapp)
    if models_missing:  # the saved choice must stay even when the model it needs is not there
        monkeypatch.setattr(resources, "speech_ready", lambda: False)
        monkeypatch.setattr(resources, "highs_model_dir", lambda: None)
        monkeypatch.setattr(resources, "voice_model_dir", lambda: None)
    folder = tmp_path / "out folder"
    folder.mkdir()

    w = _open(tmp_path)
    panel = w.settings_panel
    # a fresh install shows the defaults
    assert _shown(panel)["strength"] == "max" and _shown(panel)["where"] == "next" and _shown(panel)["highs"]
    assert _shown(panel)["video_size"] == "same"  # the picture is left as it is unless asked
    _change_everything(panel, folder, monkeypatch)
    want = {**EXPECTED, "folder": str(folder)}
    assert _shown(panel) == want
    assert not panel._strength.isEnabled() and not panel._noise.isEnabled() and not panel._loudness.isEnabled()
    w.tabs.setCurrentIndex(1)
    w.resize(905, 641)  # the offscreen screen is only 800 wide: the width is clamped there, the height is not
    height = w.height()
    _close(w)

    again = _open(tmp_path)
    try:
        panel = again.settings_panel
        assert _shown(panel) == want
        # greyed-out controls keep their value and stay greyed out where they apply
        assert not panel._strength.isEnabled() and not panel._noise.isEnabled() and not panel._loudness.isEnabled()
        assert panel._language.isVisibleTo(panel) and panel._folder_box.isVisibleTo(panel)
        # the job the next Clean starts with is built from the same values
        o = again.prefs.job_options()
        assert (o.strength, o.polish, o.loudness, o.subtitles, o.language, o.noise) == (
            "gentle", False, "broadcast", "vtt", "de", "heavy")
        assert (o.echo_track, o.restore_highs, o.rebuild_voice, o.audio_format) == (True, False, True, "flac")
        assert (o.video_size, o.short_side) == ("720", 720)
        assert again.prefs.parallel == 3 and again.prefs.output_dir() == folder
        assert again.tabs.currentIndex() == 1
        assert again.height() == height and again.width() >= 900  # not the 720 of a first start
        # turning rebuild off again shows the stored strength and noise, enabled
        _click(panel._rebuild, False)
        assert panel._strength.isEnabled() and panel._strength.value() == "gentle" and panel._noise.currentData() == "heavy"
    finally:
        _close(again)


def test_other_values_survive_too(qapp, tmp_path, monkeypatch):
    """The other end of every control (and the remaining choices), plus the folder being remembered while
    "Next to original" is picked."""
    ThemeWatcher(qapp)
    folder = tmp_path / "out"
    folder.mkdir()
    w = _open(tmp_path)
    p = w.settings_panel
    p._strength._buttons["balanced"].click()
    _pick(p._noise, "light")
    _pick(p._loudness, "youtube")
    _pick(p._subtitles, "embed")
    _pick(p._language, "auto")
    _click(p._echo, False)
    _click(p._highs, True)
    _click(p._rebuild, False)
    _pick(p._format, "opus")
    _pick(p._video_size, "2160")  # the other end of the list
    _pick(p._parallel, 0)  # Automatic is a real choice, 0, and must not read back as unset
    monkeypatch.setattr(QFileDialog, "getExistingDirectory", lambda *a, **k: str(folder))
    p._where._buttons["folder"].click()
    p._where._buttons["next"].click()
    _close(w)

    again = _open(tmp_path)
    try:
        s = _shown(again.settings_panel)
        assert s["strength"] == "balanced" and s["noise"] == "light" and s["loudness"] == "youtube"
        assert s["subtitles"] == "embed" and s["language"] == "auto" and not s["echo"] and s["highs"] and not s["rebuild"]
        assert s["format"] == "opus" and s["parallel"] == 0 and s["where"] == "next" and s["video_size"] == "2160"
        assert again.prefs.output_folder == str(folder) and again.prefs.output_dir() is None
        again.settings_panel._where._buttons["folder"].click()  # the folder is offered again without asking
        assert again.prefs.output_dir() == folder
    finally:
        _close(again)


def test_a_saved_setting_that_no_longer_exists_falls_back(qapp, tmp_path):
    ThemeWatcher(qapp)
    ini = QSettings(str(tmp_path / "settings.ini"), QSettings.Format.IniFormat)
    ini.setValue("strength", "nonsense")
    ini.setValue("video_size", "900")
    ini.setValue("parallel", "99")
    ini.setValue("output_mode", "folder")  # but no folder saved
    ini.sync()
    w = _open(tmp_path)
    try:
        s = _shown(w.settings_panel)
        assert s["strength"] == "max" and s["parallel"] == 0 and s["where"] == "next" and s["video_size"] == "same"
    finally:
        _close(w)


def test_a_note_says_why_noise_settings_are_unavailable_while_rebuilding(qapp, tmp_path):
    from PySide6.QtCore import QSettings

    from audio_cleaner_desktop.gui.preferences import Preferences
    from audio_cleaner_desktop.gui.settings_panel import SettingsPanel

    prefs = Preferences(QSettings(str(tmp_path / "s.ini"), QSettings.Format.IniFormat))
    panel = SettingsPanel(prefs)
    try:
        panel._rebuild.setChecked(True)
        assert not panel._strength.isEnabled() and not panel._noise.isEnabled() and not panel._rebuild_note.isHidden()
        panel._rebuild.setChecked(False)
        assert panel._strength.isEnabled() and panel._noise.isEnabled() and panel._rebuild_note.isHidden()
    finally:
        panel.deleteLater()


def test_the_video_size_choices_and_their_help(qapp, tmp_path):
    ThemeWatcher(qapp)
    from audio_cleaner_desktop.gui.settings_panel import VIDEO_SIZE_LABELS, InfoButton

    w = _open(tmp_path)
    try:
        box = w.settings_panel._video_size
        assert [box.itemData(i) for i in range(box.count())] == ["same", "2160", "1440", "1080", "720", "480"]
        assert [box.itemText(i) for i in range(box.count())] == [
            "Same as the original", "2160p (4K)", "1440p", "1080p (Full HD)", "720p (HD)", "480p"]
        assert "only stretches the picture" in box.toolTip()
        helps = [b._text for b in w.settings_panel.findChildren(InfoButton) if "Same as the original: the picture" in b._text]
        assert helps and "Bigger than the original only stretches the picture" in helps[0] and "stays tall" in helps[0]
        assert set(VIDEO_SIZE_LABELS) == set(box.itemData(i) for i in range(box.count()))
    finally:
        _close(w)
