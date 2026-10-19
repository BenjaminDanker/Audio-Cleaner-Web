"""Every option of the settings panel (and the window's size, position and tab) is still there after the app is
closed and opened again: the real window on an INI-backed QSettings, driven through its widgets. Also the automatic
best settings: on by default, the hand-made choices kept for when it is off."""
from __future__ import annotations

from PySide6.QtCore import QSettings
from PySide6.QtWidgets import QFileDialog

from audio_cleaner_desktop import resources
from audio_cleaner_desktop.gui.main_window import MainWindow
from audio_cleaner_desktop.gui.preferences import Preferences
from audio_cleaner_desktop.gui.settings_panel import SettingsPanel
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
        "auto": panel._auto.isChecked(), "strength": panel._strength.value(), "leveling": panel._leveling.isChecked(),
        "loudness": panel._loudness.currentData(), "subtitles": panel._subtitles.currentData(),
        "language": panel._language.currentData(), "noise": panel._noise.currentData(),
        "echo": panel._echo.isChecked(), "highs": panel._highs.isChecked(), "rebuild": panel._rebuild.isChecked(),
        "format": panel._format.currentData(), "video_size": panel._video_size.currentData(),
        "parallel": panel._parallel.currentData(),
        "where": panel._where.value(), "folder": panel._folder.text(),
    }


def _change_everything(panel, folder, monkeypatch):
    _click(panel._auto, False)  # choosing by hand: strength, noise, highs and rebuild are enabled
    panel._strength._buttons["gentle"].click()  # while rebuild is still off: strength is enabled
    _pick(panel._noise, "heavy")
    _click(panel._leveling, False)  # greys the loudness list out
    _pick(panel._loudness, "broadcast")  # still stored while greyed out
    _pick(panel._subtitles, "vtt")
    _pick(panel._language, "de")
    _click(panel._echo, True)
    _click(panel._highs, False)
    _pick(panel._format, "flac")
    _pick(panel._video_size, "720")
    _pick(panel._parallel, 0)  # Automatic is a real choice, 0, and must not read back as unset
    monkeypatch.setattr(QFileDialog, "getExistingDirectory", lambda *a, **k: str(folder))
    panel._where._buttons["folder"].click()
    _click(panel._rebuild, True)  # greys out strength and noise


EXPECTED = {
    "auto": False, "strength": "gentle", "leveling": False, "loudness": "broadcast", "subtitles": "vtt", "language": "de",
    "noise": "heavy", "echo": True, "highs": False, "rebuild": True, "format": "flac", "video_size": "720", "parallel": 0,
    "where": "folder",
}


def test_every_setting_survives_closing_and_reopening(qapp, tmp_path, monkeypatch):
    """The saved choices stay even when the models they need are not installed (they are only greyed out)."""
    ThemeWatcher(qapp)
    monkeypatch.setattr(resources, "speech_ready", lambda: False)
    monkeypatch.setattr(resources, "highs_model_dir", lambda: None)
    monkeypatch.setattr(resources, "voice_model_dir", lambda: None)
    folder = tmp_path / "out folder"
    folder.mkdir()

    w = _open(tmp_path)
    panel = w.settings_panel
    # a fresh install shows the defaults
    assert _shown(panel)["strength"] == "max" and _shown(panel)["where"] == "next" and _shown(panel)["highs"]
    assert _shown(panel)["auto"]  # automatic best settings, until unticked
    assert _shown(panel)["video_size"] == "same"  # the picture is left as it is unless asked
    _change_everything(panel, folder, monkeypatch)
    want = {**EXPECTED, "folder": str(folder)}
    assert _shown(panel) == want
    w.tabs.setCurrentIndex(1)
    w.resize(905, 641)  # the offscreen screen is only 800 wide: the width is clamped there, the height is not
    height = w.height()
    _close(w)

    again = _open(tmp_path)
    try:
        panel = again.settings_panel
        assert _shown(panel) == want
        # the job the next Clean starts with is built from the same values
        o = again.prefs.job_options()
        assert (o.strength, o.leveling, o.loudness, o.subtitles, o.language, o.noise) == (
            "gentle", False, "broadcast", "vtt", "de", "heavy")
        assert (o.echo_track, o.restore_highs, o.rebuild_voice, o.audio_format) == (True, False, True, "flac")
        assert (o.video_size, o.short_side) == ("720", 720)
        assert again.prefs.parallel == 0 and again.prefs.output_dir() == folder
        assert again.tabs.currentIndex() == 1
        assert again.height() == height and again.width() >= 900  # not the 720 of a first start
        # the folder is offered again without asking when "Next to original" is picked and then "folder" again
        again.settings_panel._where._buttons["next"].click()
        again.settings_panel._where._buttons["folder"].click()
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


def test_automatic_best_settings_choose_for_each_file_while_on_and_keep_the_hand_made_choices(qapp, tmp_path, monkeypatch):
    # neither the highs model nor the voice model is installed here
    monkeypatch.setattr(resources, "highs_model_dir", lambda: None)
    monkeypatch.setattr(resources, "voice_model_dir", lambda: None)
    ini = tmp_path / "s.ini"
    prefs = Preferences(QSettings(str(ini), QSettings.Format.IniFormat))
    panel = SettingsPanel(prefs)
    try:
        assert panel._auto.isChecked() and prefs.auto_settings  # on for a fresh install
        o = prefs.job_options()
        assert (o.strength, o.noise, o.atten_lim_db, o.restore_highs, o.rebuild_voice, o.auto_rebuild) == (
            "auto", "auto", None, True, False, True)
        assert o.auto_highs  # the highs are the model's call while everything is automatic
        _click(panel._auto, False)  # by hand
        panel._strength._buttons["strong"].click()
        _pick(panel._noise, "light")
        _click(panel._highs, False)
        o = prefs.job_options()
        assert (o.strength, o.noise, o.restore_highs, o.rebuild_voice, o.auto_rebuild) == ("strong", "light", False, False, False)
        assert not o.auto_highs and not o.decides  # by hand: nothing is measured, nothing is picked for the file
        _click(panel._rebuild, True)  # rebuilt always, the noise settings don't apply
        assert prefs.job_options().rebuild_voice and not prefs.job_options().auto_rebuild
        _click(panel._auto, True)  # ticked rebuild doesn't count while automatic: only some files are rebuilt
        o = prefs.job_options()
        assert (o.strength, o.noise, o.restore_highs, o.rebuild_voice, o.auto_rebuild) == ("auto", "auto", True, False, True)
        # the hand-made choices are kept for when it is off
        assert panel._strength.value() == "strong" and panel._noise.currentData() == "light"
        _click(panel._auto, False)
        _click(panel._rebuild, False)
    finally:
        panel.deleteLater()
    # off is remembered, and the hand-made choices come back
    prefs = Preferences(QSettings(str(ini), QSettings.Format.IniFormat))
    panel = SettingsPanel(prefs)
    try:
        assert not panel._auto.isChecked() and prefs.auto_settings is False
        assert panel._strength.value() == "strong"
    finally:
        panel.deleteLater()
