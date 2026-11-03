"""What was chosen is still there after the app is closed and opened again: the choices made on the editor screen for a whole file (they are what the
next file starts with), the ones in Settings, and the window's size: the real window on an INI-backed QSettings, driven through its widgets. Also
Automatic: on by default, the hand-made choices kept for when it is off."""
from __future__ import annotations


from PySide6.QtCore import QSettings
from PySide6.QtWidgets import QFileDialog

from audio_cleaner_desktop import labels, resources
from audio_cleaner_desktop.gui import copy
from audio_cleaner_desktop.gui.main_window import MainWindow
from audio_cleaner_desktop.gui.settings_dialog import HelpButton
from audio_cleaner_desktop.gui.theme import ThemeWatcher

import studio_helpers as sh
from conftest import wait_until


def _open(tmp_path, seconds=11.0):
    settings = QSettings(str(tmp_path / "settings.ini"), QSettings.Format.IniFormat)
    w = MainWindow([], settings=settings, history_path=tmp_path / "history.db")
    w.fakes = sh.Fakes(seconds).install(w)
    w.show()
    return w


def _close(w):
    w.close()  # closeEvent saves the geometry and shuts the queue down
    w._pool.waitForDone()
    w.deleteLater()


def _file(w, tmp_path, name="talk.wav"):
    (tmp_path / name).write_bytes(b"x" * 100)
    w.add_paths([tmp_path / name])
    assert wait_until(lambda: w.doc.model is not None, 10)
    sh.settle(w)
    return w.doc


def _pick(box, key):
    box.setCurrentIndex(box.findData(key))
    assert box.currentData() == key


def _shown(w):
    """Everything the screen and Settings show for the file in the editor, read back from the widgets."""
    i, d = w.studio.inspector, w.settings_dialog_for_tests
    m = w.doc.model
    return {
        "auto": w.studio.top.switch.isChecked(), "strength": m.strength_of(0, m.seconds), "remover": m.remover_of(0, m.seconds),
        "voice": m.voice_mode_of(0, m.seconds), "leveling": i.level.value(), "subtitles": i.subs.value(), "kind": i.files.value(),
        "language": i.language.currentData(), "echo": i.echo.isChecked(), "size": w.studio.output.size.currentData(),
        "format": d._format.currentData(), "parallel": d._parallel.currentData(), "where": d._where.value(), "folder": d._folder.text(),
    }


def _change_everything(w, folder, monkeypatch):
    i = w.studio.inspector
    w.studio.top.switch.click()  # choosing by hand
    sh.settle(w)
    i.strength._buttons["gentle"].click()
    i.remover._buttons["heavy"].click()
    i.level._buttons["broadcast"].click()
    i.subs._buttons["on"].click()
    sh.settle(w)
    i.files._buttons["vtt"].click()
    _pick(i.language, "de")
    i.language.activated.emit(i.language.currentIndex())
    i.echo.click()
    sh.settle(w)
    dialog = w.settings_dialog_for_tests
    _pick(dialog._format, "flac")
    _pick(dialog._parallel, 0)  # Automatic is a real choice, 0, and must not read back as unset
    monkeypatch.setattr(QFileDialog, "getExistingDirectory", lambda *a, **k: str(folder))
    dialog._where._buttons["folder"].click()
    sh.settle(w)


def _with_dialog(w):
    w.show_settings()
    w.settings_dialog_for_tests = w._settings_dialog


def test_every_choice_survives_closing_and_reopening(qapp, tmp_path, monkeypatch):
    ThemeWatcher(qapp)
    monkeypatch.setattr(resources, "speech_ready", lambda: False)
    monkeypatch.setattr(resources, "voice_model_dir", lambda: None)
    folder = tmp_path / "out folder"
    folder.mkdir()
    w = _open(tmp_path)
    w.fakes.facts.update(tracks=2)  # (a second audio track: the echo card is there)
    _file(w, tmp_path)
    _with_dialog(w)
    # a fresh install shows the defaults
    first = _shown(w)
    assert first["auto"] and first["where"] == "next" and first["size"] in ("same", None) and first["kind"] == ""
    _change_everything(w, folder, monkeypatch)
    want = {"auto": False, "strength": "gentle", "remover": "heavy", "voice": "noise", "leveling": "broadcast", "subtitles": "on", "kind": "vtt",
            "language": "de", "echo": True, "size": "same", "format": "flac", "parallel": 0, "where": "folder", "folder": str(folder)}
    assert _shown(w) == want
    sh.resized(w, 905, 641)
    height = w.height()
    assert height == 641
    _close(w)

    again = _open(tmp_path)
    try:
        again.fakes.facts.update(tracks=2)
        _file(again, tmp_path, "next.wav")
        _with_dialog(again)
        shown = _shown(again)
        # the next file starts with what was chosen (Automatic is off, so nothing is decided for it)
        assert shown == want, {k: (shown[k], want[k]) for k in want if shown[k] != want[k]}
        o = again.prefs.job_options()
        assert (o.strength, o.leveling, o.loudness, o.subtitles, o.language, o.noise) == ("gentle", True, "broadcast", "vtt", "de", "heavy")
        assert (o.echo_track, o.rebuild_voice, o.audio_format) == (True, False, "flac")
        assert again.prefs.parallel == 0 and again.prefs.output_dir() == folder
        wait_until(lambda: again.height() == height, 5)
        assert again.height() == height and again.width() >= 640  # (not the 900 of a first start: the offscreen screen clamps the width, not the height)
        # the folder is offered again without asking when "Next to original" is picked and then "folder" again
        again._settings_dialog._where._buttons["next"].click()
        again._settings_dialog._where._buttons["folder"].click()
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
        _file(w, tmp_path)
        _with_dialog(w)
        s = _shown(w)
        assert w.doc.model.default_noise["strength_db"] == 100.0 and s["parallel"] == 0 and s["where"] == "next" and w.doc.video_size == "same"
    finally:
        _close(w)


def test_automatic_chooses_for_each_file_while_on_and_keeps_the_hand_made_choices(qapp, tmp_path, monkeypatch):
    monkeypatch.setattr(resources, "voice_model_dir", lambda: None)  # the voice model is not installed here
    w = _open(tmp_path)
    ini = tmp_path / "settings.ini"
    try:
        doc = _file(w, tmp_path)
        assert doc.auto and w.prefs.auto_settings  # on for a fresh install
        assert wait_until(lambda: doc.applied, 10)
        options, timeline = doc.run_plan()
        assert (options.strength, options.noise, options.atten_lim_db, options.rebuild_voice, options.auto_rebuild) == ("auto", "auto", None, False, True)
        assert timeline is None and options.decides  # untouched: the run measures the file and decides for itself
        assert doc.model.strength_of(0, 11) == "balanced" and doc.model.remover_of(0, 11) == "light"  # (what it decided is shown)
        # by hand
        w.prefs.strength, w.prefs.noise = "strong", "light"
        w.studio.top.switch.click()
        sh.settle(w)
        assert not doc.auto and not w.prefs.auto_settings and doc.model.strength_of(0, 11) == "strong"
        options, timeline = doc.run_plan()
        assert (options.rebuild_voice, options.auto_rebuild, options.decides) == (False, False, False) and timeline["layers"]["noise"][0]["strength_db"] == 40.0
        w.studio.inspector.options["rebuild"].radio.click()  # rebuilt: the noise settings don't apply
        sh.settle(w)
        assert doc.model.voice_mode_of(0, 11) == "rebuild" and w.prefs.rebuild_voice and w.prefs.strength == "strong"
        # Automatic again: its choices replace them (rebuilding only where the model is there), the hand-made ones are kept for when it is off
        w.studio.top.switch.click()
        assert wait_until(lambda: doc.applied and not doc.listening, 10)
        sh.settle(w)
        assert doc.auto and doc.model.strength_of(0, 11) == "balanced" and doc.model.voice_mode_of(0, 11) == "noise"
        assert w.prefs.strength == "strong" and w.prefs.noise == "light" and w.prefs.rebuild_voice  # (kept: Automatic's picks are not remembered)
        assert w.studio.top.auto_action.text() == "Undo"  # "Undo" gives the choices before Automatic back
        w.studio.top.auto_action.click()
        sh.settle(w)
        assert not doc.auto and doc.model.voice_mode_of(0, 11) == "rebuild"
    finally:
        _close(w)
    # off is remembered, and the hand-made choices come back
    again = _open(tmp_path)
    try:
        doc = _file(again, tmp_path, "next.wav")
        assert not doc.auto and again.prefs.auto_settings is False and doc.model.voice_mode_of(0, 11) == "rebuild"
        again.studio.inspector.options["noise"].radio.click()
        assert again.doc.model.strength_of(0, 11) == "strong" and again.doc.model.remover_of(0, 11) == "light"
        assert ini.exists()
    finally:
        _close(again)


def test_the_screen_says_what_the_voice_is_rebuilt_for(qapp, tmp_path):
    """The Voice rebuild line, its tooltip and the Automatic help name the same causes the automatic choice rebuilds for (echo, a muffled voice,
    phone-quality or damaged sound, distortion from recording too loud)."""
    tip = " ".join(copy.TIPS["v.rebuild"].split())
    assert tip.startswith("For an echoey room, a muffled voice, phone-quality or damaged sound, or distortion from recording too loud:")
    assert "Keeps only speech" in tip and "highs" not in tip.lower() and "dull sound" not in tip
    line = dict((k, v) for k, _, v in copy.VOICE_MODES)["rebuild"]
    assert "echo" in line and "muffled" in line and "keeps only speech" in line
    assert any("echo, a muffled or phone-quality voice and distortion from recording too loud" in p for p in copy.REBUILD_POINTS)
    helps = [" ".join(b._text.split()) for b in _help_buttons(tmp_path)]
    extra = next(t for t in helps if t.startswith("Cleaned files get"))
    assert "highs" not in extra.lower()


def _help_buttons(tmp_path):
    from audio_cleaner_desktop.gui.preferences import Preferences
    from audio_cleaner_desktop.gui.settings_dialog import SettingsDialog
    from audio_cleaner_desktop.gui.speech_model import SpeechModelDownloader
    from audio_cleaner_desktop.updater import UpdateController

    prefs = Preferences(QSettings(str(tmp_path / "p.ini"), QSettings.Format.IniFormat))
    dialog = SettingsDialog(prefs, SpeechModelDownloader(), SpeechModelDownloader(model=resources.VOICE_MODEL), UpdateController(prefs))
    return dialog.findChildren(HelpButton)


def test_which_one_to_pick_is_said_in_plain_words(qapp, tmp_path):
    text = labels.WHICH_TEXT
    for phrase in ("keeps your voice exactly as recorded", "never changes a word", "laughs, shouts, whispers and singing stay as they were",
                   "any length", "game sounds", "fixes echo, a muffled or phone-quality voice and distortion", "can sound slightly too clean",
                   "keeps only speech", "may come out oddly", "can change a word, so listen before you publish", "slower", "many languages"):
        assert phrase.lower() in text.lower(), phrase
    assert "polish" not in text.lower() and "model" not in text.lower()
    assert "the voice was rebuilt because it sounded echoey" in labels.rebuild_reason_text("echoey") and "keeps only speech" in labels.rebuild_reason_text("echoey")
    # the screen says it in the card (a link to All the options) and in that window
    from audio_cleaner_desktop.gui.info_dialog import InfoDialog
    from PySide6.QtWidgets import QLabel

    dialog = InfoDialog()
    shown = " ".join(label.text() for label in dialog.findChildren(QLabel))
    for phrase in ("never changes words", "Laughs, shouts, whispers and singing stay as they were", "can change a word, so listen before you publish"):
        assert phrase in shown
    dialog.close()
