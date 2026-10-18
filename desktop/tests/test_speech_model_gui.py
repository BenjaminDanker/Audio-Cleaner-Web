"""The speech model download in the window: offered when subtitles need it, runs with progress, blocks
starting a subtitles job until it is there."""
from __future__ import annotations

import functools
import hashlib
import http.server
import threading

import pytest
from PySide6.QtCore import QSettings
from PySide6.QtWidgets import QMessageBox

from audio_cleaner_desktop import models
from audio_cleaner_desktop.gui import speech_model
from audio_cleaner_desktop.gui.main_window import MainWindow
from audio_cleaner_desktop.gui.preferences import Preferences
from audio_cleaner_desktop.gui.settings_panel import SettingsPanel

from conftest import wait_until


@pytest.fixture
def small_model(tmp_path, monkeypatch):
    """A 2 MB stand-in for the speech model, served over local http."""
    files = tmp_path / "served"
    files.mkdir()
    data = bytes(range(256)) * 8192
    (files / "model.bin").write_bytes(data)
    handler = functools.partial(http.server.SimpleHTTPRequestHandler, directory=str(files))
    handler.log_message = lambda *a: None
    httpd = http.server.ThreadingHTTPServer(("127.0.0.1", 0), handler)
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    monkeypatch.setenv("ACW_DESKTOP_MODELS", str(tmp_path / "models"))
    monkeypatch.setattr(models, "_ssl_context", lambda: None)
    url = f"http://127.0.0.1:{httpd.server_address[1]}/model.bin"
    yield models.SpeechModel("test-speech", "Test", "", "MIT",
                             (models.ModelFile("model.bin", url, len(data), hashlib.sha256(data).hexdigest()),))
    httpd.shutdown()


def test_panel_offers_the_download_runs_it_and_goes_away(qapp, tmp_path, small_model):
    from audio_cleaner_desktop.gui.preferences import Preferences

    prefs = Preferences(QSettings(str(tmp_path / "s.ini"), QSettings.Format.IniFormat))
    dl = speech_model.SpeechModelDownloader(model=small_model)
    panel = SettingsPanel(prefs, dl)
    panel.show()
    assert not panel.speech_panel.isVisible()  # subtitles off: nothing to ask
    panel._subtitles.setCurrentIndex(panel._subtitles.findData("srt"))
    assert panel.speech_panel.isVisible() and "one-time download of 2 MB" in panel.speech_panel._text.text()
    seen = []
    dl.progress.connect(lambda d, t: seen.append((d, t)))
    panel.speech_panel._button.click()
    full = (small_model.size, small_model.size)
    # progress arrives as queued events: wait for the last one, not just for the download to end
    assert wait_until(lambda: not dl.running and dl.ready and seen and seen[-1] == full, timeout=30)
    assert wait_until(lambda: not panel.speech_panel.isVisible(), timeout=5)


def test_a_missing_model_is_fetched_one_after_the_other_never_at_start_up(qapp, tmp_path, monkeypatch):
    monkeypatch.setattr(speech_model.SpeechModelDownloader, "ready", property(lambda self: self._fake_ready))
    monkeypatch.setattr(speech_model.SpeechModelDownloader, "_fake_ready", False, raising=False)
    started = []

    def fake_start(self):
        started.append(self.model.id)

    monkeypatch.setattr(speech_model.SpeechModelDownloader, "start", fake_start)
    shown = []
    monkeypatch.setattr(QMessageBox, "exec", lambda self: shown.append(self.text()) or QMessageBox.StandardButton.Ok)
    w = MainWindow([], settings=QSettings(str(tmp_path / "s.ini"), QSettings.Format.IniFormat), history_path=tmp_path / "h.db")
    try:
        w.show()
        qapp.processEvents()
        assert started == [] and not w.banner_models.isVisible()  # setup downloads them, not the app's start
        w.fetch_models()  # e.g. after setup's download failed
        assert started == ["whisper-large-v3-turbo"] and w.banner_models.isVisible()
        assert "Getting everything ready" in w.banner_models._label.text() and not shown  # nothing asked
        w.speech_download._fake_ready = True
        w._model_download_finished(w.speech_download, True, "")
        assert started == ["whisper-large-v3-turbo", "restore-highs"]
        w.highs_download._fake_ready = True
        w._model_download_finished(w.highs_download, True, "")
        assert started == ["whisper-large-v3-turbo", "restore-highs", "rebuild-voice"]
        w.voice_download._fake_ready = True
        w._model_download_finished(w.voice_download, True, "")
        assert not w.banner_models.isVisible()
        # a failed download says so and offers to try again
        w._model_queue = [w.highs_download]
        w._model_download_finished(w.highs_download, False, "The download stopped (timed out).")
        assert w.banner_models.isVisible() and "could not be downloaded" in w.banner_models._label.text()
        # starting a subtitles job without the speech model does not start, and offers the download
        w.speech_download._fake_ready = False
        w.prefs.subtitles = "srt"
        src = tmp_path / "a.wav"
        src.write_bytes(b"RIFF")
        w.add_paths([src])
        w.start()
        assert shown and "was not installed" in shown[-1] and not w.queue.active
    finally:
        w.queue.shutdown()
        w.history.close()
        w._pool.waitForDone()


def test_rebuilding_the_voice_offers_its_download_and_start_asks_instead_of_downgrading(qapp, tmp_path, monkeypatch):
    from audio_cleaner_desktop import engine
    from types import SimpleNamespace

    monkeypatch.setattr(speech_model.SpeechModelDownloader, "ready", property(lambda self: self.model.id != "rebuild-voice"))
    started = []
    monkeypatch.setattr(speech_model.SpeechModelDownloader, "start", lambda self: started.append(self.model.id))
    shown = []
    monkeypatch.setattr(QMessageBox, "exec", lambda self: shown.append(self.text()) or QMessageBox.StandardButton.Yes)
    monkeypatch.setattr(engine, "probe", lambda path, ff=None: SimpleNamespace(duration=5.0, has_video=False, audio_channels=1))
    settings = QSettings(str(tmp_path / "s.ini"), QSettings.Format.IniFormat)
    Preferences(settings).auto_settings = False  # rebuilding chosen by hand
    w = MainWindow([], settings=settings, history_path=tmp_path / "h.db")
    try:
        w.show()
        panel = w.settings_panel
        assert not panel.voice_panel.isVisible()  # not ticked: nothing to ask
        panel._rebuild.setChecked(True)
        assert panel.voice_panel.isVisible() and "Rebuilding the voice needs an extra model" in panel.voice_panel._text.text()
        panel.voice_panel._button.click()
        assert started == ["rebuild-voice"]
        started.clear()
        src = tmp_path / "a.wav"
        src.write_bytes(b"RIFF")
        w.add_paths([src])
        w.start()  # asks, offers the download, starts nothing
        assert shown and "model that rebuilds the voice was not installed" in shown[-1] and not w.queue.active
        assert started == ["rebuild-voice"]
        assert "rebuilding voices" in w.banner_models._label.text()
    finally:
        w.queue.shutdown()
        w.history.close()
        w._pool.waitForDone()


def test_automatic_rebuilding_offers_the_download_but_never_holds_up_cleaning(qapp, tmp_path, monkeypatch):
    """With Automatic best settings the voice is rebuilt only where its model is installed, so a missing model is offered
    in Settings but Clean doesn't stop to ask for it."""
    from types import SimpleNamespace

    from audio_cleaner_desktop import engine
    from audio_cleaner_desktop.gui import main_window

    monkeypatch.setattr(speech_model.SpeechModelDownloader, "ready", property(lambda self: self.model.id != "rebuild-voice"))
    shown = []
    monkeypatch.setattr(QMessageBox, "exec", lambda self: shown.append(self.text()) or QMessageBox.StandardButton.Yes)
    monkeypatch.setattr(engine, "probe", lambda path, ff=None: SimpleNamespace(duration=5.0, has_video=False, audio_channels=1))
    monkeypatch.setattr(main_window.resources, "check", lambda: [])
    ran = []

    def runner(path, options, **_):
        ran.append((options.auto_rebuild, options.rebuild_voice))
        out = path.with_name("a-cleaned.wav")
        out.write_bytes(b"x")
        return engine.JobResult(out, 5.0, False)

    w = MainWindow([], settings=QSettings(str(tmp_path / "s.ini"), QSettings.Format.IniFormat), history_path=tmp_path / "h.db",
                   runner=runner)
    try:
        w.show()
        panel = w.settings_panel
        assert w.prefs.auto_settings and panel.voice_panel.isVisible()  # offered
        src = tmp_path / "a.wav"
        src.write_bytes(b"RIFF")
        w.add_paths([src])
        assert wait_until(lambda: all(i.probed for i in w._items.values()), timeout=10)
        w.start()
        assert wait_until(lambda: ran, timeout=10) and ran == [(True, False)] and not shown
    finally:
        w.queue.shutdown()
        w.history.close()
        w._pool.waitForDone()
