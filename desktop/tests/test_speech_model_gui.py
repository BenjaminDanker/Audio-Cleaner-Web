"""The model downloads in the window: offered in the card that needs the model (Subtitles, Voice rebuild), run with progress, and they keep Start
waiting until the model is there (with a note that says so); the noise removal models, which the window downloads by itself."""
from __future__ import annotations

import functools
import hashlib
import http.server
import threading

import pytest
from PySide6.QtCore import QSettings
from PySide6.QtWidgets import QMessageBox

from audio_cleaner_desktop import cli, engine, models, resources
from audio_cleaner_desktop.gui import speech_model
from audio_cleaner_desktop.gui.main_window import MainWindow
from audio_cleaner_desktop.gui.preferences import Preferences

import studio_helpers as sh
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


def open_window(tmp_path, runner=None, seconds=5.0, **kw):
    w = MainWindow([], settings=QSettings(str(tmp_path / "s.ini"), QSettings.Format.IniFormat), history_path=tmp_path / "h.db",
                   **({"runner": runner} if runner else {}))
    w.fakes = sh.Fakes(seconds).install(w)
    w.show()
    return w


def add_and_wait(w, tmp_path, name="a.wav"):
    src = tmp_path / name
    src.write_bytes(b"RIFF")
    w.add_paths([src])
    assert wait_until(lambda: w.doc.model is not None, 10)
    sh.settle(w)
    return src


def test_the_subtitles_card_offers_the_download_runs_it_and_goes_away(window, tmp_path, small_model, monkeypatch):
    w = window
    monkeypatch.setattr(speech_model.SpeechModelDownloader, "ready", property(lambda self: self.model is resources.NOISE_MODEL or models.is_installed(self.model)))
    monkeypatch.setitem(models.CATALOG, small_model.id, small_model)  # (the window tidies up what the catalog does not know)
    dl = w.speech_download
    dl.model = small_model
    add_and_wait(w, tmp_path)
    panel = w.studio.inspector.speech_panel
    assert not panel.isVisibleTo(w.studio.inspector)  # subtitles off: nothing to ask
    w.studio.inspector.subs._buttons["on"].click()
    sh.settle(w)
    assert panel.isVisibleTo(w.studio.inspector) and "one-time download of 2 MB" in panel._text.text()
    assert not w.studio.start.button.isEnabled() and any("speech model" in t for t, _ in w.studio.note_lines(w.doc))  # Start waits for it
    seen = []
    dl.progress.connect(lambda d, t: seen.append((d, t)))
    panel._button.click()
    full = (small_model.size, small_model.size)
    # progress arrives as queued events: wait for the last one, not just for the download to end
    assert wait_until(lambda: not dl.running and dl.ready and seen and seen[-1] == full, timeout=30)
    sh.settle(w)
    assert wait_until(lambda: not panel.isVisibleTo(w.studio.inspector), timeout=5) and w.studio.start.button.isEnabled()


def test_a_missing_model_is_fetched_one_after_the_other_never_at_start_up(qapp, tmp_path, monkeypatch):
    monkeypatch.setattr(speech_model.SpeechModelDownloader, "ready", property(lambda self: self.model is resources.NOISE_MODEL or self._fake_ready))
    monkeypatch.setattr(speech_model.SpeechModelDownloader, "_fake_ready", False, raising=False)
    started = []

    def fake_start(self):
        started.append(self.model.id)

    monkeypatch.setattr(speech_model.SpeechModelDownloader, "start", fake_start)
    shown = []
    monkeypatch.setattr(QMessageBox, "exec", lambda self: shown.append(self.text()) or QMessageBox.StandardButton.Ok)
    w = open_window(tmp_path)
    try:
        qapp.processEvents()
        assert started == [] and not w.banner_models.isVisible()  # setup downloads them, not the app's start
        w.fetch_models()  # e.g. after setup's download failed
        assert started == ["whisper-large-v3-turbo"] and w.banner_models.isVisible()
        assert "Getting everything ready" in w.banner_models._label.text() and not shown  # nothing asked
        w.speech_download._fake_ready = True
        w._model_download_finished(w.speech_download, True, "")
        assert started == ["whisper-large-v3-turbo", "rebuild-voice"]
        w.voice_download._fake_ready = True
        w._model_download_finished(w.voice_download, True, "")
        assert not w.banner_models.isVisible()
        # a failed download says so and offers to try again
        w._model_queue = [w.voice_download]
        w._model_download_finished(w.voice_download, False, "The download stopped (timed out).")
        assert w.banner_models.isVisible() and "could not be downloaded" in w.banner_models._label.text()
        # a subtitles job without the speech model does not start, and the screen says why (no dialog: the card has the download)
        w.speech_download._fake_ready = False
        w.prefs.subtitles = "srt"
        add_and_wait(w, tmp_path)
        w.start()
        assert not shown and not w.queue.active and not w.queue.busy
        assert any("speech model" in t for t, _ in w.studio.note_lines(w.doc)) and w.studio.inspector.speech_panel.isVisibleTo(w.studio.inspector)
    finally:
        sh.shut(w)


def test_voice_rebuild_offers_its_download_and_start_waits_instead_of_downgrading(qapp, tmp_path, monkeypatch):
    monkeypatch.setattr(speech_model.SpeechModelDownloader, "ready", property(lambda self: self.model.id != "rebuild-voice"))
    started = []
    monkeypatch.setattr(speech_model.SpeechModelDownloader, "start", lambda self: started.append(self.model.id))
    shown = []
    monkeypatch.setattr(QMessageBox, "exec", lambda self: shown.append(self.text()) or QMessageBox.StandardButton.Yes)
    settings = QSettings(str(tmp_path / "s.ini"), QSettings.Format.IniFormat)
    Preferences(settings).auto_settings = False  # rebuilding chosen by hand
    monkeypatch.setattr("audio_cleaner_desktop.resources.voice_model_dir", lambda: None)
    w = MainWindow([], settings=settings, history_path=tmp_path / "h.db")
    w.fakes = sh.Fakes(5.0).install(w)
    try:
        w.show()
        add_and_wait(w, tmp_path)
        insp = w.studio.inspector
        assert not insp.rebuild_panel.isVisibleTo(insp)  # not chosen: nothing to ask
        insp.options["rebuild"].radio.click()
        sh.settle(w)
        assert insp.rebuild_panel.isVisibleTo(insp) and "Voice rebuild needs an extra model" in insp.rebuild_panel._text.text()
        assert not w.studio.start.button.isEnabled() and any("Voice rebuild needs its model" in t for t, _ in w.studio.note_lines(w.doc))
        insp.rebuild_panel._button.click()
        assert started == ["rebuild-voice"]
        w.start()  # nothing starts, nothing asks
        assert not shown and not w.queue.active
        assert "rebuilding voices" in w.banner_models._label.text() or w.voice_download.running or started
        insp.options["noise"].radio.click()  # the other choice needs no model: the way out
        sh.settle(w)
        assert w.studio.start.button.isEnabled()
    finally:
        sh.shut(w)


def test_automatic_rebuilding_offers_the_download_but_never_holds_up_cleaning(qapp, tmp_path, monkeypatch):
    """With Automatic the voice is rebuilt only where its model is installed, so a missing model is offered
    (under Voice, when it is chosen) but Start doesn't wait for it."""
    monkeypatch.setattr(speech_model.SpeechModelDownloader, "ready", property(lambda self: self.model.id != "rebuild-voice"))
    monkeypatch.setattr("audio_cleaner_desktop.resources.voice_model_dir", lambda: None)
    shown = []
    monkeypatch.setattr(QMessageBox, "exec", lambda self: shown.append(self.text()) or QMessageBox.StandardButton.Yes)
    ran = []

    def runner(path, options, **_):
        ran.append((options.auto_rebuild, options.rebuild_voice))
        out = path.with_name("a-cleaned.wav")
        out.write_bytes(b"x")
        return engine.JobResult(out, 5.0, False)

    w = open_window(tmp_path, runner)
    try:
        assert w.prefs.auto_settings
        add_and_wait(w, tmp_path)
        assert w.studio.start.button.isEnabled()
        w.start()
        assert wait_until(lambda: ran, timeout=10) and ran == [(True, False)] and not shown
    finally:
        sh.shut(w)


def test_missing_noise_models_are_downloaded_by_the_window_and_a_failure_offers_try_again(qapp, tmp_path, monkeypatch, capsys):
    """The noise removal models are required: a window without them downloads just them by itself (a silent update shows no
    download page), a failed download offers "Try again" which downloads them again, Start waits for them with a note, the command line
    points at --download-models."""
    monkeypatch.setattr(speech_model.SpeechModelDownloader, "ready", property(lambda self: self.model is not resources.NOISE_MODEL))
    started, shown = [], []
    monkeypatch.setattr(speech_model.SpeechModelDownloader, "start", lambda self: started.append(self.model.id))
    monkeypatch.setattr(QMessageBox, "exec", lambda self: shown.append(self.text()) or QMessageBox.StandardButton.No)
    w = open_window(tmp_path)
    try:
        qapp.processEvents()
        assert started == ["noise-removal"] and not shown  # by itself, only the noise models (23 MB), not 2.1 GB
        assert w.banner_models.isVisible() and "Downloading the noise removal models" in w.banner_models._label.text()
        w._model_download_finished(w.noise_download, False, "The download stopped (timed out).")
        assert w.banner_models.isVisible() and "could not be downloaded" in w.banner_models._label.text()
        assert w.banner_models._button.text() == "Try again"
        started.clear()
        w.banner_models.action.emit()  # "Try again": the noise models again
        assert started == ["noise-removal"]
        src = add_and_wait(w, tmp_path)
        w.start()
        assert not shown and not w.queue.active and not w.studio.start.button.isEnabled()
        assert any("Noise removal needs its models" in t for t, _ in w.studio.note_lines(w.doc))
    finally:
        sh.shut(w)
    monkeypatch.setattr(resources, "check", lambda: [])
    monkeypatch.setattr(resources, "noise_ready", lambda: False)
    assert cli.run([str(src)], None, engine.JobOptions()) == 1
    assert "--download-models" in capsys.readouterr().err
