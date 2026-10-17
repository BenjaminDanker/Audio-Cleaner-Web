"""A model whose catalog entry changed: downloaded again by itself when the user had it, never forced when they did
not, the outdated copy removed only once the new one is complete, and every download address on our own server."""
from __future__ import annotations

import hashlib
import re
from pathlib import Path

import pytest
from PySide6.QtCore import QSettings

from audio_cleaner_desktop import config, models, resources, updater
from audio_cleaner_desktop.gui import speech_model
from audio_cleaner_desktop.gui.main_window import MainWindow

from conftest import wait_until
from test_models import _RangeHandler, _model, server  # noqa: F401  (the local download server fixture)

DESKTOP = Path(__file__).resolve().parents[1]


def _newer(m: models.SpeechModel, payload: bytes) -> models.SpeechModel:
    """The same model id with a new revision of its (one) file, served by the same test server."""
    name = m.files[0].path
    _RangeHandler.files["/" + name] = payload
    f = models.ModelFile(name, m.files[0].url, len(payload), hashlib.sha256(payload).hexdigest())
    return models.SpeechModel(m.id, m.title, m.description, m.licence, (f,))


def _installer_root(monkeypatch, tmp_path) -> Path:
    root = tmp_path / "app" / "models"
    monkeypatch.setattr(models, "installer_models_dir", lambda: root)
    monkeypatch.setattr(models, "installer_dir", lambda model: root / model.id)
    return root


def _put_copy(d: Path, m: models.SpeechModel, payloads: dict[str, bytes]) -> None:
    """A complete copy of `m` as setup or the app leaves it (files and the `.complete` marker)."""
    for f in m.files:
        (d / f.path).parent.mkdir(parents=True, exist_ok=True)
        (d / f.path).write_bytes(payloads[f.path])
    (d / ".complete").write_text("\n".join(f.sha256 for f in m.files), encoding="utf-8")


# ---- when the user had the model ---------------------------------------------------------------------------------------
def test_a_changed_model_the_user_had_needs_an_update_and_is_downloaded_again(server):  # noqa: F811
    old = _model(server, {"model.bin": b"revision one" * 1000})
    models.download(old)
    new = _newer(old, b"revision two" * 1000)
    assert models.is_installed(old) and not models.is_installed(new)
    assert models.had_before(new) and models.needs_update(new) and models.outdated([new]) == [new]
    models.download(new)
    assert models.is_installed(new) and (models.model_dir(new) / "model.bin").read_bytes() == b"revision two" * 1000
    assert not models.needs_update(new) and models.outdated([new]) == []
    assert models.remove_stale([new]) == []  # the new copy replaced the old one in place: nothing is left over


def test_a_model_the_user_never_had_is_not_an_update(server):  # noqa: F811
    m = _model(server, {"model.bin": b"never installed"})
    assert not models.had_before(m) and not models.needs_update(m) and models.outdated([m]) == []
    (models.models_dir() / m.id).mkdir(parents=True)  # a folder without the marker (a leftover) is not "had" either
    (models.models_dir() / m.id / "model.bin").write_bytes(b"half")
    assert models.outdated([m]) == []
    models.remove(m)
    models.download(m)
    models.remove(m)  # the user deleted it themselves: gone is gone
    assert models.outdated([m]) == []


def test_the_installers_older_copy_also_counts_as_having_the_model(server, monkeypatch, tmp_path):  # noqa: F811
    root = _installer_root(monkeypatch, tmp_path)
    old = _model(server, {"model.bin": b"one"})
    _put_copy(root / old.id, old, {"model.bin": b"one"})
    new = _newer(old, b"two!")
    assert models.had_before(new) and models.outdated([new]) == [new]
    assert models.outdated([old]) == []


def test_an_archive_model_is_updated_by_one_download_of_the_new_archive(server, monkeypatch):  # noqa: F811
    from test_models import _archive_models

    monkeypatch.setattr(models, "CATALOG", {})
    _, one, two, members = _archive_models(server)
    models.register(one, two)
    models.download(one)
    _RangeHandler.requests.clear()
    # the next archive: one file of "one" changed
    changed = dict(members, **{"one/a.onnx": b"new revision" * 50_000})
    _, one2, two2, _ = _archive_models(server, changed)
    models.register(one2, two2)
    assert models.outdated() == [one2] and models.is_installed(two2)  # "two" did not change: still installed
    models.download(one2)
    assert models.outdated() == [] and _RangeHandler.requests == ["/models.zip"]
    assert (models.model_dir(one2) / "a.onnx").read_bytes() == changed["one/a.onnx"]


# ---- removing what is outdated -------------------------------------------------------------------------------------------
def test_an_outdated_copy_is_removed_only_after_the_new_one_is_complete(server, monkeypatch, tmp_path):  # noqa: F811
    root = _installer_root(monkeypatch, tmp_path)
    old = _model(server, {"model.bin": b"one"})
    _put_copy(root / old.id, old, {"model.bin": b"one"})  # setup's copy of the old revision
    new = _newer(old, b"two!")
    mine = models.model_dir(new)
    _put_copy(mine, old, {"model.bin": b"one"})  # and the app's own copy of it
    assert models.remove_stale([new]) == []  # the new revision is not here yet: both old copies are all there is
    assert (mine / "model.bin").is_file() and (root / old.id / "model.bin").is_file()
    # a half-finished download of the new revision does not change that
    models.models_dir().joinpath(f"{new.id}.partial").mkdir()
    models.models_dir().joinpath(f"{new.id}.partial", ".pin").write_text(models.pin_token(new))
    assert models.remove_stale([new]) == [] and mine.is_dir()
    # the new revision is complete in the installer's folder (e.g. a later setup): the app's old copy goes
    (root / old.id / "model.bin").write_bytes(b"two!")
    (root / old.id / ".complete").write_text(new.files[0].sha256)
    assert models.is_installed(new)
    assert models.remove_stale([new]) == [mine] and not mine.exists()
    assert (models.models_dir() / f"{new.id}.partial").is_dir()  # the current download's partial stays


def test_models_that_left_the_catalog_and_their_partials_go_once_nothing_waits_for_an_update(server):  # noqa: F811
    root = models.models_dir()
    old = _model(server, {"model.bin": b"one"})
    _put_copy(root / "retired", old, {"model.bin": b"one"})
    (root / "retired.partial").mkdir()
    (root / "Old-models-v0.partial").mkdir()
    (root / "Old-models-v0.partial" / "x.zip.part").write_bytes(b"x")
    new = _newer(old, b"two!")
    _put_copy(models.model_dir(new), old, {"model.bin": b"one"})  # the user's copy is outdated: an update is waiting
    assert models.remove_stale([new]) == []
    assert (root / "retired").is_dir()
    models.download(new)  # the update is done
    gone = {p.name for p in models.remove_stale([new])}
    assert gone == {"retired", "retired.partial", "Old-models-v0.partial"}
    assert (models.model_dir(new) / "model.bin").read_bytes() == b"two!"


def test_partials_of_other_hashes_are_discarded_and_matching_ones_resumed(server):  # noqa: F811
    data = bytes(range(256)) * 8000
    m = _model(server, {"big.bin": data})
    part = models.models_dir() / f"{m.id}.partial"
    part.mkdir(parents=True)
    (part / "big.bin.part").write_bytes(data[:500_000])
    assert models.missing_bytes(m) == len(data)  # no pin: written by something else, not counted
    (part / ".pin").write_text("0" * 64)  # started for another revision
    assert models.missing_bytes(m) == len(data)
    assert models.remove_stale([m]) == [part]
    part.mkdir()
    (part / "big.bin.part").write_bytes(data[:500_000])  # the same bytes, but no proof they are for this revision
    models.download(m)
    assert _RangeHandler.requests == ["/big.bin"]  # started from zero, not resumed
    assert (models.model_dir(m) / "big.bin").read_bytes() == data and not (models.model_dir(m) / ".pin").exists()


def test_the_licences_and_foreign_folders_are_never_deleted(server):  # noqa: F811
    root = models.models_dir()
    new = _model(server, {"model.bin": b"x"})
    models.download(new)
    (root / "LICENSES").mkdir()
    (root / "LICENSES" / "A-MIT.txt").write_text("licence")
    (root / "photos").mkdir()  # something else in a shared folder, no marker: not ours
    (root / "photos" / "a.jpg").write_text("x")
    (root / "notes.txt").write_text("x")
    assert models.remove_stale([new]) == []
    assert (root / "LICENSES" / "A-MIT.txt").is_file() and (root / "photos" / "a.jpg").is_file() and (root / "notes.txt").is_file()


def test_the_installers_folder_is_cleaned_only_where_it_can_be_written(server, monkeypatch, tmp_path):  # noqa: F811
    root = _installer_root(monkeypatch, tmp_path)
    old = _model(server, {"model.bin": b"one"})
    stale = root / old.id
    _put_copy(stale, old, {"model.bin": b"one"})
    (root / "LICENSES").mkdir()
    (root / "LICENSES" / "A-MIT.txt").write_text("licence")
    new = _newer(old, b"two!")
    models.download(new)  # the app's own copy of the new revision is complete
    real = models._writable
    monkeypatch.setattr(models, "_writable", lambda path: False)  # Program Files / /Library for a user without rights
    assert models.remove_stale([new]) == [] and stale.is_dir()
    monkeypatch.setattr(models, "_writable", real)
    assert models.remove_stale([new]) == [stale] and not stale.exists()
    assert (root / "LICENSES" / "A-MIT.txt").is_file()


@pytest.mark.skipif(not hasattr(__import__("os"), "geteuid") or __import__("os").geteuid() == 0, reason="root can write anywhere")
def test_a_read_only_installer_folder_is_left_alone_for_real(server, monkeypatch, tmp_path):  # noqa: F811
    root = _installer_root(monkeypatch, tmp_path)
    old = _model(server, {"model.bin": b"one"})
    _put_copy(root / old.id, old, {"model.bin": b"one"})
    new = _newer(old, b"two!")
    models.download(new)
    root.chmod(0o555)  # the folder cannot be changed, so a model folder in it cannot be removed
    try:
        assert models.remove_stale([new]) == [] and (root / old.id / "model.bin").is_file()
    finally:
        root.chmod(0o755)


# ---- the window ---------------------------------------------------------------------------------------------------------------
def _window(tmp_path, monkeypatch, started, outdated_ids):
    monkeypatch.setattr(speech_model.SpeechModelDownloader, "ready", property(lambda self: self.model.id in started))
    monkeypatch.setattr(speech_model.SpeechModelDownloader, "start", lambda self: started.append(self.model.id))
    monkeypatch.setattr(models, "needs_update", lambda m: m.id in outdated_ids)
    cleaned = []
    monkeypatch.setattr(models, "remove_stale", lambda *a: cleaned.append(1) or [])
    w = MainWindow([], settings=QSettings(str(tmp_path / "s.ini"), QSettings.Format.IniFormat), history_path=tmp_path / "h.db")
    return w, cleaned


def _close(w):
    w.queue.shutdown()
    w.history.close()
    w._pool.waitForDone()


def test_the_window_updates_only_the_models_that_changed_quietly_and_tidies_up_after(qapp, tmp_path, monkeypatch):
    started: list[str] = []
    w, cleaned = _window(tmp_path, monkeypatch, started, {"restore-highs", "rebuild-voice"})
    try:
        w.show()
        qapp.processEvents()
        assert started == []  # nothing happens before the scheduled start-up step
        w.update_models()
        assert started == ["restore-highs"] and w.banner_models.isVisible()
        assert "Updating the models" in w.banner_models._label.text() and w._auto_models
        w._model_download_finished(w.highs_download, True, "")
        assert started == ["restore-highs", "rebuild-voice"]  # the speech model, which the user did not have, is skipped
        w._model_download_finished(w.voice_download, True, "")
        assert not w.banner_models.isVisible() and not w._auto_models
        assert wait_until(lambda: cleaned, timeout=5)  # outdated copies are removed after the last one is complete
    finally:
        _close(w)


def test_a_failed_automatic_update_is_not_retried_and_does_not_nag(qapp, tmp_path, monkeypatch):
    started: list[str] = []
    w, cleaned = _window(tmp_path, monkeypatch, started, {"restore-highs", "rebuild-voice"})
    try:
        w.show()
        w.update_models()
        w.highs_download.last_error = "The download stopped (timed out)."
        w._model_download_finished(w.highs_download, False, "The download stopped (timed out).")
        assert started == ["restore-highs"]  # one try per start: the next model is not attempted either
        assert not w.banner_models.isVisible() and not w._auto_models and w._model_queue == []
        panel = speech_model.SpeechModelPanel(w.highs_download)
        assert "timed out" in panel._text.text() and panel._button.text() == "Try again"  # it shows in Settings
        w.update_models()  # a manual "Try again" is a normal download; update_models itself runs once per start
        assert not cleaned  # nothing was tidied while an update is still missing
    finally:
        _close(w)


def test_nothing_to_update_only_tidies_up(qapp, tmp_path, monkeypatch):
    started: list[str] = []
    w, cleaned = _window(tmp_path, monkeypatch, started, set())
    try:
        w.show()
        w.update_models()
        assert started == [] and not w.banner_models.isVisible()
        assert wait_until(lambda: cleaned, timeout=5)
    finally:
        _close(w)


# ---- our own server for every download -----------------------------------------------------------------------------------------
def test_the_download_base_is_defined_in_one_place():
    assert config.DOWNLOADS_BASE == "https://mc-multi-world.space/audio/api/downloads/"
    assert config.UPDATES_URL == config.DOWNLOADS_BASE + "desktop/" and config.MODELS_URL == config.DOWNLOADS_BASE + "models/"
    assert updater.MANIFEST_URL == config.UPDATES_URL + "update.json"
    assert updater.SIGNATURE_URL == config.UPDATES_URL + "update.json.sig"
    assert updater.DOWNLOAD_BASE == config.UPDATES_URL
    assert resources.MODELS_ARCHIVE.url == config.MODELS_URL + "AudioCleaner-models-v2.zip"
    # a domain move is one edit: nothing else spells the host out
    for path in list((DESKTOP / "audio_cleaner_desktop").rglob("*.py")) + list((DESKTOP / "packaging").glob("*.py")):
        if path.name != "config.py":
            assert "mc-multi-world" not in path.read_text(encoding="utf-8"), path


def test_every_download_of_the_catalog_is_on_our_server_or_hugging_face():
    for m in (resources.SPEECH_MODEL, resources.HIGHS_MODEL, resources.VOICE_MODEL):
        urls = [m.archive.url] if m.archive is not None else [f.url for f in m.files]
        for url in urls:
            assert url.startswith((config.MODELS_URL, "https://huggingface.co/")), url


def test_the_installers_and_the_model_check_use_the_same_addresses():
    import sys

    sys.path.insert(0, str(DESKTOP / "packaging"))
    import check_model_urls
    import installer_models

    archive = resources.MODELS_ARCHIVE
    assert (archive.url, archive.size) in check_model_urls.catalog_files()
    windows = installer_models.render()
    mac = installer_models.render_mac()
    assert archive.url in windows and archive.url in mac
    for text in (windows, mac, installer_models.render_mac_preinstall()):
        assert "github" not in text.lower() and "BenjaminDanker" not in text
    urls = set(re.findall(r"https://[^\s'\"]+", windows + mac))
    assert {u.split("/")[2] for u in urls} == {"mc-multi-world.space", "huggingface.co"}


def test_nothing_that_ships_names_the_github_repository():
    third_party = ("yxlu-0102", "sarulab-speech", "ceva-ip", "livekit", "xiph")  # other people's projects in notices
    shipped = [p for p in (DESKTOP / "audio_cleaner_desktop").rglob("*") if p.suffix in (".py", ".txt", ".md", ".json", ".qss", ".html")]
    shipped += [DESKTOP / "README.md", DESKTOP / "packaging" / "EULA.txt", *(DESKTOP / "packaging" / "licenses").glob("*")]
    shipped += [p for p in (DESKTOP / "packaging").glob("*") if p.suffix in (".py", ".iss", ".txt", ".plist") and p.name != "build_ffmpeg.sh"]
    shipped += list((DESKTOP / "packaging" / "windows").glob("*")) + list((DESKTOP / "packaging" / "macos").glob("*"))
    shipped = [p for p in shipped if p.is_file()]
    assert shipped
    for path in shipped:
        text = path.read_text(encoding="utf-8", errors="ignore")
        assert "BenjaminDanker" not in text, path
        for m in re.finditer(r"github\.com/([\w.-]+)", text):
            assert m[1].lower() in third_party, (path, m[0])
        if path.suffix in (".py", ".txt", ".md") and path.parent.name != "licenses":
            body = re.sub(r"GITHUB_\w+|\.github|github\.com/[\w.-]+", "", text)
            assert "GitHub" not in body, path
