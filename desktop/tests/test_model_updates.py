"""A model whose catalog entry changed: downloaded again by itself when the user had it, never forced when they did
not, the outdated copy removed only once the new one is complete, and every download address a file of the app's own
GitHub release."""
from __future__ import annotations

import hashlib
import re
from pathlib import Path

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


def test_the_old_models_archives_partial_download_is_cleaned_up_once_nothing_waits_for_an_update(server):  # noqa: F811
    root = models.models_dir()
    m = _model(server, {"model.bin": b"x"})
    models.download(m)
    old = root / "AudioCleaner-models-v2.partial"  # what the app of the archive days left behind after a failed download
    old.mkdir()
    (old / ".pin").write_text("0" * 64)
    (old / "AudioCleaner-models-v2.zip.part").write_bytes(b"x" * 1000)
    (root / "LICENSES").mkdir()  # the archive's shared licence texts stay (About shows them for old copies)
    assert models.remove_stale([m]) == [old] and not old.exists() and (root / "LICENSES").is_dir()


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


# ---- the window ---------------------------------------------------------------------------------------------------------------
def _window(tmp_path, monkeypatch, started, outdated_ids):
    monkeypatch.setattr(speech_model.SpeechModelDownloader, "ready", property(lambda self: self.model.id in started or self.model is resources.NOISE_MODEL))
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


# ---- the updater uses the app's GitHub releases; the models come from Hugging Face ------------------------------------------
def test_the_release_addresses_are_defined_in_one_place():
    from audio_cleaner_desktop import __releases__

    assert re.fullmatch(r"[A-Za-z0-9-]+/[A-Za-z0-9._-]+", __releases__)
    assert config.RELEASES_URL == f"https://github.com/{__releases__}/releases/"
    assert config.LATEST_URL == config.RELEASES_URL + "latest/download/"
    assert config.release_url("2.0.41") == config.RELEASES_URL + "download/v2.0.41/"
    assert updater.MANIFEST_URL == config.LATEST_URL + "update.json"  # the newest release's
    assert updater.SIGNATURE_URL == config.LATEST_URL + "update.json.sig"
    assert updater.DOWNLOAD_BASE is None  # the installer: from the release of the version the manifest names
    for gone in ("DOWNLOADS_BASE", "UPDATES_URL", "MODELS_URL"):
        assert not hasattr(config, gone)
    # a repository move is one edit (CI writes it into __init__.py): nothing else spells the repository or the host out
    for path in list((DESKTOP / "audio_cleaner_desktop").rglob("*.py")) + list((DESKTOP / "packaging").glob("*.py")):
        text = path.read_text(encoding="utf-8")
        if path.name != "__init__.py" or path.parent.name != "audio_cleaner_desktop":
            assert __releases__ not in text and "releases/latest" not in text, path
        if path.name != "config.py":
            assert "mc-multi-world" not in text, path


def test_no_model_comes_from_the_releases_and_the_installers_download_exactly_the_catalog():
    import sys

    sys.path.insert(0, str(DESKTOP / "packaging"))
    import installer_models

    catalog = (resources.NOISE_MODEL, resources.SPEECH_MODEL, resources.HIGHS_MODEL, resources.VOICE_MODEL)
    wanted = {f.url for m in catalog for f in (*m.files, *m.licences)}
    assert all(u.startswith("https://huggingface.co/") for u in wanted) and not any(u.startswith(config.RELEASES_URL) for u in wanted)
    windows = installer_models.render()
    mac = installer_models.render_mac()
    for text in (windows, mac):
        assert {u.rstrip(":.,") for u in re.findall(r"https://[^\s'\"]+", text)} == wanted  # (a log line ends one with ":")
        assert "mc-multi-world" not in text and "api/downloads" not in text and ".zip" not in text and "tar.exe" not in text
    assert "mc-multi-world" not in installer_models.render_mac_preinstall()


def _private_repository_pattern() -> str | None:
    """The leak scan's pattern for the name of the private repository (tools/public-desktop/sync.py; the name is not
    written here, this file is published). None in the public copy, which has no such tool (its scan already ran)."""
    import importlib.util

    sync = DESKTOP.parent / "tools" / "public-desktop" / "sync.py"
    if not sync.is_file():
        return None
    spec = importlib.util.spec_from_file_location("public_desktop_sync", sync)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return {why: pattern for pattern, why in module.LEAK_PATTERNS}["the name of the private repository"]


def test_nothing_that_ships_names_the_private_repository_or_our_download_server():
    from audio_cleaner_desktop import __releases__

    owner = __releases__.split("/")[0]
    third_party = ("yxlu-0102", "sarulab-speech", "ceva-ip", "livekit", "xiph")  # other people's projects in notices
    shipped = [p for p in (DESKTOP / "audio_cleaner_desktop").rglob("*") if p.suffix in (".py", ".txt", ".md", ".json", ".qss", ".html")]
    shipped += [DESKTOP / "README.md", DESKTOP / "packaging" / "EULA.txt", *(DESKTOP / "packaging" / "licenses").glob("*")]
    shipped += [p for p in (DESKTOP / "packaging").glob("*") if p.suffix in (".py", ".iss", ".txt", ".plist") and p.name != "build_ffmpeg.sh"]
    shipped += list((DESKTOP / "packaging" / "windows").glob("*")) + list((DESKTOP / "packaging" / "macos").glob("*"))
    shipped = [p for p in shipped if p.is_file()]
    assert shipped
    private = _private_repository_pattern()
    named_owner = []
    for path in shipped:
        text = path.read_text(encoding="utf-8", errors="ignore")
        if private is not None:
            assert not re.search(private, text, re.I), path
        assert "api/downloads" not in text, path  # our server serves nothing of the desktop app any more
        if owner in text:
            named_owner.append(path.relative_to(DESKTOP).as_posix())
        for m in re.finditer(r"github\.com/([\w.-]+)", text):
            assert m[1].lower() in third_party, (path, m[0])  # the app's own repository only through config.RELEASES_URL
    # The public repository (where the releases are) is named in one place: the app's __init__.py, which CI rewrites.
    assert named_owner == ["audio_cleaner_desktop/__init__.py"]
