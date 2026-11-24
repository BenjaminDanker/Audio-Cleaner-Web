"""Automatic updates: the signed manifest, the download, the 24 hour rule, the settings, the Windows command line,
the installer script's relaunch entry and the release job's signing. No network (a local HTTP server stands in for
the download server) and never the real signing key: every test makes its own key pair."""
from __future__ import annotations

import base64
import hashlib
import http.server
import json
import os
import re
import shutil
import subprocess
import sys
import threading
import time
import types
from pathlib import Path

import pytest
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
from PySide6.QtCore import QSettings

from audio_cleaner_desktop import config, models, updater
from audio_cleaner_desktop.gui.preferences import Preferences
from conftest import wait_until

ROOT = Path(__file__).resolve().parents[2]
PACKAGING = ROOT / "desktop" / "packaging"
if str(PACKAGING) not in sys.path:
    sys.path.insert(0, str(PACKAGING))

import make_update_manifest  # noqa: E402

WIN = "windows-x86_64"


def _public(key: Ed25519PrivateKey) -> str:
    return base64.b64encode(key.public_key().public_bytes(serialization.Encoding.Raw, serialization.PublicFormat.Raw)).decode()


def _manifest(version: str = "2.0.15", installer: bytes = b"installer-bytes" * 1000, **override) -> tuple[bytes, bytes]:
    """(update.json bytes, the installer they describe)."""
    name = f"AudioCleaner-{version}-Windows.exe"
    body = {
        "version": version, "published": "2026-10-16T12:00:00Z",
        "files": {
            WIN: {"name": name, "size": len(installer), "sha256": hashlib.sha256(installer).hexdigest()},
            "macos-arm64": {"name": f"AudioCleaner-{version}-Mac-Apple-silicon.pkg", "size": 5, "sha256": "a" * 64},
            "macos-x86_64": {"name": f"AudioCleaner-{version}-Mac-Intel.pkg", "size": 5, "sha256": "b" * 64},
        },
        **override,
    }
    return (json.dumps(body, indent=2, sort_keys=True) + "\n").encode(), installer


@pytest.fixture
def key() -> Ed25519PrivateKey:
    return Ed25519PrivateKey.generate()


# ---- the manifest and its signature -----------------------------------------------------------------------------
def test_the_embedded_key_is_a_valid_ed25519_public_key_that_ci_reads_the_same_way():
    from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PublicKey

    raw = base64.b64decode(updater.PUBLIC_KEY, validate=True)
    assert len(raw) == 32 and Ed25519PublicKey.from_public_bytes(raw)
    assert updater.PUBLIC_KEY == make_update_manifest.app_public_key()  # CI reads it from the source the same way


def test_only_the_exact_bytes_signed_by_the_right_key_are_accepted(key):
    data, _ = _manifest()
    sig = key.sign(data)
    update = updater.decide(data, sig, "2.0.3", WIN, _public(key))
    assert update.version == "2.0.15" and update.file.name == "AudioCleaner-2.0.15-Windows.exe"
    assert update.manifest == data
    with pytest.raises(updater.UpdateError, match="not signed correctly"):
        updater.decide(data.replace(b"2.0.15", b"9.9.99", 1), sig, "2.0.3", WIN, _public(key))  # tampered
    with pytest.raises(updater.UpdateError, match="not signed correctly"):
        updater.decide(data + b" ", sig, "2.0.3", WIN, _public(key))  # even a trailing space: the exact bytes count
    other = Ed25519PrivateKey.generate()
    with pytest.raises(updater.UpdateError, match="not signed correctly"):
        updater.decide(data, other.sign(data), "2.0.3", WIN, _public(key))
    with pytest.raises(updater.UpdateError):  # and nothing passes against the real embedded key
        updater.decide(data, other.sign(data), "2.0.3", WIN)


@pytest.mark.parametrize("sig", [b"", b"x" * 63], ids=["empty", "short"])
def test_a_missing_or_malformed_signature_is_refused(key, sig):
    data, _ = _manifest()
    with pytest.raises(updater.UpdateError):
        updater.decide(data, sig, "2.0.3", WIN, _public(key))


def test_versions_are_compared_as_numbers_and_only_a_newer_manifest_is_offered(key):
    assert updater.parse_version("2.0.15") > updater.parse_version("2.0.9") > updater.parse_version("1.99.99")
    for bad in ("", "2", "2.0.15-beta", "v2.0.15", "2.0.15\n", None):
        with pytest.raises(updater.UpdateError):
            updater.parse_version(bad)
    data, _ = _manifest("2.0.15")
    sig = key.sign(data)
    assert updater.decide(data, sig, "2.0.15", WIN, _public(key)) is None  # the same version
    assert updater.decide(data, sig, "2.0.16", WIN, _public(key)) is None  # the running app is newer
    assert updater.decide(data, sig, "2.1.0", WIN, _public(key)) is None  # numbers, not text: 2.1.0 > 2.0.15
    assert updater.decide(data, sig, "2.0.9", WIN, _public(key)) is not None  # 15 > 9 (not a string compare)


def test_a_newer_manifest_without_a_file_for_this_platform_is_an_error(key):
    data, _ = _manifest()
    with pytest.raises(updater.UpdateError, match="no installer for this computer"):
        updater.decide(data, key.sign(data), "2.0.3", None, _public(key))
    body = json.loads(data)
    del body["files"]["macos-arm64"]
    data2 = json.dumps(body, sort_keys=True).encode()
    with pytest.raises(updater.UpdateError, match="no installer for this computer"):
        updater.decide(data2, key.sign(data2), "2.0.3", "macos-arm64", _public(key))
    assert updater.decide(data2, key.sign(data2), "2.0.3", WIN, _public(key)) is not None  # the other platforms still work


@pytest.mark.parametrize("change", [
    {"version": "latest"},
    {"files": {WIN: {"name": "../evil.exe", "size": 5, "sha256": "a" * 64}}},
    {"files": {WIN: {"name": "sub/evil.exe", "size": 5, "sha256": "a" * 64}}},
    {"files": {WIN: {"name": "evil.pkg", "size": 5, "sha256": "a" * 64}}},  # the wrong kind of file for the platform
    {"files": {WIN: {"name": "a.exe", "size": -1, "sha256": "a" * 64}}},
    {"files": {WIN: {"name": "a.exe", "size": 3 << 30, "sha256": "a" * 64}}},
    {"files": {WIN: {"name": "a.exe", "size": 5, "sha256": "A" * 64}}},
], ids=lambda c: next(iter(c)) + ":" + str(next(iter(c.values())))[:40])
def test_a_signed_but_malformed_manifest_is_refused(key, change):
    data, _ = _manifest(**change)
    with pytest.raises(updater.UpdateError):
        updater.decide(data, key.sign(data), "2.0.3", WIN, _public(key))


# ---- a local server standing in for the download server ---------------------------------------------------------------
class _Handler(http.server.BaseHTTPRequestHandler):
    files: dict[str, bytes] = {}
    requests: list[str] = []

    def do_GET(self):  # noqa: N802
        self.requests.append(f"{self.path} {self.headers.get('Range', '')}".strip())
        body = self.files.get(self.path)
        if body is None:
            self.send_error(404)
            return
        start = 0
        rng = self.headers.get("Range")
        if rng and rng.startswith("bytes="):
            start = int(rng[6:].split("-")[0])
            self.send_response(206)
            self.send_header("Content-Range", f"bytes {start}-{len(body) - 1}/{len(body)}")
        else:
            self.send_response(200)
        self.send_header("Content-Length", str(len(body) - start))
        self.end_headers()
        self.wfile.write(body[start:])

    def log_message(self, *_args):
        pass


@pytest.fixture
def server(monkeypatch):
    _Handler.files, _Handler.requests = {}, []
    httpd = http.server.ThreadingHTTPServer(("127.0.0.1", 0), _Handler)
    threading.Thread(target=httpd.serve_forever, kwargs={"poll_interval": 0.01}, daemon=True).start()
    monkeypatch.setattr(models, "_ssl_context", lambda: None)  # the real addresses are https only; this one is plain http
    base = f"http://127.0.0.1:{httpd.server_address[1]}"
    yield base
    httpd.shutdown()


def _publish(base: str, key: Ed25519PrivateKey, version: str = "2.0.15", installer: bytes = b"installer-bytes" * 1000,
             sign_with: Ed25519PrivateKey | None = None) -> bytes:
    data, installer = _manifest(version, installer)
    _Handler.files["/desktop/update.json"] = data
    _Handler.files["/desktop/update.json.sig"] = (sign_with or key).sign(data)
    _Handler.files[f"/desktop/AudioCleaner-{version}-Windows.exe"] = installer
    return installer


def _endpoints(base: str) -> dict:
    return {"manifest_url": f"{base}/desktop/update.json", "signature_url": f"{base}/desktop/update.json.sig"}


def test_check_fetches_and_verifies_and_refuses_a_manifest_signed_by_someone_else(server, key):
    _publish(server, key)
    update = updater.check("2.0.3", WIN, public_key=_public(key), **_endpoints(server))
    assert update.version == "2.0.15"
    assert updater.check("2.0.15", WIN, public_key=_public(key), **_endpoints(server)) is None
    assert all(r.startswith("/desktop/update.json") for r in _Handler.requests)  # no query string, no extras
    _publish(server, key, sign_with=Ed25519PrivateKey.generate())
    with pytest.raises(updater.UpdateError, match="not signed correctly"):
        updater.check("2.0.3", WIN, public_key=_public(key), **_endpoints(server))


def test_check_says_so_when_nothing_is_published_or_the_server_is_down(server, key):
    with pytest.raises(updater.UpdateError, match="No update information is published yet"):
        updater.check("2.0.3", WIN, public_key=_public(key), **_endpoints(server))
    _Handler.files["/desktop/update.json"] = b"{}"  # but no signature
    with pytest.raises(updater.UpdateError, match="No update information is published yet"):
        updater.check("2.0.3", WIN, public_key=_public(key), **_endpoints(server))
    with pytest.raises(updater.UpdateError, match="could not be reached"):
        updater.check("2.0.3", WIN, public_key=_public(key), manifest_url="http://127.0.0.1:9/update.json",
                      signature_url="http://127.0.0.1:9/update.json.sig")


def test_check_refuses_an_oversized_answer(server, key):
    _Handler.files["/desktop/update.json"] = b"x" * (updater.MAX_MANIFEST_BYTES + 1)
    _Handler.files["/desktop/update.json.sig"] = b"x" * 64
    with pytest.raises(updater.UpdateError, match="larger than expected"):
        updater.check("2.0.3", WIN, public_key=_public(key), **_endpoints(server))


# ---- the download -------------------------------------------------------------------------------------------------
def test_download_verifies_keeps_the_manifest_and_deletes_older_downloads(server, key, tmp_path):
    installer = _publish(server, key)
    folder = tmp_path / "updates"
    folder.mkdir()
    (folder / "AudioCleaner-2.0.9-Windows.exe").write_bytes(b"old")
    (folder / "AudioCleaner-2.0.9-Windows.exe.part").write_bytes(b"older still")
    (folder / "notes.txt").write_text("not ours")
    update = updater.check("2.0.3", WIN, public_key=_public(key), **_endpoints(server))
    seen = []
    path = updater.download(update, folder, lambda d, t: seen.append((d, t)), base_url=f"{server}/desktop")
    assert path == folder / "AudioCleaner-2.0.15-Windows.exe" and path.read_bytes() == installer
    assert seen[-1] == (len(installer), len(installer))
    assert sorted(p.name for p in folder.iterdir()) == [
        "AudioCleaner-2.0.15-Windows.exe", "notes.txt", "update.json", "update.json.sig"]
    # a restart finds it again without the network, after verifying the saved manifest once more
    found = updater.pending(folder, "2.0.3", WIN, _public(key))
    assert found is not None and found[0].version == "2.0.15" and found[1] == path


def test_an_interrupted_download_resumes_and_a_finished_one_is_not_fetched_again(server, key, tmp_path):
    installer = _publish(server, key, installer=bytes(range(256)) * 8000)
    update = updater.check("2.0.3", WIN, public_key=_public(key), **_endpoints(server))
    folder = tmp_path / "updates"
    folder.mkdir()
    (folder / "AudioCleaner-2.0.15-Windows.exe.part").write_bytes(installer[:700_000])
    _Handler.requests.clear()
    path = updater.download(update, folder, base_url=f"{server}/desktop")
    assert path.read_bytes() == installer
    assert _Handler.requests == ["/desktop/AudioCleaner-2.0.15-Windows.exe bytes=700000-"]
    _Handler.requests.clear()
    updater.download(update, folder, base_url=f"{server}/desktop")
    assert _Handler.requests == []


def test_a_download_that_does_not_match_the_manifest_is_never_kept(server, key, tmp_path):
    _publish(server, key)
    update = updater.check("2.0.3", WIN, public_key=_public(key), **_endpoints(server))
    _Handler.files["/desktop/AudioCleaner-2.0.15-Windows.exe"] = b"evil" * 3750  # same size, other bytes
    folder = tmp_path / "updates"
    with pytest.raises(models.DownloadError, match="did not download correctly"):
        updater.download(update, folder, base_url=f"{server}/desktop")
    assert not (folder / "AudioCleaner-2.0.15-Windows.exe").exists()
    assert not list(folder.glob("*.part")) and not (folder / "update.json").exists()
    assert updater.pending(folder, "2.0.3", WIN, _public(key)) is None


def test_a_server_that_sends_more_than_the_pinned_size_is_cut_off_and_nothing_is_kept(server, tmp_path, monkeypatch):
    size = 100_000
    pinned = b"p" * size
    _Handler.files["/big.bin"] = pinned + b"x" * 600_000  # the pinned file, and then some
    monkeypatch.setattr(models, "_CHUNK", 64 * 1024)  # (so the first chunk fits and the second one does not)
    dest = tmp_path / "big.bin"
    on_disk = []
    with pytest.raises(models.DownloadError, match="did not download correctly"):
        models.fetch_file(f"{server}/big.bin", dest, size, hashlib.sha256(pinned).hexdigest(),
                          lambda done, total: on_disk.append(dest.with_name("big.bin.part").stat().st_size))
    assert on_disk and max(on_disk) <= size  # the extra bytes were never written
    assert not list(tmp_path.glob("big.bin*"))  # neither the part nor the file


def test_the_installer_comes_from_the_release_of_the_version_the_manifest_names(server, key, tmp_path, monkeypatch):
    # The manifest is the newest release's (config.LATEST_URL); the installer is a file of that version's own release
    # (config.release_url), whatever version the app itself is.
    monkeypatch.setattr(config, "RELEASES_URL", f"{server}/releases/")
    installer = _publish(server, key, version="2.0.15")
    _Handler.files["/releases/download/v2.0.15/AudioCleaner-2.0.15-Windows.exe"] = installer
    del _Handler.files["/desktop/AudioCleaner-2.0.15-Windows.exe"]
    update = updater.check("2.0.3", WIN, public_key=_public(key), **_endpoints(server))
    _Handler.requests.clear()
    path = updater.download(update, tmp_path / "u")  # no base_url: the default, as the app does it
    assert path.read_bytes() == installer
    assert [r.split()[0] for r in _Handler.requests] == ["/releases/download/v2.0.15/AudioCleaner-2.0.15-Windows.exe"]


def test_the_release_download_url_is_https_github_built_from_version_and_name(key, tmp_path, monkeypatch):
    data, _ = _manifest("2.0.15")
    update = updater.decide(data, key.sign(data), "2.0.3", WIN, _public(key))
    seen = []

    def fake_fetch(url, *args, **kwargs):
        seen.append(url)
        raise models.DownloadError("stop here")

    monkeypatch.setattr(models, "fetch_file", fake_fetch)
    with pytest.raises(models.DownloadError):
        updater.download(update, tmp_path / "u")
    assert seen == [config.release_url("2.0.15") + "AudioCleaner-2.0.15-Windows.exe"]  # the URL is not from the manifest
    assert seen[0].startswith("https://github.com/")
    assert not (tmp_path / "u" / "update.json").exists()  # nothing is remembered without the installer


def test_pending_is_none_for_a_damaged_installer_and_cleans_up_after_an_install(server, key, tmp_path):
    installer = _publish(server, key)
    update = updater.check("2.0.3", WIN, public_key=_public(key), **_endpoints(server))
    folder = tmp_path / "updates"
    path = updater.download(update, folder, base_url=f"{server}/desktop")
    path.write_bytes(installer[:-1] + b"X")  # same size, one byte changed
    assert updater.pending(folder, "2.0.3", WIN, _public(key)) is None
    path.write_bytes(installer)
    assert updater.pending(folder, "2.0.3", WIN, _public(key)) is not None
    (folder / "update.json").write_bytes(b'{"version": "9.9.9"}')  # tampered after the download
    assert updater.pending(folder, "2.0.3", WIN, _public(key)) is None
    assert not list(folder.iterdir())  # an unusable saved manifest takes the download with it
    # the update was installed: the running app is now 2.0.15, so nothing is kept
    path = updater.download(update, folder, base_url=f"{server}/desktop")
    assert updater.pending(folder, "2.0.15", WIN, _public(key)) is None
    assert not path.exists() and not list(folder.iterdir())


def test_prune_keeps_only_the_named_installer(tmp_path):
    for n in ("AudioCleaner-2.0.8-Windows.exe", "AudioCleaner-2.0.9-Windows.exe", "AudioCleaner-2.0.9-Windows.exe.part",
              "AudioCleaner-2.0.8-Mac-Intel.pkg", "update.json", "other.txt"):
        (tmp_path / n).write_bytes(b"x")
    updater.prune(tmp_path, "AudioCleaner-2.0.9-Windows.exe")
    assert sorted(p.name for p in tmp_path.iterdir()) == [
        "AudioCleaner-2.0.9-Windows.exe", "AudioCleaner-2.0.9-Windows.exe.part", "other.txt", "update.json"]
    updater.prune(tmp_path, None)
    assert [p.name for p in tmp_path.iterdir()] == ["other.txt"]


# ---- the controller: the 24 hour rule, the setting, the whole flow ------------------------------------------------
def _prefs(tmp_path) -> Preferences:
    return Preferences(QSettings(str(tmp_path / "prefs.ini"), QSettings.Format.IniFormat))


def _controller(qapp, tmp_path, server, key, current="2.0.3", clock=None, **kw):
    prefs = kw.pop("prefs", None) or _prefs(tmp_path)
    ctl = updater.UpdateController(
        prefs, enabled=True, folder=tmp_path / "updates", current=current, key=WIN, public_key=_public(key),
        download_base=f"{server}/desktop", clock=clock or (lambda: 1_000_000.0), **_endpoints(server), **kw)
    return ctl, prefs


def test_the_24_hour_rule(qapp, tmp_path):
    now = [1_000_000.0]
    prefs = _prefs(tmp_path)
    ctl = updater.UpdateController(prefs, enabled=True, clock=lambda: now[0])
    assert ctl.due()  # never checked
    prefs.update_last_check = now[0]
    assert not ctl.due()
    now[0] += 23 * 3600 + 3000
    assert not ctl.due()
    now[0] += 1000
    assert ctl.due()  # more than 24 h
    prefs.update_last_check = now[0] + 5 * 3600  # the clock was set back: do not wait for days
    assert ctl.due()
    prefs.update_last_check = now[0]
    prefs.check_updates = False
    now[0] += 3 * 24 * 3600
    assert not ctl.due()  # switched off: never


def test_startup_check_downloads_once_a_day_and_remembers_the_update(qapp, tmp_path, server, key):
    installer = _publish(server, key)
    now = [2_000_000.0]
    ctl, prefs = _controller(qapp, tmp_path, server, key, clock=lambda: now[0])
    seen, ready = [], []
    ctl.statusChanged.connect(seen.append)
    ctl.readyChanged.connect(lambda: ready.append(ctl.ready_version))
    ctl.startup_check()
    assert wait_until(lambda: ready, timeout=30)
    assert ready == ["2.0.15"] and ctl.ready_path.read_bytes() == installer and not ctl.busy
    assert "Checking for updates..." in seen and "Downloading 2.0.15..." in seen
    assert prefs.update_last_check == now[0]

    # the same day, another start: no request to the server at all, the downloaded update is found on disk
    _Handler.requests.clear()
    now[0] += 3600
    ctl2, _ = _controller(qapp, tmp_path, server, key, clock=lambda: now[0], prefs=prefs)
    ready2 = []
    ctl2.readyChanged.connect(lambda: ready2.append(ctl2.ready_version))
    ctl2.startup_check()
    assert wait_until(lambda: ready2, timeout=30)
    assert ready2 == ["2.0.15"] and _Handler.requests == []

    # a day later it asks again, finds the same version and downloads nothing
    now[0] += 24 * 3600
    ctl3, _ = _controller(qapp, tmp_path, server, key, clock=lambda: now[0], prefs=prefs)
    ready3 = []
    ctl3.readyChanged.connect(lambda: ready3.append(1))
    ctl3.startup_check()
    assert wait_until(lambda: ready3 and not ctl3.busy, timeout=30)
    assert [r.split()[0] for r in _Handler.requests] == ["/desktop/update.json", "/desktop/update.json.sig"]
    assert prefs.update_last_check == now[0]


def test_check_now_reports_the_result_and_a_bad_manifest_downloads_nothing(qapp, tmp_path, server, key):
    _publish(server, key, "2.0.3")  # the newest release is the running version
    ctl, prefs = _controller(qapp, tmp_path, server, key)
    ctl.check_now()
    assert wait_until(lambda: not ctl.busy and ctl.status.startswith("You have"), timeout=30)
    assert ctl.status == "You have the newest version (2.0.3)." and ctl.ready_version is None

    _publish(server, key, sign_with=Ed25519PrivateKey.generate())
    ctl.check_now()
    assert wait_until(lambda: not ctl.busy and "signed" in ctl.status, timeout=30)
    assert ctl.ready_version is None
    assert not list((tmp_path / "updates").glob("AudioCleaner-*"))  # nothing from a bad manifest is downloaded


def test_a_failed_check_does_not_use_up_the_day(qapp, tmp_path, server, key):
    ctl, prefs = _controller(qapp, tmp_path, server, key)
    ctl.check_now()
    assert wait_until(lambda: not ctl.busy and ctl.status.startswith("Could not"), timeout=30)
    assert prefs.update_last_check == 0 and ctl.due()


def test_the_update_setting_defaults_to_on_and_is_remembered(qapp, tmp_path):
    ini = tmp_path / "s.ini"
    assert Preferences(QSettings(str(ini), QSettings.Format.IniFormat)).check_updates is True
    p = Preferences(QSettings(str(ini), QSettings.Format.IniFormat))
    p.check_updates = False
    p._store.sync()
    assert Preferences(QSettings(str(ini), QSettings.Format.IniFormat)).check_updates is False


def _offline_controller(tmp_path, key, enabled=True, **kw) -> tuple[updater.UpdateController, Preferences]:
    prefs = _prefs(tmp_path)
    return updater.UpdateController(prefs, enabled=enabled, folder=tmp_path / "updates", current="2.0.3", key=WIN,
                                    public_key=_public(key), **kw), prefs


def _ready(ctl: updater.UpdateController, key: Ed25519PrivateKey, version: str = "2.0.15") -> Path:
    """A downloaded update waiting in the controller's folder (the installer, update.json and its signature, and a file
    of someone else's). Returns the folder."""
    data, installer = _manifest(version)
    folder = ctl.folder
    folder.mkdir(parents=True, exist_ok=True)
    (folder / f"AudioCleaner-{version}-Windows.exe").write_bytes(installer)
    (folder / "update.json").write_bytes(data)
    (folder / "update.json.sig").write_bytes(key.sign(data))
    (folder / "notes.txt").write_text("not ours")
    ctl.ready_update = updater.decide(data, key.sign(data), "2.0.3", WIN, _public(key))
    ctl.ready_path = folder / f"AudioCleaner-{version}-Windows.exe"
    return folder


def test_turning_updates_off_forgets_and_deletes_a_downloaded_update(qapp, tmp_path, key, monkeypatch):
    ctl, prefs = _offline_controller(tmp_path, key)
    folder = _ready(ctl, key)
    ready = []
    ctl.readyChanged.connect(lambda: ready.append(ctl.ready_version))
    ctl.relaunch = True  # "Restart now" had been chosen
    prefs.check_updates = False
    ctl.setting_changed()
    assert ctl.ready_version is None and ctl.ready_path is None and not ctl.relaunch
    assert ready == [None]  # the window hears of it once
    assert [p.name for p in folder.iterdir()] == ["notes.txt"]  # no installer, no update.json, no signature
    started = []
    monkeypatch.setattr(updater, "start_windows_installer", lambda *args: started.append(args))
    with monkeypatch.context() as m:
        m.setattr(updater.sys, "platform", "win32")
        assert not ctl.install_on_close()
    assert started == []


def test_turning_updates_on_keeps_a_downloaded_update(qapp, tmp_path, key):
    ctl, prefs = _offline_controller(tmp_path, key)
    folder = _ready(ctl, key)
    ready = []
    ctl.readyChanged.connect(lambda: ready.append(1))
    ctl.setting_changed()  # the setting is on
    assert ctl.ready_version == "2.0.15" and ready == []
    assert (folder / "AudioCleaner-2.0.15-Windows.exe").is_file() and (folder / "update.json").is_file()


def test_turning_updates_off_while_a_check_runs_forgets_the_update_at_once_and_deletes_it_when_the_check_ends(
        qapp, tmp_path, server, key, monkeypatch):
    _publish(server, key)
    ctl, prefs = _controller(qapp, tmp_path, server, key)
    folder = _ready(ctl, key)
    real, reached, go = updater.check, threading.Event(), threading.Event()

    def held(*args, **kwargs):  # the worker waits here (the check is under way) until the test lets it go
        reached.set()
        go.wait(30)
        return real(*args, **kwargs)

    monkeypatch.setattr(updater, "check", held)
    ready = []
    ctl.readyChanged.connect(lambda: ready.append(ctl.ready_version))
    ctl.relaunch = True
    ctl.check_now()
    assert wait_until(reached.is_set, timeout=30) and ctl.running
    prefs.check_updates = False
    ctl.setting_changed()
    assert ctl.ready_version is None and ctl.ready_path is None and not ctl.relaunch and ready == [None]  # at once
    assert (folder / "AudioCleaner-2.0.15-Windows.exe").is_file()  # the files go when the worker has stopped
    go.set()
    assert wait_until(lambda: not ctl.busy, timeout=30)
    assert ctl.ready_version is None and ready == [None]  # (nothing is announced twice, nothing comes back)
    assert [p.name for p in folder.iterdir()] == ["notes.txt"]


def _leftover_download(tmp_path, key, monkeypatch, enabled=True):
    """A controller of a freshly started app that finds a download in its folder (the last run quit before deleting it)."""
    ctl, prefs = _offline_controller(tmp_path, key, enabled=enabled)
    folder = _ready(ctl, key)
    ctl.ready_update = ctl.ready_path = None  # nothing is remembered, only the files are there
    timers = []
    monkeypatch.setattr(updater, "QTimer", types.SimpleNamespace(singleShot=lambda ms, fn: timers.append((ms, fn))))
    return ctl, prefs, folder, timers


def test_the_startup_check_with_updates_off_deletes_a_leftover_download_and_starts_no_check(qapp, tmp_path, key, monkeypatch):
    ctl, prefs, folder, timers = _leftover_download(tmp_path, key, monkeypatch)
    prefs.check_updates = False
    ctl.schedule_startup_check()
    assert [p.name for p in folder.iterdir()] == ["notes.txt"]  # no installer, no update.json, no signature
    assert timers == [] and not ctl.busy and not ctl.running


def test_the_startup_check_with_updates_on_is_scheduled_and_keeps_a_downloaded_update(qapp, tmp_path, key, monkeypatch):
    ctl, prefs, folder, timers = _leftover_download(tmp_path, key, monkeypatch)
    ctl.schedule_startup_check()
    assert timers == [(updater.STARTUP_DELAY_MS, ctl.startup_check)]
    assert (folder / "AudioCleaner-2.0.15-Windows.exe").is_file() and (folder / "update.json").is_file()


def test_the_startup_check_of_a_copy_that_is_not_installed_does_nothing_even_with_updates_off(qapp, tmp_path, key, monkeypatch):
    ctl, prefs, folder, timers = _leftover_download(tmp_path, key, monkeypatch, enabled=False)
    prefs.check_updates = False
    ctl.schedule_startup_check()
    assert timers == [] and (folder / "AudioCleaner-2.0.15-Windows.exe").is_file() and (folder / "update.json").is_file()


def test_an_update_that_finishes_downloading_after_updates_were_turned_off_is_not_kept(qapp, tmp_path, server, key):
    _publish(server, key)
    ctl, prefs = _controller(qapp, tmp_path, server, key)
    ready = []
    ctl.readyChanged.connect(lambda: ready.append(ctl.ready_version))
    ctl.check_now()
    ctl.wait(30)  # the download is done and its result waits for the GUI thread (nothing has processed events)
    assert not ctl.running
    prefs.check_updates = False  # turned off in the meantime
    assert wait_until(lambda: not ctl.busy, timeout=30)
    assert ctl.ready_version is None and ready == [] and ctl.status == ""
    assert not list((tmp_path / "updates").glob("AudioCleaner-*")) and not (tmp_path / "updates" / "update.json").exists()


def test_check_now_with_updates_off_only_says_that_a_newer_version_exists(qapp, tmp_path, server, key):
    _publish(server, key)
    ctl, prefs = _controller(qapp, tmp_path, server, key)
    prefs.check_updates = False
    ctl.check_now()
    assert wait_until(lambda: not ctl.busy and ctl.status.startswith("Version"), timeout=30)
    assert ctl.status == "Version 2.0.15 is available. Updates are off, so nothing was downloaded."
    assert [r.split()[0] for r in _Handler.requests] == ["/desktop/update.json", "/desktop/update.json.sig"]  # no installer asked for
    assert not list((tmp_path / "updates").glob("AudioCleaner-*")) and ctl.ready_version is None


def test_a_cancelled_download_ends_the_check_so_the_controller_is_not_busy_any_more(qapp, tmp_path, server, key, monkeypatch):
    _publish(server, key)
    ctl, _ = _controller(qapp, tmp_path, server, key)
    real, reached, go = updater.download, threading.Event(), threading.Event()

    def held(*args, **kwargs):  # the worker waits here until the test has cancelled
        reached.set()
        go.wait(30)
        return real(*args, **kwargs)

    monkeypatch.setattr(updater, "download", held)
    ctl.check_now()
    assert wait_until(reached.is_set, timeout=30) and ctl.busy
    ctl.cancel()
    go.set()
    assert wait_until(lambda: not ctl.busy, timeout=30)
    assert ctl.ready_version is None and ctl.status == "" and not (tmp_path / "updates" / "update.json").exists()
    ctl.check_now()  # and a new check can start
    assert ctl.busy
    ctl.wait(30)
    assert wait_until(lambda: not ctl.busy, timeout=30) and ctl.ready_version == "2.0.15"


def test_the_installer_is_not_opened_when_updates_are_off(qapp, tmp_path, key, monkeypatch):
    ctl, prefs = _offline_controller(tmp_path, key)
    _ready(ctl, key)
    opened = []
    monkeypatch.setattr(updater, "open_mac_installer", opened.append)
    assert ctl.open_installer() and len(opened) == 1  # (the setting is on)
    prefs.check_updates = False
    assert not ctl.open_installer() and len(opened) == 1


# ---- the window ------------------------------------------------------------------------------------------------
def test_closing_starts_the_installer_once_per_version(qapp, tmp_path, monkeypatch):
    started, handed, opened = [], [], []

    def locked_installer(path, entry):  # what Windows' locked_installer does (size + SHA-256 through the open file)
        f = open(path, "rb")
        data = f.read()
        if len(data) == entry.size and hashlib.sha256(data).hexdigest() == entry.sha256:
            f.seek(0)
            opened.append(f)
            return f
        f.close()
        return None

    def start_windows_installer(path, relaunch, locked=None):
        started.append((path.name, relaunch))
        handed.append(locked)

    monkeypatch.setattr(updater, "locked_installer", locked_installer)
    monkeypatch.setattr(updater, "start_windows_installer", start_windows_installer)
    prefs = _prefs(tmp_path)
    ctl = updater.UpdateController(prefs, enabled=True)
    entry_bytes = b"setup" * 50
    path = tmp_path / "AudioCleaner-2.0.15-Windows.exe"
    path.write_bytes(entry_bytes)
    ctl.ready_update = updater.Update("2.0.15", updater.UpdateFile(
        path.name, len(entry_bytes), hashlib.sha256(entry_bytes).hexdigest()), b"{}", b"")
    ctl.ready_path = path

    monkeypatch.setattr(updater.sys, "platform", "linux")
    assert not ctl.install_on_close()  # not Windows: nothing starts
    monkeypatch.setattr(updater.sys, "platform", "win32")
    try:
        assert ctl.install_on_close() and started == [(path.name, False)]
        assert handed == opened and opened[0].closed  # setup got the locked file, and this app let go of it afterwards
        assert prefs.update_install_tried == "2.0.15"
        assert not ctl.install_on_close() and len(started) == 1  # a cancelled setup is not started at every close
        ctl.relaunch = True  # "Restart now" is a clear request
        assert ctl.install_on_close() and started[-1] == (path.name, True)
        assert handed == opened and len(opened) == 2 and opened[1].closed
        prefs.check_updates = False
        assert not ctl.install_on_close()  # switched off since: nothing starts
        prefs.check_updates = True
        path.write_bytes(entry_bytes[:-1] + b"X")  # the file changed since it was verified
        assert not ctl.install_on_close() and len(started) == 2 and len(opened) == 2
    finally:
        monkeypatch.undo()


def test_the_window_drops_its_banner_and_a_waiting_restart_when_updates_are_turned_off(qapp, tmp_path, key, monkeypatch):
    import studio_helpers as sh
    from audio_cleaner_desktop.gui.main_window import MainWindow

    settings = QSettings(str(tmp_path / "s.ini"), QSettings.Format.IniFormat)
    prefs = Preferences(settings)
    ctl = updater.UpdateController(prefs, enabled=False, folder=tmp_path / "updates", current="2.0.3", key=WIN,
                                   public_key=_public(key))
    _ready(ctl, key)
    w = MainWindow([], settings=settings, history_path=tmp_path / "h.db", updater=ctl)
    closed = []
    monkeypatch.setattr(w, "close", lambda: closed.append(1))
    try:
        w.show()
        ctl.readyChanged.emit()
        assert w.banner_update.isVisible()
        w._restart_when_idle = True  # "Restart now" was pressed while files were being cleaned
        prefs.check_updates = False
        ctl.setting_changed()
        assert not w._restart_when_idle and not w.banner_update.isVisible()
        w._restart_for_update()  # e.g. the queue went idle
        assert closed == [] and not ctl.relaunch
    finally:
        sh.shut(w)


# ---- Windows setup: the command line and the script -------------------------------------------------------------------
def test_the_silent_windows_command_line():
    setup = r"C:\Users\Zoe\AppData\Local\Audio Cleaner\Audio Cleaner\updates\AudioCleaner-2.0.15-Windows.exe"
    env = {"LOCALAPPDATA": r"C:\Users\Zoe\AppData\Local", "ProgramFiles": r"C:\Program Files",
           "ProgramFiles(x86)": r"C:\Program Files (x86)"}
    base = [setup, "/VERYSILENT", "/SUPPRESSMSGBOXES", "/NORESTART"]

    def cmd(exe, relaunch=False):
        return updater.windows_installer_command(setup, exe, relaunch, env)

    # the mode of the running install is kept (Windows paths, compared the way Windows does)
    assert cmd(r"C:\Users\Zoe\AppData\Local\Programs\Audio Cleaner\AudioCleaner.exe") == [*base, "/CURRENTUSER"]
    assert cmd(r"c:\users\zoe\appdata\local\programs\audio cleaner\audiocleaner.exe") == [*base, "/CURRENTUSER"]
    assert cmd(r"C:\Program Files\Audio Cleaner\AudioCleaner.exe") == [*base, "/ALLUSERS"]
    assert cmd(r"C:\Program Files (x86)\Audio Cleaner\AudioCleaner.exe") == [*base, "/ALLUSERS"]
    # another folder: no switch, Setup reuses the previous install's mode itself (UsePreviousPrivileges)
    assert cmd(r"D:\Apps\Audio Cleaner\AudioCleaner.exe") == base
    assert cmd(r"C:\Program Files Evil\AudioCleaner.exe") == base  # a folder that only starts with the same letters
    assert cmd(r"C:\Program Files\Audio Cleaner\AudioCleaner.exe", True) == [*base, "/ALLUSERS", "/RELAUNCH=1"]
    assert updater.windows_installer_command(setup, r"D:\a.exe", False, {}) == base  # no environment, no switch
    # nothing in it asks for a visible setup, a restart of the computer, or a dialog
    assert {"/VERYSILENT", "/SUPPRESSMSGBOXES", "/NORESTART"} <= set(base) and "/SILENT" not in base


def test_the_installer_is_started_detached_without_a_console_and_the_mac_one_with_open(monkeypatch, tmp_path):
    calls = []
    monkeypatch.setattr(updater.subprocess, "Popen", lambda cmd, **kw: calls.append((cmd, kw)))
    updater.start_windows_installer(tmp_path / "AudioCleaner-2.0.15-Windows.exe", True)
    cmd, kw = calls[0]
    assert cmd[0].endswith("AudioCleaner-2.0.15-Windows.exe") and cmd[-1] == "/RELAUNCH=1" and "/VERYSILENT" in cmd
    assert kw["creationflags"] & updater.DETACHED_PROCESS and kw["creationflags"] & updater.CREATE_NEW_PROCESS_GROUP
    assert kw["close_fds"] is True and kw["stdin"] == subprocess.DEVNULL and kw["stdout"] == subprocess.DEVNULL
    opened = []
    monkeypatch.setattr(updater.subprocess, "Popen", lambda cmd, **kw: opened.append(cmd))
    updater.open_mac_installer(tmp_path / "AudioCleaner-2.0.15-Mac-Intel.pkg")
    assert opened == [["open", str(tmp_path / "AudioCleaner-2.0.15-Mac-Intel.pkg")]]


# ---- the lock on the installer (Windows only: it is the Windows file sharing rules that make it work) ---------------------
windows_only = pytest.mark.skipif(sys.platform != "win32", reason="file locking by sharing mode is a Windows feature")


def _installer_copy(tmp_path) -> tuple[Path, updater.UpdateFile]:
    """A real program that exits by itself (ping) under the name of an installer, and its manifest entry."""
    path = tmp_path / "AudioCleaner-2.0.15-Windows.exe"
    shutil.copyfile(Path(os.environ.get("SystemRoot", r"C:\Windows")) / "System32" / "PING.EXE", path)
    data = path.read_bytes()
    return path, updater.UpdateFile(path.name, len(data), hashlib.sha256(data).hexdigest())


@windows_only
def test_a_locked_installer_cannot_be_written_replaced_renamed_or_deleted_but_can_be_read(tmp_path):
    path, entry = _installer_copy(tmp_path)
    other = tmp_path / "other.exe"
    other.write_bytes(b"something else")
    locked = updater.locked_installer(path, entry)
    assert locked is not None
    try:
        with pytest.raises(OSError):
            open(path, "r+b")
        with pytest.raises(OSError):
            os.replace(other, path)
        with pytest.raises(OSError):
            os.rename(path, tmp_path / "moved.exe")
        with pytest.raises(OSError):
            path.unlink()
        with open(path, "rb") as reader:  # reading (and so running) it is still possible
            assert reader.read(2) == b"MZ"
    finally:
        locked.close()
    os.rename(path, tmp_path / "moved.exe")  # let go: it is an ordinary file again


@windows_only
@pytest.mark.parametrize("change", ["longer", "other bytes"])
def test_an_installer_that_does_not_match_is_not_locked_and_no_handle_is_left_open(tmp_path, change):
    path, entry = _installer_copy(tmp_path)
    data = path.read_bytes()
    path.write_bytes(data + b"x" if change == "longer" else data[:-1] + bytes([data[-1] ^ 1]))
    assert updater.locked_installer(path, entry) is None
    os.rename(path, tmp_path / "moved.exe")  # nothing holds it
    assert updater.locked_installer(tmp_path / "missing.exe", entry) is None


@windows_only
def test_setup_started_with_the_locked_file_keeps_it_locked_after_the_app_let_go(tmp_path, monkeypatch):
    path, entry = _installer_copy(tmp_path)
    # the real start (detached, handle handed down), with a program that runs for a few seconds in place of setup
    monkeypatch.setattr(updater, "windows_installer_command", lambda installer, exe, relaunch, env=None: [str(path), "-n", "4", "127.0.0.1"])
    locked = updater.locked_installer(path, entry)
    assert locked is not None
    try:
        updater.start_windows_installer(path, False, locked)
    finally:
        locked.close()  # the app's own handle is gone; the running program still holds the one it was given
    moved = tmp_path / "moved.exe"
    with pytest.raises(OSError):
        os.rename(path, moved)
    for _ in range(60):  # it ends after about three seconds; then the file is free again
        try:
            os.rename(path, moved)
            break
        except OSError:
            time.sleep(0.25)
    assert moved.is_file() and not path.exists()


def _junction_to(real_dir: Path, junction: Path) -> None:
    import _winapi

    real_dir.mkdir(exist_ok=True)
    _winapi.CreateJunction(str(real_dir), str(junction))


@windows_only
def test_final_path_names_the_real_file_not_the_junction_it_was_opened_through(tmp_path):
    real_dir, junction = tmp_path / "the-real-folder", tmp_path / "via-junction"
    _junction_to(real_dir, junction)
    try:
        (real_dir / "Setup.exe").write_bytes(b"x")
        with updater.open_locked(junction / "Setup.exe") as f:
            got = updater.final_path(f)
        assert not got.startswith("\\\\?\\")  # an ordinary path, without the long-path prefix
        assert got.lower().endswith("\\the-real-folder\\setup.exe") and "via-junction" not in got
        assert os.path.samefile(got, real_dir / "Setup.exe")
    finally:
        os.rmdir(junction)  # the link only, not the folder it points to


@windows_only
def test_setup_is_started_by_the_real_path_of_the_locked_file_not_the_junction_it_was_found_through(tmp_path, monkeypatch):
    real_dir, junction = tmp_path / "the-real-folder", tmp_path / "via-junction"
    _junction_to(real_dir, junction)
    commands, started = [], []
    real_popen = subprocess.Popen

    def popen(cmd, **kwargs):  # the real start, only with a short program's arguments in place of setup's
        commands.append(cmd)
        started.append(real_popen([cmd[0], "-n", "2", "127.0.0.1"], **kwargs))

    monkeypatch.setattr(updater.subprocess, "Popen", popen)
    try:
        path, entry = _installer_copy(real_dir)
        locked = updater.locked_installer(junction / path.name, entry)  # found through the junction
        assert locked is not None
        try:
            updater.start_windows_installer(junction / path.name, False, locked)
        finally:
            locked.close()
        assert os.path.samefile(commands[0][0], path) and "via-junction" not in commands[0][0]
        assert "/VERYSILENT" in commands[0]
        started[0].wait(timeout=30)  # and that file did run: it started and ended
    finally:
        for process in started:
            if process.poll() is None:
                process.kill()
        os.rmdir(junction)


ISS = (PACKAGING / "windows" / "AudioCleaner.iss").read_text(encoding="utf-8")


def test_the_inno_script_relaunches_only_for_a_silent_update_that_asks_for_it():
    run = ISS.split("[Run]")[1].split("[Code]")[0]
    entries = [ln for ln in run.splitlines() if ln.startswith("Filename:")]
    assert len(entries) == 2
    wizard, relaunch = entries
    assert "postinstall skipifsilent" in wizard and "Check:" not in wizard  # the normal "Launch" tick box is unchanged
    assert "nowait" in relaunch and "runasoriginaluser" in relaunch  # not waiting, not elevated
    assert "postinstall" not in relaunch and "Check: RelaunchAfterSilentUpdate" in relaunch
    code = ISS.split("[Code]")[1]
    assert "Result := WizardSilent and (ExpandConstant('{param:RELAUNCH|0}') = '1');" in code
    assert ISS.index("function RelaunchAfterSilentUpdate") < ISS.index("procedure CurUninstallStepChanged")
    # what the updater passes is what the script reads, and the settings an update relies on are still there
    assert "/RELAUNCH=1" in updater.windows_installer_command(Path("s.exe"), Path("a.exe"), True, {})
    for line in ("UsePreviousAppDir=yes", "UsePreviousPrivileges=yes", "PrivilegesRequiredOverridesAllowed=dialog",
                 "CloseApplications=yes"):
        assert line in ISS


# ---- the release job and the manifest script -------------------------------------------------------------------------
def _openssl_3() -> str | None:
    """An OpenSSL 3 (the release job signs with Ubuntu's: `pkeyutl -rawin`, Ed25519 over the raw file). macOS's own
    openssl is LibreSSL, which has neither; Homebrew's openssl@3 is used when it is there."""
    for candidate in (shutil.which("openssl"), "/opt/homebrew/opt/openssl@3/bin/openssl",
                      "/usr/local/opt/openssl@3/bin/openssl"):
        if not candidate or not Path(candidate).is_file():
            continue
        try:
            version = subprocess.run([candidate, "version"], capture_output=True, text=True, timeout=30).stdout
        except (OSError, subprocess.SubprocessError):
            continue
        match = re.match(r"OpenSSL (\d+)\.", version)
        if match and int(match.group(1)) >= 3:
            return candidate
    return None


OPENSSL = _openssl_3()
openssl = pytest.mark.skipif(OPENSSL is None, reason="needs OpenSSL 3 (pkeyutl -rawin, as on the release job's Ubuntu); "
                                                     "the openssl found is missing or older, or LibreSSL as on macOS")


def test_the_manifest_has_sorted_keys_and_the_sizes_and_hashes_of_the_installers(tmp_path):
    contents = {}
    for key, pattern in make_update_manifest.INSTALLERS.items():
        name = pattern.format(version="2.0.15")
        contents[key] = f"installer {key}".encode()
        (tmp_path / name).write_bytes(contents[key])
    data = make_update_manifest.build_manifest("2.0.15", tmp_path, published="2026-10-16T12:00:00Z")
    body = json.loads(data)
    assert list(body) == sorted(body) == ["files", "published", "version"]
    assert body["version"] == "2.0.15" and body["published"] == "2026-10-16T12:00:00Z"
    assert "notes_url" not in body and "github" not in data.decode().lower()
    assert sorted(body["files"]) == ["macos-arm64", "macos-x86_64", "windows-x86_64"]
    for key, entry in body["files"].items():
        assert list(entry) == ["name", "sha256", "size"]
        assert entry["size"] == len(contents[key]) and entry["sha256"] == hashlib.sha256(contents[key]).hexdigest()
    assert body["files"]["windows-x86_64"]["name"] == "AudioCleaner-2.0.15-Windows.exe"
    assert data == (json.dumps(body, indent=2, sort_keys=True) + "\n").encode()
    (tmp_path / "AudioCleaner-2.0.15-Mac-Intel.pkg").unlink()
    with pytest.raises(SystemExit, match="is missing"):
        make_update_manifest.build_manifest("2.0.15", tmp_path)
    with pytest.raises(SystemExit):
        make_update_manifest.build_manifest("latest", tmp_path)


@openssl
def test_a_manifest_signed_the_way_ci_does_it_verifies_in_the_app(tmp_path):
    """The CI command (openssl pkeyutl -sign -rawin) produces what the app's check accepts, and a manifest the app
    builds its trust on only with the matching key."""
    pem = tmp_path / "k.pem"
    subprocess.run([OPENSSL, "genpkey", "-algorithm", "ed25519", "-out", str(pem)], check=True, capture_output=True)
    pub = make_update_manifest.public_key_of(pem)
    for key, pattern in make_update_manifest.INSTALLERS.items():
        (tmp_path / pattern.format(version="2.0.15")).write_bytes(b"x" + key.encode())
    manifest = tmp_path / "update.json"
    assert make_update_manifest.main(["manifest", "--version", "2.0.15", "--dir", str(tmp_path)]) == 0
    sig = tmp_path / "update.json.sig"
    subprocess.run([OPENSSL, "pkeyutl", "-sign", "-rawin", "-inkey", str(pem), "-in", str(manifest), "-out", str(sig)],
                   check=True, capture_output=True)
    assert sig.stat().st_size == 64
    update = updater.decide(manifest.read_bytes(), sig.read_bytes(), "2.0.3", WIN, pub)
    assert update.version == "2.0.15" and update.file.name == "AudioCleaner-2.0.15-Windows.exe"
    with pytest.raises(updater.UpdateError):
        updater.decide(manifest.read_bytes() + b"\n", sig.read_bytes(), "2.0.3", WIN, pub)
    # a throwaway key is not the key the app trusts, so the release job would publish no manifest for it
    assert make_update_manifest.main(["check-key", str(pem)]) == 1


_WORKFLOW_FILE = ROOT / ".github" / "workflows" / "desktop.yml"
WORKFLOW = _WORKFLOW_FILE.read_text(encoding="utf-8") if _WORKFLOW_FILE.is_file() else ""


_has_release = pytest.mark.skipif("\n  release:\n" not in WORKFLOW, reason="this copy of the repository has no release job")


def _jobs() -> dict[str, str]:
    """{job name: its text} of the workflow (a job starts at its two-space `name:` line under `jobs:`)."""
    parts = re.split(r"^  ([a-z][\w-]*):\n", WORKFLOW.split("\njobs:\n", 1)[1], flags=re.M)
    return dict(zip(parts[1::2], parts[2::2]))


def _jobs_yaml() -> dict:
    return pytest.importorskip("yaml").safe_load(WORKFLOW)["jobs"]


def _checkouts(job: dict) -> list[dict]:
    return [step.get("with", {}) for step in job["steps"] if str(step.get("uses", "")).startswith("actions/checkout@")]


def _needs(job: dict) -> list[str]:
    return [job["needs"]] if isinstance(job["needs"], str) else list(job["needs"])


@_has_release
def test_the_release_is_three_jobs_and_no_repository_code_runs_next_to_a_secret():
    jobs, structure = _jobs(), _jobs_yaml()
    update_info, sign, release = jobs["update-info"], jobs["sign"], jobs["release"]
    assert _needs(structure["sign"]) == ["update-info"] and sorted(_needs(structure["release"])) == ["sign", "update-info"]
    # the key only in sign, the tokens only in release, no secret at all in the job that runs the repository's scripts
    assert [name for name, text in jobs.items() if "UPDATE_SIGNING_KEY" in text] == ["sign"]
    assert [name for name, text in jobs.items() if "secrets.PUBLIC_RELEASE_TOKEN" in text] == ["release"]
    assert [name for name, text in jobs.items() if "github.token" in text] == ["release"]
    assert "UPDATE_SIGNING_KEY: ${{ secrets.UPDATE_SIGNING_KEY }}" in sign
    assert re.findall(r"secrets\.(\w+)", sign) == ["UPDATE_SIGNING_KEY"]
    assert "secrets." not in update_info
    assert structure["update-info"]["permissions"] == {"contents": "read"}
    assert structure["sign"]["permissions"] == {"contents": "read"}
    assert "version.py" in update_info and "make_update_manifest.py" in update_info  # the scripts run here, with no secret
    # nothing from the repository runs in sign or release: no Python, and release has no checkout at all
    assert "python" not in sign.lower() and "python" not in release.lower()
    assert "actions/checkout" not in release
    # the jobs that do check out read nothing but what they need, and keep no token in .git
    assert [c["persist-credentials"] for c in _checkouts(structure["update-info"])] == [False]
    assert [c["sparse-checkout"].split() for c in _checkouts(structure["update-info"])] == [
        ["desktop/audio_cleaner_desktop/*.py", "desktop/packaging/version.py", "desktop/packaging/make_update_manifest.py"]]
    only = _checkouts(structure["sign"])  # only updater.py, read as text for the public key
    assert len(only) == 1 and only[0]["persist-credentials"] is False and only[0]["sparse-checkout-cone-mode"] is False
    assert only[0]["sparse-checkout"].split() == ["desktop/audio_cleaner_desktop/updater.py"]
    # release takes its version from update-info and checks that it is a version number (also in the public repository step)
    for step in (_step(release, "Create the release"), _step(release, "Publish the release to the public repository")):
        assert "VERSION: ${{ needs.update-info.outputs.version }}" in step
        assert '[[ "$VERSION" =~ ^[0-9]+(\\.[0-9]+){1,3}$ ]]' in step and 'version="$VERSION"' in step


@_has_release
def test_the_release_job_signs_the_manifest_and_fails_closed():
    jobs = _jobs()
    sign, release = jobs["sign"], jobs["release"]
    assert "openssl pkeyutl -sign -rawin" in sign and "openssl pkeyutl -verify -rawin" in sign
    assert 'if [ -z "$UPDATE_SIGNING_KEY" ]' in sign and "::warning::" in sign  # no secret: no manifest, a warning
    # a secret that is not the app's key publishes no manifest either: its public half is compared with PUBLIC_KEY
    assert 'openssl pkey -in "$key" -pubout -outform DER' in sign and '!= "$trusted"' in sign and "::error::" in sign
    assert "PUBLIC_KEY" in sign and "check-key" not in sign  # (no repository script reads the key)
    # the key file is removed whatever happens, and it is never printed
    trap = re.search(r"""trap 'rm -f "\$key"[^']*' EXIT""", sign)
    assert trap and trap.start() < sign.index('printf \'%s\\n\' "$UPDATE_SIGNING_KEY" > "$key"')
    assert "echo \"$UPDATE_SIGNING_KEY\"" not in sign
    # signed=true only after the signature was verified, and the signed files are what the next job downloads
    assert sign.count("signed=true") == 1
    assert sign.index("openssl pkeyutl -verify") < sign.index('echo "signed=true" >> "$GITHUB_OUTPUT"')
    assert "signed: ${{ steps.sign.outputs.signed }}" in sign and "if: steps.sign.outputs.signed == 'true'" in sign
    # release publishes them in the same release as the installers, only when signing happened
    assert "if: needs.sign.outputs.signed == 'true'" in release and "name: update-information" in release
    add = "[ ! -f update.json ] || files+=(update.json update.json.sig)"
    assert release.index("files=(") < release.index(add) < release.index("gh release create")
    assert WORKFLOW.count("--require-hashes") >= 2 and "--require-hashes -r desktop/requirements.lock" in WORKFLOW


def _sed_pattern() -> str:
    """The sed command of the sign job that reads the key the app trusts out of updater.py, as a Python regex."""
    script = next(step["run"] for step in _jobs_yaml()["sign"]["steps"] if step.get("id") == "sign")
    bre = re.search(r"sed -n 's/(.+?)/\\1/p' desktop/audio_cleaner_desktop/updater\.py", script).group(1)
    return re.sub(r"\\\{(\d+)\\\}", r"{\1}", bre.replace(r"\(", "(").replace(r"\)", ")"))


@_has_release
def test_the_sign_job_reads_exactly_the_public_key_the_app_trusts():
    rx = re.compile(_sed_pattern())
    source = (ROOT / "desktop" / "audio_cleaner_desktop" / "updater.py").read_text(encoding="utf-8").splitlines()
    assert [m.group(1) for m in map(rx.search, source) if m] == [updater.PUBLIC_KEY]
    assert not rx.search(f'    PUBLIC_KEY = "{updater.PUBLIC_KEY}"') and not rx.search(f'PUBLIC_KEY = "{updater.PUBLIC_KEY[:-1]}"')


def _run_sign_step(tmp_path, public: str, secret: str) -> tuple[subprocess.CompletedProcess, Path, Path]:
    """The sign job's own script, run as the runner runs it, in a checkout that has only updater.py and the downloaded
    update.json. Returns the process, the GITHUB_OUTPUT file and the folder with update.json."""
    script = next(step["run"] for step in _jobs_yaml()["sign"]["steps"] if step.get("id") == "sign")
    (tmp_path / "sign.sh").write_text(script, encoding="utf-8")
    work, runner, output = tmp_path / "work", tmp_path / "runner", tmp_path / "output"
    (work / "desktop" / "audio_cleaner_desktop").mkdir(parents=True)
    (work / "info").mkdir()
    runner.mkdir()
    output.touch()
    (work / "desktop" / "audio_cleaner_desktop" / "updater.py").write_text(f'DOWNLOAD = "x"\nPUBLIC_KEY = "{public}"\n')
    (work / "info" / "update.json").write_bytes(_manifest()[0])
    env = {**os.environ, "RUNNER_TEMP": str(runner), "GITHUB_OUTPUT": str(output), "UPDATE_SIGNING_KEY": secret}
    done = subprocess.run(["bash", "--noprofile", "--norc", "-eo", "pipefail", str(tmp_path / "sign.sh")], cwd=work, env=env,
                          capture_output=True, text=True, timeout=120, check=False)
    return done, output, work / "info"


def _pem(key: Ed25519PrivateKey) -> str:
    return key.private_bytes(serialization.Encoding.PEM, serialization.PrivateFormat.PKCS8, serialization.NoEncryption()).decode()


linux_bash = pytest.mark.skipif(sys.platform != "linux" or shutil.which("bash") is None,
                                reason="the sign job runs on Ubuntu (bash, GNU sed, stat and base64)")


@_has_release
@openssl
@linux_bash
def test_the_sign_job_signs_with_the_trusted_key_and_leaves_no_key_behind(tmp_path, key):
    done, output, info = _run_sign_step(tmp_path, _public(key), _pem(key))
    assert done.returncode == 0, done.stderr
    assert output.read_text() == "signed=true\n"
    update = updater.decide((info / "update.json").read_bytes(), (info / "update.json.sig").read_bytes(), "2.0.3", WIN, _public(key))
    assert update.version == "2.0.15"  # the signature is what the app accepts
    assert list((tmp_path / "runner").iterdir()) == []  # the key file (and the public halves) are gone


@_has_release
@openssl
@linux_bash
@pytest.mark.parametrize("secret", ["empty", "other key", "not a key"])
def test_the_sign_job_publishes_nothing_without_the_trusted_key(tmp_path, key, secret):
    value, marker = {"empty": ("", "::warning::"), "other key": (_pem(Ed25519PrivateKey.generate()), "::error::"),
                     "not a key": ("not a key", "::error::")}[secret]
    done, output, info = _run_sign_step(tmp_path, _public(key), value)
    assert done.returncode == 0 and marker in done.stdout  # (the installers are still published)
    assert output.read_text() == "" and not (info / "update.json.sig").exists()  # no signed=true: release publishes no update.json
    assert list((tmp_path / "runner").iterdir()) == []


def test_cryptography_is_pinned_with_hashes_for_every_platform():
    lock = (ROOT / "desktop" / "requirements.lock").read_text(encoding="utf-8")
    blocks = {m[1]: m[0] for m in re.findall(r"^((\S+)==\S+[^\n]*\\\n(?:    --hash=\S+(?: \\)?\n?)+)", lock, re.M)}
    for name in ("cryptography", "cffi", "pycparser"):
        assert name in blocks, f"{name} is not pinned in requirements.lock"
        assert blocks[name].count("--hash=sha256:") >= 2
    # one wheel per platform at least: manylinux/musllinux, Windows, macOS universal2 (Intel and Apple silicon)
    assert blocks["cryptography"].count("--hash=sha256:") >= 20


def _public_copy_leak_patterns() -> dict[str, str]:
    """{why: pattern} of the public copy's leak scan (in this repository wherever the release job is)."""
    import importlib.util

    spec = importlib.util.spec_from_file_location("public_desktop_sync", ROOT / "tools" / "public-desktop" / "sync.py")
    sync = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(sync)
    return {why: pattern for pattern, why in sync.LEAK_PATTERNS}


def _step(job: str, name: str) -> str:
    body = job.split(f"- name: {name}")[1]
    return body.split("\n      - name:")[0].split("\n      - uses:")[0]


@pytest.mark.skipif("\n  release:\n" not in WORKFLOW, reason="this copy of the repository publishes no releases")
def test_the_release_is_published_to_the_public_repository_with_the_installers_and_the_update_information():
    release = WORKFLOW.split("\n  release:\n")[1]
    name = "Publish the release to the public repository"
    assert f"- name: {name}" in release and "Copy the release to the public repository" not in WORKFLOW
    step = _step(release, name)
    assert "GH_TOKEN: ${{ secrets.PUBLIC_RELEASE_TOKEN }}" in step
    assert "continue-on-error: true" in step and "::warning::" in step  # never fails the official release
    assert 'if [ -z "$GH_TOKEN" ]' in step and "exit 0" in step  # no secret: skipped
    assert 'tag="v$version"' in step and '--target "$branch"' in step and "--clobber" in step
    # no model is in the release (the app and setup download them from Hugging Face)
    assert "release_models" not in step and "models/" not in step and "whisper" not in step.lower()
    # everything the app downloads from here: the installers and the signed update information (when this build has it)
    files = re.search(r"files=\((.*?)\)", step).group(1).split()
    assert files == ['"AudioCleaner-$version-Windows.exe"', '"AudioCleaner-$version-Mac-Apple-silicon.pkg"',
                     '"AudioCleaner-$version-Mac-Intel.pkg"']
    assert "[ ! -f update.json ] || files+=(update.json update.json.sig)" in step
    # made as a draft, filled, then published and marked latest: no app and no person ever sees half a release
    create = step.index("gh release create")
    upload = step.index('gh release upload "$tag"')
    publish = step.index('gh release edit "$tag"')
    assert create < upload < publish
    assert "--draft " in step[create:upload] and "--latest" not in step[create:upload]
    assert "--draft=false" in step[publish:] and "--latest" in step[publish:]
    assert 'if ! gh release view "$tag"' in step  # a rerun reuses the release (its files are replaced)
    # releases there are never removed: every version keeps its installers (an app updating from an older one downloads it)
    assert "gh release delete" not in step and "automatic build" not in step
    # The notes are written for the community: which file is for which computer, that setup downloads the models once
    # and that the app keeps itself up to date.
    notes = re.search(r'notes="(Audio Cleaner \$version\..*?[^\\])"\n', step, re.S).group(1)
    picks = dict(re.findall(r"^\s*- (.+?): (AudioCleaner-\$version-\S+)$", notes, re.M))
    assert set(picks.values()) == {f.strip('"') for f in files}  # every installer is named, once
    by_file = {f: computer for computer, f in picks.items()}
    assert "Windows" in by_file["AudioCleaner-$version-Windows.exe"]
    assert "Apple silicon" in by_file["AudioCleaner-$version-Mac-Apple-silicon.pkg"]
    assert "Intel" in by_file["AudioCleaner-$version-Mac-Intel.pkg"]
    assert "Pick the file that matches your computer" in notes and "About This Mac" in notes
    assert "downloads the models once" in notes and "updates itself" in notes
    assert "audiocleaner.live" not in notes  # not offered on the website
    # and nothing about where the build runs: no private repository, commit or link
    assert "GITHUB_REPOSITORY" not in step and "GITHUB_SHA" not in step and "github.com" not in step
    # The public copy's leak scan (tools/public-desktop/sync.py) names the private repository; that name is not written
    # here (this file is published), the scan's own pattern is used. The notes match none of its patterns.
    leaks = _public_copy_leak_patterns()
    assert not re.search(leaks["the name of the private repository"], step, re.I)
    assert [why for pattern, why in leaks.items() if re.search(pattern, notes, re.I)] == []
    assert release.index("gh release create") < release.index(name)  # after the official release


@pytest.mark.skipif("\n  release:\n" not in WORKFLOW, reason="this copy of the repository publishes no releases")
def test_the_app_is_built_for_the_repository_the_release_is_published_to():
    from audio_cleaner_desktop import __releases__

    app = WORKFLOW.split("\n  app:\n")[1].split("\n  tests:\n")[0]
    step = _step(app, "Version")
    built_for = re.search(r"ACW_RELEASES_REPO: (\$\{\{.*?\}\})", step).group(1)
    assert 'python desktop/packaging/version.py --write' in step
    publish = _step(WORKFLOW.split("\n  release:\n")[1], "Publish the release to the public repository")
    published_to = re.search(r"PUBLIC_REPO: (\$\{\{.*?\}\})", publish).group(1)
    assert built_for == published_to  # the app looks for its releases (and models) where they are published
    assert f"'{__releases__}'" in built_for  # with the same default as the source


def test_the_model_check_job_runs_the_url_guard_and_the_release_job_needs_no_model_script():
    if "\n  model-urls:\n" not in WORKFLOW:
        pytest.skip("this copy of the repository has no workflow")
    job = WORKFLOW.split("\n  model-urls:\n")[1].split("\n  ffmpeg:\n")[0]
    assert "run: python desktop/packaging/check_model_urls.py" in job and "secrets." not in job
    assert "release_models" not in WORKFLOW
