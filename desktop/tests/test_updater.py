"""Automatic updates: the signed manifest, the download, the 24 hour rule, the settings, the Windows command line,
the installer script's relaunch entry and the release job's signing. No network (a local HTTP server stands in for
the download server) and never the real signing key: every test makes its own key pair."""
from __future__ import annotations

import base64
import hashlib
import http.server
import json
import re
import shutil
import subprocess
import sys
import threading
from pathlib import Path

import pytest
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
from PySide6.QtCore import QSettings

from audio_cleaner_desktop import models, updater
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
def test_the_embedded_key_is_a_valid_ed25519_public_key():
    from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PublicKey

    raw = base64.b64decode(updater.PUBLIC_KEY, validate=True)
    assert len(raw) == 32 and Ed25519PublicKey.from_public_bytes(raw)
    assert updater.PUBLIC_KEY == make_update_manifest.app_public_key()  # CI reads it from the source the same way


def test_a_good_signature_is_accepted(key):
    data, _ = _manifest()
    update = updater.decide(data, key.sign(data), "2.0.3", WIN, _public(key))
    assert update.version == "2.0.15" and update.file.name == "AudioCleaner-2.0.15-Windows.exe"
    assert update.manifest == data


def test_a_tampered_manifest_is_refused(key):
    data, _ = _manifest()
    sig = key.sign(data)
    tampered = data.replace(b"2.0.15", b"9.9.99", 1)
    with pytest.raises(updater.UpdateError, match="not signed correctly"):
        updater.decide(tampered, sig, "2.0.3", WIN, _public(key))
    with pytest.raises(updater.UpdateError, match="not signed correctly"):
        updater.decide(data + b" ", sig, "2.0.3", WIN, _public(key))  # even a trailing space: the exact bytes count


def test_a_signature_from_another_key_is_refused(key):
    data, _ = _manifest()
    other = Ed25519PrivateKey.generate()
    with pytest.raises(updater.UpdateError, match="not signed correctly"):
        updater.decide(data, other.sign(data), "2.0.3", WIN, _public(key))
    with pytest.raises(updater.UpdateError):  # and nothing passes against the real embedded key
        updater.decide(data, other.sign(data), "2.0.3", WIN)


@pytest.mark.parametrize("sig", [b"", b"x" * 63, b"x" * 65, b"\0" * 64], ids=["empty", "short", "long", "zeros"])
def test_a_missing_or_malformed_signature_is_refused(key, sig):
    data, _ = _manifest()
    with pytest.raises(updater.UpdateError):
        updater.decide(data, sig, "2.0.3", WIN, _public(key))


def test_a_manifest_that_is_not_newer_is_ignored(key):
    data, _ = _manifest("2.0.15")
    sig = key.sign(data)
    assert updater.decide(data, sig, "2.0.15", WIN, _public(key)) is None  # the same version
    assert updater.decide(data, sig, "2.0.16", WIN, _public(key)) is None  # the running app is newer
    assert updater.decide(data, sig, "2.1.0", WIN, _public(key)) is None  # numbers, not text: 2.1.0 > 2.0.15
    assert updater.decide(data, sig, "2.0.9", WIN, _public(key)) is not None  # 15 > 9 (not a string compare)


def test_a_newer_manifest_without_a_file_for_this_platform_is_an_error(key):
    data, _ = _manifest()
    sig = key.sign(data)
    with pytest.raises(updater.UpdateError, match="no installer for this computer"):
        updater.decide(data, sig, "2.0.3", None, _public(key))
    body = json.loads(data)
    del body["files"]["macos-arm64"]
    data2 = json.dumps(body, sort_keys=True).encode()
    with pytest.raises(updater.UpdateError, match="no installer for this computer"):
        updater.decide(data2, key.sign(data2), "2.0.3", "macos-arm64", _public(key))
    assert updater.decide(data2, key.sign(data2), "2.0.3", WIN, _public(key)) is not None  # the other platforms still work


@pytest.mark.parametrize("change", [
    {"version": "latest"}, {"version": "2"}, {"version": "2.0.x"},
    {"files": []}, {"files": {WIN: "x"}},
    {"files": {WIN: {"name": "../evil.exe", "size": 5, "sha256": "a" * 64}}},
    {"files": {WIN: {"name": "sub/evil.exe", "size": 5, "sha256": "a" * 64}}},
    {"files": {WIN: {"name": "evil.pkg", "size": 5, "sha256": "a" * 64}}},  # the wrong kind of file for the platform
    {"files": {WIN: {"name": "a.exe", "size": -1, "sha256": "a" * 64}}},
    {"files": {WIN: {"name": "a.exe", "size": True, "sha256": "a" * 64}}},
    {"files": {WIN: {"name": "a.exe", "size": 3 << 30, "sha256": "a" * 64}}},
    {"files": {WIN: {"name": "a.exe", "size": 5, "sha256": "A" * 64}}},
    {"files": {WIN: {"name": "a.exe", "size": 5, "sha256": "abc"}}},
], ids=lambda c: next(iter(c)) + ":" + str(next(iter(c.values())))[:40])
def test_a_signed_but_malformed_manifest_is_refused(key, change):
    data, _ = _manifest(**change)
    with pytest.raises(updater.UpdateError):
        updater.decide(data, key.sign(data), "2.0.3", WIN, _public(key))


def test_versions_are_compared_as_numbers():
    assert updater.parse_version("2.0.15") > updater.parse_version("2.0.9") > updater.parse_version("1.99.99")
    for bad in ("", "2", "2.0.15-beta", "v2.0.15", "2.0.15\n", None):
        with pytest.raises(updater.UpdateError):
            updater.parse_version(bad)


def test_the_platform_key_names_one_of_the_three_installers():
    assert updater.platform_key() in (None, *updater.PLATFORMS)


# ---- a local server standing in for the download server ---------------------------------------------------------------------
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
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
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


def test_check_fetches_and_verifies(server, key):
    _publish(server, key)
    update = updater.check("2.0.3", WIN, public_key=_public(key), **_endpoints(server))
    assert update.version == "2.0.15"
    assert updater.check("2.0.15", WIN, public_key=_public(key), **_endpoints(server)) is None
    assert all(r.startswith("/desktop/update.json") for r in _Handler.requests)  # no query string, no extras


def test_check_refuses_a_manifest_signed_by_someone_else(server, key):
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


def test_the_request_names_the_app_and_its_version(server, key, monkeypatch):
    seen = []
    orig = _Handler.do_GET

    def spy(self):
        seen.append(self.headers.get("User-Agent"))
        orig(self)

    monkeypatch.setattr(_Handler, "do_GET", spy)
    _publish(server, key)
    updater.check("2.0.3", WIN, public_key=_public(key), **_endpoints(server))
    assert seen and set(seen) == {"AudioCleaner/2.0.3"}


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


def test_the_download_url_comes_from_the_version_and_name_not_from_the_manifest(server, key, tmp_path):
    _publish(server, key)
    update = updater.check("2.0.3", WIN, public_key=_public(key), **_endpoints(server))
    _Handler.requests.clear()
    updater.download(update, tmp_path / "u", base_url=f"{server}/desktop")
    assert [r.split()[0] for r in _Handler.requests] == ["/desktop/AudioCleaner-2.0.15-Windows.exe"]


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
    updater.prune(tmp_path / "missing", None)  # no folder, no problem


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
    assert "2.0.15" in ctl.status and "ready" in ctl.status
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


def test_a_newer_release_replaces_the_downloaded_one(qapp, tmp_path, server, key):
    _publish(server, key, "2.0.15")
    ctl, prefs = _controller(qapp, tmp_path, server, key)
    ctl.check_now()
    assert wait_until(lambda: ctl.ready_version == "2.0.15" and not ctl.busy, timeout=30)
    _publish(server, key, "2.0.16", installer=b"newer" * 5000)
    ctl.check_now()  # "Check now" ignores the 24 hour rule
    assert wait_until(lambda: ctl.ready_version == "2.0.16" and not ctl.busy, timeout=30)
    assert sorted(p.name for p in (tmp_path / "updates").iterdir()) == [
        "AudioCleaner-2.0.16-Windows.exe", "update.json", "update.json.sig"]


def test_check_now_reports_the_result_in_a_line_of_text(qapp, tmp_path, server, key):
    _publish(server, key, "2.0.3")  # the newest release is the running version
    ctl, prefs = _controller(qapp, tmp_path, server, key)
    ctl.check_now()
    assert wait_until(lambda: not ctl.busy and ctl.status.startswith("You have"), timeout=30)
    assert ctl.status == "You have the newest version (2.0.3)." and ctl.ready_version is None

    _Handler.files.clear()  # a failure is a quiet line, never a dialog or an exception
    ctl.check_now()
    assert wait_until(lambda: not ctl.busy and ctl.status.startswith("Could not"), timeout=30)
    assert "No update information is published yet" in ctl.status

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


def test_nothing_happens_when_the_setting_is_off_or_from_source(qapp, tmp_path, server, key):
    _publish(server, key)
    prefs = _prefs(tmp_path)
    prefs.check_updates = False
    ctl, _ = _controller(qapp, tmp_path, server, key, prefs=prefs)
    ctl.startup_check()
    ctl.schedule_startup_check(0)
    assert ctl._thread is None and _Handler.requests == []
    # from source (not frozen) there is no automatic update, and "Check now" says why without any request
    src = updater.UpdateController(_prefs(tmp_path), current="2.0.3", manifest_url=f"{server}/desktop/update.json")
    assert not src.enabled
    src.startup_check()
    src.check_now()
    assert src._thread is None and _Handler.requests == []
    assert src.status == "Updates are only installed in the installed app."


def test_the_setting_defaults_to_on_and_is_remembered(qapp, tmp_path):
    ini = tmp_path / "s.ini"
    assert Preferences(QSettings(str(ini), QSettings.Format.IniFormat)).check_updates is True
    p = Preferences(QSettings(str(ini), QSettings.Format.IniFormat))
    p.check_updates = False
    p._store.sync()
    assert Preferences(QSettings(str(ini), QSettings.Format.IniFormat)).check_updates is False


def test_the_settings_panel_shows_the_checkbox_and_check_now(qapp, tmp_path):
    from audio_cleaner_desktop.gui.settings_panel import SettingsPanel

    prefs = _prefs(tmp_path)
    panel = SettingsPanel(prefs)
    try:
        assert panel._updates.isChecked() and panel._check_now.text() == "Check now"
        panel._updates.click()
        assert prefs.check_updates is False
        panel._updates.click()
        assert prefs.check_updates is True
        panel._check_now.click()  # from source: says why, in a line of text
        assert panel._update_status.text() == "Updates are only installed in the installed app."
        assert not panel._update_status.isHidden()
    finally:
        panel.deleteLater()


def test_the_checkbox_survives_closing_and_reopening_the_window(qapp, tmp_path):
    from audio_cleaner_desktop.gui.main_window import MainWindow

    def open_():
        w = MainWindow([], settings=QSettings(str(tmp_path / "settings.ini"), QSettings.Format.IniFormat),
                       history_path=tmp_path / "history.db")
        w.show()
        return w

    def close(w):
        w.close()
        w._pool.waitForDone()
        w.deleteLater()

    w = open_()
    assert w.settings_panel._updates.isChecked()
    w.settings_panel._updates.click()
    close(w)
    again = open_()
    try:
        assert not again.settings_panel._updates.isChecked() and again.prefs.check_updates is False
    finally:
        close(again)


def test_about_says_whether_updates_are_on(qapp):
    from audio_cleaner_desktop.gui.about import AboutDialog

    for on, text in ((True, "Updates: on"), (False, "Updates: off")):
        d = AboutDialog(updates_on=on)
        try:
            assert d.updates_label.text() == text
        finally:
            d.deleteLater()


# ---- the window ------------------------------------------------------------------------------------------------
def _window(tmp_path):
    from audio_cleaner_desktop.gui.main_window import MainWindow

    w = MainWindow([], settings=QSettings(str(tmp_path / "settings.ini"), QSettings.Format.IniFormat),
                   history_path=tmp_path / "history.db")
    w.show()
    return w


def _fake_ready(w, tmp_path, version="2.0.15"):
    installer = b"installer" * 100
    path = tmp_path / f"AudioCleaner-{version}-Windows.exe"
    path.write_bytes(installer)
    entry = updater.UpdateFile(path.name, len(installer), hashlib.sha256(installer).hexdigest())
    w.updater.ready_update = updater.Update(version, entry, b"{}", b"")
    w.updater.ready_path = path
    w.updater.readyChanged.emit()


@pytest.mark.skipif(sys.platform == "darwin", reason="the banner says Install on a Mac")
def test_the_banner_offers_restart_and_waits_for_a_running_batch(qapp, tmp_path, monkeypatch):
    w = _window(tmp_path)
    try:
        assert w.banner_update.isHidden()
        _fake_ready(w, tmp_path)
        assert not w.banner_update.isHidden()
        assert "2.0.15 is ready. It installs when you close the app." in w.banner_update._label.text()
        assert w.banner_update._button.text() == "Restart now"
        monkeypatch.setattr(type(w.queue), "busy", property(lambda self: True))
        w._update_action()  # files are being cleaned: never interrupt them
        assert w._restart_when_idle and w.isVisible() and not w.updater.relaunch
        assert "when the files are finished" in w.banner_update._label.text()
        monkeypatch.undo()
        w._update_action()  # idle: closes the app and asks setup to start the new version
        assert w._closing and w.updater.relaunch and not w._restart_when_idle
    finally:
        w._closing = True
        w.deleteLater()


def test_closing_starts_the_installer_once_per_version(qapp, tmp_path, monkeypatch):
    started = []
    monkeypatch.setattr(updater, "start_windows_installer", lambda path, relaunch: started.append((path.name, relaunch)))
    prefs = _prefs(tmp_path)
    ctl = updater.UpdateController(prefs, enabled=True)
    entry_bytes = b"setup" * 50
    path = tmp_path / "AudioCleaner-2.0.15-Windows.exe"
    path.write_bytes(entry_bytes)
    ctl.ready_update = updater.Update("2.0.15", updater.UpdateFile(
        path.name, len(entry_bytes), hashlib.sha256(entry_bytes).hexdigest()), b"{}", b"")
    ctl.ready_path = path

    assert not ctl.install_on_close()  # not Windows (these tests run on Linux, too): nothing starts
    monkeypatch.setattr(updater.sys, "platform", "win32")
    try:
        assert ctl.install_on_close() and started == [(path.name, False)]
        assert prefs.update_install_tried == "2.0.15"
        assert not ctl.install_on_close() and len(started) == 1  # a cancelled setup is not started at every close
        ctl.relaunch = True  # "Restart now" is a clear request
        assert ctl.install_on_close() and started[-1] == (path.name, True)
        prefs.check_updates = False
        assert not ctl.install_on_close()  # switched off since: nothing starts
        prefs.check_updates = True
        path.write_bytes(entry_bytes[:-1] + b"X")  # the file changed since it was verified
        assert not ctl.install_on_close() and len(started) == 2
    finally:
        monkeypatch.undo()


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
    assert cmd(r"D:\Apps\Audio Cleaner\AudioCleaner.exe", True) == [*base, "/RELAUNCH=1"]
    assert updater.windows_installer_command(setup, r"D:\a.exe", False, {}) == base  # no environment, no switch
    # nothing in it asks for a visible setup, a restart of the computer, or a dialog
    assert {"/VERYSILENT", "/SUPPRESSMSGBOXES", "/NORESTART"} <= set(base) and "/SILENT" not in base


def test_the_installer_is_started_detached_without_a_console(monkeypatch, tmp_path):
    calls = []
    monkeypatch.setattr(updater.subprocess, "Popen", lambda cmd, **kw: calls.append((cmd, kw)))
    updater.start_windows_installer(tmp_path / "AudioCleaner-2.0.15-Windows.exe", True)
    cmd, kw = calls[0]
    assert cmd[0].endswith("AudioCleaner-2.0.15-Windows.exe") and cmd[-1] == "/RELAUNCH=1" and "/VERYSILENT" in cmd
    assert kw["creationflags"] & updater.DETACHED_PROCESS and kw["creationflags"] & updater.CREATE_NEW_PROCESS_GROUP
    assert kw["close_fds"] is True and kw["stdin"] == subprocess.DEVNULL and kw["stdout"] == subprocess.DEVNULL


def test_the_mac_installer_is_opened_with_open(monkeypatch, tmp_path):
    calls = []
    monkeypatch.setattr(updater.subprocess, "Popen", lambda cmd, **kw: calls.append(cmd))
    updater.open_mac_installer(tmp_path / "AudioCleaner-2.0.15-Mac-Intel.pkg")
    assert calls == [["open", str(tmp_path / "AudioCleaner-2.0.15-Mac-Intel.pkg")]]


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
    assert "function RelaunchAfterSilentUpdate: Boolean;" in code
    assert "Result := WizardSilent and (ExpandConstant('{param:RELAUNCH|0}') = '1');" in code
    assert ISS.index("function RelaunchAfterSilentUpdate") < ISS.index("procedure CurUninstallStepChanged")
    # what the updater passes is what the script reads, and the settings an update relies on are still there
    assert "/RELAUNCH=1" in updater.windows_installer_command(Path("s.exe"), Path("a.exe"), True, {})
    for line in ("UsePreviousAppDir=yes", "UsePreviousPrivileges=yes", "PrivilegesRequiredOverridesAllowed=dialog",
                 "CloseApplications=yes"):
        assert line in ISS


# ---- the release job and the manifest script -------------------------------------------------------------------------------
openssl = pytest.mark.skipif(shutil.which("openssl") is None, reason="openssl is needed")


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
    subprocess.run(["openssl", "genpkey", "-algorithm", "ed25519", "-out", str(pem)], check=True, capture_output=True)
    pub = make_update_manifest.public_key_of(pem)
    for key, pattern in make_update_manifest.INSTALLERS.items():
        (tmp_path / pattern.format(version="2.0.15")).write_bytes(b"x" + key.encode())
    manifest = tmp_path / "update.json"
    assert make_update_manifest.main(["manifest", "--version", "2.0.15", "--dir", str(tmp_path)]) == 0
    sig = tmp_path / "update.json.sig"
    subprocess.run(["openssl", "pkeyutl", "-sign", "-rawin", "-inkey", str(pem), "-in", str(manifest), "-out", str(sig)],
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


@pytest.mark.skipif("\n  release:\n" not in WORKFLOW, reason="this copy of the repository has no release job")
def test_the_release_job_signs_the_manifest_and_fails_closed():
    release = WORKFLOW.split("\n  release:\n")[1]
    assert "UPDATE_SIGNING_KEY: ${{ secrets.UPDATE_SIGNING_KEY }}" in release
    assert 'openssl pkeyutl -sign -rawin -inkey "$key" -in update.json -out update.json.sig' in release
    assert "files+=(update.json update.json.sig)" in release
    assert release.index("files+=(update.json update.json.sig)") < release.index("gh release create")  # same release as the installers
    assert 'if [ -z "$UPDATE_SIGNING_KEY" ]' in release and "::warning::" in release  # no secret: no manifest, a warning
    assert "check-key" in release  # a secret that is not the app's key publishes no manifest either
    assert 'rm -f "$key"' in release and "echo \"$UPDATE_SIGNING_KEY\"" not in release
    assert "desktop/audio_cleaner_desktop/updater.py" in WORKFLOW.split("sparse-checkout: |")[1]
    assert WORKFLOW.count("--require-hashes") >= 2 and "--require-hashes -r desktop/requirements.lock" in WORKFLOW


def test_cryptography_is_pinned_with_hashes_for_every_platform():
    lock = (ROOT / "desktop" / "requirements.lock").read_text(encoding="utf-8")
    blocks = {m[1]: m[0] for m in re.findall(r"^((\S+)==\S+[^\n]*\\\n(?:    --hash=\S+(?: \\)?\n?)+)", lock, re.M)}
    for name in ("cryptography", "cffi", "pycparser"):
        assert name in blocks, f"{name} is not pinned in requirements.lock"
        assert blocks[name].count("--hash=sha256:") >= 2
    # one wheel per platform at least: manylinux/musllinux, Windows, macOS universal2 (Intel and Apple silicon)
    assert blocks["cryptography"].count("--hash=sha256:") >= 20
    assert "cryptography==48.0.1" in blocks["cryptography"]  # the newest release that still has a macOS Intel wheel


@pytest.mark.skipif("\n  release:\n" not in WORKFLOW, reason="this copy of the repository publishes no releases")
def test_the_release_is_copied_to_the_public_repository_without_the_update_files():
    release = WORKFLOW.split("\n  release:\n")[1]
    assert "- name: Copy the release to the public repository" in release
    step = release.split("- name: Copy the release to the public repository")[1].split("\n      - name:")[0]
    assert "GH_TOKEN: ${{ secrets.PUBLIC_RELEASE_TOKEN }}" in step
    assert "continue-on-error: true" in step and "::warning::" in step  # never fails the official release
    assert 'if [ -z "$GH_TOKEN" ]' in step and "exit 0" in step  # no secret: skipped
    assert 'tag="v$version"' in step and "--latest" in step and '--target "$branch"' in step and "--clobber" in step
    assert "update.json" not in step  # only the installers go there
    assert step.count("-Windows.exe") >= 1 and "-Mac-Apple-silicon.pkg" in step and "-Mac-Intel.pkg" in step
    # the notes are written for the community: they point at the download page and say nothing about where the build runs
    assert "https://mc-multi-world.space/audio/" in step
    assert "GITHUB_REPOSITORY" not in step and "GITHUB_SHA" not in step and "github.com" not in step
    assert release.index("gh release create") < release.index("Copy the release to the public repository")
