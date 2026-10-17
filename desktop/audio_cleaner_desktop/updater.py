"""Automatic updates: look for a newer version in the app's GitHub releases, download it, and install it.

How it works (docs/desktop.md has the owner's side of it):

* Every desktop release carries two small files next to the installers: `update.json` (the version, the file
  name, size and SHA-256 of the installer of each platform) and `update.json.sig`, a raw Ed25519 signature over the exact
  bytes of `update.json`. CI makes the signature with a private key that never leaves CI; the matching
  public key is `PUBLIC_KEY` below. Without a valid signature nothing is accepted (an unsigned release simply never
  updates anybody), so whoever controls a download link still can't make the app install anything else.
* The newest release's files are always at `<config.LATEST_URL>update.json`. The app fetches it and the signature,
  verifies it, compares the version with its own (a manifest that is not newer is ignored), picks the file of this
  platform, and downloads that installer into `<app data>/updates/` with the model downloader's care (https only,
  resumable, size and SHA-256 checked) from that version's release (config.release_url): the URL is built from the
  version and the file name, never taken from the manifest. Only the newest download is kept. The models of the new
  version come from its release too: the updated app finds that a model it had changed (models.outdated) and
  downloads the new one by itself.
* Windows installs it when the app is closed (silently, over the existing install); a Mac opens the .pkg for the
  user, because the macOS installer asks for the password anyway.

This module has two parts: functions that need no Qt (the manifest, its verification, the download, the Windows command
line; they are what the tests drive), and `UpdateController`, the small QObject the window uses.
"""
from __future__ import annotations

import base64
import binascii
import hashlib
import json
import logging
import ntpath
import os
import platform
import re
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path

from PySide6.QtCore import QObject, QTimer, Signal

from . import __version__, config, models

log = logging.getLogger(__name__)

MANIFEST_URL = config.LATEST_URL + "update.json"
SIGNATURE_URL = config.LATEST_URL + "update.json.sig"
DOWNLOAD_BASE = None  # the installer comes from the release of the version the manifest names (config.release_url)
# The public half of the key CI signs update.json with (raw Ed25519, base64). To rotate the key see docs/desktop.md.
PUBLIC_KEY = "lkLV2nf15eOuttdcqpieY2QJFyPsF4WOIwidisbp6Hw="
MANIFEST_NAME = "update.json"
SIGNATURE_NAME = "update.json.sig"

CHECK_EVERY = 24 * 3600  # seconds between automatic checks
STARTUP_DELAY_MS = 10_000  # the first check waits this long after the window opened
MAX_MANIFEST_BYTES = 64 * 1024
MAX_INSTALLER_BYTES = 2 << 30  # an installer is never bigger than this
_TIMEOUT = 20

PLATFORMS = ("windows-x86_64", "macos-arm64", "macos-x86_64")
_SUFFIX = {"windows-x86_64": ".exe", "macos-arm64": ".pkg", "macos-x86_64": ".pkg"}
_VERSION = re.compile(r"\d{1,6}(?:\.\d{1,6}){1,3}")
_FILE_NAME = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,120}")
_SHA256 = re.compile(r"[0-9a-f]{64}")


class UpdateError(Exception):
    """Something went wrong with an update that is worth a line in the log and in Settings (no dialog)."""


# ---- the manifest ---------------------------------------------------------------------------------------------
@dataclass(frozen=True)
class UpdateFile:
    name: str
    size: int
    sha256: str


@dataclass(frozen=True)
class Manifest:
    version: str
    published: str
    files: Mapping[str, UpdateFile]


@dataclass(frozen=True)
class Update:
    """A verified newer version and the installer of this platform. `manifest` and `signature` are the exact bytes
    that were verified (kept next to the download, so a restart can verify them again without the network)."""
    version: str
    file: UpdateFile
    manifest: bytes
    signature: bytes


def parse_version(text: str) -> tuple[int, ...]:
    """'2.0.15' -> (2, 0, 15). Only dotted numbers count; anything else is an error (never guess)."""
    if not isinstance(text, str) or _VERSION.fullmatch(text) is None:
        raise UpdateError(f"{text!r} is not a version number")
    return tuple(int(p) for p in text.split("."))


def platform_key() -> str | None:
    """Which installer fits this computer, or None where there is none (Linux: only a development system)."""
    machine = platform.machine().lower()
    if sys.platform == "win32":
        return "windows-x86_64" if machine in ("amd64", "x86_64") else None
    if sys.platform == "darwin":
        return "macos-arm64" if machine in ("arm64", "aarch64") else "macos-x86_64"
    return None


def is_frozen() -> bool:
    return bool(getattr(sys, "frozen", False))


def verify_signature(data: bytes, signature: bytes, public_key: str = PUBLIC_KEY) -> None:
    """Raises UpdateError unless `signature` (64 raw bytes) is the Ed25519 signature of exactly `data`."""
    from cryptography.exceptions import InvalidSignature
    from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PublicKey

    if len(signature) != 64:
        raise UpdateError("The update information is not signed correctly, so it was ignored.")
    try:
        key = Ed25519PublicKey.from_public_bytes(base64.b64decode(public_key, validate=True))
        key.verify(signature, data)
    except (InvalidSignature, ValueError, binascii.Error) as e:
        raise UpdateError("The update information is not signed correctly, so it was ignored.") from e


def parse_manifest(data: bytes) -> Manifest:
    """The checked contents of an update.json (call verify_signature first; this only validates the shape)."""
    try:
        raw = json.loads(data.decode("utf-8"))
    except (UnicodeDecodeError, ValueError) as e:
        raise UpdateError("The update information is not readable.") from e
    if not isinstance(raw, dict) or not isinstance(raw.get("files"), dict):
        raise UpdateError("The update information is not in the expected form.")
    version = raw.get("version")
    parse_version(version)
    # A "notes_url" in the manifest (older releases had one) is signed but never used: the app shows no release page.
    files: dict[str, UpdateFile] = {}
    for key, entry in raw["files"].items():
        if key not in PLATFORMS:
            continue  # a platform this version of the app does not know about
        if not isinstance(entry, dict):
            raise UpdateError("The update information is not in the expected form.")
        name, size, sha = entry.get("name"), entry.get("size"), entry.get("sha256")
        if (not isinstance(name, str) or _FILE_NAME.fullmatch(name) is None or not name.endswith(_SUFFIX[key])
                or ".." in name):
            raise UpdateError("The update information names an unexpected file.")
        if not isinstance(size, int) or isinstance(size, bool) or not 0 < size <= MAX_INSTALLER_BYTES:
            raise UpdateError("The update information has an unexpected file size.")
        if not isinstance(sha, str) or _SHA256.fullmatch(sha) is None:
            raise UpdateError("The update information has an unexpected checksum.")
        files[key] = UpdateFile(name, size, sha)
    published = raw.get("published")
    return Manifest(version, published if isinstance(published, str) else "", files)


def decide(manifest_bytes: bytes, signature: bytes, current: str, key: str | None,
           public_key: str = PUBLIC_KEY) -> Update | None:
    """Verify a manifest and say whether it offers something: an Update for this platform, or None when the manifest
    is not newer than `current`. Raises UpdateError for a bad signature, a broken manifest, or a newer version that has no
    installer for this platform."""
    verify_signature(manifest_bytes, signature, public_key)
    manifest = parse_manifest(manifest_bytes)
    if parse_version(manifest.version) <= parse_version(current):
        return None
    if key is None or key not in manifest.files:
        raise UpdateError(f"Version {manifest.version} has no installer for this computer.")
    return Update(manifest.version, manifest.files[key], manifest_bytes, signature)


def user_agent(version: str = __version__) -> str:
    """What a download tells the server about us: the app and its version, as any browser would."""
    return f"{config.APP_NAME.replace(' ', '')}/{version}"


def _get_small(url: str, limit: int, opener, agent: str) -> bytes:
    request = urllib.request.Request(url, headers={"User-Agent": agent})
    try:
        with opener.open(request, timeout=_TIMEOUT) as response:
            data = response.read(limit + 1)
    except urllib.error.HTTPError as e:
        if e.code == 404:
            raise UpdateError("No update information is published yet.") from e
        raise UpdateError(f"The update server answered with an error ({e.code}).") from e
    except models.DownloadError as e:
        raise UpdateError(str(e)) from e
    except OSError as e:
        raise UpdateError(f"The update server could not be reached ({getattr(e, 'reason', e)}).") from e
    if len(data) > limit:
        raise UpdateError("The update information is larger than expected, so it was ignored.")
    return data


def check(current: str = __version__, key: str | None = None, *, opener=None, manifest_url: str = MANIFEST_URL,
          signature_url: str = SIGNATURE_URL, public_key: str = PUBLIC_KEY, agent: str | None = None) -> Update | None:
    """Fetch and verify the newest release's manifest. None: nothing newer. Raises UpdateError."""
    opener = opener or models.https_opener()
    agent = agent or user_agent(current)
    manifest = _get_small(manifest_url, MAX_MANIFEST_BYTES, opener, agent)
    signature = _get_small(signature_url, 256, opener, agent)
    return decide(manifest, signature, current, key if key is not None else platform_key(), public_key)


# ---- the download ----------------------------------------------------------------------------------------------
def updates_dir() -> Path:
    return models.app_data_dir() / "updates"


def _sha256_of(path: Path, cancel: threading.Event | None = None) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(1 << 20):
            if cancel is not None and cancel.is_set():
                raise models.Cancelled
            h.update(chunk)
    return h.hexdigest()


def is_complete(path: Path, entry: UpdateFile, cancel: threading.Event | None = None) -> bool:
    """The file is there with the manifest's size and SHA-256."""
    try:
        return path.is_file() and path.stat().st_size == entry.size and _sha256_of(path, cancel) == entry.sha256
    except OSError:
        return False


def _write_atomic(path: Path, data: bytes) -> None:
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_bytes(data)
    os.replace(tmp, path)


def prune(folder: Path, keep: str | None) -> None:
    """Delete every download in `folder` except the installer called `keep` (and its partial file), so only the
    newest one stays. `keep=None` deletes them all, and the saved manifest with them."""
    if not folder.is_dir():
        return
    for p in folder.iterdir():
        if not p.is_file():
            continue
        is_download = p.name.startswith("AudioCleaner-") and p.suffix in (".exe", ".pkg", ".part", ".tmp")
        wanted = keep is not None and (p.name == keep or p.name == keep + ".part")
        if (is_download and not wanted) or (keep is None and p.name in (MANIFEST_NAME, SIGNATURE_NAME)):
            try:
                p.unlink()
            except OSError:
                log.warning("could not delete the old download %s", p, exc_info=True)


def download(update: Update, folder: Path, progress: models.Progress | None = None,
             cancel: threading.Event | None = None, *, opener=None, base_url: str | None = DOWNLOAD_BASE,
             agent: str | None = None) -> Path:
    """Download (or finish downloading) the installer of `update` into `folder`, check it, remember the verified
    manifest beside it and delete older downloads. Returns the installer's path. Raises DownloadError, Cancelled."""
    dest = folder / update.file.name
    folder.mkdir(parents=True, exist_ok=True)
    if not is_complete(dest, update.file, cancel):
        dest.unlink(missing_ok=True)
        base = base_url if base_url is not None else config.release_url(update.version)
        url = base.rstrip("/") + "/" + update.file.name
        models.fetch_file(url, dest, update.file.size, update.file.sha256, progress, cancel, agent or user_agent(), opener)
    _write_atomic(folder / MANIFEST_NAME, update.manifest)
    _write_atomic(folder / SIGNATURE_NAME, update.signature)
    prune(folder, update.file.name)
    return dest


def pending(folder: Path, current: str = __version__, key: str | None = None, public_key: str = PUBLIC_KEY,
            cancel: threading.Event | None = None) -> tuple[Update, Path] | None:
    """A newer version downloaded earlier and still intact: its manifest is verified again (no network needed) and
    the installer's size and SHA-256 are checked. Leftovers of versions that are not newer than `current` (the
    update has been installed) are deleted."""
    try:
        manifest = (folder / MANIFEST_NAME).read_bytes()
        signature = (folder / SIGNATURE_NAME).read_bytes()
    except OSError:
        return None
    try:
        update = decide(manifest, signature, current, key if key is not None else platform_key(), public_key)
    except UpdateError as e:
        log.info("the saved update information is not usable: %s", e)
        prune(folder, None)
        return None
    if update is None:  # already installed (or the app is newer): nothing to keep
        prune(folder, None)
        return None
    path = folder / update.file.name
    if not is_complete(path, update.file, cancel):
        return None
    return update, path


# ---- installing -----------------------------------------------------------------------------------------------
DETACHED_PROCESS = 0x00000008
CREATE_NEW_PROCESS_GROUP = 0x00000200


def install_mode_switch(exe: str | Path, env: Mapping[str, str] | None = None) -> str:
    """Which Inno Setup switch keeps an update in the mode of the running install: /CURRENTUSER for a copy in the
    user's profile, /ALLUSERS for one in Program Files, none for any other folder (Setup then reuses the previous
    install's mode on its own: UsePreviousPrivileges). Windows paths: compared as Windows does (no case, either slash)."""
    env = os.environ if env is None else env

    def inside(var: str) -> bool:
        base = env.get(var)
        if not base:
            return False
        folder = ntpath.normcase(ntpath.normpath(base)).rstrip("\\")
        return ntpath.normcase(ntpath.normpath(str(exe))).startswith(folder + "\\")

    if inside("LOCALAPPDATA"):
        return "/CURRENTUSER"
    if any(inside(v) for v in ("ProgramFiles", "ProgramFiles(x86)", "ProgramW6432")):
        return "/ALLUSERS"
    return ""


def windows_installer_command(installer: str | Path, exe: str | Path, relaunch: bool, env: Mapping[str, str] | None = None) -> list[str]:
    """The command line that installs `installer` silently over the running install. `exe` is the running
    AudioCleaner.exe. With `relaunch`, setup starts the new version when it is done (/RELAUNCH=1, read by the
    [Run] entry of AudioCleaner.iss)."""
    cmd = [str(installer), "/VERYSILENT", "/SUPPRESSMSGBOXES", "/NORESTART"]
    switch = install_mode_switch(exe, env)
    if switch:
        cmd.append(switch)
    if relaunch:
        cmd.append("/RELAUNCH=1")
    return cmd


def start_windows_installer(installer: Path, relaunch: bool) -> None:
    """Start setup detached from this process (it outlives the app) and without a console window."""
    cmd = windows_installer_command(installer, Path(sys.executable), relaunch)
    log.info("starting the update: %s", cmd)
    subprocess.Popen(
        cmd, creationflags=DETACHED_PROCESS | CREATE_NEW_PROCESS_GROUP, close_fds=True,
        stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
    )


def open_mac_installer(pkg: Path) -> None:
    """Hand the .pkg to the macOS installer (it asks for the password itself)."""
    subprocess.Popen(["open", str(pkg)])


# ---- the part the window uses ----------------------------------------------------------------------------------
@dataclass(frozen=True)
class _Outcome:
    checked: bool  # the server was reached and its answer verified: counts for the 24 h rule
    message: str
    update: Update | None = None
    path: Path | None = None


class UpdateController(QObject):
    """Checks for updates in the background and keeps the downloaded one ready. Everything that touches the
    preferences runs on the GUI thread; the network work runs on one worker thread whose results arrive as signals.

    From source (not frozen) nothing happens by itself and "Check now" only says so. Failures never open a dialog:
    they go to the log and to the status line in Settings."""

    statusChanged = Signal(str)
    readyChanged = Signal()  # a downloaded update is now waiting, or is gone
    _outcome = Signal(object, bool)  # internal: worker -> GUI thread

    def __init__(self, prefs, parent: QObject | None = None, *, enabled: bool | None = None, folder: Path | None = None,
                 current: str = __version__, key: str | None = None, clock: Callable[[], float] = time.time,
                 public_key: str = PUBLIC_KEY, manifest_url: str = MANIFEST_URL, signature_url: str = SIGNATURE_URL,
                 download_base: str | None = DOWNLOAD_BASE):
        super().__init__(parent)
        self._prefs = prefs
        self.enabled = is_frozen() if enabled is None else enabled
        self._folder = folder
        self._current = current
        self._key = key if key is not None else platform_key()
        self._clock = clock
        self._public_key = public_key
        self._urls = (manifest_url, signature_url, download_base)
        self._thread: threading.Thread | None = None
        self._cancel = threading.Event()
        self.status = ""
        self.ready_update: Update | None = None
        self.ready_path: Path | None = None
        self.busy = False  # a check or download is under way (GUI thread)
        self.relaunch = False  # set by "Restart now": setup starts the new version when it is done
        self._outcome.connect(self._on_outcome)

    # -- state -----------------------------------------------------------------------------------------------
    @property
    def folder(self) -> Path:
        return self._folder if self._folder is not None else updates_dir()

    @property
    def running(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    @property
    def ready_version(self) -> str | None:
        return self.ready_update.version if self.ready_update is not None else None

    def due(self) -> bool:
        """An automatic check is wanted: the setting is on and the last one is more than 24 hours ago."""
        if not self._prefs.check_updates:
            return False
        last = self._prefs.update_last_check
        now = self._clock()
        return last <= 0 or last > now or now - last >= CHECK_EVERY  # a clock set back counts as due

    def _set_status(self, text: str) -> None:
        self.status = text
        self.statusChanged.emit(text)

    # -- starting a check -------------------------------------------------------------------------------------
    def schedule_startup_check(self, delay_ms: int = STARTUP_DELAY_MS) -> None:
        """Check once, shortly after the window opened (the app has to be frozen, and the setting on)."""
        if self.enabled and self._prefs.check_updates:
            QTimer.singleShot(delay_ms, self.startup_check)

    def startup_check(self) -> None:
        if self.enabled and self._prefs.check_updates:
            self._start(manual=False)

    def check_now(self) -> None:
        """The "Check now" button: ignores the 24 hour rule."""
        if not self.enabled:
            self._set_status("Updates are only installed in the installed app.")
            return
        if self.busy:
            return
        self._start(manual=True)

    def _start(self, manual: bool) -> None:
        if self.busy:
            return
        self._cancel.clear()
        wanted = manual or self.due()
        self.busy = True
        self._set_status("Checking for updates..." if wanted else self.status)
        self._thread = threading.Thread(target=self._work, args=(wanted, manual), name="update-check", daemon=True)
        self._thread.start()

    def cancel(self) -> None:
        self._cancel.set()

    def wait(self, timeout: float | None = None) -> None:
        if self._thread is not None:
            self._thread.join(timeout)

    # -- the worker thread --------------------------------------------------------------------------------------
    def _work(self, network: bool, manual: bool) -> None:
        try:
            outcome = self._run(network)
        except models.Cancelled:
            return
        except (UpdateError, models.DownloadError) as e:
            log.info("update check failed: %s", e)
            outcome = _Outcome(False, f"Could not check for updates: {e}")
        except Exception as e:  # noqa: BLE001 - an update problem must never reach the user as a crash
            log.exception("update check failed")
            outcome = _Outcome(False, f"Could not check for updates ({type(e).__name__}).")
        self._outcome.emit(outcome, manual)

    def _run(self, network: bool) -> _Outcome:
        folder = self.folder
        have = pending(folder, self._current, self._key, self._public_key, self._cancel)
        if have is not None and not network:
            return _Outcome(False, "", *have)
        if not network:
            return _Outcome(False, "")
        opener = models.https_opener()
        manifest_url, signature_url, base = self._urls
        update = check(self._current, self._key, opener=opener, manifest_url=manifest_url, signature_url=signature_url,
                       public_key=self._public_key)
        if update is None:
            prune(folder, None)
            return _Outcome(True, f"You have the newest version ({self._current}).")
        if have is not None and have[0].version == update.version:
            return _Outcome(True, "", *have)
        last = [-1]

        def progress(done: int, total: int) -> None:
            pct = int(100 * done / max(1, total))
            if pct != last[0]:
                last[0] = pct
                self._set_status_threadsafe(f"Downloading {update.version}... {pct}%")

        self._set_status_threadsafe(f"Downloading {update.version}...")
        path = download(update, folder, progress, self._cancel, opener=opener, base_url=base)
        return _Outcome(True, "", update, path)

    def _set_status_threadsafe(self, text: str) -> None:
        self.status = text
        self.statusChanged.emit(text)  # queued onto the GUI thread: this QObject lives there

    # -- back on the GUI thread ---------------------------------------------------------------------------------
    def _on_outcome(self, outcome: _Outcome, manual: bool) -> None:
        self.busy = False
        if outcome.checked:
            self._prefs.update_last_check = self._clock()
        if outcome.update is not None and outcome.path is not None:
            changed = self.ready_version != outcome.update.version
            self.ready_update, self.ready_path = outcome.update, outcome.path
            self._set_status(self._ready_text(outcome.update.version))
            if changed:
                self.readyChanged.emit()
            return
        if outcome.message:
            self._set_status(outcome.message)
        elif self.status.startswith(("Checking", "Downloading")):
            self._set_status("")
        else:
            self.statusChanged.emit(self.status)  # the check ended: buttons may be enabled again

    @staticmethod
    def _ready_text(version: str) -> str:
        if sys.platform == "darwin":
            return f"Version {version} is downloaded and ready to install."
        return f"Version {version} is ready. It installs when you close Audio Cleaner."

    # -- installing -----------------------------------------------------------------------------------------------
    def install_on_close(self) -> bool:
        """Windows, when the app is closing: start setup for the downloaded version (silently; with `relaunch` it
        starts the new version afterwards). Once per version unless the user chose "Restart now": a setup that
        was cancelled (the administrator prompt) is not started again at every close. True when it was started."""
        if sys.platform != "win32" or self.ready_update is None or self.ready_path is None:
            return False
        version = self.ready_update.version
        if not self._prefs.check_updates:
            return False
        if not self.relaunch and self._prefs.update_install_tried == version:
            return False
        try:
            if not is_complete(self.ready_path, self.ready_update.file):
                log.warning("the downloaded update is not intact any more; not installing it")
                return False
            self._prefs.update_install_tried = version
            start_windows_installer(self.ready_path, self.relaunch)
            return True
        except Exception:  # noqa: BLE001 - closing the app must always work
            log.exception("could not start the update")
            return False

    def open_installer(self) -> bool:
        """macOS: open the downloaded .pkg."""
        if self.ready_path is None or self.ready_update is None:
            return False
        try:
            if not is_complete(self.ready_path, self.ready_update.file):
                log.warning("the downloaded update is not intact any more")
                return False
            open_mac_installer(self.ready_path)
            return True
        except Exception:  # noqa: BLE001
            log.exception("could not open the installer")
            return False
