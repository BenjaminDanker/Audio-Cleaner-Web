"""Models that are downloaded once instead of shipping in the installer.

The installer stays small: noise removal (libDF + the DeepFilterNet3 models, a few MB) works straight
after installing, and the large models (subtitles, restoring highs, rebuilding the voice) are fetched by the
installer's own setup. The app only downloads them itself when the user asks for it (a Download button
after a setup download failed). Every file is pinned by URL (a fixed revision or release), size and
SHA-256. A download goes into
`<id>.partial/`, can resume after an interruption, is checked file by file and only then becomes
`<id>/`, so a half-finished or tampered model is never used. After that the app works offline.
"""
from __future__ import annotations

import hashlib
import os
import shutil
import ssl
import sys
import threading
import urllib.parse
import urllib.request
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from . import config

_CHUNK = 1 << 20
_TIMEOUT = 30
_COMPLETE = ".complete"


@dataclass(frozen=True)
class ModelFile:
    path: str  # relative to the model's folder
    url: str
    size: int
    sha256: str


@dataclass(frozen=True)
class SpeechModel:
    id: str
    title: str
    description: str
    licence: str  # one line for the About box
    files: tuple[ModelFile, ...]

    @property
    def size(self) -> int:
        return sum(f.size for f in self.files)


class DownloadError(Exception):
    """Something the user can act on (network, disk space, a file that does not match)."""


class Cancelled(Exception):
    pass


Progress = Callable[[int, int], None]  # bytes done, bytes total


CATALOG: dict[str, SpeechModel] = {}  # filled in by the subtitle engine (engine.SPEECH_MODELS)


def register(*models: SpeechModel) -> None:
    for m in models:
        CATALOG[m.id] = m


def app_data_dir() -> Path:
    """The app's data folder (Qt's AppLocalDataLocation layout, without creating it): downloaded models,
    scratch files, history and logs live in it."""
    org, app = config.ORG_NAME, config.APP_NAME
    if sys.platform == "win32":
        base = Path(os.environ.get("LOCALAPPDATA") or Path.home() / "AppData" / "Local")
    elif sys.platform == "darwin":
        base = Path.home() / "Library" / "Application Support"
    else:
        base = Path(os.environ.get("XDG_DATA_HOME") or Path.home() / ".local" / "share")
    return base / org / app


def models_dir() -> Path:
    """Where downloaded models live: next to the app's other data, or $ACW_DESKTOP_MODELS."""
    env = os.environ.get("ACW_DESKTOP_MODELS")
    if env:
        return Path(env)
    return app_data_dir() / "models"


def model_dir(model: SpeechModel) -> Path:
    """Where the app itself downloads `model` to."""
    return models_dir() / model.id


def installer_dir(model: SpeechModel) -> Path | None:
    """Where setup put `model` in a frozen build: next to the program on Windows (models/<id>), in
    /Library/Application Support/Audio Cleaner/models/<id> on a Mac (the .pkg's postinstall)."""
    if not getattr(sys, "frozen", False):
        return None
    if sys.platform == "darwin":
        return Path("/Library/Application Support") / config.APP_NAME / "models" / model.id
    return Path(sys.executable).resolve().parent / "models" / model.id


def _sizes_match(d: Path, model: SpeechModel) -> bool:
    return all((d / f.path).is_file() and (d / f.path).stat().st_size == f.size for f in model.files)


def _installer_copy_ok(d: Path, model: SpeechModel) -> bool:
    """The installer checked every file's SHA-256 before installing it and then wrote the `.complete` marker
    (the hashes it installed), which must match this catalog entry. An install made before the marker existed
    has none: sizes are all there is to compare then."""
    try:
        marker = (d / _COMPLETE).read_text(encoding="utf-8").split()
    except OSError:
        return _sizes_match(d, model)
    return sorted(marker) == sorted(f.sha256 for f in model.files) and _sizes_match(d, model)


def location(model: SpeechModel) -> Path | None:
    """The folder holding a complete copy of `model`: the app's own download, else the installer's copy."""
    if _downloaded(model):
        return model_dir(model)
    d = installer_dir(model)
    if d is not None and _installer_copy_ok(d, model):
        return d
    return None


def is_installed(model: SpeechModel) -> bool:
    return location(model) is not None


def _downloaded(model: SpeechModel) -> bool:
    """All files present with the pinned sizes and the marker written after the hashes were checked."""
    d = model_dir(model)
    try:
        marker = (d / _COMPLETE).read_text(encoding="utf-8").split()
    except OSError:
        return False
    if sorted(marker) != sorted(f.sha256 for f in model.files):
        return False  # made from another version of the catalog entry
    return _sizes_match(d, model)


def _installer_file(model: SpeechModel, f: ModelFile) -> Path | None:
    """The installer's copy of one file of `model` when it is there with the right size."""
    d = installer_dir(model)
    if d is None:
        return None
    p = d / f.path
    try:
        return p if p.is_file() and p.stat().st_size == f.size else None
    except OSError:
        return None


def missing_bytes(model: SpeechModel) -> int:
    """How much is still to download (a resumed download counts what is already there, and so do the
    files setup managed to install)."""
    if is_installed(model):
        return 0
    part = models_dir() / f"{model.id}.partial"
    have = 0
    for f in model.files:
        p = part / f.path
        if p.is_file():
            have += min(p.stat().st_size, f.size)
        elif (part / (f.path + ".part")).is_file():
            have += min((part / (f.path + ".part")).stat().st_size, f.size)
        elif _installer_file(model, f) is not None:
            have += f.size
    return max(0, model.size - have)


def _ssl_context() -> ssl.SSLContext:
    try:  # the operating system's certificate store (corporate proxies, macOS without certifi)
        import truststore

        return truststore.SSLContext(ssl.PROTOCOL_TLS_CLIENT)
    except Exception:  # noqa: BLE001
        return ssl.create_default_context()


def _sha256(path: Path, cancel: threading.Event | None) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(_CHUNK):
            if cancel is not None and cancel.is_set():
                raise Cancelled
            h.update(chunk)
    return h.hexdigest()


class _HttpsRedirects(urllib.request.HTTPRedirectHandler):
    """Follow redirects (Hugging Face and GitHub hand out the files from a CDN), but never to plain http."""

    def redirect_request(self, req, fp, code, msg, headers, newurl):
        if urllib.parse.urlparse(newurl).scheme != "https" and urllib.parse.urlparse(req.full_url).scheme == "https":
            raise DownloadError("The download was sent to an address that is not secure, so it was stopped.")
        return super().redirect_request(req, fp, code, msg, headers, newurl)


def _seed(model: SpeechModel, f: ModelFile, dest: Path, cancel: threading.Event | None) -> bool:
    """Reuse the installer's copy of `f` instead of downloading it again (setup got this far before it
    failed). Only a copy whose SHA-256 matches is used. True when `dest` is in place."""
    src = _installer_file(model, f)
    if src is None:
        return False
    try:
        if _sha256(src, cancel) != f.sha256:
            return False
        tmp = dest.with_name(dest.name + ".seed")
        shutil.copyfile(src, tmp)
        os.replace(tmp, dest)
        return True
    except OSError:
        return False


def _fetch(f: ModelFile, dest: Path, done_before: int, total: int, progress: Progress | None,
           cancel: threading.Event | None, opener) -> None:
    part = dest.with_name(dest.name + ".part")
    have = part.stat().st_size if part.is_file() else 0
    if have > f.size:
        part.unlink()
        have = 0
    if have < f.size:
        request = urllib.request.Request(f.url, headers={"User-Agent": f"{config.APP_NAME} desktop"})
        if have:
            request.add_header("Range", f"bytes={have}-")
        try:
            with opener.open(request, timeout=_TIMEOUT) as response:
                if have and response.status != 206:  # the server ignored the range: start over
                    have = 0
                with open(part, "ab" if have else "wb") as out:
                    while chunk := response.read(_CHUNK):
                        if cancel is not None and cancel.is_set():
                            raise Cancelled
                        out.write(chunk)
                        have += len(chunk)
                        if progress is not None:
                            progress(done_before + min(have, f.size), total)
        except Cancelled:
            raise
        except OSError as e:  # URLError, timeouts, connection resets
            raise DownloadError(f"The download stopped ({getattr(e, 'reason', e)}). Try again to continue where it left off.") from e
    if part.stat().st_size != f.size or _sha256(part, cancel) != f.sha256:
        part.unlink(missing_ok=True)
        raise DownloadError(f"{f.path} did not download correctly. Please try again.")
    os.replace(part, dest)


def download(model: SpeechModel, progress: Progress | None = None, cancel: threading.Event | None = None) -> Path:
    """Download (or finish downloading) `model`; returns its folder. Raises DownloadError or Cancelled."""
    final = model_dir(model)
    if is_installed(model):
        return location(model)
    root = models_dir()
    root.mkdir(parents=True, exist_ok=True)
    need = missing_bytes(model)
    free = shutil.disk_usage(root).free
    if need + (64 << 20) > free:
        raise DownloadError(
            f"Not enough free disk space: this download needs {need / 1e9:.1f} GB, {free / 1e9:.1f} GB is free."
        )
    part = root / f"{model.id}.partial"
    opener = urllib.request.build_opener(urllib.request.HTTPSHandler(context=_ssl_context()), _HttpsRedirects())
    done = 0
    for f in model.files:
        dest = part / f.path
        dest.parent.mkdir(parents=True, exist_ok=True)
        if not (dest.is_file() and dest.stat().st_size == f.size) and not _seed(model, f, dest, cancel):
            _fetch(f, dest, done, model.size, progress, cancel, opener)
        done += f.size
        if progress is not None:
            progress(done, model.size)
    (part / _COMPLETE).write_text("\n".join(f.sha256 for f in model.files), encoding="utf-8")
    if final.exists():
        shutil.rmtree(final)
    os.replace(part, final)
    return final


def remove(model: SpeechModel) -> None:
    for d in (model_dir(model), models_dir() / f"{model.id}.partial"):
        shutil.rmtree(d, ignore_errors=True)
