"""Models that are downloaded once instead of shipping in the installer.

The installer stays small: it holds the program (with libDF and FFmpeg) and no model. Every model, the noise
removal ones (23 MB, required) and the large ones (subtitles, restoring highs, rebuilding the voice), is fetched by
the installer's own setup. The app only downloads them itself when the user asks for it (a Download button
after a setup download failed). Every file is pinned by URL (a fixed revision), size and
SHA-256, and every model is downloaded on its own: someone who only wants one model downloads only its files.
A download goes into `<id>.partial/`, can resume after an interruption, is checked file by file and only then becomes
`<id>/`, so a half-finished or tampered model is never used. After that the app works offline.

A model can also have licence texts (`licences`, paths like `LICENSES/<name>` inside its folder). They are downloaded
with it, checked the same way, and are not part of what makes the model complete: they are not in `files`, not in the
`.complete` marker and not in the `.partial` pin, so a copy installed before the texts were fetched separately (the
old models archive, setup of an older version) still counts as installed and is never downloaded again for them.

Changing a model (a new revision) is safe: the old copy no longer matches the catalog, so it does not
count as installed, and nothing half-finished from the old one is resumed (a `.partial` folder is pinned to the hashes
it is for). The app downloads the new version by itself when the old one was there (outdated()), and removes what is
left of the old one afterwards (remove_stale(), which also clears the `.partial` folders of the models archive that
older versions used).
"""
from __future__ import annotations

import hashlib
import logging
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

log = logging.getLogger(__name__)

_CHUNK = 1 << 20
_TIMEOUT = 30
_COMPLETE = ".complete"
_PIN = ".pin"  # in a .partial folder: the hashes the download in it is for (see _claim)


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
    licences: tuple[ModelFile, ...] = ()  # licence texts that come with it (see the module docstring)
    required: bool = False  # the app cannot do its main job without it: setup insists, the app asks for it when missing

    @property
    def size(self) -> int:
        """Size of the model's files once installed."""
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


def licence_dirs() -> list[Path]:
    """The folders with licence texts that came with the downloaded models: each installed model's own `LICENSES`
    folder, and the shared `LICENSES` folder next to the models that setup or the old models archive left."""
    dirs = [d / "LICENSES" for m in CATALOG.values() if (d := location(m)) is not None]
    dirs.append(models_dir() / "LICENSES")
    installed = installer_models_dir()
    if installed is not None:
        dirs.append(installed / "LICENSES")
    seen: list[Path] = []
    for d in dirs:
        if d.is_dir() and d not in seen:
            seen.append(d)
    return seen


def model_dir(model: SpeechModel) -> Path:
    """Where the app itself downloads `model` to."""
    return models_dir() / model.id


def installer_models_dir() -> Path | None:
    """Where setup put the models in a frozen build: next to the program on Windows (models), in
    /Library/Application Support/Audio Cleaner/models on a Mac (the .pkg's postinstall)."""
    if not getattr(sys, "frozen", False):
        return None
    if sys.platform == "darwin":
        return Path("/Library/Application Support") / config.APP_NAME / "models"
    return Path(sys.executable).resolve().parent / "models"


def installer_dir(model: SpeechModel) -> Path | None:
    """Where setup put `model` (installer_models_dir()/<id>)."""
    root = installer_models_dir()
    return None if root is None else root / model.id


def _sizes_match(d: Path, model: SpeechModel) -> bool:
    return all((d / f.path).is_file() and (d / f.path).stat().st_size == f.size for f in model.files)


def pin_token(model: SpeechModel) -> str:
    """What a `.partial` folder of `model` is pinned to: the hashes of its files."""
    return "\n".join(f.sha256 for f in model.files)


def _pinned(d: Path, token: str) -> bool:
    try:
        return (d / _PIN).read_text(encoding="utf-8").strip() == token
    except OSError:
        return False


def _claim(d: Path, token: str) -> None:
    """Use the partial folder `d` for a download of `token`. What it holds is resumed only when it was started for
    exactly these hashes; anything else (another revision of the model, an older version of the app that wrote no pin)
    is thrown away, so a stale half-download can never end up in a new model."""
    if _pinned(d, token):
        return
    shutil.rmtree(d, ignore_errors=True)
    d.mkdir(parents=True, exist_ok=True)
    (d / _PIN).write_text(token, encoding="utf-8")


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
    pinned = _pinned(part, pin_token(model))
    have = 0
    for f in model.files:
        p = part / f.path
        if not pinned:
            if _installer_file(model, f) is not None:
                have += f.size
        elif p.is_file():
            have += min(p.stat().st_size, f.size)
        elif (part / (f.path + ".part")).is_file():
            have += min((part / (f.path + ".part")).stat().st_size, f.size)
        elif _installer_file(model, f) is not None:
            have += f.size
    return max(0, model.size - have)


def download_bytes(models_: "list[SpeechModel] | tuple[SpeechModel, ...]") -> int:
    """What downloading all of `models_` takes."""
    return sum(m.size for m in models_)


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
    """Follow redirects (GitHub hands out release files from its storage host), but never to plain http."""

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


def https_opener() -> urllib.request.OpenerDirector:
    """The opener every download of the app uses: the operating system's certificates, redirects only to https."""
    return urllib.request.build_opener(urllib.request.HTTPSHandler(context=_ssl_context()), _HttpsRedirects())


def _fetch(f: ModelFile, dest: Path, done_before: int, total: int, progress: Progress | None,
           cancel: threading.Event | None, opener, user_agent: str | None = None) -> None:
    part = dest.with_name(dest.name + ".part")
    have = part.stat().st_size if part.is_file() else 0
    if have > f.size:
        part.unlink()
        have = 0
    if have < f.size:
        request = urllib.request.Request(f.url, headers={"User-Agent": user_agent or f"{config.APP_NAME} desktop"})
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


def fetch_file(url: str, dest: Path, size: int, sha256: str, progress: Progress | None = None,
               cancel: threading.Event | None = None, user_agent: str | None = None, opener=None) -> None:
    """Download one file pinned by size and SHA-256 to `dest` with the same care as a model: resumes from
    `dest.part` after an interruption, checks the size and the hash, and only then puts it at `dest`. Raises
    DownloadError or Cancelled. (The updater uses it for the installer of a new version.)"""
    dest.parent.mkdir(parents=True, exist_ok=True)
    part = dest.with_name(dest.name + ".part")
    have = part.stat().st_size if part.is_file() else 0
    _check_disk(dest.parent, max(0, size - min(have, size)))
    f = ModelFile(dest.name, url, size, sha256)
    _fetch(f, dest, 0, size, progress, cancel, opener or https_opener(), user_agent)


def _check_disk(root: Path, need: int) -> None:
    free = shutil.disk_usage(root).free
    if need + (64 << 20) > free:
        raise DownloadError(
            f"Not enough free disk space: this download needs {need / 1e9:.1f} GB, {free / 1e9:.1f} GB is free."
        )


def download(model: SpeechModel, progress: Progress | None = None, cancel: threading.Event | None = None) -> Path:
    """Download (or finish downloading) `model`; returns its folder. Raises DownloadError or Cancelled."""
    final = model_dir(model)
    if is_installed(model):
        return location(model)
    root = models_dir()
    root.mkdir(parents=True, exist_ok=True)
    part = root / f"{model.id}.partial"
    _claim(part, pin_token(model))
    _check_disk(root, missing_bytes(model) + sum(f.size for f in model.licences))
    opener = https_opener()
    done = 0
    for f in model.files:
        dest = part / f.path
        dest.parent.mkdir(parents=True, exist_ok=True)
        if not (dest.is_file() and dest.stat().st_size == f.size) and not _seed(model, f, dest, cancel):
            _fetch(f, dest, done, model.size, progress, cancel, opener)
        done += f.size
        if progress is not None:
            progress(done, model.size)
    for f in model.licences:  # small texts: no progress; they are checked like the files but are not part of the marker
        dest = part / f.path
        dest.parent.mkdir(parents=True, exist_ok=True)
        if not (dest.is_file() and dest.stat().st_size == f.size):
            _fetch(f, dest, done, model.size, None, cancel, opener)
    return _put_in_place(model, part, final)


def _put_in_place(model: SpeechModel, part: Path, final: Path) -> Path:
    (part / _PIN).unlink(missing_ok=True)
    (part / _COMPLETE).write_text("\n".join(f.sha256 for f in model.files), encoding="utf-8")
    if final.exists():
        shutil.rmtree(final)
    os.replace(part, final)
    return final


def remove(model: SpeechModel) -> None:
    for d in (model_dir(model), models_dir() / f"{model.id}.partial"):
        shutil.rmtree(d, ignore_errors=True)


# ---- when the catalog changes a model --------------------------------------------------------------------------------
def _marker(d: Path) -> list[str] | None:
    """The hashes in the `.complete` marker of model folder `d`, or None when there is no marker."""
    try:
        return sorted((d / _COMPLETE).read_text(encoding="utf-8").split())
    except OSError:
        return None


def had_before(model: SpeechModel) -> bool:
    """The user had this model: a complete copy (it has the marker that is only written once every file was checked)
    from some version of the catalog entry is in the app's folder or the installer's. Says nothing about whether that
    copy is the current one."""
    return any(d is not None and _marker(d) is not None for d in (model_dir(model), installer_dir(model)))


def needs_update(model: SpeechModel) -> bool:
    """The user had an older copy of `model` and the current catalog entry is not installed: the app downloads the new
    one by itself (a model the user never had is only offered, never forced)."""
    return not is_installed(model) and had_before(model)


def outdated(catalog: "list[SpeechModel] | tuple[SpeechModel, ...] | None" = None) -> list[SpeechModel]:
    """The models that need_update(), in catalog order."""
    return [m for m in (CATALOG.values() if catalog is None else catalog) if needs_update(m)]


def _writable(path: Path) -> bool:
    """`path` can be deleted: it and the folder it is in are writable by this user."""
    return os.access(path, os.W_OK) and os.access(path.parent, os.W_OK)


def _delete(path: Path, removed: list[Path]) -> None:
    try:
        shutil.rmtree(path)
    except FileNotFoundError:
        return
    except OSError:
        log.warning("could not delete the outdated model data %s", path, exc_info=True)
        return
    removed.append(path)


def remove_stale(catalog: "list[SpeechModel] | tuple[SpeechModel, ...] | None" = None) -> list[Path]:
    """Delete what older versions of the models left behind (up to 1.6 GB each), but only once the current ones are
    there. Call it when no download is running. Returns what was deleted.

    * A model folder (it has a `.complete` marker) in the app's models folder whose marker names other hashes than the
      catalog's entry of that id goes when the current model is installed (in either place).
    * A model folder of an id that is not in the catalog any more, and a `.partial` folder that is not the current
      download of a catalog model (another revision; the folders of the old models archive), go when no model still
      waits for its update.
    * The same for the installer's folder ({app}\\models, /Library/Application Support/...), but only where this user
      may delete: otherwise it stays (the next interactive setup forgets what is stale).
    * Nothing else is touched: the LICENSES folder, folders without a marker and files are left alone, so a models
      folder that also holds other things is safe.
    """
    cat = {m.id: m for m in (CATALOG.values() if catalog is None else catalog)}
    waiting = any(needs_update(m) for m in cat.values())
    removed: list[Path] = []

    def stale_models(root: Path, own: bool) -> None:
        try:
            children = sorted(p for p in root.iterdir() if p.is_dir())
        except OSError:
            return
        for d in children:
            if own and d.name.endswith(".partial"):
                stem = d.name[: -len(".partial")]
                if stem in cat:
                    keep = _pinned(d, pin_token(cat[stem]))
                else:
                    keep = waiting  # an orphan: only after the updates are done
                if not keep:
                    _delete(d, removed)
                continue
            have = _marker(d)
            if have is None:
                continue  # not a model folder (LICENSES, .unpack, someone else's): never ours to delete
            model = cat.get(d.name)
            if model is not None:
                if have == sorted(f.sha256 for f in model.files) or not is_installed(model):
                    continue  # current, or its replacement is not here yet: this copy stays
            elif waiting:
                continue
            if own or _writable(d):
                _delete(d, removed)

    stale_models(models_dir(), True)
    installed = installer_models_dir()
    if installed is not None and installed.resolve() != models_dir().resolve():
        stale_models(installed, False)
    return removed
