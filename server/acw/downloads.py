"""The release files primary-ram serves under our own domain: what is on disk and which files may be served.

acw/mirror.py copies verified release files from the build service (GitHub Releases) into
<data>/downloads/<channel>/<directory>/<file>; acw/api/downloads_routes.py serves them. This module is what both
agree on: the layout, the per-channel state file, and the strict allow-list.

  <data>/downloads/
      mirror-status.json              when each channel was last mirrored (read by the monitor)
      <channel>/state.json            what is served: written LAST, atomically, after every file is verified
      <channel>/<directory>/<file>    desktop and obs-plugin: <directory> is the version; models: the release tag
      <channel>/.staging-*            downloads in progress (never served)

state.json:  {"current": {"version", "published_at", "files": {name: {"size", "sha256", "dir", ...}}, "platforms": {...}},
              "previous": [{... "retired_at"}], "seen": {"version", "since"} | null}
A file is served only if its name is in `current` (or, for installers and archives, in a `previous` entry that is
younger than KEEP_PREVIOUS_S: clients that started a download of the old version finish it), and only with the size
recorded when it was verified.
"""
from __future__ import annotations

import json
import os
import re
import stat
import threading
import time
from pathlib import Path

from .config import get_settings

CHANNELS = ("desktop", "obs-plugin", "models")
KEEP_PREVIOUS_S = 7 * 86400
# Every served name, and every directory name, is checked against this before it is ever joined to a path.
NAME_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9._+-]{0,150}")
DIR_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9._+-]{0,60}")
# Manifest files, by channel: (stored name, public name). Served from `current` only.
MANIFESTS = {
    "desktop": (("update.json", "update.json"), ("update.json.sig", "update.json.sig")),
    "obs-plugin": (("obs-plugin-manifest.json", "manifest.json"), ("obs-plugin-manifest.json.sig", "manifest.json.sig")),
}
# Public platform names of the plugin's download buttons -> the keys of the plugin manifest.
OBS_PLATFORMS = {"windows": "windows-x64", "windows-setup": "windows-setup", "mac": "macos", "ubuntu-debian": "ubuntu-debian",
                 "linux-other": "linux-other", "source": "source"}


def root() -> Path:
    return get_settings().data_dir / "downloads"


def channel_dir(channel: str) -> Path:
    if channel not in CHANNELS:
        raise ValueError(channel)
    return root() / channel


def vtuple(version: object) -> tuple[int, ...] | None:
    """'2.0.15' -> (2, 0, 15); None for anything that is not 2 to 4 dotted numbers."""
    if not isinstance(version, str) or re.fullmatch(r"\d{1,6}(?:\.\d{1,6}){1,3}", version) is None:
        return None
    return tuple(int(p) for p in version.split("."))


# ---- state file ----------------------------------------------------------------------------------------------------

_cache: dict[str, tuple[tuple[int, int, int], dict | None]] = {}
_cache_lock = threading.Lock()


def load_state(channel: str) -> dict | None:
    """The state of a channel, None when nothing was ever published. Cached until the file changes, so a burst of
    range requests does not re-read it."""
    path = channel_dir(channel) / "state.json"
    try:
        st = path.stat()
    except OSError:
        return None
    sig = (st.st_ino, st.st_mtime_ns, st.st_size)
    key = str(path)
    with _cache_lock:
        hit = _cache.get(key)
        if hit is not None and hit[0] == sig:
            return hit[1]
    try:
        state = json.loads(path.read_text("utf-8"))
        if not isinstance(state, dict):
            state = None
    except (OSError, ValueError):
        state = None
    with _cache_lock:
        _cache[key] = (sig, state)
    return state


def write_json_atomic(path: Path, data: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    tmp = path.with_name(path.name + f".tmp{os.getpid()}")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=1, sort_keys=True)
        f.write("\n")
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)
    try:
        fd = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(fd)
        finally:
            os.close(fd)
    except OSError:
        pass


def write_state(channel: str, state: dict) -> None:
    write_json_atomic(channel_dir(channel) / "state.json", state)


# ---- the allow-list --------------------------------------------------------------------------------------------------

def _entry_ok(name: str, entry: object) -> bool:
    return (
        isinstance(entry, dict)
        and NAME_RE.fullmatch(name) is not None
        and isinstance(entry.get("dir"), str) and DIR_RE.fullmatch(entry["dir"]) is not None
        and isinstance(entry.get("size"), int) and not isinstance(entry.get("size"), bool) and entry["size"] > 0
        and isinstance(entry.get("sha256"), str) and re.fullmatch(r"[0-9a-f]{64}", entry["sha256"]) is not None
    )


def lookup(channel: str, name: str, *, now: float | None = None, manifest_only: bool = False) -> dict | None:
    """The entry for a file the mirror verified and may serve, else None. `name` is the stored file name.
    Manifest files come from `current` only; other files from `current` or a still-kept previous version."""
    if NAME_RE.fullmatch(name) is None:
        return None
    state = load_state(channel)
    if not state:
        return None
    now = time.time() if now is None else now
    cur = state.get("current")
    stored_manifests = {m[0] for m in MANIFESTS.get(channel, ())}
    if isinstance(cur, dict) and isinstance(cur.get("files"), dict):
        entry = cur["files"].get(name)
        if _entry_ok(name, entry):
            return {**entry, "name": name, "manifest": name in stored_manifests, "version": cur.get("version")}
    if manifest_only or name in stored_manifests:
        return None
    for prev in state.get("previous") or []:
        if not isinstance(prev, dict) or not isinstance(prev.get("files"), dict):
            continue
        if now - float(prev.get("retired_at") or 0) > KEEP_PREVIOUS_S:
            continue
        entry = prev["files"].get(name)
        if _entry_ok(name, entry):
            return {**entry, "name": name, "manifest": False, "version": prev.get("version")}
    return None


def platform_file(channel: str, platform_key: str) -> str | None:
    """The stored file name of a platform in the current version (the plugin's `latest/<platform>` links)."""
    state = load_state(channel)
    cur = (state or {}).get("current")
    if not isinstance(cur, dict):
        return None
    name = (cur.get("platforms") or {}).get(platform_key)
    return name if isinstance(name, str) and NAME_RE.fullmatch(name) else None


def open_nofollow(channel: str, entry: dict) -> int:
    """File descriptor of a served file. Every component is opened without following symbolic links, relative to the
    descriptor of the one before; raises OSError if anything is not what the state says it is (a link, a directory,
    a different size)."""
    flags = os.O_RDONLY | os.O_CLOEXEC | os.O_NOFOLLOW
    parts = [str(root()), channel, entry["dir"]]
    dir_fd = os.open(parts[0], flags | os.O_DIRECTORY)
    try:
        for part in parts[1:]:
            nxt = os.open(part, flags | os.O_DIRECTORY, dir_fd=dir_fd)
            os.close(dir_fd)
            dir_fd = nxt
        fd = os.open(entry["name"], flags, dir_fd=dir_fd)
    finally:
        os.close(dir_fd)
    try:
        st = os.fstat(fd)
        if not stat.S_ISREG(st.st_mode) or st.st_size != entry["size"]:
            raise OSError("not the file that was verified")
    except BaseException:
        os.close(fd)
        raise
    return fd


def served_intact(channel: str, files: dict) -> bool:
    """True when every file of an entry set exists with its recorded size (a cheap check; `acw.admin mirror --verify`
    hashes them)."""
    for name, entry in files.items():
        try:
            path = channel_dir(channel) / entry["dir"] / name
            if not path.is_file() or path.stat().st_size != entry["size"]:
                return False
        except (OSError, KeyError, TypeError):
            return False
    return True
