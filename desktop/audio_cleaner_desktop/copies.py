"""Saved copies of a job: what lets a result be changed later without running a model again.

Every cleaned file gets a folder of its own in the app's data folder (`jobs/<32 hex digits>/`), next to the history:

    job.json          what the job remembers: the original's path with a fingerprint of it, the choices it was made with, the
                      copies below (with the sizes they were written with), the results saved from it
    noise.flac        the noise removal at full strength, before the strength and Leveling (24-bit FLAC, `acw_common.kept`)
    rebuilt.flac      the rebuilt voice, before Leveling, when Voice rebuild ran
    input.flac        the recording with the speaker echo taken out, when that ran (the strength blends it back in)
    segments.json     the transcript, when subtitles were made
    blocks/           the block cache of the per-part editor (acw_common.blocks.FlacCache, 24-bit FLAC): what each step made, block by
                      block (the noise removal at full strength, the rebuilt voice, the echo-free recording). A normal run fills it, an
                      edit reads it and only makes the blocks it does not have. It replaces noise.flac / rebuilt.flac / input.flac
                      for jobs made now; jobs made before keep those copies until they are opened or changed (editor_job.make_ready
                      turns them into blocks then)

The results and the original are NOT copied: they are the user's own files and stay where they are. The job only remembers where
the original was, and its fingerprint (size, modified time and a hash of 16 parts of it) to know whether it is still the same
file. The copies stay until the user deletes the job (`delete_job_dir`), which never touches anything outside this folder.
A copy only counts when its file is there with the size it was written with, so a job that failed halfway, a half-written
file, or a file someone removed by hand just means that copy is not there.

No Qt in here.
"""
from __future__ import annotations

import hashlib
import json
import logging
import os
import re
import shutil
import stat
import sys
import time
import uuid
from dataclasses import dataclass
from pathlib import Path

from acw_common import captions
from acw_common.ffmpeg import FFmpeg

log = logging.getLogger(__name__)

JOBS_DIR = "jobs"
MANIFEST = "job.json"
VERSION = 1
NOISE, REBUILT, INPUT, TRANSCRIPT, BLOCKS = "noise", "rebuilt", "input", "transcript", "blocks"
AUDIO_KINDS = (NOISE, REBUILT, INPUT)
BLOCKS_DIR = "blocks"
TRANSCRIPT_FILE = "segments.json"
KIND_NAMES = {
    NOISE: "Noise removal at full strength", REBUILT: "Rebuilt voice", INPUT: "Recording without the speaker echo",
    TRANSCRIPT: "Transcript", BLOCKS: "Processed parts (what each step made, to change one part later)",
}
_NAME = re.compile(r"^[0-9a-f]{32}$")
PART_SIZE = 16 * 1024
PART_COUNT = 16
ORPHAN_SECONDS = 24 * 3600  # a job folder nothing in the list points to is swept away after this long


def source_id_of(fp: dict) -> str:
    """What identifies the samples of an original in every block's key: its size and a hash of 16 parts of it (`fingerprint`)."""
    return "d" + hashlib.sha1(f"{fp.get('size')}:{fp.get('parts')}".encode()).hexdigest()[:20]


def jobs_root(data_dir: Path) -> Path:
    return Path(data_dir) / JOBS_DIR


def is_job_name(name: str) -> bool:
    return bool(_NAME.match(name or ""))


def new_job_dir(root: Path) -> Path:
    """A new, empty folder for one job in `root`."""
    root.mkdir(parents=True, exist_ok=True)
    path = root / uuid.uuid4().hex
    path.mkdir()
    return path


def plan_job_dir(root: Path) -> Path:
    """The folder a job's copies will go to (made when the first one is written), so a job that never runs leaves nothing behind."""
    return root / uuid.uuid4().hex


def job_dir(root: Path, name: str) -> Path | None:
    """The folder of the job called `name` (None for anything that is not a name this app gives)."""
    return root / name if is_job_name(name) else None


# ---------------------------------------------------------------------------------------------------- the original

def fingerprint(path: Path | str) -> dict:
    """Size, modified time and a hash of up to 16 parts of the file (the first, the last and 14 spread between): cheap even for a
    very large video, and any change to the length or to those parts is noticed. Raises OSError."""
    path = Path(path)
    st = path.stat()
    size = st.st_size
    h = hashlib.sha256(str(size).encode())
    with open(path, "rb") as f:
        if size <= PART_SIZE * PART_COUNT:
            h.update(f.read())
        else:
            for i in range(PART_COUNT):
                f.seek(round(i * (size - PART_SIZE) / (PART_COUNT - 1)))
                h.update(f.read(PART_SIZE))
    return {"size": size, "mtime_ns": st.st_mtime_ns, "parts": h.hexdigest()}


@dataclass(frozen=True)
class OriginalCheck:
    state: str  # "ok", "missing" or "changed"
    text: str = ""  # plain words for the user when it is not ok

    @property
    def ok(self) -> bool:
        return self.state == "ok"


def check_original(path: Path | str, fp: dict | None) -> OriginalCheck:
    """Is the file still the one the job was made from? Same size and modified time: yes, without reading it. A different
    modified time (the file was copied back, or touched): only if its parts hash the same."""
    path = Path(path)
    if not fp:
        return OriginalCheck("changed", "This job does not remember what its original looked like, so it can't tell whether the file "
                                        "is still the same.")
    try:
        if not path.is_file():
            return OriginalCheck("missing", f"The original file is not at {path} any more. It was moved, renamed or deleted.")
        st = path.stat()
        if st.st_size != fp.get("size"):
            return OriginalCheck("changed", "The original file has changed since it was cleaned (it is a different size).")
        if st.st_mtime_ns == fp.get("mtime_ns"):
            return OriginalCheck("ok")
        if fingerprint(path)["parts"] == fp.get("parts"):
            return OriginalCheck("ok")
    except OSError:
        return OriginalCheck("missing", f"The original file at {path} can't be read right now.")
    return OriginalCheck("changed", "The original file has changed since it was cleaned.")


# ---------------------------------------------------------------------------------------------------- disk

def scan_bytes(path: Path) -> int:
    """Bytes the regular files under `path` take (links are not followed)."""
    total = 0
    try:
        with os.scandir(path) as it:
            for e in it:
                try:
                    if e.is_symlink():
                        continue
                    if e.is_dir(follow_symlinks=False):
                        total += scan_bytes(Path(e.path))
                    elif e.is_file(follow_symlinks=False):
                        total += e.stat(follow_symlinks=False).st_size
                except OSError:
                    continue
    except OSError:
        return 0
    return total


def _forget_readonly(func, path, _exc) -> None:  # noqa: ANN001 - shutil's callback
    try:
        os.chmod(path, stat.S_IWRITE)
        func(path)
    except OSError:
        pass


def _rmtree(path: Path) -> None:
    if sys.version_info >= (3, 12):
        shutil.rmtree(path, onexc=_forget_readonly)
    else:
        shutil.rmtree(path, onerror=_forget_readonly)


def delete_job_dir(root: Path, name: str) -> bool:
    """Remove the folder of one job (its saved copies and its job.json) and nothing else: only a folder directly inside `root` with
    a name this app gives, and a link is removed as a link, never followed. True when it is gone (also when it never was there)."""
    path = job_dir(root, name)
    if path is None:
        return False
    try:
        if path.is_symlink():
            path.unlink()
        elif path.exists():
            _rmtree(path)
    except OSError:
        log.warning("could not remove the saved copies in %s", path, exc_info=True)
    return not os.path.lexists(path)


def job_names(root: Path) -> list[str]:
    """The folders in `root` that look like jobs."""
    try:
        return sorted(e.name for e in os.scandir(root) if is_job_name(e.name) and e.is_dir(follow_symlinks=False))
    except OSError:
        return []


def sweep_orphans(root: Path, known: set[str], *, now: float | None = None, older_than: float = ORPHAN_SECONDS) -> int:
    """Remove job folders no entry of the list points to (a crash between making the folder and listing the job), once they are
    `older_than` seconds old. Returns how many went."""
    now = time.time() if now is None else now
    removed = 0
    for name in job_names(root):
        if name in known:
            continue
        try:
            newest = max([os.lstat(root / name).st_mtime, *(e.stat().st_mtime for e in os.scandir(root / name))])
        except OSError:
            newest = 0.0
        if now - newest >= older_than and delete_job_dir(root, name):
            removed += 1
    return removed


# ---------------------------------------------------------------------------------------------------- one job's folder

def _atomic_write(path: Path, text: str) -> None:
    tmp = path.with_name(path.name + ".part")
    tmp.write_text(text, encoding="utf-8")
    os.replace(tmp, path)


def _plain_file(directory: Path, name: str, size: int | None) -> bool:
    """`name` is a regular file directly in `directory` (not a link), with the size it was written with."""
    if not isinstance(name, str) or not name or name != Path(name).name or name in (".", ".."):
        return False
    try:
        st = os.lstat(directory / name)
    except OSError:
        return False
    return stat.S_ISREG(st.st_mode) and (size is None or st.st_size == size)


class JobCopies:
    """One job's folder: job.json and the copies. `data` is the manifest."""

    def __init__(self, directory: Path, data: dict):
        self.dir = Path(directory)
        self.data = data

    # -- making and reading ----------------------------------------------------------------------------

    @classmethod
    def create(cls, directory: Path, original: Path, fp: dict, media: dict) -> "JobCopies":
        jc = cls(directory, {
            "version": VERSION, "created": time.time(),
            "original": {"path": str(original), **fp}, "media": dict(media), "samples": 0, "echo": False,
            "made": {}, "copies": {}, "saves": [],
        })
        jc.save()
        return jc

    @classmethod
    def load(cls, directory: Path) -> "JobCopies | None":
        """The job in `directory`, or None when there is no readable job.json of a version this app knows."""
        try:
            data = json.loads((Path(directory) / MANIFEST).read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return None
        if not isinstance(data, dict) or data.get("version") != VERSION or not isinstance(data.get("original"), dict):
            return None
        for key, kind in (("copies", dict), ("made", dict), ("media", dict), ("saves", list)):
            if not isinstance(data.get(key), kind):
                data[key] = kind()
        return cls(directory, data)

    def save(self) -> None:
        _atomic_write(self.dir / MANIFEST, json.dumps(self.data, indent=1, sort_keys=True))

    # -- the original ---------------------------------------------------------------------------------------

    @property
    def original_path(self) -> Path:
        return Path(self.data["original"].get("path", ""))

    @property
    def fingerprint(self) -> dict:
        o = self.data["original"]
        return {k: o.get(k) for k in ("size", "mtime_ns", "parts")}

    def check_original(self) -> OriginalCheck:
        return check_original(self.original_path, self.fingerprint)

    def relocate(self, path: Path) -> OriginalCheck:
        """The user pointed at the file the original moved to: accepted only if it is the same file (size and parts)."""
        path = Path(path)
        try:
            fp = fingerprint(path)
        except OSError:
            return OriginalCheck("missing", "That file can't be read.")
        mine = self.fingerprint
        if fp["size"] != mine.get("size") or fp["parts"] != mine.get("parts"):
            return OriginalCheck("changed", "That is not the same file as the original of this job.")
        self.data["original"] = {"path": str(path), **fp}
        self.save()
        return OriginalCheck("ok")

    @property
    def media(self) -> dict:
        return self.data["media"]

    @property
    def samples(self) -> int:
        return int(self.data.get("samples") or 0)

    @property
    def echo(self) -> bool:
        return bool(self.data.get("echo"))

    @property
    def made(self) -> dict:
        return self.data["made"]

    @property
    def saves(self) -> list[dict]:
        return self.data["saves"]

    # -- the copies -----------------------------------------------------------------------------------------

    def entry(self, kind: str) -> dict | None:
        """The description of a copy as `kept.load` wants it, or None when it is not (completely) there."""
        e = self.data["copies"].get(kind)
        if not isinstance(e, dict):
            return None
        if kind == BLOCKS:
            return e if self._blocks_there(e) else None
        if kind == TRANSCRIPT:
            return e if _plain_file(self.dir, e.get("file"), e.get("bytes")) else None
        try:
            cols, scale, samples = [int(c) for c in e["channels"]], float(e["scale"]), int(e["samples"])
        except (KeyError, TypeError, ValueError):
            return None
        if not (1 <= len(cols) <= 2 and all(c in (0, 1) for c in cols) and 0.0 < scale <= 1.0 and samples > 0):
            return None
        if not _plain_file(self.dir, e.get("file"), e.get("bytes")):
            return None
        return e

    def has(self, kind: str) -> bool:
        return self.entry(kind) is not None

    # -- the block cache (the per-part editor) --------------------------------------------------------------

    def _blocks_there(self, e: dict) -> bool:
        """The manifest names a block folder that is a real folder (not a link) holding at least one block."""
        if e.get("dir") != BLOCKS_DIR or not isinstance(e.get("source_id"), str):
            return False
        path = self.dir / BLOCKS_DIR
        try:
            if not stat.S_ISDIR(os.lstat(path).st_mode):
                return False
            with os.scandir(path) as it:
                return any(x.name.endswith(".flac") and x.is_file(follow_symlinks=False) for x in it)
        except OSError:
            return False

    @property
    def has_blocks(self) -> bool:
        return self.has(BLOCKS)

    @property
    def source_id(self) -> str:
        """What identifies the samples of the original in every block's key (its size and a hash of 16 parts of it)."""
        return source_id_of(self.fingerprint)

    def set_blocks(self, samples: int, echo: bool) -> None:
        """Record that the job's blocks are being kept (written before the first block is made, so a job that fails halfway keeps
        the blocks it had)."""
        self.data["samples"] = int(samples)
        self.data["echo"] = bool(echo)
        self.data["copies"][BLOCKS] = {"dir": BLOCKS_DIR, "source_id": self.source_id}
        self.save()

    def block_cache(self, ff: FFmpeg):
        """The job's block cache (made when it is first asked for)."""
        from acw_common import blocks

        return blocks.FlacCache(self.dir / BLOCKS_DIR, ff)

    @property
    def editor(self) -> dict:
        """What the per-part editor remembers about the job: the timeline of the last result, the decisions "automatic" made, the
        recording's facts and waveform, the suggestions (editor_job.py)."""
        e = self.data.get("editor")
        if not isinstance(e, dict):
            e = self.data["editor"] = {}
        return e
    def put_audio(self, kind: str, entry: dict, **extra) -> None:
        """Record a copy `kept.save` has just written (its size is read from disk)."""
        entry = {**entry, **extra}
        entry["bytes"] = (self.dir / entry["file"]).stat().st_size
        self.data["copies"][kind] = entry
        self.save()

    def put_transcript(self, segments: list[captions.Segment], language: str | None, echo: bool = False) -> None:
        text = json.dumps({"language": language, "segments": captions.segments_to_json(segments)})
        _atomic_write(self.dir / TRANSCRIPT_FILE, text)
        self.data["copies"][TRANSCRIPT] = {
            "file": TRANSCRIPT_FILE, "language": language, "echo": bool(echo),
            "bytes": (self.dir / TRANSCRIPT_FILE).stat().st_size,
        }
        self.save()

    def transcript(self) -> tuple[list[captions.Segment], str | None] | None:
        """(segments, language) of the saved transcript, or None."""
        e = self.entry(TRANSCRIPT)
        if e is None:
            return None
        try:
            data = json.loads((self.dir / e["file"]).read_text(encoding="utf-8"))
            return captions.segments_from_json(data["segments"]), data.get("language")
        except (OSError, ValueError, KeyError, TypeError):
            return None

    def drop(self, kind: str) -> None:
        """Forget a copy and delete its file (e.g. before it is written again)."""
        e = self.data["copies"].pop(kind, None)
        if kind == BLOCKS:
            path = self.dir / BLOCKS_DIR
            if path.is_dir() and not path.is_symlink():
                _rmtree(path)
        elif isinstance(e, dict) and _plain_file(self.dir, e.get("file"), None):
            try:
                (self.dir / e["file"]).unlink()
            except OSError:
                pass
        self.save()

    def kinds(self) -> list[str]:
        return [k for k in (BLOCKS, NOISE, REBUILT, INPUT, TRANSCRIPT) if self.has(k)]

    def __bool__(self) -> bool:  # a job with no copy at all is not worth a folder
        return bool(self.kinds())

    def sizes(self) -> list[tuple[str, int]]:
        """(kind, bytes) of each copy that is there."""
        return [(k, scan_bytes(self.dir / BLOCKS_DIR) if k == BLOCKS else int(self.data["copies"][k]["bytes"])) for k in self.kinds()]

    def bytes(self) -> int:
        """Disk the whole folder takes."""
        return scan_bytes(self.dir)

    def add_save(self, files: list[Path], settings: dict) -> None:
        """Remember a result saved from this job (a new one each time; the files are the user's own)."""
        self.data["saves"].append({"at": time.time(), "files": [str(f) for f in files], "settings": settings})
        self.save()


# ---------------------------------------------------------------------------------------------------- written while a job runs

class QuietCache:
    """The job's block cache as a run sees it that must not fail over it: a block that cannot be written (disk full, an encoder that
    failed) is reported (`on_lost`) and kept in memory for the rest of this run only (the renderer reads back what it made to put the
    result together), so the job goes on and the result is whole; the job just doesn't keep that block for later."""

    def __init__(self, cache, on_lost) -> None:
        self.cache, self.on_lost = cache, on_lost
        self._here: dict = {}

    def has(self, key: str) -> bool:
        return key in self._here or self.cache.has(key)

    def get(self, key: str):
        return self._here[key] if key in self._here else self.cache.get(key)

    def put(self, key: str, block) -> None:
        try:
            self.cache.put(key, block)
        except (OSError, RuntimeError) as e:  # a full disk, an encoder that failed: the job itself goes on
            log.warning("could not keep a block: %s", e)
            self._here[key] = block
            self.on_lost(e)


class Keeper:
    """Writes a job's copies as the job goes (engine.process): each one is recorded in job.json as soon as it is complete, so a job
    that fails or is stopped later still has the copies it had made. Nothing here may fail the job: a copy that can't be written
    (disk full) is logged and left out, and `lost` says so."""

    def __init__(self, directory: Path, src: Path, media: dict, ff: FFmpeg):
        self.dir, self.ff = Path(directory), ff
        self.lost: str | None = None
        self.job: JobCopies | None = None
        try:
            self.dir.mkdir(parents=True, exist_ok=True)
            self.job = JobCopies.create(self.dir, src.absolute(), fingerprint(src), media)
        except OSError as e:
            log.warning("saved copies are off for %s: %s", src, e)
            self.lost = "the folder for the saved copies can't be written"

    @classmethod
    def for_job(cls, job: JobCopies, ff: FFmpeg) -> "Keeper":
        """A keeper that adds to a job that exists already (a change that runs a step again: redo.py)."""
        keeper = cls.__new__(cls)
        keeper.dir, keeper.ff, keeper.lost, keeper.job = job.dir, ff, None, job
        return keeper

    def transcript(self, segments: list[captions.Segment], language: str | None, echo: bool) -> None:
        if self.job is None:
            return
        try:
            self.job.put_transcript(segments, language, echo)
        except OSError as e:
            log.warning("could not keep the transcript: %s", e)
            self.lost = self.lost or "it could not be written"

    def quiet(self, cache):
        """`cache` as a run that must not fail over it sees it (QuietCache); what it could not write is `lost`."""
        def lost(e: BaseException) -> None:
            self.lost = "the disk is full" if getattr(e, "errno", None) == 28 else "it could not be written"

        return QuietCache(cache, lost)

    def blocks(self, samples: int, echo: bool) -> None:
        """The job keeps its block cache from here on (engine.process, before the first model runs)."""
        if self.job is None:
            return
        try:
            self.job.set_blocks(samples, echo)
        except OSError as e:
            log.warning("could not start the block cache: %s", e)
            self.lost = self.lost or "it could not be written"

    def made(self, **choices) -> None:
        if self.job is not None:
            self.job.made.update(choices)
            try:
                self.job.save()
            except OSError:
                pass

    def finish(self) -> bool:
        """The job is over (done, or failed after some copies were made): keep the folder if it holds anything worth keeping,
        otherwise remove it. Returns whether it is kept."""
        if self.job is not None and self.job:
            return True
        self.discard()
        return False

    def discard(self) -> None:
        delete_job_dir(self.dir.parent, self.dir.name)
