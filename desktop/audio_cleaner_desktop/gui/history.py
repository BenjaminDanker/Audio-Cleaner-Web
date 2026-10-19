"""The job list: every cleaned (or failed) file, kept in a small SQLite file on this computer, with the folder of its saved copies."""
from __future__ import annotations

import json
import sqlite3
import time
from collections.abc import Iterable
from dataclasses import dataclass, field
from pathlib import Path

from .. import copies, labels

_SCHEMA = """
CREATE TABLE IF NOT EXISTS history (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    finished_at REAL NOT NULL,
    name TEXT NOT NULL,
    source TEXT NOT NULL,
    duration REAL NOT NULL,
    has_video INTEGER NOT NULL,
    settings TEXT NOT NULL,
    seconds_taken REAL NOT NULL,
    outputs TEXT NOT NULL,
    status TEXT NOT NULL,
    message TEXT NOT NULL
)
"""
# Added with the saved copies (copies.py): the folder of the job's copies ("" for an entry made before, or one that kept none), the
# bytes it takes, and the settings told in words (what the search reads).
_COLUMNS = (("job_dir", "TEXT NOT NULL DEFAULT ''"), ("copies_bytes", "INTEGER NOT NULL DEFAULT 0"), ("summary", "TEXT NOT NULL DEFAULT ''"))
DONE, FAILED = "done", "failed"
SORTS = {
    "newest": "finished_at DESC, id DESC",
    "oldest": "finished_at ASC, id ASC",
    "name": "name COLLATE NOCASE ASC, finished_at DESC, id DESC",
    "disk": "copies_bytes DESC, finished_at DESC, id DESC",
    "length": "duration DESC, finished_at DESC, id DESC",
}


@dataclass
class Entry:
    name: str
    source: str
    duration: float
    has_video: bool
    settings: dict
    seconds_taken: float
    outputs: list[str]
    status: str
    message: str = ""
    finished_at: float = field(default_factory=time.time)
    id: int = 0
    job_dir: str = ""  # folder name of the saved copies in the jobs folder; "" when there are none
    copies_bytes: int = 0
    summary: str = ""


@dataclass(frozen=True)
class Totals:
    files: int
    audio_seconds: float
    video_seconds: float
    seconds_spent: float


@dataclass(frozen=True)
class CopiesTotal:
    jobs: int  # jobs that have saved copies
    bytes: int  # disk they take together


def _like(term: str) -> str:
    return "%" + term.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_") + "%"


class HistoryStore:
    """The job list: one row per cleaned (or failed) file, kept in a small SQLite file on this computer. The saved copies of a job
    live in a folder under `jobs_root` (copies.py); deleting a row deletes that folder and nothing else."""

    def __init__(self, path: Path | str, jobs_root: Path | str | None = None):
        self._db = sqlite3.connect(str(path))
        self._db.row_factory = sqlite3.Row
        self._db.execute(_SCHEMA)
        have = {r["name"] for r in self._db.execute("PRAGMA table_info(history)")}
        for column, decl in _COLUMNS:
            if column not in have:
                self._db.execute(f"ALTER TABLE history ADD COLUMN {column} {decl}")
        self._db.commit()
        self.jobs_root = Path(jobs_root) if jobs_root is not None else Path(path).parent / copies.JOBS_DIR
        self._backfill_summaries()

    def close(self) -> None:
        self._db.close()

    def _backfill_summaries(self) -> None:
        """Entries from before the summary column get theirs (so the search finds "Gentle" or "leveled" in them too)."""
        rows = self._db.execute("SELECT id, settings, has_video FROM history WHERE summary = ''").fetchall()
        for r in rows:
            try:
                text = labels.describe_settings(json.loads(r["settings"]), bool(r["has_video"]))
            except (ValueError, KeyError, TypeError):
                text = ""
            self._db.execute("UPDATE history SET summary = ? WHERE id = ?", (text or "-", r["id"]))
        if rows:
            self._db.commit()

    def add(self, entry: Entry) -> int:
        try:
            summary = entry.summary or labels.describe_settings(entry.settings, entry.has_video) or "-"
        except (KeyError, TypeError):
            summary = "-"
        cur = self._db.execute(
            "INSERT INTO history (finished_at, name, source, duration, has_video, settings, seconds_taken, outputs, status,"
            " message, job_dir, copies_bytes, summary) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)",
            (
                entry.finished_at, entry.name, entry.source, entry.duration, int(entry.has_video),
                json.dumps(entry.settings), entry.seconds_taken, json.dumps(entry.outputs), entry.status, entry.message,
                entry.job_dir, entry.copies_bytes, summary,
            ),
        )
        self._db.commit()
        return int(cur.lastrowid)

    def _where(self, text: str) -> tuple[str, list[str]]:
        """Every word typed must be found in the name, the original's path, the results' paths, the message, the settings (told in
        words and as stored) or the status."""
        clauses, args = [], []
        for term in text.split():
            like = _like(term)
            clauses.append("(" + " OR ".join(f"{col} LIKE ? ESCAPE '\\'" for col in
                                              ("name", "source", "outputs", "message", "summary", "settings", "status")) + ")")
            args += [like] * 7
        return (" WHERE " + " AND ".join(clauses)) if clauses else "", args

    def search(self, text: str = "", limit: int | None = None, offset: int = 0, sort: str = "newest") -> list[Entry]:
        """Jobs matching `text` (every word of it), `limit` of them from the `offset`th on (all by default), in the order `sort`
        says (SORTS; newest first by default)."""
        where, args = self._where(text)
        rows = self._db.execute(
            f"SELECT * FROM history{where} ORDER BY {SORTS.get(sort, SORTS['newest'])} LIMIT ? OFFSET ?",
            (*args, -1 if limit is None else int(limit), max(0, int(offset))),
        )
        return [_entry(r) for r in rows]

    def count(self, text: str = "") -> int:
        where, args = self._where(text)
        return int(self._db.execute(f"SELECT COUNT(*) FROM history{where}", args).fetchone()[0])

    def get(self, entry_id: int) -> Entry | None:
        r = self._db.execute("SELECT * FROM history WHERE id = ?", (entry_id,)).fetchone()
        return _entry(r) if r is not None else None

    def recent_folders(self, limit: int = 200) -> list[Path]:
        """The folders of the latest entries' originals and results: where the app has written files."""
        folders: dict[Path, None] = {}
        for r in self._db.execute("SELECT source, outputs FROM history ORDER BY finished_at DESC, id DESC LIMIT ?", (limit,)):
            for name in (r["source"], *json.loads(r["outputs"])):
                folders[Path(name).parent] = None
        return list(folders)

    # -- saved copies --------------------------------------------------------------------------------------

    def set_copies(self, entry_id: int, job_dir: str, size: int | None = None) -> None:
        """The job's copies are in this folder (counted now, unless `size` is given)."""
        if size is None:
            size = copies.scan_bytes(self.jobs_root / job_dir) if job_dir else 0
        self._db.execute("UPDATE history SET job_dir = ?, copies_bytes = ? WHERE id = ?", (job_dir, int(size), entry_id))
        self._db.commit()

    def set_outputs(self, entry_id: int, outputs: list[str]) -> None:
        self._db.execute("UPDATE history SET outputs = ? WHERE id = ?", (json.dumps(outputs), entry_id))
        self._db.commit()

    def set_source(self, entry_id: int, source: str) -> None:
        self._db.execute("UPDATE history SET source = ? WHERE id = ?", (source, entry_id))
        self._db.commit()

    def set_job_result(self, entry_id: int, outputs: list[str], settings: dict, job_dir: str, copies_bytes: int) -> None:
        """A result was saved again from the job's copies: the entry now lists it (it is a done job whatever it was), told with the
        choices it was made with."""
        r = self._db.execute("SELECT has_video FROM history WHERE id = ?", (entry_id,)).fetchone()
        if r is None:
            return
        summary = labels.describe_settings(settings, bool(r["has_video"])) or "-"
        self._db.execute(
            "UPDATE history SET outputs = ?, settings = ?, summary = ?, job_dir = ?, copies_bytes = ?, status = ?, message = '' WHERE id = ?",
            (json.dumps(outputs), json.dumps(settings), summary, job_dir, int(copies_bytes), DONE, entry_id),
        )
        self._db.commit()

    def set_status(self, entry_id: int, status: str, message: str = "") -> None:
        self._db.execute("UPDATE history SET status = ?, message = ? WHERE id = ?", (status, message, entry_id))
        self._db.commit()

    def copies_total(self) -> CopiesTotal:
        r = self._db.execute("SELECT COUNT(*) AS n, COALESCE(SUM(copies_bytes), 0) AS b FROM history WHERE job_dir != ''").fetchone()
        return CopiesTotal(int(r["n"]), int(r["b"]))

    def reconcile(self, *, now: float | None = None, protect: Iterable[str] = ()) -> int:
        """Make the list and the folders agree: a job whose folder is gone (deleted by hand) no longer claims copies, the sizes
        are counted again, and folders no job points to (and none of `protect`, the ones being written now) go after a day.
        Returns how many folders were swept."""
        known = set(protect)
        for r in self._db.execute("SELECT id, job_dir, copies_bytes FROM history WHERE job_dir != ''").fetchall():
            folder = self.jobs_root / r["job_dir"]
            if not copies.is_job_name(r["job_dir"]) or not folder.is_dir() or copies.JobCopies.load(folder) is None:
                self._db.execute("UPDATE history SET job_dir = '', copies_bytes = 0 WHERE id = ?", (r["id"],))
                continue
            known.add(r["job_dir"])
            size = copies.scan_bytes(folder)
            if size != r["copies_bytes"]:
                self._db.execute("UPDATE history SET copies_bytes = ? WHERE id = ?", (size, r["id"]))
        self._db.commit()
        return copies.sweep_orphans(self.jobs_root, known, now=now)

    def delete(self, entry_ids: Iterable[int]) -> list[int]:
        """Delete jobs from the list, with their saved copies (their folders). The original and the results are never touched.
        A job whose copies could not be removed (a file in use) stays in the list. Returns the ids that stayed."""
        stayed = []
        for entry_id in entry_ids:
            r = self._db.execute("SELECT job_dir FROM history WHERE id = ?", (entry_id,)).fetchone()
            if r is None:
                continue
            if r["job_dir"] and not copies.delete_job_dir(self.jobs_root, r["job_dir"]):
                stayed.append(entry_id)
                continue
            self._db.execute("DELETE FROM history WHERE id = ?", (entry_id,))
        self._db.commit()
        return stayed

    def clear(self) -> list[int]:
        """Delete every job and every saved copy. Returns the ids that stayed (see delete)."""
        stayed = self.delete([r["id"] for r in self._db.execute("SELECT id FROM history")])
        if not stayed:
            for name in copies.job_names(self.jobs_root):  # copies of jobs that were never listed
                copies.delete_job_dir(self.jobs_root, name)
        return stayed

    def totals(self) -> Totals:
        row = self._db.execute(
            "SELECT COUNT(*) AS n,"
            " COALESCE(SUM(CASE WHEN has_video = 0 THEN duration END), 0) AS audio,"
            " COALESCE(SUM(CASE WHEN has_video = 1 THEN duration END), 0) AS video,"
            " COALESCE(SUM(seconds_taken), 0) AS spent FROM history WHERE status = ?",
            (DONE,),
        ).fetchone()
        return Totals(row["n"], row["audio"], row["video"], row["spent"])


def _entry(r: sqlite3.Row) -> Entry:
    return Entry(
        name=r["name"], source=r["source"], duration=r["duration"], has_video=bool(r["has_video"]),
        settings=json.loads(r["settings"]), seconds_taken=r["seconds_taken"], outputs=json.loads(r["outputs"]),
        status=r["status"], message=r["message"], finished_at=r["finished_at"], id=r["id"],
        job_dir=r["job_dir"], copies_bytes=r["copies_bytes"], summary=r["summary"],
    )
