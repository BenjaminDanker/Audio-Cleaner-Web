"""The history of cleaned files, kept in a small SQLite file on this computer."""
from __future__ import annotations

import json
import sqlite3
import time
from collections.abc import Iterable
from dataclasses import dataclass, field
from pathlib import Path

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
DONE, FAILED = "done", "failed"
LIST_LIMIT = 500


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


@dataclass(frozen=True)
class Totals:
    files: int
    audio_seconds: float
    video_seconds: float
    seconds_spent: float


class HistoryStore:
    def __init__(self, path: Path | str):
        self._db = sqlite3.connect(str(path))
        self._db.row_factory = sqlite3.Row
        self._db.execute(_SCHEMA)
        self._db.commit()

    def close(self) -> None:
        self._db.close()

    def add(self, entry: Entry) -> int:
        cur = self._db.execute(
            "INSERT INTO history (finished_at, name, source, duration, has_video, settings, seconds_taken, outputs, status,"
            " message) VALUES (?,?,?,?,?,?,?,?,?,?)",
            (
                entry.finished_at, entry.name, entry.source, entry.duration, int(entry.has_video),
                json.dumps(entry.settings), entry.seconds_taken, json.dumps(entry.outputs), entry.status, entry.message,
            ),
        )
        self._db.commit()
        return int(cur.lastrowid)

    def search(self, text: str = "", limit: int = LIST_LIMIT) -> list[Entry]:
        """Newest first (at most `limit`); `text` matches file names and messages."""
        like = "%" + text.strip().replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_") + "%"
        rows = self._db.execute(
            "SELECT * FROM history WHERE name LIKE ? ESCAPE '\\' OR message LIKE ? ESCAPE '\\'"
            " ORDER BY finished_at DESC, id DESC LIMIT ?",
            (like, like, min(limit, LIST_LIMIT)),
        )
        return [_entry(r) for r in rows]

    def recent_folders(self, limit: int = 200) -> list[Path]:
        """The folders of the latest entries' originals and results: where the app has written files."""
        folders: dict[Path, None] = {}
        for r in self._db.execute("SELECT source, outputs FROM history ORDER BY finished_at DESC, id DESC LIMIT ?", (limit,)):
            for name in (r["source"], *json.loads(r["outputs"])):
                folders[Path(name).parent] = None
        return list(folders)

    def delete(self, entry_ids: Iterable[int]) -> None:
        self._db.executemany("DELETE FROM history WHERE id = ?", [(i,) for i in entry_ids])
        self._db.commit()

    def clear(self) -> None:
        self._db.execute("DELETE FROM history")
        self._db.commit()

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
    )
