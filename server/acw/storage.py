"""How big the things that grow are (docs/operations.md, "What grows and what bounds it").

`python -m acw.admin storage` prints it; the monitor alerts when the data directory (without the job files) or the
backup directory is larger than it should ever be, so a leak is noticed in days, not when the disk is full.
Job files are not part of the 5 GB check: they are uploads and results that legitimately take many GB while work
is going on, and they are bounded by their 24 hours and the free-space alert instead.
"""
from __future__ import annotations

import os
from pathlib import Path

GB = 1024**3


def _limit(name: str, default_gb: float) -> int:
    try:
        return int(float(os.environ.get(name, "") or default_gb) * GB)
    except ValueError:
        return int(default_gb * GB)


def data_limit() -> int:
    """ACW_DATA_ALERT_GB (default 5): the data directory except jobs/."""
    return _limit("ACW_DATA_ALERT_GB", 5)


def backups_limit() -> int:
    """ACW_BACKUPS_ALERT_GB (default 20): the backup directory."""
    return _limit("ACW_BACKUPS_ALERT_GB", 20)


def dir_bytes(path: Path, skip: tuple[str, ...] = ()) -> int:
    """Total size of the files under `path` (not following links); `skip` names top-level entries to leave out."""
    total = 0
    stack = [(path, True)]
    while stack:
        d, top = stack.pop()
        try:
            with os.scandir(d) as it:
                for e in it:
                    if top and e.name in skip:
                        continue
                    try:
                        if e.is_symlink():
                            continue
                        if e.is_dir(follow_symlinks=False):
                            stack.append((Path(e.path), False))
                        else:
                            total += e.stat(follow_symlinks=False).st_size
                    except OSError:
                        continue
        except OSError:
            continue
    return total


def sizes(data_dir: Path, backup_dir: Path) -> dict[str, int]:
    db_bytes = sum(p.stat().st_size for p in data_dir.glob("acw.sqlite3*") if p.is_file()) if data_dir.is_dir() else 0
    return {
        "data": dir_bytes(data_dir, skip=("jobs",)),
        "database": db_bytes,
        "jobs": dir_bytes(data_dir / "jobs"),
        "backups": dir_bytes(backup_dir),
    }
