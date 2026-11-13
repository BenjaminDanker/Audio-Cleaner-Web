"""A file the window knows about: added, waiting for its turn, running, or finished. (The queue itself is worker.py; the screen is studio.py.)"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from .. import engine

WAITING, RUNNING, DONE, FAILED, CANCELLED = "waiting", "running", "done", "failed", "cancelled"


def format_size(num_bytes: int) -> str:
    size = float(num_bytes)
    for unit in ("B", "KB", "MB", "GB"):
        if size < 1000 or unit == "GB":
            return f"{size:.0f} {unit}" if unit == "B" else f"{size:.1f} {unit}"
        size /= 1000
    return ""


@dataclass
class FileItem:
    id: int
    path: Path
    state: str = WAITING
    progress: float = 0.0
    duration: float = 0.0
    has_video: bool = False
    channels: int = 2
    message: str = ""
    outputs: list[Path] = field(default_factory=list)
    started_at: float = 0.0
    probed: bool = False
    options: engine.JobOptions | None = None
    picked_strength: str | None = None  # what "Automatic" picked for the last run (engine.JobResult.strength)
    picked_rebuild: str | None = None  # ... and why it rebuilt the voice (engine.JobResult.rebuilt)
    stopping: bool = False  # asked to stop; the file finishes its current step first
    restart: bool = False  # stopping in order to start again with the current choices
    # What the result is called: the text of the name box (the ending says the type). While `out_custom` is False it is the usual name and
    # follows the choices; typing another name makes it the user's.
    source: engine.Source | None = None
    out_name: str = ""
    out_custom: bool = False
    job_name: str = ""  # the name asked for in the job that is running (or ran last)
    keep_dir: Path | None = None  # the folder the saved copies of the run in line (or running) go to
    timeline: dict | None = None  # what to do to which part (None: the whole file with the options as they are)
    entry_id: int = 0  # the job list's entry once the run is recorded
    start_when_read: bool = False  # Start all was pressed while the file was still being read: it starts once it is
    doc: object | None = None  # the editor's state for this file (doc.Doc)
