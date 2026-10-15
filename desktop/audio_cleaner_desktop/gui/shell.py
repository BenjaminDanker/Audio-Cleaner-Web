"""Hand files and folders to the operating system."""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

from PySide6.QtCore import QDir, QFile, QUrl
from PySide6.QtGui import QDesktopServices

TRASH_NAME = "Recycle Bin" if sys.platform == "win32" else "Trash"


def move_to_trash(path: Path) -> bool:
    """Into the Recycle Bin / Trash (so it can be restored). False where the system has none."""
    return QFile.moveToTrash(str(path))[0]


def open_path(path: Path) -> bool:
    return QDesktopServices.openUrl(QUrl.fromLocalFile(str(path)))


def reveal_command(path: Path, platform: str | None = None) -> list[str] | None:
    """The command that shows `path` selected in the file manager, or None where that is not supported. On
    Windows the path is its own argument after "/select,": a single "/select,C:\\My Videos\\a.wav" argument
    gets quoted whole when the path has a space, and Explorer then selects nothing."""
    platform = platform or sys.platform
    if platform == "win32":
        return ["explorer", "/select,", QDir.toNativeSeparators(str(path))]
    if platform == "darwin":
        return ["open", "-R", str(path)]
    return None


def reveal_in_folder(path: Path) -> None:
    """Show `path` selected in the file manager where that is supported, else open its folder."""
    if path.exists():
        command = reveal_command(path)
        if command is not None:
            try:
                subprocess.Popen(command)
                return
            except OSError:
                pass
    folder = path if path.is_dir() else path.parent
    while not folder.exists() and folder != folder.parent:
        folder = folder.parent
    open_path(folder)
