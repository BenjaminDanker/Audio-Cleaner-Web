"""Hand files and folders to the operating system."""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

from PySide6.QtCore import QUrl
from PySide6.QtGui import QDesktopServices


def open_path(path: Path) -> bool:
    return QDesktopServices.openUrl(QUrl.fromLocalFile(str(path)))


def reveal_in_folder(path: Path) -> None:
    """Show `path` selected in the file manager where that is supported, else open its folder."""
    if path.exists():
        try:
            if sys.platform == "win32":
                subprocess.Popen(["explorer", f"/select,{path}"])
                return
            if sys.platform == "darwin":
                subprocess.Popen(["open", "-R", str(path)])
                return
        except OSError:
            pass
    folder = path if path.is_dir() else path.parent
    while not folder.exists() and folder != folder.parent:
        folder = folder.parent
    open_path(folder)
