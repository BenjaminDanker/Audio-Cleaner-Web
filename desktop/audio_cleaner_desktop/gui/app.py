"""Start the application: logging, style, theme and the main window."""
from __future__ import annotations

import logging
import sys
import traceback
from logging.handlers import RotatingFileHandler
from pathlib import Path

from PySide6.QtCore import Qt
from PySide6.QtGui import QGuiApplication, QIcon
from PySide6.QtWidgets import QApplication, QMessageBox

from .. import __version__, config
from . import tips
from .main_window import MainWindow, log_dir
from .theme import ThemeWatcher

ICON = Path(__file__).resolve().parent.parent / "assets" / "icon.png"
log = logging.getLogger("audio_cleaner_desktop")


def setup_logging() -> Path:
    path = log_dir() / "audio-cleaner.log"
    handler = RotatingFileHandler(path, maxBytes=1_000_000, backupCount=3, encoding="utf-8")
    handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s"))
    root = logging.getLogger()
    root.setLevel(logging.INFO)
    root.addHandler(handler)
    log.info("%s %s starting", config.APP_NAME, __version__)
    return path


def _install_excepthook(log_path: Path) -> None:
    def hook(exc_type, exc, tb) -> None:
        log.critical("unhandled exception", exc_info=(exc_type, exc, tb))
        if issubclass(exc_type, KeyboardInterrupt) or QApplication.instance() is None:
            return
        box = QMessageBox(
            QMessageBox.Icon.Critical, config.APP_NAME,
            "Something unexpected went wrong. Audio Cleaner may not work properly until you restart it.",
        )
        box.setInformativeText(f"Details were saved to:\n{log_path}")
        box.setDetailedText("".join(traceback.format_exception(exc_type, exc, tb)))
        box.exec()

    sys.excepthook = hook


def create_application(argv: list[str]) -> QApplication:
    QGuiApplication.setHighDpiScaleFactorRoundingPolicy(Qt.HighDpiScaleFactorRoundingPolicy.PassThrough)
    app = QApplication(argv)
    app.setOrganizationName(config.ORG_NAME)
    app.setOrganizationDomain(config.ORG_DOMAIN)
    app.setApplicationName(config.APP_NAME)
    app.setApplicationDisplayName(config.APP_NAME)
    app.setApplicationVersion(__version__)
    app.setWindowIcon(QIcon(str(ICON)))
    app.setStyle("Fusion")
    tips.install(app)
    return app


def run(files: list[str]) -> int:
    app = create_application(sys.argv[:1])
    _install_excepthook(setup_logging())
    ThemeWatcher(app)
    window = MainWindow(files)
    window.show()
    return app.exec()
