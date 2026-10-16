"""Start the application: logging, style, theme and the main window."""
from __future__ import annotations

import logging
import sys
import traceback
from logging.handlers import RotatingFileHandler
from pathlib import Path

from PySide6.QtCore import QDir, QEvent, Qt, Signal
from PySide6.QtGui import QGuiApplication, QIcon
from PySide6.QtWidgets import QApplication, QMessageBox

from .. import __version__, config, sysinfo
from . import single_instance, tips
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


class Application(QApplication):
    """Passes on the files the system hands to the running app: on a Mac, dropping files on the Dock icon,
    "Open With" and double-clicking a file arrive as an event, not on the command line."""

    fileOpened = Signal(str)

    def event(self, event: QEvent) -> bool:  # noqa: N802
        if event.type() == QEvent.Type.FileOpen:
            path = event.file()  # from a file:// URL: forward slashes also on Windows
            if path:
                self.fileOpened.emit(QDir.toNativeSeparators(path))
            return True
        return super().event(event)


def create_application(argv: list[str]) -> Application:
    QGuiApplication.setHighDpiScaleFactorRoundingPolicy(Qt.HighDpiScaleFactorRoundingPolicy.PassThrough)
    app = Application(argv)
    app.setOrganizationName(config.ORG_NAME)
    app.setOrganizationDomain(config.ORG_DOMAIN)
    app.setApplicationName(config.APP_NAME)
    app.setApplicationDisplayName(config.APP_NAME)
    app.setApplicationVersion(__version__)
    app.setWindowIcon(QIcon(str(ICON)))
    app.setStyle("Fusion")
    tips.install(app)
    return app


def bring_forward(window: MainWindow, paths: list[str]) -> None:
    """Show the window to the user and queue the files another start handed over."""
    if paths:
        window.add_paths(paths)
    if window.isMinimized():
        window.showNormal()
    window.raise_()
    window.activateWindow()
    QApplication.alert(window, 0)


def run(files: list[str]) -> int:
    app = create_application(sys.argv[:1])
    name = single_instance.server_name()
    if single_instance.forward(name, [str(Path(f).absolute()) for f in files]):
        return 0  # Audio Cleaner is already open: it got the files
    _install_excepthook(setup_logging())
    sysinfo.lower_priority()
    ThemeWatcher(app)
    window = MainWindow(files)
    server = single_instance.InstanceServer(name, app)
    if not server.listen():
        log.warning("could not listen for other starts of the app")
    server.received.connect(lambda paths: bring_forward(window, paths))
    app.fileOpened.connect(lambda path: bring_forward(window, [path]))
    window.show()
    window.updater.schedule_startup_check()  # frozen app, setting on: a quiet check ~10 s from now
    window.schedule_model_update()  # a model that changed since it was installed is fetched again, quietly
    return app.exec()
