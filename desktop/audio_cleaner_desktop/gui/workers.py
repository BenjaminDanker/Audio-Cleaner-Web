"""Background work for the screen: `Worker` runs one function off the window's thread, with progress, a way to stop it and the outcome as signals."""
from __future__ import annotations

import logging
import threading
from collections.abc import Callable

from PySide6.QtCore import QThread, Signal

from acw_common.ffmpeg import MediaError

from .. import engine

log = logging.getLogger(__name__)


class Worker(QThread):
    """Runs `fn(cancel, progress)` off the window's thread; exactly one of done / failed / stopped follows."""

    done = Signal(object)
    failed = Signal(str)
    stopped = Signal()
    progress = Signal(float)

    def __init__(self, fn: Callable, parent=None):
        super().__init__(parent)
        self._fn = fn
        self.cancel = threading.Event()

    def run(self) -> None:
        try:
            result = self._fn(self.cancel, self.progress.emit)
        except engine.Cancelled:
            self.stopped.emit()
        except MediaError as e:
            self.failed.emit(str(e))
        except Exception as e:  # noqa: BLE001 - one bad file must not take the app down
            log.exception("background work failed")
            self.failed.emit("The disk is full." if getattr(e, "errno", None) == 28 else "Something went wrong; see the log.")
        else:
            self.done.emit(result)

    def stop(self) -> None:
        self.cancel.set()
