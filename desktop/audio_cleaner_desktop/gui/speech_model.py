"""The one-time download of a model (the noise removal models, the speech model for subtitles, the one that restores
highs, the one that rebuilds the voice): a background download with progress, and the line under the setting that offers it, shows
it running, or says what went wrong."""
from __future__ import annotations

import logging
import threading

from PySide6.QtCore import QObject, Signal
from PySide6.QtWidgets import QHBoxLayout, QLabel, QProgressBar, QPushButton, QVBoxLayout, QWidget

from .. import models, resources
from .theme import set_property

log = logging.getLogger(__name__)


def size_text(num_bytes: int) -> str:
    return f"{num_bytes / 1e9:.1f} GB" if num_bytes >= 1e9 else f"{num_bytes / 1e6:.0f} MB"


class SpeechModelDownloader(QObject):
    """Runs models.download() on a worker thread; the signals arrive on the GUI thread."""

    progress = Signal(object, object)  # bytes done, bytes total (Python ints: sizes exceed 32 bits' comfort)
    finished = Signal(bool, str)  # ok, message for the user ("" when ok or cancelled)
    changed = Signal()  # started, finished or cancelled: anything showing the state should refresh

    def __init__(self, parent: QObject | None = None, model: models.SpeechModel | None = None):
        super().__init__(parent)
        self.model = model or resources.SPEECH_MODEL  # any catalog model
        self._thread: threading.Thread | None = None
        self._cancel = threading.Event()
        self.last_error = ""

    @property
    def running(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    @property
    def ready(self) -> bool:
        if self.model is resources.SPEECH_MODEL:
            return resources.speech_ready()  # also counts the test model when run from source
        if self.model is resources.NOISE_MODEL:
            return resources.noise_ready()  # also counts vendor/models and the server's models when run from source
        if self.model is resources.HIGHS_MODEL:
            return resources.highs_model_dir() is not None
        if self.model is resources.VOICE_MODEL:
            return resources.voice_model_dir() is not None
        return models.is_installed(self.model)

    def start(self) -> None:
        if self.running or models.is_installed(self.model):
            return
        self._cancel.clear()
        self.last_error = ""
        self._thread = threading.Thread(target=self._run, name="speech-model-download", daemon=True)
        self._thread.start()
        self.changed.emit()

    def cancel(self) -> None:
        self._cancel.set()

    def wait(self, timeout: float | None = None) -> None:
        if self._thread is not None:
            self._thread.join(timeout)

    def _run(self) -> None:
        try:
            models.download(self.model, progress=lambda d, t: self.progress.emit(d, t), cancel=self._cancel)
        except models.Cancelled:
            ok, message = False, ""
        except models.DownloadError as e:
            ok, message = False, str(e)
        except Exception as e:  # noqa: BLE001 - anything else (disk full mid-way, permissions): say it, log it
            log.exception("speech model download failed")
            ok, message = False, f"The download failed: {e}"
        else:
            ok, message = True, ""
        self.last_error = message
        self.finished.emit(ok, message)
        self.changed.emit()


class SpeechModelPanel(QWidget):
    """Under the Subtitles setting: what is needed, the download while it runs, or an error."""

    def __init__(self, downloader: SpeechModelDownloader, parent: QWidget | None = None, *,
                 need: str = "Subtitles need the speech model", what: str = "the speech model"):
        super().__init__(parent)
        self._dl = downloader
        self._need, self._what = need, what
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 2, 0, 0)
        layout.setSpacing(6)
        self._text = QLabel()
        self._text.setObjectName("Small")
        self._text.setWordWrap(True)
        layout.addWidget(self._text)
        row = QHBoxLayout()
        row.setSpacing(8)
        self._bar = QProgressBar()
        self._bar.setRange(0, 1000)
        self._bar.setTextVisible(False)
        self._button = QPushButton()
        self._button.clicked.connect(self._clicked)
        row.addWidget(self._bar, 1)
        row.addWidget(self._button)
        layout.addLayout(row)
        downloader.progress.connect(self._progress)
        downloader.changed.connect(self.refresh)
        self.refresh()

    def showEvent(self, event) -> None:  # noqa: N802
        self.refresh()  # the model may have been installed meanwhile (by setup, or the download queue)
        super().showEvent(event)

    def refresh(self) -> None:
        dl = self._dl
        total = dl.model.size
        if dl.running:
            self._button.setText("Cancel")
            self._button.setToolTip("Stop the download. Starting it again continues where it left off.")
            self._bar.show()
            if not self._text.text().startswith("Downloading"):
                self._text.setText(f"Downloading {self._what} ({size_text(total)})...")
            set_property(self._text, "state", "")
        elif dl.last_error:
            self._text.setText(dl.last_error)
            set_property(self._text, "state", "failed")
            self._button.setText("Try again")
            self._button.setToolTip("Continue the download where it stopped")
            self._bar.hide()
        else:
            left = models.missing_bytes(dl.model)
            more = f" ({size_text(left)} left)" if 0 < left < total else ""
            self._text.setText(
                f"{self._need}: a one-time download of {size_text(total)}{more}. After that everything works offline."
            )
            set_property(self._text, "state", "")
            self._button.setText("Download")
            self._button.setToolTip(f"Download {self._what} now")
            self._bar.hide()

    def _clicked(self) -> None:
        if self._dl.running:
            self._dl.cancel()
        else:
            self._dl.start()

    def _progress(self, done: int, total: int) -> None:
        self._bar.setValue(int(1000 * done / max(1, total)))
        self._text.setText(f"Downloading {self._what}: {size_text(done)} of {size_text(total)}")
