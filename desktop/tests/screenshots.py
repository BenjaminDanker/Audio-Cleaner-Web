"""Screenshots of the editor screen, taken offscreen with the real engine on a real recording (not collected by pytest):

    QT_QPA_PLATFORM=offscreen PYTHONPATH=desktop:common python desktop/tests/screenshots.py [OUT_DIR] [SPEECH_FILE]

The empty screen, a file loaded, a part chosen, a run in progress and the finished result at 1440 x 900, the same two states in a narrow window, and
the finished result in a dark palette. OUT_DIR defaults to ~/audio-cleaner/dev/desktop-shots, SPEECH_FILE to ~/audio-cleaner/dev/live_opus.wav."""
from __future__ import annotations

import os
import shutil
import sys
import tempfile
import threading
import time
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
ROOT = Path(__file__).resolve().parents[2]
for sub in ("desktop", "common"):
    sys.path.insert(0, str(ROOT / sub))


def main() -> None:
    out = Path(sys.argv[1]) if len(sys.argv) > 1 else Path.home() / "audio-cleaner" / "dev" / "desktop-shots"
    speech = Path(sys.argv[2]) if len(sys.argv) > 2 else Path.home() / "audio-cleaner" / "dev" / "live_opus.wav"
    out.mkdir(parents=True, exist_ok=True)

    from PySide6.QtCore import QSettings, QStandardPaths
    from PySide6.QtGui import QColor, QPalette

    QStandardPaths.setTestModeEnabled(True)
    from audio_cleaner_desktop import engine
    from audio_cleaner_desktop.gui.app import create_application
    from audio_cleaner_desktop.gui.main_window import MainWindow
    from audio_cleaner_desktop.gui.theme import ThemeWatcher

    app = create_application(["screenshots"])
    ThemeWatcher(app)
    tmp = Path(tempfile.mkdtemp(prefix="acw-shots-"))
    gate = threading.Event()
    hold = {"on": False}

    def runner(path, options, *, progress, **kw):
        def slowed(fraction: float) -> None:
            progress(fraction)
            if hold["on"] and fraction >= 0.45:
                gate.wait(60)  # (the screenshot of "a run in progress" is taken here)

        return engine.process(path, options, progress=slowed, **kw)

    w = MainWindow([], settings=QSettings(str(tmp / "s.ini"), QSettings.Format.IniFormat), history_path=tmp / "h.db", runner=runner)
    w.doc_env.auto_suggest = True  # (moments worth a look are looked for by themselves, as in the app)
    w.resize(1440, 900)
    w.show()

    def pump(seconds=0.4, until=None, limit=180.0) -> bool:
        end = time.time() + (limit if until else seconds)
        while time.time() < end:
            app.processEvents()
            time.sleep(0.01)
            if until and until():
                app.processEvents()
                time.sleep(0.15)
                app.processEvents()
                return True
        return not until

    def shot(name: str, widget=None, size=None) -> None:
        if size:
            w.resize(*size)
            pump(0.8)
        (widget or w).grab().save(str(out / f"{name}.png"))
        print("saved", out / f"{name}.png")

    pump(0.6)
    shot("1-empty-1440x900")
    src = tmp / "interview.wav"
    shutil.copy(speech, src)
    w.add_paths([src])
    doc = w.doc
    pump(until=lambda: doc.model is not None and doc.wave == "ready" and doc.applied and not doc.listening and doc.sg != "running")
    shot("2-file-loaded-1440x900")
    doc.model.select(3.0, 7.0)
    pump(0.5)
    shot("3-part-chosen-1440x900")
    doc.model.clear_selection()
    pump(0.3)
    hold["on"] = True
    w.studio.start.button.click()
    pump(until=lambda: doc.run_fraction is not None and doc.run_fraction >= 0.45)
    pump(0.8)
    shot("4-run-in-progress-1440x900")
    hold["on"] = False
    gate.set()
    pump(until=lambda: doc.stage == "done", limit=240)
    pump(until=lambda: doc.heard is not None and doc.session_state == "ready", limit=60)
    pump(0.6)
    shot("5-finished-1440x900")
    # a second file next to it (jobs column with a waiting file and a finished job)
    src2 = tmp / "second talk.wav"
    shutil.copy(speech, src2)
    w.add_paths([src2])
    pump(until=lambda: w.doc.model is not None and w.doc.wave == "ready" and w.doc.applied and not w.doc.listening)
    shot("6-second-file-and-a-finished-job-1440x900")
    # narrow: the settings and the jobs go under the editor (the whole page is taken, tall)
    w.open_entry(doc.entry.id)
    pump(until=lambda: w.doc.session_state == "ready", limit=60)
    w.resize(760, 2300)
    pump(1.0)
    shot("7-narrow-finished-760")
    w.open_item(w._items[2].id)
    pump(0.6)
    shot("8-narrow-file-loaded-760")
    w.resize(1440, 900)
    pump(0.6)
    # the empty screen in a narrow window
    w._show(None)
    w.resize(760, 1500)
    pump(0.8)
    shot("9-narrow-empty-760")
    # a dark palette
    w.resize(1440, 900)
    dark = QPalette()
    for role, color in ((QPalette.ColorRole.Window, "#121826"), (QPalette.ColorRole.WindowText, "#e7ecf5"), (QPalette.ColorRole.Base, "#151b26"),
                        (QPalette.ColorRole.AlternateBase, "#1c2434"), (QPalette.ColorRole.Text, "#e7ecf5"), (QPalette.ColorRole.Button, "#1c2434"),
                        (QPalette.ColorRole.ButtonText, "#e7ecf5"), (QPalette.ColorRole.ToolTipBase, "#1c2434"), (QPalette.ColorRole.ToolTipText, "#e7ecf5"),
                        (QPalette.ColorRole.PlaceholderText, "#a0abc0")):
        dark.setColor(role, QColor(color))
    app.setPalette(dark)
    pump(0.6)
    w.open_entry(doc.entry.id)
    pump(until=lambda: w.doc.session_state == "ready", limit=60)
    pump(0.8)
    shot("10-finished-dark-1440x900")
    w._closing = True
    w.queue.shutdown()
    w._show(None)
    for item in list(w._items.values()):
        if item.doc is not None:
            item.doc.dispose()
    w.history.close()
    shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    main()
