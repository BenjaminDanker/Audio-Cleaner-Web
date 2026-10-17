"""The History tab: totals, search and one row per cleaned (or failed) file."""
from __future__ import annotations

from datetime import datetime
from pathlib import Path

from PySide6.QtCore import Qt, QTimer
from PySide6.QtWidgets import (
    QFrame, QHBoxLayout, QLabel, QLineEdit, QMessageBox, QPushButton, QScrollArea, QVBoxLayout, QWidget,
)

from .. import engine
from . import shell
from .history import DONE, Entry, HistoryStore, Totals
from .settings_panel import FORMAT_LABELS, LOUDNESS_SHORT, STRENGTH_LABELS, SUBTITLE_LABELS, VIDEO_SIZE_LABELS
from .theme import set_property
from .widgets import ElidedLabel, IconButton, format_time


def format_hours(seconds: float) -> str:
    if seconds < 3600:
        return f"{round(seconds / 60)} min" if seconds >= 60 else ("0 min" if seconds == 0 else "< 1 min")
    return f"{seconds / 3600:.1f} h"


def format_spent(seconds: float) -> str:
    minutes = round(seconds / 60)
    if minutes < 60:
        return f"{minutes} min" if minutes else ("0 min" if seconds == 0 else "< 1 min")
    return f"{minutes // 60} h {minutes % 60:02d} min"


def describe_settings(s: dict, has_video: bool) -> str:
    if s.get("rebuild_voice"):
        parts = ["voice rebuilt"]
    elif s.get("picked_rebuild"):
        parts = [f"voice rebuilt (it sounded {s['picked_rebuild']})"]
    else:
        picked = s.get("picked_strength") if s.get("strength") == engine.AUTO_STRENGTH else None
        if s.get("strength") == engine.AUTO_STRENGTH:
            parts = [f"{dict(STRENGTH_LABELS)[picked]} noise reduction (picked for this file)" if picked in engine.STRENGTHS
                     else "noise reduction picked for the file"]
        else:
            parts = [dict(STRENGTH_LABELS).get(s.get("strength"), "Maximum") + " noise reduction"]
    if s.get("noise", "auto") != "auto":
        parts.append(f"for {s['noise']} noise")
    if s.get("echo_track"):
        parts.append("speaker echo removed")
    if s.get("restore_highs") is False:
        parts.append("highs not restored")
    if s.get("leveling", True):
        parts.append("leveled (" + LOUDNESS_SHORT.get(s.get("loudness"), "podcast") + ")")
    if not has_video and s.get("audio_format", "same") != "same":
        parts.append(FORMAT_LABELS[s["audio_format"]])
    if s.get("video_size", "same") != "same":
        parts.append("video " + VIDEO_SIZE_LABELS.get(s["video_size"], s["video_size"] + "p"))
    if s.get("subtitles", "off") != "off":
        parts.append("subtitles: " + SUBTITLE_LABELS[s["subtitles"]].lower())
    return ", ".join(parts)


def settings_dict(options: engine.JobOptions) -> dict:
    return {
        "strength": options.strength, "leveling": options.leveling, "loudness": options.loudness,
        "audio_format": options.audio_format, "subtitles": options.subtitles, "language": options.language,
        "noise": options.noise, "echo_track": options.echo_track, "restore_highs": options.restore_highs,
        "rebuild_voice": options.rebuild_voice, "auto_rebuild": options.auto_rebuild, "video_size": options.video_size,
    }


PAGE_SIZE = 50  # rows drawn at first, and added by each "Show more"


class _Stat(QFrame):
    def __init__(self, caption: str, parent: QWidget | None = None):
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(2)
        self.value = QLabel("0")
        self.value.setObjectName("Heading")
        label = QLabel(caption)
        label.setObjectName("Small")
        layout.addWidget(self.value)
        layout.addWidget(label)


class _HistoryRow(QFrame):
    def __init__(self, entry: Entry, page: "HistoryPage"):
        super().__init__()
        self.entry = entry
        self.setObjectName("Row")
        outputs = [Path(p) for p in entry.outputs]
        layout = QHBoxLayout(self)
        layout.setContentsMargins(14, 10, 10, 10)
        layout.setSpacing(12)

        badge = QLabel("VIDEO" if entry.has_video else "AUDIO")
        badge.setObjectName("Badge")
        badge.setAlignment(Qt.AlignmentFlag.AlignCenter)
        badge.setFixedSize(48, 22)
        set_property(badge, "kind", "video" if entry.has_video else "audio")
        layout.addWidget(badge, 0, Qt.AlignmentFlag.AlignVCenter)

        middle = QVBoxLayout()
        middle.setSpacing(3)
        name = ElidedLabel(entry.name)
        name.setToolTip("\n".join([entry.source, *entry.outputs]))
        when = datetime.fromtimestamp(entry.finished_at).strftime("%d %b %Y, %H:%M")
        if entry.status == DONE:
            meta = f"{when}  ·  {format_time(entry.duration)}  ·  took {format_spent(entry.seconds_taken)}  ·  " \
                   + describe_settings(entry.settings, entry.has_video)
        else:
            meta = f"{when}  ·  {entry.message}"
        detail = ElidedLabel(meta)
        detail.setObjectName("Small")
        detail.setToolTip(meta)
        middle.addWidget(name)
        middle.addWidget(detail)
        layout.addLayout(middle, 1)

        status = QLabel("Done" if entry.status == DONE else "Failed")
        set_property(status, "state", "done" if entry.status == DONE else "failed")
        layout.addWidget(status)

        if entry.status == DONE and outputs:
            open_btn = IconButton("play", "Open the cleaned file with your default player")
            open_btn.clicked.connect(lambda: self._open(outputs[0]))
            reveal = IconButton("folder", "Show the cleaned file in its folder")
            reveal.clicked.connect(lambda: shell.reveal_in_folder(outputs[0]))
            for b in (open_btn, reveal):
                b.setFixedSize(32, 32)
                layout.addWidget(b)
        remove = IconButton("trash", "Remove this entry from the history (the files stay)")
        remove.setFixedSize(32, 32)
        remove.clicked.connect(lambda: page.delete_entry(entry.id))
        layout.addWidget(remove)

    def _open(self, path: Path) -> None:
        if not path.exists() or not shell.open_path(path):
            QMessageBox.information(self, "Open", "This file has been moved or deleted.")


class HistoryPage(QWidget):
    def __init__(self, store: HistoryStore, parent: QWidget | None = None):
        super().__init__(parent)
        self._store = store
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 16, 0, 0)
        layout.setSpacing(14)

        totals = QFrame()
        totals.setObjectName("Card")
        totals_row = QHBoxLayout(totals)
        totals_row.setContentsMargins(24, 16, 24, 16)
        self._stats = {k: _Stat(c) for k, c in (
            ("files", "Files cleaned"), ("audio", "Audio cleaned"), ("video", "Video cleaned"), ("spent", "Time spent cleaning"),
        )}
        for stat in self._stats.values():
            totals_row.addWidget(stat, 1)
        layout.addWidget(totals)

        bar = QHBoxLayout()
        self._search = QLineEdit()
        self._search.setPlaceholderText("Search by file name")
        self._search.setClearButtonEnabled(True)
        self._search.setToolTip("Type part of a file name to find it in the history")
        self._clear = QPushButton("Clear history")
        self._clear.setToolTip("Forget every entry below. Your files are not touched.")
        self._clear.clicked.connect(self._clear_all)
        bar.addWidget(self._search, 1)
        bar.addWidget(self._clear)
        layout.addLayout(bar)

        card = QFrame()
        card.setObjectName("Card")
        card_layout = QVBoxLayout(card)
        card_layout.setContentsMargins(0, 0, 0, 0)
        self._empty = QLabel()
        self._empty.setObjectName("Muted")
        self._empty.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._scroll = QScrollArea()
        self._scroll.setWidgetResizable(True)
        self._scroll.setFrameShape(QFrame.Shape.NoFrame)
        self._scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        card_layout.addWidget(self._empty, 1)
        card_layout.addWidget(self._scroll, 1)
        layout.addWidget(card, 1)

        self._debounce = QTimer(self)
        self._debounce.setSingleShot(True)
        self._debounce.setInterval(150)
        self._debounce.timeout.connect(self._search_changed)
        self._search.textChanged.connect(lambda _: self._debounce.start())
        self._shown = PAGE_SIZE
        self.refresh()

    def _search_changed(self) -> None:
        self._shown = PAGE_SIZE
        self.refresh()

    def _show_more(self) -> None:
        self._shown += PAGE_SIZE
        self.refresh()

    def refresh(self) -> None:
        """Rebuild the list: the newest PAGE_SIZE entries (more on request), not all of them at once."""
        totals = self._store.totals()
        self._show_totals(totals)
        found = self._store.search(self._search.text(), self._shown + 1)
        entries, more = found[: self._shown], len(found) > self._shown
        holder = QWidget()
        holder.setObjectName("RowHolder")
        rows = QVBoxLayout(holder)
        rows.setContentsMargins(8, 8, 8, 8)
        rows.setSpacing(2)
        for entry in entries:
            rows.addWidget(_HistoryRow(entry, self))
        if more:
            button = QPushButton("Show more")
            button.setToolTip("Show older entries")
            button.clicked.connect(self._show_more)
            rows.addWidget(button, 0, Qt.AlignmentFlag.AlignCenter)
        rows.addStretch(1)
        old = self._scroll.takeWidget()
        if old is not None:
            old.deleteLater()
        self._scroll.setWidget(holder)
        searching = bool(self._search.text().strip())
        self._scroll.setVisible(bool(entries))
        self._empty.setVisible(not entries)
        self._empty.setText(
            "Nothing matches your search." if searching else "Files you clean will be listed here.\nNothing leaves your computer."
        )
        self._clear.setEnabled(totals.files > 0 or bool(entries))

    def _show_totals(self, t: Totals) -> None:
        self._stats["files"].value.setText(str(t.files))
        self._stats["audio"].value.setText(format_hours(t.audio_seconds))
        self._stats["video"].value.setText(format_hours(t.video_seconds))
        self._stats["spent"].value.setText(format_spent(t.seconds_spent))

    def delete_entry(self, entry_id: int) -> None:
        self._store.delete([entry_id])
        self.refresh()

    def _clear_all(self) -> None:
        answer = QMessageBox.question(
            self, "Clear history", "Remove every entry from the history?\nYour cleaned files are not deleted.",
        )
        if answer == QMessageBox.StandardButton.Yes:
            self._store.clear()
            self.refresh()
