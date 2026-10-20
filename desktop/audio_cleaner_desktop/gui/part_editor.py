"""The per-part editor ("Fix a moment"): the strip under the player with the suggestion pins, "Hear" chips for each layer, and the
Moment -> Part -> Change panel; "Show all layers" turns the strip into one row per layer. It is used in two places: in the job view (the
player plays the job's live sound; the bar at the bottom processes and saves the changes) and, before a file is cleaned, in plan mode (the
parts to process are chosen; the bar says how long that takes).

This widget draws and edits an `editor_model.EditModel` and tells its owner what to do through signals; it plays, processes and saves
nothing itself.
"""
from __future__ import annotations

from PySide6.QtCore import QTimer, Qt, Signal
from PySide6.QtGui import QColor
from PySide6.QtWidgets import (
    QComboBox, QFrame, QHBoxLayout, QLabel, QProgressBar, QPushButton, QStackedWidget, QToolButton, QVBoxLayout, QWidget,
)

from .. import editor_job, labels
from ..editor_model import (
    LAYER_NAMES, LAYER_ORDER, REMOVERS, STRENGTH_HINTS, STRENGTHS, EditModel, clock, span_text,
)
from .theme import set_property
from .wave_strip import LAYER_COLORS, WaveStrip
from .widgets import Segmented, WrapCheck

STEPS = (("home", "Moment"), ("range", "Part"), ("change", "Change"))
LOUDNESS_CHOICES = [(v, labels.LOUDNESS_LABELS[k]) for k, v in (("podcast", -16.0), ("youtube", -14.0), ("broadcast", -23.0), ("keep", None))]
DEBOUNCE_MS = 120


def tint(color: str, alpha: float) -> str:
    c = QColor(color)
    return f"rgba({c.red()}, {c.green()}, {c.blue()}, {alpha})"


def bar_text(model: EditModel, quote: "editor_job.Quote | None") -> tuple[str, str, bool]:
    """(the bar's heading, its line under it, whether Save works) for the changes in the draft."""
    if model.plan_mode:
        total = model.seconds
        if model.only and not model.chosen:
            return "Nothing will be processed", "Pick a part, then choose Only this part.", False
        if model.only:
            secs = model.processed_seconds()
            head = f"Chosen parts · {editor_job.say_time(quote.seconds)}" if quote else "Chosen parts"
            return head, f"Processes {secs:.0f} s of {clock(total)}, with a little sound around each part.", True
        head = f"Whole file · {editor_job.say_time(quote.seconds)}" if quote else "Whole file"
        return head, "Every layer you have on is on for the whole file. Choose parts to process less.", True
    if not model.dirty:
        return "No changes yet", "Pick a part to change it. Strength changes and the Hear switches are heard right away.", False
    if quote is None:
        return "Process changes", "Saves a new file with these changes.", True
    if quote.nothing:
        extra = f" Leveling runs again ({editor_job.say_time(quote.seconds)})." if quote.seconds > 10 else ""
        return "Changes: nothing to process", "Uses what is already processed." + extra, True
    steps = quote.steps or {}
    names = {"light": "noise removal", "heavy": "noise removal", "rebuild": "Voice rebuild", "echo": "echo removal"}
    what = ", ".join(sorted({names[s] for s in steps if s in names}))
    return f"Process changes · {editor_job.say_time(quote.seconds)}", f"Makes the parts that are not ready yet ({what}), then saves a new file." if what else "", True


class _Card(QFrame):
    def __init__(self, focus: bool = False, parent: QWidget | None = None):
        super().__init__(parent)
        self.setObjectName("Sugg")
        set_property(self, "focus", focus)


class PartEditor(QWidget):
    hearChanged = Signal(str, bool)
    seekRequested = Signal(float)
    playPartRequested = Signal(float, float)
    draftChanged = Signal()  # the draft was edited (debounced): the owner updates what is heard and the quote
    saveRequested = Signal()
    stopRequested = Signal()
    findRequested = Signal()  # "Find moments to fix"
    suggestionsChanged = Signal()  # one was applied, undone or dismissed (the owner keeps it)

    def __init__(self, model: EditModel, peaks: list[int] | None = None, parent: QWidget | None = None):
        super().__init__(parent)
        self.model = model
        self.view = "home"
        self._playhead = 0.0
        self._building = False
        self._finding: float | None = None
        self._find_text = ""
        self._can_find = True
        self._rebuild_available = True
        self._debounce = QTimer(self, singleShot=True, interval=DEBOUNCE_MS, timeout=self.draftChanged.emit)

        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(10)
        grid = QHBoxLayout()
        grid.setSpacing(18)
        outer.addLayout(grid, 1)

        # -- the left: whatever the owner puts there (the player), the strip, the Hear chips
        self.left = QVBoxLayout()
        self.left.setSpacing(8)
        self.slot = QVBoxLayout()
        self.left.addLayout(self.slot)
        self.strip = WaveStrip(model, peaks)
        self.left.addWidget(self.strip)
        chips = QHBoxLayout()
        chips.setSpacing(6)
        self._hear_label = QLabel("Hear:")
        self._hear_label.setObjectName("Muted")
        chips.addWidget(self._hear_label)
        self._chips: dict[str, QPushButton] = {}
        for layer in LAYER_ORDER:
            chip = QPushButton(LAYER_NAMES[layer])
            chip.setVisible(layer != "echo" or model.echo_available)
            chip.setCheckable(True)
            chip.setChecked(True)
            chip.setToolTip(f"Hear the {LAYER_NAMES[layer].lower()} in what plays. This only changes what you hear, not what is saved.")
            chip.setAccessibleName(f"Hear {LAYER_NAMES[layer]}")
            chip.toggled.connect(lambda on, k=layer: self._hear_toggled(k, on))
            chips.addWidget(chip)
            self._chips[layer] = chip
        chips.addStretch(1)
        self._rows = QPushButton("Show all layers")
        self._rows.setCheckable(True)
        self._rows.setToolTip("One row for each layer, with all its parts. Drag across a row to choose a part.")
        self._rows.toggled.connect(self._rows_toggled)
        self.chips_row = QWidget()
        self.chips_row.setLayout(chips)
        chips.setContentsMargins(0, 0, 0, 0)
        self.left.addWidget(self.chips_row)
        rows_line = QHBoxLayout()
        rows_line.addWidget(self._rows)
        rows_line.addStretch(1)
        self.left.addLayout(rows_line)
        self._heard_note = QLabel()
        self._heard_note.setObjectName("Small")
        self._heard_note.setWordWrap(True)
        self.left.addWidget(self._heard_note)
        self.left.addStretch(1)
        left_box = QWidget()
        left_box.setLayout(self.left)
        self.left.setContentsMargins(0, 0, 0, 0)
        grid.addWidget(left_box, 11)

        # -- the right: the three steps
        self.panel = QFrame()
        self.panel.setObjectName("Card")
        pl = QVBoxLayout(self.panel)
        pl.setContentsMargins(14, 12, 14, 14)
        pl.setSpacing(8)
        steps = QHBoxLayout()
        self._step_buttons: dict[str, QPushButton] = {}
        for i, (key, name) in enumerate(STEPS, 1):
            b = QPushButton(f"{i}  {name}")
            b.setObjectName("Flat")
            b.clicked.connect(lambda _=False, k=key: self.goto(k))
            steps.addWidget(b)
            self._step_buttons[key] = b
        pl.addLayout(steps)
        self.pages = QStackedWidget()
        pl.addWidget(self.pages, 1)
        self._home = QWidget()
        self._home_layout = QVBoxLayout(self._home)
        self._home_layout.setContentsMargins(0, 0, 0, 0)
        self._home_layout.setSpacing(8)
        self._range = QWidget()
        self._build_range()
        self._change = QWidget()
        self._build_change()
        for page in (self._home, self._range, self._change):
            self.pages.addWidget(page)
        grid.addWidget(self.panel, 10)

        # -- the bar at the bottom
        self.bar = QFrame()
        self.bar.setObjectName("Card")
        bl = QHBoxLayout(self.bar)
        bl.setContentsMargins(16, 10, 16, 10)
        text = QVBoxLayout()
        text.setSpacing(1)
        self.bar_title = QLabel()
        self.bar_title.setObjectName("CardTitle")
        self.bar_detail = QLabel()
        self.bar_detail.setObjectName("Small")
        self.bar_detail.setWordWrap(True)
        self.progress = QProgressBar()
        self.progress.setRange(0, 1000)
        self.progress.setTextVisible(False)
        self.progress.hide()
        text.addWidget(self.bar_title)
        text.addWidget(self.bar_detail)
        text.addWidget(self.progress)
        bl.addLayout(text, 1)
        self.stop_button = QPushButton("Stop")
        self.stop_button.clicked.connect(self.stopRequested)
        self.stop_button.hide()
        self.save_button = QPushButton("Save")
        self.save_button.setObjectName("Primary")
        self.save_button.clicked.connect(self.saveRequested)
        bl.addWidget(self.stop_button)
        bl.addWidget(self.save_button)
        outer.addWidget(self.bar)

        # -- wiring
        model.on_change(self._model_changed)
        self.strip.seekRequested.connect(self._strip_seek)
        self.strip.selectionDragged.connect(self._strip_selected)
        self.strip.pinClicked.connect(self._pin_clicked)
        if model.plan_mode:
            self.chips_row.hide()
            self.save_button.hide()
            self._heard_note.setText("Nothing is processed yet, so you hear the recording as it is. Choose parts, then clean the file.")
        else:
            self.save_button.setText("Save")
        self._restyle()
        self._refresh()

    # -- plan mode and what the owner can switch -------------------------------------------------------------------------
    def set_hear_available(self, on: bool) -> None:
        """The chips only mean something while there is live sound (a job with its blocks laid out)."""
        self.chips_row.setVisible(on and not self.model.plan_mode)

    def set_rebuild_available(self, on: bool) -> None:
        self._rebuild_available = on
        self._refresh()

    def set_playhead(self, seconds: float) -> None:
        self._playhead = seconds
        self.strip.set_playhead(seconds)

    def set_peaks(self, peaks: list[int]) -> None:
        self.strip.set_peaks(peaks)

    def set_heard_note(self, text: str) -> None:
        self._heard_note.setText(text)
        self._heard_note.setVisible(bool(text))

    def set_finding(self, fraction: float | None, text: str = "", can: bool = True) -> None:
        """Suggestions are being looked for (a fraction), or not (None); `can`: there is a button to ask for them."""
        self._finding, self._find_text, self._can_find = fraction, text, can
        self._refresh_home()

    def show_quote(self, quote: "editor_job.Quote | None") -> None:
        title, detail, can = bar_text(self.model, quote)
        self.bar_title.setText(title)
        self.bar_detail.setText(detail)
        self.save_button.setEnabled(can and not self.progress.isVisible())

    def set_busy(self, busy: bool, fraction: float | None = None) -> None:
        """A save is running: the bar shows its progress and a Stop button."""
        self.progress.setVisible(busy)
        self.stop_button.setVisible(busy)
        self.stop_button.setEnabled(True)
        self.save_button.setText("Saving..." if busy else "Save")
        if busy:
            self.save_button.setEnabled(False)
            self.progress.setValue(int((fraction or 0.0) * 1000))
        for w in (self.panel, self.strip, self.chips_row):
            w.setEnabled(not busy)

    # -- the steps --------------------------------------------------------------------------------------------------------
    def goto(self, view: str) -> None:
        if view != "home" and self.model.selection is None:
            self.model.around(self._playhead)  # (this refreshes through the model)
        self.view = view
        self._refresh()

    def _restyle(self) -> None:
        for layer, chip in self._chips.items():
            c = LAYER_COLORS[layer]
            chip.setStyleSheet(
                f"QPushButton {{ border: 1.5px solid {c}; border-radius: 13px; padding: 3px 10px; }}"
                f"QPushButton:checked {{ background: {tint(c, 0.2)}; }}")

    def _hear_toggled(self, layer: str, on: bool) -> None:
        self.strip.set_hear({k: b.isChecked() for k, b in self._chips.items()})
        self.hearChanged.emit(layer, on)

    def _rows_toggled(self, on: bool) -> None:
        self.strip.set_rows(on)
        self._rows.setText("Show fewer layers" if on else "Show all layers")

    def _strip_seek(self, t: float) -> None:
        self.set_playhead(t)
        self.seekRequested.emit(t)

    def _strip_selected(self, a: float, b: float) -> None:
        self.model.focus = None
        self.model.select(a, b)
        if self.view == "home":
            self.view = "range"
            self._refresh()

    def _pin_clicked(self, sid: str) -> None:
        self.model.show(sid)
        self.view = "change"
        self.seekRequested.emit(self.model.suggestion(sid).start)
        self._refresh()

    def _model_changed(self) -> None:
        if self._building:
            return
        self._refresh()
        self._debounce.start()

    # -- the home page --------------------------------------------------------------------------------------------------------
    def _refresh(self) -> None:
        if self.model.selection is None and self.view != "home":
            self.view = "home"
        self.pages.setCurrentWidget({"home": self._home, "range": self._range, "change": self._change}[self.view])
        for key, b in self._step_buttons.items():
            set_property(b, "on", key == self.view)
        self.strip.update()
        if self.view == "home":
            self._refresh_home()
        elif self.view == "range":
            self._refresh_range()
        else:
            self._refresh_change()

    def _refresh_home(self) -> None:
        lay = self._home_layout
        while lay.count():
            item = lay.takeAt(0)
            if item.widget() is not None:
                item.widget().hide()
                item.widget().deleteLater()
            elif item.layout() is not None:
                _clear(item.layout())
        model = self.model
        shown = [s for s in model.suggestions if s.state != "dismissed"]
        head = QHBoxLayout()
        title = QLabel("We listened to your file" if model.plan_mode else "Moments worth a look")
        title.setObjectName("CardTitle")
        head.addWidget(title)
        if model.open_suggestions():
            badge = QLabel(f"{len(model.open_suggestions())} new")
            badge.setObjectName("Badge")
            badge.setFixedHeight(20)
            head.addWidget(badge)
        head.addStretch(1)
        lay.addLayout(head)
        for s in shown:
            lay.addWidget(self._suggestion_card(s))
        if self._finding is not None:
            bar = QProgressBar()
            bar.setRange(0, 1000)
            bar.setValue(int(self._finding * 1000))
            bar.setTextVisible(False)
            note = QLabel(self._find_text or "Listening to your file for moments that need other settings...")
            note.setObjectName("Small")
            note.setWordWrap(True)
            stop = QPushButton("Stop")
            stop.clicked.connect(self.stopRequested)
            row = QHBoxLayout()
            row.addWidget(bar, 1)
            row.addWidget(stop)
            lay.addWidget(note)
            lay.addLayout(row)
        elif not shown:
            note = QLabel(self._find_text or ("No moments found that need other settings." if model.suggestions else
                                              "Nothing has been looked for yet."))
            note.setObjectName("Small")
            note.setWordWrap(True)
            lay.addWidget(note)
        if self._finding is None and self._can_find:
            find = QPushButton("Find moments to fix")
            find.setToolTip("Listens to the whole file for parts that sound different (more noise, distortion, quiet) and suggests settings "
                            "for them. On a long recording this takes a while; you can stop it.")
            find.clicked.connect(self.findRequested)
            lay.addWidget(find, 0, Qt.AlignmentFlag.AlignLeft)
        if model.plan_mode:
            seg = Segmented([("whole", "Whole file"), ("only", "Only parts")])
            seg.set_value("only" if model.only else "whole")
            seg.setToolTip("Process the whole file, or only the parts you choose")
            seg.changed.connect(lambda k: model.set_only(k == "only") if k == "whole" or model.chosen else self._need_part())
            row = QHBoxLayout()
            lab = QLabel("What to process")
            lab.setObjectName("CardTitle")
            row.addWidget(lab)
            row.addStretch(1)
            lay.addLayout(row)
            lay.addWidget(seg)
            for i, (a, b) in enumerate(model.chosen):
                chosen = QHBoxLayout()
                chosen.addWidget(QLabel(f"Chosen: {span_text(a, b)}"))
                chosen.addStretch(1)
                drop = QPushButton("Remove")
                drop.setObjectName("Flat")
                drop.clicked.connect(lambda _=False, k=i: model.unchoose(k))
                chosen.addWidget(drop)
                lay.addLayout(chosen)
        layers = QLabel("Layers")
        layers.setObjectName("CardTitle")
        lay.addWidget(layers)
        for layer in LAYER_ORDER:
            if layer == "echo" and not model.echo_available:
                continue
            row = QHBoxLayout()
            name = QLabel(f"<span style='color:{LAYER_COLORS[layer]}'>■</span> <b>{LAYER_NAMES[layer]}</b>")
            name.setMinimumWidth(130)
            row.addWidget(name)
            runs = model.runs(layer)
            if not runs:
                off = QLabel("Off")
                off.setObjectName("Muted")
                row.addWidget(off)
            for r in runs[:6]:
                whole = r["start"] <= 0.01 and r["end"] >= model.seconds - 0.01
                text = (r["label"] + " · " if r["label"] else "") + ("whole file" if whole else span_text(r["start"], r["end"]))
                chip = QPushButton(text)
                chip.setToolTip("Change this part")
                chip.setStyleSheet(f"QPushButton {{ border: 1.5px {'dashed' if r['pending'] else 'solid'} {LAYER_COLORS[layer]}; "
                                   f"border-radius: 12px; padding: 2px 10px; background: {tint(LAYER_COLORS[layer], 0.1)}; }}")
                chip.clicked.connect(lambda _=False, a=r["start"], b=r["end"]: self._chip_clicked(a, b))
                row.addWidget(chip)
            if len(runs) > 6:
                more = QLabel(f"+{len(runs) - 6} more")
                more.setObjectName("Muted")
                row.addWidget(more)
            row.addStretch(1)
            lay.addLayout(row)
        hint = QLabel("Every layer starts on the whole file. Pick a part to change just that.")
        hint.setObjectName("Small")
        hint.setWordWrap(True)
        lay.addWidget(hint)
        pick = QPushButton("Pick a part myself")
        pick.clicked.connect(lambda: self.goto("range"))
        lay.addWidget(pick, 0, Qt.AlignmentFlag.AlignLeft)
        lay.addStretch(1)

    def _need_part(self) -> None:
        self.model.set_only(False)
        self._heard_note.setText("Pick a part first, then choose Only this part.")

    def _chip_clicked(self, a: float, b: float) -> None:
        self.model.focus = None
        self.model.select(a, b)
        self.view = "change"
        self._refresh()

    def _suggestion_card(self, s) -> QWidget:
        card = _Card(focus=self.model.focus == s.id)
        v = QVBoxLayout(card)
        v.setContentsMargins(10, 8, 10, 8)
        v.setSpacing(5)
        if s.state == "applied":
            line = QLabel(f"<b>{span_text(s.start, s.end)}</b> done: {s.action.lower()}.")
            line.setWordWrap(True)
            undo = QPushButton("Undo")
            undo.setObjectName("Flat")
            undo.clicked.connect(lambda: self._sug(s.id, "undo"))
            row = QHBoxLayout()
            row.addWidget(line, 1)
            row.addWidget(undo)
            v.addLayout(row)
            return card
        line = QLabel(f"<b>{span_text(s.start, s.end)}</b> {s.what}. {s.ask}")
        line.setWordWrap(True)
        v.addWidget(line)
        row = QHBoxLayout()
        row.setSpacing(6)
        for text, act, primary in ((s.action, "apply", True), ("Show me", "show", False), ("Dismiss", "dismiss", False)):
            b = QPushButton(text)
            b.setObjectName("Primary" if primary else "Flat" if act == "dismiss" else "")
            b.clicked.connect(lambda _=False, a=act: self._sug(s.id, a))
            row.addWidget(b)
        row.addStretch(1)
        v.addLayout(row)
        same_remover = bool(s.noise) and self.model.remover_of(s.start, s.end) == s.noise.get("remover")
        note = QLabel("You hear it right away" if same_remover and not s.rebuild else "Needs processing")
        note.setObjectName("Small")
        v.addWidget(note)
        return card

    def _sug(self, sid: str, act: str) -> None:
        m = self.model
        if act == "apply":
            m.apply(sid)
        elif act == "undo":
            m.undo(sid)
        elif act == "dismiss":
            m.dismiss(sid)
        else:
            m.show(sid)
            self.seekRequested.emit(m.suggestion(sid).start)
            self.view = "change"
            self._refresh()
            return
        self.suggestionsChanged.emit()

    # -- the Part page ----------------------------------------------------------------------------------------------------------
    def _build_range(self) -> None:
        v = QVBoxLayout(self._range)
        v.setContentsMargins(0, 0, 0, 0)
        v.setSpacing(8)
        self._range_title = QLabel()
        self._range_title.setObjectName("CardTitle")
        v.addWidget(self._range_title)
        hint = QLabel("Drag across the waveform or the handles above, or set the start and end here.")
        hint.setObjectName("Small")
        hint.setWordWrap(True)
        v.addWidget(hint)
        row = QHBoxLayout()
        self._time_labels: dict[str, QLabel] = {}
        for which, name in (("s", "Start"), ("e", "End")):
            box = QFrame()
            box.setObjectName("Sugg")
            bv = QVBoxLayout(box)
            bv.setContentsMargins(8, 6, 8, 6)
            cap = QLabel(name)
            cap.setObjectName("Small")
            bv.addWidget(cap)
            line = QHBoxLayout()
            minus = QPushButton("−")
            minus.setToolTip(f"{name} one second earlier")
            minus.setAccessibleName(f"{name} one second earlier")
            minus.setFixedWidth(34)
            minus.clicked.connect(lambda _=False, w=which: self.model.nudge(w, -1.0))
            plus = QPushButton("+")
            plus.setToolTip(f"{name} one second later")
            plus.setAccessibleName(f"{name} one second later")
            plus.setFixedWidth(34)
            plus.clicked.connect(lambda _=False, w=which: self.model.nudge(w, 1.0))
            label = QLabel()
            label.setAlignment(Qt.AlignmentFlag.AlignCenter)
            f = label.font()
            f.setBold(True)
            f.setPointSizeF(f.pointSizeF() * 1.3)
            label.setFont(f)
            self._time_labels[which] = label
            line.addWidget(minus)
            line.addWidget(label, 1)
            line.addWidget(plus)
            bv.addLayout(line)
            use = QPushButton("Use playhead")
            use.setObjectName("Flat")
            use.clicked.connect(lambda _=False, w=which: self.model.use_playhead(w, self._playhead))
            bv.addWidget(use)
            row.addWidget(box, 1)
        v.addLayout(row)
        row = QHBoxLayout()
        around = QPushButton("Around playhead")
        around.clicked.connect(lambda: self.model.around(self._playhead))
        whole = QPushButton("Whole file")
        whole.clicked.connect(self.model.whole)
        play = QPushButton("Play part")
        play.clicked.connect(self._play_part)
        for b in (around, whole, play):
            row.addWidget(b)
        self._only_button = QPushButton("Only this part")
        self._only_button.setToolTip("Process only this part (and the other parts you choose) when the file is cleaned")
        self._only_button.clicked.connect(lambda: (self.model.choose(), self.goto("home")))
        row.addWidget(self._only_button)
        row.addStretch(1)
        v.addLayout(row)
        v.addStretch(1)
        nav = QHBoxLayout()
        cancel = QPushButton("Cancel")
        cancel.setObjectName("Flat")
        cancel.clicked.connect(lambda: (self.model.clear_selection(), self.goto("home")))
        nxt = QPushButton("Next: what to change")
        nxt.setObjectName("Primary")
        nxt.clicked.connect(lambda: self.goto("change"))
        nav.addWidget(cancel)
        nav.addStretch(1)
        nav.addWidget(nxt)
        v.addLayout(nav)

    def _refresh_range(self) -> None:
        a, b = self.model.selection
        self._range_title.setText(f"Which part? {span_text(a, b)}")
        self._time_labels["s"].setText(clock(a))
        self._time_labels["e"].setText(clock(b))
        self._only_button.setVisible(self.model.plan_mode)

    def _play_part(self) -> None:
        if self.model.selection is not None:
            self.playPartRequested.emit(*self.model.selection)

    # -- the Change page ------------------------------------------------------------------------------------------------------------
    def _build_change(self) -> None:
        v = QVBoxLayout(self._change)
        v.setContentsMargins(0, 0, 0, 0)
        v.setSpacing(8)
        top = QHBoxLayout()
        self._part_label = QLabel()
        top.addWidget(self._part_label, 1)
        self._play_btn = QPushButton("Play this part")
        self._play_btn.clicked.connect(self._play_part)
        self._undo_btn = QPushButton("Undo changes")
        self._undo_btn.setObjectName("Flat")
        self._undo_btn.setToolTip("Put this part back the way the saved file has it")
        self._undo_btn.clicked.connect(lambda: self.model.undo_changes())
        done = QPushButton("Done")
        done.setObjectName("Flat")
        done.clicked.connect(lambda: (self.model.clear_selection(), self.goto("home")))
        top.addWidget(self._play_btn)
        top.addWidget(self._undo_btn)
        top.addWidget(done)
        v.addLayout(top)
        self._sugg_slot = QVBoxLayout()
        v.addLayout(self._sugg_slot)

        v.addWidget(self._layer_title("noise", "Noise reduction"))
        self._strength = Segmented(list(STRENGTHS))
        self._strength.setToolTip("How much of the noise to remove in this part. Off leaves the sound as it is.")
        self._strength.changed.connect(self._strength_picked)
        v.addWidget(self._strength)
        self._strength_hint = QLabel()
        self._strength_hint.setObjectName("Small")
        self._strength_hint.setWordWrap(True)
        v.addWidget(self._strength_hint)
        rem = QHBoxLayout()
        rem.addWidget(QLabel("Background noise"))
        self._remover = QComboBox()
        for key, label in REMOVERS:
            self._remover.addItem(label, key)
        self._remover.setToolTip("Which kind of background noise this part has. Automatic measures the part.")
        self._remover.activated.connect(lambda i: self._remover_picked(self._remover.itemData(i)))
        rem.addWidget(self._remover, 1)
        v.addLayout(rem)

        v.addWidget(self._layer_title("rebuild", "Rebuild the voice"))
        self._rebuild = WrapCheck("Rebuild the voice in this part")
        self._rebuild.setToolTip("Makes the voice again from scratch in this part: for echo, a muffled or phone-quality voice or distortion.")
        self._rebuild.toggled.connect(lambda on: self._layer_picked("rebuild", on))
        v.addWidget(self._rebuild)
        self._wins = QLabel("Where Rebuild the voice and noise reduction are both on, Rebuild the voice wins: the noise reduction is not "
                            "used there.")
        self._wins.setObjectName("Small")
        self._wins.setWordWrap(True)
        v.addWidget(self._wins)
        self._no_voice = QLabel("The model that rebuilds the voice is not installed. Download it in Settings to use this.")
        self._no_voice.setObjectName("Small")
        self._no_voice.setWordWrap(True)
        v.addWidget(self._no_voice)

        self._echo_title = self._layer_title("echo", "Echo removal")
        v.addWidget(self._echo_title)
        self._echo = WrapCheck("Remove speaker echo from this part")
        self._echo.setToolTip("Uses the file's second audio track (your desktop sound) to take the speaker echo out of the microphone track.")
        self._echo.toggled.connect(lambda on: self._layer_picked("echo", on))
        v.addWidget(self._echo)

        v.addWidget(self._layer_title("leveling", "Leveling"))
        self._leveling = WrapCheck("Even out the volume in this part")
        self._leveling.setToolTip("Evens out the voice and sets the loudness. It also listens a little around the part to match what is next to it.")
        self._leveling.toggled.connect(lambda on: self._layer_picked("leveling", on))
        v.addWidget(self._leveling)
        lrow = QHBoxLayout()
        lrow.addWidget(QLabel("Loudness of the file"))
        self._loudness = QComboBox()
        for value, label in LOUDNESS_CHOICES:
            self._loudness.addItem(label, value)
        self._loudness.setToolTip("The loudness every leveled part is set to. It is one choice for the whole job.")
        self._loudness.activated.connect(lambda i: self.model.set_loudness(self._loudness.itemData(i)))
        lrow.addWidget(self._loudness, 1)
        v.addLayout(lrow)

        self._which = QToolButton()
        self._which.setText(labels.WHICH_TITLE)
        self._which.setCheckable(True)
        self._which.setArrowType(Qt.ArrowType.RightArrow)
        self._which.setToolButtonStyle(Qt.ToolButtonStyle.ToolButtonTextBesideIcon)
        self._which.setToolTip("How noise reduction and Rebuild the voice differ")
        self._which_text = QLabel(labels.WHICH_TEXT)
        self._which_text.setObjectName("Small")
        self._which_text.setWordWrap(True)
        self._which_text.hide()
        self._which.toggled.connect(lambda on: (self._which_text.setVisible(on),
                                                self._which.setArrowType(Qt.ArrowType.DownArrow if on else Qt.ArrowType.RightArrow)))
        v.addWidget(self._which)
        v.addWidget(self._which_text)
        v.addStretch(1)

    def _layer_title(self, layer: str, text: str) -> QLabel:
        label = QLabel(f"<span style='color:{LAYER_COLORS[layer]}'>■</span> <b>{text}</b>")
        return label

    def _strength_picked(self, key: str) -> None:
        if self._building or self.model.selection is None or key == "mixed":
            return
        self.model.set_strength(key)

    def _remover_picked(self, remover: str) -> None:
        if not self._building and self.model.selection is not None:
            self.model.set_remover(remover)

    def _layer_picked(self, layer: str, on: bool) -> None:
        if not self._building and self.model.selection is not None:
            self.model.set_layer(layer, on)

    def _refresh_change(self) -> None:
        m = self.model
        a, b = m.selection
        self._building = True
        try:
            self._part_label.setText(f"Part <b>{span_text(a, b)}</b> · {b - a:.0f} s")
            self._undo_btn.setVisible(m.dirty and not m.plan_mode and m.done is not None)
            while self._sugg_slot.count():
                item = self._sugg_slot.takeAt(0)
                if item.widget() is not None:
                    item.widget().hide()
                    item.widget().deleteLater()
            focus = m.suggestion(m.focus) if m.focus else None
            if focus is not None and focus.state != "dismissed":
                self._sugg_slot.addWidget(self._suggestion_card(focus))
            key = m.strength_of(a, b)
            self._strength.set_value(key) if key in dict(STRENGTHS) else self._strength.clear()
            self._strength_hint.setText(
                STRENGTH_HINTS.get(key, "Leaves this part as it is." if key == "off" else "This part has different strengths.")
                + (" Strength changes are heard right away." if key in STRENGTH_HINTS and not m.plan_mode else ""))
            rem = m.remover_of(a, b)
            self._remover.setEnabled(rem is not None)
            self._remover.setCurrentIndex(max(0, self._remover.findData(rem if rem in ("auto", "light", "heavy") else "auto")))
            rb = m.switch_of("rebuild", a, b)
            self._rebuild.setCheckState(Qt.CheckState.PartiallyChecked if rb is None else Qt.CheckState.Checked if rb else Qt.CheckState.Unchecked)
            self._rebuild.setEnabled(self._rebuild_available or bool(rb))
            self._no_voice.setVisible(not self._rebuild_available)
            self._wins.setVisible(m.rebuild_wins(a, b))
            self._echo_title.setVisible(m.echo_available)
            self._echo.setVisible(m.echo_available)
            self._echo.setChecked(m.switch_of("echo", a, b) is True)
            self._leveling.setChecked(m.switch_of("leveling", a, b) is True)
            self._loudness.setCurrentIndex(max(0, self._loudness.findData(m.loudness())))
            self._loudness.setEnabled(bool(m.draft.layers["leveling"]) or self._leveling.isChecked())
        finally:
            self._building = False


def _clear(layout) -> None:
    while layout.count():
        item = layout.takeAt(0)
        if item.widget() is not None:
            item.widget().hide()
            item.widget().deleteLater()
        elif item.layout() is not None:
            _clear(item.layout())
