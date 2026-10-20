"""What the per-part editor shows and changes, without any widgets: the timeline being edited (the draft), the one the latest result was made
from, the part that is selected, the suggestions and what was done with them, and, before a file is cleaned, the parts chosen to be
processed. No Qt in here; gui/part_editor.py draws it and tells it what the user did.

Every edit goes through `Timeline.paint` / `erase` (acw_common.timeline), so the draft is always a valid timeline: a part painted over old
ones replaces them, neighbours with the same settings merge.
"""
from __future__ import annotations

import copy
from collections.abc import Callable
from dataclasses import dataclass, field

from acw_common.timeline import DEFAULT_LOUDNESS, LAYERS, Part, Timeline

from . import editor_session

#: The strengths of a part, in the order of the buttons (Off takes the noise removal off the part).
STRENGTHS = (("off", "Off"), ("gentle", "Gentle"), ("balanced", "Balanced"), ("strong", "Strong"), ("max", "Maximum"))
STRENGTH_DB = {"gentle": 12.0, "balanced": 24.0, "strong": 40.0, "max": 100.0}
STRENGTH_NAME = dict(STRENGTHS)
STRENGTH_HINTS = {
    "gentle": "Takes the edge off. Keeps some room sound and music.",
    "balanced": "Most of the noise goes; the room still sounds like a room.",
    "strong": "Nearly all of the noise goes.",
    "max": "Removes as much as it can. Best for noisy places.",
}
REMOVERS = (("auto", "Automatic"), ("light", "Light noise"), ("heavy", "Heavy noise"))
LAYER_NAMES = {"echo": "Echo removal", "noise": "Noise removal", "rebuild": "Voice rebuild", "leveling": "Leveling"}
#: What the Hear switches and the layer chips call each layer, in the order they are listed.
LAYER_ORDER = ("noise", "rebuild", "echo", "leveling")
AROUND_S = 5.0
MIN_PART_S = 1.0


def clock(seconds: float) -> str:
    s = int(round(max(0.0, seconds)))
    return f"{s // 3600}:{s % 3600 // 60:02d}:{s % 60:02d}" if s >= 3600 else f"{s // 60}:{s % 60:02d}"


def span_text(a: float, b: float) -> str:
    return f"{clock(a)}–{clock(b)}"


def strength_key(db: float) -> str:
    """The name of a strength in dB ("custom" when it is none of the four)."""
    return next((k for k, v in STRENGTH_DB.items() if abs(v - db) < 0.5), "custom")


@dataclass
class Suggestion:
    """A part the app found that other settings would suit (editor_job.find_suggestions), and what the user did with it."""
    id: str
    start: float
    end: float
    kind: str  # "part" | "clipping"
    rebuild: bool
    noise: dict | None
    reasons: list[str]
    state: str = "open"  # "open" | "applied" | "dismissed"
    prev: dict | None = field(default=None, repr=False)  # the parts it replaced, by layer (for Undo)

    @classmethod
    def from_json(cls, d: dict) -> "Suggestion":
        return cls(str(d["id"]), float(d["start"]), float(d["end"]), d.get("kind", "part"), bool(d.get("rebuild")), d.get("noise"),
                   list(d.get("reasons") or []), d.get("state", "open"), d.get("prev"))

    def to_json(self) -> dict:
        return {"id": self.id, "start": self.start, "end": self.end, "kind": self.kind, "rebuild": self.rebuild, "noise": self.noise,
                "reasons": self.reasons, "state": self.state, **({"prev": self.prev} if self.prev is not None else {})}

    @property
    def what(self) -> str:
        return " and ".join(self.reasons) if self.reasons else "may sound better with other settings"

    @property
    def ask(self) -> str:
        if self.rebuild:
            return "Rebuild just this part?"
        if self.noise:
            return f"{STRENGTH_NAME.get(strength_key(self.noise['strength_db']), 'Custom')} here?"
        return "Change this part?"

    @property
    def action(self) -> str:
        if self.rebuild:
            return "Rebuild this part"
        if self.noise:
            return f"Use {STRENGTH_NAME.get(strength_key(self.noise['strength_db']), 'this strength')} here"
        return "Apply"


class EditModel:
    """The draft timeline and everything around it. `changed` callbacks run after every change (the widgets redraw)."""

    def __init__(self, seconds: float, draft: Timeline, done: Timeline | None = None, *, echo_available: bool = False,
                 plan_mode: bool = False, loudness: float | None = DEFAULT_LOUDNESS):
        self.seconds = float(seconds)
        # the job's loudness target: the one its leveled parts have, else the one given (None keeps the level)
        self._loudness = draft.layers["leveling"][0].settings["loudness_lufs"] if draft.layers["leveling"] else loudness
        self.draft = draft.copy()
        self.done = done.copy() if done is not None else None
        self.echo_available = echo_available
        self.plan_mode = plan_mode
        self.selection: tuple[float, float] | None = None
        self.suggestions: list[Suggestion] = []
        self.chosen: list[tuple[float, float]] = []  # plan mode: the parts to process when "Only parts" is on
        self.only = False
        self.focus: str | None = None  # the suggestion being looked at
        self._listeners: list[Callable[[], None]] = []

    # -- listening ----------------------------------------------------------------------------------------------------
    def on_change(self, fn: Callable[[], None]) -> None:
        self._listeners.append(fn)

    def _changed(self) -> None:
        for fn in list(self._listeners):
            fn()

    # -- the part that is selected --------------------------------------------------------------------------------------
    def select(self, a: float, b: float) -> None:
        a, b = sorted((a, b))
        a, b = max(0.0, a), min(self.seconds, b)
        if b - a < MIN_PART_S:
            b = min(self.seconds, a + MIN_PART_S)
            a = max(0.0, b - MIN_PART_S)
        self.selection = (a, b)
        self._changed()

    def clear_selection(self) -> None:
        self.selection = None
        self.focus = None
        self._changed()

    def around(self, t: float, radius: float = AROUND_S) -> None:
        self.select(t - radius, t + radius)

    def whole(self) -> None:
        self.selection = (0.0, self.seconds)
        self._changed()

    def nudge(self, which: str, delta: float) -> None:
        if self.selection is None:
            return
        a, b = self.selection
        self.select(a + delta, b) if which == "s" else self.select(a, b + delta)

    def use_playhead(self, which: str, t: float) -> None:
        if self.selection is None:
            return
        a, b = self.selection
        if which == "s":
            self.select(min(t, b - MIN_PART_S), b)
        else:
            self.select(a, max(t, a + MIN_PART_S))

    def _range(self, a: float | None, b: float | None) -> tuple[float, float]:
        if a is None or b is None:
            if self.selection is None:
                raise ValueError("no part is selected")
            return self.selection
        return a, b

    # -- reading the draft ------------------------------------------------------------------------------------------------
    @property
    def dirty(self) -> bool:
        return self.done is None or self.draft != self.done

    def parts(self, layer: str, a: float, b: float) -> list[Part]:
        return [p for p in self.draft.layers[layer] if p.end > a and p.start < b]

    def _covers_all(self, layer: str, a: float, b: float) -> bool:
        pos = a
        for p in self.draft.layers[layer]:
            if p.end <= pos:
                continue
            if p.start > pos + 1e-6:
                return False
            pos = max(pos, p.end)
            if pos >= b - 1e-6:
                return True
        return pos >= b - 1e-6

    def strength_of(self, a: float, b: float) -> str:
        """"off", one of the four names, "custom" or "mixed" for what the noise removal is on [a, b)."""
        ps = self.parts("noise", a, b)
        if not ps:
            return "off"
        keys = {strength_key(p.settings["strength_db"]) for p in ps}
        if len(keys) == 1 and self._covers_all("noise", a, b):
            return keys.pop()
        return "mixed"

    def remover_of(self, a: float, b: float) -> str | None:
        """"auto" / "light" / "heavy" for the noise removal on [a, b), "mixed", or None when it is off there."""
        ps = self.parts("noise", a, b)
        if not ps:
            return None
        rs = {p.settings["remover"] for p in ps}
        return rs.pop() if len(rs) == 1 else "mixed"

    def switch_of(self, layer: str, a: float, b: float) -> bool | None:
        """Is the layer on for all of [a, b) (True), none of it (False) or part of it (None)?"""
        ps = self.parts(layer, a, b)
        if not ps:
            return False
        return True if self._covers_all(layer, a, b) else None

    def rebuild_wins(self, a: float, b: float) -> bool:
        """Voice rebuild and noise removal both cover some of [a, b): the noise removal is not used there."""
        for r in self.parts("rebuild", a, b):
            lo, hi = max(a, r.start), min(b, r.end)
            if any(n.end > lo and n.start < hi for n in self.draft.layers["noise"]):
                return True
        return False

    def loudness(self) -> float | None:
        return self._loudness

    def runs(self, layer: str) -> list[dict]:
        """The parts of a layer for the lists and the rows: {"start", "end", "label", "pending"}."""
        out = []
        for p in self.draft.layers[layer]:
            label = STRENGTH_NAME.get(strength_key(p.settings["strength_db"]), f"{p.settings['strength_db']:.0f} dB") if layer == "noise" else ""
            if layer == "noise" and p.settings["remover"] != "auto":
                label += f" ({p.settings['remover']} noise)"
            pending = self.done is not None and not any(
                d.start <= p.start + 1e-6 and d.end >= p.end - 1e-6 and d.settings == p.settings for d in self.done.layers[layer])
            out.append({"start": p.start, "end": p.end, "label": label, "pending": pending})
        return out

    # -- editing the draft ------------------------------------------------------------------------------------------------
    def set_strength(self, key: str, a: float | None = None, b: float | None = None) -> None:
        """Noise removal on [a, b) (the selection by default) at a strength, or Off. A part keeps the remover it has; new ones are automatic."""
        a, b = self._range(a, b)
        if key == "off":
            self.draft.erase("noise", a, b)
        else:
            db = STRENGTH_DB[key]
            pieces, pos = [], a
            for p in self.parts("noise", a, b):
                lo, hi = max(a, p.start), min(b, p.end)
                if lo > pos:
                    pieces.append((pos, lo, "auto"))
                pieces.append((lo, hi, p.settings["remover"]))
                pos = hi
            if pos < b:
                pieces.append((pos, b, "auto"))
            for lo, hi, rem in pieces:
                self.draft.paint("noise", lo, hi, {"strength_db": db, "remover": rem})
        self._changed()

    def set_remover(self, remover: str, a: float | None = None, b: float | None = None) -> None:
        a, b = self._range(a, b)
        for p in self.parts("noise", a, b):
            lo, hi = max(a, p.start), min(b, p.end)
            self.draft.paint("noise", lo, hi, {"strength_db": p.settings["strength_db"], "remover": remover})
        self._changed()

    def set_layer(self, layer: str, on: bool, a: float | None = None, b: float | None = None) -> None:
        """Voice rebuild, Echo removal or Leveling on or off for [a, b)."""
        a, b = self._range(a, b)
        if layer == "echo" and on and not self.echo_available:
            return
        if on:
            self.draft.paint(layer, a, b, {"loudness_lufs": self._loudness} if layer == "leveling" else None)
        else:
            self.draft.erase(layer, a, b)
        self._changed()

    def set_loudness(self, lufs: float | None) -> None:
        """The loudness target of the job: every leveled part gets it."""
        self._loudness = lufs
        for p in list(self.draft.layers["leveling"]):
            self.draft.paint("leveling", p.start, p.end, {"loudness_lufs": lufs})
        self._changed()

    def undo_changes(self, a: float | None = None, b: float | None = None) -> None:
        """Put [a, b) back as the latest result has it."""
        a, b = self._range(a, b)
        if self.done is None:
            return
        for layer in LAYERS:
            self._restore(layer, self.done.layers[layer], a, b)
        self._changed()

    def _restore(self, layer: str, parts: list[Part], a: float, b: float) -> None:
        self.draft.erase(layer, a, b)
        for p in parts:
            lo, hi = max(a, p.start), min(b, p.end)
            if hi > lo:
                self.draft.paint(layer, lo, hi, dict(p.settings))

    # -- suggestions ------------------------------------------------------------------------------------------------------
    def set_suggestions(self, items: list[dict] | None) -> None:
        self.suggestions = [Suggestion.from_json(d) for d in (items or [])]
        self._changed()

    def suggestions_json(self) -> list[dict]:
        return [s.to_json() for s in self.suggestions]

    def open_suggestions(self) -> list[Suggestion]:
        return [s for s in self.suggestions if s.state == "open"]

    def suggestion(self, sid: str) -> Suggestion | None:
        return next((s for s in self.suggestions if s.id == sid), None)

    def apply(self, sid: str) -> None:
        s = self.suggestion(sid)
        if s is None or s.state != "open":
            return
        s.prev = {layer: [{"start": p.start, "end": p.end, **p.settings} for p in self.draft.layers[layer] if p.end > s.start and p.start < s.end]
                  for layer in LAYERS}
        if s.rebuild:
            self.draft.paint("rebuild", s.start, s.end)
        else:
            self.draft.erase("rebuild", s.start, s.end)
            if s.noise:
                self.draft.paint("noise", s.start, s.end, dict(s.noise))
        s.state = "applied"
        self._changed()

    def undo(self, sid: str) -> None:
        s = self.suggestion(sid)
        if s is None or s.state != "applied":
            return
        for layer in LAYERS:
            parts = [Part(d["start"], d["end"], {k: v for k, v in d.items() if k not in ("start", "end")}) for d in (s.prev or {}).get(layer, [])]
            self._restore(layer, parts, s.start, s.end)
        s.state, s.prev = "open", None
        self._changed()

    def dismiss(self, sid: str) -> None:
        s = self.suggestion(sid)
        if s is not None and s.state == "open":
            s.state = "dismissed"
            self._changed()

    def show(self, sid: str) -> None:
        s = self.suggestion(sid)
        if s is not None:
            self.focus = sid
            self.select(s.start, s.end)

    # -- before a file is cleaned: the parts to process ------------------------------------------------------------------------
    def choose(self, a: float | None = None, b: float | None = None) -> None:
        """Add [a, b) (the selection by default) to the parts that are processed when "Only parts" is on."""
        a, b = self._range(a, b)
        ranges = sorted([*self.chosen, (a, b)])
        merged: list[list[float]] = []
        for lo, hi in ranges:
            if merged and lo <= merged[-1][1] + 1e-6:
                merged[-1][1] = max(merged[-1][1], hi)
            else:
                merged.append([lo, hi])
        self.chosen = [(lo, hi) for lo, hi in merged]
        self.only = True
        self._changed()

    def unchoose(self, index: int) -> None:
        if 0 <= index < len(self.chosen):
            self.chosen.pop(index)
            if not self.chosen:
                self.only = False
            self._changed()

    def set_only(self, on: bool) -> None:
        self.only = bool(on)
        self._changed()

    def effective(self) -> Timeline:
        """The timeline to process: the draft, and with "Only parts" nothing but the chosen parts."""
        if self.plan_mode and self.only:
            return editor_session.only_parts(self.draft, self.chosen)
        return self.draft.copy()

    def processed_seconds(self) -> float:
        """Plan mode: how much of the file is processed."""
        if self.plan_mode and self.only:
            return sum(b - a for a, b in self.chosen)
        return self.seconds

    # -- remembering the state of the job --------------------------------------------------------------------------------------
    def snapshot(self) -> dict:
        return {"draft": self.draft.to_json(), "suggestions": self.suggestions_json()}

    def copy_draft(self) -> Timeline:
        return copy.deepcopy(self.draft)
