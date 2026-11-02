"""What the editor screen shows and changes, without any widgets: the timeline being edited (the draft), the one the latest result was made
from, the part that is selected, the suggestions and what was done with them, and Undo and Redo of all that. No Qt in here; gui/studio.py draws it
(gui/timeline.py, gui/inspector.py) and tells it what the user did.

Every edit goes through `Timeline.paint` / `erase` (acw_common.timeline), so the draft is always a valid timeline: a part painted over old
ones replaces them, neighbours with the same settings merge.
"""
from __future__ import annotations

import copy
from collections.abc import Callable
from dataclasses import dataclass, field

from acw_common.timeline import DEFAULT_LOUDNESS, LAYERS, Part, Timeline

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
                 loudness: float | None = DEFAULT_LOUDNESS):
        self.seconds = float(seconds)
        # the job's loudness target: the one its leveled parts have, else the one given (None keeps the level)
        self._loudness = draft.layers["leveling"][0].settings["loudness_lufs"] if draft.layers["leveling"] else loudness
        self.draft = draft.copy()
        self.done = done.copy() if done is not None else None
        self.echo_available = echo_available
        self.selection: tuple[float, float] | None = None
        self.suggestions: list[Suggestion] = []
        self.focus: str | None = None  # the suggestion being looked at
        self.default_noise: dict = {"strength_db": STRENGTH_DB["max"], "remover": "auto"}  # what Noise removal starts as where it was off
        self._past: list[tuple] = []  # undo / redo: what the draft (and the suggestions) were before each edit
        self._future: list[tuple] = []
        self._listeners: list[Callable[[], None]] = []

    # -- listening ----------------------------------------------------------------------------------------------------
    def on_change(self, fn: Callable[[], None]) -> None:
        self._listeners.append(fn)

    def _changed(self) -> None:
        for fn in list(self._listeners):
            fn()

    # -- undo and redo (the draft, its loudness and what was done with the suggestions) -----------------------------------
    def _state(self) -> tuple:
        return (self.draft.copy(), self._loudness, [(s.id, s.state, s.prev) for s in self.suggestions])

    def _remember(self) -> None:
        """Called before every edit: one step of Undo."""
        self._past.append(self._state())
        del self._past[:-100]
        self._future.clear()

    def _restore_state(self, state: tuple) -> None:
        draft, loudness, sugg = state
        self.draft, self._loudness = draft.copy(), loudness
        for sid, st, prev in sugg:
            s = self.suggestion(sid)
            if s is not None:
                s.state, s.prev = st, prev

    @property
    def can_undo(self) -> bool:
        return bool(self._past)

    @property
    def can_redo(self) -> bool:
        return bool(self._future)

    def undo_edit(self) -> None:
        if not self._past:
            return
        self._future.append(self._state())
        self._restore_state(self._past.pop())
        self._changed()

    def redo_edit(self) -> None:
        if not self._future:
            return
        self._past.append(self._state())
        self._restore_state(self._future.pop())
        self._changed()

    def clear_history(self) -> None:
        self._past.clear()
        self._future.clear()

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

    def voice_mode_of(self, a: float, b: float) -> str:
        """What the voice is on [a, b): "off", "noise" (Noise removal), "rebuild" (Voice rebuild) or "mixed"."""
        reb = self.switch_of("rebuild", a, b)
        noise = self.switch_of("noise", a, b)
        if reb is True:
            return "rebuild"  # (where it is on it wins over the noise removal)
        if noise is True and reb is False:
            return "noise"
        if noise is False and reb is False:
            return "off"
        return "mixed"

    def level_of(self, a: float, b: float) -> str:
        """Leveling on [a, b): "off", "on" or "mixed"."""
        on = self.switch_of("leveling", a, b)
        return "mixed" if on is None else "on" if on else "off"

    def sections(self, layer: str) -> list[dict]:
        """The whole file as consecutive stretches of one layer: {"start", "end", "on", "label", "pending", "settings"}; the stretches the layer
        does not cover are "on": False. (The rows of the timeline.)"""
        out, pos = [], 0.0
        runs = {(r["start"], r["end"]): r for r in self.runs(layer)}
        for p in self.draft.layers[layer]:
            if p.start > pos + 1e-6:
                out.append({"start": pos, "end": p.start, "on": False, "label": "", "pending": False, "settings": None})
            r = runs.get((p.start, p.end), {})
            out.append({"start": p.start, "end": p.end, "on": True, "label": r.get("label", ""), "pending": r.get("pending", False),
                        "settings": dict(p.settings)})
            pos = p.end
        if pos < self.seconds - 1e-6:
            out.append({"start": pos, "end": self.seconds, "on": False, "label": "", "pending": False, "settings": None})
        return out

    def voice_sections(self) -> list[dict]:
        """The whole file as consecutive stretches of the voice: "off", Noise removal ("noise", with its strength and remover) or Voice rebuild
        ("rebuild": where it is on it wins over the noise removal). {"start", "end", "mode", "label", "pending", "settings"}."""
        edges = {0.0, self.seconds}
        for layer in ("noise", "rebuild"):
            for p in self.draft.layers[layer]:
                edges.update((p.start, p.end))
        cuts = sorted(e for e in edges if 0.0 <= e <= self.seconds)
        out: list[dict] = []
        for lo, hi in zip(cuts, cuts[1:]):
            if hi - lo < 1e-6:
                continue
            mid = (lo + hi) / 2
            reb = next((p for p in self.draft.layers["rebuild"] if p.start <= mid < p.end), None)
            noi = next((p for p in self.draft.layers["noise"] if p.start <= mid < p.end), None)
            if reb is not None:
                row = {"mode": "rebuild", "label": "Rebuild", "settings": dict(reb.settings)}
                pending = self.done is not None and not any(d.start <= lo + 1e-6 and d.end >= hi - 1e-6 for d in self.done.layers["rebuild"])
            elif noi is not None:
                key = strength_key(noi.settings["strength_db"])
                label = "Max" if key == "max" else STRENGTH_NAME.get(key, f"{noi.settings['strength_db']:.0f} dB")  # (the rows say Max, as the web's do)
                if noi.settings["remover"] != "auto":
                    label += f" · {dict(REMOVERS)[noi.settings['remover']].split()[0]}"
                row = {"mode": "noise", "label": label, "settings": dict(noi.settings)}
                pending = self.done is not None and not any(
                    d.start <= lo + 1e-6 and d.end >= hi - 1e-6 and d.settings == noi.settings for d in self.done.layers["noise"])
            else:
                row = {"mode": "off", "label": "", "settings": None}
                pending = False
            row.update(start=lo, end=hi, pending=pending)
            if out and out[-1]["mode"] == row["mode"] and out[-1]["settings"] == row["settings"] and out[-1]["pending"] == pending:
                out[-1]["end"] = hi
            else:
                out.append(row)
        return out

    # -- the web's voice choice ---------------------------------------------------------------------------------------------------
    def set_voice(self, mode: str, a: float | None = None, b: float | None = None) -> None:
        """Voice on [a, b): "off", "noise" (Noise removal: the strength it had, else `default_noise`) or "rebuild" (Voice rebuild)."""
        a, b = self._range(a, b)
        if mode not in ("off", "noise", "rebuild"):
            raise ValueError(f"unknown voice choice {mode!r}")
        self._remember()
        if mode == "rebuild":
            self.draft.erase("noise", a, b)
            self.draft.paint("rebuild", a, b)
        elif mode == "off":
            self.draft.erase("noise", a, b)
            self.draft.erase("rebuild", a, b)
        else:
            self.draft.erase("rebuild", a, b)
            pieces, pos = [], a
            for p in self.parts("noise", a, b):
                lo, hi = max(a, p.start), min(b, p.end)
                if lo > pos:
                    pieces.append((pos, lo, dict(self.default_noise)))
                pieces.append((lo, hi, dict(p.settings)))
                pos = hi
            if pos < b:
                pieces.append((pos, b, dict(self.default_noise)))
            for lo, hi, st in pieces:
                self.draft.paint("noise", lo, hi, st)
        self._changed()

    def set_level(self, lufs: float | None, on: bool, a: float | None = None, b: float | None = None) -> None:
        """Leveling on [a, b) at a loudness (every leveled part of the file gets it: it is one choice for the whole file), or Off."""
        a, b = self._range(a, b)
        self._remember()
        if on:
            self._loudness = lufs
            for p in list(self.draft.layers["leveling"]):
                self.draft.paint("leveling", p.start, p.end, {"loudness_lufs": lufs})
            self.draft.paint("leveling", a, b, {"loudness_lufs": lufs})
        else:
            self.draft.erase("leveling", a, b)
        self._changed()

    def leave_the_rest(self, a: float | None = None, b: float | None = None) -> None:
        """Only [a, b) is processed: every layer is taken off everything outside it (the rest is left as it is)."""
        a, b = self._range(a, b)
        self._remember()
        for layer in LAYERS:
            if a > 0:
                self.draft.erase(layer, 0.0, a)
            if b < self.seconds:
                self.draft.erase(layer, b, self.seconds)
        self._changed()

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
        self._remember()
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
        self._remember()
        for p in self.parts("noise", a, b):
            lo, hi = max(a, p.start), min(b, p.end)
            self.draft.paint("noise", lo, hi, {"strength_db": p.settings["strength_db"], "remover": remover})
        self._changed()

    def set_layer(self, layer: str, on: bool, a: float | None = None, b: float | None = None) -> None:
        """Voice rebuild, Echo removal or Leveling on or off for [a, b)."""
        a, b = self._range(a, b)
        if layer == "echo" and on and not self.echo_available:
            return
        self._remember()
        if on:
            self.draft.paint(layer, a, b, {"loudness_lufs": self._loudness} if layer == "leveling" else None)
        else:
            self.draft.erase(layer, a, b)
        self._changed()

    def set_loudness(self, lufs: float | None) -> None:
        """The loudness target of the job: every leveled part gets it."""
        self._remember()
        self._loudness = lufs
        for p in list(self.draft.layers["leveling"]):
            self.draft.paint("leveling", p.start, p.end, {"loudness_lufs": lufs})
        self._changed()

    def undo_changes(self, a: float | None = None, b: float | None = None) -> None:
        """Put [a, b) back as the latest result has it."""
        a, b = self._range(a, b)
        if self.done is None:
            return
        self._remember()
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
        self._remember()
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
        self._remember()
        for layer in LAYERS:
            parts = [Part(d["start"], d["end"], {k: v for k, v in d.items() if k not in ("start", "end")}) for d in (s.prev or {}).get(layer, [])]
            self._restore(layer, parts, s.start, s.end)
        s.state, s.prev = "open", None
        self._changed()

    def dismiss(self, sid: str) -> None:
        s = self.suggestion(sid)
        if s is not None and s.state == "open":
            self._remember()
            s.state = "dismissed"
            self._changed()

    def show(self, sid: str) -> None:
        s = self.suggestion(sid)
        if s is not None:
            self.focus = sid
            self.select(s.start, s.end)

    def effective(self) -> Timeline:
        """The timeline to process: the draft as it is."""
        return self.draft.copy()

    # -- remembering the state of the job --------------------------------------------------------------------------------------
    def snapshot(self) -> dict:
        return {"draft": self.draft.to_json(), "suggestions": self.suggestions_json()}

    def copy_draft(self) -> Timeline:
        return copy.deepcopy(self.draft)
