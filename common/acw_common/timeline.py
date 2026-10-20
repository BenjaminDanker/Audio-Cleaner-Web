"""The timeline of the per-part editor: what is done to which part of a recording. The one format the server's API and the
desktop app use (JSON-able, versioned, validated).

A recording has LAYERS, one per step, and each layer covers PARTS of the timeline, each part with its own settings:

    echo      remove the speaker echo (needs the file's second track): no settings
    noise     remove the noise: strength_db 12 Gentle / 24 Balanced / 40 Strong / 100 Maximum (or any value above 0), remover
              "light" | "heavy" | "auto" (measured per part: editor.resolve_removers)
    rebuild   rebuild the voice: no settings. Where noise removal and Voice rebuild both cover a moment, Voice rebuild wins
              (measured: chaining them adds nothing), so the noise removal there is not run
    leveling  even out the voice and set the loudness: loudness_lufs (-70..0; null keeps the level)

and, from version 2, a layer that is not part of the sound:

    subtitles  quality "free" (the browser transcribes it) | "standard" | "accurate" (the server transcribes it), language (a
               code, null: detect) and translate (up to 3 codes, not the part's language, only with standard or accurate)

The default, `Timeline.whole_file`, is every layer you ask for on the whole file: what `pipeline.clean_audio` does (no subtitles).
The JSON:

    {"version": 1, "seconds": 123.4, "layers": {
        "noise": [{"start": 0, "end": 123.4, "strength_db": 40, "remover": "auto"}],
        "rebuild": [{"start": 30.5, "end": 60}],
        "leveling": [{"start": 0, "end": 123.4, "loudness_lufs": -16}]}}

VERSION 2 adds two things: the `subtitles` layer, and `"browser": true` on a noise part (the browser cleaned that range itself and
uploaded it; the remover is then not used and left out of the JSON, and a `strength_db` that was sent is kept (so a page can show the strength
the person chose) but never used in processing or pricing; an echo part may not overlap it). `to_json` says
version 2 only for a timeline that uses one of them, so every timeline the first version could say is written exactly as before
(old builds of the desktop app read only version 1); `from_json` reads both, and a version 1 document may not use the new things.

Times are seconds, kept to the millisecond and clamped to [0, seconds]. The parts of a layer never overlap: a part painted
over old ones replaces them (`Timeline.paint`; in JSON the later part wins), neighbours with the same settings are merged, an
empty part (after clamping) is dropped. A layer that is missing has no parts. Anything else (an unknown key, layer or setting,
a bad value, a version this code does not know) raises `TimelineError`.
"""
from __future__ import annotations

import json
import math
import re
from collections.abc import Iterator
from dataclasses import dataclass, field
from typing import Any

SAMPLE_RATE = 48_000
VERSION = 1  # the version a timeline without anything new is written in
VERSION_2 = 2  # ... and the one that has subtitles or browser-cleaned parts
VERSIONS = (VERSION, VERSION_2)
#: The layers that make the sound, in the order the steps run in; each layer's input is the result of the ones before it.
LAYERS = ("echo", "noise", "rebuild", "leveling")
SUBTITLES = "subtitles"
ALL_LAYERS = (*LAYERS, SUBTITLES)
QUALITIES = ("free", "standard", "accurate")
#: The languages of the website's lists (web/src/jobs.js LANGUAGES, server/acw/translate.py): what free and standard subtitles take
#: and what a translation can be made into. Accurate subtitles also take any other 2-3 letter code the engine knows.
LANGUAGES = ("en", "es", "fr", "de", "it", "pt", "nl", "pl", "ru", "uk", "tr", "ar", "hi", "ja", "ko", "zh", "sv", "da", "fi", "no",
             "cs", "el", "he", "id", "vi", "hu", "ro", "th")
MAX_TRANSLATIONS = 3
REMOVERS = ("auto", "light", "heavy")
STRENGTHS = {"gentle": 12.0, "balanced": 24.0, "strong": 40.0, "max": 100.0, "maximum": 100.0}
MAX_PARTS = 500  # per layer
DEFAULT_LOUDNESS = -16.0


class TimelineError(ValueError):
    """The timeline is not valid (the message says what, in plain words)."""


def ms(t: float) -> float:
    """A time in seconds rounded to the millisecond."""
    return round(float(t) * 1000.0) / 1000.0


def _number(v: Any, what: str) -> float:
    if isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v):
        raise TimelineError(f"{what} must be a number")
    return float(v)


def _language(raw: object, quality: str) -> str | None:
    """A subtitle language: a code of LANGUAGES (accurate: any 2-3 letter code), or None to detect it."""
    if raw is None or raw == "":
        return None
    if not isinstance(raw, str):
        raise TimelineError("language must be a code such as en, or null to detect it")
    code = raw.strip().lower()
    if code in LANGUAGES or (quality == "accurate" and re.fullmatch(r"[a-z]{2,3}", code)):
        return code
    raise TimelineError(f"unknown language {raw[:8]!r} for {quality} subtitles")


def _subtitles(raw: dict) -> dict:
    quality = raw.pop("quality", None)
    if quality not in QUALITIES:
        raise TimelineError("subtitles need a quality: free, standard or accurate")
    language = _language(raw.pop("language", None), quality)
    translate = raw.pop("translate", None)
    if translate is None:
        translate = []
    if not isinstance(translate, list) or not all(isinstance(t, str) for t in translate):
        raise TimelineError("translate must be a list of language codes")
    codes = list(dict.fromkeys(t.strip().lower() for t in translate if t.strip()))
    if len(codes) > MAX_TRANSLATIONS:
        raise TimelineError(f"at most {MAX_TRANSLATIONS} translation languages for a part")
    for code in codes:
        if code not in LANGUAGES:
            raise TimelineError(f"unknown translation language {code[:8]!r}")
        if code == language:
            raise TimelineError("a translation language is the same as the spoken language of the part")
    if codes and quality == "free":
        raise TimelineError("Translations need Standard or Accurate subtitles: free subtitles are made in the browser and can't be translated")
    return {"quality": quality, "language": language, "translate": codes}


def _clone(settings: dict) -> dict:
    return {k: (list(v) if isinstance(v, list) else v) for k, v in settings.items()}


def _strength(raw: dict) -> float:
    """A noise strength in dB, from `strength_db` or its short name `strength`, a number or a word (gentle, balanced, strong, max)."""
    strength = raw.pop("strength_db", raw.pop("strength", 100.0))
    if isinstance(strength, str):
        if strength.lower() not in STRENGTHS:
            raise TimelineError(f"unknown strength {strength!r} (gentle, balanced, strong, max, or a number of dB)")
        strength = STRENGTHS[strength.lower()]
    strength = _number(strength, "strength_db")
    if not 0.0 < strength <= 100.0:
        raise TimelineError("strength_db must be above 0 and at most 100")
    return float(strength)


def _settings(layer: str, raw: dict | None) -> dict:
    """The settings of a part of `layer`, checked and complete (defaults filled in)."""
    raw = dict(raw or {})
    if layer == SUBTITLES:
        out = _subtitles(raw)
    elif layer == "noise":
        browser = raw.pop("browser", False)
        if not isinstance(browser, bool):
            raise TimelineError("browser must be true or false")
        if browser:  # cleaned by the browser: no remover; a strength that was sent is kept for the page, and used for nothing
            raw.pop("remover", None)
            sent = "strength_db" in raw or "strength" in raw
            if sent and raw.get("strength_db", raw.get("strength")) is None:
                raw.pop("strength_db", None), raw.pop("strength", None)
                sent = False
            out = {"browser": True, **({"strength_db": _strength(raw)} if sent else {})}
        else:
            strength = _strength(raw)
            remover = raw.pop("remover", "auto")
            if remover not in REMOVERS:
                raise TimelineError(f"remover must be one of {', '.join(REMOVERS)}")
            out = {"strength_db": strength, "remover": remover}
    elif layer == "leveling":
        lufs = raw.pop("loudness_lufs", DEFAULT_LOUDNESS)
        if lufs is not None:
            lufs = _number(lufs, "loudness_lufs")
            if not -70.0 <= lufs <= 0.0:
                raise TimelineError("loudness_lufs must be between -70 and 0 (or null to keep the level)")
        out = {"loudness_lufs": lufs}
    elif layer in ("echo", "rebuild"):
        out = {}
    else:
        raise TimelineError(f"unknown layer {layer!r}")
    if raw:
        raise TimelineError(f"unknown setting {sorted(raw)[0]!r} for {layer}")
    return out


@dataclass(frozen=True)
class Part:
    """A stretch [start, end) of the timeline (seconds) and its settings."""
    start: float
    end: float
    settings: dict = field(default_factory=dict)

    @property
    def seconds(self) -> float:
        return self.end - self.start

    def to_json(self) -> dict:
        return {"start": self.start, "end": self.end, **self.settings}


class Timeline:
    """The layers and their parts for a recording of `seconds`. Mutate with `paint` / `erase`; every state is valid."""

    def __init__(self, seconds: float, layers: dict[str, list[Part]] | None = None) -> None:
        seconds = _number(seconds, "seconds")
        if seconds < 0:
            raise TimelineError("seconds must not be negative")
        self.seconds = ms(seconds)
        self.layers: dict[str, list[Part]] = {name: [] for name in ALL_LAYERS}
        for name, parts in (layers or {}).items():
            if name not in self.layers:
                raise TimelineError(f"unknown layer {name!r}")
            for p in parts:
                self.paint(name, p.start, p.end, p.settings)

    # ---- building ------------------------------------------------------------------------------------------------
    @classmethod
    def whole_file(cls, seconds: float, *, echo: bool = False, noise: dict | bool | None = True, rebuild: bool = False,
                   leveling: dict | bool | None = True) -> "Timeline":
        """Every layer asked for on the whole file (the default: noise removal and Leveling). `noise` / `leveling`: True for
        the default settings, a dict for these settings, False or None for not at all."""
        t = cls(seconds)
        for name, on in (("echo", echo), ("noise", noise), ("rebuild", rebuild), ("leveling", leveling)):
            if on:
                t.paint(name, 0.0, t.seconds, on if isinstance(on, dict) else None)
        return t

    @classmethod
    def from_clean_options(cls, seconds: float, opts, *, echo: bool = False) -> "Timeline":
        """The timeline that `pipeline.clean_audio(x, ..., opts)` stands for (opts.atten_lim_db None, the automatic strength,
        has no timeline: measure first)."""
        if opts.atten_lim_db is None:
            raise TimelineError("an automatic strength is decided by measuring: pass the strength it chose")
        t = cls.whole_file(seconds, echo=echo, noise=False, leveling=False)
        if opts.restore_voice:
            t.paint("rebuild", 0.0, t.seconds)
        elif opts.denoise:
            t.paint("noise", 0.0, t.seconds, {"strength_db": opts.atten_lim_db, "remover": opts.noise})
        if opts.leveling:
            t.paint("leveling", 0.0, t.seconds, {"loudness_lufs": opts.loudness_lufs})
        return t

    def copy(self) -> "Timeline":
        t = Timeline(self.seconds)
        t.layers = {name: [Part(p.start, p.end, _clone(p.settings)) for p in parts] for name, parts in self.layers.items()}
        return t

    def audio(self) -> "Timeline":
        """What the result audio depends on: only the layers that make the sound (no subtitles), and a part the browser cleaned without the
        strength the page keeps for it (neighbours that are then equal are one part, so two timelines that sound the same are equal)."""
        t = Timeline(self.seconds)
        for name in LAYERS:
            for p in self.layers[name]:
                t.paint(name, p.start, p.end, {"browser": True} if p.settings.get("browser") else p.settings)
        return t

    def _layer(self, layer: str) -> list[Part]:
        if layer not in self.layers:
            raise TimelineError(f"unknown layer {layer!r}")
        return self.layers[layer]

    def erase(self, layer: str, start: float, end: float) -> "Timeline":
        """Take [start, end) out of the layer (a part that reaches in is shortened, one that contains it is split)."""
        parts = self._layer(layer)
        a, b = ms(max(0.0, start)), ms(min(self.seconds, end))
        if b <= a:
            return self
        out: list[Part] = []
        for p in parts:
            if p.end <= a or p.start >= b:
                out.append(p)
                continue
            if p.start < a:
                out.append(Part(p.start, a, _clone(p.settings)))
            if p.end > b:
                out.append(Part(b, p.end, _clone(p.settings)))
        self.layers[layer] = out
        return self

    def paint(self, layer: str, start: float, end: float, settings: dict | None = None) -> "Timeline":
        """Put a part on the layer, replacing whatever it covers (times clamped to the file; an empty part is ignored)."""
        start, end = _number(start, "start"), _number(end, "end")
        if end < start:
            raise TimelineError("a part must not end before it starts")
        s = _settings(layer, settings)
        a, b = ms(max(0.0, start)), ms(min(self.seconds, end))
        if b <= a:
            return self
        self.erase(layer, a, b)
        parts = self._layer(layer)
        parts.append(Part(a, b, s))
        parts.sort(key=lambda p: p.start)
        merged: list[Part] = []
        for p in parts:  # neighbours with the same settings are one part
            if merged and merged[-1].end == p.start and merged[-1].settings == p.settings:
                merged[-1] = Part(merged[-1].start, p.end, merged[-1].settings)
            else:
                merged.append(p)
        if len(merged) > MAX_PARTS:
            raise TimelineError(f"at most {MAX_PARTS} parts per layer")
        self.layers[layer] = merged
        return self

    # ---- reading -------------------------------------------------------------------------------------------------
    def parts(self, layer: str) -> list[Part]:
        return list(self._layer(layer))

    def covers(self, layer: str, t: float) -> bool:
        return any(p.start <= t < p.end for p in self._layer(layer))

    def browser_parts(self) -> list[Part]:
        """The noise parts the browser cleaned itself (`browser: true`)."""
        return [p for p in self.layers["noise"] if p.settings.get("browser")]

    def free_parts(self) -> list[Part]:
        """The subtitle parts the browser transcribes itself (quality free)."""
        return [p for p in self.layers[SUBTITLES] if p.settings.get("quality") == "free"]

    def uses_v2(self) -> bool:
        """Whether the timeline has something the first version could not say (subtitles, a browser-cleaned part)."""
        return bool(self.layers[SUBTITLES]) or any(p.settings.get("browser") for p in self.layers["noise"])

    def check(self) -> "Timeline":
        """Raises TimelineError for what no single part can tell: an echo part overlapping a part the browser cleaned (echo removal
        runs before the noise removal, and the browser can't do it)."""
        for b in self.browser_parts():
            if any(e.start < b.end and b.start < e.end for e in self.layers["echo"]):
                raise TimelineError("Echo removal can't be used on a part cleaned in the browser")
        return self

    def samples(self, layer: str, n: int) -> list[tuple[int, int, dict]]:
        """The parts of a layer as sample ranges (a, b, settings) of a recording of `n` samples, clamped, empty ones dropped."""
        out = []
        for p in self._layer(layer):
            a, b = max(0, min(n, int(round(p.start * SAMPLE_RATE)))), max(0, min(n, int(round(p.end * SAMPLE_RATE))))
            if b > a:
                out.append((a, b, p.settings))
        return out

    def __iter__(self) -> Iterator[tuple[str, Part]]:
        for name in LAYERS:
            for p in self.layers[name]:
                yield name, p

    def __eq__(self, other: object) -> bool:
        return isinstance(other, Timeline) and self.seconds == other.seconds and self.layers == other.layers

    def __repr__(self) -> str:
        return f"Timeline({self.seconds}, {self.to_json()['layers']})"

    # ---- JSON ----------------------------------------------------------------------------------------------------
    def to_json(self) -> dict:
        return {"version": VERSION_2 if self.uses_v2() else VERSION, "seconds": self.seconds,
                "layers": {name: [p.to_json() for p in self.layers[name]] for name in ALL_LAYERS if self.layers[name]}}

    def dumps(self) -> str:
        return json.dumps(self.to_json(), sort_keys=True, separators=(",", ":"))

    @classmethod
    def from_json(cls, obj: dict | str | bytes, seconds: float | None = None) -> "Timeline":
        """Check and read a timeline. `seconds`: the real length of the recording (it wins over the one in the JSON, and
        everything is clamped to it)."""
        if isinstance(obj, (str, bytes)):
            try:
                obj = json.loads(obj)
            except ValueError as exc:
                raise TimelineError("the timeline is not valid JSON") from exc
        if not isinstance(obj, dict):
            raise TimelineError("the timeline must be an object")
        extra = sorted(set(obj) - {"version", "seconds", "layers"})
        if extra:
            raise TimelineError(f"unknown key {extra[0]!r}")
        version = obj.get("version")
        if isinstance(version, bool) or version not in VERSIONS:
            raise TimelineError(f"unsupported timeline version {version!r} (this code reads versions {VERSION} and {VERSION_2})")
        length = seconds if seconds is not None else obj.get("seconds")
        if length is None:
            raise TimelineError("the timeline needs seconds (the length of the recording)")
        t = cls(length)
        layers = obj.get("layers", {})
        if not isinstance(layers, dict):
            raise TimelineError("layers must be an object")
        for name, parts in layers.items():
            if name not in ALL_LAYERS:
                raise TimelineError(f"unknown layer {name!r}")
            if name == SUBTITLES and version < VERSION_2:
                raise TimelineError(f"the subtitles layer needs timeline version {VERSION_2}")
            if not isinstance(parts, list):
                raise TimelineError(f"{name} must be a list of parts")
            if len(parts) > 4 * MAX_PARTS:
                raise TimelineError(f"at most {MAX_PARTS} parts per layer")
            for p in parts:
                if not isinstance(p, dict) or "start" not in p or "end" not in p:
                    raise TimelineError(f"every part of {name} needs start and end")
                rest = {k: v for k, v in p.items() if k not in ("start", "end")}
                t.paint(name, p["start"], p["end"], rest)  # later parts replace earlier ones
        if version < VERSION_2 and t.uses_v2():
            raise TimelineError(f"a part cleaned in the browser needs timeline version {VERSION_2}")
        return t.check()
