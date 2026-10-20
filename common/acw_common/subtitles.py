"""Subtitles per part (the timeline's `subtitles` layer, acw_common.timeline): the pure steps between a recogniser's segments and the
job's caption files, shared by the sandboxed runner, the web process and the tests. No models, no files except where said.

    ranges      a part and the little extra sound around it that a recogniser is given for context (`padded`)
    clipping    what a recogniser said about the padded sound, cut to the part (`clip_segments`)
    cues        the cues a browser transcribed for a free part, as segments (`segments_from_cues`)
    combining   the parts' segments (and each translated language's) as the job's `segments.json` / `segments.<lang>.json`
                (`combine`), which the usual caption writer turns into srt / vtt / txt per language

Everything is in FILE time (seconds from the start of the recording).
"""
from __future__ import annotations

import re

from .captions import Segment, Word, _join, is_unspaced, segments_to_json

#: Seconds of sound a recogniser also hears before and after a part (its words are cut away again).
PAD_SECONDS = 2.0
MAX_CUES = 20_000  # cues a browser may upload for one part
MAX_CUE_CHARS = 1_000
NAME_RE = re.compile(r"^[a-z0-9-]{1,40}$")


def padded(start: float, end: float, total: float, pad: float = PAD_SECONDS) -> tuple[float, float]:
    """The range of the recording a recogniser is given for the part [start, end): the part and `pad` seconds each side, inside the file."""
    return max(0.0, start - pad), min(total, end + pad)


def shift(segments: list[Segment], offset: float) -> list[Segment]:
    """The segments `offset` seconds later (a recogniser times a stretch from its own start; the file's time is that plus where it began)."""
    return [Segment(s.start + offset, s.end + offset, s.text, [Word(w.start + offset, w.end + offset, w.text) for w in s.words], s.speaker)
            for s in segments]


def clip_segments(segments: list[Segment], start: float, end: float) -> list[Segment]:
    """The segments that belong to the part [start, end): a segment whose middle lies inside it, its times cut to the part; its words are
    the ones whose middle lies inside (the text follows them when some were cut away). Speech of the padding around the part is dropped."""
    out: list[Segment] = []
    for s in segments:
        if not (start <= (s.start + s.end) / 2 < end):
            continue
        a, b = max(s.start, start), min(s.end, end)
        if b <= a:
            continue
        words = [w for w in s.words if start <= (w.start + w.end) / 2 < end]
        text = s.text
        if s.words and len(words) != len(s.words):
            if not words:
                continue
            text = _join([w.text.strip() for w in words], is_unspaced(None, s.text)).strip()
        words = [Word(max(w.start, a), min(w.end, b), w.text) for w in words]
        if text.strip():
            out.append(Segment(a, b, text, words, s.speaker))
    return out


def check_cues(raw: object, start: float, end: float) -> list[dict]:
    """The cues a browser uploaded for the free part [start, end): a list of {start, end, text} in file time, inside the part (a hair,
    0.5 s, past its edges is cut to it), the text not empty. Raises ValueError with a plain reason."""
    if not isinstance(raw, list):
        raise ValueError("The cues must be a list of {start, end, text}.")
    if len(raw) > MAX_CUES:
        raise ValueError(f"At most {MAX_CUES} cues for a part.")
    slack, out = 0.5, []
    for i, c in enumerate(raw):
        if not isinstance(c, dict) or set(c) - {"start", "end", "text"}:
            raise ValueError(f"Cue {i + 1} must be {{start, end, text}}.")
        a, b, text = c.get("start"), c.get("end"), c.get("text")
        if any(isinstance(v, bool) or not isinstance(v, (int, float)) or v != v or abs(v) == float("inf") for v in (a, b)):
            raise ValueError(f"Cue {i + 1}: start and end must be numbers.")
        if not isinstance(text, str):
            raise ValueError(f"Cue {i + 1}: text must be a string.")
        if a < start - slack or b > end + slack:
            raise ValueError(f"Cue {i + 1} is outside the part ({start:g} to {end:g} s).")
        a, b = max(a, start), min(b, end)
        if b <= a:
            raise ValueError(f"Cue {i + 1} must end after it starts.")
        text = " ".join(text.split())[:MAX_CUE_CHARS]
        if text:
            out.append({"start": round(a, 3), "end": round(b, 3), "text": text})
    out.sort(key=lambda c: (c["start"], c["end"]))
    return out


def segments_from_cues(cues: list[dict], start: float, end: float) -> list[Segment]:
    """A free part's uploaded cues as segments (no word timings: the caption writer spreads a cue's time over its words), cut to the part."""
    return clip_segments([Segment(float(c["start"]), float(c["end"]), str(c["text"])) for c in cues], start, end)


def dominant_language(parts: list[dict]) -> str | None:
    """The language most of the speech is in: each part's declared or detected language, weighted by its seconds."""
    seconds: dict[str, float] = {}
    for p in parts:
        lang = p.get("language")
        if lang:
            seconds[lang] = seconds.get(lang, 0.0) + max(0.0, float(p["end"]) - float(p["start"]))
    return max(seconds, key=lambda k: (seconds[k], k == "en")) if seconds else None


def combine(parts: list[dict]) -> tuple[dict | None, dict[str, dict]]:
    """The job's segment files from the parts. Each part is {start, end, language (declared or detected, or None), engine, segments:
    [Segment] (clipped, file time), translations: {lang: [Segment]} (the part's segments translated; same timing)}.

    Returns (segments.json's content, {lang: segments.<lang>.json's content}): the original text of every part in one list in time order,
    and each translated language holding the translations of the parts that asked for it. (None, {}) when no part said anything."""
    ordered = sorted(parts, key=lambda p: (p["start"], p["end"]))
    main = [s for p in ordered for s in p["segments"]]
    main.sort(key=lambda s: (s.start, s.end))
    langs: list[str] = []
    for p in ordered:
        for lang in p.get("translations") or {}:
            if lang not in langs:
                langs.append(lang)
    dom = dominant_language(ordered)
    if not main and not langs:
        return None, {}
    engines = {p.get("engine") for p in ordered if p.get("segments")}
    payload = {"language": dom, "engine": engines.pop() if len(engines) == 1 else "parts", "segments": segments_to_json(main)}
    translated: dict[str, dict] = {}
    for lang in langs:
        segs = [s for p in ordered for s in (p.get("translations") or {}).get(lang, [])]
        segs.sort(key=lambda s: (s.start, s.end))
        translated[lang] = {"language": lang, "engine": "translation", "source_language": dom,
                            "segments": [{**d, "words": []} for d in segments_to_json(segs)]}
    return payload, translated
