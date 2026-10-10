"""Caption formatting: timed words/segments -> readable cues -> SRT / WebVTT / text."""
from __future__ import annotations

from dataclasses import dataclass, field

MAX_LINE = 42
MAX_LINES = 2
MAX_CUE_SECONDS = 6.0
MIN_CUE_SECONDS = 0.8


@dataclass
class Word:
    start: float
    end: float
    text: str


@dataclass
class Segment:
    start: float
    end: float
    text: str
    words: list[Word] = field(default_factory=list)
    speaker: str | None = None


@dataclass
class Cue:
    start: float
    end: float
    lines: list[str]


def _wrap(text: str) -> list[str]:
    words, lines, cur = text.split(), [], ""
    for w in words:
        if cur and len(cur) + 1 + len(w) > MAX_LINE:
            lines.append(cur)
            cur = w
        else:
            cur = f"{cur} {w}" if cur else w
    if cur:
        lines.append(cur)
    return lines


def _words_of(seg: Segment) -> list[Word]:
    if seg.words:
        return seg.words
    # No word timings: spread the segment duration over its words by length.
    toks = seg.text.split()
    if not toks:
        return []
    total = sum(len(t) + 1 for t in toks)
    t, out = seg.start, []
    for tok in toks:
        d = (seg.end - seg.start) * (len(tok) + 1) / total
        out.append(Word(t, t + d, tok))
        t += d
    return out


def build_cues(segments: list[Segment]) -> list[Cue]:
    """Pack words into cues of at most two 42-char lines and 6 s, breaking
    early at sentence ends so cues read naturally."""
    cues: list[Cue] = []
    for seg in segments:
        buf: list[Word] = []

        def flush() -> None:
            if not buf:
                return
            text = " ".join(w.text.strip() for w in buf).strip()
            if text:
                cues.append(Cue(buf[0].start, max(buf[-1].end, buf[0].start + MIN_CUE_SECONDS), _wrap(text)))
            buf.clear()

        for w in _words_of(seg):
            if not w.text.strip():
                continue
            candidate = " ".join(x.text.strip() for x in [*buf, w])
            if buf and (len(_wrap(candidate)) > MAX_LINES or w.end - buf[0].start > MAX_CUE_SECONDS):
                flush()
            buf.append(w)
            if w.text.rstrip().endswith((".", "?", "!")) and w.end - buf[0].start > 1.5:
                flush()
        flush()
    # Never overlap: clip each cue to the next one's start.
    for a, b in zip(cues, cues[1:]):
        if a.end > b.start:
            a.end = max(a.start + 0.1, b.start - 0.01)
    return cues


def _ts(t: float, sep: str) -> str:
    ms = int(round(max(t, 0.0) * 1000))
    h, ms = divmod(ms, 3_600_000)
    m, ms = divmod(ms, 60_000)
    s, ms = divmod(ms, 1000)
    return f"{h:02d}:{m:02d}:{s:02d}{sep}{ms:03d}"


def to_srt(cues: list[Cue]) -> str:
    out = []
    for i, c in enumerate(cues, 1):
        out.append(f"{i}\n{_ts(c.start, ',')} --> {_ts(c.end, ',')}\n" + "\n".join(c.lines) + "\n")
    return "\n".join(out)


def to_vtt(cues: list[Cue]) -> str:
    out = ["WEBVTT\n"]
    for c in cues:
        out.append(f"{_ts(c.start, '.')} --> {_ts(c.end, '.')}\n" + "\n".join(c.lines) + "\n")
    return "\n".join(out)


def to_text(segments: list[Segment]) -> str:
    """Plain transcript with paragraph breaks at long pauses."""
    paras, cur, last_end = [], [], None
    for s in segments:
        t = s.text.strip()
        if not t:
            continue
        if last_end is not None and s.start - last_end > 2.0 and cur:
            paras.append(" ".join(cur))
            cur = []
        cur.append(t)
        last_end = s.end
    if cur:
        paras.append(" ".join(cur))
    return "\n\n".join(paras) + "\n"


def segments_to_json(segments: list[Segment]) -> list[dict]:
    return [
        {
            "start": s.start,
            "end": s.end,
            "text": s.text,
            "words": [[w.start, w.end, w.text] for w in s.words],
            **({"speaker": s.speaker} if s.speaker else {}),
        }
        for s in segments
    ]


def segments_from_json(data: list[dict]) -> list[Segment]:
    return [
        Segment(
            float(d["start"]),
            float(d["end"]),
            str(d["text"]),
            [Word(float(a), float(b), str(c)) for a, b, c in d.get("words", [])],
            d.get("speaker"),
        )
        for d in data
    ]
