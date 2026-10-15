"""Caption formatting: timed words/segments -> readable cues -> SRT / WebVTT / text."""
from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass, field

MAX_LINE = 42
MAX_LINES = 2
MAX_CUE_SECONDS = 6.0
MIN_CUE_SECONDS = 0.8


# Languages written without spaces between words: the recogniser hands back single characters or short
# runs, which are joined without a space and wrapped by character count.
UNSPACED_LANGUAGES = frozenset({"zh", "yue", "ja", "th", "lo", "km", "my"})
_UNSPACED_CHAR = (
    "\u0e00-\u0eff"  # Thai, Lao
    "\u1000-\u109f\u1780-\u17ff"  # Burmese, Khmer
    "\u2e80-\u2fdf\u3000-\u303f\u3040-\u30ff\u3100-\u312f\u31f0-\u31ff\u3400-\u4dbf\u4e00-\u9fff"
    "\uf900-\ufaff\uff00-\uffef"  # CJK, kana, bopomofo, full-width forms
)
_UNSPACED_RE = re.compile(f"[{_UNSPACED_CHAR}]")
_TOKEN_RE = re.compile(f"[{_UNSPACED_CHAR}]|[^\\s{_UNSPACED_CHAR}]+")


def is_unspaced(language: str | None, text: str = "") -> bool:
    """True when `language` (or, for language None, the text itself) is written without spaces."""
    if language:
        return language.split("-")[0].lower() in UNSPACED_LANGUAGES
    letters = [c for c in text if not c.isspace()]
    return bool(letters) and sum(1 for c in letters if _UNSPACED_RE.match(c)) * 2 > len(letters)


def _join(parts: list[str], unspaced: bool) -> str:
    if not unspaced:
        return " ".join(parts)
    out = ""
    for p in parts:
        if out and out[-1].isascii() and out[-1].isalnum() and p[:1].isascii() and p[:1].isalnum():
            out += " "  # two Latin words inside unspaced text keep their space
        out += p
    return out


def _width(ch: str) -> int:
    return 2 if unicodedata.east_asian_width(ch) in ("W", "F") else 1


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


def _wrap_unspaced(text: str) -> list[str]:
    """Break at the line width in columns (a CJK character is two columns wide), at a space when there is one."""
    lines, cur, cols = [], "", 0
    for ch in text:
        w = _width(ch)
        if cur and cols + w > MAX_LINE:
            cut = cur.rfind(" ")
            if cut > 0 and ch != " ":
                lines.append(cur[:cut])
                cur = cur[cut + 1:]
                cols = sum(_width(c) for c in cur)
            else:
                lines.append(cur)
                cur, cols = "", 0
        if ch == " " and not cur:
            continue
        cur += ch
        cols += w
    if cur.strip():
        lines.append(cur.rstrip())
    return lines


def _wrap(text: str, unspaced: bool = False) -> list[str]:
    if unspaced:
        return _wrap_unspaced(text)
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


def _words_of(seg: Segment, unspaced: bool = False) -> list[Word]:
    if seg.words:
        return seg.words
    # No word timings: spread the segment duration over its words by length.
    toks = _TOKEN_RE.findall(seg.text) if unspaced else seg.text.split()
    if not toks:
        return []
    total = sum(len(t) + 1 for t in toks)
    t, out = seg.start, []
    for tok in toks:
        d = (seg.end - seg.start) * (len(tok) + 1) / total
        out.append(Word(t, t + d, tok))
        t += d
    return out


def build_cues(segments: list[Segment], language: str | None = None) -> list[Cue]:
    """Pack words into cues of at most two 42-char lines and 6 s, breaking
    early at sentence ends so cues read naturally. For Chinese, Japanese, Thai and other languages written
    without spaces (`language`, or the text itself when it is None) the words are joined without spaces and
    lines are broken by width."""
    cues: list[Cue] = []
    for seg in segments:
        buf: list[Word] = []
        unspaced = is_unspaced(language, seg.text)

        def flush() -> None:
            if not buf:
                return
            text = _join([w.text.strip() for w in buf], unspaced).strip()
            if text:
                cues.append(Cue(buf[0].start, max(buf[-1].end, buf[0].start + MIN_CUE_SECONDS), _wrap(text, unspaced)))
            buf.clear()

        for w in _words_of(seg, unspaced):
            if not w.text.strip():
                continue
            candidate = _join([x.text.strip() for x in [*buf, w]], unspaced)
            if buf and (len(_wrap(candidate, unspaced)) > MAX_LINES or w.end - buf[0].start > MAX_CUE_SECONDS):
                flush()
            buf.append(w)
            if w.text.rstrip().endswith((".", "?", "!", "\u3002", "\uff1f", "\uff01")) and w.end - buf[0].start > 1.5:
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


def to_text(segments: list[Segment], language: str | None = None) -> str:
    """Plain transcript with paragraph breaks at long pauses."""
    paras, cur, last_end = [], [], None
    for s in segments:
        t = s.text.strip()
        if not t:
            continue
        if last_end is not None and s.start - last_end > 2.0 and cur:
            paras.append(_join(cur, is_unspaced(language, cur[0])))
            cur = []
        cur.append(t)
        last_end = s.end
    if cur:
        paras.append(_join(cur, is_unspaced(language, cur[0])))
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
