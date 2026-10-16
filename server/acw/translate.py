"""Translations of captions and transcripts (files and live), through a text model of our speech partner.

Runs in the web process (the only service allowed to reach the internet). Only TEXT is sent: the lines that were
already transcribed, never audio. The model is ACW_OPENAI_TRANSLATE_MODEL, default gpt-6-luna with reasoning off:
in our test (2026-10-16, gaming/stream lines into Spanish, Japanese, German) it was the cheapest current model, about
1.3 s per live line, and it got slang ("cheesed", "one-tapped") and grammar right where gpt-5.4-mini/nano and
gpt-5.6-luna slipped; gpt-6.1-sol was a little better but ~20x the price and slower. About a cent per hour of speech
per language.

Files: `translate_job(job_dir, targets, source)` reads segments.json and writes segments.<lang>.json with the same
timing and the translated text, one language at a time; a language that fails is left out (the job then says so and
doesn't charge for it). Live: `translate_line()` translates one final caption, with the last few lines as context.
"""
from __future__ import annotations

import asyncio
import json
import logging
import os
import weakref
from pathlib import Path

import httpx

from . import alerts
from .config import get_settings

log = logging.getLogger(__name__)

URL = "https://api.openai.com/v1/chat/completions"
BATCH_LINES = 40  # lines per request for files: enough context, and a short answer to check
BACKOFF_SECONDS = (3, 15, 60)
# Live captions: late text is useless, and a line stuck behind 3 + 15 + 60 s of retries is up to 78 s late while the
# lines behind it are dropped. One quick retry, then the line is skipped.
LIVE_BACKOFF_SECONDS = (0.5,)
# Requests to the text model in flight at once in this process, all sessions and jobs together (20 live sessions with 3
# languages each would otherwise send bursts into the provider's rate limits and the web process's event loop).
MAX_IN_FLIGHT = 8
_gates: "weakref.WeakKeyDictionary[asyncio.AbstractEventLoop, asyncio.Semaphore]" = weakref.WeakKeyDictionary()


def _gate() -> asyncio.Semaphore:
    loop = asyncio.get_running_loop()
    gate = _gates.get(loop)
    if gate is None:
        gate = _gates[loop] = asyncio.Semaphore(MAX_IN_FLIGHT)
    return gate

# The languages offered (web/src/jobs.js LANGUAGES), with the names the model is told.
LANGUAGE_NAMES = {
    "en": "English", "es": "Spanish", "fr": "French", "de": "German", "it": "Italian", "pt": "Portuguese",
    "nl": "Dutch", "pl": "Polish", "ru": "Russian", "uk": "Ukrainian", "tr": "Turkish", "ar": "Arabic",
    "hi": "Hindi", "ja": "Japanese", "ko": "Korean", "zh": "Chinese (Simplified)", "sv": "Swedish", "da": "Danish",
    "fi": "Finnish", "no": "Norwegian", "cs": "Czech", "el": "Greek", "he": "Hebrew", "id": "Indonesian",
    "vi": "Vietnamese",
}
MAX_TARGETS = 3


class TranslationError(Exception):
    pass


def model() -> str:
    return os.environ.get("ACW_OPENAI_TRANSLATE_MODEL", "gpt-6-luna")


def _tuning() -> dict:
    """Reasoning models take no temperature; with reasoning off they answer as fast as a plain model."""
    m = model()
    if m.startswith(("gpt-5", "gpt-6", "o")):
        return {"reasoning_effort": os.environ.get("ACW_OPENAI_TRANSLATE_EFFORT", "none")}
    return {"temperature": 0.2}


def check_targets(targets: object, spoken: str | None = None) -> list[str]:
    """A clean list of target languages, or ValueError with a plain reason."""
    if targets is None:
        return []
    if not isinstance(targets, list) or not all(isinstance(t, str) for t in targets):
        raise ValueError("translate must be a list of language codes")
    out = list(dict.fromkeys(t.strip().lower() for t in targets if t.strip()))
    if len(out) > MAX_TARGETS:
        raise ValueError(f"at most {MAX_TARGETS} translation languages")
    for t in out:
        if t not in LANGUAGE_NAMES:
            raise ValueError(f"unknown translation language: {t[:8]}")
        if spoken and t == spoken[:2].lower():
            raise ValueError("a translation language is the same as the spoken language")
    return out


def _system_prompt(source: str | None, target: str) -> str:
    src = LANGUAGE_NAMES.get((source or "")[:2], "the original language") if source else "the original language"
    return (
        f"You translate subtitles of spoken audio from {src} into {LANGUAGE_NAMES[target]}. "
        "Translate the meaning naturally and briefly, as subtitles; keep names, numbers and technical words; "
        "do not add notes, quotes or explanations. Never refuse: if a line is unclear, translate it as well as you can."
    )


async def _post(http: httpx.AsyncClient, body: dict, backoff: tuple[float, ...] | None = None) -> dict:
    key = get_settings().openai_api_key
    if not key:
        raise TranslationError("translations are not configured")
    last: Exception | None = None
    for wait in (0, *(BACKOFF_SECONDS if backoff is None else backoff)):
        if wait:
            await asyncio.sleep(wait)  # waiting for a retry does not hold one of the in-flight places
        try:
            async with _gate():
                r = await http.post(URL, headers={"Authorization": f"Bearer {key}"}, json=body)
        except httpx.HTTPError as e:
            last = e
            continue
        if r.status_code == 200:
            return r.json()
        last = TranslationError(f"status {r.status_code}")
        try:
            err_body = r.json()
        except ValueError:
            err_body = None
        alerts.report_openai("translation", r.status_code, err_body)
        if r.status_code not in (408, 409, 429) and r.status_code < 500:
            break  # a request we got wrong: retrying won't help
    raise TranslationError(str(last))


async def translate_lines(http: httpx.AsyncClient, lines: list[str], source: str | None, target: str) -> list[str]:
    """The same number of lines, translated. Raises TranslationError if the model keeps answering wrongly."""
    if not lines:
        return []
    body = {
        "model": model(),
        **_tuning(),
        "response_format": {"type": "json_object"},
        "messages": [
            {"role": "system", "content": _system_prompt(source, target)
             + ' You get JSON {"lines": [...]} and answer JSON {"lines": [...]} with exactly one translated line per input line, in order.'},
            {"role": "user", "content": json.dumps({"lines": lines}, ensure_ascii=False)},
        ],
    }
    for _ in range(2):
        data = await _post(http, body)
        try:
            out = json.loads(data["choices"][0]["message"]["content"])["lines"]
        except (KeyError, IndexError, TypeError, ValueError):
            continue
        if isinstance(out, list) and len(out) == len(lines) and all(isinstance(x, str) for x in out):
            return [x.strip() for x in out]
    # The batch came back with a different number of lines twice: one line at a time keeps the timing right.
    return [await translate_line(http, line, source, target, live=False) for line in lines]


async def translate_line(http: httpx.AsyncClient, text: str, source: str | None, target: str, context: list[str] | None = None,
                         live: bool = True) -> str:
    """One caption line (live), with up to a few earlier lines as context (not translated again). Live lines are
    retried once, quickly; with live=False (a file's fallback) the full back-off applies."""
    ctx = "\n".join(context[-3:]) if context else ""
    body = {
        "model": model(),
        **_tuning(),
        "messages": [
            {"role": "system", "content": _system_prompt(source, target) + " Answer with the translation of the LAST line only."},
            {"role": "user", "content": (f"Earlier lines, for context only:\n{ctx}\n\n" if ctx else "") + f"Line to translate:\n{text}"},
        ],
    }
    data = await _post(http, body, LIVE_BACKOFF_SECONDS if live else None)
    try:
        return str(data["choices"][0]["message"]["content"]).strip()
    except (KeyError, IndexError, TypeError) as e:
        raise TranslationError("unexpected answer") from e


def _read_segments(job_dir: Path) -> dict:
    path = job_dir / "segments.json"
    fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW)  # written by the sandboxed runner: never follow a link
    with os.fdopen(fd, "rb") as f:
        if os.fstat(f.fileno()).st_size > 20_000_000:
            raise TranslationError("segments too large")
        return json.loads(f.read().decode("utf-8"))


def _write(path: Path, text: str) -> None:
    tmp = path.with_name(path.name + ".tmp")
    fd = os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_TRUNC | os.O_NOFOLLOW, 0o600)
    with os.fdopen(fd, "wb") as f:
        f.write(text.encode("utf-8"))
    os.replace(tmp, path)


async def translate_job(job_dir: Path, targets: list[str]) -> list[str]:
    """Writes segments.<lang>.json for each target that worked (kept across restarts: a language already done is
    not paid for twice). Returns the languages that are done."""
    data = await asyncio.to_thread(_read_segments, job_dir)
    source = data.get("language")
    segs = data.get("segments") or []
    texts = [str(s.get("text", "")) for s in segs]
    done = []
    async with httpx.AsyncClient(timeout=httpx.Timeout(90, connect=20)) as http:
        for lang in targets:
            out_path = job_dir / f"segments.{lang}.json"
            if out_path.exists():
                done.append(lang)
                continue
            try:
                translated: list[str] = []
                for i in range(0, len(texts), BATCH_LINES):
                    translated += await translate_lines(http, texts[i:i + BATCH_LINES], source, lang)
            except TranslationError:
                log.exception("translation into %s failed for %s", lang, job_dir.name)
                continue
            payload = {
                "language": lang,
                "engine": "translation",
                "source_language": source,
                # same timing; no word timings (they belong to the original words)
                "segments": [{**s, "text": t, "words": []} for s, t in zip(segs, translated)],
            }
            await asyncio.to_thread(_write, out_path, json.dumps(payload, ensure_ascii=False))
            done.append(lang)
    return done
