"""Premium file transcription through the OpenAI API.

Runs in the web process (the only service allowed to reach the internet). It
only ever reads the speech segments the sandboxed runner produced (16 kHz mono
mp3, at most 15 s each), never the user's original upload.

`gpt-transcribe` is OpenAI's current file-transcription model (the gpt-4o
transcribe models and whisper-1 shut down on 2027-02-26). It returns text only,
so each request is one speech segment and the segment boundaries give the
subtitle timing.
"""
from __future__ import annotations

import asyncio
import json
import logging
import os
import stat
from pathlib import Path

import httpx

from acw_common.captions import Segment, segments_to_json
from .. import alerts
from ..config import get_settings

log = logging.getLogger(__name__)

MODEL = "gpt-transcribe"
URL = "https://api.openai.com/v1/audio/transcriptions"
CONCURRENCY = 4  # requests in flight per job (two jobs run at once)
# Waits between attempts on one segment: an outage of a few minutes costs the job some waiting, not the job.
BACKOFF_SECONDS = (5, 20, 60, 180)


class CloudASRError(Exception):
    pass


def _read_text_no_follow(path: Path) -> str | None:
    """The text saved for a segment, or None. Never follows a link (the job directory is written by the sandboxed runner)."""
    try:
        fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW)
    except OSError:
        return None
    with os.fdopen(fd, "rb") as f:
        if not stat.S_ISREG(os.fstat(f.fileno()).st_mode) or os.fstat(f.fileno()).st_size > 200_000:
            return None
        return f.read().decode("utf-8", errors="replace")


def _save_text(path: Path, text: str) -> None:
    tmp = path.with_name(path.name + ".tmp")
    fd = os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_TRUNC | os.O_NOFOLLOW, 0o600)
    with os.fdopen(fd, "wb") as f:
        f.write(text.encode("utf-8"))
    os.replace(tmp, path)


def _open_chunk(path: Path) -> int:
    """A chunk file as written by the runner: a regular file on the same device as the job directory, never a link."""
    fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW)
    try:
        st = os.fstat(fd)
        if not stat.S_ISREG(st.st_mode) or st.st_dev != os.stat(path.parent).st_dev or st.st_size > 20_000_000:
            raise CloudASRError("bad chunk file")
    except BaseException:
        os.close(fd)
        raise
    return fd


async def _transcribe_one(http: httpx.AsyncClient, key: str, path: Path, language: str | None) -> str:
    data = {"model": MODEL, "response_format": "json"}
    if language:
        data["languages[]"] = language  # a hint; speech in another language is still transcribed as spoken
    r = None
    for attempt in range(len(BACKOFF_SECONDS) + 1):
        try:
            try:
                fd = _open_chunk(path)
            except OSError:
                raise CloudASRError("bad chunk file")
            with os.fdopen(fd, "rb") as f:
                r = await http.post(URL, headers={"Authorization": f"Bearer {key}"}, data=data,
                                    files={"file": (path.name, f, "audio/mpeg")})
        except httpx.TransportError as e:  # timeouts, connection resets: the provider is struggling
            log.warning("openai transcription transport error (attempt %s): %s", attempt + 1, type(e).__name__)
            r = None
        else:
            if r.status_code >= 400:
                alerts.report_openai("file transcription", r.status_code, r.content)  # status/error type only
            if r.status_code < 500 and r.status_code != 429:
                break
            log.warning("openai transcription retry %s: %s", attempt + 1, r.status_code)
        if attempt < len(BACKOFF_SECONDS):
            await asyncio.sleep(BACKOFF_SECONDS[attempt])
    if r is None or r.status_code != 200:
        raise CloudASRError(f"transcription API returned {r.status_code if r else 'nothing'}")
    return (r.json().get("text") or "").strip()


async def transcribe_chunks(chunk_dir: Path, language: str | None = None, max_seconds: float | None = None) -> list[Segment]:
    key = get_settings().openai_api_key
    if not key:
        raise CloudASRError("premium transcription is not configured")
    index_text = _read_text_no_follow(chunk_dir / "index.json")
    if index_text is None:
        raise CloudASRError("bad chunk index")
    chunks = json.loads(index_text)["chunks"]
    paths = []
    for chunk in chunks:
        name = str(chunk["file"])
        if "/" in name or not name.endswith(".mp3") or name.startswith("."):
            raise CloudASRError("bad chunk index")
        path = chunk_dir / name
        if path.parent != chunk_dir:
            raise CloudASRError("bad chunk index")
        paths.append(path)
    # Each second sent costs us money: never send (much) more than the length the job is billed for.
    if max_seconds is not None and sum(float(c["seconds"]) for c in chunks) > max_seconds:
        raise CloudASRError("more speech than the billed length")
    gate = asyncio.Semaphore(CONCURRENCY)

    async def one(path: Path) -> str:
        # Segments finished earlier (before a restart, or by an attempt that later failed) are not paid for twice.
        saved = _read_text_no_follow(path.with_suffix(".txt"))
        if saved is not None:
            return saved.strip()
        async with gate:
            text = await _transcribe_one(http, key, path, language)
        await asyncio.to_thread(_save_text, path.with_suffix(".txt"), text)
        return text

    async with httpx.AsyncClient(timeout=httpx.Timeout(120, connect=20)) as http:
        texts = await asyncio.gather(*(one(p) for p in paths))
    return [
        Segment(float(c["offset"]), float(c["offset"]) + float(c["seconds"]), text)
        for c, text in zip(chunks, texts)
        if text
    ]


async def run_for_job(job_dir: Path, language: str | None, billed_seconds: float | None = None) -> None:
    limit = {} if billed_seconds is None else {"max_seconds": billed_seconds * 1.05 + 1}
    segs = await transcribe_chunks(job_dir / "asr_chunks", language, **limit)
    payload = {
        "language": language,
        "engine": "premium",
        "segments": segments_to_json(segs),
    }
    await asyncio.to_thread(_save_text, job_dir / "segments.json", json.dumps(payload))
