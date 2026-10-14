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
from pathlib import Path

import httpx

from ..captions import Segment, segments_to_json
from ..config import get_settings

log = logging.getLogger(__name__)

MODEL = "gpt-transcribe"
URL = "https://api.openai.com/v1/audio/transcriptions"
CONCURRENCY = 6


class CloudASRError(Exception):
    pass


async def _transcribe_one(http: httpx.AsyncClient, key: str, path: Path, language: str | None) -> str:
    data = {"model": MODEL, "response_format": "json"}
    if language:
        data["languages[]"] = language  # a hint; speech in another language is still transcribed as spoken
    r = None
    for attempt in range(4):
        with path.open("rb") as f:
            r = await http.post(URL, headers={"Authorization": f"Bearer {key}"}, data=data,
                                files={"file": (path.name, f, "audio/mpeg")})
        if r.status_code < 500 and r.status_code != 429:
            break
        log.warning("openai transcription retry %s: %s", attempt + 1, r.status_code)
        await asyncio.sleep(2 ** attempt)
    if r is None or r.status_code != 200:
        raise CloudASRError(f"transcription API returned {r.status_code if r else 'nothing'}")
    return (r.json().get("text") or "").strip()


async def transcribe_chunks(chunk_dir: Path, language: str | None = None) -> list[Segment]:
    key = get_settings().openai_api_key
    if not key:
        raise CloudASRError("premium transcription is not configured")
    chunks = json.loads((chunk_dir / "index.json").read_text())["chunks"]
    paths = []
    for chunk in chunks:
        path = chunk_dir / chunk["file"]
        if path.parent != chunk_dir or not path.is_file():
            raise CloudASRError("bad chunk index")
        paths.append(path)
    gate = asyncio.Semaphore(CONCURRENCY)

    async def one(path: Path) -> str:
        async with gate:
            return await _transcribe_one(http, key, path, language)

    async with httpx.AsyncClient(timeout=httpx.Timeout(120, connect=20)) as http:
        texts = await asyncio.gather(*(one(p) for p in paths))
    return [
        Segment(float(c["offset"]), float(c["offset"]) + float(c["seconds"]), text)
        for c, text in zip(chunks, texts)
        if text
    ]


async def run_for_job(job_dir: Path, language: str | None) -> None:
    segs = await transcribe_chunks(job_dir / "asr_chunks", language)
    payload = {
        "language": language,
        "engine": "premium",
        "segments": segments_to_json(segs),
    }
    tmp = job_dir / "segments.json.tmp"
    tmp.write_text(json.dumps(payload))
    tmp.replace(job_dir / "segments.json")
