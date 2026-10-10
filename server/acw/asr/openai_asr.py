"""Premium file transcription through the OpenAI API.

Runs in the web process (the only service allowed to reach the internet). It
only ever reads audio chunks the sandboxed runner produced (16 kHz mono mp3,
each well under the 25 MB API limit), never the user's original upload.

`gpt-4o-transcribe-diarize` is used because it is the current non-legacy
model that returns segment timings (diarized_json), which subtitles need;
speaker labels come along for free.
"""
from __future__ import annotations

import json
import logging
from pathlib import Path

import httpx

from ..captions import Segment, segments_to_json
from ..config import get_settings

log = logging.getLogger(__name__)

MODEL = "gpt-4o-transcribe-diarize"
URL = "https://api.openai.com/v1/audio/transcriptions"


class CloudASRError(Exception):
    pass


async def transcribe_chunks(chunk_dir: Path, language: str | None = None) -> list[Segment]:
    key = get_settings().openai_api_key
    if not key:
        raise CloudASRError("premium transcription is not configured")
    index = json.loads((chunk_dir / "index.json").read_text())
    segments: list[Segment] = []
    async with httpx.AsyncClient(timeout=httpx.Timeout(600, connect=20)) as http:
        for chunk in index["chunks"]:
            path = chunk_dir / chunk["file"]
            if path.parent != chunk_dir or not path.is_file():
                raise CloudASRError("bad chunk index")
            data = {"model": MODEL, "response_format": "diarized_json", "chunking_strategy": "auto"}
            if language:
                data["language"] = language
            for attempt in range(3):
                with path.open("rb") as f:
                    r = await http.post(
                        URL,
                        headers={"Authorization": f"Bearer {key}"},
                        data=data,
                        files={"file": (path.name, f, "audio/mpeg")},
                    )
                if r.status_code < 500 and r.status_code != 429:
                    break
                log.warning("openai transcription retry %s: %s", attempt + 1, r.status_code)
            if r.status_code != 200:
                raise CloudASRError(f"transcription API returned {r.status_code}")
            body = r.json()
            offset = float(chunk["offset"])
            for s in body.get("segments", []):
                text = (s.get("text") or "").strip()
                if not text:
                    continue
                segments.append(
                    Segment(offset + float(s["start"]), offset + float(s["end"]), text, speaker=s.get("speaker"))
                )
    return segments


async def run_for_job(job_dir: Path, language: str | None) -> None:
    segs = await transcribe_chunks(job_dir / "asr_chunks", language)
    payload = {
        "language": language,
        "engine": MODEL,
        "segments": segments_to_json(segs),
    }
    tmp = job_dir / "segments.json.tmp"
    tmp.write_text(json.dumps(payload))
    tmp.replace(job_dir / "segments.json")
