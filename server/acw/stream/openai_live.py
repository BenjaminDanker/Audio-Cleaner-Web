"""Premium live captions through the OpenAI Realtime API (transcription session).

Fed with the echo-free 48 kHz original (before noise removal, like every other caption path: recognisers
score worse on denoised speech); downsampled statefully to 24 kHz PCM16 as the API expects. Turns are committed by our own pause detector (the
transcription model does not do server-side VAD). Interim text arrives as
deltas and is forwarded as non-final captions.

Endpoint and model are configurable because this API surface moves fast:
ACW_OPENAI_REALTIME_URL, ACW_OPENAI_LIVE_MODEL.
"""
from __future__ import annotations

import asyncio
import base64
import hashlib
import json
import logging
import os

import numpy as np
from scipy.signal import butter, sosfilt, sosfilt_zi

from .. import alerts
from ..config import get_settings

log = logging.getLogger("acw.stream.premium")

REALTIME_URL = os.environ.get("ACW_OPENAI_REALTIME_URL", "wss://api.openai.com/v1/realtime?intent=transcription")
LIVE_MODEL = os.environ.get("ACW_OPENAI_LIVE_MODEL", "gpt-live-transcribe")


class PremiumCaptions:
    SPEECH_DB = -45.0
    END_SILENCE_S = 0.6
    MAX_TURN_S = 12.0

    SEND_EVERY_S = 0.1  # audio is sent in 100 ms pieces: a fifth of the messages of 20 ms ones, for 80 ms of delay

    def __init__(self, languages: list[str] | str | None, emit, user_hint: str = "", on_fail=None):
        # One language, or the few a speaker switches between (the model then follows the switches).
        self.languages = [languages] if isinstance(languages, str) else list(languages or [])
        self.emit = emit
        self.on_fail = on_fail  # called once, when the captions stop working (the session stops billing them)
        self.user_hint = hashlib.sha256(user_hint.encode()).hexdigest()[:32] if user_hint else None
        self.ws = None
        self.reader_task: asyncio.Task | None = None
        self.sos = butter(8, 11_000, btype="lowpass", fs=48_000, output="sos")
        self.zi = sosfilt_zi(self.sos) * 0.0
        self.in_turn = False
        self.turn_s = 0.0
        self.silence_s = 0.0
        self.partial: dict[str, str] = {}
        self.t = 0.0
        self.failed = False
        self._pending = bytearray()  # audio not sent yet
        self._pending_s = 0.0

    async def connect(self) -> None:
        from websockets.asyncio.client import connect

        headers = {"Authorization": f"Bearer {get_settings().openai_api_key}"}
        if self.user_hint:
            headers["OpenAI-Safety-Identifier"] = self.user_hint
        try:
            # No compression: base64 audio shrinks little, and deflating every message costs our event loop.
            self.ws = await connect(REALTIME_URL, additional_headers=headers, max_size=2**22, open_timeout=15, compression=None)
            transcription = {"model": LIVE_MODEL}
            if self.languages:
                transcription["languages"] = self.languages
            await self.ws.send(json.dumps({
                "type": "session.update",
                "session": {
                    "type": "transcription",
                    "audio": {"input": {"format": {"type": "audio/pcm", "rate": 24000},
                                        "transcription": transcription, "turn_detection": None}},
                },
            }))
            self.reader_task = asyncio.create_task(self._read())
        except Exception as e:  # noqa: BLE001
            log.exception("premium captions connect failed")
            resp = getattr(e, "response", None)  # websockets InvalidStatus: the HTTP answer
            if resp is not None:
                alerts.report_openai("live captions", getattr(resp, "status_code", None), getattr(resp, "body", None))
            self._fail()

    def _fail(self) -> None:
        if not self.failed:
            self.failed = True
            if self.on_fail is not None:
                self.on_fail()
            self.emit({"type": "warning", "code": "captions_unavailable",
                       "message": "Extra-accurate captions are unavailable for now. You won’t be charged for them from here on."})

    async def _read(self) -> None:
        try:
            async for raw in self.ws:
                ev = json.loads(raw)
                t = ev.get("type", "")
                if t.endswith("input_audio_transcription.delta"):
                    item = ev.get("item_id", "")
                    self.partial[item] = self.partial.get(item, "") + ev.get("delta", "")
                    self.emit({"type": "caption", "text": self.partial[item].strip(), "final": False, "id": item})
                elif t.endswith("input_audio_transcription.completed"):
                    item = ev.get("item_id", "")
                    self.partial.pop(item, None)
                    text = (ev.get("transcript") or "").strip()
                    if text:
                        self.emit({"type": "caption", "text": text, "final": True, "id": item})
                elif t == "error":
                    log.warning("realtime error: %s", ev.get("error"))
                    err = ev.get("error") if isinstance(ev.get("error"), dict) else {}
                    alerts.report_openai("live captions", None, err_type=err.get("type"), err_code=err.get("code"))
        except Exception:  # noqa: BLE001
            log.exception("premium captions stream ended")
        if not self.failed:
            self._fail()

    async def feed(self, y48: np.ndarray) -> None:
        if self.ws is None or self.failed:
            return
        lp, self.zi = sosfilt(self.sos, y48, zi=self.zi)
        y24 = np.clip(lp[::2], -1.0, 1.0)
        pcm = (y24 * 32767.0).astype("<i2").tobytes()
        dur = len(y48) / 48_000
        rms = float(np.sqrt(np.mean(y48 * y48) + 1e-12))
        speaking = 20 * np.log10(rms) > self.SPEECH_DB
        if not speaking and not self.in_turn:
            return  # don't bill the API for silence between turns
        try:
            self._pending += pcm
            self._pending_s += dur
            if speaking:
                self.in_turn, self.silence_s = True, 0.0
            elif self.in_turn:
                self.silence_s += dur
            commit = False
            if self.in_turn:
                self.turn_s += dur
                commit = self.silence_s >= self.END_SILENCE_S or self.turn_s >= self.MAX_TURN_S
            if commit or self._pending_s >= self.SEND_EVERY_S:
                await self.ws.send(json.dumps({"type": "input_audio_buffer.append",
                                               "audio": base64.b64encode(bytes(self._pending)).decode()}))
                self._pending.clear()
                self._pending_s = 0.0
            if commit:
                await self.ws.send(json.dumps({"type": "input_audio_buffer.commit"}))
                self.in_turn, self.turn_s, self.silence_s = False, 0.0, 0.0
        except Exception:  # noqa: BLE001
            self._fail()

    async def close(self) -> None:
        if self.reader_task:
            self.reader_task.cancel()
        if self.ws is not None:
            try:
                await self.ws.close()
            except Exception:  # noqa: BLE001
                pass
