"""Live caption translations for one session (docs/streaming-protocol.md, "translation" events).

Every FINAL caption is translated into each chosen language, one language queue each, in order, with the last
few lines as context. A translation that fails is skipped; after FAIL_LIMIT failures in a row the language is
switched off for the rest of the session (a `translation_unavailable` warning), and nothing more is counted for it.
Billing (stream/routes.py) is by the seconds of speech that were actually translated, per language.
"""
from __future__ import annotations

import asyncio
import logging
import time

import httpx

from .. import translate

log = logging.getLogger(__name__)

FAIL_LIMIT = 3
QUEUE_LINES = 8  # a language that can't keep up drops its oldest waiting lines (live: late text is useless)
MAX_LINE_AGE = 20.0  # seconds: a line that waited this long is skipped, not translated (and not billed)
CONTEXT_LINES = 3


class LiveTranslator:
    def __init__(self, targets: list[str], source: str | None, emit):
        self.targets = list(targets)
        self.source = source
        self.emit = emit  # emit(event dict); called on the event loop
        self.seconds = {t: 0.0 for t in self.targets}  # of speech translated, per language
        self.off: set[str] = set()
        self._queues = {t: asyncio.Queue(maxsize=QUEUE_LINES) for t in self.targets}
        self._context: list[str] = []
        self._http = httpx.AsyncClient(timeout=httpx.Timeout(15, connect=5))
        self._tasks = [asyncio.create_task(self._run(t)) for t in self.targets]

    def offer(self, ev: dict) -> None:
        """A caption event from the recognizer: final ones are queued for every language that is still on."""
        if ev.get("type") != "caption" or not ev.get("final"):
            return
        text = str(ev.get("text") or "").strip()
        if not text:
            return
        item = (ev.get("id"), text, ev.get("start"), ev.get("end"), list(self._context), time.monotonic())
        self._context = (self._context + [text])[-CONTEXT_LINES:]
        for t, q in self._queues.items():
            if t in self.off:
                continue
            if q.full():
                q.get_nowait()  # too far behind: the oldest line goes
            q.put_nowait(item)

    async def _run(self, lang: str) -> None:
        q = self._queues[lang]
        fails = 0
        while True:
            cid, text, start, end, context, queued_at = await q.get()
            if time.monotonic() - queued_at > MAX_LINE_AGE:
                continue  # it waited behind slow answers: showing it now would only confuse
            try:
                out = await translate.translate_line(self._http, text, self.source, lang, context)
            except Exception:  # noqa: BLE001 - one line failing must not end the session
                fails += 1
                log.warning("live translation into %s failed (%d in a row)", lang, fails)
                if fails >= FAIL_LIMIT:
                    self.off.add(lang)
                    self.emit({"type": "warning", "code": "translation_unavailable", "lang": lang,
                               "message": f"{translate.LANGUAGE_NAMES.get(lang, lang)} translations stopped working, "
                                          "so they’re off for the rest of this session and not charged from here."})
                    return
                continue
            fails = 0
            if not out:
                continue
            if isinstance(start, (int, float)) and isinstance(end, (int, float)) and end > start:
                self.seconds[lang] += float(end) - float(start)
            else:
                # Extra-accurate captions come without times: count the line's speaking time from its length
                # (about 2.5 words a second, at least one second).
                self.seconds[lang] += max(1.0, len(text.split()) / 2.5)
            self.emit({"type": "translation", "lang": lang, "text": out, "of": cid,
                       **({"start": start} if start is not None else {}), **({"end": end} if end is not None else {})})

    def total_seconds(self) -> float:
        return sum(self.seconds.values())

    async def close(self, wait: float = 3.0) -> None:
        """Lets the last lines be translated (a few seconds at most), then stops."""
        try:
            await asyncio.wait_for(asyncio.gather(*(self._drain(t) for t in self.targets)), timeout=wait)
        except asyncio.TimeoutError:
            pass
        for task in self._tasks:
            task.cancel()
        await asyncio.gather(*self._tasks, return_exceptions=True)
        await self._http.aclose()

    async def _drain(self, lang: str) -> None:
        q = self._queues[lang]
        while not q.empty() and lang not in self.off:
            await asyncio.sleep(0.05)
