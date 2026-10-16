"""Capacity limits from the 2026-10-16 audit: caption streams are capped and shed instead of queued."""
import numpy as np

from acw import cpusplit
from acw.stream import engine, worker


def _fresh_worker(monkeypatch):
    monkeypatch.setattr(worker, "_monitor", None)
    monkeypatch.setattr(cpusplit, "units_per_core", lambda: 2.4)
    monkeypatch.setattr(worker, "_units", 0.0)
    monkeypatch.setattr(worker, "_active", 0)
    monkeypatch.setattr(worker, "_caption_streams", 0)


def test_local_caption_streams_are_capped_whatever_the_units_say(monkeypatch):
    _fresh_worker(monkeypatch)
    monkeypatch.setattr(worker, "MAX_CAPTION_STREAMS", 10)
    cfg = {"denoise": None, "captions": "local"}
    monkeypatch.setattr(worker, "_caption_streams", 9)
    assert worker.admit(cfg) == (True, [])
    monkeypatch.setattr(worker, "_caption_streams", 10)
    ok, options = worker.admit(cfg)
    assert not ok and options


def test_full_caption_streams_offer_what_still_fits(monkeypatch):
    _fresh_worker(monkeypatch)
    monkeypatch.setattr(worker, "_caption_streams", worker.caption_cap())
    # denoise + captions: the denoise alone still fits, so only the captions change
    ok, options = worker.admit({"denoise": "light", "captions": "local"})
    assert not ok and options == [{"captions": "premium"}]
    # captions only, nothing to keep: extra-accurate on its own
    ok, options = worker.admit({"denoise": None, "captions": "local"})
    assert not ok and options and all(o["captions"] == "premium" for o in options)
    # sessions without local captions are not affected by the caption cap
    assert worker.admit({"denoise": "light", "captions": None}) == (True, [])
    assert worker.admit({"denoise": None, "captions": "premium"}) == (True, [])


def test_caption_cap_can_be_set_with_the_environment(monkeypatch):
    import importlib

    monkeypatch.setenv("ACW_STREAM_MAX_CAPTION_STREAMS", "4")
    try:
        assert importlib.reload(worker).MAX_CAPTION_STREAMS == 4
    finally:
        monkeypatch.delenv("ACW_STREAM_MAX_CAPTION_STREAMS")
        assert importlib.reload(worker).MAX_CAPTION_STREAMS == 0


def test_without_a_setting_the_caption_cap_follows_the_workers_cores(monkeypatch):
    monkeypatch.setattr(worker, "MAX_CAPTION_STREAMS", 0)
    monkeypatch.setattr(worker, "_cores", [2, 3])
    assert worker.caption_cap() == 2 * worker.CAPTION_STREAMS_PER_CORE


def test_shed_backlog_keeps_the_newest_audio():
    chunks = [np.full(480, i, np.float32) for i in range(10)]
    kept, dropped = engine.shed_backlog(chunks, 1200)
    joined = np.concatenate(kept)
    assert len(joined) == 1200 and dropped == 4800 - 1200
    assert joined[-1] == 9 and joined[0] == 7  # the last 1200 samples: 240 of chunk 7, then 8 and 9
    kept, dropped = engine.shed_backlog(chunks[:2], 5000)
    assert dropped == 0 and len(kept) == 2


class _FakeStream:
    def __init__(self):
        self.fed = 0

    def accept_waveform(self, rate, x):
        self.fed += len(x)


class _FakeRecognizer:
    def create_stream(self):
        return _FakeStream()


class _FakeEngine:
    rec = _FakeRecognizer()

    def add(self, c):
        pass

    def remove(self, c):
        pass


def _talk():
    """A 20 ms frame of someone talking (loud enough not to count as silence)."""
    return np.full(960, 0.1, np.float32)


def _captioner(monkeypatch):
    monkeypatch.setattr(engine, "shared_recognizer", lambda *a, **k: _FakeEngine())
    events = []
    return engine.StreamingCaptioner(None, events.append), events


def test_a_stream_that_falls_behind_sheds_audio_and_warns_once_per_incident(monkeypatch):
    c, events = _captioner(monkeypatch)
    sec = engine.SAMPLE_RATE
    for _ in range(300):  # 6 s of 20 ms frames waiting for the decoder
        c.feed(_talk())
    c._take_audio()
    assert c.stream.fed <= engine.KEEP_BACKLOG_SAMPLES // 3 + 10  # about half a second reached the recognizer
    assert [e["code"] for e in events] == ["captions_behind"]
    assert events[0]["type"] == "warning" and events[0]["message"]
    assert c._fed >= 6 * sec // 3 - 10  # the caption clock still follows the talk
    for _ in range(300):  # still behind: same incident, no second warning
        c.feed(_talk())
    c._take_audio()
    assert len(events) == 1
    c.feed(_talk())  # caught up: a normal tick ends the incident
    c._take_audio()
    for _ in range(300):
        c.feed(_talk())
    c._take_audio()
    assert len(events) == 2


def test_a_stream_that_keeps_up_loses_nothing(monkeypatch):
    c, events = _captioner(monkeypatch)
    for _ in range(100):
        for _ in range(3):  # 60 ms between decoder ticks
            c.feed(_talk())
        c._take_audio()
    assert events == [] and abs(c.stream.fed - 100 * 3 * 960 // 3) < 40  # the decimator holds back a few samples


def test_silence_is_skipped_but_counted_and_the_words_after_it_are_kept(monkeypatch):
    c, _ = _captioner(monkeypatch)
    quiet = np.full(960, 10 ** (-70 / 20), np.float32)
    for _ in range(25):  # half a second of talk
        c.feed(_talk())
    for i in range(500):  # 10 s of silence: decoded until the line has ended, then skipped
        c.feed(quiet)
        if i % 25 == 0:
            c._take_audio()  # the decoder's ticks
    c._take_audio()
    decoded = c.stream.fed
    assert abs(decoded - (0.5 + engine.GATE_AFTER_S) * 16_000) < 400
    c.feed(_talk())  # someone talks again: the moments before it are decoded with it
    c._take_audio()
    assert c.stream.fed - decoded == (engine.PREROLL_FRAMES + 1) * 320
    assert abs(c._fed - 10.52 * 16_000) < 100  # the caption clock still counts the skipped silence
    assert abs(c.skipped_total - (500 - engine.GATE_AFTER_S * 50 - engine.PREROLL_FRAMES) * 960) <= 960


def test_a_denoised_level_decides_what_is_silence(monkeypatch):
    c, _ = _captioner(monkeypatch)
    noisy = np.full(960, 0.05, np.float32)  # a loud fan: not silent by itself
    for i in range(300):
        c.feed(noisy, -80.0)  # but nothing is left of it after noise removal
        if i % 25 == 0:
            c._take_audio()
    c._take_audio()
    assert c.skipped_total > 0 and c.stream.fed < 300 * 320
