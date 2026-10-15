"""Shared helpers for the ASR benchmarks: manifests, text normalisation, WER, resource tracking.

All scripts expect the venv at ~/audio-cleaner/research/venv and data prepared by prep_data.py.
"""
import json
import os
import re
import subprocess
import threading
import time
from pathlib import Path

import numpy as np
import soundfile as sf

ROOT = Path.home() / "audio-cleaner/research"
EVAL = ROOT / "data/eval"
MODELS = ROOT / "models"
RESULTS = Path(__file__).resolve().parent / "results"
RESULTS.mkdir(exist_ok=True)

SETS_EN = ["clean", "noisy5", "noisy5_dfn3"]
SETS_FLEURS = ["fleurs_es", "fleurs_de", "fleurs_fr"]


def load_manifest(name):
    return [json.loads(l) for l in open(EVAL / f"manifest_{name}.jsonl")]


def load_audio(path):
    x, sr = sf.read(str(path), dtype="float32")
    assert sr == 16000 and x.ndim == 1, (path, sr, x.shape)
    return x


_en_norm = None
_basic_norm = None


def normalize(text, lang="en"):
    """Whisper normalisers: EnglishTextNormalizer for en, BasicTextNormalizer otherwise."""
    global _en_norm, _basic_norm
    if lang == "en":
        if _en_norm is None:
            from whisper_normalizer.english import EnglishTextNormalizer
            _en_norm = EnglishTextNormalizer()
        return _en_norm(text)
    if _basic_norm is None:
        from whisper_normalizer.basic import BasicTextNormalizer
        _basic_norm = BasicTextNormalizer()
    return _basic_norm(text)


def wer(refs, hyps, lang="en"):
    """Corpus WER (%) after normalisation. Returns (wer, n_words, errors dict)."""
    import jiwer
    r = [normalize(x, lang) for x in refs]
    h = [normalize(x, lang) for x in hyps]
    pairs = [(a, b) for a, b in zip(r, h) if a.strip()]
    r, h = [p[0] for p in pairs], [p[1] if p[1].strip() else "" for p in pairs]
    out = jiwer.process_words(r, h)
    n = out.hits + out.substitutions + out.deletions
    return 100.0 * (out.substitutions + out.deletions + out.insertions) / max(n, 1), n, dict(
        sub=out.substitutions, dele=out.deletions, ins=out.insertions)


def loadavg():
    return open("/proc/loadavg").read().split()[:3]


def peak_rss_mb():
    for line in open("/proc/self/status"):
        if line.startswith("VmHWM"):
            return int(line.split()[1]) / 1024.0
    return float("nan")


def rss_mb():
    for line in open("/proc/self/status"):
        if line.startswith("VmRSS"):
            return int(line.split()[1]) / 1024.0
    return float("nan")


class GpuMem:
    """Samples total used VRAM via NVML (pynvml / nvidia-ml-py) in a thread; reports peak above baseline."""

    def __init__(self, interval=0.1):
        self.interval, self.peak, self.base, self._stop = interval, 0, 0, False
        try:
            import pynvml
            pynvml.nvmlInit()
            self.h = pynvml.nvmlDeviceGetHandleByIndex(0)
            self.nv = pynvml
        except Exception:
            self.h = None

    def _used(self):
        return self.nv.nvmlDeviceGetMemoryInfo(self.h).used / 2**20

    def __enter__(self):
        if self.h is not None:
            self.base = self._used()
            self.peak = self.base
            self._t = threading.Thread(target=self._run, daemon=True)
            self._t.start()
        return self

    def _run(self):
        while not self._stop:
            self.peak = max(self.peak, self._used())
            time.sleep(self.interval)

    def __exit__(self, *a):
        self._stop = True
        if self.h is not None:
            self._t.join()
            self.peak = max(self.peak, self._used())

    @property
    def peak_above_base_mb(self):
        return self.peak - self.base if self.h is not None else None


def words_from_tokens(tokens, starts, end_time=None, durs=None, min_dur=0.04, max_tail=0.4):
    """Group SentencePiece tokens ('\u2581' or a leading space marks word start) into words (text, start, end).
    With per-token durations (sherpa-onnx TDT) the word end is the end of its last token; otherwise (onnx-asr
    only returns token start times) it is the start of the next word, capped at start+1.0 s."""
    words = []
    for i, (tok, t) in enumerate(zip(tokens, starts)):
        d = durs[i] if durs is not None and i < len(durs) else None
        if tok[:1] in ("\u2581", " ") or not words:  # sherpa keeps '\u2581', onnx-asr turns it into ' '
            words.append([tok.lstrip("\u2581 "), float(t), float(t), float(t) + (d or 0.0)])
        else:
            words[-1][0] += tok
            words[-1][2] = float(t)
            words[-1][3] = float(t) + (d or 0.0)
    out = []
    for i, (w, s_, last, tok_end) in enumerate(words):
        if durs is not None:
            e = max(tok_end, s_ + min_dur)
        else:
            nxt = words[i + 1][1] if i + 1 < len(words) else (end_time if end_time else last + max_tail)
            e = min(max(nxt, s_ + min_dur), s_ + 1.0)
        out.append((w, round(s_, 3), round(e, 3)))
    return out


def save_result(tag, obj):
    obj["loadavg_end"] = loadavg()
    p = RESULTS / f"{tag}.json"
    json.dump(obj, open(p, "w"), indent=1)
    print("saved", p)
