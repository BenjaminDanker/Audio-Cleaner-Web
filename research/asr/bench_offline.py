#!/usr/bin/env python
"""Offline (file transcription) ASR benchmark: WER on several sets, RTF, load time, RAM/VRAM, timestamps.

Usage (research venv; GPU runs must be wrapped:  flock ~/audio-cleaner/dev/gpu.lock <cmd>):
  source ~/audio-cleaner/research/env.sh     # LD_LIBRARY_PATH for the nvidia pip wheels
  python bench_offline.py --engine fw:/path/to/ct2-model:cpu:int8   --tag fw-turbo-cpu-int8 [--threads 8]
  python bench_offline.py --engine fw:/path/to/ct2-model:cuda:float32:batched8 --tag ...
  python bench_offline.py --engine onnxasr:/path/to/parakeet-onnx:int8:cpu --tag pk2-int8-cpu
  python bench_offline.py --engine sherpa:/path/to/sherpa-nemo-dir:int8:cpu --tag pk2-sherpa-int8
Options: --sets clean,noisy5,noisy5_dfn3,fleurs_es,fleurs_de,fleurs_fr   (default: the 3 English sets)
         --long   also run the 10-min long-form file (RTF + WER + word timestamps, production-like config)
         --limit N  only first N utterances per set (smoke tests)
         --ts       word timestamps for the short sets too (default off for speed; always on for --long)
Writes results/<tag>.json (WER per set, RTF per set, long-form RTF, load time, peak RSS/VRAM, CPU-seconds,
`uptime` load average before/after, sample hypotheses and word timings).

Engines
  fw      faster-whisper (CTranslate2): <dir>:<device>:<compute_type>[:batchedN]  (beam 5, no prev-text conditioning)
  onnxasr onnx-asr (NeMo TDT/RNNT/CTC ONNX exports): <dir>:<quant|none>:<cpu|cuda>; long-form = silero VAD segments
  sherpa  sherpa-onnx OfflineRecognizer.from_transducer (NeMo TDT): <dir>:<int8|fp32>:<cpu|cuda>
"""
import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from common import *  # noqa

SR = 16000


class FW:
    def __init__(self, spec, threads):
        from faster_whisper import WhisperModel
        parts = spec.split(":")
        path, dev, ct = parts[0], parts[1], parts[2]
        self.batch = int(parts[3][len("batched"):]) if len(parts) > 3 else 0
        t0 = time.time()
        self.model = WhisperModel(path, device=dev, compute_type=ct, cpu_threads=threads, num_workers=1)
        if self.batch:
            from faster_whisper import BatchedInferencePipeline  # faster-whisper >= 1.1 only
            self.pipe = BatchedInferencePipeline(self.model)
        else:
            self.pipe = None
        self.load_s = time.time() - t0

    def _run(self, audio, lang, words, vad):
        kw = dict(language=lang, beam_size=5, word_timestamps=words, condition_on_previous_text=False)
        if vad:
            kw.update(vad_filter=True, vad_parameters={"min_silence_duration_ms": 500})
        if self.pipe is not None:
            kw["batch_size"] = self.batch
            kw.pop("condition_on_previous_text")
            segs, info = self.pipe.transcribe(audio, **kw)
        else:
            segs, info = self.model.transcribe(audio, **kw)
        text, ws = [], []
        for s in segs:
            text.append(s.text.strip())
            for w in (s.words or []):
                ws.append((w.word.strip(), round(w.start, 3), round(w.end, 3)))
        return " ".join(text), ws

    def short(self, audio, lang, words=False):
        return self._run(audio, lang, words, vad=False)

    def long(self, audio, lang):
        return self._run(audio, lang, True, vad=True)


class OnnxAsr:
    def __init__(self, spec, threads):
        import onnx_asr
        import onnxruntime as ort
        path, quant, prov = spec.split(":")
        so = ort.SessionOptions()
        so.intra_op_num_threads = threads if prov == "cpu" else 2
        so.inter_op_num_threads = 1
        # cudnn_conv_algo_search=HEURISTIC is essential: the default EXHAUSTIVE re-benchmarks cuDNN for every new
        # input length (~50 s per utterance on the GTX 960 instead of 0.2 s)
        providers = ["CPUExecutionProvider"] if prov == "cpu" else [
            ("CUDAExecutionProvider", {"cudnn_conv_algo_search": "HEURISTIC"}), "CPUExecutionProvider"]
        t0 = time.time()
        self.m = onnx_asr.load_model("nemo-conformer-tdt", path, quantization=None if quant == "none" else quant,
                                     sess_options=so, providers=providers, cpu_preprocessing=(prov == "cpu") or None)
        # silero VAD (MIT) for long-form segmentation; CPU always
        self.vad = onnx_asr.load_vad("silero", path=str(MODELS / "silero-vad-onnx")) if (MODELS / "silero-vad-onnx").exists() else onnx_asr.load_vad("silero")
        self.load_s = time.time() - t0
        self.mt = self.m.with_timestamps()
        # VAD segments are decoded in batches; 8x30 s does not fit next to the 2.4 GB fp32 weights on a 4 GB card
        self.vad_batch = 8 if prov == "cpu" else 2

    def short(self, audio, lang, words=False):
        if words:
            r = self.mt.recognize(audio)
            return r.text, words_from_tokens(r.tokens, r.timestamps, len(audio) / SR)
        return self.m.recognize(audio), []

    def long(self, audio, lang):
        segs = self.m.with_vad(self.vad, max_speech_duration_s=30, min_silence_duration_ms=500, batch_size=self.vad_batch).with_timestamps().recognize(audio)
        text, ws = [], []
        for s in segs:
            text.append(s.text.strip())
            ws += [(w, round(a + s.start, 3), round(b + s.start, 3))
                   for w, a, b in words_from_tokens(s.tokens, s.timestamps, s.end - s.start)]
        return " ".join(text), ws


class Sherpa:
    def __init__(self, spec, threads):
        import sherpa_onnx
        path, quant, prov = spec.split(":")
        p = Path(path)
        sfx = ".int8.onnx" if quant == "int8" else ".onnx"
        t0 = time.time()
        self.rec = sherpa_onnx.OfflineRecognizer.from_transducer(
            encoder=str(p / f"encoder{sfx}"), decoder=str(p / f"decoder{sfx}"), joiner=str(p / f"joiner{sfx}"),
            tokens=str(p / "tokens.txt"), num_threads=threads, sample_rate=SR, feature_dim=128 if "v3" in path or "v2" in path else 80,
            decoding_method="greedy_search", model_type="nemo_transducer", provider=prov)
        vad_cfg = sherpa_onnx.VadModelConfig()
        vad_cfg.silero_vad.model = str(MODELS / "sherpa-dl/silero_vad.onnx")
        vad_cfg.silero_vad.min_silence_duration = 0.5
        vad_cfg.silero_vad.max_speech_duration = 30.0
        vad_cfg.sample_rate = SR
        self.vad_cfg = vad_cfg
        self.sherpa = sherpa_onnx
        self.load_s = time.time() - t0

    def _dec(self, audios):
        ss = []
        for a in audios:
            s = self.rec.create_stream()
            s.accept_waveform(SR, a)
            ss.append(s)
        self.rec.decode_streams(ss)
        return [s.result for s in ss]

    def short(self, audio, lang, words=False):
        r = self._dec([audio])[0]
        ws = words_from_tokens(r.tokens, r.timestamps, len(audio) / SR, getattr(r, 'durations', None) or None) if words else []
        return r.text.strip(), ws

    def long(self, audio, lang):
        vad = self.sherpa.VoiceActivityDetector(self.vad_cfg, buffer_size_in_seconds=60)
        win = self.vad_cfg.silero_vad.window_size
        segs = []
        for i in range(0, len(audio), win):
            vad.accept_waveform(audio[i:i + win])
            while not vad.empty():
                segs.append((vad.front.start / SR, vad.front.samples)); vad.pop()
        vad.flush()
        while not vad.empty():
            segs.append((vad.front.start / SR, vad.front.samples)); vad.pop()
        text, ws = [], []
        for k in range(0, len(segs), 8):
            chunk = segs[k:k + 8]
            for (st, a), r in zip(chunk, self._dec([c[1] for c in chunk])):
                text.append(r.text.strip())
                ws += [(w, round(x + st, 3), round(y + st, 3)) for w, x, y in words_from_tokens(r.tokens, r.timestamps, len(a) / SR, getattr(r, 'durations', None) or None)]
        return " ".join(text), ws


ENGINES = {"fw": FW, "onnxasr": OnnxAsr, "sherpa": Sherpa}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--engine", required=True)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--sets", default=",".join(SETS_EN))
    ap.add_argument("--long", action="store_true")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--ts", action="store_true")
    a = ap.parse_args()
    kind, spec = a.engine.split(":", 1)
    res = dict(tag=a.tag, engine=a.engine, threads=a.threads, loadavg_start=loadavg(), sets={})
    gm = GpuMem()
    with gm:
        c0 = time.time(); p0 = time.process_time()
        eng = ENGINES[kind](spec, a.threads)
        res["load_s"] = round(eng.load_s, 2)
        res["rss_after_load_mb"] = round(rss_mb())
        # warm-up (not timed): lets CUDA/ORT allocate kernels & workspaces
        w = load_audio(load_manifest("clean")[0]["path"])
        eng.short(w, "en", True)
        eng.short(w, "en", False)
        for name in a.sets.split(","):
            rows = load_manifest(name)
            if a.limit:
                rows = rows[:a.limit]
            lang = rows[0]["lang"]
            hyps, dur, wordlists = [], 0.0, {}
            audios = [load_audio(r["path"]) for r in rows]
            t0, cpu0 = time.time(), time.process_time()
            for r, x in zip(rows, audios):
                text, ws = eng.short(x, lang, a.ts)
                hyps.append(text); dur += len(x) / SR
                if a.ts:
                    wordlists[r["id"]] = ws
            el, cpu = time.time() - t0, time.process_time() - cpu0
            w_, n, errs = wer([r["ref"] for r in rows], hyps, lang)
            res["sets"][name] = dict(wer=round(w_, 2), n_words=n, errs=errs, audio_s=round(dur, 1), wall_s=round(el, 1),
                                     rtf=round(el / dur, 4), cpu_s_per_audio_s=round(cpu / dur, 3), n=len(rows),
                                     sample=[(rows[i]["ref"], hyps[i]) for i in range(min(3, len(rows)))])
            res["sets"][name]["hyps"] = {r["id"]: h for r, h in zip(rows, hyps)}
            if a.ts:
                (RESULTS / "words").mkdir(exist_ok=True)
                json.dump(wordlists, open(RESULTS / "words" / f"{a.tag}_{name}.json", "w"))
            print(f"[{a.tag}] {name}: WER {w_:.2f}% ({n} words) RTF {el / dur:.3f} ({el:.0f}s for {dur:.0f}s audio)", flush=True)
        save_result(a.tag + "_partial", dict(res))
        if a.long:
            x = load_audio(EVAL / "long/long10.wav")
            ref = json.load(open(EVAL / "long/long10.json"))
            res["loadavg_long_start"] = loadavg()
            t0, cpu0 = time.time(), time.process_time()
            text, ws = eng.long(x, "en")
            el, cpu = time.time() - t0, time.process_time() - cpu0
            w_, n, errs = wer([ref["ref"]], [text], "en")
            res["long"] = dict(wer=round(w_, 2), n_words=n, errs=errs, audio_s=round(len(x) / SR, 1), wall_s=round(el, 1),
                               rtf=round(el / (len(x) / SR), 4), cpu_s_per_audio_s=round(cpu / (len(x) / SR), 3),
                               n_words_ts=len(ws), words_head=ws[:25])
            res["long_words"] = ws
            res["long_text"] = text
            print(f"[{a.tag}] LONG: WER {w_:.2f}% RTF {el / (len(x) / SR):.3f} ({el:.0f}s for {len(x) / SR:.0f}s) words_ts={len(ws)}", flush=True)
            res["loadavg_long_end"] = loadavg()
    res["peak_rss_mb"] = round(peak_rss_mb())
    res["peak_vram_mb"] = None if gm.peak_above_base_mb is None else round(gm.peak_above_base_mb)
    save_result(a.tag, res)


if __name__ == "__main__":
    main()
