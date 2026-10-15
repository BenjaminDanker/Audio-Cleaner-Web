#!/usr/bin/env python
"""Streaming ASR benchmark with sherpa-onnx OnlineRecognizer (CPU).

  python bench_stream.py --model nemotron-160 --wer-limit 60         # WER (clean/noisy5/noisy5_dfn3) + compute RTF
  python bench_stream.py --model nemotron-160 --live 1 --threads 2    # real-time paced run, N concurrent streams
Models (all int8, from the sherpa-onnx `asr-models` release, extracted under ~/audio-cleaner/research/models/sherpa/):
  zip-en        streaming zipformer en 2023-06-26 (icefall, trained on LibriSpeech; Apache-2.0)          chunk 320 ms
  nemo-fc-80    NeMo FastConformer-transducer streaming, 80 ms chunks  (nvidia, CC-BY-4.0)
  nemo-fc-480   same, 480 ms chunks
  nemotron-160 / nemotron-560   nvidia/nemotron-speech-streaming-en-0.6b, 160 / 560 ms (NVIDIA Open Model License)
  nemotron35-160 / nemotron35-560   nvidia/nemotron-3.5-asr-streaming-0.6b (35+ languages, OpenMDW-1.1), 160 / 560 ms
Live mode: N worker processes each stream the same ~60 s excerpt (9 LibriSpeech utterances with 0.8 s gaps) in 100 ms
hops paced in real time. Per word latency = time the word's last token appeared in the partial result minus the
MFA reference end time of that word (LibriSpeech Alignments). Also reports hop processing delay and CPU% per stream.
"""
import argparse, difflib, json, multiprocessing as mp, os, re, sys, time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from common import *  # noqa

SH = MODELS / "sherpa"
SR = 16000
HOP = 0.1
D = {
    "zip-en": ("sherpa-onnx-streaming-zipformer-en-2023-06-26", "{}-epoch-99-avg-1-chunk-16-left-128.int8.onnx", 80),
    "nemo-fc-80": ("sherpa-onnx-nemo-streaming-fast-conformer-transducer-en-80ms-int8", "{}.int8.onnx", 80),
    "nemo-fc-480": ("sherpa-onnx-nemo-streaming-fast-conformer-transducer-en-480ms-int8", "{}.int8.onnx", 80),
    "nemotron-160": ("sherpa-onnx-nemotron-speech-streaming-en-0.6b-160ms-int8-2026-04-25", "{}.int8.onnx", 128),
    "nemotron-560": ("sherpa-onnx-nemotron-speech-streaming-en-0.6b-560ms-int8-2026-04-25", "{}.int8.onnx", 128),
    "nemotron35-160": ("sherpa-onnx-nemotron-3.5-asr-streaming-0.6b-160ms-int8-2026-06-11", "{}.int8.onnx", 128),
    "nemotron35-560": ("sherpa-onnx-nemotron-3.5-asr-streaming-0.6b-560ms-int8-2026-06-11", "{}.int8.onnx", 128),
}


def make(model, threads):
    import sherpa_onnx
    d, pat, fdim = D[model]
    p = SH / d
    kw = dict(tokens=str(p / "tokens.txt"), encoder=str(p / pat.format("encoder")) if model != "zip-en" else str(p / pat.format("encoder")),
              decoder=str(p / pat.format("decoder")), joiner=str(p / pat.format("joiner")), num_threads=threads,
              sample_rate=SR, feature_dim=fdim, decoding_method="greedy_search", provider="cpu",
              enable_endpoint_detection=False)
    return sherpa_onnx.OnlineRecognizer.from_transducer(**kw)


def new_stream(rec, model):
    s = rec.create_stream()
    if model.startswith("nemotron35"):
        s.set_option("language", "en")
    return s


def result(rec, s):
    r = rec.get_result_all(s)
    return r.text, list(r.tokens), list(r.timestamps)


def group_words(tokens, emit):
    """tokens (sentencepiece, '▁' = word start) + emission time per token -> [(word, t_last_token)]."""
    words = []
    for tok, t in zip(tokens, emit):
        if tok[:1] in ("▁", " ") or not words:
            words.append([tok.lstrip("▁ "), t])
        else:
            words[-1][0] += tok
            words[-1][1] = t
    return [(w, t) for w, t in words if w]


def clean_w(w):
    return re.sub(r"[^a-z0-9']", "", w.lower())


# ---------------------------------------------------------------- reference excerpt
def excerpt(n_utts=9, gap=0.8, lead=0.5):
    rows = load_manifest("clean")[:n_utts]
    ali = json.load(open(EVAL / "align_clean.json"))
    parts, ref_words, t = [np.zeros(int(lead * SR), np.float32)], [], lead
    for r in rows:
        x = load_audio(r["path"])
        ref_words += [(clean_w(w), s + t, e + t) for w, s, e in ali[r["id"]] if clean_w(w)]
        parts += [x, np.zeros(int(gap * SR), np.float32)]
        t += len(x) / SR + gap
    return np.concatenate(parts), ref_words


# ---------------------------------------------------------------- live worker
def live_worker(args):
    model, threads, wid, barrier, q = args
    import psutil
    rec = make(model, threads)
    audio, ref = excerpt()
    # warm-up: 2 s of audio through a throw-away stream
    s = new_stream(rec, model)
    s.accept_waveform(SR, audio[: 2 * SR])
    while rec.is_ready(s):
        rec.decode_stream(s)
    s = new_stream(rec, model)
    proc = psutil.Process()
    barrier.wait()
    c0 = sum(proc.cpu_times()[:2]); t0 = time.monotonic()
    hop_n = int(HOP * SR)
    n_tok, emit, delays = 0, [], []
    for k in range(0, len(audio), hop_n):
        arrive = t0 + (k + hop_n) / SR  # last sample of this hop is spoken/captured at this time
        dt = arrive - time.monotonic()
        if dt > 0:
            time.sleep(dt)
        s.accept_waveform(SR, audio[k:k + hop_n])
        while rec.is_ready(s):
            rec.decode_stream(s)
        now = time.monotonic()
        delays.append(now - arrive)
        text, toks, _ = result(rec, s)
        emit += [now - t0] * (len(toks) - len(emit))
    s.input_finished()
    while rec.is_ready(s):
        rec.decode_stream(s)
    text, toks, _ = result(rec, s)
    emit += [time.monotonic() - t0] * (len(toks) - len(emit))
    cpu = sum(proc.cpu_times()[:2]) - c0
    wall = time.monotonic() - t0
    words = group_words(toks, emit)
    sm = difflib.SequenceMatcher(None, [r[0] for r in ref], [clean_w(w) for w, _ in words], autojunk=False)
    lat = []
    for tag, i1, i2, j1, j2 in sm.get_opcodes():
        if tag == "equal":
            lat += [words[j][1] - ref[i][2] for i, j in zip(range(i1, i2), range(j1, j2))]
    q.put(dict(wid=wid, cpu_s=cpu, wall_s=wall, audio_s=len(audio) / SR, text=text, lat=lat, hop_delay=delays,
               rss_mb=rss_mb(), n_ref=len(ref), n_match=len(lat), n_hyp=len(words), emit_last=emit[-1] if emit else None))


def run_live(model, n, threads):
    ctx = mp.get_context("spawn")
    barrier, q = ctx.Barrier(n), ctx.Queue()
    ps = [ctx.Process(target=live_worker, args=((model, threads, i, barrier, q),)) for i in range(n)]
    la0 = loadavg()
    for p in ps: p.start()
    rs = [q.get() for _ in ps]
    for p in ps: p.join()
    lat = np.concatenate([r["lat"] for r in rs]); hd = np.concatenate([r["hop_delay"] for r in rs])
    out = dict(model=model, streams=n, threads_per_stream=threads, loadavg_start=la0, loadavg_end=loadavg(),
               cpu_pct_per_stream=round(100 * np.mean([r["cpu_s"] / r["audio_s"] for r in rs]), 1),
               word_lat_median_s=round(float(np.median(lat)), 3), word_lat_p90_s=round(float(np.percentile(lat, 90)), 3),
               word_lat_mean_s=round(float(lat.mean()), 3), matched_words=int(np.mean([r["n_match"] for r in rs])),
               ref_words=rs[0]["n_ref"], hyp_words=rs[0]["n_hyp"], text_tail=rs[0]["text"][-150:], hop_delay_p50_ms=round(1000 * float(np.median(hd))),
               hop_delay_p95_ms=round(1000 * float(np.percentile(hd, 95))), hop_delay_max_ms=round(1000 * float(hd.max())),
               rss_mb_per_stream=round(np.mean([r["rss_mb"] for r in rs])), text0=rs[0]["text"][:200])
    return out


# ---------------------------------------------------------------- WER mode
def run_wer(model, threads, limit, sets):
    rec = make(model, threads)
    res = dict(model=model, threads=threads, loadavg_start=loadavg(), sets={})
    w = load_audio(load_manifest("clean")[0]["path"])
    s = new_stream(rec, model); s.accept_waveform(SR, w[:SR * 3])
    while rec.is_ready(s): rec.decode_stream(s)
    for name in sets:
        rows = load_manifest(name)[:limit]
        hyps, dur, el, cpu = [], 0.0, 0.0, 0.0
        for r in rows:
            x = load_audio(r["path"])
            c0, t0 = time.process_time(), time.time()
            s = new_stream(rec, model)
            for k in range(0, len(x), int(HOP * SR)):
                s.accept_waveform(SR, x[k:k + int(HOP * SR)])
                while rec.is_ready(s): rec.decode_stream(s)
            s.accept_waveform(SR, np.zeros(int(0.5 * SR), np.float32))  # trailing silence so last words are flushed
            s.input_finished()
            while rec.is_ready(s): rec.decode_stream(s)
            text = result(rec, s)[0]
            el += time.time() - t0; cpu += time.process_time() - c0
            hyps.append(text); dur += len(x) / SR
        w_, n, errs = wer([r["ref"] for r in rows], hyps, "en")
        res["sets"][name] = dict(wer=round(w_, 2), n_words=n, errs=errs, n=len(rows), audio_s=round(dur, 1), rtf=round(el / dur, 4),
                                 cpu_s_per_audio_s=round(cpu / dur, 3), sample=[(rows[i]["ref"], hyps[i]) for i in range(2)])
        print(f"[{model}] {name}: WER {w_:.2f}% ({n} words) RTF {el / dur:.3f}", flush=True)
    res["peak_rss_mb"] = round(peak_rss_mb())
    return res


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, choices=list(D))
    ap.add_argument("--wer-limit", type=int, default=0, help="run WER mode on first N utts of each set")
    ap.add_argument("--live", type=int, default=0, help="run live mode with N concurrent streams")
    ap.add_argument("--threads", type=int, default=2)
    ap.add_argument("--sets", default=",".join(SETS_EN))
    a = ap.parse_args()
    if a.wer_limit:
        t0 = time.time(); r = run_wer(a.model, a.threads, a.wer_limit, a.sets.split(","))
        save_result(f"stream-{a.model}-wer-{'noisy-' if 'noisy5' in a.sets else ''}t{a.threads}", r)
    if a.live:
        r = run_live(a.model, a.live, a.threads)
        print(json.dumps(r, indent=1))
        save_result(f"stream-{a.model}-live{a.live}-t{a.threads}", r)
