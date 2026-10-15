"""Run one model over the fixed VoiceBank-DEMAND subset and write enhanced wavs.

  [flock ~/audio-cleaner/dev/gpu.lock] python -I run_offline.py MODEL --device cuda --subset 200

Output: ~/audio-cleaner/research/denoise-out/<MODEL>/<utt>.wav  (+ _times.json with per-clip seconds and load average)
"""
import argparse, json, os, sys, time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.dont_write_bytecode = True
os.environ.setdefault("TQDM_DISABLE", "1")

import numpy as np
from common import OUT, SR, VB_CLEAN, VB_NOISY, loadavg, peak_rss_mb, read_mono, resample, subset_ids, write_wav
import models


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("model")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--subset", type=int, default=200)
    ap.add_argument("--first", type=int, default=None, help="only the first N ids of the subset")
    ap.add_argument("--offset", type=int, default=0)
    a = ap.parse_args()

    ids = subset_ids(a.subset)[a.offset:]
    if a.first:
        ids = ids[: a.first]
    out = OUT / a.model
    out.mkdir(parents=True, exist_ok=True)
    tj = out / "_times.json"
    times = json.loads(tj.read_text()) if tj.exists() else {}

    t0 = time.time()
    m = models.make(a.model, a.device, a.threads)
    load_s = time.time() - t0
    meta = {"load_s": load_s, "device": a.device, "threads": a.threads}
    if a.model in models.LIVE:  # measure the input->output lag on clean speech
        meta["latency_ms"] = m.measure_delay(read_mono(VB_CLEAN / (subset_ids(a.subset)[0] + ".wav")))
    if a.device == "cuda":
        import torch
    for i in ids:
        f = out / f"{i}.wav"
        if f.exists():
            continue
        x = read_mono(VB_NOISY / f"{i}.wav")
        t = time.perf_counter()
        if a.model.startswith("mfgan16"):
            try:
                y = m.enhance16(resample(x, SR, 16000))
            except Exception as e:  # CUDA OOM on the 4 GB card with a 10 s window: retry with 6 s windows
                if "out of memory" not in str(e).lower():
                    raise
                import torch
                torch.cuda.empty_cache()
                if not hasattr(m, "_w6"):
                    m._w6 = models.MossFormerGANSE16k(a.device, a.threads, window_s=6.0)
                y = m._w6.enhance16(resample(x, SR, 16000))
                meta.setdefault("oom_fallback_w6", []).append(i)
            sr_out = 16000
        else:
            y = m(x); sr_out = SR
        if a.device == "cuda":
            torch.cuda.synchronize()
        dt = time.perf_counter() - t
        write_wav(f, y, sr_out)
        times[i] = {"sec": dt, "audio_s": len(x) / SR, "load": loadavg()}
    meta["peak_rss_mb"] = peak_rss_mb()
    if a.device == "cuda":
        meta["peak_vram_mb"] = torch.cuda.max_memory_allocated() / 1e6
    times["_meta"] = {**times.get("_meta", {}), **meta}
    tj.write_text(json.dumps(times))
    tot = sum(v["sec"] for k, v in times.items() if k != "_meta")
    aud = sum(v["audio_s"] for k, v in times.items() if k != "_meta")
    print(f"{a.model}: {len(times)-1} clips, {aud:.0f}s audio in {tot:.0f}s -> RTF {tot/aud:.3f} (device={a.device}, load {loadavg()}) meta={meta}")


if __name__ == "__main__":
    main()
