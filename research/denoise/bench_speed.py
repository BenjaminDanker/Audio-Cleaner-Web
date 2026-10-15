"""Speed / memory benchmark for one (model, device, threads) combination on the long file.

  [flock ~/audio-cleaner/dev/gpu.lock] python -I bench_speed.py MODEL --device cuda|cpu --threads N [--seconds 600]

Appends one JSON line to ~/audio-cleaner/research/denoise-out/speed_results.jsonl with RTF, model load time,
peak RSS, peak VRAM and the `uptime` load averages before/after (the box is shared, so timings carry noise).
If --seconds is shorter than the 600 s file the first N seconds are used and the RTF is a linear extrapolation
(all adapters are chunked, so cost is linear in duration).
"""
import argparse, json, os, sys, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent)); sys.dont_write_bytecode = True
os.environ.setdefault("TQDM_DISABLE", "1")
import numpy as np
from common import DATA, LONG_FILE, OUT, SR, loadavg, peak_rss_mb, read_mono
import models

ap = argparse.ArgumentParser()
ap.add_argument("model"); ap.add_argument("--device", default="cpu"); ap.add_argument("--threads", type=int, default=8)
ap.add_argument("--seconds", type=float, default=600.0); ap.add_argument("--note", default="")
a = ap.parse_args()

x = read_mono(LONG_FILE)[: int(a.seconds * SR)]
l0 = loadavg()
t = time.perf_counter(); m = models.make(a.model, a.device, a.threads); load_s = time.perf_counter() - t
if a.device == "cuda":
    import torch
# warm-up (kernel selection / allocator) on 6 s, not counted
if a.model.startswith("mfgan16"):
    _ = m.enhance16(x[: 6 * SR][::3])
else:
    _ = m(x[: 6 * SR])
if a.device == "cuda":
    torch.cuda.synchronize(); torch.cuda.reset_peak_memory_stats()
l1 = loadavg()
t = time.perf_counter()
y = m(x) if not a.model.startswith("mfgan16") else m(x)
if a.device == "cuda":
    torch.cuda.synchronize()
dt = time.perf_counter() - t
rec = {"model": a.model, "device": a.device, "threads": a.threads, "audio_s": len(x) / SR, "wall_s": dt, "rtf": dt / (len(x) / SR),
       "load_s": load_s, "peak_rss_mb": peak_rss_mb(), "load_before": l1, "load_after": loadavg(), "note": a.note,
       "finite": bool(np.isfinite(y).all()), "when": time.strftime("%Y-%m-%d %H:%M:%S")}
if a.device == "cuda":
    rec["peak_vram_alloc_mb"] = torch.cuda.max_memory_allocated() / 1e6
    rec["peak_vram_reserved_mb"] = torch.cuda.max_memory_reserved() / 1e6
OUT.mkdir(exist_ok=True, parents=True)
with open(OUT / "speed_results.jsonl", "a") as f:
    f.write(json.dumps(rec) + "\n")
print(json.dumps(rec))
