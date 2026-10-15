"""Per-frame CPU cost of the causal candidates, single thread pinned to one core.

  python -I bench_live.py [--frames 3000] [--reps 3] [--core 7] [MODEL ...]

For every model the first N hops of the 10-minute noisy file are streamed frame by frame and the time of each
model call is recorded (STFT/iSTFT included, resampling of 16 kHz models measured separately).
Results are appended to denoise-out/live_results.jsonl together with the `uptime` load average.
"""
import argparse, json, os, sys, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent)); sys.dont_write_bytecode = True
for k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[k] = "1"
import numpy as np
from scipy.signal import resample_poly
from common import DATA, LONG_FILE, OUT, SR, VB_CLEAN, loadavg, read_mono, subset_ids
import models

ap = argparse.ArgumentParser()
ap.add_argument("models", nargs="*", default=["dfn3_live", "dfn3_ll", "dpdfnet2_48khz_hr", "dpdfnet2", "dpdfnet4", "baseline", "gtcrn_dns3"])
ap.add_argument("--frames", type=int, default=3000); ap.add_argument("--reps", type=int, default=3); ap.add_argument("--core", type=int, default=7)
a = ap.parse_args()
os.sched_setaffinity(0, {a.core})
x48 = read_mono(LONG_FILE)[: SR * 40]
clean = read_mono(VB_CLEAN / (subset_ids(200)[0] + ".wav"))
OUT.mkdir(exist_ok=True, parents=True)
rows = []
for name in a.models:
    m = models.make(name, "cpu", 1)
    lat = m.measure_delay(clean)
    xn = m._native(x48)[: m.hop * a.frames]
    m.stream(xn[: m.hop * 200])  # warm-up
    per_rep = []
    for r in range(a.reps):
        m.reset()
        l0 = loadavg()
        _, t = m.stream(xn)
        per_rep.append(t)
    t = np.concatenate(per_rep)
    hop_ms = 1000.0 * m.hop / m.sr
    rec = {"model": name, "native_sr": m.sr, "hop_ms": hop_ms, "frames": int(len(t)), "latency_ms": lat,
           "mean_ms": float(t.mean() * 1e3), "median_ms": float(np.median(t) * 1e3), "p95_ms": float(np.percentile(t, 95) * 1e3),
           "p99_ms": float(np.percentile(t, 99) * 1e3), "max_ms": float(t.max() * 1e3),
           "core_fraction": float(t.mean() / (hop_ms / 1e3)), "ms_per_10ms_audio": float(t.mean() * 1e3 * 10.0 / hop_ms),
           "per_rep_mean_ms": [float(p.mean() * 1e3) for p in per_rep], "load": loadavg(), "core": a.core,
           "when": time.strftime("%Y-%m-%d %H:%M:%S")}
    if m.sr != SR:  # cost of the 48k<->native resamplers for one 10 ms block (stateless polyphase, indicative)
        blk = x48[: 4800]
        up, down = SR // m.sr, 1
        tt = []
        for _ in range(300):
            t0 = time.perf_counter(); y = resample_poly(blk, 1, SR // m.sr); z = resample_poly(y, SR // m.sr, 1); tt.append(time.perf_counter() - t0)
        # 4800 samples = 100 ms; scale to 10 ms
        rec["resample_ms_per_10ms_audio"] = float(np.median(tt) * 1e3 / 10.0)
    rows.append(rec)
    print(json.dumps(rec), flush=True)
with open(OUT / "live_results.jsonl", "a") as f:
    for r in rows:
        f.write(json.dumps(r) + "\n")
