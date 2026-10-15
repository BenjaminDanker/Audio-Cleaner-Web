#!/usr/bin/env python
"""Throughput with N concurrent file-transcription jobs on the CPU (Parakeet via onnx-asr + silero VAD).

  python -I bench_concurrent_files.py <model_dir> <int8|none> N THREADS_PER_JOB [reps]
Each worker loads its own model copy, waits at a barrier, then transcribes the 10-min long-form file `reps` times.
Prints per-job wall time, aggregate throughput (audio-seconds per wall-second) and peak RSS per worker.
"""
import multiprocessing as mp, sys, time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))


def worker(model_dir, quant, threads, barrier, q, reps):
    from common import EVAL, load_audio, rss_mb, peak_rss_mb
    import onnx_asr, onnxruntime as ort
    so = ort.SessionOptions(); so.intra_op_num_threads = threads; so.inter_op_num_threads = 1
    m = onnx_asr.load_model("nemo-conformer-tdt", model_dir, quantization=None if quant == "none" else quant,
                            sess_options=so, providers=["CPUExecutionProvider"], cpu_preprocessing=True)
    vad = onnx_asr.load_vad("silero", path=str(Path.home() / "audio-cleaner/research/models/silero-vad-onnx"))
    x = load_audio(EVAL / "long/long10.wav")
    list(m.with_vad(vad, max_speech_duration_s=30, min_silence_duration_ms=500, batch_size=8).recognize(x[:16000 * 20]))  # warm-up
    barrier.wait()
    t0 = time.time()
    for _ in range(reps):
        list(m.with_vad(vad, max_speech_duration_s=30, min_silence_duration_ms=500, batch_size=8).recognize(x))
    q.put((time.time() - t0, len(x) / 16000 * reps, peak_rss_mb()))


if __name__ == "__main__":
    model_dir, quant, n, threads = sys.argv[1], sys.argv[2], int(sys.argv[3]), int(sys.argv[4])
    reps = int(sys.argv[5]) if len(sys.argv) > 5 else 1
    ctx = mp.get_context("spawn")
    barrier, q = ctx.Barrier(n), ctx.Queue()
    ps = [ctx.Process(target=worker, args=(model_dir, quant, threads, barrier, q, reps)) for _ in range(n)]
    la = open("/proc/loadavg").read().split()[:3]
    for p in ps: p.start()
    rs = [q.get() for _ in ps]
    for p in ps: p.join()
    wall = max(r[0] for r in rs)
    audio = sum(r[1] for r in rs)
    print(f"jobs={n} threads/job={threads} model={Path(model_dir).name}:{quant}  per-job wall {[round(r[0]) for r in rs]} s  "
          f"aggregate {audio / wall:.1f} audio-s per s (RTF per job {wall / rs[0][1]:.3f})  peak RSS/job {max(r[2] for r in rs):.0f} MB  loadavg_before {la}")
