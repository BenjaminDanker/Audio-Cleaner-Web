#!/usr/bin/env python
"""Print markdown result tables from results/*.json (offline engines). Usage: python -I summarize.py"""
import glob, json, sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from common import *  # noqa

R = {}
for f in sorted(glob.glob(str(RESULTS / "*.json"))):
    if f.endswith("_partial.json"):
        continue
    r = json.load(open(f))
    if "sets" in r and "engine" in r:
        R[r["tag"]] = r
order = [t for t in ["fw-turbo-cpu-int8", "fw-distil35-cpu-int8", "pk2-int8-cpu", "pk2-fp32-cpu", "pk3-int8-cpu", "pk3-fp32-cpu",
                     "sherpa-pk2-int8-cpu", "pk2-fp32-gpu", "pk3-fp32-gpu", "fw-distil35-gpu-fp32", "fw-turbo-gpu-fp32"] if t in R]


def common_wer(tag, setname, ids):
    s = R[tag]["sets"].get(setname)
    if not s or "hyps" not in s:
        return None
    rows = {r["id"]: r for r in load_manifest(setname)}
    if not all(i in s["hyps"] for i in ids):
        return None
    w, n, _ = wer([rows[i]["ref"] for i in ids], [s["hyps"][i] for i in ids], rows[ids[0]]["lang"])
    return w


print("| tag | load s | clean (n) | noisy 5 dB | noisy+DFN3 | long-form WER | long RTF | short-utt RTF | CPU-s/audio-s (long) | peak RSS MB | peak VRAM MB |")
print("|---|---|---|---|---|---|---|---|---|---|---|")
for t in order:
    r = R[t]; S = r["sets"]; L = r.get("long", {})
    def c(n):
        return f"{S[n]['wer']:.2f}" if n in S else "-"
    nn = S["clean"]["n"] if "clean" in S else "-"
    rtf = S["clean"]["rtf"] if "clean" in S else "-"
    print(f"| {t} | {r['load_s']} | {c('clean')} ({nn}) | {c('noisy5')} | {c('noisy5_dfn3')} | {L.get('wer', '-')} | {L.get('rtf', '-')} | {rtf} | {L.get('cpu_s_per_audio_s', '-')} | {r['peak_rss_mb']} | {'-' if '-cpu' in t else r.get('peak_vram_mb')} |")

print("\nCommon-subset WER (first 20 utterances of each English set; all engines scored on identical utterances):")
print("| tag | clean-20 | noisy5-20 | dfn3-20 |")
print("|---|---|---|---|")
for t in order:
    vals = []
    for n in SETS_EN:
        ids = [r["id"] for r in load_manifest(n)][:20]
        v = common_wer(t, n, ids)
        vals.append("-" if v is None else f"{v:.2f}")
    print(f"| {t} | " + " | ".join(vals) + " |")

print("\nFLEURS (20 utts es/de/fr; Whisper CPU rows use the first 10):")
print("| tag | es | de | fr |")
print("|---|---|---|---|")
for t in order:
    S = R[t]["sets"]
    if any(k.startswith("fleurs") for k in S):
        print(f"| {t} | " + " | ".join(f"{S[k]['wer']:.2f} (n={S[k]['n']})" if k in S else "-" for k in SETS_FLEURS) + " |")
for t in ["fw-turbo-cpu-int8-fleurs"]:
    if t in R:
        S = R[t]["sets"]
        print(f"| {t} | " + " | ".join(f"{S[k]['wer']:.2f} (n={S[k]['n']})" if k in S else "-" for k in SETS_FLEURS) + " |")
    if t in R:
        # parakeet v3 on the same first-10 utterances
        for pt in ["pk3-fp32-cpu", "pk3-int8-cpu"]:
            vals = []
            for k in SETS_FLEURS:
                ids = [r["id"] for r in load_manifest(k)][:10]
                v = common_wer(pt, k, ids)
                vals.append("-" if v is None else f"{v:.2f} (n=10)")
            print(f"| {pt} (same first 10) | " + " | ".join(vals) + " |")

# ------------------------------------------------------------------ streaming
S = {}
for f in sorted(glob.glob(str(RESULTS / "stream-*.json"))):
    r = json.load(open(f))
    S[Path(f).stem] = r
live = [r for k, r in S.items() if "-live" in k]
if live:
    print("\nStreaming, real-time paced (100 ms hops), 60 s excerpt (9 LibriSpeech utterances):")
    print("| model | streams x threads | CPU % of one core per stream | word latency median / p90 (s) | hop delay p95 / max (ms) | RSS MB/stream | loadavg at start |")
    print("|---|---|---|---|---|---|---|")
    for r in sorted(live, key=lambda r: (r["model"], r["streams"])):
        print(f"| {r['model']} | {r['streams']} x {r['threads_per_stream']} | {r['cpu_pct_per_stream']} | {r['word_lat_median_s']} / {r['word_lat_p90_s']} | {r['hop_delay_p95_ms']} / {r['hop_delay_max_ms']} | {r['rss_mb_per_stream']} | {r['loadavg_start'][0]} |")
wers = {}
for k, r in S.items():
    if "-wer" in k:
        for sname, v in r["sets"].items():
            wers.setdefault(r["model"], {})[sname] = (v["wer"], v["n"], v["cpu_s_per_audio_s"], v["rtf"])
if wers:
    print("\nStreaming WER (final transcript after end of audio; first 40 clean / 30 noisy utterances), 2 threads:")
    print("| model | clean | noisy 5 dB | noisy+DFN3 | CPU-s per audio-s (clean) |")
    print("|---|---|---|---|---|")
    for m, d in wers.items():
        c = lambda n: f"{d[n][0]:.2f} (n={d[n][1]})" if n in d else "-"
        print(f"| {m} | {c('clean')} | {c('noisy5')} | {c('noisy5_dfn3')} | {d['clean'][2] if 'clean' in d else '-'} |")
