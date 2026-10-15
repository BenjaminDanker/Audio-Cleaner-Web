"""Turn denoise-out/*/metrics*.csv, speed_results.jsonl and live_results.jsonl into markdown tables.
  python -I report.py > tables.md
"""
import csv, json, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent)); sys.dont_write_bytecode = True
import numpy as np
from common import OUT

COLS = [("pesq", "PESQ-WB"), ("stoi", "STOI"), ("estoi", "ESTOI"), ("si_sdr", "SI-SDR"), ("lsd_hf", "LSD>8k"),
        ("dnsmos_sig", "SIG"), ("dnsmos_bak", "BAK"), ("dnsmos_ovrl", "OVRL")]


def mean_row(path):
    if not path.exists():
        return None
    rows = list(csv.DictReader(open(path)))
    return len(rows), {k: float(np.nanmean([float(r[k]) for r in rows])) for k, _ in COLS}


def qtable(title, entries):
    print(f"\n### {title}\n")
    print("| System | n | " + " | ".join(h for _, h in COLS) + " |")
    print("|---|---|" + "---|" * len(COLS))
    for label, path in entries:
        r = mean_row(OUT / path)
        if r:
            n, m = r
            print(f"| {label} | {n} | " + " | ".join(f"{m[k]:.2f}" if k not in ("stoi", "estoi") else f"{m[k]:.3f}" for k, _ in COLS) + " |")


studio = [("Noisy input", "_noisy/metrics.csv"),
          ("DFN3 (libDF) raw", "dfn3/metrics.csv"), ("**DFN3 + clarity (production)**", "dfn3/metrics_clarity.csv"),
          ("MossFormer2_SE_48K raw", "mf2_48k/metrics.csv"), ("MossFormer2_SE_48K + clarity", "mf2_48k/metrics_clarity.csv"),
          ("MossFormerGAN_SE_16K as-is raw", "mfgan16/metrics.csv"), ("MossFormerGAN_SE_16K as-is + clarity", "mfgan16/metrics_clarity.csv"),
          ("Hybrid GAN<7.5k + DFN3 raw", "hybrid_mfgan_dfn3/metrics.csv"), ("Hybrid + clarity", "hybrid_mfgan_dfn3/metrics_clarity.csv"),
          ("Resemble denoiser raw", "resemble_denoise/metrics.csv"), ("Resemble denoiser + clarity", "resemble_denoise/metrics_clarity.csv"),
          ("Resemble denoiser+enhancer (NFE 32) raw", "resemble_enhance_nfe32/metrics.csv"), ("Resemble denoiser+enhancer (NFE 32) + clarity", "resemble_enhance_nfe32/metrics_clarity.csv")]
live = [("DFN3 (libDF, 30 ms)", "dfn3/metrics.csv"), ("DFN3-LL (libDF, 10 ms)", "dfn3_ll/metrics.csv"),
        ("DPDFNet2 48 kHz HR", "dpdfnet2_48khz_hr/metrics.csv"), ("DPDFNet2 16 kHz", "dpdfnet2/metrics.csv"), ("DPDFNet4 16 kHz", "dpdfnet4/metrics.csv"),
        ("DPDFNet baseline 16 kHz", "baseline/metrics.csv"), ("GTCRN (DNS3-trained, 16 kHz)", "gtcrn_dns3/metrics.csv"), ("GTCRN (VCTK-trained, 16 kHz)", "gtcrn_vctk/metrics.csv")]
qtable("Studio candidates, VoiceBank-DEMAND test subset (200 of 824)", studio)
qtable("Live candidates, same subset (raw model output)", live)

sp = OUT / "speed_results.jsonl"
if sp.exists():
    print("\n### Speed on the 10-minute file (RTF = wall / audio; lower is better)\n")
    print("| Model | Device | Threads | Audio s | RTF | Load s | Peak RSS MB | Peak VRAM MB (alloc/reserved) | Load avg before / after | Note |")
    print("|---|---|---|---|---|---|---|---|---|---|")
    for l in sp.read_text().splitlines():
        r = json.loads(l)
        vr = f"{r['peak_vram_alloc_mb']:.0f} / {r['peak_vram_reserved_mb']:.0f}" if "peak_vram_alloc_mb" in r else "-"
        print(f"| {r['model']} | {r['device']} | {r['threads']} | {r['audio_s']:.0f} | {r['rtf']:.3f} | {r['load_s']:.1f} | {r['peak_rss_mb']:.0f} | {vr} | {r['load_before']} / {r['load_after']} | {r['note']} |")

lp = OUT / "live_results.jsonl"
if lp.exists():
    print("\n### Live: per-frame cost, single thread pinned to one core\n")
    print("| Model | Rate | Hop ms | Latency ms (I/O lag) | Mean ms/frame | p99 ms | Max ms | Core fraction / stream | Streams per core @70 % | Resample ms/10 ms | Load |")
    print("|---|---|---|---|---|---|---|---|---|---|---|")
    for l in lp.read_text().splitlines():
        r = json.loads(l)
        f = r["core_fraction"] + r.get("resample_ms_per_10ms_audio", 0) / 10.0
        print(f"| {r['model']} | {r['native_sr']} | {r['hop_ms']:.1f} | {r['latency_ms']:.0f} | {r['mean_ms']:.2f} | {r['p99_ms']:.2f} | {r['max_ms']:.2f} | {r['core_fraction']:.3f} | {0.7 / f:.1f} | {r.get('resample_ms_per_10ms_audio', 0):.3f} | {r['load']} |")
