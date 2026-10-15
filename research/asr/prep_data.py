#!/usr/bin/env python
"""Build the ASR evaluation sets under ~/audio-cleaner/research/data/eval/.

Run with the research venv:  ~/audio-cleaner/research/venv/bin/python prep_data.py <step>
Steps (in order; each is idempotent):
  librispeech   extract LibriSpeech test-clean (CC BY 4.0), pick 150 fixed utterances
                (seed 1234) -> eval/clean/*.wav (16 kHz mono int16) + manifest_clean.jsonl,
                and a disjoint ~10 min long-form file eval/long/long10.wav (+ reference).
  noise         (after fetch_demand.py) mix the 150 utterances with DEMAND noise (CC BY 4.0, 48 kHz, ch01 of
                DLIVING / PCAFETER / STRAFFIC / OOFFICE) at 5 dB SNR
                -> eval/noisy5_48k/*.wav (48 kHz, input to DeepFilterNet3) and
                   eval/noisy5/*.wav (16 kHz) + manifest_noisy5.jsonl
  dfn_in        (nothing to do - run denoise_dfn.py with the *production* venv python)
  dfn_out       resample the DFN3 output to 16 kHz -> eval/noisy5_dfn3/*.wav + manifest
  fleurs        first 20 test utterances of FLEURS es_419 / de_de / fr_fr (CC BY 4.0)
                streamed from the HF tar.gz -> eval/fleurs_{es,de,fr}/ + manifests
Raw downloads live in ~/audio-cleaner/research/data/raw/ (untrusted data; only read via
soundfile / tarfile with filter="data" / zipfile).
"""
import json, random, sys, tarfile, zipfile, io, urllib.request
from pathlib import Path

import numpy as np
import soundfile as sf
from scipy.signal import resample_poly

ROOT = Path.home() / "audio-cleaner/research/data"
RAW = ROOT / "raw"
EVAL = ROOT / "eval"
SR = 16000


def write_wav(path, x, sr):
    path.parent.mkdir(parents=True, exist_ok=True)
    x = np.clip(x, -1.0, 1.0)
    sf.write(str(path), (x * 32767.0).astype(np.int16), sr, subtype="PCM_16")


def read_wav(path):
    x, sr = sf.read(str(path), dtype="float32", always_2d=False)
    if x.ndim > 1:
        x = x.mean(axis=1)
    return x, sr


def write_manifest(name, rows):
    with open(EVAL / f"manifest_{name}.jsonl", "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")


def read_manifest(name):
    return [json.loads(l) for l in open(EVAL / f"manifest_{name}.jsonl")]


def step_librispeech():
    ex = RAW / "librispeech" / "ex"
    if not ex.exists():
        ex.mkdir(parents=True)
        with tarfile.open(RAW / "librispeech" / "test-clean.tar.gz") as tf:
            tf.extractall(ex, filter="data")
    base = ex / "LibriSpeech" / "test-clean"
    utts = []
    for trans in sorted(base.glob("*/*/*.trans.txt")):
        for line in open(trans):
            uid, text = line.strip().split(" ", 1)
            utts.append((uid, text, trans.parent / f"{uid}.flac"))
    rng = random.Random(1234)
    rng.shuffle(utts)
    # 150 utterances for WER; next ones for the long-form file
    eval_set, rest = utts[:150], utts[150:]
    rows = []
    for uid, text, p in eval_set:
        x, sr = read_wav(p)
        assert sr == SR
        write_wav(EVAL / "clean" / f"{uid}.wav", x, SR)
        rows.append(dict(id=uid, path=str(EVAL / "clean" / f"{uid}.wav"), ref=text.lower(), dur=len(x) / SR, lang="en"))
    write_manifest("clean", rows)
    print("clean set:", len(rows), "utts,", sum(r["dur"] for r in rows) / 60, "min")
    # long-form: ~10 min, pauses 0.3-1.5 s
    rng2 = random.Random(99)
    parts, refs, spans, t = [], [], [], 0.0
    for uid, text, p in rest:
        if t >= 600:
            break
        x, _ = read_wav(p)
        gap = rng2.uniform(0.3, 1.5)
        spans.append([round(t, 3), round(t + len(x) / SR, 3), uid])
        parts += [x, np.zeros(int(gap * SR), dtype=np.float32)]
        refs.append(text.lower())
        t += len(x) / SR + gap
    long = np.concatenate(parts)
    write_wav(EVAL / "long" / "long10.wav", long, SR)
    json.dump(dict(ref=" ".join(refs), spans=spans, dur=len(long) / SR), open(EVAL / "long" / "long10.json", "w"))
    print("long-form:", len(long) / SR / 60, "min,", len(spans), "utts")


def rms(x):
    return float(np.sqrt(np.mean(x.astype(np.float64) ** 2) + 1e-12))


def step_noise():
    noises = {}
    for env in ["DLIVING", "PCAFETER", "STRAFFIC", "OOFFICE"]:
        # produced by fetch_demand.py (range-read of ch01 from the DEMAND 48 kHz zip)
        x, sr = sf.read(str(RAW / "demand" / f"{env}_ch01.wav"), dtype="float32")
        assert sr == 48000
        noises[env] = x
    envs = list(noises)
    rng = np.random.default_rng(4321)
    rows48, rows16 = [], []
    for i, r in enumerate(read_manifest("clean")):
        x16, _ = read_wav(r["path"])
        x48 = resample_poly(x16, 3, 1).astype(np.float32)
        env = envs[i % len(envs)]
        nz = noises[env]
        off = int(rng.integers(0, len(nz) - len(x48) - 1))
        n = nz[off:off + len(x48)]
        snr = 5.0
        g = rms(x48) / (rms(n) * 10 ** (snr / 20))
        mix = x48 + g * n
        peak = np.max(np.abs(mix))
        if peak > 0.95:
            mix *= 0.95 / peak
        write_wav(EVAL / "noisy5_48k" / f"{r['id']}.wav", mix, 48000)
        m16 = resample_poly(mix, 1, 3).astype(np.float32)
        write_wav(EVAL / "noisy5" / f"{r['id']}.wav", m16, SR)
        rows16.append({**r, "path": str(EVAL / "noisy5" / f"{r['id']}.wav"), "noise": env})
    write_manifest("noisy5", rows16)
    print("noisy set written:", len(rows16))


def step_dfn_out():
    rows = []
    for r in read_manifest("noisy5"):
        p = EVAL / "noisy5_dfn3_48k" / f"{r['id']}.wav"
        x48, sr = read_wav(p)
        assert sr == 48000
        x16 = resample_poly(x48, 1, 3).astype(np.float32)
        write_wav(EVAL / "noisy5_dfn3" / f"{r['id']}.wav", x16, SR)
        rows.append({**r, "path": str(EVAL / "noisy5_dfn3" / f"{r['id']}.wav")})
    write_manifest("noisy5_dfn3", rows)
    print("dfn3 set:", len(rows))


FLEURS = {"es": "es_419", "de": "de_de", "fr": "fr_fr"}


def step_fleurs(n=20):
    for lang, cfg in FLEURS.items():
        base = "https://huggingface.co/datasets/google/fleurs/resolve/main/data/%s" % cfg
        tsv = urllib.request.urlopen(base + "/test.tsv").read().decode()
        refs = {}
        for line in tsv.splitlines():
            c = line.split("\t")
            refs[c[1]] = c[3]  # file_name -> normalized transcription
        rows = []
        resp = urllib.request.urlopen(base + "/audio/test.tar.gz")
        with tarfile.open(fileobj=resp, mode="r|gz") as tf:
            for m in tf:
                name = Path(m.name).name
                if not m.isfile() or name not in refs:
                    continue
                x, sr = sf.read(io.BytesIO(tf.extractfile(m).read()), dtype="float32")
                if sr != SR:
                    x = resample_poly(x, SR, sr).astype(np.float32)
                p = EVAL / f"fleurs_{lang}" / name
                write_wav(p, x, SR)
                rows.append(dict(id=name[:-4], path=str(p), ref=refs[name], dur=len(x) / SR, lang=lang))
                if len(rows) >= n:
                    break
        resp.close()
        write_manifest(f"fleurs_{lang}", rows)
        print(lang, len(rows), "utts", sum(r["dur"] for r in rows), "s")


if __name__ == "__main__":
    {"librispeech": step_librispeech, "noise": step_noise, "dfn_out": step_dfn_out, "fleurs": step_fleurs}[sys.argv[1]]()
