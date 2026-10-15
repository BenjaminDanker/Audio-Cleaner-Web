"""Metrics for enhanced wavs in denoise-out/<model>/ against the VoiceBank clean references.

  python -I eval_metrics.py MODEL [MODEL ...] [--clarity] [--noisy]
Writes denoise-out/<model>/metrics[_clarity].csv and prints the mean row.
PESQ-WB / STOI / ESTOI on 16 kHz; SI-SDR at 48 kHz (16 kHz for models that output 16 kHz); DNSMOS P.835 (16 kHz).
"""
import argparse, csv, sys
from multiprocessing import Pool
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.dont_write_bytecode = True
import numpy as np
import soundfile as sf
from common import DNSMOS, OUT, SR, VB_CLEAN, VB_NOISY, intrusive_metrics, read_mono, resample, subset_ids, si_sdr

_dn = None
_clarity = None


def _init():
    global _dn, _clarity
    _dn = DNSMOS()
    sys.path.insert(0, "/home/silver/repos/Audio-Cleaner-Web/server")
    from acw.dsp import clarity
    _clarity = clarity


def one(args):
    mdir, uid, use_clarity = args
    c48 = read_mono(VB_CLEAN / f"{uid}.wav")
    noisy = read_mono(VB_NOISY / f"{uid}.wav")
    if mdir == "NOISY":
        y, sr = noisy, SR
    else:
        y, sr = sf.read(str(OUT / mdir / f"{uid}.wav"), dtype="float32")
    if use_clarity:
        y48 = resample(y, sr, SR)
        y48 = np.pad(y48, (0, max(0, len(noisy) - len(y48))))[: len(noisy)]  # 16 kHz outputs can differ by a sample
        y48 = _clarity.process(noisy[:, None], y48[:, None], SR)[:, 0]
        y, sr = y48, SR
    if sr == SR:
        m = intrusive_metrics(c48, y)
        dn = _dn(y)
    else:  # native 16 kHz output: everything at 16 kHz
        c16 = resample(c48, SR, 16000)
        y48 = resample(y, sr, SR)
        m = intrusive_metrics(c48, y48)           # PESQ/STOI are 16 kHz anyway
        m["si_sdr"] = si_sdr(c16, y)               # SI-SDR on the native band only
        dn = _dn(y48)
    return {"utt": uid, **m, **{f"dnsmos_{k}": v for k, v in dn.items()}}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("models", nargs="+")
    ap.add_argument("--clarity", action="store_true")
    ap.add_argument("--subset", type=int, default=200)
    ap.add_argument("--workers", type=int, default=4)
    a = ap.parse_args()
    ids = subset_ids(a.subset)
    for mdir in a.models:
        if mdir != "NOISY":
            have = {p.stem for p in (OUT / mdir).glob("*.wav")}
            todo = [i for i in ids if i in have]
        else:
            todo = ids
        with Pool(a.workers, initializer=_init) as pool:
            rows = pool.map(one, [(mdir, i, a.clarity) for i in todo], chunksize=4)
        keys = [k for k in rows[0] if k != "utt"]
        mean = {k: float(np.nanmean([r[k] for r in rows])) for k in keys}
        outdir = OUT / ("_noisy" if mdir == "NOISY" else mdir)
        outdir.mkdir(exist_ok=True, parents=True)
        f = outdir / ("metrics_clarity.csv" if a.clarity else "metrics.csv")
        with open(f, "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=["utt"] + keys)
            w.writeheader(); w.writerows(rows)
        print(f"{mdir}{'+clarity' if a.clarity else ''} n={len(rows)} " + " ".join(f"{k}={v:.3f}" for k, v in mean.items()))


if __name__ == "__main__":
    main()
