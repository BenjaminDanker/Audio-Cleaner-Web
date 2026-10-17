"""Server-side measurements for the calibration set: noise figure (pipeline.noise_figure_db), bandwidth, and SI-SDR (against the clean clip) of the
noisy input and of DeepFilterNet3 at 12/24/40/100 dB attenuation limits and DPDFNet-8, as pipeline.clean_audio runs them."""
import os as _os
from pathlib import Path as _Path
# Work folder with libri/ (LibriSpeech stream16k.wav), reco/ (generated material and results) and effiles/. Set RECO_ROOT to move it.
ROOT = _os.environ.get('RECO_ROOT', str(_Path.home() / 'audio-cleaner' / 'research' / 'reco-calibration'))
REPO = str(_Path(__file__).resolve().parents[2])
import json, os, sys, time
from multiprocessing import Pool
from pathlib import Path
import numpy as np, soundfile as sf

sys.path.insert(0, REPO + '/common')
from acw_common import pipeline, bwe, dfn, dpdfnet

S = Path(ROOT + '/reco')
LIB = _os.path.expanduser('~/audio-cleaner/bin/libdf.so')
MODEL = _os.path.expanduser('~/audio-cleaner/models/DeepFilterNet3_onnx.tar.gz')
HEAVY = _os.path.expanduser('~/audio-cleaner/models/dpdfnet/dpdfnet8_48khz_hr.onnx')


def sisdr(est, ref):
    n = min(len(est), len(ref))
    est, ref = est[:n].astype(np.float64), ref[:n].astype(np.float64)
    # tiny lag search (the models delay the signal a little; the pipeline compensates, this is belt and braces)
    best = None
    for lag in (0,):
        e, r = est, ref
        a = np.dot(e, r) / (np.dot(r, r) + 1e-12)
        t = a * r
        v = 10 * np.log10(np.sum(t ** 2) / (np.sum((e - t) ** 2) + 1e-12) + 1e-12)
        best = v if best is None else max(best, v)
    return float(best)


def one(m):
    try:
        x, _ = sf.read(S / 'set' / f"{m['name']}.wav", dtype='float32')
        c, _ = sf.read(S / m['clean'], dtype='float32')
        c = c[:len(x)]
        r = dict(m)
        r['nd'] = pipeline.noise_figure_db(x, LIB, MODEL)
        r['bw'] = bwe.effective_bandwidth(x)
        r['sisdr_in'] = sisdr(x, c)
        for db in (12, 24, 40, 100):
            y = dfn.denoise_array(x, LIB, MODEL, atten_lim_db=db, threads=1)
            r[f'sisdr_dfn{db}'] = sisdr(y, c)
            if db == 100:
                r['rms_dfn'] = float(10 * np.log10(np.mean(y ** 2) + 1e-12))
        y = dpdfnet.denoise_array(x, atten_lim_db=100, threads=1, model_path=HEAVY)
        r['sisdr_heavy'] = sisdr(y, c)
        return r
    except Exception as e:
        return dict(m, error=repr(e))


if __name__ == '__main__':
    meta = json.load(open(S / 'meta.json'))
    first = int(sys.argv[1]) if len(sys.argv) > 1 else 12
    subset = [m for m in meta if m['clip'] < first]
    out = open(S / f'srv_{first}.jsonl', 'w')
    t0 = time.time()
    with Pool(int(os.environ.get('NPROC', '5'))) as p:
        for i, r in enumerate(p.imap_unordered(one, subset), 1):
            out.write(json.dumps(r) + '\n'); out.flush()
            if i % 50 == 0:
                print(i, len(subset), round(time.time() - t0), flush=True)
