"""Second round of server measurements: for every strength (12/24/40/100 dB) of DeepFilterNet3 (all mixes) and of DPDFNet-8 (mixes whose
noise figure is under 21): SI-SDR against the clean clip AND the noise left in the pauses relative to the speech level (dB)."""
import os as _os
from pathlib import Path as _Path
# Work folder with libri/ (LibriSpeech stream16k.wav), reco/ (generated material and results) and effiles/. Set RECO_ROOT to move it.
ROOT = _os.environ.get('RECO_ROOT', str(_Path.home() / 'audio-cleaner' / 'research' / 'reco-calibration'))
REPO = str(_Path(__file__).resolve().parents[2])
import json, os, sys
from multiprocessing import Pool
from pathlib import Path
import numpy as np, soundfile as sf
sys.path.insert(0, REPO + '/common')
from acw_common import dfn, dpdfnet
S = Path(ROOT + '/reco')
LIB = _os.path.expanduser('~/audio-cleaner/bin/libdf.so')
MODEL = _os.path.expanduser('~/audio-cleaner/models/DeepFilterNet3_onnx.tar.gz')
HEAVY = _os.path.expanduser('~/audio-cleaner/models/dpdfnet/dpdfnet8_48khz_hr.onnx')


def sisdr(est, ref):
    n = min(len(est), len(ref)); e, r = est[:n].astype(np.float64), ref[:n].astype(np.float64)
    t = (np.dot(e, r) / (np.dot(r, r) + 1e-12)) * r
    return float(10 * np.log10(np.sum(t ** 2) / (np.sum((e - t) ** 2) + 1e-12) + 1e-12))


def frames(x):
    n = len(x) // 480
    return np.mean(x[:n * 480].reshape(n, 480).astype(np.float64) ** 2, axis=1)


def one(r):
    x, _ = sf.read(S / 'set' / f"{r['name']}.wav", dtype='float32')
    c, _ = sf.read(S / r['clean'], dtype='float32'); c = c[:len(x)]
    fc = frames(c)
    order = np.argsort(fc)
    pause = order[:max(5, len(order) // 5)]          # the quietest fifth of the clean clip
    speech_pow = np.mean(fc[fc >= np.median(fc)])    # active speech level
    a = np.dot(x, c) / (np.dot(c, c) + 1e-12)        # the mix may have been rescaled to avoid clipping
    d = dict(name=r['name'])

    def res(y):
        fy = frames(y)
        return float(10 * np.log10(np.mean(fy[pause]) / (a * a * speech_pow) + 1e-12))

    d['res_in'] = res(x)
    for db in (12, 24, 40, 100):
        y = dfn.denoise_array(x, LIB, MODEL, atten_lim_db=db, threads=1)
        d[f'si_l{db}'] = sisdr(y, c); d[f'res_l{db}'] = res(y)
    if r['nd'] < 21:
        for db in (12, 24, 40, 100):
            y = dpdfnet.denoise_array(x, atten_lim_db=db, threads=1, model_path=HEAVY)
            d[f'si_h{db}'] = sisdr(y, c); d[f'res_h{db}'] = res(y)
    return d


if __name__ == '__main__':
    rows = [json.loads(l) for l in open(S / 'srv_12.jsonl')]
    rows = [r for r in rows if 'error' not in r]
    rows.sort(key=lambda r: -r['nd'])   # slow (heavy) ones last
    out = open(S / 'srv2.jsonl', 'w')
    with Pool(int(os.environ.get('NPROC', '5'))) as p:
        for i, d in enumerate(p.imap_unordered(one, rows), 1):
            out.write(json.dumps(d) + '\n'); out.flush()
            if i % 50 == 0: print(i, len(rows), flush=True)
