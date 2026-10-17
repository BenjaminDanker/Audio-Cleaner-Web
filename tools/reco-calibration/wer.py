"""How much does Standard (Parakeet) change its words with noise and fast speech? WER of each condition against the transcript of the
same clip clean at normal speed."""
import os as _os
from pathlib import Path as _Path
# Work folder with libri/ (LibriSpeech stream16k.wav), reco/ (generated material and results) and effiles/. Set RECO_ROOT to move it.
ROOT = _os.environ.get('RECO_ROOT', str(_Path.home() / 'audio-cleaner' / 'research' / 'reco-calibration'))
REPO = str(_Path(__file__).resolve().parents[2])
import json, sys, re
from pathlib import Path
import numpy as np, soundfile as sf
from scipy.signal import resample_poly
sys.path.insert(0, REPO + '/common')
from acw_common import transcribe
S = Path(ROOT + '/reco')
M = Path(_os.path.expanduser('~/audio-cleaner/models'))
model, vad = M / 'parakeet-v2', M / 'silero-vad'

def text(path):
    x, sr = sf.read(path, dtype='float32')
    if x.ndim > 1: x = x.mean(1)
    y = resample_poly(x, 1, 3).astype(np.float32)
    segs = transcribe.transcribe_parakeet(y, model, vad, threads=2)
    return ' '.join(s.text for s in segs)

def norm(t): return re.sub(r"[^a-z' ]", ' ', t.lower()).split()

def wer(ref, hyp):
    r, h = norm(ref), norm(hyp)
    d = np.zeros((len(r) + 1, len(h) + 1), int)
    d[:, 0] = range(len(r) + 1); d[0, :] = range(len(h) + 1)
    for i in range(1, len(r) + 1):
        for j in range(1, len(h) + 1):
            d[i, j] = min(d[i-1, j] + 1, d[i, j-1] + 1, d[i-1, j-1] + (r[i-1] != h[j-1]))
    return d[-1, -1], len(r)

out = open(S / 'wer.jsonl', 'w')
for i in range(16):
    c = f'c{i:03d}'
    ref = text(S / f'clean/{c}.wav')
    conds = [('clean', 99, S / f'clean/{c}.wav')]
    for t in ('1.3', '1.6', '1.9'): conds.append(('tempo', t, S / f'fast/t{t}_{c}.wav'))
    for kind in ('white', 'babble', 'cafe', 'traffic'):
        for snr in (0, 5, 10, 15, 20, 25):
            conds.append((kind, snr, S / f'set/n_{kind}_{snr:02d}_{c}.wav'))
    for kind, lvl, p in conds:
        if not p.exists(): continue
        e, n = wer(ref, text(p)) if kind != 'clean' else (0, len(norm(ref)))
        out.write(json.dumps(dict(clip=c, kind=kind, level=lvl, errors=int(e), words=int(n))) + '\n'); out.flush()
