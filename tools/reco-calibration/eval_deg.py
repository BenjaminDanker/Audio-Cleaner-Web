import os as _os
from pathlib import Path as _Path
# Work folder with libri/ (LibriSpeech stream16k.wav), reco/ (generated material and results) and effiles/. Set RECO_ROOT to move it.
ROOT = _os.environ.get('RECO_ROOT', str(_Path.home() / 'audio-cleaner' / 'research' / 'reco-calibration'))
REPO = str(_Path(__file__).resolve().parents[2])
import json, collections, sys
import numpy as np, soundfile as sf
sys.path.insert(0, REPO + '/common')
from acw_common import advice, spectrum  # noqa: F401
S = ROOT + '/reco/'
meta = {m['name']: m for m in json.load(open(S + 'meta_deg.json'))}
rows = []
for l in open(S + 'feat_deg_reco.jsonl'):
    j = json.loads(l); n = j['file'].split('/')[-1][:-4]
    m = meta.get(n, dict(name=n, cat='clean'))
    rows.append({**m, **j})
def rate(sel, key):
    v = [r for r in rows if sel(r) and r['reco']]
    return sum(bool(key(r['reco'])) for r in v), len(v)
print('clean (86 libri clips, 16 kHz source):  rebuild %d/%d' % rate(lambda r: r['cat']=='clean', lambda x: x['rebuild']))
for rt in (0.15, 0.3, 0.5, 0.7, 1.0, 1.4):
    a, n = rate(lambda r: r['cat']=='reverb' and r['rt60']==rt, lambda x: x['rebuild']); print(f'reverb RT60 {rt}: rebuild recommended {a}/{n}')
for fc in (1200, 2000, 3000):
    a, n = rate(lambda r: r['cat']=='muffled' and r['cut']==fc, lambda x: x['rebuild']); print(f'muffled (one-pole low-pass {fc} Hz): rebuild recommended {a}/{n}')
for fc in (3400, 4000, 5500, 7000, 9500):
    a, n = rate(lambda r: r['cat']=='bandlimit' and r['cut']==fc, lambda x: x['rebuild'])
    py = collections.Counter(bool(r['bandwidthHz'] <= advice.PHONE_HZ) for r in rows if r['cat']=='bandlimit' and r['cut']==fc)
    print(f'band-limited at {fc} Hz: phone-like (voice rebuilt) {a}/{n}; at or below the phone line {dict(py)}')
for tag in ('amr', 'g722', 'mp3_32', 'opus_12'):
    a, n = rate(lambda r: r['cat']=='codec' and r['tag']==tag, lambda x: x['rebuild']); print(f'codec {tag}: rebuilt {a}/{n}')
