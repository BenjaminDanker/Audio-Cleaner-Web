import os as _os
from pathlib import Path as _Path
# Work folder with libri/ (LibriSpeech stream16k.wav), reco/ (generated material and results) and effiles/. Set RECO_ROOT to move it.
ROOT = _os.environ.get('RECO_ROOT', str(_Path.home() / 'audio-cleaner' / 'research' / 'reco-calibration'))
REPO = str(_Path(__file__).resolve().parents[2])
import json, collections
from pathlib import Path
import numpy as np, soundfile as sf
S = Path(ROOT + '/reco')
(S / 'long').mkdir(exist_ok=True)
meta = json.load(open(S / 'meta.json'))
by = collections.defaultdict(dict)
for m in meta:
    if m['cat'] == 'noise': by[(m['kind'], m['snr'])][m['clip']] = m['name']
n = 0
for (kind, snr), clips in sorted(by.items()):
    ids = sorted(clips)
    for gi, group in enumerate((ids[:12], ids[8:20])):
        if len(group) < 10: continue
        x = np.concatenate([sf.read(S / 'set' / f'{clips[c]}.wav', dtype='float32')[0] for c in group])
        sf.write(S / 'long' / f'long_{kind}_{snr:02d}_g{gi}.wav', x, 48000, subtype='FLOAT'); n += 1
print(n, 'long files')
