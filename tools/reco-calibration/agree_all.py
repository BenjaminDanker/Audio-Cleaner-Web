"""Browser analysis (Node) against the server functions, on every file of the calibration material."""
import os as _os
from pathlib import Path as _Path
# Work folder with libri/ (LibriSpeech stream16k.wav), reco/ (generated material and results) and effiles/. Set RECO_ROOT to move it.
ROOT = _os.environ.get('RECO_ROOT', str(_Path.home() / 'audio-cleaner' / 'research' / 'reco-calibration'))
REPO = str(_Path(__file__).resolve().parents[2])
import glob, json, subprocess, sys
import numpy as np, soundfile as sf
sys.path.insert(0, REPO + '/common')
from acw_common import pipeline, bwe
S = ROOT + '/'
LIB = _os.path.expanduser('~/audio-cleaner/bin/libdf.so')
MODEL = _os.path.expanduser('~/audio-cleaner/models/DeepFilterNet3_onnx.tar.gz')
files = sorted(glob.glob(S + 'reco/set/*.wav') + glob.glob(S + 'reco/deg/*.wav') + glob.glob(S + 'effiles/*.wav') + glob.glob(S + 'reco/nosp/*.wav') + glob.glob(S + 'reco/long/*.wav'))
out = subprocess.run(['node', REPO + '/web/tests/reco-cli.mjs', '--reco'] + files, capture_output=True, text=True, cwd=REPO + '/web')
assert out.returncode == 0, out.stderr[-500:]
js = {json.loads(l)['file']: json.loads(l) for l in out.stdout.splitlines()}
dnd = []; dbw = 0; pick = 0; heavy_ok = 0; layout_ok = 0; n = 0; reb = 0
for f in files:
    x, _ = sf.read(f, dtype='float32', always_2d=True)
    j = js[f]; n += 1
    nd = pipeline.noise_figure_db(x, LIB, MODEL); dnd.append(abs(nd - j['noiseDb']))
    bw = bwe.effective_bandwidth(x); dbw = max(dbw, abs(bw - j['bandwidthHz'])); pick += bwe.pick_model(bw) == bwe.pick_model(j['bandwidthHz'])
    heavy_ok += (j['noiseDb'] < pipeline.HEAVY_BELOW_DB) == (nd < pipeline.HEAVY_BELOW_DB)
    plan = tuple(pipeline.channel_plan(x))
    want = {(0,): 'mono', (0, 1): 'stereo', (0, 0): 'dual-mono', (None, 1): 'one-side', (0, None): 'one-side'}[plan]
    layout_ok += (j['layout'] == want) or (j['layout'] == 'silent')
print(f'{n} files: max |noise figure diff| {max(dnd):.6f} dB; max |bandwidth diff| {dbw} Hz; same bwe generator {pick}/{n}; same heavy decision {heavy_ok}/{n}; same channel layout {layout_ok}/{n}')
noisy_reb = [f for f in files if '/set/n_' in f and js[f]['reco'] and js[f]['reco']['rebuild']]
clean_reb = [f for f in files if ('/set/clean_' in f) and js[f]['reco'] and js[f]['reco']['rebuild']]
print('rebuild recommended on noisy mixes:', len(noisy_reb), 'of', sum('/set/n_' in f for f in files), '; on clean clips:', len(clean_reb), 'of', sum('/set/clean_' in f for f in files))
