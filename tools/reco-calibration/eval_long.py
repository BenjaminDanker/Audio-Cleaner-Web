"""Long files and the live window: is a ~minute sampled from a long file (and the first 10 s of a microphone) judged like the whole file?
Compares: server (Python, whole file) / browser analysis of 6 sampled stretches / browser analysis of the first 10 s, and whether
the plugin's C++ analyser hears speech in the first 10 s where the browser does (its noise figure comes from the live engine)."""
import os as _os
from pathlib import Path as _Path
# Work folder with libri/ (LibriSpeech stream16k.wav), reco/ (generated material and results) and effiles/. Set RECO_ROOT to move it.
ROOT = _os.environ.get('RECO_ROOT', str(_Path.home() / 'audio-cleaner' / 'research' / 'reco-calibration'))
REPO = str(_Path(__file__).resolve().parents[2])
import json, subprocess, sys, os, collections, glob
import numpy as np, soundfile as sf
sys.path.insert(0, REPO + '/common')
from acw_common import pipeline
S = ROOT + '/reco/'
LIB = _os.path.expanduser('~/audio-cleaner/bin/libdf.so')
MODEL = _os.path.expanduser('~/audio-cleaner/models/DeepFilterNet3_onnx.tar.gz')
files = sorted(glob.glob(S + 'long/*.wav'))
def node(args):
    o = subprocess.run(['node', REPO + '/web/tests/reco-cli.mjs'] + args + files, capture_output=True, text=True, cwd=REPO + '/web')
    assert o.returncode == 0, o.stderr[-500:]
    return {json.loads(l)['file']: json.loads(l) for l in o.stdout.splitlines()}
full = node(['--reco']); samp = node(['--sampled', '--reco']); first = node(['--first=10', '--reco'])
cpp = {}
env = dict(os.environ, LIMIT_S='10')
for f in files:
    o = subprocess.run([S + 'advice_cli', f], capture_output=True, text=True, env=env).stdout.split()
    cpp[f] = dict(speech=bool(int(o[1])))
def cls(j):
    r = j['reco']
    if r is None or r['noise'] == 'off': return 'off'
    return 'heavy' if r['heavy'] else 'light'
rows = []
for f in files:
    x, _ = sf.read(f, dtype='float32')
    nd = pipeline.noise_figure_db(x, LIB, MODEL)
    rows.append(dict(f=f, secs=len(x) / 48000, nd=nd, server='heavy' if nd < pipeline.HEAVY_BELOW_DB else 'light', full=cls(full[f]), samp=cls(samp[f]), first=cls(first[f]),
                     nd_full=full[f]['noiseDb'], nd_samp=samp[f]['noiseDb'], nd_first=first[f]['noiseDb'], str_full=full[f]['reco']['strength'], str_samp=samp[f]['reco']['strength'], str_first=first[f]['reco']['strength'],
                     speech_cpp=cpp[f]['speech'], speech_first=first[f]['hasSpeech']))
n = len(rows)
print(f'{n} long files ({np.mean([r["secs"] for r in rows]):.0f} s each)')
d = lambda a, b: np.array([r[a] - r[b] for r in rows])
print(f'noise figure, browser on the whole file vs server: max |diff| {np.abs(d("nd_full","nd")).max():.5f} dB')
print(f'noise figure, 6 sampled stretches (60 s) vs server whole file: mean |diff| {np.abs(d("nd_samp","nd")).mean():.2f} dB, max {np.abs(d("nd_samp","nd")).max():.2f} dB, 90% within {np.percentile(np.abs(d("nd_samp","nd")),90):.2f} dB')
print(f'noise figure, first 10 s vs server whole file: mean |diff| {np.abs(d("nd_first","nd")).mean():.2f} dB, max {np.abs(d("nd_first","nd")).max():.2f} dB')
print(f'speech heard, plugin (C++) first 10 s vs website first 10 s: the same in {sum(r["speech_cpp"] == r["speech_first"] for r in rows)}/{n}')
agree = lambda a, b: 100 * np.mean([r[a] == r[b] for r in rows])
print(f'remover class (off / light / heavy): sampled vs whole file in the browser {agree("samp","full"):.1f}%;  first 10 s vs whole {agree("first","full"):.1f}%')
print(f'strength: sampled vs whole {100*np.mean([r["str_samp"]==r["str_full"] for r in rows]):.1f}%;  first 10 s vs whole {100*np.mean([r["str_first"]==r["str_full"] for r in rows]):.1f}%')
srv_heavy = [r for r in rows if r['server'] == 'heavy']
print(f'heavy: sampled says heavy for {sum(r["samp"]=="heavy" for r in rows)} files, and the server agrees in {sum(r["samp"]=="heavy" and r["server"]=="heavy" for r in rows)}; first 10 s says heavy for {sum(r["first"]=="heavy" for r in rows)}, server agrees in {sum(r["first"]=="heavy" and r["server"]=="heavy" for r in rows)}; the server picks heavy for {len(srv_heavy)}')
bad = [r for r in rows if r['samp'] == 'heavy' and r['server'] != 'heavy']
print('wrongly heavy (sampled):', [(os.path.basename(r['f']), round(r['nd'], 1), round(r['nd_samp'], 1)) for r in bad])
bad = [r for r in rows if r['first'] == 'heavy' and r['server'] != 'heavy']
print('wrongly heavy (first 10 s):', [(os.path.basename(r['f']), round(r['nd'], 1), round(r['nd_first'], 1)) for r in bad])
