"""Does the recommendation match the best choice by the server's own measurements?  (final run; numbers go into docs/research/recommend-2026-10.md)
Rows: srv_12.jsonl + srv2.jsonl: SI-SDR against the clean clip and the noise left in the pauses, for the server's own noise removers at 4 strengths.
The recommendation is web/src/recommend.js run by Node on the same files (sampled mode, as in the browser). Subtitles: wer.jsonl."""
import os as _os
from pathlib import Path as _Path
# Work folder with libri/ (LibriSpeech stream16k.wav), reco/ (generated material and results) and effiles/. Set RECO_ROOT to move it.
ROOT = _os.environ.get('RECO_ROOT', str(_Path.home() / 'audio-cleaner' / 'research' / 'reco-calibration'))
REPO = str(_Path(__file__).resolve().parents[2])
import json, subprocess, sys, collections
import numpy as np
S = ROOT + '/reco/'
base = {}
for l in open(S + 'srv_12.jsonl'):
    r = json.loads(l)
    if 'error' not in r: base[r['name']] = r
rows = []
for l in open(S + 'srv2.jsonl'):
    d = json.loads(l); rows.append({**base[d['name']], **d})
files = [S + 'set/' + r['name'] + '.wav' for r in rows]
out = subprocess.run(['node', REPO + '/web/tests/reco-cli.mjs', '--sampled', '--reco'] + files, capture_output=True, text=True, cwd=REPO + '/web')
assert out.returncode == 0, out.stderr
js = {json.loads(l)['file'].split('/')[-1][:-4]: json.loads(l) for l in out.stdout.splitlines()}
STEPS = [12, 24, 40, 100]
ID = {'gentle': 12, 'balanced': 24, 'strong': 40, 'max': 100}
for r in rows:
    r['js'] = js[r['name']]
    r['reco'] = r['js']['reco']
    r['server_heavy'] = r['nd'] < 19
    m = 'h' if r['server_heavy'] else 'l'
    r['g'] = {s: r[f'si_{m}{s}'] - r['sisdr_in'] for s in STEPS}
    r['res'] = {s: r[f'res_{m}{s}'] for s in STEPS}
    r['gl'] = r['si_l100'] - r['sisdr_in']
    r['gh'] = (r['si_h100'] - r['sisdr_in']) if 'si_h100' in r else r['sisdr_heavy'] - r['sisdr_in']
    r['gh'] = r['sisdr_heavy'] - r['sisdr_in']
    gmax = max(r['g'].values()); rmin = min(r['res'].values())
    r['minS'] = next((s for s in STEPS if r['g'][s] >= gmax - 0.75 and r['res'][s] <= rmin + 4), 100)

def best_class(r, gmin):
    if max(r['gl'], r['gh']) < gmin: return 'off'
    return 'heavy' if r['gh'] - r['gl'] > 0.3 else 'light'

def rec_class(r):
    x = r['reco']
    if x is None or x['noise'] == 'off': return 'off'
    return 'heavy' if x['heavy'] else 'light'

n = len(rows)
print(f'{n} mixes ({collections.Counter(r["kind"] for r in rows)})')
stationary = {'white', 'pink', 'traffic', 'hum', 'clean'}
for gmin in (1.0, 2.0, 3.0):
    conf = collections.Counter((best_class(r, gmin), rec_class(r)) for r in rows)
    agree = sum(v for (b, c), v in conf.items() if b == c)
    st = [r for r in rows if r['kind'] in stationary]
    agree_st = sum(best_class(r, gmin) == rec_class(r) for r in st)
    base_auto = sum(best_class(r, gmin) == ('heavy' if r['server_heavy'] else 'light') for r in rows)
    onoff = sum((best_class(r, gmin) == 'off') == (rec_class(r) == 'off') for r in rows)
    print(f'\nworth-it threshold {gmin} dB of SI-SDR gain')
    print('  best \\ recommended   off light heavy')
    for b in ('off', 'light', 'heavy'):
        print(f'  {b:>18}  {conf[(b,"off")]:4d} {conf[(b,"light")]:5d} {conf[(b,"heavy")]:5d}')
    print(f'  3-way agreement {100*agree/n:.1f}% (stationary noise only: {100*agree_st/len(st):.1f}% of {len(st)})   remove-or-not {100*onoff/n:.1f}%   baseline "always run noise removal, auto": {100*base_auto/n:.1f}%')
    if gmin == 2.0:
        main = (conf, agree, agree_st, len(st), onoff, base_auto)
# heavy validity
rec_h = [r for r in rows if rec_class(r) == 'heavy']
srv_h = [r for r in rows if r['server_heavy']]
print(f'\nheavy recommended for {len(rec_h)} mixes; the server (whole-clip noise figure < 19) picks the stronger remover for {sum(r["server_heavy"] for r in rec_h)} of them ({100*sum(r["server_heavy"] for r in rec_h)/max(1,len(rec_h)):.1f}%). '
      f'Of the {len(srv_h)} mixes the server sends to the stronger remover, {sum(rec_class(r)=="heavy" for r in srv_h)} are recommended heavy (the rest are within the 1.5 dB margin below the cut: {sum(1 for r in srv_h if rec_class(r)=="light")} said light, {sum(1 for r in srv_h if rec_class(r)=="off")} said off)')
# strength
use = [r for r in rows if rec_class(r) != 'off']
def ev(name, f):
    ok = np.mean([r['minS'] <= f(r) for r in use]); ex = np.mean([r['minS'] == f(r) for r in use]); over = np.mean([f(r) > r['minS'] for r in use])
    w1 = np.mean([abs(STEPS.index(f(r)) - STEPS.index(r['minS'])) <= 1 for r in use])
    print(f'  {name:30s} never under-cleans {100*ok:5.1f}%   exact {100*ex:5.1f}%   within one step {100*w1:5.1f}%   stronger than needed {100*over:5.1f}%')
print(f'\nstrength on the {len(use)} mixes where noise removal is recommended (needed = gentlest step within 0.75 dB SI-SDR and 4 dB of pause noise of Maximum):')
ev('recommended', lambda r: ID[r['reco']['strength']])
ev('always Maximum (today)', lambda r: 100)
ev('always Balanced', lambda r: 24)
# by kind
print('\n3-way agreement by noise kind (threshold 2 dB):', {k: f'{100*np.mean([best_class(r,2)==rec_class(r) for r in rows if r["kind"]==k]):.0f}% of {sum(1 for r in rows if r["kind"]==k)}' for k in sorted(set(r["kind"] for r in rows))})
# subtitles
wer = [json.loads(l) for l in open(S + 'wer.jsonl')]
pts = []
for w in wer:
    if w['kind'] in ('clean', 'tempo'): continue
    name = f"n_{w['kind']}_{w['level']:02d}_{w['clip']}"
    if name in js: pts.append((js[name]['reco']['subs'] if js[name]['reco'] else None, w['errors'] / max(1, w['words']), js[name]['noiseDb']))
need = [p[1] > 0.10 for p in pts]; flag = [p[0] == 'accurate' for p in pts]
tp = sum(a and b for a, b in zip(need, flag)); fp = sum((not a) and b for a, b in zip(need, flag)); fn = sum(a and not b for a, b in zip(need, flag))
print(f'\nsubtitles: {len(pts)} noisy clips; Standard changes more than 10% of the words in {sum(need)} of them. Extra accurate recommended for {sum(flag)}: '
      f'recall {100*tp/max(1,tp+fn):.0f}%, precision {100*tp/max(1,tp+fp):.0f}%, agreement {100*sum(a==b for a,b in zip(need,flag))/len(pts):.0f}% (always-Standard: {100*sum(not a for a in need)/len(pts):.0f}%)')
tw = {}
for w in wer:
    if w['kind'] == 'tempo': tw.setdefault(w['level'], [0, 0]); tw[w['level']][0] += w['errors']; tw[w['level']][1] += w['words']
print('speed (changes vs normal speed, Standard):', {k: f'{100*v[0]/v[1]:.1f}%' for k, v in tw.items()})
