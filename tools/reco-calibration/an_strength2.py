import os as _os
from pathlib import Path as _Path
# Work folder with libri/ (LibriSpeech stream16k.wav), reco/ (generated material and results) and effiles/. Set RECO_ROOT to move it.
ROOT = _os.environ.get('RECO_ROOT', str(_Path.home() / 'audio-cleaner' / 'research' / 'reco-calibration'))
REPO = str(_Path(__file__).resolve().parents[2])
import json, collections, itertools, numpy as np, sys
S=ROOT + '/reco/'
base={}
for l in open(S+'srv_12.jsonl'):
    r=json.loads(l)
    if 'error' not in r: base[r['name']]=r
rows=[]
for l in open(S+'srv2.jsonl'):
    d=json.loads(l); r={**base[d['name']],**d}
    heavy=r['nd']<19; m='h' if heavy else 'l'
    r['g']={s:r[f'si_{m}{s}']-r['sisdr_in'] for s in (12,24,40,100)}
    r['res']={s:r[f'res_{m}{s}'] for s in (12,24,40,100)}
    rows.append(r)
def minS(r,tsi,tres):
    gmax=max(r['g'].values()); rmin=min(r['res'].values())
    for s in (12,24,40,100):
        if r['g'][s]>=gmax-tsi and r['res'][s]<=rmin+tres: return s
    return 100
worth=[r for r in rows if r['nd']<34]
for tsi,tres in ((0.5,2),(0.5,4),(0.75,4),(1.0,5),(1.0,6)):
    for r in rows: r['minS']=minS(r,tsi,tres)
    c=collections.Counter()
    best=[]
    for t in itertools.combinations([6,9,12,15,18,21,24,27,30,33,36],3):
        rule=lambda nd: 100 if nd<t[0] else 40 if nd<t[1] else 24 if nd<t[2] else 12
        ok=np.mean([r['minS']<=rule(r['nd']) for r in worth])     # the recommended strength is at least as strong as needed
        exact=np.mean([r['minS']==rule(r['nd']) for r in worth])
        over=np.mean([rule(r['nd'])>r['minS'] for r in worth])
        best.append((exact+0.5*ok,ok,exact,over,t))
    best.sort(reverse=True)
    print(f'tol SI-SDR {tsi} dB, pause residual {tres} dB:')
    for sc,ok,exact,over,t in best[:3]: print(f'   t={t}: never under-cleans {100*ok:.0f}%  exact {100*exact:.0f}%  stronger than needed {100*over:.0f}%')

print('\n--- candidate final rules, rows where noise removal is recommended (nd < 29), tolerance SI-SDR 0.75 dB / residual 4 dB')
for r in rows: r['minS']=minS(r,0.75,4)
use=[r for r in rows if r['nd']<29]
def evaluate(name, rule):
    ok=np.mean([r['minS']<=rule(r['nd']) for r in use]); ex=np.mean([r['minS']==rule(r['nd']) for r in use]); over=np.mean([rule(r['nd'])>r['minS'] for r in use])
    # within one step
    steps=[12,24,40,100]
    w1=np.mean([abs(steps.index(rule(r['nd']))-steps.index(r['minS']))<=1 for r in use])
    print(f'{name:38s} n={len(use)} never under-cleans {100*ok:.1f}%  exact {100*ex:.1f}%  within one step {100*w1:.1f}%  stronger than needed {100*over:.1f}%')
evaluate('always Maximum (today)', lambda nd: 100)
evaluate('always Balanced', lambda nd: 24)
evaluate('18/21/26: Max<18, Strong<21, Bal<26, Gentle', lambda nd: 100 if nd<18 else 40 if nd<21 else 24 if nd<26 else 12)
evaluate('18/21/27', lambda nd: 100 if nd<18 else 40 if nd<21 else 24 if nd<27 else 12)
evaluate('18/21/29 (no gentle)', lambda nd: 100 if nd<18 else 40 if nd<21 else 24)
evaluate('15/18/26', lambda nd: 100 if nd<15 else 40 if nd<18 else 24 if nd<26 else 12)
evaluate('18/24/27', lambda nd: 100 if nd<18 else 40 if nd<24 else 24 if nd<27 else 12)
