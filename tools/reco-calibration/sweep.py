import os as _os
from pathlib import Path as _Path
# Work folder with libri/ (LibriSpeech stream16k.wav), reco/ (generated material and results) and effiles/. Set RECO_ROOT to move it.
ROOT = _os.environ.get('RECO_ROOT', str(_Path.home() / 'audio-cleaner' / 'research' / 'reco-calibration'))
REPO = str(_Path(__file__).resolve().parents[2])
import json, numpy as np, collections, itertools, sys
S=ROOT + '/reco/'
rows=[json.loads(l) for l in open(S+'srv_12.jsonl')]; rows=[r for r in rows if 'error' not in r]
feat={}
for l in open(S+'feat_full.jsonl'):
    j=json.loads(l); feat[j['file'].split('/')[-1][:-4]]=j
for r in rows:
    f=feat[r['name']]; r['floor']=f['floorDb']; r['level']=f['levelDb']
    g={s:r[f'sisdr_dfn{s}']-r['sisdr_in'] for s in (12,24,40,100)}
    r['g']=g; r['gl']=g[100]; r['gh']=r['sisdr_heavy']-r['sisdr_in']
def label(r,gmin):
    if max(r['gl'],r['gh'])<gmin: return 'off'
    return 'heavy' if r['gh']-r['gl']>0.3 else 'light'
HEAVY_CUT=17.5
def rec(r,c,fl):
    if r['nd']>=c and r['floor']<=fl: return 'off'
    return 'heavy' if r['nd']<HEAVY_CUT else 'light'
for gmin in (2,3):
    print('gmin',gmin, 'label dist', collections.Counter(label(r,gmin) for r in rows))
    best=[]
    for c in range(24,46,2):
        for fl in (-40,-44,-48,-52,-56,0):
            acc=np.mean([rec(r,c,fl)==label(r,gmin) for r in rows])
            # stationary-only (white, pink, traffic, hum, clean)
            st=[r for r in rows if r['kind'] in ('white','pink','traffic','hum','clean')]
            acc_st=np.mean([rec(r,c,fl)==label(r,gmin) for r in st])
            best.append((acc,acc_st,c,fl))
    best.sort(reverse=True)
    print(' top (all, stationary, c, floor):',[(round(a,3),round(b,3),c,fl) for a,b,c,fl in best[:6]])
    # what is the always-run baseline (server auto)?
    print(' baseline always-denoise auto:', round(np.mean([('heavy' if r['nd']<19 else 'light')==label(r,gmin) for r in rows]),3))

print('\n--- binary: remove or not (gmin 2 / 3), by nd cut, floor<=-48 or none')
for gmin in (2,3):
    for fl in (-48,0):
        res=[]
        for c in range(22,46,2):
            acc=np.mean([ (not (r['nd']>=c and r['floor']<=fl)) == (label(r,gmin)!='off') for r in rows])
            res.append((c,round(acc,3)))
        print(gmin,fl,res)
print('\n--- among files where removal is worth it (gmin 2): heavy vs light by nd cut')
w=[r for r in rows if label(r,2)!='off']
for cut in (15,17.5,19,21,23,25):
    print(cut, round(np.mean([('heavy' if r['nd']<cut else 'light')==label(r,2) for r in w]),3), 'n',len(w))
print('mean gh-gl by nd bucket among worthwhile:')
for a,b in zip((0,10,14,17,19,21,23,26,30),(10,14,17,19,21,23,26,30,60)):
    v=[r for r in w if a<=r['nd']<b]
    if v: print(f'  {a}-{b}: n={len(v)} gh-gl mean {np.mean([r["gh"]-r["gl"] for r in v]):.2f}  heavy better>0.3: {np.mean([r["gh"]-r["gl"]>0.3 for r in v]):.2f}')
