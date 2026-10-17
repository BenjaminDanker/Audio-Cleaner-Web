# Evaluation of skipping noisy pauses (docs/research/skip-noisy-pauses-2026-10.md). L = a scratch directory holding
# utts/ (LibriSpeech utterances as 16 kHz wav + refs.json), dfout/libdf.so (the build), item3/ (made by the make_*.py).
import sys, json, numpy as np, soundfile as sf
from scipy.signal import resample_poly
L=sys.argv[1]; SR=48000
refs=json.load(open(L+'/utts/refs.json'))
utts=[sf.read(f"{L}/utts/{r['id']}.wav",dtype='float32')[0] for r in refs]
db=lambda v:20*np.log10(v+1e-12); rms=lambda x: float(np.sqrt(np.mean(np.square(x,dtype=np.float64))+1e-30))
for env,seed in (('DLIVING',21),('OOFFICE',22),('DKITCHEN',23)):
    nz0,_=sf.read(f'{L}/demand/{env}_ch01.wav',dtype='float32')
    for snr in (15,25,35):
        rng=np.random.default_rng(seed*100+snr)
        order=rng.permutation(len(utts)); parts=[]; mask=[]
        def add(x,m): parts.append(x); mask.append(np.full(len(x),m,bool))
        add(np.zeros(int(rng.uniform(0.5,1.5)*SR),np.float32),False)
        for i in order:
            u=resample_poly(utts[i],3,1).astype(np.float32); fr=480; n=len(u)//fr
            e=10*np.log10(np.mean(u[:n*fr].reshape(n,fr)**2,axis=1)+1e-12); act=e>np.percentile(e,95)-38
            m=np.zeros(len(u),bool); m[:n*fr]=np.repeat(act,fr); add(u,m)
            gap=rng.uniform(1.5,6.0) if rng.random()<0.4 else rng.uniform(0.25,1.0); add(np.zeros(int(gap*SR),np.float32),False)
        sp=np.concatenate(parts); gt=np.concatenate(mask); sp*=10**((-23-db(rms(sp[gt])))/20)
        off=int(rng.integers(0,len(nz0))); nz=np.resize(np.roll(nz0,-off),len(sp)).astype(np.float64)
        nz*=10**((-23-snr-db(rms(nz)))/20)
        pk=float(np.abs(sp+nz).max()); g=min(1.0,0.99/pk); sp=sp*g; nz=nz*g; x=(sp+nz).astype(np.float32)
        name=f'{env}_{snr}'; sf.write(f'{L}/item3/{name}.wav',x,SR,subtype='FLOAT'); np.save(f'{L}/item3/{name}.gt.npy',gt); np.save(f'{L}/item3/{name}.clean.npy',sp.astype(np.float32))
        print(name,f'{len(x)/SR:.0f}s noise {db(rms(nz)):.1f} dBFS')
