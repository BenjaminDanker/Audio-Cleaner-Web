import os as _os
from pathlib import Path as _Path
# Work folder with libri/ (LibriSpeech stream16k.wav), reco/ (generated material and results) and effiles/. Set RECO_ROOT to move it.
ROOT = _os.environ.get('RECO_ROOT', str(_Path.home() / 'audio-cleaner' / 'research' / 'reco-calibration'))
REPO = str(_Path(__file__).resolve().parents[2])
import json
from pathlib import Path
import numpy as np, soundfile as sf
S = Path(ROOT + '/reco'); (S/'nosp').mkdir(exist_ok=True)
SR=48000; rng=np.random.default_rng(3)
def pink(n):
    f=np.fft.rfftfreq(n,1/SR); sp=(rng.standard_normal(len(f))+1j*rng.standard_normal(len(f)))/np.sqrt(np.maximum(f,20)); return np.fft.irfft(sp,n)
def brown(n):
    f=np.fft.rfftfreq(n,1/SR); sp=(rng.standard_normal(len(f))+1j*rng.standard_normal(len(f)))/np.maximum(f,20); return np.fft.irfft(sp,n)
def norm(x,db): return (x*(10**(db/20)/np.sqrt(np.mean(x**2)))).astype(np.float32)
n=12*SR; t=np.arange(n)/SR
sig={}
for db in (-50,-30,-20):
    sig[f'white{db}']=norm(rng.standard_normal(n),db)
    sig[f'pink{db}']=norm(pink(n),db)
    sig[f'brown{db}']=norm(brown(n),db)
sig['hum-40']=norm(sum(np.sin(2*np.pi*50*k*t)/k for k in range(1,6)),-40)
sig['tone-30']=norm(np.sin(2*np.pi*440*t),-30)
sig['fan-35']=norm(pink(n)*(1+0.25*np.sin(2*np.pi*0.7*t)),-35)
sig['aircon-30']=norm(brown(n)*(1+0.4*np.sin(2*np.pi*0.2*t))+0.2*pink(n),-30)
sig['silence']=np.zeros(n,np.float32)
sig['lownoise-80']=norm(rng.standard_normal(n),-80)
# music-like: notes with ADSR
def music(bpm):
    y=np.zeros(n); step=int(60/bpm*SR/2)
    for k,pos in enumerate(range(0,n-step*2,step)):
        f=220*2**(rng.integers(0,12)/12); L=step*2
        tt=np.arange(L)/SR; env=np.minimum(1,tt/0.01)*np.exp(-tt*3)
        y[pos:pos+L]+=sum(np.sin(2*np.pi*f*h*tt)/h for h in range(1,5))*env
    return y
sig['music-100']=norm(music(100),-25); sig['music-140']=norm(music(140),-25)
# noise bursts (a dog bark / door slam like): speech-like levels but not speech
b=np.zeros(n)
for pos in range(SR,n-SR,int(1.7*SR)):
    L=int(0.25*SR); b[pos:pos+L]+=rng.standard_normal(L)*np.exp(-np.arange(L)/(0.05*SR))
sig['bursts']=norm(b+0.003*rng.standard_normal(n),-28)
for k,v in sig.items(): sf.write(S/f'nosp/{k}.wav',v,SR,subtype='FLOAT')
