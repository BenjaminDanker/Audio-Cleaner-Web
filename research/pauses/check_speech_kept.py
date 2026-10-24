# Evaluation of skipping noisy pauses (docs/research/skip-noisy-pauses-2026-10.md). L = a scratch directory holding
# utts/ (LibriSpeech utterances as 16 kHz wav + refs.json), dfout/libdf.so (the build), item3/ (made by the make_*.py).
import sys, glob, os, numpy as np, soundfile as sf
from scipy.ndimage import distance_transform_edt
sys.path.insert(0,'/home/silver/repos/Audio-Cleaner-Web/common')
sys.path.insert(0, __import__('os').path.dirname(__file__))
import pauses
L=sys.argv[1]; SR=48000; fr=480; VAD='/home/silver/audio-cleaner/models/silero-vad/silero_vad.onnx'
tot=0
for f in sorted(glob.glob(L+'/item3/*.full.npy')):
    n=os.path.basename(f)[:-9]; x,_=sf.read(f'{L}/item3/{n}.wav',dtype='float32'); clean=np.load(f'{L}/item3/{n}.clean.npy'); gt0=np.load(f'{L}/item3/{n}.gt.npy')
    nf=len(clean)//fr; e=10*np.log10(np.mean(clean[:nf*fr].reshape(nf,fr)**2,axis=1)+1e-12)
    act=(e>np.percentile(e[gt0[::fr][:nf]],95)-28)&gt0[::fr][:nf]
    strict=np.zeros(len(clean),bool); strict[:nf*fr]=np.repeat(act,fr)
    probs,en=pauses.speech_probs(x,VAD); holes=pauses.candidate_holes(probs,len(x))
    hm=np.zeros(len(x),bool)
    for s,e_ in holes: hm[s:e_]=True
    dist=distance_transform_edt(~strict)/SR   # seconds to nearest truth speech sample
    md=min((dist[s:e_].min() for s,e_ in holes),default=float('nan'))
    print(f'{n:9s} candidate holes {len(holes):2d} ({hm.mean()*100:4.1f}% of file): truth-speech samples inside holes: loose {int((hm&gt0).sum())} strict {int((hm&strict).sum())}; nearest truth speech to any hole: {md:.2f} s')
