# Evaluation of skipping noisy pauses (docs/research/skip-noisy-pauses-2026-10.md). L = a scratch directory holding
# utts/ (LibriSpeech utterances as 16 kHz wav + refs.json), dfout/libdf.so (the build), item3/ (made by the make_*.py).
import sys, os, numpy as np, soundfile as sf
sys.path.insert(0,'/home/silver/repos/Audio-Cleaner-Web/common')
sys.path.insert(0, __import__('os').path.dirname(__file__))
import pauses
from acw_common import dfn
L=sys.argv[1]; LIB=L+'/dfout/libdf.so'; M='/home/silver/audio-cleaner/models/DeepFilterNet3_onnx.tar.gz'; VAD='/home/silver/audio-cleaner/models/silero-vad/silero_vad.onnx'
for n in sys.argv[2:]:
    x,_=sf.read(f'{L}/item3/{n}.wav',dtype='float32'); st={}
    h=pauses.plan(x,VAD,lambda w: dfn.denoise_array(w,LIB,M,threads=1),stats=st)
    print(n,len(h),sum(e-s for s,e in h)/48000,{k:round(v,1) for k,v in st.items()})
