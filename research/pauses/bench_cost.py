# Evaluation of skipping noisy pauses (docs/research/skip-noisy-pauses-2026-10.md). L = a scratch directory holding
# utts/ (LibriSpeech utterances as 16 kHz wav + refs.json), dfout/libdf.so (the build), item3/ (made by the make_*.py).
import sys, os, time, numpy as np, soundfile as sf
sys.path.insert(0,'/home/silver/repos/Audio-Cleaner-Web/common')
sys.path.insert(0, __import__('os').path.dirname(__file__))
import pauses
L=sys.argv[1]; LIB=L+'/dfout/libdf.so'; M='/home/silver/audio-cleaner/models/DeepFilterNet3_onnx.tar.gz'; VAD='/home/silver/audio-cleaner/models/silero-vad/silero_vad.onnx'
n=sys.argv[2]; x,_=sf.read(f'{L}/item3/{n}.wav',dtype='float32'); X=x[:,None]
def run(skip):
    rep={}; t=time.perf_counter(); c=time.process_time()
    y=pauses.denoise_with_pauses(x,LIB,M,VAD,threads=4,skip=skip,report=rep)
    return y,time.perf_counter()-t,time.process_time()-c,rep
run(True) if False else None
res={True:[],False:[]}; outs={}
for i in range(3):
    for skip in (False,True):
        y,w,c,rep=run(skip); res[skip].append((w,c)); outs[skip]=y
        print(f'{n} skip_pauses={skip}: wall {w:.1f}s cpu {c:.1f}s {rep.get("pauses_skipped_s","")}',flush=True)
snr=10*np.log10(np.sum(outs[False].astype(np.float64)**2)/np.sum((outs[False]-outs[True]).astype(np.float64)**2))
b=np.median([c for w,c in res[False]]); a=np.median([c for w,c in res[True]]); bw=np.median([w for w,c in res[False]]); aw=np.median([w for w,c in res[True]])
print(f'{n}: median cpu {b:.1f}s -> {a:.1f}s ({100*(1-a/b):.1f}% less), wall {bw:.1f}s -> {aw:.1f}s ({100*(1-aw/bw):.1f}%); SNR of skipped vs full run (4 threads, 60 s segments) {snr:.1f} dB')
