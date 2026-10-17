# Evaluation of skipping noisy pauses (docs/research/skip-noisy-pauses-2026-10.md). L = a scratch directory holding
# utts/ (LibriSpeech utterances as 16 kHz wav + refs.json), dfout/libdf.so (the build), item3/ (made by the make_*.py).
import sys, os, time, json, numpy as np, soundfile as sf
sys.path.insert(0,'/home/silver/repos/acw-left/common')
sys.path.insert(0, __import__('os').path.dirname(__file__))
import pauses
L=sys.argv[1]; THREADS=int(os.environ.get('THREADS','1')); LIB=L+'/dfout/libdf.so'; M='/home/silver/audio-cleaner/models/DeepFilterNet3_onnx.tar.gz'
VAD='/home/silver/audio-cleaner/models/silero-vad/silero_vad.onnx'
db=lambda v:20*np.log10(v+1e-6); rms=lambda x: float(np.sqrt(np.mean(np.square(x,dtype=np.float64))))
snr=lambda r,t: 10*np.log10(np.sum(r.astype(np.float64)**2)/np.sum((r.astype(np.float64)-t)**2))
for n in sys.argv[2:]:
    x,_=sf.read(f'{L}/item3/{n}.wav',dtype='float32'); gt=np.load(f'{L}/item3/{n}.gt.npy')
    ref=np.load(f'{L}/item3/{n}.contig.npy')
    rep={}; t=time.time(); c=time.process_time()
    y=pauses.denoise_with_pauses(x,LIB,M,VAD,threads=THREADS,report=rep)
    w=time.time()-t; cp=time.process_time()-c
    d=y-ref; skipped=rep.get('pauses_skipped_s',0)
    print(f"{n:9s} skipped {skipped:5.1f}s ({100*skipped*48000/len(x):4.1f}%) SNR vs uncut {snr(ref,y):6.1f} dB | wall {w:5.1f}s cpu {cp:5.1f}s",flush=True)
    np.save(f'{L}/item3/{n}.real_t{THREADS}.npy',y)
