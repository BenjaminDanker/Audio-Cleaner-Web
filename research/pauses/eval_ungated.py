# Evaluation of skipping noisy pauses (docs/research/skip-noisy-pauses-2026-10.md). L = a scratch directory holding
# utts/ (LibriSpeech utterances as 16 kHz wav + refs.json), dfout/libdf.so (the build), item3/ (made by the make_*.py).
import sys, glob, os, json, time, numpy as np, soundfile as sf
from scipy.signal import resample_poly
from scipy.ndimage import binary_dilation
sys.path.insert(0,'/home/silver/repos/Audio-Cleaner-Web/common')
from acw_common import dfn
from faster_whisper.vad import get_vad_model
L=sys.argv[1]; SR=48000; LIB=L+'/dfout/libdf.so'; M='/home/silver/audio-cleaner/models/DeepFilterNet3_onnx.tar.gz'
db=lambda v:20*np.log10(v+1e-6); rms=lambda x: float(np.sqrt(np.mean(np.square(x,dtype=np.float64))))
THR=float(os.environ.get('THR','0.1')); PAD=float(os.environ.get('PAD','0.3')); GAP=1.5; CTX=float(os.environ.get('CTX','0.5'))
def speech_probs(x):
    a=resample_poly(x,1,3).astype(np.float32); n=(len(a)+511)//512*512
    p=get_vad_model()(np.pad(a,(0,n-len(a)))); return p.reshape(-1)  # per 512 samples @16k = 32 ms
def stretches(x):
    p=speech_probs(x); sp=p>THR
    sp=binary_dilation(sp,iterations=int(round(PAD/0.032)))
    out=[]; i=0
    while i<len(sp):
        if sp[i]: i+=1; continue
        j=i
        while j<len(sp) and not sp[j]: j+=1
        a,b=int(i*512*3),min(len(x),int(j*512*3))
        if (b-a)>=GAP*SR: out.append((a,b))
        i=j
    return out
def kept_spans(n, skips):
    spans=[]; cur=0
    for a,b in skips:
        s,e=-(-(a+int(CTX*SR))//480)*480,(b-int(CTX*SR))//480*480   # on the 10 ms hop grid: the audio after the hole keeps its hop phase
        if e-s<=0: continue
        spans.append((cur,s)); cur=e
    spans.append((cur,n)); return spans,[(s,e) for (_,s),(e,_) in zip(spans[:-1],spans[1:])]
SPLICE=os.environ.get('SPLICE','1')=='1'
def run(x, spans):
    y=np.zeros(len(x),np.float32)
    if not SPLICE:
        for a,b in spans: y[a:b]=dfn.denoise_array(x[a:b],LIB,M,threads=int(os.environ.get('THREADS','4')))
        return y
    cat=np.concatenate([x[a:b] for a,b in spans]); yc=dfn.denoise_array(cat,LIB,M,threads=int(os.environ.get('THREADS','4'))); p=0
    for a,b in spans: y[a:b]=yc[p:p+b-a]; p+=b-a
    return y
def fill(x,y,holes,mode):
    z=y.copy(); q=int(0.25*SR); f=int(float(os.environ.get('FADE','0.04'))*SR)
    for s,e in holes:
        if mode=='zero': z[s:e]=0
        else:
            gl=rms(y[s-q:s])/(rms(x[s-q:s])+1e-12); gr=rms(y[e:e+q])/(rms(x[e:e+q])+1e-12)
            g=np.linspace(gl,gr,e-s).astype(np.float32); z[s:e]=x[s:e]*g
        # short fades on the model's own output at both joins (20 ms) so nothing steps
        if mode=='zero':
            z[s-f:s]*=np.linspace(1,0,f,dtype=np.float32); z[e:e+f]*=np.linspace(0,1,f,dtype=np.float32)
        else:
            w=np.linspace(0,1,f,dtype=np.float32)
            z[s-f:s]=y[s-f:s]*(1-w)+x[s-f:s]*np.linspace(gl,gl,f)*w
            z[e:e+f]=x[e:e+f]*gr*(1-w)+y[e:e+f]*w
    return z
def snr(ref,t): return 10*np.log10(np.sum(ref.astype(np.float64)**2)/(np.sum((ref.astype(np.float64)-t)**2)+1e-30))
res={}
for f in sorted(glob.glob(L+'/item3/*.full.npy')):
    n=os.path.basename(f)[:-9]
    if len(sys.argv)>2 and n not in sys.argv[2:]: continue
    x,_=sf.read(f'{L}/item3/{n}.wav',dtype='float32'); full=np.load(f) if os.environ.get('REF','full')=='full' else np.load(f'{L}/item3/{n}.contig.npy'); gt=np.load(f'{L}/item3/{n}.gt.npy')
    t=time.time(); sk=stretches(x); tv=time.time()-t
    spans,holes=kept_spans(len(x),sk)
    skipped=sum(e-s for s,e in holes)
    # never skip speech: ground-truth speech frames inside any hole
    hole_mask=np.zeros(len(x),bool)
    for s,e in holes: hole_mask[s:e]=True
    speech_lost=int(np.sum(hole_mask&gt)); 
    y=run(x,spans)
    r={'skipped_pct':100*skipped/len(x),'vad_s':tv,'speech_samples_in_holes':speech_lost,'holes':len(holes)}
    for mode in ('zero','scaled'):
        z=fill(x,y,holes,mode)
        r[mode]={'snr_all':snr(full,z),'snr_nonhole':snr(full[~hole_mask],z[~hole_mask]) if (~hole_mask).any() else None,
                 'hole_rms_full_dbfs':db(rms(full[hole_mask])) if hole_mask.any() else None,'hole_err_dbfs':db(rms((full-z)[hole_mask])) if hole_mask.any() else None}
        # level step at joins: 20 ms rms before vs after the join, skip result minus full run
        steps=[]
        for s,e in holes:
            for j in (s,e):
                b=lambda v: db(rms(v[j-int(0.02*SR):j])); a=lambda v: db(rms(v[j:j+int(0.02*SR)]))
                steps.append(abs((a(z)-b(z))-(a(full)-b(full))))
        r[mode]['max_join_step_db']=max(steps) if steps else 0
        r[mode]['max_abs_diff_at_joins']=float(max([np.max(np.abs(z[j-480:j+480]-full[j-480:j+480])) for s,e in holes for j in (s,e)] or [0]))
        if mode=='zero': np.save(f'{L}/item3/{n}.skip_zero_s{int(SPLICE)}.npy',z)
        else: np.save(f'{L}/item3/{n}.skip_scaled_s{int(SPLICE)}.npy',z)
    res[n]=r
    F=lambda v: 'n/a' if v is None else f'{v:.1f}'
    print(n, f"skipped {r['skipped_pct']:.1f}% in {r['holes']} holes, speech samples lost {speech_lost}, VAD {tv:.1f}s | zero: SNR {F(r['zero']['snr_all'])} dB (non-hole {F(r['zero']['snr_nonhole'])}), hole full-run level {F(r['zero']['hole_rms_full_dbfs'])} dBFS, max join step {r['zero']['max_join_step_db']:.1f} dB | scaled: SNR {F(r['scaled']['snr_all'])} (non-hole {F(r['scaled']['snr_nonhole'])}), step {r['scaled']['max_join_step_db']:.1f} dB",flush=True)
json.dump(res,open(f'{L}/item3/skip_results_splice{int(SPLICE)}_ctx{CTX}_thr{THR}_pad{PAD}.json','w'),indent=1)
