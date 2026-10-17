"""Degraded versions of the clean clips: band-limited (filters, phone / low-bitrate codecs), reverberant, muffled."""
import os as _os
from pathlib import Path as _Path
# Work folder with libri/ (LibriSpeech stream16k.wav), reco/ (generated material and results) and effiles/. Set RECO_ROOT to move it.
ROOT = _os.environ.get('RECO_ROOT', str(_Path.home() / 'audio-cleaner' / 'research' / 'reco-calibration'))
REPO = str(_Path(__file__).resolve().parents[2])
import json, subprocess
from pathlib import Path
import numpy as np, soundfile as sf
from scipy.signal import fftconvolve, lfilter, butter, sosfilt

S = Path(ROOT + '/reco')
FF = _os.path.expanduser('~/audio-cleaner/bin/ffmpeg')
SR = 48000
rng = np.random.default_rng(7)
(S / 'deg').mkdir(exist_ok=True)
meta = []

def ff(args):
    r = subprocess.run([FF, '-hide_banner', '-loglevel', 'error', '-y'] + args, capture_output=True, text=True)
    if r.returncode: raise RuntimeError(r.stderr)

def rir(rt60, n=None):
    n = n or int(min(2.5, rt60 * 1.6) * SR)
    t = np.arange(n) / SR
    h = rng.standard_normal(n) * np.exp(-6.908 * t / rt60)
    h[0] = 0
    h = h / np.sqrt(np.sum(h ** 2))
    h[0] = 1.0            # direct path
    return h

for i in range(0, 40, 1):
    cp = S / f'clean/c{i:03d}.wav'
    x, _ = sf.read(cp, dtype='float32')
    for fc in (3400, 4000, 5500, 7000, 9500):
        sos = butter(14, fc, fs=SR, output='sos')
        y = sosfilt(sos, x).astype(np.float32)
        sf.write(S / f'deg/bl_{fc}_c{i:03d}.wav', y, SR, subtype='FLOAT')
        meta.append(dict(name=f'bl_{fc}_c{i:03d}', clip=i, cat='bandlimit', cut=fc))
    for tag, enc in (('amr', ['-ar', '8000', '-c:a', 'libopencore_amrnb', '-b:a', '12.2k', '-f', 'amr']),
                     ('g722', ['-ar', '16000', '-c:a', 'g722', '-f', 'wav']),
                     ('mp3_32', ['-c:a', 'libmp3lame', '-b:a', '32k', '-f', 'mp3']),
                     ('opus_12', ['-c:a', 'libopus', '-b:a', '12k', '-f', 'ogg'])):
        if i >= 20 and tag != 'amr': continue
        tmp = S / f'deg/tmp_{tag}'
        try:
            ff(['-i', str(cp)] + enc + [str(tmp)])
            ff(['-i', str(tmp), '-ar', '48000', '-ac', '1', '-c:a', 'pcm_f32le', str(S / f'deg/codec_{tag}_c{i:03d}.wav')])
            meta.append(dict(name=f'codec_{tag}_c{i:03d}', clip=i, cat='codec', tag=tag))
        except RuntimeError as e:
            print('skip', tag, str(e)[:100])
    for rt in (0.15, 0.3, 0.5, 0.7, 1.0, 1.4):
        y = fftconvolve(x, rir(rt))[:len(x)].astype(np.float32)
        y *= np.sqrt(np.mean(x ** 2) / np.mean(y ** 2))
        sf.write(S / f'deg/rev_{int(rt*100):03d}_c{i:03d}.wav', y, SR, subtype='FLOAT')
        meta.append(dict(name=f'rev_{int(rt*100):03d}_c{i:03d}', clip=i, cat='reverb', rt60=rt))
    for fc in (1200, 2000, 3000):
        a = np.exp(-2 * np.pi * fc / SR)
        y = lfilter([1 - a], [1, -a], x).astype(np.float32)
        y *= np.sqrt(np.mean(x ** 2) / np.mean(y ** 2))
        sf.write(S / f'deg/muf_{fc}_c{i:03d}.wav', y, SR, subtype='FLOAT')
        meta.append(dict(name=f'muf_{fc}_c{i:03d}', clip=i, cat='muffled', cut=fc))
json.dump(meta, open(S / 'meta_deg.json', 'w'))
print(len(meta))
