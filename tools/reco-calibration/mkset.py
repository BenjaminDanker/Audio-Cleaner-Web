"""Builds the calibration material: LibriSpeech clips (scratchpad/libri/stream16k.wav, 16 kHz source, upsampled to 48 kHz),
noise of several kinds at 0-30 dB SNR, band-limited / reverberant versions."""
import os as _os
from pathlib import Path as _Path
# Work folder with libri/ (LibriSpeech stream16k.wav), reco/ (generated material and results) and effiles/. Set RECO_ROOT to move it.
ROOT = _os.environ.get('RECO_ROOT', str(_Path.home() / 'audio-cleaner' / 'research' / 'reco-calibration'))
REPO = str(_Path(__file__).resolve().parents[2])
import json, subprocess, sys
from pathlib import Path
import numpy as np, soundfile as sf
from scipy.signal import resample_poly, butter, sosfilt, fftconvolve

S = Path(ROOT + '')
OUT = S / 'reco'
FF = _os.path.expanduser('~/audio-cleaner/bin/ffmpeg')
SR = 48000
CLIP = 8.0
rng = np.random.default_rng(20261017)


def clips():
    x, sr = sf.read(S / 'libri/stream16k.wav', dtype='float32')
    x = resample_poly(x, 3, 1).astype(np.float32)
    n = int(CLIP * SR)
    out = []
    pos = 0
    while pos + n + SR < len(x):
        # cut at the quietest 50 ms within 0.8 s of the nominal boundary
        hi = pos + n
        seg = x[hi - int(0.8 * SR):hi + int(0.8 * SR)]
        w = int(0.05 * SR)
        e = np.convolve(seg ** 2, np.ones(w) / w, mode='valid')
        cut = hi - int(0.8 * SR) + int(np.argmin(e)) + w // 2
        out.append(x[pos:cut])
        pos = cut
    return out


def pink(n):
    f = np.fft.rfftfreq(n, 1 / SR)
    spec = (rng.standard_normal(len(f)) + 1j * rng.standard_normal(len(f))) / np.sqrt(np.maximum(f, 20.0))
    return np.fft.irfft(spec, n)


def brown(n):
    f = np.fft.rfftfreq(n, 1 / SR)
    spec = (rng.standard_normal(len(f)) + 1j * rng.standard_normal(len(f))) / np.maximum(f, 20.0)
    return np.fft.irfft(spec, n)


def babble(n, pool, k):
    y = np.zeros(n)
    for _ in range(k):
        c = pool[rng.integers(len(pool))]
        c = np.tile(c, 1 + n // len(c))
        o = rng.integers(len(c) - n)
        y += c[o:o + n]
    return y


def clatter(n):
    y = np.zeros(n)
    for _ in range(int(n / SR * 2)):
        o = rng.integers(n - 4000)
        burst = rng.standard_normal(4000) * np.exp(-np.arange(4000) / rng.uniform(150, 900))
        y[o:o + 4000] += burst * rng.uniform(0.3, 1.0)
    return y


def noise_of(kind, n, pool):
    if kind == 'white':
        return rng.standard_normal(n)
    if kind == 'pink':
        return pink(n)
    if kind == 'babble':
        return babble(n, pool, 5)
    if kind == 'traffic':
        t = np.arange(n) / SR
        return brown(n) * (1 + 0.5 * np.sin(2 * np.pi * rng.uniform(0.1, 0.4) * t + rng.uniform(0, 6)))
    if kind == 'cafe':
        b = babble(n, pool, 3)
        return b / np.std(b) + 0.8 * pink(n) / np.std(pink(n)) + 0.6 * clatter(n) / (np.std(clatter(n)) + 1e-9)
    if kind == 'hum':
        t = np.arange(n) / SR
        h = sum(np.sin(2 * np.pi * 50 * k * t + rng.uniform(0, 6)) / k for k in range(1, 8))
        return h / np.std(h) + 0.15 * rng.standard_normal(n)
    if kind == 'keys':
        return clatter(n) / (np.std(clatter(n)) + 1e-9) + 0.05 * pink(n) / np.std(pink(n))
    raise ValueError(kind)


KINDS = ['white', 'pink', 'babble', 'traffic', 'cafe', 'hum', 'keys']
SNRS = [0, 5, 10, 15, 20, 25, 30]


def mix(clean, noise, snr):
    noise = noise[:len(clean)]
    g = np.sqrt(np.mean(clean ** 2) / np.mean(noise ** 2)) * 10 ** (-snr / 20)
    y = clean + g * noise
    return y


def save(path, y):
    y = np.asarray(y, dtype=np.float32)
    pk = np.max(np.abs(y))
    if pk > 0.95:
        y = y * (0.95 / pk)
    sf.write(path, y, SR, subtype='FLOAT')
    return y


def main():
    cl = clips()
    print(len(cl), 'clips', file=sys.stderr)
    (OUT / 'clean').mkdir(exist_ok=True)
    (OUT / 'set').mkdir(exist_ok=True)
    meta = []
    # keep the clean clips at a sensible level (about -23 dBFS RMS) so the mixes are comparable
    cl = [c * (0.07 / np.sqrt(np.mean(c ** 2))) for c in cl]
    for i, c in enumerate(cl):
        sf.write(OUT / f'clean/c{i:03d}.wav', c.astype(np.float32), SR, subtype='FLOAT')
    n_noisy = 40  # clips used with noise (the rest stay as extra clean / band-limited material)
    for i in range(n_noisy):
        c = cl[i].astype(np.float64)
        pool = [x.astype(np.float64) for j, x in enumerate(cl) if j != i]
        for kind in KINDS:
            for snr in SNRS:
                if kind in ('hum', 'keys') and snr in (0, 5):
                    continue
                if rng.random() > (0.5 if kind in ('white', 'pink', 'traffic') else 0.6) and i >= 20:
                    continue  # thin out: the first 20 clips get every combination, the rest about half
                y = mix(c, noise_of(kind, len(c), pool), snr)
                name = f'n_{kind}_{snr:02d}_c{i:03d}'
                save(OUT / f'set/{name}.wav', y)
                meta.append(dict(name=name, clip=i, kind=kind, snr=snr, clean=f'clean/c{i:03d}.wav', cat='noise'))
    # clean clips, a few with very low noise
    for i in range(len(cl)):
        meta.append(dict(name=f'clean_c{i:03d}', clip=i, kind='clean', snr=99, clean=f'clean/c{i:03d}.wav', cat='clean'))
        p = OUT / f'set/clean_c{i:03d}.wav'
        sf.write(p, cl[i].astype(np.float32), SR, subtype='FLOAT')
    json.dump(meta, open(OUT / 'meta.json', 'w'))


if __name__ == '__main__':
    main()
