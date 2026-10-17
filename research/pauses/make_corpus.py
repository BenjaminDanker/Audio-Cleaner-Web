# Evaluation of skipping noisy pauses (docs/research/skip-noisy-pauses-2026-10.md). L = a scratch directory holding
# utts/ (LibriSpeech utterances as 16 kHz wav + refs.json), dfout/libdf.so (the build), item3/ (made by the make_*.py).
"""Test recordings: LibriSpeech utterances with pauses (a good share of them >= 1.5 s) over continuous room noise."""
import json, sys
import numpy as np, soundfile as sf
from scipy.signal import resample_poly, lfilter, butter, sosfilt
L = sys.argv[1]; OUT = L + '/item3'
refs = json.load(open(L + '/utts/refs.json'))
utts = [sf.read(f"{L}/utts/{r['id']}.wav", dtype='float32')[0] for r in refs]
SR = 48000
def rms(x): return float(np.sqrt(np.mean(np.square(x, dtype=np.float64)) + 1e-30))
def db(v): return 20 * np.log10(v + 1e-30)
def pink(n, rng):
    w = rng.standard_normal(n); f = np.fft.rfft(w); f /= np.sqrt(np.maximum(np.arange(len(f)), 1)); return np.fft.irfft(f, n)
def hum(n, rng):
    t = np.arange(n) / SR; x = sum(np.sin(2 * np.pi * 120 * k * t + rng.uniform(0, 6)) / k for k in range(1, 12)); return x + 0.5 * pink(n, rng) * 0.3 * rms(x) / rms(pink(n, rng))
def babble(n, rng):
    out = np.zeros(n)
    for _ in range(6):
        seq = np.concatenate([resample_poly(u, 3, 1)[::-1] for u in rng.permutation(len(utts))[:8] for u in [utts[u]]])
        k = rng.integers(0, max(1, len(seq) - n)) if len(seq) > n else 0
        seq = np.resize(seq, n) if len(seq) < n else seq[k:k + n]
        out += seq
    return out + 0.3 * rms(out) * pink(n, rng) / rms(pink(n, rng))
def clicks(n, rng):
    x = pink(n, rng); x *= 0.5 / rms(x)
    for c in rng.integers(0, n - 4000, n // SR // 3):
        k = int(rng.integers(300, 1500)); x[c:c + k] += rng.standard_normal(k) * np.exp(-np.arange(k) / (k / 5)) * rng.uniform(1, 6) * 0.5
    return x
def hiss(n, rng):
    return lfilter([1], [1, -0.5], rng.standard_normal(n))
def room(n, rng):  # HVAC rumble + hiss + a little pink
    r = sosfilt(butter(2, 200, 'low', fs=SR, output='sos'), rng.standard_normal(n)); r *= 1.0 / rms(r)
    h = hiss(n, rng); h *= 0.6 / rms(h); return r + h + 0.3 * pink(n, rng) / rms(pink(n, rng))
CASES = {  # name: (noise fn, SNR dB during speech, speech level dBFS rms, seed)
    'pink15': (pink, 15, -23, 1), 'hum10': (hum, 10, -23, 2), 'babble5': (babble, 5, -23, 3),
    'quiet20': (pink, 20, -45, 4), 'whisper5': (pink, 5, -50, 5), 'clicks12': (clicks, 12, -23, 6), 'pink0': (pink, 0, -23, 7),
    'hiss50': (hiss, 27, -23, 8), 'hiss60': (hiss, 37, -23, 9), 'hiss40': (hiss, 17, -23, 10), 'room45': (room, 22, -23, 11), 'hiss70q': (hiss, 37, -33, 12),
}
for name, (nf, snr, lvl, seed) in CASES.items():
    if len(sys.argv) > 2 and name not in sys.argv[2:]: continue
    rng = np.random.default_rng(seed)
    order = rng.permutation(len(utts))
    parts, mask = [], []
    def add(x, m): parts.append(x); mask.append(np.full(len(x), m, bool))
    add(np.zeros(int(rng.uniform(0.5, 1.5) * SR), np.float32), False)
    long_total = 0
    for i in order:
        u = resample_poly(utts[i], 3, 1).astype(np.float32)
        # ground-truth speech frames: 10 ms frames within 38 dB of the utterance's loud level (pauses inside a sentence are not speech)
        fr = SR // 100; n = len(u) // fr
        e = 10 * np.log10(np.mean(u[:n * fr].reshape(n, fr) ** 2, axis=1) + 1e-12)
        act = e > np.percentile(e, 95) - 38
        m = np.zeros(len(u), bool); m[:n * fr] = np.repeat(act, fr)
        add(u, m)
        gap = rng.uniform(1.5, 6.0) if rng.random() < 0.4 else rng.uniform(0.25, 1.0)
        if gap >= 1.5: long_total += gap
        add(np.zeros(int(gap * SR), np.float32), False)
    sp = np.concatenate(parts); gt = np.concatenate(mask)
    # level: active speech rms to lvl, then noise to give the SNR
    sp *= 10 ** ((lvl - db(rms(sp[gt]))) / 20)
    nz = nf(len(sp), rng); nz *= 10 ** ((lvl - snr - db(rms(nz))) / 20)
    x = (sp + nz).astype(np.float32)
    assert np.abs(x).max() < 1
    sf.write(f'{OUT}/{name}.wav', x, SR, subtype='FLOAT'); np.save(f'{OUT}/{name}.gt.npy', gt); np.save(f'{OUT}/{name}.clean.npy', sp.astype(np.float32))
    print(name, f'{len(x)/SR:.0f}s speech {gt.mean()*100:.0f}% long-gap time {long_total:.0f}s noise {db(rms(nz)):.1f} dBFS peak {np.abs(x).max():.2f}')
