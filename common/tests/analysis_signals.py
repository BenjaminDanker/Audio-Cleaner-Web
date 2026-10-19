"""Made-up recordings for the tests of analysis.py and reco_model.py: the same samples on every machine (only integer bit
generator output, sines and filters), written as 16-bit WAV files when the website's own measuring code is run on them
(make_reco_fixtures.py, test_analysis.py). Words with pauses (a glottal pulse train and some hiss), in a quiet room, in an
echoey one, behind a gentle low-pass, over noise, through a phone's band, clipped, gated, in stereo."""
from __future__ import annotations

import wave
from pathlib import Path

import numpy as np
import scipy.signal as ss

SR = 48000


def noise(n: int, seed: int) -> np.ndarray:
    """Unit-variance noise, from the raw output of numpy's bit generator (which numpy keeps the same between versions)."""
    raw = np.random.PCG64(seed).random_raw(4 * n).reshape(4, n) >> np.uint64(11)
    return ((raw.astype(np.float64) / 2.0**53).sum(axis=0) - 2.0) / np.sqrt(4 / 12)


def talk(seconds: float, seed: int = 0) -> np.ndarray:
    """Words of 0.2-0.5 s with 0.3-0.8 s pauses: a pulse train at a gliding ~120 Hz (harmonics to the top) and hiss, switched
    with 10 ms ramps, over a quiet room (-70 dB)."""
    n = int(seconds * SR)
    u = noise(2 * n, seed + 1000)  # u[:n] is the hiss, u[n:] the room; the lengths of the words come from the bit generator too
    f0 = 120 * (1 + 0.15 * np.sin(2 * np.pi * 0.4 * np.arange(n) / SR + seed))
    pulses = np.zeros(n)
    pulses[np.nonzero(np.diff(np.floor(np.cumsum(f0) / SR)))[0]] = 1.0
    voiced = ss.sosfilt(ss.butter(2, 900, fs=SR, output="sos"), pulses)
    hiss = ss.sosfilt(ss.butter(2, 3000, "high", fs=SR, output="sos"), u[:n])
    v = voiced / np.std(voiced) + 0.6 * hiss
    lengths = np.random.PCG64(seed + 77).random_raw(4096) / 2.0**64  # uniform 0..1
    env = np.zeros(n)
    ramp = int(0.01 * SR)
    pos = int(0.2 * SR)
    k = 0
    while pos + SR // 2 < n:
        w = int((0.2 + 0.3 * lengths[k]) * SR)
        env[pos:pos + w] = np.minimum(1, np.minimum(np.arange(w), np.arange(w)[::-1]) / ramp)[:max(0, min(w, n - pos))]
        pos += w + int((0.3 + 0.5 * lengths[k + 1]) * SR)
        k += 2
    x = v * env
    return 0.3 * x / np.max(np.abs(x)) + 3e-4 * u[n:]


def room(x: np.ndarray, rt60: float, seed: int = 0) -> np.ndarray:
    """In a room that rings for `rt60` seconds (-60 dB): a decaying noise impulse response."""
    m = int(rt60 * 1.5 * SR)
    h = noise(m, seed + 50) * 10 ** (-3 * np.arange(m) / SR / rt60)
    h[0] = 1.0
    y = ss.fftconvolve(x, h)[:len(x)]
    return 0.3 * y / np.max(np.abs(y)) + 3e-4 * noise(len(y), seed + 60)


def over_noise(x: np.ndarray, snr_db: float, seed: int, pink: bool = False) -> np.ndarray:
    """`x` with noise at this signal-to-noise ratio (pink: more low than high), at a peak of 0.5."""
    n = noise(len(x), seed + 100)
    if pink:
        f = np.fft.rfftfreq(len(x), 1 / SR)
        n = np.fft.irfft(np.fft.rfft(n) / np.sqrt(np.maximum(f, 30.0)), len(x))
    n *= np.sqrt(np.mean(x ** 2) / np.mean(n ** 2)) * 10 ** (-snr_db / 20)
    y = x + n
    return y * (0.5 / np.max(np.abs(y)))


def quantised(x: np.ndarray) -> np.ndarray:
    """What a 16-bit WAV file holds of it, back as float32 (samples, channels)."""
    x = np.asarray(x, dtype=np.float64)
    x = x[:, None] if x.ndim == 1 else x
    return (np.clip(np.round(x * 32767), -32768, 32767).astype("<i2").astype(np.float32) / 32768.0)


def signals() -> dict[str, np.ndarray]:
    """name -> float64 samples, (n,) or (n, 2)."""
    close = talk(8, 0)
    gated = over_noise(talk(8, 1), 45, 3)
    gated[np.abs(gated) < 0.004] = 0.0  # a gated recording: digital silence between the words
    t = np.arange(8 * SR) / SR
    long = np.concatenate([over_noise(talk(25, 10 + k), 30 - 6 * k, 20 + k, pink=bool(k % 2)) for k in range(4)])
    out = {
        "close": close,
        "echoey": room(close, 0.8, 0),
        "muffled": ss.sosfilt(ss.butter(2, 2000, fs=SR, output="sos"), close),
        "phone": ss.sosfilt(ss.butter(14, 3400, fs=SR, output="sos"), over_noise(talk(8, 2), 45, 4)),
        "noisy": over_noise(talk(8, 3), 3, 5, pink=True),
        "clipped": np.clip(6 * talk(6, 4), -1, 1),
        "gated": gated,
        "stereo": np.stack([over_noise(talk(8, 5), 25, 6), over_noise(talk(8, 6), 20, 7, pink=True)], axis=1),
        "dual_mono": np.stack([close, close], axis=1),
        "hum": 0.2 * np.sin(2 * np.pi * 50 * t) + 0.05 * np.sin(2 * np.pi * 150 * t) + 0.002 * noise(len(t), 9),
        "silence": np.zeros(5 * SR),
        "short": talk(2.5, 7)[:int(1.2 * SR)],
        "long": long,  # 100 s: measured on six stretches, like a long file
        "long_stereo": np.stack([long[:80 * SR], over_noise(talk(80, 30), 12, 31)], axis=1),
    }
    return {k: quantised(v)[:, 0] if np.ndim(v) == 1 else quantised(v) for k, v in out.items()}


def write_wav(path: Path, x: np.ndarray) -> None:
    """16-bit PCM WAV of float32 samples that already are what such a file holds (see `quantised`)."""
    x = x[:, None] if x.ndim == 1 else x
    q = np.round(x * 32768.0).astype("<i2")
    with wave.open(str(path), "wb") as w:
        w.setnchannels(q.shape[1])
        w.setsampwidth(2)
        w.setframerate(SR)
        w.writeframes(q.tobytes())
