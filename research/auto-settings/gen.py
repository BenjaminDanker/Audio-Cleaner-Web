"""Simulated degradations of clean speech (LibriTTS-R, CC BY 4.0). Every clip records its ground truth.
usage: gen.py <split> <count> <seed> <speech_root_list> ; writes clips/<split>/<id>.wav (48 kHz PCM16, what the browser gets),
clips/<split>/<id>.ref16.npy (int16 16 kHz clean reference for the scorer, the same chain without the noise) and clips/<split>/<id>.json.
"""
import json, os, random, subprocess, sys, glob
os.environ.setdefault("OMP_NUM_THREADS", "1")
import numpy as np
import soundfile as sf
from scipy import signal

ROOT = os.environ.get("AC_ROOT", os.path.expanduser("~/audio-cleaner/research/auto-settings"))
FF = subprocess.run([sys.executable, "-c", "import imageio_ffmpeg as i;print(i.get_ffmpeg_exe())"],
                    capture_output=True, text=True).stdout.strip()
FS = 48000
SPLIT_SPEAKERS = {"train": "dev", "val": "dev", "test": "test"}  # dev-clean speakers -> train/val, test-clean -> test

def speakers(sub):
    """speaker dirs of a LibriTTS-R subset (dev or test), sorted for reproducibility"""
    base = os.path.join(ROOT, "libri", "LibriTTS_R", f"{sub}-clean")
    return sorted(glob.glob(os.path.join(base, "*")))

def utter_list(spk_dirs):
    out = {}
    for d in spk_dirs:
        files = sorted(glob.glob(os.path.join(d, "*", "*.wav")))
        if files:
            out[os.path.basename(d)] = files
    return out

def load48(path):
    x, fs = sf.read(path, dtype="float32")
    if x.ndim > 1:
        x = x.mean(1)
    if fs != FS:
        x = signal.resample_poly(x, FS, fs).astype(np.float32)
    return x

def active_rms(x):
    fr = x[: len(x) // 480 * 480].reshape(-1, 480)
    e = 10 * np.log10(np.mean(fr ** 2, 1) + 1e-12)
    m = e > e.max() - 35
    return float(np.sqrt(np.mean(fr[m] ** 2)) + 1e-9)

def speech_clip(rng, spk_files, dur=10.0):
    """one speaker's utterances joined to dur seconds"""
    files = list(spk_files)
    rng.shuffle(files)
    out, n = [], 0
    gap = np.zeros(int(0.2 * FS), np.float32)
    for f in files:
        y = load48(f)
        out += [y, gap]
        n += len(y) + len(gap)
        if n >= dur * FS:
            break
    x = np.concatenate(out)[: int(dur * FS)]
    if len(x) < int(dur * FS):
        x = np.pad(x, (0, int(dur * FS) - len(x)))
    return x

def colored(rng, n, beta):  # rng: numpy Generator
    """noise with power spectrum 1/f^beta (beta 0 white, 1 pink, 2 brown)"""
    w = rng.standard_normal(n)
    W = np.fft.rfft(w)
    f = np.fft.rfftfreq(n, 1 / FS)
    f[0] = f[1]
    W = W / f ** (beta / 2)
    y = np.fft.irfft(W, n)
    return (y / (np.std(y) + 1e-12)).astype(np.float32)

def make_noise(rng, nprng, fam, n, babble_pool):
    """rng: random.Random (choices, babble), nprng: numpy Generator (signal noise)"""
    if fam == "white": y = colored(nprng, n, 0)
    elif fam == "pink": y = colored(nprng, n, 1)
    elif fam == "brown": y = colored(nprng, n, rng.uniform(1.5, 2.2))
    elif fam == "hum":
        f0 = rng.choice([50, 60]); t = np.arange(n) / FS
        y = sum(np.sin(2 * np.pi * f0 * k * t + rng.uniform(0, 6)) / k for k in range(1, 9)).astype(np.float32)
        y = y / (np.std(y) + 1e-12) + 0.2 * colored(nprng, n, 1)
    elif fam == "fan":
        y = colored(nprng, n, 1.0)
        sos = signal.butter(4, rng.uniform(1200, 2500), "low", fs=FS, output="sos"); y = signal.sosfilt(sos, y)
        t = np.arange(n) / FS; y = y * (1 + 0.3 * np.sin(2 * np.pi * rng.uniform(0.2, 2) * t))
    elif fam == "babble":
        y = sum(speech_clip(rng, s, dur=n / FS) for s in rng.sample(babble_pool, k=rng.randint(3, 6)))
    elif fam == "rumble":  # test only: low traffic rumble with passing bursts
        y = colored(nprng, n, 2.0)
        sos = signal.butter(2, rng.uniform(200, 400), "low", fs=FS, output="sos"); y = signal.sosfilt(sos, y)
        t = np.arange(n) / FS; y = y * (1 + 0.8 * np.sin(2 * np.pi * rng.uniform(0.05, 0.3) * t) ** 2)
    elif fam == "ac":  # test only: air conditioning, harmonics at other base frequencies plus broadband hiss
        f0 = rng.uniform(90, 180); t = np.arange(n) / FS
        y = sum(np.sin(2 * np.pi * f0 * k * t + rng.uniform(0, 6)) / k ** 0.7 for k in range(1, 12)).astype(np.float32)
        y = y / (np.std(y) + 1e-12) + 0.5 * colored(nprng, n, 0.5)
    elif fam == "clicks":  # test only: keyboard and mouse clicks over room tone
        y = 0.1 * colored(nprng, n, 1.5)
        for _ in range(int(rng.uniform(2, 12) * n / FS)):
            p = rng.randrange(0, n - 500); L = rng.randint(80, 400)
            y[p:p + L] += (rng.uniform(0.5, 1.0) * np.exp(-np.arange(L) / 40) * nprng.standard_normal(L)).astype(np.float32)
    else:
        raise ValueError(fam)
    return (y / (np.std(y) + 1e-12)).astype(np.float32)

def make_rir(rng, pra, group):
    """simulated shoebox room; group selects room sizes (train and test rooms are drawn from disjoint size ranges)"""
    lo, hi = (3.0, 7.0) if group == "train" else (6.0, 11.0)
    dims = [rng.uniform(lo, hi), rng.uniform(lo, hi), rng.uniform(2.4, 3.6)]
    rt60 = rng.choice([rng.uniform(0.15, 0.4), rng.uniform(0.4, 0.8), rng.uniform(0.8, 1.2)])
    e, mo = pra.inverse_sabine(rt60, dims)
    room = pra.ShoeBox(dims, fs=FS, materials=pra.Material(e), max_order=min(mo, 14))
    src = [rng.uniform(0.5, dims[0] - 0.5), rng.uniform(0.5, dims[1] - 0.5), rng.uniform(1.2, 1.8)]
    d = rng.uniform(0.6, 2.5)
    sgn = 1 if src[0] + d < dims[0] - 0.3 else -1
    mic = [float(np.clip(src[0] + sgn * d, 0.3, dims[0] - 0.3)), float(np.clip(src[1], 0.3, dims[1] - 0.3)), 1.2]
    room.add_source(src); room.add_microphone(mic)
    room.compute_rir()
    h = np.asarray(room.rir[0][0], np.float32)[: int(1.2 * FS)]
    h = h / (np.max(np.abs(h)) + 1e-12)
    pk = int(np.argmax(np.abs(h)))
    direct = h[max(0, pk - 120): pk + 120]
    e_dir = float(np.sum(direct ** 2))
    rest = np.sum(h[pk + 120:] ** 2) + 1e-12
    drr = 10 * np.log10(e_dir / rest + 1e-12)
    e50 = np.sum(h[: pk + int(0.05 * FS)] ** 2) + 1e-12
    c50 = 10 * np.log10(e50 / (np.sum(h[pk + int(0.05 * FS):] ** 2) + 1e-12) + 1e-12)
    return h, dict(rt60_target=float(rt60), drr_db=float(drr), c50_db=float(c50), dims=[round(v, 2) for v in dims])

def shelf(x, gain_db, fc):
    """RBJ high-shelf (a dull, muffled microphone): gain_db < 0 cuts the top of the band"""
    A = 10 ** (gain_db / 40); w = 2 * np.pi * fc / FS; cw, sw = np.cos(w), np.sin(w)
    alpha = sw / 2 * np.sqrt((A + 1 / A) * (1 / 0.7 - 1) + 2)
    b0 = A * ((A + 1) + (A - 1) * cw + 2 * np.sqrt(A) * alpha)
    b1 = -2 * A * ((A - 1) + (A + 1) * cw)
    b2 = A * ((A + 1) + (A - 1) * cw - 2 * np.sqrt(A) * alpha)
    a0 = (A + 1) - (A - 1) * cw + 2 * np.sqrt(A) * alpha
    a1 = 2 * ((A - 1) - (A + 1) * cw)
    a2 = (A + 1) - (A - 1) * cw - 2 * np.sqrt(A) * alpha
    return signal.lfilter([b0 / a0, b1 / a0, b2 / a0], [1, a1 / a0, a2 / a0], x).astype(np.float32)

def ffmpeg(args):
    subprocess.run([FF, "-hide_banner", "-loglevel", "error", "-nostdin", "-y"] + args, check=True)

def codec(x, kind, kbps, tmp):
    """encode and decode through a real codec (ffmpeg), back to 48 kHz mono float"""
    src, mid, out = f"{tmp}_in.wav", f"{tmp}_mid", f"{tmp}_out.wav"
    sf.write(src, x, FS, subtype="PCM_16")
    if kind == "phone":  # G.711 A-law at 8 kHz
        ffmpeg(["-i", src, "-ar", "8000", "-ac", "1", "-c:a", "pcm_alaw", f"{mid}.wav"])
        ffmpeg(["-i", f"{mid}.wav", "-ar", str(FS), "-ac", "1", out])
    else:
        ext = {"mp3": "mp3", "opus": "opus", "aac": "m4a"}[kind]
        enc = {"mp3": ["-c:a", "libmp3lame"], "opus": ["-c:a", "libopus"], "aac": ["-c:a", "aac"]}[kind]
        ffmpeg(["-i", src, "-ar", "48000" if kind != "mp3" else "44100", "-ac", "1", *enc, "-b:a", f"{kbps}k", f"{mid}.{ext}"])
        ffmpeg(["-i", f"{mid}.{ext}", "-ar", str(FS), "-ac", "1", out])
    y, _ = sf.read(out, dtype="float32")
    for p in [src, out, f"{mid}.wav", f"{mid}.mp3", f"{mid}.opus", f"{mid}.m4a"]:
        if os.path.exists(p): os.remove(p)
    n = len(x)
    return np.pad(y, (0, max(0, n - len(y))))[:n]

def lowpass(x, fc):
    sos = signal.butter(6, fc, "low", fs=FS, output="sos")
    return signal.sosfilt(sos, x).astype(np.float32)

def highpass(x, fc):
    sos = signal.butter(4, fc, "high", fs=FS, output="sos")
    return signal.sosfilt(sos, x).astype(np.float32)

def measured_cut(x):
    """Frequency above which the long-term spectrum falls 40 dB below its level at 2 kHz (an objective bandwidth of the clean path)"""
    f, P = signal.welch(x, FS, nperseg=2048)
    ref = 10 * np.log10(np.mean(P[(f > 1800) & (f < 2200)]) + 1e-20)
    db = 10 * np.log10(P + 1e-20) - ref
    ok = np.where((db > -40) & (f > 1800))[0]
    return float(f[ok[-1]]) if len(ok) else 2000.0

def main():
    import pyroomacoustics as pra
    split, count, seed = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])
    rng = random.Random(seed * 7919 + 1); nprng = np.random.default_rng(seed)
    test = split == "test"
    dev_spk = utter_list(speakers("dev")); test_spk = utter_list(speakers("test"))
    pool = test_spk if test else dev_spk
    names = sorted(pool)
    random.Random(12345).shuffle(names)  # the speaker split is the same for every seed
    nval = max(1, len(names) // 4)
    if split == "val": names = names[:nval]
    elif split == "train": names = names[nval:]
    else: names = names
    # babble: speech of the same split's other speakers (train babble never uses validation or test speakers)
    babble_pool = [pool[k] for k in names] if len(names) >= 4 else [pool[k] for k in sorted(pool)]
    out = os.path.join(ROOT, "clips", split); os.makedirs(out, exist_ok=True)
    tmp = os.path.join(ROOT, "work", "tmp", f"gen_{split}_{seed}")
    noise_fams = ["rumble", "ac", "clicks", "babble"] if test else ["white", "pink", "brown", "hum", "fan", "babble"]
    for i in range(count):
        try:
            cid = f"{split}_{seed}_{i:05d}"
            if os.path.exists(os.path.join(out, cid + ".json")):
                continue
            spk = names[i % len(names)]
            files = pool[spk]
            x = speech_clip(rng, files)
            # speech level
            x = x / (active_rms(x) + 1e-9) * 10 ** (rng.uniform(-35, -18) / 20)
            meta = dict(id=cid, split=split, speaker=spk, speech_files=len(files))
            # 1. muffled (dull) microphone: a high-shelf cut, clean otherwise
            dull = rng.random() < 0.3
            if dull:
                g = rng.uniform(-14, -4); fc = rng.uniform(1500, 4000)
                x = shelf(x, g, fc)
                meta.update(dull_db=round(g, 2), dull_fc=round(fc, 1))
            else:
                meta.update(dull_db=0.0, dull_fc=None)
            # 2. room
            try:
                rir = make_rir(rng, pra, "test" if test else "train") if rng.random() < 0.7 else None
            except Exception as ex:  # a degenerate room: dry instead
                print("rir failed", cid, ex, flush=True); rir = None
            if rir is not None:
                h, rinfo = rir
                from scipy.signal import fftconvolve
                y = fftconvolve(x, h)[: len(x)].astype(np.float32)
                x = y / (active_rms(y) + 1e-9) * 10 ** (rng.uniform(-35, -18) / 20) * (1 if True else 1)
                meta.update(rinfo, reverb=True)
            else:
                meta.update(rt60_target=0.0, drr_db=None, c50_db=None, dims=None, reverb=False)
            # 3. chain for the reference: band/codec only (no noise)
            kind_r = rng.random()
            if kind_r < 0.45: codec_kind = ("none", 0)
            elif kind_r < 0.6: codec_kind = ("lowpass", rng.choice([4000, 5000, 6000, 7000, 8000, 11000]))
            elif kind_r < 0.7: codec_kind = ("phone", 0)
            elif kind_r < 0.8: codec_kind = ("opus", rng.choice([6, 8, 12, 16, 24]))
            elif kind_r < 0.9: codec_kind = ("mp3", rng.choice([16, 24, 32, 64, 128]))
            else: codec_kind = ("aac", rng.choice([16, 24, 32, 48, 96]))
            def chain(sig):
                k, v = codec_kind
                if k == "lowpass": return lowpass(sig, v)
                if k == "phone": return highpass(lowpass(codec(sig, "phone", 0, tmp), 3400), 300)
                if k in ("opus", "mp3", "aac"): return codec(sig, k, v, tmp)
                return sig
            clean_path = chain(x)
            # 4. noise at a known SNR (none for 25%)
            fam = rng.choice(noise_fams)
            snr = None if rng.random() < 0.25 else rng.uniform(-3, 30)
            y = clean_path.copy()
            if snr is not None:
                nz = make_noise(rng, np.random.default_rng(rng.randrange(1 << 30)), fam, len(x), babble_pool)
                sp = active_rms(clean_path)
                nz = nz / (active_rms(nz) + 1e-9) * sp * 10 ** (-snr / 20)
                # the noise also goes through the codec path (it is part of the recording): add after the band cut
                y = clean_path + nz.astype(np.float32)
            # clipping for some loud recordings
            clip_frac = 0.0
            if rng.random() < 0.12:
                gain = 10 ** (rng.uniform(4, 12) / 20)
                z = y * gain
                clip_frac = float(np.mean(np.abs(z) >= 0.999))
                y = np.clip(z, -1, 1)
            # level
            target = rng.uniform(-38, -10)
            y = y / (np.sqrt(np.mean(y ** 2)) + 1e-9) * 10 ** (target / 20)
            y = np.clip(y, -1, 1).astype(np.float32)
            ref = chain(x)
            ref = ref / (np.sqrt(np.mean(ref ** 2)) + 1e-9) * 10 ** (target / 20)
            ref = np.clip(ref, -1, 1)
            cut = measured_cut(ref)
            # ground-truth labels by construction (see report: label mapping)
            echoey_gt = meta["reverb"] and meta["rt60_target"] >= 0.5
            codec_bad = codec_kind[0] == "opus" and codec_kind[1] <= 8 or (codec_kind[0] == "mp3" and codec_kind[1] <= 16)
            damage = codec_bad or clip_frac >= 0.005
            meta.update(
                noise_family=fam if snr is not None else None, snr_db=None if snr is None else round(snr, 2),
                codec=codec_kind[0], codec_value=codec_kind[1], cut_hz=round(cut, 1), clip_frac_set=clip_frac,
                target_lufs_rms_db=round(target, 2), gt_rebuild=bool(echoey_gt or damage), gt_echoey=bool(echoey_gt),
                gt_restore_highs=bool(cut <= 7000), gt_dull=dull,
            )
            sf.write(os.path.join(out, cid + ".wav"), y, FS, subtype="PCM_16")
            from scipy.signal import resample_poly
            r16 = np.clip(resample_poly(ref, 1, 3), -1, 1)
            np.save(os.path.join(out, cid + ".ref16.npy"), (r16 * 32767).astype(np.int16))
            with open(os.path.join(out, cid + ".json"), "w") as f:
                json.dump(meta, f)
            if i % 50 == 0: print(split, i, flush=True)
        except Exception as ex:  # one bad clip must not stop the run
            print('skipped', i, repr(ex)[:200], flush=True)

if __name__ == "__main__":
    main()
