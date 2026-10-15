"""Listening samples for the owner (192 kbps mp3, 48 kHz mono).

  flock ~/audio-cleaner/dev/gpu.lock python -I make_samples.py [--nfe 16]

Writes ~/audio-cleaner/research/samples/<clip>/{noisy,dfn3_clarity,...}.mp3 and samples/index.json.
Studio candidates are run through the same clarity chain as production (level match, HPF, compressor, -16 LUFS,
limiter); live candidates are raw model output gain-matched to -16 LUFS integrated (no other processing).
The CC BY-NC DeepFilterNet clip is named internal_* and flagged unpublishable.
"""
import argparse, json, os, subprocess, sys, tempfile, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent)); sys.dont_write_bytecode = True
os.environ.setdefault("TQDM_DISABLE", "1")
import numpy as np
import soundfile as sf
from scipy.signal import fftconvolve
from common import ASSETS, DNSMOS, SAMPLES, SR, VB_CLEAN, VB_NOISY, mix_snr, read_mono, resample, si_sdr, subset_ids, write_wav
import hybrid, models

FFMPEG = str(Path.home() / "audio-cleaner" / "bin" / "ffmpeg")
sys.path.insert(0, "/home/silver/repos/Audio-Cleaner-Web/server")
from acw.dsp import clarity


def pick_clips():
    ids = subset_ids(200)
    rows = []
    for i in ids:
        n, c = read_mono(VB_NOISY / f"{i}.wav"), read_mono(VB_CLEAN / f"{i}.wav")
        d = len(n) / SR
        if 3.0 <= d <= 7.0:
            rows.append((si_sdr(c, n), i, d))
    rows.sort()
    out = {}
    for spk in ("p232", "p257"):
        r = [x for x in rows if x[1].startswith(spk)]
        out[f"vb_{r[0][1]}"] = r[0][1]            # hardest of this speaker
        out[f"vb_{r[len(r)//4][1]}"] = r[len(r)//4][1]  # moderately hard
    return out


def reverb_clip(rng):
    ids = [i for i in subset_ids(200) if i.startswith("p257")]
    uid = ids[len(ids) // 2]
    clean = read_mono(VB_CLEAN / f"{uid}.wav")
    rir = read_mono(ASSETS / "rir_sim_1001_w11.7_l2.6_h2.5_rt60_0.7919.wav")
    rir = rir / np.abs(rir).max()
    rev = fftconvolve(clean, rir)[: len(clean)].astype(np.float32)
    rev *= np.sqrt(np.mean(clean ** 2) / np.mean(rev ** 2))
    noise = read_mono(ASSETS / "noise_freesound_573577.wav")  # CC0 1.0
    return uid, mix_snr(rev, noise, 8.0, rng), clean


def lufs_match(x, target=-16.0):
    import pyloudnorm as pyln
    m = pyln.Meter(SR)
    try:
        l = m.integrated_loudness(x.astype(np.float64)[:, None])
    except ValueError:
        return x
    if not np.isfinite(l):
        return x
    y = x * 10 ** ((target - l) / 20)
    return clarity.limit(y.astype(np.float32), SR, -1.0, 5.0, 60.0)


def mp3(x, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory() as td:
        w = Path(td) / "a.wav"
        write_wav(w, x, SR)
        subprocess.run([FFMPEG, "-y", "-loglevel", "error", "-i", str(w), "-codec:a", "libmp3lame", "-b:a", "192k", "-ar", "48000", "-ac", "1", str(path)], check=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--nfe", type=int, default=16)
    a = ap.parse_args()
    rng = np.random.default_rng(7)
    clips = {}
    for name, uid in pick_clips().items():
        clips[name] = {"noisy": read_mono(VB_NOISY / f"{uid}.wav"), "src": f"VoiceBank-DEMAND test {uid} (CC BY 4.0)", "pub": True}
    uid, rv, _ = reverb_clip(rng)
    clips[f"reverb_{uid}"] = {"noisy": rv, "src": f"VoiceBank-DEMAND clean {uid} convolved with DeepFilterNet rir_sim_1001 (RT60 0.79 s) + CC0 freesound 573577 noise at 8 dB SNR", "pub": True}
    clips["internal_noisy_snr0"] = {"noisy": read_mono(ASSETS / "noisy_snr0.wav"), "src": "DeepFilterNet assets/noisy_snr0.wav (derived from freesound 33711, CC BY-NC 3.0): INTERNAL LISTENING ONLY, never publish", "pub": False}

    variants = {}  # name -> (callable, kind, description)
    t0 = time.time()
    studio = [("dfn3_clarity", "dfn3", "Production chain: DeepFilterNet3 (libDF) + clarity chain"),
              ("mossformer2_se_48k_clarity", "mf2_48k", "ClearerVoice MossFormer2_SE_48K + clarity chain"),
              ("mossformergan_se_16k_clarity", "mfgan16_w6", "ClearerVoice MossFormerGAN_SE_16K as-is (16 kHz, band-limited to 8 kHz) + clarity chain"),
              ("hybrid_gan_dfn3_clarity", None, "MossFormerGAN 0-7.5 kHz + DFN3 above (linear-phase crossover) + clarity chain"),
              ("resemble_denoise_clarity", "resemble_denoise", "Resemble Enhance denoiser only (44.1 kHz) + clarity chain"),
              (f"resemble_enhance_nfe{a.nfe}_clarity", f"resemble_enhance_nfe{a.nfe}", f"Resemble Enhance denoiser + enhancer (NFE {a.nfe}) + clarity chain")]
    live = [("live_dfn3", "dfn3_live", "Live: DeepFilterNet3 (30 ms)"), ("live_dfn3_ll", "dfn3_ll", "Live: DeepFilterNet3-LL (10 ms)"),
            ("live_dpdfnet2_48khz_hr", "dpdfnet2_48khz_hr", "Live: DPDFNet2 48 kHz HR"), ("live_gtcrn_dns3", "gtcrn_dns3", "Live: GTCRN (16 kHz, band-limited)")]
    results = {c: {"noisy": (v["noisy"], "Unprocessed input")} for c, v in clips.items()}
    gan_raw = {}
    for fname, mname, desc in studio + live:
        is_live = fname.startswith("live_")
        if mname is None:
            continue
        m = models.make(mname, "cuda" if not is_live else "cpu", 2)
        for c, v in clips.items():
            x = v["noisy"]
            if mname.startswith("mfgan16"):
                y16 = m.enhance16(resample(x, SR, 16000)); gan_raw[c] = y16; y = resample(y16, 16000, SR)[: len(x)]
            else:
                y = m(x)
            y = lufs_match(y) if is_live else clarity.process(x[:, None], y[:, None], SR)[:, 0]
            results[c][fname] = (y, desc)
        del m
        print(f"{fname} done at {time.time()-t0:.0f}s", flush=True)
        if mname == "dfn3":
            dfn_raw = {c: None for c in clips}
    # hybrid needs raw DFN3 + raw GAN
    m = models.make("dfn3", "cpu", 2)
    for c, v in clips.items():
        y = hybrid.hybrid(gan_raw[c], m(v["noisy"]))
        results[c]["hybrid_gan_dfn3_clarity"] = (clarity.process(v["noisy"][:, None], y[:, None], SR)[:, 0], studio[3][2])
    index = {"generated": time.strftime("%Y-%m-%d %H:%M:%S"), "format": "mp3 192 kbps CBR, 48 kHz mono",
             "notes": ["*_clarity = model output through the production clarity chain (level match, 80 Hz HPF, compressor, -16 LUFS, limiter)",
                       "live_* = raw causal-model output, gain-matched to -16 LUFS integrated with a -1 dBFS limiter, nothing else",
                       "files named internal_* must never leave this machine (CC BY-NC source audio)"], "clips": {}}
    dn = DNSMOS()
    index["dnsmos_note"] = "DNSMOS P.835 (non-personalized) on the delivered file (post clarity chain / gain match). Internal metric only."
    for c, d in results.items():
        files = {}
        for fname, (y, desc) in d.items():
            mp3(y, SAMPLES / c / f"{fname}.mp3")
            files[f"{fname}.mp3"] = {"what": desc, "dnsmos": {k: round(v, 3) for k, v in dn(y).items()}}
        index["clips"][c] = {"source": clips[c]["src"], "duration_s": round(len(clips[c]["noisy"]) / SR, 2), "publishable": clips[c]["pub"], "files": files}
    (SAMPLES / "index.json").write_text(json.dumps(index, indent=1))
    print("samples written:", SAMPLES)


if __name__ == "__main__":
    main()
