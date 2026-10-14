# Speech denoise / enhancement benchmark (2026-10-13 to 10-14)

Measured on primary-ram (Ryzen 7 1700X, 31 GB, GTX 960 4 GB, driver 580, Ubuntu 24.04, Python 3.12.3).
Candidates and licences come from [denoise-candidates.md](denoise-candidates.md). Scripts and pins are in
`research/denoise/`. Nothing here was committed to the product code.

## 1. Bottom line

| Question | Answer |
|---|---|
| (a) Studio clean file tier | **Keep DeepFilterNet3 + clarity chain as the default.** No candidate beats it clearly. The only measurable upgrade is a hybrid (MossFormerGAN 0-7.5 kHz + DFN3 above): +0.17 PESQ and +0.04 DNSMOS OVRL over the production chain on VoiceBank-DEMAND, but it is not ahead on the 6 real-world-style clips, needs the whole GPU for about 1.0x real time, and its weights are 16 kHz. Offer it only as an optional GPU "Studio+" tier, and only if the owner prefers it in a blind A/B on `~/audio-cleaner/research/samples/`. |
| (b) Live medium / high (<= ~40 ms) | **Medium = DFN3 (30 ms, 0.107 core per stream). High / low-latency = DFN3-LL (10 ms, 0.28 core per stream), same quality.** Nothing else qualifies: DPDFNet measured 50 ms stream lag and is not better than DFN3 on this data, GTCRN is 16 kHz-only and clearly worse (OVRL -0.28). GTCRN is usable only as an economy tier (7 % of a core). |
| (c) Does the GPU help? | **Not for the product as it stands.** DFN3 already runs at RTF 0.046 to 0.103 on CPU. The GPU only makes heavier models feasible (MF2 0.05, Resemble denoiser 0.05, MossFormerGAN 0.98), and those do not improve quality enough. Live models gain nothing from it (one 10 ms frame per call). The 4 GB card is shared with ASR through `gpu.lock`, and MossFormerGAN needs 2.2 GB (6 s windows) to 3.6 GB (10 s). |

Limits of this report: nobody listened to the samples yet. The "observations" in section 5 come from DNSMOS and
spectral metrics, not from ears. The 6 sample clips are too few for statistics. Metrics are in-domain-ish for
some models (see section 8).

## 2. Setup that works on sm_52

- venv `~/audio-cleaner/research/se-venv` (deleted after the run; rebuild with `research/denoise/setup_env.sh`, about 25 min, wheels are in the pip cache).
- **`torch==2.11.0+cu126`, `torchaudio==2.11.0+cu126`** from `https://download.pytorch.org/whl/cu126`. `torch.cuda.get_arch_list()` is `sm_50 sm_60 sm_70 sm_75 sm_80 sm_86 sm_90`; the sm_50 kernels run on the sm_52 card (matmul, conv1d, attention all verified). torch 2.12 to 2.14 cu126 wheels exist but have no matching torchaudio (stops at 2.11). cu128 and later were not tested (upstream dropped the oldest archs there).
- CUDA libs come as pip wheels: `nvidia-cublas-cu12==12.6.4.1`, `nvidia-cudnn-cu12==9.10.2.21`, `nvidia-cufft-cu12==11.3.0.4`, `nvidia-cuda-runtime-cu12==12.6.77`, `nvidia-cuda-nvrtc-cu12==12.6.85`, `nvidia-nvjitlink-cu12==12.6.85` (full list in `~/audio-cleaner/research/keep/denoise/pip-freeze.txt`). `triton==3.6.0` is pulled in but unused.
- Other pins: numpy 2.5.4, scipy 1.18.1, onnxruntime 1.31.0 (CPU), soundfile 0.14.0, librosa 1.0.0, pesq 0.0.4 (builds from source, needs gcc), pystoi 0.4.1, pyloudnorm 0.2.0, einops 0.8.2, rotary-embedding-torch 0.9.1, yamlargparse 1.31.1, omegaconf 2.4.0.
- Fixes needed: Resemble Enhance imports `deepspeed` even for inference (stubbed in `models.py`, deepspeed not installed); its CFM solver breaks on NumPy >= 2.5 (`float()` of a 1-element array; patched at runtime, no repo edit); its chunker can leave a final chunk of a few hundred samples that `torch.stft` rejects (zero tail added and trimmed).
- All checkpoints (MossFormer2, MossFormerGAN, Resemble) load with `torch.load(weights_only=True)`. No full unpickling was needed. DPDFNet used ONNX files; GTCRN's torch checkpoints also loaded with `weights_only=True`. Resemble's `hparams.yaml` has a `!!python/object/apply:pathlib.PosixPath` tag (only that tag; checked with grep); it was parsed by OmegaConf.
- Third-party downloads each went into their own new directory, were read only by `python -I`, and large files used `curl --limit-rate 40M`.

## 3. Method

- **Data:** VoiceBank-DEMAND test set, 824 pairs, 48 kHz, from Edinburgh DataShare 10283/2791 (test files only). The record's licence text is **CC BY 4.0** (confirmed from the bitstream). Fixed random subset of **200 pairs** (seed 1234, `research/denoise/subset200.json`, 523 s of audio, clips 1.2 to 8.5 s). Full set not run.
- **Metrics:** PESQ-WB and STOI/ESTOI on 16 kHz resampled audio, SI-SDR at 48 kHz (16 kHz for 16 kHz-output models), LSD>8k = log-spectral distance above 8 kHz on active frames (sees what 16 kHz metrics cannot), DNSMOS P.835 (SIG/BAK/OVRL, DNS-Challenge `sig_bak_ovr.onnx`, non-personalized). DNSMOS and PESQ are for internal evaluation only.
- **Pipeline check:** the noisy input scores PESQ 1.90 / STOI 0.921 (literature 1.97 / 0.921) and GTCRN's VCTK-trained checkpoint scores PESQ 2.83 / STOI 0.941 (paper 2.87 / 0.940), so the metric chain is sane.
- **Baseline:** `acw.dsp.dfn.denoise_array` (libDF, DeepFilterNet3) then `acw.dsp.clarity.process`. Studio candidates are reported raw and after the same clarity chain. The clarity chain compresses and loudness-normalizes, so SI-SDR is meaningless after it (about 5.8 for every system); use PESQ/STOI/DNSMOS there.
- **Speed:** 10-minute file = 236 VoiceBank noisy utterances (every 3rd of the sorted list, both speakers and all noise types). Slow models used a shorter excerpt of the same file (noted) and the RTF is linear (all adapters are chunked). **The box is shared: the ASR benchmark was running, so every CPU RTF carries a load average (column in the speed table). Treat torch CPU numbers as upper bounds.** GPU runs held `gpu.lock`.
- **Live cost:** frame by frame, one thread pinned to one core, 3 x 3000 frames (30 s) per model; repeat means agree within 1 %. Latency = measured input-to-output lag on clean speech with a hop-by-hop host (see 6.2).
- **Real-world-style clips (6):** 4 VoiceBank noisy utterances (hardest and moderately hard per speaker), 1 reverberant clip (VoiceBank clean convolved with `rir_sim_1001` RT60 0.79 s plus CC0 freesound 573577 noise at 8 dB), and `noisy_snr0.wav` (CC BY-NC, internal only, named `internal_*`).

## 4. Results

### 4.1 Master table

CPU RTF = CPU seconds per audio second at the stated threads (lower is better; the 10-minute-file requirement is <= 1.0). GPU RTF is on the GTX 960 in FP32. Live models show CPU cost as core fraction per stream instead.
Scores are on the 200-clip subset, **raw model output**.

| Model | Native rate | CPU RTF (threads) | GPU RTF | VRAM | PESQ / STOI / SI-SDR | DNSMOS SIG / BAK / OVRL | Licence: code / weights; commercial use |
|---|---|---|---|---|---|---|---|
| Noisy input | 48k | - | - | - | 1.90 / 0.921 / 8.4 | 3.32 / 3.09 / 2.68 | - |
| **DFN3 (libDF), production** | 48k | **0.103 (1), 0.049 (4), 0.046 (8)** | not run | - | 3.14 / 0.944 / 18.7 | 3.46 / 4.08 / 3.20 | MIT or Apache-2.0 / shipped in the same repo, no separate weights licence. Yes. |
| DFN3 + clarity (production chain) | 48k | same | - | - | 3.05 / 0.940 / (5.8) | 3.42 / 4.02 / 3.15 | as above |
| MossFormer2_SE_48K | 48k | 0.33 (8; load 5.6), 1.29 (16; contended) | **0.051** | 0.26 GB | 3.08 / 0.955 / 19.4 | 3.44 / 4.08 / 3.19 | Apache-2.0 / Apache-2.0 (HF tag). Yes, but card says trained on "open-sourced and private data": get a provenance statement. |
| MossFormerGAN_SE_16K as-is | 16k | 7.5 (8) | **0.98** (6 s windows) | 2.2 GB (6 s), 3.6 GB (10 s) | 3.30 / 0.956 / 19.6 (at 16k) | 3.53 / 4.14 / 3.30 | Apache-2.0 / Apache-2.0 (HF tag); same private-data caveat. |
| Hybrid MossFormerGAN <7.5 kHz + DFN3 | 48k | GAN + 0.05 | about 0.98 (GAN) | 2.2 GB | 3.30 / 0.956 / 19.5 | 3.53 / 4.15 / 3.30 | both parts as above |
| Resemble denoiser only | 44.1k | 0.42 (8; load 13) | 0.050 | 1.9 GB | 2.75 / 0.947 / 12.2 | 3.42 / 3.98 / 3.11 | MIT / MIT (HF tag). Yes; training data not stated. |
| Resemble denoiser + enhancer, NFE 32 | 44.1k | 6.3 at NFE 16 (8) | **1.34** (NFE 16: 0.96, NFE 8: 0.77) | 2.5 GB | 2.31 / 0.911 / -19.7 | 3.46 / 3.99 / 3.15 | as above |
| DFN3-LL (live) | 48k | 0.284 core / stream | not run | - | 3.06 / 0.949 / 18.6 | 3.48 / 4.06 / 3.21 | as DFN3 (same release) |
| DPDFNet2 48 kHz HR (live) | 48k | 0.306 core | not run | - | 3.00 / 0.938 / 18.7 | 3.44 / 4.08 / 3.19 | Apache-2.0 / Apache-2.0 (HF model repo tag). Yes; training data includes FSD50K (per-clip CC licences, some NC). |
| DPDFNet8 48 kHz HR | 48k | 0.738 core | - | - | 2.89 / 0.933 / 18.8 | 3.43 / 4.09 / 3.18 | as above |
| DPDFNet2 / 4 / baseline, 16 kHz | 16k | 0.226 / 0.329 / 0.118 core (+0.005 resample) | - | - | 2.56 / 0.918 / 18.2 ; 2.54 / 0.918 / 18.3 ; 2.52 / 0.915 / 17.9 | 3.43 / 4.13 / 3.20 ; 3.44 / 4.15 / 3.22 ; 3.42 / 4.10 / 3.17 | as above |
| GTCRN (DNS3-trained) | 16k | 0.070 core (16 ms hop) | - | - | 2.51 / 0.917 / 14.6 | 3.24 / 3.91 / 2.92 | MIT / checkpoints in repo under the repo MIT licence (no separate statement). Yes; training data licences (DEMAND is ShareAlike per the DNS-Challenge listing) not reviewed. |
| GTCRN (VCTK-trained) | 16k | same | - | - | 2.83 / 0.941 / 18.0 | 3.37 / 4.03 / 3.09 | as above. In-domain for this test set (trained on VCTK-DEMAND), so flattering. |

The SI-SDR for 16 kHz-output models is measured at 16 kHz. LSD>8k tells the bandwidth story: **8.1** (DFN3), 9.9 (MF2), 7.0 (DFN3-LL), 11.5 (DPDFNet2 HR) against **about 25** for every 16 kHz-only output (MossFormerGAN as-is, DPDFNet 16k, GTCRN). The hybrid is back to **8.1** at identical 16 kHz scores.

### 4.2 Studio candidates after the clarity chain (production-equivalent)

| System | PESQ-WB | STOI | ESTOI | LSD>8k | SIG | BAK | OVRL |
|---|---|---|---|---|---|---|---|
| Noisy input | 1.90 | 0.921 | 0.785 | 12.3 | 3.32 | 3.09 | 2.68 |
| **DFN3 + clarity (production)** | 3.05 | 0.940 | 0.858 | 8.9 | 3.42 | 4.02 | 3.15 |
| MossFormer2_SE_48K + clarity | 3.01 | 0.951 | 0.880 | 9.9 | 3.37 | 4.02 | 3.11 |
| MossFormerGAN as-is + clarity | 3.22 | 0.951 | 0.875 | 24.4 | 3.44 | 4.07 | 3.19 |
| **Hybrid GAN + DFN3 + clarity** | **3.22** | 0.951 | 0.875 | 8.7 | 3.44 | 4.06 | 3.19 |
| Resemble denoiser + clarity | 2.72 | 0.943 | 0.861 | 12.8 | 3.41 | 3.98 | 3.12 |
| Resemble denoiser+enhancer NFE 32 + clarity | 2.29 | 0.907 | 0.793 | 13.8 | 3.39 | 3.84 | 3.04 |

Paired differences against the production chain (mean over 200 clips, 95 % CI):

| Comparison | PESQ | STOI | DNSMOS OVRL |
|---|---|---|---|
| Hybrid + clarity vs DFN3 + clarity | +0.170 +/- 0.039 | +0.011 +/- 0.003 | +0.043 +/- 0.016 |
| MF2 + clarity vs DFN3 + clarity | -0.039 +/- 0.037 | +0.010 +/- 0.003 | -0.041 +/- 0.018 |
| Resemble enhancer NFE 32 + clarity vs DFN3 + clarity | -0.758 +/- 0.058 | -0.034 +/- 0.003 | -0.108 +/- 0.026 |
| DFN3-LL vs DFN3 (raw) | -0.079 +/- 0.031 | +0.005 +/- 0.002 | +0.009 +/- 0.010 |
| DPDFNet2 HR vs DFN3 (raw) | -0.135 +/- 0.041 | -0.006 +/- 0.003 | -0.014 +/- 0.019 |
| GTCRN (DNS3) vs DFN3 (raw) | -0.629 +/- 0.044 | -0.028 +/- 0.004 | -0.279 +/- 0.030 |

### 4.3 Speed on the 10-minute file

| Model | Device | Threads | Audio | RTF | Load s | Peak RSS | Peak VRAM | Load avg before / after |
|---|---|---|---|---|---|---|---|---|
| mf2_48k | GPU | 4 | 600 s | **0.051** (30.6 s) | 2.2 | 1.7 GB | 0.26 GB | 0.10 / 0.99 |
| resemble_denoise | GPU | 4 | 600 s | **0.050** | 5.2 | 2.8 GB | 1.9 GB | 0.99 / 1.18 |
| mfgan16, 6 s windows | GPU | 4 | 180 s excerpt | **0.977** | 2.1 | 1.5 GB | 2.2 GB | 1.15 / 1.13 |
| mfgan16, 10 s windows | GPU | 4 | 60 s excerpt | **CUDA OOM** (needs > 3.9 GB) | | | | |
| resemble_enhance NFE 8 / 16 / 32 | GPU | 4 | 120 s excerpt | 0.77 / 0.96 / **1.34** | 5.4 | 2.8 GB | 2.5 GB | 1.3 to 2.4 |
| dfn3 | CPU | 1 / 4 / 8 | 600 s | 0.103 (62 s) / 0.049 / 0.046 (27 s) | 0.0 | 0.5 to 0.7 GB | - | 1.7 / 2.2 to 3.7 |
| mf2_48k | CPU | 8 | 60 s excerpt | 0.328 | 2.0 | 1.2 GB | - | 5.6 / 7.3 |
| mf2_48k | CPU | 16 | 60 s excerpt | 1.289 (contended, SMT oversubscription) | | | | 8.2 / 14.4 |
| resemble_denoise | CPU | 8 | 60 s excerpt | 0.418 | 4.8 | 3.0 GB | - | 13.0 / 10.5 |
| mfgan16 (6 s) | CPU | 8 | 30 s excerpt | 7.49 | 2.0 | 3.1 GB | - | 9.9 / 9.0 |
| resemble_enhance NFE 16 | CPU | 8 | 15 s excerpt | 6.28 | 4.9 | 3.4 GB | - | 7.8 / 8.3 |

Earlier 200-clip runs (clips <= 8.5 s, each processed in one pass) gave GPU RTF 0.067 (MF2), 0.053 (Resemble denoiser), 0.65 (MossFormerGAN with 10 s windows, 3.6 GB peak), 1.15 (Resemble enhancer NFE 32). A 6 s window makes the GAN fit the card but costs the 0.75-stride overlap (1.33x compute).

### 4.4 Live: per-frame cost and latency (one thread, pinned core)

| Model | Hop | Latency (stream lag) | Mean ms / frame | p99 | Max | Core fraction per stream | Streams per core at 70 % budget |
|---|---|---|---|---|---|---|---|
| DFN3 (libDF) | 10 ms | **30 ms** | 1.07 | 1.32 | 1.98 | **0.107** | 6.5 |
| DFN3-LL (libDF) | 10 ms | **10 ms** | 2.84 | 3.73 | 23.6 (one outlier, shared box) | 0.284 | 2.5 |
| DPDFNet2 48k HR | 10 ms | 50 ms | 3.06 | 3.36 | 6.8 | 0.306 | 2.3 |
| DPDFNet8 48k HR | 10 ms | 50 ms | 7.38 | 7.96 | 10.2 | 0.738 | 0.9 |
| DPDFNet2 16k | 10 ms | 50 ms | 2.26 | 2.91 | 3.2 | 0.226 (+0.005 resample) | 3.0 |
| DPDFNet4 16k | 10 ms | 50 ms | 3.29 | 3.98 | 4.3 | 0.329 | 2.1 |
| DPDFNet baseline 16k | 10 ms | 50 ms | 1.18 | 1.43 | 1.7 | 0.118 | 5.7 |
| GTCRN (DNS3) | 16 ms | 16 ms | 1.12 | 1.37 | 1.7 | 0.070 | 9.3 |

Production's own figure (about 0.1 core per stream, architecture.md) is reproduced. Pure model time only; add websocket, Opus and sandbox overhead from the real stream worker.

## 5. Observations (metrics, not listening)

1. **16 kHz-only outputs lose everything above 8 kHz** (LSD>8k about 25 dB against 8 dB). PESQ-WB, STOI and DNSMOS cannot see this, so MossFormerGAN's headline PESQ 3.30 is not a like-for-like win over a 48 kHz model. The linear-phase crossover (`hybrid.py`, 1535-tap Kaiser FIR, -6 dB at 7.5 kHz, complementary high-pass) restores the band from DFN3 with no change to the 16 kHz scores.
2. **MossFormer2_SE_48K is DFN3-level:** PESQ -0.04 (not significant), STOI +0.01, DNSMOS OVRL -0.04. It is 7x the CPU cost of DFN3 (0.33 vs 0.046) for no gain.
3. **Resemble:** the denoiser alone is worse than DFN3 on every measure (PESQ 2.75, SI-SDR 12.2). The enhancer is generative: SI-SDR about -20 dB and PESQ 2.3 mean the waveform is re-synthesised, which is a hallucination risk; its DNSMOS is near the baseline on average and swings per clip (best of all systems on the low SNR internal clip at 3.34 against 2.87, worst on VoiceBank p257_199 at 2.39 against 2.88). At NFE 32 it is 1.34x real time on this GPU, so it fails the Studio speed bar; NFE 16 passes (0.96) but its quality was not measured (only NFE 32 was).
4. **Reverberation:** on the reverberant clip DFN3 scored best among the studio candidates (DNSMOS OVRL 3.11 vs 2.64 MF2, 2.74 GAN, 2.73 hybrid, 2.31 Resemble denoiser, 2.94 Resemble enhancer).
5. **The 6 sample clips (mean DNSMOS OVRL, post clarity for studio, gain-matched for live):** noisy 2.47; DFN3+clarity **3.07**; hybrid 3.02; Resemble enhancer 3.02; MossFormerGAN 3.01; MF2 2.91; Resemble denoiser 2.83. Live: DFN3 3.07, DFN3-LL 3.06, DPDFNet2 HR 2.96, GTCRN 2.69. So the VoiceBank PESQ gain of the GAN models does not carry over to these clips. Likely cause: MossFormerGAN is advertised on VoiceBank-DEMAND (its card quotes PESQ 3.47 there), i.e. close to in-domain.
6. **DPDFNet:** the vendor's DNS4-blind gains over DFN3 were not reproduced here (VoiceBank PESQ -0.14, OVRL -0.01, both within noise or worse), and its streaming path adds up to 50 ms lag (model shifts the signal by 40 ms plus one hop). The 16 kHz variants score about 0.45 PESQ lower than the 48 kHz HR variant after our 48-16-48 conversion.
7. **DFN3-LL** matches DFN3 on DNSMOS (+0.009 +/- 0.010) with a better SIG, lower PESQ (-0.08), and 2.65x the CPU.
8. **GTCRN** is cheap but clearly worse (OVRL -0.28, SIG -0.22, 8 kHz band). Its DNS3 checkpoint is the general-purpose one; the VCTK checkpoint scores higher here only because it is trained on this test set's source.

Samples: `~/audio-cleaner/research/samples/<clip>/*.mp3` (192 kbps, 48 kHz mono) with `index.json` listing every file and its DNSMOS. Studio files carry the clarity chain; `live_*` files are raw model output gain-matched to -16 LUFS. `internal_noisy_snr0/` must stay on this machine.

## 6. Recommendations and integration

### 6.1 (a) Studio clean

Keep `dfn.denoise_array` + `clarity.process` (CPU, 4 to 8 threads, a 10-minute file in about 30 s). To add a GPU "Studio+" tier later, use the hybrid:

1. Runtime: a separate venv with the pins in section 2 (the production venv has no torch). Code is vendored under `~/audio-cleaner/research/keep/denoise/code/clearvoice/` (Apache-2.0; ship the LICENSE and keep the copyright header). Weights go to `~/audio-cleaner/models/mossformergan_se_16k/{last_best_checkpoint,last_best_checkpoint.pt}` (38.8 MB, load with `weights_only=True`).
2. Per channel: resample to 16 kHz, MossFormerGAN with **6 s windows, 4.5 s stride** (upstream 0.75 stride, global RMS normalisation over the whole file, discard the overlapped edges), 16 to 48 kHz, merge with the DFN3 output via `hybrid.merge` (FIR crossover at 7.5 kHz), then the clarity chain. 10 s windows do not fit 4 GB for long inputs; stay at or below 7 s.
3. Throughput: about 1.0x real time on the GPU (a 10-minute file takes about 10 minutes of exclusive GPU), plus 30 s of CPU for DFN3 in parallel. Model load 2 s. 2.2 GB VRAM, 1.5 GB RAM. CPU fallback is not viable (7.5x real time).
4. Sharing: take `~/audio-cleaner/dev/gpu.lock` (flock) around each job; the ASR work uses the same card (3.8 GB observed), so jobs must be serialised and GPU transcription cannot run at the same time as a 10 s-window GAN job. The Landlocked runner needs read access to the venv, the model directory and the NVIDIA device nodes (`/dev/nvidia*`, `/dev/nvidiactl`, `/dev/nvidia-uvm`) and its seccomp/network rules stay as they are.
5. Gate it on the owner's blind A/B of `samples/`. Expect a small gain (about +0.04 DNSMOS OVRL, +0.17 PESQ on VoiceBank) and no gain on reverberant or very noisy material.

Not recommended: MossFormer2_SE_48K (no gain, 7x CPU), Resemble Enhance (fails real time at NFE 32, denoiser-only is worse, enhancer is generative; NFE 16 untested).

### 6.2 (b) Live medium / high

- No change to `acw-stream` for medium: DFN3, hop 480, delay 1440 samples (30 ms). Capacity about 6 streams per core at a 70 % budget.
- "High" (lowest latency): DFN3-LL, `~/audio-cleaner/models/DeepFilterNet3_ll_onnx.tar.gz` through the same libDF API. Hop 480, **delay 480 samples (10 ms)** instead of `MODEL_DELAY = 1440`, so the stream worker needs the delay per model. Cost 0.284 core per stream: 2.5 streams per core, so plan for about 2.6x more CPU per high-tier stream.
- Latency convention used here: lag between the input stream and the output stream of a hop-by-hop host, measured by cross-correlation on clean speech (DFN3 30 ms, DFN3-LL 10 ms, GTCRN 16 ms, DPDFNet 50 ms). DPDFNet's official `StreamEnhancer` gives the same lag, and our driver is bit-identical to it.
- If an economy tier is wanted: GTCRN 16 kHz (MIT, `stream/onnx_models/gtcrn.onnx` in its repo, 352 KB, 7 % core) with an 8 kHz bandwidth and OVRL about 0.28 lower. Not recommended for paid tiers.

### 6.3 (c) GPU

Use it only for an optional Studio+ tier. For the existing Studio and live tiers it adds contention with ASR and no quality.

## 7. Licence notes

- **Code and weights of the winners are fine commercially:** DFN3 (MIT or Apache-2.0), MossFormerGAN_SE_16K and MossFormer2_SE_48K (Apache-2.0, HF card tags), Resemble Enhance (MIT, HF tag), DPDFNet (Apache-2.0 repo and HF model repo), GTCRN (MIT).
- **Provenance risk for the ClearerVoice weights:** both model cards say "trained on large scale datasets including open-sourced and private data". Ask Alibaba/ModelScope for a written statement before charging for output from them. Apache-2.0 applies to the weights file; it does not warrant the training data.
- DPDFNet training data (paper section III-A): DNS4, Multilingual LibriSpeech, MUSAN, FSD50K. FSD50K and DNS4 contain per-clip CC licences, some non-commercial or ShareAlike. Legal status of weights trained on them is unsettled; it is the usual risk, same as DFN3's DNS4 training.
- **DNSMOS** weights (`DNS-Challenge/DNSMOS/DNSMOS/sig_bak_ovr.onnx`): the repo LICENSE is CC BY 4.0, no weights-specific statement. Used for internal evaluation only; do not put them in the product. **PESQ** (ITU P.862) is licensed by ITU members; internal use only. `pesq 0.0.4` and `pystoi` are not shipped.
- **VoiceBank-DEMAND:** CC BY 4.0 (record licence text). Keep the attribution if you publish results or samples. The samples derived from it are CC BY 4.0 with that attribution.
- **`noisy_snr0.wav`:** CC BY-NC 3.0 (freesound 33711). Only used for the `internal_*` sample set, never in the repo.
- The reverberant sample uses freesound 573577 (CC0) and the DeepFilterNet RIR.

## 8. Problems and limits

- CPU timings for torch models were taken while the ASR benchmark ran (load average 5 to 13). They are upper bounds; the MF2 16-thread figure (1.29) is clearly contended.
- Resemble Enhance NFE 16 and NFE 8 have speed but no quality measurement (a 50-clip slice crashed on the tail-chunk bug before the fix; the partial result was discarded).
- The GAN's long-file windows (6 s) were timed (RTF 0.98) but their effect on quality was not measured; VoiceBank clips are <= 8.5 s, so quality was measured with 10 s windows (one pass per clip, no OOM).
- No human listening. 6 clips only. 200 of 824 pairs; the 95 % CIs above cover only the utterance sampling.
- The 16 kHz DPDFNet scores (PESQ about 2.5) are well below the 48 kHz HR variant and below the vendor's claims; the pipeline is bit-identical to the official streaming code, so this is the models under 48-16-48 conversion on VoiceBank, not an integration bug that we found.

## 9. Reproduce, and what was kept or deleted

- Scripts: `research/denoise/` (`setup_env.sh`, `fetch_models.sh`, `requirements.txt`, `common.py`, `models.py`, `live.py`, `hybrid.py`, `run_offline.py`, `eval_metrics.py`, `bench_speed.py`, `bench_live.py`, `make_long.py`, `make_samples.py`, `report.py`, `subset200.json`, `run_quality_*.sh`, `run_speed.sh`, `run_post.sh`). Order: `setup_env.sh`, `fetch_models.sh` (needs the test-set zips extracted flat into `denoise-data/vbd48_{clean,noisy}`), `run_quality_*.sh`, `run_speed.sh`, `run_post.sh`, `report.py`. Every GPU command is wrapped in `flock ~/audio-cleaner/dev/gpu.lock`.
- Kept in `~/audio-cleaner/research/keep/denoise/` (192 MB): MossFormerGAN_SE_16K weights (38.8 MB), the minimal ClearerVoice code (372 KB) and LICENSE, the 200 clean/noisy eval pairs, the 10-minute file, the DNSMOS onnx, all metric CSVs and speed/live JSON, run logs, `pip-freeze.txt`. A keep-only run (`ACW_DENOISE_KEEP_ONLY=1`) reproduced the hybrid output on 3 clips to 1e-4. Samples stay in `~/audio-cleaner/research/samples/` (6.9 MB).
- Deleted: all code clones, MossFormer2 (222 MB), Resemble (713 MB), DPDFNet and GTCRN files, the full test set and zips, all enhanced wavs, `se-venv` (6.7 GB, rebuild with `setup_env.sh`). About 9.9 GB freed. The shared pip cache (`~/.cache/pip`, 5.9 GB, also used by ASR) was left alone.
