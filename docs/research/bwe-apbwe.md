# Bandwidth extension with AP-BWE (2026-10-15)

Goal: restore the highs of band-limited recordings (phone calls, VoIP, old tape: speech that ends at 3.4 to 12 kHz)
when Audio Cleaner processes a recorded file. Nothing here is wired into `pipeline.py` yet; the building blocks are
`common/acw_common/bwe.py` (runtime), `common/tests/test_bwe.py` and `tools/export_apbwe_onnx.py` (one-time export).

## Model and licence

AP-BWE (Lu, Ai, Du, Ling; IEEE/ACM TASLP 2024, github.com/yxlu-0102/AP-BWE). All-convolutional GAN generator
(ConvNeXt, 8 layers x 2 streams, 512 channels, 29.8 M parameters) on 48 kHz STFT frames (n_fft 1024, win 320, hop 80).
Input: the narrow-band signal resampled back to 48 kHz. Output: wide-band log-magnitude (residual on the input) and
wide-band phase. Four official 48 kHz generators, named after the input sampling rate: `8k` (content to 4 kHz), `12k`
(6 kHz), `16k` (8 kHz), `24k` (12 kHz). Same architecture and the same STFT, only the weights differ.

Licence checked in the repo clone: `LICENSE` is MIT (c) 2023 Ye-Xin Lu; `weights_LICENSE.txt` says the weights are MIT
as well. Both files are copied to `~/audio-cleaner/models/ap-bwe/`. Caveat: the README does not state the training
data licence (VCTK-0.92, CC BY 4.0); the weights file itself carries the MIT statement.

## Files

`~/audio-cleaner/models/ap-bwe/` (not in git): `8kto48k.onnx`, `12kto48k.onnx`, `16kto48k.onnx`, `24kto48k.onnx`
(119 093 242 bytes each, fp32, opset 17, dynamic time axis), `SHA256SUMS` (the ONNX files),
`SOURCE_CHECKPOINTS.sha256` (the official `.pt` checkpoints), the two licence files.

| file | sha256 (onnx) | sha256 (official checkpoint) |
|---|---|---|
| 8kto48k | 788d45ed37d399c0eeffca0462d7b9c2928871ae8523af67e96f9395720137a3 | 8f0387a4f1cdf1ec5dc557a9e8a303326b45b7706b37843f771f24ee94d2e521 |
| 12kto48k | eaa2d5822e1094d2ec93fd233be8b1f2a20ae76a41b2702d404aa0b7ac00d618 | defa766ac25ccb5eda300860a7cbad5c7bfa02ee2b5e28d07c1daa470f8e8f39 |
| 16kto48k | ca91ce8a19b91555e6c49f1cce4e44547bb1073af7f32eae62edad9ba83ecd44 | 305a05dcab7dc29ffba09d32692d7a34550fc8fbdf338013641ff5d39a3cb285 |
| 24kto48k | e662a7fe7243507fe766092a2b62869f55d75952fcce5ef8b1deb6d99076cf55 | 00950c7782a076e218f9c5a9bbc3b59a5beb0c4617b62ccea836df5f616a63dd |

Weights come from the Google Drive folder in the README (`gdown --folder`). The checkpoints hold only the
`generator` state dict and load with `torch.load(weights_only=True)`.

Reproduce: `tools/export_apbwe_onnx.py` header lists the pins (torch 2.5.1+cpu, numpy 2.5.4, onnx 1.23.2, onnxscript
0.7.2). Sources and venv live in `~/audio-cleaner/research/ap-bwe-src/` and `ap-bwe-venv/`.

## Export and fidelity

The ONNX graph is the network only (mag, pha in; mag_wb and the re/im of the predicted phase out). STFT, iSTFT,
`exp`, and `atan2` normalisation are numpy in `bwe.py`.

| check | result |
|---|---|
| my re-implemented torch model vs the official `APNet_BWE_Model`, same weights | identical (max abs 0) |
| ONNX (onnxruntime) vs torch, network outputs (log-mag, phase re/im), 3 s of noise | max abs 2.4e-4 (8k), 8.9e-5 (12k), 7.0e-5 (16k), 1.9e-5 (24k) |
| ONNX + torch STFT/iSTFT vs the official inference path, waveform | max abs 4e-7, rms 7e-8 |
| numpy STFT vs `torch.stft` | max abs 7.5e-7 (relative 1.8e-7); iSTFT round trip 6e-8; vs `torch.istft` 9e-8 |
| numpy STFT + ONNX vs fully-torch path on speech (10 s, 8k and 16k models) | rms diff -34 dB re signal, max 4.5 % of peak |

The last line is not an export error. The network is very sensitive to the *phase of near-empty bins* (the
band-limited input has |S| ~ 1e-6 there, so the phase is rounding noise). Torch itself shows the same: float32 STFT
vs float64 STFT of the same signal, same torch network, differs by rms 7e-4 (about -37 dB). Real recordings have a
noise floor above that, and the output is statistically the same.

Blocks: the receptive field is +-27 frames (2160 samples), so block processing is overlap-save with 4000 samples of
context per side; output of 3 s blocks is bit-identical to a single pass in the tests (and 0.5 s blocks too).
Peak memory is independent of the file length.

## What the model does, and what `extend` does with it

* The generator regenerates the **whole** band (phase is predicted from scratch, magnitude is a residual). The paper's
  inference script writes its output directly. `extend` therefore blends the *original* spectrum back in below a
  crossover (default 0.9 x measured cut-off, at most 0.95 x the model's input Nyquist, linear ramp 375 Hz wide) so
  the recorded band is kept: the change below 2.8 kHz in the 3.4 kHz test is -61 dB re signal. Only the missing
  band comes from the network.
* The generators were trained on inputs that are full up to lr/2 (the resampler's cut-off). A recording that ends
  lower (3.4 kHz phone speech into the 4 kHz model) leaves a hole in the input and the model then hardly generates
  anything (4-8 kHz level -75 dB vs -48 dB in the original). The hole is filled with the mirrored spectrum just below
  the real edge (gain 0.6, "spectral folding") before the network sees it; 4-8 kHz then comes out at -61 dB.
  The generated highs are still 10 to 25 dB below the true ones on the test clip: the model is conservative.
* Guards: generated bin magnitude <= 2 x loudest input bin of the frame (white noise, clicks made the network
  produce peaks of 300+ otherwise); faded out where the input is (nearly) digital silence (otherwise a hiss of about
  -75 dBFS appears in gaps). Input is used at its actual level; gain from 0.1 to 3 changed the generated/in-band
  ratio by < 1 dB.
* Detection: `effective_bandwidth` finds the first frequency above 2.4 kHz where the long-term spectrum of the active
  frames is 20 dB below its 0.6-2.4 kHz level and stays there, accepts it only if the level falls >= 18 dB between just
  below and just above it, and reports the point 12 dB below the level before it. On sharp low-passes (FIR/resampler/
  codec) it lands within a few percent of the true cut-off (3.4k -> 3.45k, 4k -> 4.03k, 6k -> 5.7k, 8k -> 8.0k,
  12k -> 11.5k); gentle filters (6th-order Butterworth) read 10 to 40 % high. `pick_model` limits: 4.4 / 6.6 / 8.8 /
  12.6 kHz, anything wider is None. On the repo's real clips: full-band noisy/Opus/MP3 inputs read 17 to 24 kHz (None);
  the 16 kHz-model denoiser outputs read 7.6 to 8.7 kHz (16k); some DFN3 outputs of short clips read 5.2 kHz (their
  highs really are gone), which is why extension should run on the *input* file, before denoising.

## Speed and memory (primary-ram, Ryzen 7 1700X, machine busy: load average about 4, `nice -n 19`)

60 s of audio, ONNX file 119 MB, fp32, onnxruntime 1.31 CPU, 3 s blocks unless noted:

| model | threads | time | RTF | peak RSS |
|---|---|---|---|---|
| 8k->48k | 1 | 51 s (8 s blocks) | 0.86 | 680 MB (8 s blocks) |
| 8k->48k | 4 | 13 s | 0.22 | 450 MB |
| 16k->48k | 1 | 47 s (8 s blocks) | 0.79 | 680 MB (8 s blocks) |
| 16k->48k | 4 | 14 s (8 s blocks) | 0.24 | 674 MB (8 s blocks) |

(20 s blocks: 1.1 GB, no faster.) About 30 GFLOP per second of audio, so 4 threads on a busy machine give 4 to 5x
real time, one thread about 1.2x. Cost is the same for all four models. The paper's 18x real time on a CPU needs a
faster/unloaded core set. Session load is about 0.5 s; one session is cached.

## Not done / open

* Not in the pipeline; `extend(x, model_dir, threads, progress, cancel, out)` is ready. Needs onnxruntime (already a
  dependency of the other models), numpy and scipy only.
* Low-cut recordings (phone speech is 300 to 3400 Hz) keep their missing 100 to 300 Hz; this is high-band only.
* Nobody has listened to the results; judgement so far is from spectra on one 10 s clip.
* The 2k/4k/8k->16k generators in the same Drive folder were not exported (16 kHz output, not needed for 48 kHz files).
