# Sidon as an optional "restore" step for recorded files (2026-10-15)

Goal: an optional, slow step for recorded files that removes room echo / reverb and noise and rebuilds the band up to
24 kHz by re-synthesising the voice at 48 kHz. Building blocks (nothing is wired into `pipeline.py`):
`common/acw_common/restore.py` (runtime), `common/tests/test_restore.py`, and a one-time conversion script
`tools/convert_sidon_onnx.py` (its logic is described below).

Result in one line: it works and runs at about 0.8x real time on 4 CPU threads with onnxruntime (no new dependency, no
torch), but it is a generative model and on strongly reverberant speech it changed words. Ship it as an explicit,
off-by-default option.

## Licence (verified 2026-10-15)

| item | licence | where checked |
|---|---|---|
| `sarulab-speech/sidon-v0.1` weights | MIT | HF API `cardData.license = mit`; the model card text itself is only contributors + acknowledgements |
| `github.com/sarulab-speech/Sidon` code (training, configs) | MIT, (c) 2025 sarulab-speech | `LICENSE` file in the repo, GitHub API `spdx_id: MIT` |
| base model `facebook/w2v-bert-2.0` (first 8 layers are inside Sidon) | MIT | HF API |
| training data `google/fleurs-r`, `parler-tts/libritts_r_filtered` | CC BY 4.0 | HF API (attribution only; says nothing about the weights) |
| `cstr/Sidon-GGUF` | MIT (model), runtime CrispASR / ggml MIT | HF API, GitHub `LICENSE` |

So the earlier "no licence" note in `denoise-candidates.md` is wrong for the weights and the repo. Remaining caveat: the
model card does not repeat the licence in prose and the trained weights inherit nothing stricter that we could find.
Attribution lines to ship in the app's notices: Sidon (Nakata, Saito, et al., sarulab-speech, MIT), w2v-BERT 2.0 (Meta, MIT).

## Official inference recipe

From the authors' HF Space `sarulab-speech/sidon_demo_beta` (`app.py`) and `export.py` in the GitHub repo. The repo has no
inference script for single-speaker Sidon; the TorchScript files are the release.

1. Peak-normalise (irrelevant: the fbank is mean-normalised, so the model is gain invariant), 50 Hz high-pass biquad,
   resample to **16 kHz** mono.
2. Zero-pad the end by 1.5 s; cut into chunks of 96 s; pad each chunk by 160 samples per side.
3. `SeamlessM4TFeatureExtractor` of `facebook/w2v-bert-2.0`: 80 Kaldi mel bins, 25 ms povey window, 10 ms hop, 2-frame
   stack -> `(T, 160)`, mean/variance normalised **per chunk**.
4. `feature_extractor_cpu.pt` (TorchScript, w2v-BERT 2.0 first 8 layers + merged LoRA): `{"last_hidden_state": (1, T, 1024)}`,
   50 features/s.
5. `decoder_cpu.pt` (TorchScript, DAC decoder, rates 8x5x4x3x2 = 960): `(1, 1024, T) -> (1, 1, T*960)`, i.e. 48 kHz. The last
   960 samples are dropped; the result is cropped to the original length.

`restore.py` reimplements step 3 in numpy (matches `transformers` to 1e-6; a test checks it when `transformers` is
installed), so `transformers` is not a runtime dependency.

## Files

Pinned: HF revision `b3b02d8bbd55fdbc410e6e46e76ef95ace4fbf52` (2026-10-15). `~/audio-cleaner/models/sidon/`:

| file | size | sha256 |
|---|---|---|
| `feature_extractor_cpu.pt` (official, matches HF LFS oid) | 795 498 929 | `fd9abc906a9048b3c047bd3d246a25f6485d09c2d6bb85e098f527f844efc019` |
| `decoder_cpu.pt` (official, matches HF LFS oid) | 209 824 877 | `34dcc80fab75bd1336369ba5b2063f6da475887aa824db94b495c4059c5fdea4` |
| `sidon_fe.onnx` (ours, from the two above) | 397 857 704 | `76222cc7f4dd95eb3d137d81783e73b69b00cab66b65a91790e0e72c96452a75` |
| `sidon_dec.onnx` (ours) | 104 975 475 | `08c7547572f000d964f7dc3ef120fdf155c2d936721d9d320d20c6c4a1d652c4` |
| `sidon_fe_q8.onnx` (ours, optional int8 variant of the extractor) | 276 861 944 | `3c34fa7296eef1121f76b4a0ed7c066706d53ac3c2bc7248e4f57132e469cf3b` |

The CUDA variants and the GGUF files were not kept in that folder (GGUF test copies are under
`~/audio-cleaner/research/sidon-src/gguf/`).

### The ONNX conversion

`torch.onnx.export(dynamo=False, opset 17)` of the TorchScript modules (the extractor needs a scripted wrapper that returns
`["last_hidden_state"]` because ONNX cannot return a dict), then: weights, which the exporter emits as `Constant` nodes, are
turned into initializers; large float32 initializers are stored as float16 with a `Cast` back to float32 (onnxruntime folds
it at load; compute stays float32). Check against TorchScript on 10 s of speech: extractor cosine 0.99999 (min 0.999991
over frames), decoder relative error 7e-4; with fp32 storage the error is 6e-6. The optional int8 file is
`quantize_dynamic(QUInt8, per_channel, MatMul only)`: feature relative error 5%, mean cosine 0.9988. (QInt8 is far worse:
cosine 0.98, 20% error, AVX2 saturation; do not use it.) The ONNX files need a one-time conversion with torch installed
(the sidon-venv); runtime needs neither torch nor the conversion script.

## Speed and memory

Machine: Ryzen 7 1700X (Zen 1, AVX2 only, 8 cores / 16 threads), a **live production server shared with other jobs**
(load average 2 to 8 during the runs), always `nice -n 19`. 42.4 s of speech (the 10.6 s test clip x4), `restore()` end to
end including front end, windowing and resampling. Wall times are therefore pessimistic and noisy; CPU seconds are steadier.

| backend | threads | wall | RTF | CPU time | peak RAM |
|---|---|---|---|---|---|
| onnxruntime, fp16-stored weights | 4 | 35 to 42 s | **0.84 (0.77 to 0.98)** | 111 to 126 s | 2.2 to 2.3 GB |
| onnxruntime | 8 | 28 to 40 s | 0.67 (0.93 when the machine was busy) | 138 to 201 s | 2.2 GB |
| onnxruntime, int8 extractor | 4 | 35 to 37 s | 0.82 to 0.88 | 110 to 117 s | **1.8 GB** |
| torch 2.14.1+cpu, TorchScript | 4 | 61 s | 1.43 | 180 s | 1.6 GB |
| torch, TorchScript | 8 | 53 s | 1.24 | 284 s | 1.65 GB |
| CrispASR + GGUF q8_0 (10.6 s clip) | 4 | 22 s | 2.1 (1.8 with `CRISPASR_SIDON_FASTCONV=1`) | 84 s | 1.0 to 1.3 GB |

- 180 s of speech: RTF 0.77 at 4 threads, peak RAM 2.47 GB (42 s: 2.3 GB) so memory is bounded by the window size, not the
  file length (the rest is the models, about 1.1 GB for the extractor alone, and the input/output arrays).
- Where the time goes (ORT, 4 threads): decoder about 0.6 of real time (DAC convolutions at up to 48 kHz, 28% of it Snake
  activations), feature extractor 0.14 to 0.25. Eight threads help little (0.84 -> 0.67): the decoder does not scale.
- int8 gives **no speed-up** on this CPU (no VNNI); it only shrinks the download and the RAM.
- ONNX beats TorchScript by 1.7x, almost entirely in the decoder (2.9 s vs 4.9 s for 5 s of audio).
- The GGUF route is 2.5x slower than ONNX: ggml's convolutions dominate (DAC stage 15 to 19 s of the 19 to 22 s total;
  predictor 2.8 s, comparable to ORT). CrispASR is a 30 MB C++ binary with `--s2s`, a C ABI and Python bindings, no torch,
  but it caps one call at about 60 s of input unless split, so it is not worth it when onnxruntime is already shipped.
- Model load: 2.2 s (page cache warm). `restore.unload()` frees about 1.1 GB.
- `torch` CPU install size, if the TorchScript fallback were ever shipped: PyPI/pytorch.org CPU wheel 196 MB (Linux x86_64),
  124 MB (Windows), 127 MB (macOS arm64); installed 770 MB for `torch/` alone plus about 150 MB of dependencies on Linux.
  onnxruntime 1.31.0 is already in `server/` and `desktop/requirements.lock` (wheel 14 to 23 MB).
- A modern desktop CPU (AVX-512 or Apple silicon) should be at least 2x faster than this Zen 1; not measured.

## Chunking

Windows of 16 s with 2 s linear cross-fade **in feature space**, then the decoder on 6 s blocks with 12 frames (0.24 s) of
context on each side that are cropped away. Decoder blocking is bit-exact (a test checks `max |diff| < 1e-4`; with 10 context
frames the difference is exactly 0). The window seams are the only approximation: features differ a little between windows
(attention context and per-window normalisation), and the cross-fade hides it. Chunked vs one single window on the same
speech: mean log-spectral difference 2.1 dB over 40 bands, no seam outliers; waveforms differ completely in phase, which is
normal for a generative vocoder (so never compare the waveforms of two runs sample by sample). The output is about 8 to 15 ms
late relative to the input (generative phase, measured with envelope correlation); not compensated.

Cancel is checked before every window and after every decoded block: latency is a few seconds. Stereo: the mid signal is
restored once and written to both channels when the side signal is more than 20 dB below it, otherwise every channel is done
separately (2x time). The output is peak-limited to 0.99 and its level is the model's own, so loudness normalisation has to
run after it.

## Word-change check (faster-whisper small, int8, English, beam 5, T=0)

Reference = the clean `live_opus.wav` transcript ("We will not be held responsible for any hearing impairments or damage
caused to you from excessive exposure to this sound."). "changed" = word substitutions/insertions/deletions between the
transcript of the input and of the restored output (21 words for the long clips).

| input | words | input transcript differs from clean | restored transcript differs from input | notes |
|---|---|---|---|---|
| live_opus.wav (clean) | 21 | - | **0** | identical text |
| + synthetic RIR, RT60 0.7 s, direct-to-reverb 0 dB | 21 | 1 ("sun") | **5** | "impairments" -> "impendence", "damage caused" -> "died hoarse", "to" -> "of"; "sun" -> "sound" fixed |
| + RIR, RT60 0.4 s, +5 dB | 21 | 1 | **1** | the only change fixes "sun" -> "sound" |
| + RIR, RT60 1.0 s, -3 dB | 21 | 1 | **2** | "impairments" -> "impendence"; "sun" fixed |
| VoiceBank noisy clips (4, 3 to 4 s) | 9 to 11 | - | 0, 0, 0, **1** | "singer" -> "singlet" |
| `reverb_` sample clip (2 s) | 4 | - | 0 | |
| noisy clip, SNR 0 dB (10.6 s, local-only source) | 21 | - | 0 | |

Against the clean reference the 0.7 s reverb input had 1 word error and the restored output has 5, so **on heavy reverb
Sidon trades acoustic quality for content errors**; on clean, noisy and mildly reverberant speech it was neutral or better.
This agrees with the paper's own tables (WER 0.045 vs 0.040 noisy on LibriTTS). Only ten short clips, one voice for the
reverb tests; treat the numbers as indicative. Stored weights in fp32, fp16 (shipped) and the int8 extractor give the same
word-level result on all clips, so the smaller files cost nothing measurable here.

Speaker proxy (no speaker-embedding model on this machine): median F0 225 -> 227 Hz (clean), 246 -> 232 Hz (reverb input;
the reverberant pitch track is itself unreliable), long-term average spectrum correlation 0.97 to 0.99 (150 Hz to 8 kHz).
The paper reports speaker similarity 0.971 on LibriTTS (not checked). Treat "same person" as likely, not verified. The band
above 8 kHz: the restored files carry 7 to 14 dB less relative energy at 8 to 24 kHz than the noisy/opus input for the test
clip (it also removes the input's noise there), and about the same on VoiceBank; I did not see highs "added" to full-band material.

## Packaging cost

- Dependencies: **none new** (numpy, scipy, onnxruntime are already locked for server and desktop). torch is never imported
  unless `backend="torch"`. `transformers` and `pytest` only for tests (the fbank test skips without transformers).
- Download for the desktop: 503 MB (`sidon_fe.onnx` 398 MB + `sidon_dec.onnx` 105 MB), or 382 MB with the int8 extractor
  (277 + 105 MB, a 5% feature error, same words in the tests). Float weights do not compress. Needs the usual model-pack
  route (the same as the other models).
- RAM while running: 2.3 GB (1.8 GB with the int8 extractor) plus the audio; the desktop should warn below about 4 GB free.
- Disk during conversion (not needed at runtime): the 1 GB official TorchScript files.
- Intel Macs use onnxruntime 1.23.2 in the lock; the models are opset 17 with `Cast`, `LayerNormalization`, `Conv`,
  `ConvTranspose`: should load, **not tested**.

## Recommendation

- **(a) Server file jobs (budget about 1x real time on 4 threads):** feasible but with no margin. 0.77 to 0.98 RTF measured on
  a busy production host at nice 19; with a 30-minute file that is 25 to 30 minutes of 4 cores, and with 8 threads only 20%
  faster. Offer it as an explicit per-job option ("Studio restore, slow"), off by default, run through the existing
  worker queue with the cancel hook, one job at a time, and show a warning that voices and (on very reverberant audio)
  words can change. I would cap it (for example 60 minutes of audio) and estimate the time from `0.9 x duration`.
- **(b) Desktop (up to 3x on 4 threads is acceptable):** yes. It runs 3x inside the allowance on this old CPU and
  probably 0.3 to 0.5 on current hardware, adds only the 380 to 500 MB model download and no dependency. Same warning text;
  default off; keep the model as an on-demand download, not in the installer.
- Not recommended: torch/TorchScript (1.7x slower, 770 MB runtime), the GGUF/CrispASR route (2.5x slower), int8 for speed
  (no gain on AVX2-only CPUs; it is only a size option).
- Order in the pipeline: after the DFN denoise (or instead of it for reverberant recordings), before loudness
  normalisation. Not for live audio, not for music, singing or other languages' tonal speech without a listening check.

## Not verified / open

- Same-speaker identity (proxy only); reverb tests used one synthetic RIR family and one voice; no listening test.
- Speed on other CPUs (AVX-512, Apple, Windows laptops); machine was shared, so RTF has a +/- 15% band.
- Intel-Mac onnxruntime 1.23.2 and Windows runs; the 503 MB pack route and a hash check for the ONNX files in the app.
- Whether the ~10 ms output delay matters for lip-sync (not compensated).
- Very long inputs in a single call: memory is bounded, but the 16 kHz copy (1/3 of the input size) is held per channel.

## Reproduce

```
python3 -m venv sidon-venv && sidon-venv/bin/pip install torch numpy scipy soundfile onnx onnxruntime \
    --extra-index-url https://download.pytorch.org/whl/cpu          # torch 2.14.1+cpu
curl -L -o decoder_cpu.pt            https://huggingface.co/sarulab-speech/sidon-v0.1/resolve/b3b02d8bbd55fdbc410e6e46e76ef95ace4fbf52/decoder_cpu.pt
curl -L -o feature_extractor_cpu.pt  https://huggingface.co/sarulab-speech/sidon-v0.1/resolve/b3b02d8bbd55fdbc410e6e46e76ef95ace4fbf52/feature_extractor_cpu.pt
python -I convert_onnx.py MODEL_DIR  # writes sidon_fe.onnx, sidon_dec.onnx, sidon_fe_q8.onnx
nice -n 19 python -m pytest common/tests/test_restore.py     # ACW_SIDON_MODEL=MODEL_DIR, ACW_SPEECH_WAV=clip.wav
```

Scratch outputs (inputs, restored WAVs, transcripts): `/tmp/claude-1000/-home-silver/8e7d3362-a882-56d7-9723-cae3e2c616d9/scratchpad/sidon-*.wav`.
