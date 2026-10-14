# Speech recognition benchmark (2026-10-13 to 10-14)

Which ASR engines should replace or sit next to production's
faster-whisper large-v3-turbo (int8, CPU) for file transcription and live
captions, on the Ryzen 7 1700X (8C/16T) and the GTX 960 (4 GB, sm_52).
Scripts and raw results: `research/asr/` (untracked; `summarize.py` prints the tables).

## 1. Bottom line

| Use | Today | Recommended | Why |
|---|---|---|---|
| File transcription, English | Whisper turbo int8 CPU, RTF 0.41 | **Parakeet TDT 0.6B v2** (CPU fp32 RTF 0.09; GPU RTF 0.04) | 4.5x faster on CPU, long-form WER 2.2 vs 5.3, word timestamps |
| File transcription, 24 other European languages | Whisper turbo | **Parakeet TDT 0.6B v3** | es/de on par with Whisper, fr much better (5.2 vs 10.8) |
| File transcription, other languages | Whisper turbo | keep Whisper turbo | Parakeet covers 25 languages only |
| Live captions, English | Whisper small, utterance-based (caption after each pause) | **NeMo FastConformer streaming 480 ms** (or zipformer-en) | words appear ~0.65 s after they are spoken, 0.2-0.3 core/stream |
| Live captions, other languages | Whisper small utterances | keep (or premium OpenAI) | Nemotron-3.5 streaming is multilingual but ~2 cores/stream |
| Input to ASR | denoised audio for denoise+transcribe jobs | **original audio** (done) | every engine is worse on DFN3 output (table 4.1) |

The GTX 960 is useful for exactly one thing: Parakeet file transcription with
ONNX Runtime (RTF 0.04 and ~0.04 CPU-seconds per audio-second, i.e. it takes
transcription off the CPU almost entirely). It needs 3.7 GB of its 4 GB, so it
runs one job at a time. Whisper on the GPU works only with old CTranslate2 3.x
builds and is slower than Parakeet on the CPU.

## 2. Setup that works on sm_52 (Maxwell)

- **ONNX Runtime (Parakeet, GPU):** `onnxruntime-gpu==1.19.2` is the last PyPI
  build whose CUDA provider has sm_52 kernels (1.21+ start at sm_60; 1.31 is
  CUDA 13, which dropped Maxwell). CUDA 12 runtime from pip wheels
  (`nvidia-cublas-cu12==12.9.2.10`, `nvidia-cuda-runtime-cu12==12.9.79`,
  `nvidia-cufft-cu12==11.4.1.4`, `nvidia-curand-cu12==10.3.10.19`,
  `nvidia-cuda-nvrtc-cu12==12.9.86`, `nvidia-nvjitlink-cu12==12.9.86`) on
  `LD_LIBRARY_PATH`, and **cuDNN 9.7.1.26** (9.9+ deprecates pre-Turing; 9.27
  fails with `CUDNN_STATUS_EXECUTION_FAILED_CUDART`):
  `pip install --no-deps --target <dir> nvidia-cudnn-cu12==9.7.1.26`.
  Full pins: `research/asr/requirements-gpu.txt`.
- **CTranslate2 (Whisper, GPU):** 4.x wheels have no sm_52 code
  (`cudaErrorInvalidDevice`). `ctranslate2==3.24.0` (CUDA 11.2 build, sm_50
  SASS runs on sm_52) with `faster-whisper==0.10.1 --no-deps`, `av==19.0.1`,
  cuBLAS/cuDNN cu11 wheels. Pins: `research/asr/requirements-ct2old.txt`. Not recommended.
- **CPU:** `onnx-asr==0.12.0` (MIT) or `sherpa-onnx==1.13.8` (Apache-2.0, bundles
  its own ONNX Runtime and Silero VAD support). Pins: `research/asr/requirements.txt`.
- Driver 580.178.04, GPU shared between jobs with `flock ~/audio-cleaner/dev/gpu.lock`.

## 3. Method

- English: LibriSpeech test-clean, 150 utterances for WER (20-utterance common
  subset for slow engines), the same mixed with DEMAND noise at 5 dB SNR
  ("noisy 5 dB"), and that noisy set after DFN3 ("noisy+DFN3"). Long-form: a
  disjoint ~10-minute file of concatenated utterances with 0.3-1.5 s pauses.
- Other languages: FLEURS es/de/fr, 20 utterances each (Whisper CPU: first 10).
- WER with jiwer after Whisper's English/basic normalizers. RTF = wall time / audio
  time; "CPU-s/audio-s" counts all cores. CPU numbers were taken while other
  benchmarks ran on the box, so treat them as upper bounds.
- Streaming: real-time paced feed in 100 ms hops of a 60 s excerpt, word
  latency = time a word first appears minus its end time in the LibriSpeech
  forced alignments; N concurrent streams.

## 4. Results

### 4.1 Offline (files)

| Engine | clean WER (n) | noisy 5 dB | noisy+DFN3 | long-form WER | long RTF | CPU-s/audio-s | RSS MB | VRAM MB |
|---|---|---|---|---|---|---|---|---|
| Whisper large-v3-turbo int8 CPU (production) | 1.26 (20) | 1.89 | 2.32 | 5.30 | 0.405 | 2.39 | 1697 | - |
| distil-large-v3.5 int8 CPU | 2.11 (20) | 2.11 | 3.58 | 4.83 | 0.334 | 1.99 | 1592 | - |
| distil-large-v3.5 fp32 GPU (ct2 3.24) | 1.97 (50) | 3.47 | 4.23 | 5.70 | 0.109 | 0.12 | 3095 | 3868 |
| Parakeet TDT v2 fp32 CPU (onnx-asr) | 1.59 (150) | 2.20 | 2.81 | **2.21** | **0.090** | 0.71 | 4371 | - |
| Parakeet TDT v2 int8 CPU (onnx-asr) | 1.44 (150) | 2.07 | 2.90 | 2.28 | 0.174 | 1.08 | 3499 | - |
| Parakeet TDT v2 int8 CPU (sherpa-onnx + VAD) | 1.31 (150) | 2.23 | 2.90 | 2.55 | 0.109 | 0.69 | 3668 | - |
| Parakeet TDT v2 fp32 GPU | 1.59 (150) | 2.20 | 2.81 | 2.21 | **0.040** | **0.04** | 1062 | 3657 |
| Parakeet TDT v3 fp32 CPU | 1.69 (150) | - | - | 3.69 | 0.093 | 0.73 | 4370 | - |
| Parakeet TDT v3 fp32 GPU | 1.69 (150) | 2.39 | 3.25 | 3.69 | 0.043 | 0.04 | 1057 | 3781 |

On the identical 20-utterance subset: Whisper turbo 1.26 / 1.89 / 2.32,
Parakeet v2 1.47 / 2.53 / 2.95 (clean / noisy / noisy+DFN3). Whisper is
slightly better on short clean utterances. Parakeet is much better on long-form
audio, where Whisper drifts and drops or repeats text.

Throughput with several jobs (Parakeet v2 fp32 CPU): 1 job × 8 threads gives
8.5 audio-s per s, 2 × 4 gives 7.8, and 4 × 2 gives 9.8. That's about
**8-10 hours of audio per wall-clock hour** on the CPU, whatever the split
(Whisper turbo: about 2.5 h/h).

### 4.2 Other languages (FLEURS WER)

| Engine | es | de | fr |
|---|---|---|---|
| Whisper turbo int8 CPU (n=10) | 1.70 | 3.27 | 10.78 |
| Parakeet v3 fp32 (same 10) | 3.40 | 3.27 | 5.23 |
| Parakeet v3 fp32 (n=20) | 2.51 | 4.44 | 5.77 |
| Parakeet v3 int8 (n=20) | 3.13 | 8.67 | 8.49 |

int8 Parakeet v3 loses a lot on de/fr: use fp32.

### 4.3 Live captions (streaming)

| Model (sherpa-onnx, int8) | streams × threads | core per stream | word latency median / p90 | WER clean / noisy / noisy+DFN3 | RSS/stream | Licence |
|---|---|---|---|---|---|---|
| zipformer-en 2023-06-26 (320 ms chunks) | 4 × 1 | 0.11 | 0.61 / 0.76 s | 2.52 / 4.80 / 5.65 | 176 MB | Apache-2.0 |
| zipformer-en | 12 × 1 | 0.17 | 0.63 / 0.78 s | | | |
| NeMo FastConformer 480 ms | 4 × 1 | 0.19 | 0.69 / 0.99 s | 3.07 / 3.95 / 5.23 | 268 MB | CC-BY-4.0 |
| NeMo FastConformer 80 ms | 4 × 1 | 0.48 | 0.36 / 0.55 s | 2.63 / - / - | 266 MB | CC-BY-4.0 |
| Nemotron speech streaming en 0.6B, 560 ms | 1 × 4 | 1.42 | 0.59 / 1.01 s | 2.52 / 2.40 / 4.24 | 972 MB | NVIDIA Open Model Licence |
| Nemotron-3.5 ASR streaming 0.6B (35+ langs), 560 ms | 1 × 4 | 1.89 | 0.67 / 0.95 s | 3.18 / - / - | 991 MB | OpenMDW-1.1 |
| Nemotron 160 ms | 1 × 4 | 4.8 (falls behind) | 2.7 s | 2.96 | | |

Production's whisper-small captioner shows a line only after each pause
(typically 2-6 s after the first word) and was not part of this run.

## 5. Recommendations and integration

1. **Original audio for ASR (done).** Denoise+transcribe jobs now transcribe the
   decoded original, not the DFN3 output. Live local captions still segment and
   recognise the denoised signal. Feeding the recogniser raw audio there needs a
   delay-aligned raw buffer, because the energy VAD should keep using the denoised
   signal. Do it when live captions move to a streaming model.
2. **File transcription:** add a Parakeet engine (`acw/asr/parakeet.py`)
   built on onnx-asr with the fp32 models, Silero VAD segmentation (30 s max,
   500 ms min silence) and token timestamps for SRT/VTT. This is exactly the
   measured `pk2-fp32-cpu` setup. Route English to v2 and the other 24 Parakeet
   v3 languages to v3. Route everything else to Whisper turbo, including `auto`
   when Whisper's language ID on the first 30 s picks such a language. Keep the
   price; queue time drops ~4x. RAM is ~4.3 GB per job (fp32), so 2 workers need
   ~9 GB. The sandbox needs read access to the model dir. `onnx-asr` and
   `onnxruntime` (CPU) go into `requirements.lock`.
3. **GPU (optional, after 2):** one GPU worker running Parakeet with
   onnxruntime-gpu 1.19.2 + cuDNN 9.7.1.26 in its own venv, holding the GPU lock.
   CPU workers stay as the fallback. Landlock read access to `/dev/nvidia*`
   and the CUDA wheels is needed.
4. **Live captions, English:** replace the utterance captioner with NeMo
   FastConformer 480 ms streaming. It's more robust to noise than zipformer and
   still cheap. Partial results become possible (the plugin sends CEA-608 roll-up
   captions already). Keep whisper-small utterances for other languages.
5. **Licences:** Parakeet v2/v3 and NeMo FastConformer weights are CC-BY-4.0,
   so add an attribution line to the licences page and docs when shipping. Zipformer
   (icefall) is Apache-2.0. sherpa-onnx is Apache-2.0, onnx-asr MIT, ONNX
   Runtime MIT. Nemotron's NVIDIA Open Model Licence allows commercial use, but
   it's too expensive per stream here anyway.

## 6. What was kept

Winners are kept under `~/audio-cleaner/research/keep/asr/`: Parakeet TDT v2 and
v3 fp32 ONNX (onnx-asr layout, 3 GB each, also on Hugging Face as
`istupakov/parakeet-tdt-0.6b-v{2,3}-onnx`), Silero VAD, and the sherpa-onnx
streaming models NeMo FastConformer 480 ms and zipformer-en. Everything else was
deleted: distil-whisper, Nemotron, the int8 Parakeet exports, wheel caches, the
CTranslate2-3 venv and the evaluation audio. The research venvs can be rebuilt
from the pins in `research/asr/requirements*.txt`.
