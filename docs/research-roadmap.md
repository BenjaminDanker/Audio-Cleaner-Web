# Research roadmap (after launch-readiness)

Ordered by expected payoff. Each item ends in a benchmark table on this
machine plus a listening/WER check, and is wired in behind a tier name so
prices/tiers don't change shape.

## 1. Faster local transcription (biggest bottleneck)

Today: faster-whisper large-v3-turbo int8 = RTF ~0.45 on 8 threads (Zen 1 has
no AVX-512/VNNI, so int8 gains are small).

Candidates:
- **NVIDIA Parakeet TDT 0.6B v2 (English) / v3 (25 European languages)** via
  ONNX (sherpa-onnx or onnx-asr). Reported many times faster than Whisper on
  CPU with competitive WER, and it gives word timestamps. Good fit for the
  "standard" tier and for **true streaming live captions** (cache-aware
  streaming FastConformer variants) instead of the utterance-based approach.
- Streaming Zipformer transducers (sherpa-onnx) for very cheap live English
  captions.
- distil-large-v3.5 (English-only Whisper distillation) as a drop-in.
- Keep Whisper turbo for languages the others don't cover.

## 2. Better offline denoise for "Studio clean"

DFN3 is a real-time model. Offline jobs can afford much more compute:
- ClearerVoice-Studio MossFormer2_SE_48K (Apache-2.0), 48 kHz.
- Resemble Enhance (MIT): denoise + bandwidth extension ("studio" sound). Heavy;
  likely GPU-only.
- Newer 2025-2026 generative/regression SE models (FlowSE, SEMamba, etc.) -
  check licences (many are research-only) and CPU cost.
- Evaluate with DNSMOS / PESQ / SI-SDR on a fixed test set plus blind A/B listening.
  Keep DFN3 as a fallback tier.

## 3. Live denoise tiers - done (see research/denoise.md)

- Shipped: `light` = DFN3 (30 ms, ~0.11 core/stream). A `low_latency` tier
  (DFN3-LL, 10 ms, ~0.28 core/stream, same quality) was shipped, then dropped
  by the owner (10-20 ms isn't worth 2.6x the CPU); the model file can be
  deleted. No causal candidate
  (DPDFNet, GTCRN) beat DFN3, so there is no separate "high quality" live tier.
- Studio clean stays DFN3 + clarity; an optional GPU hybrid
  (MossFormerGAN-SE-16k below 7.5 kHz + DFN3) is pending a blind A/B listen.

## 4. GTX 960 (Maxwell, sm_52, 2-4 GB)

Blocked on the NVIDIA driver, which needs sudo and a reboot:
`sudo apt install nvidia-driver-580` (Ubuntu's recommended branch for the
GM206, and the last NVIDIA branch with Maxwell support, so GPU work stays on
CUDA 12.x).

Constraints to verify:
- Recent CUDA 13 / PyTorch builds dropped Maxwell. CUDA 12.x and the
  matching onnxruntime-gpu / CTranslate2 builds still support it.
- No FP16 tensor cores. Use FP32 (≈2.3 TFLOPS). int8 on GPU needs sm_61+,
  so it's unavailable here.
- Keeping weights resident on the GPU avoids load time per job/stream.

Plan:
1. Install the driver, check `nvidia-smi`, check CUDA 12 runtime availability.
2. Benchmark Whisper turbo (CTranslate2 cuda float32), Parakeet (onnxruntime
   CUDA EP), and batched DFN3 (ONNX on CUDA) for many concurrent streams.
3. If the GPU wins, run one GPU service that keeps models resident, behind
   the same Unix-socket interface as acw-stream, with CPU fallback.
4. For live audio, measure per-frame latency on the GPU. Small batched
   kernels on Maxwell may be slower than CPU per frame, but cheaper per stream.

## 5. Product follow-ups

- Opus on the live uplink (the plugin encodes; the stream worker decodes, still sandboxed).
- Resumable uploads (tus) for multi-GB videos on flaky connections.
- Burned-in subtitles as an optional paid render (x264 re-encode; CPU heavy).
- Move to a dedicated domain/subdomain; dedicated system user + system units.
