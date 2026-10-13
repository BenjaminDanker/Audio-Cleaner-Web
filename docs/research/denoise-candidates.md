# Speech denoise / enhancement candidates (snapshot 2026-10-13)

Compiled from web search and page reads only. Nothing was installed or run.

Conventions:
- "not stated" means the page I read did not give the value. Treat it as unknown, not as permissive.
- "unverified" means the item appeared only in a search snippet or a third-party page.
- Every licence cell cites a source ID [S#]. The URLs are listed at the end.

## 0. Read first

- **Code licence is not weights licence.** Many repos say "MIT" with nothing separate for the checkpoints. Both are listed. Where neither is stated, the weights are unclear, not permissive.
- **No offline (A) candidate has a measured real-time factor on the Ryzen 1700X or the GTX 960.** The compute figures below are only what the sources state. The ≤0.5x target needs a benchmark.
- **DeepFilterNet3-LL does not appear in any source I found.** The nearest real variant is DeepFilterNet3 (48 kHz, causal). Sources disagree on its latency (see section 2).
- **Conflicts noted in the sources:**
  - DeepFilterNet3 latency: 20 ms lookahead [S46], 30 ms [S48], ~40 ms [S47].
  - GTCRN parameters: 23.7K in the original paper, 48.2K in the updated README [S6].
  - DNSMOS: the GitHub page read still showed a DNSMOS folder [S33], but a mirror commit titled "Delete DNSMOS/" (29 Mar 2024) turned up in search (unverified; see section 6).

## 1. (A) Offline "studio" candidates

| Name | Year | Task | Sample rate | Causal? / latency | Params / compute | Code licence | WEIGHTS licence | Weights location | Runtime | Reported quality | Notes |
|---|---|---|---|---|---|---|---|---|---|---|---|
| ClearerVoice MossFormer2_SE_48K | 2025 [S19] | denoise | 48 kHz | No (offline; not stated) | not stated | Apache-2.0 [S1] | Apache-2.0 (HF tag) [S2]. Authors say training used open and private data [S2]. | HF alibabasglab/MossFormer2_SE_48K [S2]; ModelScope iic/ClearerVoice-Studio [S1] | PyTorch via ClearVoice package [S2]; no ONNX mentioned | VB-DMD 48k: PESQ 3.15, CSIG 4.23, STOI 0.95, SI-SDR 19.36. DNS2020 16k: PESQ 2.94, STOI 0.97 [S1] | Authors flag LLR/LSD anomalies at 48k. No DNSMOS/NISQA reported. |
| ClearerVoice MossFormerGAN_SE_16K | 2025 [S19] | denoise | 16 kHz | No (not stated) | not stated | Apache-2.0 [S1] | Apache-2.0 (HF tag) [S3] | HF alibabasglab/MossFormerGAN_SE_16K [S3] | PyTorch [S3] | VB-DMD 16k: PESQ 3.47, CSIG 4.40, STOI 0.96, SI-SDR 19.45. DNS2020 16k: PESQ 3.57, STOI 0.98 [S1] | GAN output. Hallucination not measured. Needs an upsampler for 48k delivery. |
| FRCRN (ClearVoice, FRCRN_SE_16K) | 2022 paper [S20]; in ClearVoice 2025 [S1] | denoise | 16 kHz model; ClearVoice also offers 48 kHz output [S1] | not stated | not stated | Apache-2.0 (ClearVoice repo) [S1] | not stated separately [S1] | ModelScope iic; HF alibabasglab (unverified) | PyTorch | not verified (one aggregator lists PESQ-WB 3.23, unverified) | Same toolkit as MossFormer. |
| Resemble Enhance | 2024 (not verified) | denoise + enhancement (quality and bandwidth restoration) | Trained on 44.1 kHz data; inference rate not stated [S4] | No (offline CLI; generative) | not stated | MIT [S4] | MIT (HF tag) [S5] | HF ResembleAI/resemble-enhance [S5]; HF Space demo [S4] | PyTorch [S4] | none published [S4][S5] | Two stages: denoiser, then latent conditional flow-matching enhancer [S4]. VRAM and RTF not stated. |
| MP-SENet | 2023 Interspeech; 2025 Neural Networks [S8] | denoise | 16 kHz (VB-DEMAND setup) [S8] | No (not stated) | not stated | MIT [S8] | not stated separately; checkpoint in best_ckpt/ in repo [S8] | GitHub repo [S8] | PyTorch [S8] | VB-DEMAND table is an image; not extracted | Parallel magnitude and phase estimation. |
| SEMamba | 2024 [S50] | denoise | 16 kHz (VB-DEMAND checkpoint) [S9] | not stated | not stated | **No licence stated** [S9] | not stated; ckpts/vd.pth trained on VCTK-Demand + DNS-2020 [S9] | GitHub repo [S9] | PyTorch [S9] | VB-DEMAND: PESQ 3.56, STOI 0.96; with PCS PESQ 3.75 [S9] | Do not ship without a written licence from the author. |
| StoRM | 2023 TASLP [S10] | denoise + dereverb (diffusion stochastic regeneration) | 16 kHz likely; not stated on page | No (diffusion, multi-step) | NCSN++ 27.8M params per repo [S10] | MIT [S10] | not stated [S10] | GitHub + Google Drive checkpoints [S10] | PyTorch [S10] | not extracted | Multi-step diffusion; unlikely to meet 0.5x on the 1700X. |
| SGMSE+ | 2022 Interspeech [S11] (before the window; included as baseline) | denoise + dereverb (score-based diffusion) | 16 kHz configs; 48 kHz EARS checkpoints [S11] | No (not stated) | not stated | MIT [S11] | not stated [S11] | GitHub sp-uhh/sgmse [S11] | PyTorch [S11] | not extracted | Diffusion, multi-step. |
| VoiceFixer | 2021 arXiv [S12] (before the window) | general restoration: denoise, dereverb, declip, low-resolution (bandwidth) | input 2 to 44.1 kHz; vocoder 44.1 kHz [S12] | No (not stated) | not stated | MIT [S12] | not stated; downloaded on first run [S12] | GitHub; CPU Docker image [S12] | PyTorch; CPU path (cuda=False) [S12] | none on page | Runs on CPU. Quality numbers not on page. Old. |
| Apollo | 2024 arXiv; ICASSP 2025 [S13] | music / general audio restoration of lossy-coded audio | operating rate not stated; GPU timing quoted at 44.1 kHz [S13] | not stated | not stated | CC BY-SA 4.0 [S13] | not stated [S13] | HF JusperLee/Apollo [S13] | PyTorch | not extracted | Music-focused. CC BY-SA on code is an awkward fit for a product. |
| LLaSE-G1 | 2025, ACL 2025 (arXiv 2503.00493) [S15] | general SE: denoise, target speaker extraction, PLC, AEC, separation | not stated | not stated | not stated. LLaMA-based; WavLM features in, X-Codec2 tokens out [S15] | not verified (repo linked from card: Kevin-naticl/LLaSE-G1 [S52]) | Apache-2.0 (HF tag and card) [S15] | HF ASLP-lab/LLaSE-G1 [S15] | PyTorch [S15] | none on the card; refers to paper | Card warns task-type selection can be unstable [S15]. An LLM this size is unlikely to fit a 4 GB GTX 960 (estimate). |
| FlowSE (Honee-W; arXiv 2505.19476) | 2025 [S18] | denoise (generative; optional transcript) | not stated; folder name suggests 24 kHz Vocos mel (unverified) [S14] | not stated | not stated | **none stated** [S14] | **none stated**; HF flowse/wenetspeech4tts_Premium.pt.tar [S14] | GitHub Honee-W/FlowSE [S14] | PyTorch | none | Unclear. Do not ship. A different "FlowSE" (arXiv 2508.06840) has no code found [S60]. |
| MeanFlowSE (MeanSE repo) | 2025 [S59] | denoise (one-step generative) | not stated | not stated | not stated | **none stated** [S26] | none; no checkpoints in repo [S26] | GitHub Twinkzzzzz/MeanSE [S26] | PyTorch | none extracted | Not usable yet. |
| GenSE | 2025 (arXiv 2502.02942) [S16] | general SE (LM over codec tokens) | not stated | not stated | not stated | not found | not found | paper says code is public; link not verified [S16] | not found | none extracted | Unclear. |
| AnyEnhance | 2025 (arXiv 2501.15417) [S17] | universal: denoise, dereverb, declip, SR, target speaker extraction (speech and singing) | not stated | not stated | not stated | no code found | no weights found | demo page only [S17] | none | none extracted | Do not rely on it. |
| Sidon | 2025 (arXiv 2509.17052) [S25] | general speech restoration, multilingual; built for dataset cleaning | 48 kHz output [S25] | not stated; feature predictor + vocoder, offline | 250M total (198M w2v-BERT 2.0 feature predictor + 52.4M vocoder); RTF 0.0023 on H200 bf16 at batch 1 [S25] | not stated [S25] | not stated [S25] | project page hf.co/spaces/Wataru/SidonSamples [S25] | PyTorch (unverified) | LibriTTS test-clean: NISQA 4.79, DNSMOS 3.30. FLEURS: DNSMOS 3.39 [S25] | Strongest open restoration candidate in this set, but no licence. CPU and GTX 960 speed not measured. |
| AudioSR | 2023 (arXiv 2309.07314) [S28] | bandwidth extension (2 to 16 kHz to 48 kHz), general audio | 48 kHz output [S28] | No (diffusion) | not stated; third-party mirror suggests 50 to 100 steps [S29] | MIT per third-party mirror; unverified [S29] | not stated; only third-party mirror weights found [S29] | project page audioldm.github.io/audiosr [S28] | PyTorch | none extracted | Use only as a bandwidth-extension stage. Slow. |
| NVIDIA Maxine Audio Effects SDK | SDK; 2021 licence text [S30] | denoise (product) | not stated | not stated | not stated | proprietary EULA [S30] | proprietary; NGC-gated; forum staff say NVAIE licence is required [S31] | NGC catalog [S31] | proprietary SDK | none | Not open. Commercial use appears to need NVAIE; current EULA not fetched. |
| Miipher-2 | 2025 (arXiv 2505.04457) [S32] | universal restoration (Google) | not stated | not stated | RTF 0.0078 on consumer accelerators (paper) [S32] | n/a | **not released**; authors state code and checkpoints will not be released [S32] | none | none | paper-only metrics | Not usable. Sidon is the open substitute cited [S25]. |
| URGENT 2025 top systems (TS-URGENet, USEMamba) | 2025 [S39][S62] | universal (7 distortion types) | not stated | not stated | not stated | not verified | not verified | none found | none | not extracted | Released weights not confirmed for the top systems. |

## 2. (B) Real-time causal candidates (≤ ~40 ms algorithmic latency)

| Name | Year | Task | Sample rate | Causal? / latency | Params / compute | Code licence | WEIGHTS licence | Weights location | Runtime | Reported quality | Notes |
|---|---|---|---|---|---|---|---|---|---|---|---|
| DeepFilterNet3 (baseline) | 2023 [S7] | denoise | 48 kHz [S7] | Yes (causal STFT with lookahead) | Latency sources disagree: 20 ms lookahead [S46], 30 ms [S48], ~40 ms [S47] | 2.1M params [S46] | dual MIT or Apache-2.0, "all code" [S7]. Weights not stated separately. | GitHub Rikorose/DeepFilterNet (models folder) [S7] | PyTorch, Rust binary, LADSPA [S7]. ONNX only via third-party mirrors. | DNS4 blind DNSMOS OVRL 2.958 (as reported by DPDFNet paper) [S24] | Incumbent and fallback. Not an upgrade. |
| GTCRN | 2024 (ICASSP 2024) [S6] | denoise | 16 kHz per paper; README does not state it [S6] | Yes (streaming version in stream/); algorithmic latency not stated [S6] | 48.2K params (updated; original 23.7K) [S6]; 33.0 MMACs/s [S6]; streaming RTF 0.07 on i5-12400 CPU [S6] | MIT [S6] | not stated separately; checkpoints bundled in repo [S6] | GitHub Xiaobin-Rong/gtcrn [S6] | PyTorch; ONNX not in repo (third-party sherpa-onnx) [S6] | VCTK-DEMAND: PESQ 2.87, STOI 0.940, SI-SNR 18.83 [S6]. DNS3 blind DNSMOS OVRL reported as 2.70; column alignment uncertain [S6] | Cheapest viable model. 16 kHz only, so 48 kHz streams need resampling. |
| DPDFNet (Ceva) | 2025 (arXiv 2512.16420) [S24] | denoise | 16 kHz (baseline, dpdfnet2/4/8); 8 kHz variants; 48 kHz HR variants [S23] | Yes (causal dual-path RNN) [S24]. Algorithmic latency not stated. STFT 20 ms window, 10 ms hop, DF look-ahead 2 frames [S24] | 2.31 to 3.63M params; 0.36 to 7.17 GMACs across variants. 48 kHz HR dpdfnet2: 2.58M, 2.42 GMACs [S23]. MAC basis (per second or per frame) not stated [S24] | Apache-2.0 [S23] | **not stated**; models hosted on HF, not bundled [S23] | GitHub ceva-ip/DPDFNet [S23]; PyTorch .pth, ONNX (CPU path), TFLite [S23] | DNS4 blind: DNSMOS OVRL 3.105 (DF2 3.019, DF3 2.958), SIG 3.389, BAK 4.047. Multilingual low-SNR: PESQ 2.82 vs DF3 2.45 [S24] | Vendor-reported. Targets NPU deployment. Best medium-tier candidate. |
| ULCNet / Fast-ULCNet | 2023 (arXiv 2312.08132) [S21]; Fast-ULCNet 2026 (arXiv 2601.14925) [S22] | denoise | 16 kHz [S22] | Causality not stated; STFT 32 ms window, 16 ms hop [S22] | ULCNet 0.685M params, 2.06M MACs; Fast-ULCNet 0.338M, 1.69M MACs [S22] | **not stated** (code at narrietal/Fast-ULCNet [S49]) | not stated | GitHub [S49] | not stated | DNS2020 10 s synthetic: ULCNet OVRL 3.10, PESQ 2.62; Fast-ULCNet OVRL 3.09, PESQ 2.51 [S22] | Cheapest tier. Paper gives RTF on a Raspberry Pi 3B+ and Cortex-A53 [S22]. |
| UL-UNAS | 2025 (TASLP accepted) [S37] | denoise | not stated | Partly: causal time-frequency attention. Full causality not stated [S37] | not stated | **not stated** [S37] | not stated | GitHub Xiaobin-Rong/ul-unas [S37] | not stated | not extracted; claims to beat ultra-light models at equal or lower complexity [S37] | Check licence before use. |
| FullSubNet / FullSubNet+ | 2021 / 2022 [S41][S42] | denoise | 16 kHz (unverified) | not verified | not verified | **not found** [S41][S42] | checkpoints in repo (unverified) | GitHub [S41][S42] | PyTorch | not extracted | Outside the window. Winner status in DNS not verified. |

Out of window and not shortlisted: the Facebook Denoiser dns64 (2020) is causal at 16 kHz, but its repo was archived in Oct 2023 [S57].

## 3. Shortlists (commercially usable, based on verified licences)

### (A) Offline studio: best 3

1. **ClearerVoice MossFormer2_SE_48K.** This is the only native-48 kHz candidate with an explicit Apache-2.0 licence on both code and weights [S1][S2]. Quality is middling (VB-DEMAND 48k PESQ 3.15) [S1]. The authors say training included private data [S2], so get a written provenance statement before commercial use. Compute is not stated, so benchmark it on the Ryzen 1700X first.
2. **ClearerVoice MossFormerGAN_SE_16K.** Apache-2.0 on code and weights [S1][S3]. It has the best reported metrics in this set (VB-DEMAND 16k PESQ 3.47; DNS2020 PESQ 3.57) [S1]. Two limits: it is 16 kHz only, so a 48 kHz output needs an upsampler whose licence is also clean, and GAN hallucination was not measured.
3. **Resemble Enhance.** MIT code and MIT weights [S4][S5]. It pairs a denoiser with a generative enhancer, which suits general restoration better than a pure denoiser. Weaknesses: no published metrics [S4][S5], no VRAM or RTF figures, and generative hallucination risk. It is probably GPU-bound, but that is unverified.

Not shortlisted despite decent quality, because the weights licence is missing or restrictive: Sidon (no licence stated [S25]), StoRM and SGMSE+ (weights not separately licensed; diffusion is too slow), VoiceFixer (weights unclear, but the most CPU-friendly option if its weights licence is confirmed [S12]), SEMamba (no licence [S9]), Apollo (CC BY-SA on code, music-focused [S13]).

### (B) Real-time causal: best 3

1. **DPDFNet (Ceva).** Apache-2.0 code [S23]. It is causal [S24] and reports higher DNSMOS than DF2 and DF3 on DNS4 (OVRL 3.105 vs 3.019 and 2.958) [S24]. It ships ONNX and TFLite [S23], and there is a 48 kHz high-resolution variant [S23]. Caveats: the weights licence is not stated [S23], the numbers are vendor-reported, and algorithmic latency is not stated. Get the weights licence in writing before use.
2. **GTCRN.** MIT code, with checkpoints bundled in the repo [S6]. It is very cheap (48.2K params, 33 MMACs/s, streaming RTF 0.07 on an i5-12400) [S6]. Caveats: it is 16 kHz only, its quality is below DPDFNet's, and the weights licence is just the repo's MIT.
3. **DeepFilterNet3.** Dual MIT or Apache-2.0 [S7], 48 kHz, causal, and already in production. It is the baseline and fallback, not an upgrade.

Only two candidates clear the bar on verified licences. Fast-ULCNet [S22] and UL-UNAS [S37] are cheap and causal-leaning, but their licences are not stated.

## 4. Evaluation datasets (clean+noisy pairs)

| Dataset | Sample rate | Pairs | Licence | Download | Notes |
|---|---|---|---|---|---|
| VoiceBank-DEMAND (Edinburgh DataShare 10283/2791) | 48 kHz original [S34]; 16 kHz derivative is common (unverified) | Parallel clean/noisy: 28- and 56-speaker train sets plus test set [S34] | DataShare page lists "End-user Licence"; text not captured [S34]. CC BY 4.0 from memory, **unconfirmed**. | https://datashare.ed.ac.uk/download/DS_10283_2791.zip [S34] | Confirm the licence on the record before commercial use. |
| DNS Challenge datasets (microsoft/DNS-Challenge) | 16 kHz wideband and 48 kHz fullband variants (unverified in repo) | Synthetic clean/noisy test sets plus a blind test set | Code MIT [S33]. Datasets "under the original terms that Microsoft received" [S33]. Sources: VCTK ODC-By 1.0; DEMAND CC BY-SA 3.0; Freesound CC0 only; VoxCeleb CC BY 4.0; AudioSet CC BY 4.0; LibriVox public domain [S33]. Test-set terms **not stated**. | Repo download scripts; aka.ms/dns-challenge [S33] | Treat the test-set licence as unclear for commercial work. |
| EARS (facebookresearch/ears_dataset; project page by sp-uhh) | 48 kHz anechoic [S43] | About 100 h of clean speech from 107 speakers. Noisy pairs must be made by mixing [S43]. | Repo says "CC-NC 4.0 International license" [S43]. Non-commercial. | Per-speaker zips in GitHub releases [S43] | Research only. Check with counsel before using it in a commercial benchmark. |
| URGENT 2024 / 2025 / 2026 data | Multi-rate [S38][S39] | Simulated distortions over public corpora [S38] | Not verified; each source keeps its own licence | Challenge sites (not verified) | Overview papers: [S38], [S39], [S40]. |

## 5. Non-intrusive metric models

| Model | Measures | Code licence | Weights licence | Notes |
|---|---|---|---|---|
| DNSMOS P.835 (Microsoft) | SIG, BAK, OVRL (and P.808) | MIT [S33] | **Not stated.** Repo conflict: the GitHub page showed a DNSMOS folder [S33], but a mirror commit "Delete DNSMOS/" (29 Mar 2024) turned up in search (unverified). A third-party torch conversion has MIT code and CC BY 4.0 converted weights [S45]. | Wrapper speechmos is archived [S44]. Azure hosting terms may govern commercial use (unverified). Treat as unclear. |
| UTMOS (UTMOS22 strong learner) | MOS naturalness (speech) | MIT [S36] | **Not stated** [S36] | Downloads fairseq checkpoints via script [S36]. Input rate not stated. Best option if the weights terms are confirmed. |
| NISQA v2 | MOS, NOI, COL, DIS, LOUD | MIT [S35] | **CC BY-NC-SA 4.0 [S35]. Not commercial.** | Do not use weights in a paid product. |

PESQ, POLQA, ViSQOL and torchaudio SQUIM were not checked for licence.

## 6. Not covered and next steps

- Found but not checked: MambAttention (arXiv 2507.00966) [search]; TIGER; ZipEnhancer; LiSenNet. Their licences and metrics are unknown.
- NoLACE (arXiv 2309.14521) is a codec post-filter for Opus, not a general denoiser [S27]. Its licence was not verified.
- Benchmark plan for the 0.5x target: 10 s clips, PyTorch and ONNX, on the Ryzen 1700X (1 thread and 8 threads) and on the GTX 960 in FP32. Candidates in order: GTCRN, DPDFNet, MossFormer2_SE_48K, MossFormerGAN_SE_16K, Resemble Enhance.
- Ask upstream in writing about weights terms for DPDFNet, MossFormer and Resemble Enhance (the last one is MIT-tagged [S5]), and about dataset provenance for MossFormer2, since it used private data [S2].

## Sources

- [S1] https://github.com/modelscope/ClearerVoice-Studio (Apache-2.0; model list and quality tables)
- [S2] https://huggingface.co/alibabasglab/MossFormer2_SE_48K (licence tag apache-2.0)
- [S3] https://huggingface.co/alibabasglab/MossFormerGAN_SE_16K (licence tag apache-2.0)
- [S4] https://github.com/resemble-ai/resemble-enhance (MIT, sidebar)
- [S5] https://huggingface.co/ResembleAI/resemble-enhance (licence tag mit)
- [S6] https://github.com/Xiaobin-Rong/gtcrn (MIT; params, MMACs, RTF, PESQ)
- [S7] https://github.com/Rikorose/DeepFilterNet (MIT or Apache-2.0)
- [S8] https://github.com/yxlu-0102/MP-SENet (MIT, LICENSE file listed)
- [S9] https://github.com/RoyChao19477/SEMamba (no licence stated)
- [S10] https://github.com/sp-uhh/storm (MIT)
- [S11] https://github.com/sp-uhh/sgmse (MIT)
- [S12] https://github.com/haoheliu/voicefixer (MIT)
- [S13] https://github.com/JusperLee/Apollo (CC BY-SA 4.0)
- [S14] https://github.com/Honee-W/FlowSE (no code licence)
- [S15] https://huggingface.co/ASLP-lab/LLaSE-G1 (Apache-2.0)
- [S16] https://arxiv.org/abs/2502.02942 (GenSE; no licence found)
- [S17] https://arxiv.org/abs/2501.15417 (AnyEnhance; no code found)
- [S18] https://arxiv.org/abs/2505.19476 (FlowSE paper)
- [S19] https://arxiv.org/abs/2506.19398 (ClearerVoice paper, Interspeech 2025)
- [S20] https://arxiv.org/abs/2206.07293 (FRCRN paper)
- [S21] https://arxiv.org/abs/2312.08132 (ULCNet paper)
- [S22] https://arxiv.org/html/2601.14925v1 (Fast-ULCNet; code linked)
- [S23] https://github.com/ceva-ip/DPDFNet (Apache-2.0; model table)
- [S24] https://arxiv.org/html/2512.16420v1 (DPDFNet paper)
- [S25] https://arxiv.org/html/2509.17052v2 (Sidon paper)
- [S26] https://github.com/Twinkzzzzz/MeanSE (no licence; no checkpoints)
- [S27] https://arxiv.org/abs/2309.14521 (NoLACE)
- [S28] https://arxiv.org/abs/2309.07314 (AudioSR; project page audioldm.github.io/audiosr)
- [S29] https://huggingface.co/drbaph/AudioSR (third-party mirror; MIT claim unverified)
- [S30] https://developer.download.nvidia.com/licenses/Maxine_SDK_License_1Apr2021_updated.pdf
- [S31] https://forums.developer.nvidia.com/t/impossible-to-download-maxine-audio-effects-sdk/343193
- [S32] https://arxiv.org/abs/2505.04457 (Miipher-2; release statement)
- [S33] https://github.com/microsoft/DNS-Challenge (MIT code; dataset terms)
- [S34] https://datashare.ed.ac.uk/handle/10283/2791 (VoiceBank-DEMAND)
- [S35] https://github.com/gabrielmittag/NISQA (MIT code; CC BY-NC-SA 4.0 weights)
- [S36] https://github.com/sarulab-speech/UTMOS22 (MIT code)
- [S37] https://arxiv.org/abs/2503.00340 (UL-UNAS; code https://github.com/Xiaobin-Rong/ul-unas)
- [S38] https://arxiv.org/abs/2406.04660 (URGENT 2024)
- [S39] https://arxiv.org/abs/2505.23212 (URGENT 2025)
- [S40] https://arxiv.org/abs/2601.13531 (URGENT 2026 overview)
- [S41] https://github.com/haoxiangsnr/FullSubNet (licence not found)
- [S42] https://github.com/hit-thusz-RookieCJ/FullSubNet-plus (licence not found)
- [S43] https://github.com/facebookresearch/ears_dataset (CC-NC 4.0 International)
- [S44] https://pypi.org/project/speechmos/ (archived DNSMOS wrapper)
- [S45] https://huggingface.co/prj-beatrice/dnsmos-torch-native (third-party conversion; MIT code, CC BY 4.0 weights)
- [S46] https://huggingface.co/aufklarer/DeepFilterNet3-MLX (2.1M params; 20 ms lookahead)
- [S47] https://huggingface.co/thoratsr7/deepfilternet3-onnx (ONNX mirror; ~40 ms claim per search summary)
- [S48] https://huggingface.co/iky1e/DeepFilterNet3-Streaming-CoreML (30 ms delay)
- [S49] https://github.com/narrietal/Fast-ULCNet (code; licence not checked)
- [S50] https://arxiv.org/abs/2405.06573 (SEMamba paper)
- [S52] https://github.com/Kevin-naticl/LLaSE-G1 (linked from HF card; licence not checked)
- [S57] https://github.com/facebookresearch/denoiser (archived Oct 2023)
- [S59] https://arxiv.org/abs/2509.21214 (MeanSE paper)
- [S60] https://arxiv.org/abs/2508.06840 (separate FlowSE paper; no code found)
- [S62] https://arxiv.org/abs/2505.21198 (USEMamba, URGENT 2025)
