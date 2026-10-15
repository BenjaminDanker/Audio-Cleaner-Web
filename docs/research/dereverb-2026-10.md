# Speech dereverberation models: research report (as of 2026-10-15)

Scope: web research only. No models were downloaded or run. Every number below comes from the cited page; anything not checked is marked **unverified**. "Not stated" means the cited page did not give the value.

Products: A = free desktop app (CPU, quality first, CC BY-NC acceptable, no-licence or research-only excluded). B = paid website, offline, commercial licence only, ~1x real time on 4 threads of an 8-core Ryzen. C = paid live streaming, causal, algorithmic latency <= ~40 ms, low CPU per stream, commercial licence only.

---

## 1. Picks

### Product A (desktop, free, quality first)

**Best pick: Sidon v0.1 (sarulab-speech).** Restoration model (noise, reverb and bandwidth), MIT, 48 kHz output. Not dereverb-only, and the paper reports no reverb-specific numbers. The one CPU run found is 0.87x real time (slower than real time) on an unnamed CPU, which is inside A's 3x budget.

**Runner-up: URGENT 2025 TF-GridNet baseline (kohei0209/tfgridnet_urgent25).** Universal enhancement that includes reverberation. CC-BY-4.0 on the HF card (unconfirmed as the official release). About 8.5M params. Offline (causality not stated).

Other A notes:
- UniPASE has the strongest published dereverb numbers in this set, but its weights have no stated licence. It becomes the A pick if the weights licence is confirmed as permissive.
- SGMSE+ EARS-Reverb has dereverb-specific numbers, but its checkpoint has no stated licence, so it is excluded under the strict rule.

### Product B (paid website, commercial licence, 1x RT on 4 threads)

**Best pick (conditional): Sidon v0.1.** Licence is MIT on the HF card. Quality is strong on restoration benchmarks. Budget risk: the only published CPU number is 0.87x real time (RTF about 1.15) on an unnamed CPU with unknown threads, so it does not meet the 1x budget as measured. Keep it only if a 4-thread benchmark on the Ryzen box reaches >= 1x real time. Only an FP16 GGUF exists, so no quantized CPU path is verified.

**Runner-up: URGENT 2025 TF-GridNet baseline (CC-BY-4.0 per HF card, about 8.5M params).** Much cheaper to run, so it is the likely budget fallback. Quality is lower than Sidon on the one shared test set (see table 2). Causality and CPU speed not verified.

### Product C (paid live streaming, causal, <= 40 ms, commercial licence)

**No pick. No candidate was verified against all constraints** (causal + dereverb + commercial licence + 48 kHz + published CPU numbers).

Closest verified options, for prototyping only:
- **nara_wpe (MIT), frame-online WPE mode.** Classical, causal-capable dereverb with a documented online mode. Latency, CPU cost and single-channel quality are **unverified**.
- **DeepFilterNet (MIT/Apache-2.0 dual code), already in use.** 48 kHz and real-time. The README does not claim dereverberation. The 2023 demo paper reports RTF 0.19 on a single-thread notebook CPU, but that is for the original DFN, not DFN3. Dereverb extensions exist in the literature (Sensors 2025), but no released code was found. Model-weights licence is not stated separately, so it is unverified.

---

## 2. Quality evidence (for the picks and runners-up)

### Sidon v0.1 (restoration, LibriTTS, from the Sidon paper)

| Set | Model | WER | SpkSim | NISQA | DNSMOS |
|---|---|---|---|---|---|
| LibriTTS test-clean | Noisy input | 0.040 | - | 4.093 | 3.179 |
| LibriTTS test-clean | Miipher (text-conditioned) | 0.047 | 0.942 | 4.688 | 3.134 |
| LibriTTS test-clean | **Sidon** | 0.045 | 0.971 | 4.790 | 3.303 |
| LibriTTS test-other | Noisy input | 0.079 | - | 3.623 | 2.949 |
| LibriTTS test-other | **Sidon** | 0.095 | 0.961 | 4.698 | 3.219 |

Source: https://arxiv.org/html/2509.17052v2. Reverberation is one of six simulated degradations (RT60 U(0.1, 2.0) s). No separate reverb-only result is reported.

### URGENT 2025 non-blind test set (from the NVIDIA RE-USE paper, Table 1 and Table 2)

Source: https://arxiv.org/html/2603.02641 (paper: arXiv 2603.02641).

| System | DNSMOS | NISQA | UTMOS | PESQ | ESTOI | SDR | CAcc |
|---|---|---|---|---|---|---|---|
| Noisy | 1.84 | 1.69 | 1.56 | 1.37 | 0.61 | 2.53 | 81.29 |
| Baseline (TF-GridNet) | 2.94 | 2.89 | 2.11 | 2.43 | 0.80 | 11.29 | 84.96 |
| Rank 1 (Bobbsun), challenge entry | 3.01 | 3.41 | 2.40 | 2.95 | 0.86 | 14.33 | 88.92 |
| RE-USE, shifted-anechoic + GAN | 3.26 | 4.12 | 2.80 | 2.38 | 0.76 | 8.18 | 89.88 |
| ClearerVoice (48 kHz subset) | 2.97 | 3.38 | 3.02 | 2.09 | 0.72 | 11.55 | 89.90 |
| Resemble Enhance (44.1 kHz subset) | 3.13 | 3.68 | 2.11 | 1.33 | 0.45 | -15.01 | 47.20 |

Notes from the same paper: the intrusive metrics are not directly comparable across training targets. RE-USE is non-causal and "not suited to real-time use." No RTF is reported.

### UniPASE (DNS 2020 with-reverb test set, Table V; source https://arxiv.org/html/2604.14606)

| Model | DNSMOS | UTMOS | PESQ | ESTOI | SpkSim | dWER % |
|---|---|---|---|---|---|---|
| Noisy | 1.39 | 1.30 | 1.16 | 0.35 | 0.70 | 10.23 |
| TF-GridNet | 2.63 | 1.42 | 1.51 | 0.49 | 0.70 | 8.86 |
| **UniPASE** | 3.33 | 3.62 | 1.74 | 0.76 | 0.79 | 8.16 |

The paper also says LLaSE-G1 and AnyEnhance get higher DNSMOS/UTMOS but have dWER of 41.66% and 14.16%. These are the authors' own numbers.

### EARS-Reverb (SGMSE+, from the EARS paper, Table 7; source https://arxiv.org/html/2406.06185v1)

| Model | PESQ | ESTOI | SI-SDR (dB) |
|---|---|---|---|
| Reverberant input | 1.48 +/- 0.37 | 0.52 +/- 0.17 | -16.17 +/- 9.77 |
| SGMSE+ | 3.03 +/- 0.67 | 0.85 +/- 0.09 | 5.79 +/- 7.97 |

EARS-Reverb is 48 kHz fullband, with 2313 RIRs. Source: same page.

---

## 3. Candidate table (all candidates checked)

Columns: task | quality evidence | rate | causal / latency | params | CPU speed | licence (code / weights) | date | links.

| # | Model | Task | Quality evidence (dataset) | Rate | Causal / latency | Params | CPU speed | Licence code / weights | Date | Links | Fit |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | **Sidon v0.1** | Restoration (noise, reverb, bandwidth); no dereverb-only claim | NISQA 4.790, DNSMOS 3.303, WER 0.045 on LibriTTS test-clean (see sec 2) | 16 kHz in (per RTF caption); 48 kHz out | Offline; GGUF page says it "processes a complete clip" | 250M (198M predictor + 52.4M vocoder) | CPU 10.53 s for 9.14 s input = 0.87x real time (CPU model and threads not stated, third-party GGUF page). GPU H200 RTF 0.00226 | MIT / MIT (HF card) | arXiv Sep 2025 (ID-derived); HF release date unverified | https://huggingface.co/sarulab-speech/sidon-v0.1 ; https://arxiv.org/html/2509.17052v2 ; https://github.com/sarulab-speech/Sidon ; https://huggingface.co/KevinAHM/Sidon-GGUF | **A pick; B conditional pick** |
| 2 | URGENT 2025 TF-GridNet baseline | Universal SE incl. reverb (1 of 7 distortions) | Non-blind: DNSMOS 2.94, NISQA 2.89, UTMOS 2.11, PESQ 2.43, ESTOI 0.80, SDR 11.29 (RE-USE Table 1). Blind-test baseline T10: DNSMOS 2.85, NISQA 2.77, PESQ 2.24 (URGENT 2025 paper, split unlabeled) | 8 to 48 kHz task; baseline rate not stated | Not stated (TF-GridNet offline by design, unverified) | ~8.5M | Not stated | Code ESPnet (licence unverified); weights cc-by-4.0 per HF card (unconfirmed as official) | Interspeech 2025 (arXiv May 2025) | https://arxiv.org/pdf/2505.23212 ; https://huggingface.co/kohei0209/tfgridnet_urgent25 | **A and B runner-up** |
| 3 | UniPASE (xiaobin-rong) | Universal SE incl. reverb | DNS 2020 with-reverb: DNSMOS 3.33, UTMOS 3.62, PESQ 1.74, ESTOI 0.76, dWER 8.16 (authors' numbers) | Input resampled to 16 kHz internally; PostNet gives 48 kHz output; URGENT rates 8 to 48 kHz | Offline only (authors) | 545.7M, 79.2 GMAC/s | RTX 4090: 44 ms for 1 s at 48 kHz, 3.2 GB; no CPU numbers | MIT code; weights: released on HF, licence not stated | arXiv Apr 2026 | https://arxiv.org/abs/2604.14606 ; https://github.com/xiaobin-rong/unipase | **A candidate, excluded until weights licence confirmed**; B/C excluded (offline, huge) |
| 4 | RE-USE (NVIDIA) | Universal SE; reverb target changed to shifted anechoic | URGENT 2025 non-blind table (sec 2) | 8 to 48 kHz | Non-causal, "not suited to real-time" (authors) | 9.6M (HF page) | Not reported | NVIDIA One-Way Noncommercial License (NSCLv1), "research and development only" | arXiv 2603.02641 (3 Mar 2026); HF 2026-03-18 | https://huggingface.co/nvidia/RE-USE ; https://arxiv.org/abs/2603.02641 | **Excluded** (research-only, no redistribution) |
| 5 | SGMSE+ EARS-Reverb checkpoint (sp-uhh) | Dereverb (EARS-Reverb); WSJ0-Reverb also listed | PESQ 3.03, ESTOI 0.85, SI-SDR 5.79 on EARS-Reverb (sec 2) | EARS-Reverb 48 kHz; repo page cites sr=16000 for one checkpoint | Non-causal diffusion; 30 steps (paper) | Not stated (repo says 27.8M for NCSN++ lighter config, paper config) | Not stated (RTF not found) | Code MIT (sgmse repo); checkpoints: **no licence stated**; EARS dataset CC BY-NC 4.0 | arXiv 2208.05830 (2022); EARS 2024 | https://github.com/sp-uhh/sgmse ; https://arxiv.org/abs/2208.05830 ; https://sp-uhh.github.io/ears_dataset/ ; https://github.com/facebookresearch/ears_dataset | **Excluded** (no weights licence); runner-up for A if licence is obtained |
| 6 | StoRM (sp-uhh) | Denoise + WSJ0-Reverb dereverb checkpoint | Not checked numerically | Not stated | Not stated | Not stated | Not stated | Code MIT; checkpoint licence not stated | Not checked | https://github.com/sp-uhh/storm | **Excluded** (no weights licence, no numbers) |
| 7 | Facebook Denoiser | Causal denoise + room reverb removal (README claim) | No quality numbers found | Not stated (16/24/48 not stated) | Causal, waveform domain | Not stated | Quad-core i5 2 GHz: RTF 0.6 (H=48, 4 threads), 0.8 (H=48, 1 thread) | **CC-BY-NC 4.0** (repo LICENSE); archived 31 Oct 2023 | arXiv 2006.12847 (2020) | https://github.com/facebookresearch/denoiser ; https://arxiv.org/pdf/2006.12847 | **A only** (non-commercial OK); B/C excluded |
| 8 | VoiceFixer (haoheliu) | Restoration incl. reverberation (README) | No dereverb numbers found | 2 kHz to 44.1 kHz; vocoder 44.1 kHz | Not stated | Not stated | Not stated | Code MIT; checkpoints: licence not stated | Not checked (2021 per general knowledge, unverified) | https://github.com/haoheliu/VoiceFixer | **Excluded** pending weights licence and numbers |
| 9 | Resemble Enhance (ResembleAI) | Denoise + enhance (distortion, bandwidth). No dereverb claim | RE-USE table: weak on reverb subset (sec 2) | 44.1 kHz training; inference rate not stated | Not stated | Not stated | Not stated | Code MIT (README) | Not checked | https://github.com/ResembleAI/resemble-enhance | **Excluded** for dereverb |
| 10 | ClearerVoice-Studio (MossFormer2_SE_48K etc.) | Speech enhancement; no dereverb listed on repo page | RE-USE 48 kHz subset numbers (sec 2) | 16 and 48 kHz (SE) | Not stated | Not stated | Not stated | Code Apache-2.0; model licence not stated | Not checked | https://github.com/modelscope/ClearerVoice-Studio | **Excluded** for dereverb; possible SE-only B candidate, unverified |
| 11 | MossFormer2 (alibabasglab) | Separation (WHAM!/WHAMR! benchmarks) | Only text claims; table is an image | 8 kHz pipeline | Not stated | Not stated | Not stated | Code MIT; pretrained licence not stated | Not checked | https://github.com/alibabasglab/MossFormer2 | **Excluded** (8 kHz, separation not dereverb) |
| 12 | FullSubNet / Improved FullSubNet (Westlake) | Denoise; no dereverb claim on repo page | None | Improved variant "48 and 24 KHz" | Not stated | Not stated | Not stated | MIT | Not checked | https://github.com/Audio-WestlakeU/FullSubNet | **Excluded** (no dereverb claim) |
| 13 | LLaSE-G1 (ASLP-lab) | NS, TSE, PLC, AEC, separation; dereverb not listed | Only UniPASE's dWER comparison | Not stated | Not stated (LLM-based) | Not stated | Not stated | Apache-2.0 per HF card and repo | arXiv Mar 2025 (2503.00493), ACL 2025 | https://arxiv.org/pdf/2503.00493 ; https://huggingface.co/ASLP-lab/LLaSE-G1 | **Excluded** (no dereverb task; LLM is not CPU-cheap) |
| 14 | DeepFilterNet / DFN3 (Rikorose) | Denoise; README makes no dereverb claim | None verified | 48 kHz | Real-time; lookahead exists (README); community CoreML card says ~40 ms latency (not authoritative) | Not stated | Original DFN: RTF 0.19 single-thread notebook CPU (2023 demo) | Code dual MIT / Apache-2.0; model licence not stated | DFN 2023 demo (arXiv 2305.08227); DFN3 date not verified | https://github.com/Rikorose/DeepFilterNet ; https://arxiv.org/abs/2305.08227 | **C closest, not a dereverb model** |
| 15 | DFN3 dereverb-related 2024 Sensors extension | Dereverb (late reverb) on DFN | Snippet only; full text not fetched (MDPI 403) | 48 kHz (DFN base) | Lightweight, edge target | Not stated | Not stated | No code link found | Feb 2025 | https://www.mdpi.com/1424-8220/25/3/630 (not fetched) | **Unverified**; no code |
| 16 | DFN3 room-acoustics retraining (Milo et al.) | Denoise only (no dereverb claim) | No numbers available | Not stated | Not stated | Same DFN3 | Not stated | Not stated | Submitted 21 Aug 2026 (IWAENC 2026) | https://arxiv.org/abs/2608.20971 | Not a dereverb model; excluded |
| 17 | GTCRN (Xiaobin-Rong) | Denoise; no dereverb claim | VCTK-DEMAND, DNS3 (not repeated) | 16 kHz (paper benchmarks; not restated) | Streaming version in repo; causal not explicitly stated | 48.2K | RTF 0.07 on i5-12400 (threads not stated); 33.0 MMAC/s | MIT (repo LICENSE) | ICASSP 2024; arXiv Feb 2025 | https://github.com/Xiaobin-Rong/gtcrn ; https://arxiv.org/pdf/2502.14224v2 | **Excluded** for dereverb; CPU budget reference only |
| 18 | LiSenNet (claroche1) | Denoise only | PESQ wideband 3.006 to 3.084 on VoiceBank-DEMAND-16k | 16 kHz | Causal; 16 ms hop; 1.02 s block graph | 36K to 46K | STM32N6570 NPU RTF 0.072 / 0.174 (no CPU) | MIT | Not checked (arXiv 2409.13285 per search, unverified) | https://huggingface.co/claroche1/LiSenNet | **Excluded** (no dereverb, 16 kHz) |
| 19 | nara_wpe (fgnt) | Classical WPE dereverb; offline, block-online and frame-online modes | No numbers in page | Not stated | Frame-online and block-online modes exist; no ms latency | Not applicable | Not stated | MIT | Released 2018 (Drude et al.) | https://github.com/fgnt/nara_wpe | **C prototype candidate** (quality and latency unverified) |
| 20 | Low-delay WPE + separation (Mo et al.) | Online separation + dereverb | Not checked numerically | Not stated | Claims delay kept near base method | Not stated | Not stated | Not stated | arXiv 2406.09821 (2024) | https://arxiv.org/pdf/2406.09821 | Not a licensed model; research only |
| 21 | Frame-online FCP / Kronecker product dereverb | Monaural frame-online dereverb | "competitive" only (abstract) | Not stated | Frame online (abstract) | Not stated | "substantially reduces computation" (no numbers) | No code link; paper CC BY 4.0 | arXiv 2510.24471 (Oct 2025) | https://arxiv.org/abs/2510.24471 | **Excluded** (no code, no numbers) |
| 22 | Binary residual masking dereverb (Williams) | Dereverb (STFT residual mask) | Not stated in abstract | Not stated | "low-latency", no ms | Not stated | Not stated | No code link | arXiv 2510.00356 (30 Sep 2025) | https://arxiv.org/abs/2510.00356 | **Excluded** (no code, no numbers) |
| 23 | Dereverb_MetricGAN-U (pleomax0730) | Dereverb (VoiceBank-reverb, 16 kHz) | PESQ 2.03 (VoiceBank), 1.74 (other set) | 16 kHz | Not stated | Not stated | Not stated | Code Apache-2.0 (SpeechBrain-based); no released weights | Not checked | https://github.com/pleomax0730/Dereverb_MetricGAN-U | **Excluded** (16 kHz, no weights) |
| 24 | URGENT 2026 challenge overview (ICASSP 2026) | Track 1 universal SE | No rankings in fetched text (site not fetched) | 8 to 48 kHz | Mixed; top systems "hybrid generative and discriminative" | Not stated | Not stated | Challenge; no system licences | arXiv 2601.13531 (Jan 2026) | https://arxiv.org/abs/2601.13531 ; https://arxiv.org/html/2601.13531v1 | Context only; rankings unverified |
| 25 | URGENT 2025 rank 1 (T1, "Bobbsun") | Universal SE | See sec 2 (RE-USE table) | Not stated | Discriminative sub-band RNN, about 102M params (URGENT 2025 paper) | ~102M | Not stated | No released code/licence found | May 2025 | https://arxiv.org/pdf/2505.23212 | **Excluded** (no released system) |
| 26 | Google patent US11404055 | Simultaneous dereverb + denoise, low latency | n/a | n/a | Low latency (claims) | n/a | n/a | Patent (legal status not assessed) | Patent page | https://patents.google.com/patent/US11404055 | **Flag for C** (FTO risk, not assessed) |
| 27 | Miipher / Miipher-2 (Google) | Restoration | Cited as comparison in Sidon paper | 24 kHz (Miipher-2) | Not stated | Not stated | Not stated | Not open-sourced (no licence found) | 2023 / 2025 | https://arxiv.org/html/2509.17052v2 | Excluded (no open weights) |
| 28 | TRU-Net (Tiny Recurrent U-Net) | Denoise + dereverb, real-time | Not checked | Not stated | Claimed online | Not stated | Not stated | Not stated | 2021 | https://paperswithcode.com/paper/real-time-denoising-and-dereverberation-with | Excluded (licence and numbers unverified) |
| 29 | PercepNet (48 kHz real-time) | Denoise | Not checked (no URL fetched) | 48 kHz | ~30 ms lookahead (search snippet; not checked) | Not stated | Not stated | Not checked | Not checked | n/a | Excluded (not checked, non-causal per snippet) |

---

## 4. Excluded models and reasons

- **RE-USE (nvidia/RE-USE):** NVIDIA One-Way Noncommercial License (NSCLv1), "for research and development only." Excluded for all three products. https://huggingface.co/nvidia/RE-USE
- **SGMSE+ EARS-Reverb checkpoint:** code is MIT, but the checkpoint and dataset terms carry no licence for the checkpoint (dataset is CC BY-NC 4.0). Strictly excluded. Non-causal, so excluded for B and C anyway. https://github.com/sp-uhh/sgmse ; https://sp-uhh.github.io/ears_dataset/
- **Facebook Denoiser:** repo LICENSE is CC-BY-NC 4.0 and the repo is archived. Eligible for A only. Note: an earlier search summary claimed MIT; the repo page shows CC-BY-NC. https://github.com/facebookresearch/denoiser
- **UniPASE:** code MIT, but the weights licence is not stated, and it is offline only with 545.7M params. Excluded for B and C. Excluded for A until the weights licence is confirmed. https://github.com/xiaobin-rong/unipase
- **StoRM, VoiceFixer:** code MIT, but no weights licence stated. VoiceFixer's dereverb claim is README-only. https://github.com/sp-uhh/storm ; https://github.com/haoheliu/VoiceFixer
- **LLaSE-G1:** Apache-2.0, but no dereverb task (NS, TSE, PLC, AEC, SS). LLM-based, not CPU-cheap. https://huggingface.co/ASLP-lab/LLaSE-G1
- **MossFormer2:** MIT code, but the released pipeline is 8 kHz separation. Not dereverb. https://github.com/alibabasglab/MossFormer2
- **ClearerVoice-Studio:** Apache-2.0 code, but its repo page does not list dereverb. Model licences not stated. https://github.com/modelscope/ClearerVoice-Studio
- **Resemble Enhance:** MIT, but no dereverb claim, and RE-USE's numbers on the reverb subset are poor (SDR -15.01, CAcc 47.20). https://github.com/ResembleAI/resemble-enhance
- **FullSubNet:** MIT, but no dereverb claim. https://github.com/Audio-WestlakeU/FullSubNet
- **GTCRN, LiSenNet:** MIT, causal and cheap (GTCRN RTF 0.07 on i5-12400), but denoise only, and LiSenNet is 16 kHz. https://github.com/Xiaobin-Rong/gtcrn ; https://huggingface.co/claroche1/LiSenNet
- **Frame-online FCP (2510.24471), binary residual masking (2510.00356):** no code and no numbers. https://arxiv.org/abs/2510.24471 ; https://arxiv.org/abs/2510.00356
- **Dereverb_MetricGAN-U:** 16 kHz, no released weights, low PESQ. https://github.com/pleomax0730/Dereverb_MetricGAN-U
- **Google patent US11404055:** low-latency simultaneous dereverb and denoise. Not a model release; flagged as a freedom-to-operate risk for C. Legal status not assessed. https://patents.google.com/patent/US11404055
- **Miipher / Miipher-2:** not open. Cited in the Sidon paper as the comparison. https://arxiv.org/html/2509.17052v2

---

## 5. Gaps and next steps (unverified items)

1. **Sidon CPU speed.** The only CPU number is from a third-party GGUF page (0.87x real time, CPU model and threads not stated). Benchmark on the Ryzen 8-core at 4 threads before committing to B. https://huggingface.co/KevinAHM/Sidon-GGUF
2. **Sidon reverb-specific quality.** Not published. Ask for or measure a reverb-only test (e.g. EARS-Reverb or DNS with-reverb) before claiming dereverb. https://arxiv.org/html/2509.17052v2
3. **URGENT 2025 TF-GridNet baseline.** Confirm the official licence and the sample rate and causality. https://huggingface.co/kohei0209/tfgridnet_urgent25
4. **Causal dereverb with commercial licence (C).** No candidate found in this session. Options: build on nara_wpe frame-online WPE (MIT) and measure its latency and quality, or search further for 2025 to 2026 causal 48 kHz dereverb releases (the search did not surface any).
5. **DFN3 dereverb.** Upstream README does not claim it. Check the DFN3 model licence and any dereverb training branch. https://github.com/Rikorose/DeepFilterNet
6. **URGENT 2026 rankings and team systems.** Not fetched (challenge site not reached). https://arxiv.org/html/2601.13531v1

Note on sources: MDPI (mdpi.com) returned 403 on fetch, so the Sensors 2025 numbers were not read. Official challenge leaderboard pages were not fetched.
