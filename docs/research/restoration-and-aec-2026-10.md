# Speech restoration, noise suppression and neural AEC: candidate survey (as of 2026-10-15)

Scope: web research only (papers, model cards, repos, leaderboards). Nothing was downloaded or run. Every number below comes from the cited page. "unverified" means I could not check it: the page was not fetched, returned 401/404, or did not state the item. Licences are reported for CODE and WEIGHTS separately because they often differ.

Product constraints used for the picks:
- A: desktop, free non-commercial distribution OK (CC BY-NC OK), quality first, CPU only, ~3x real time.
- B: paid website, CPU, commercial licence only, ~1x real time on 4 threads.
- C: live, commercial licence only, causal, algorithmic latency <= 40 ms, ~0.11 core per stream.

---

## 1. Picks

### 1a. Noise suppression

| Product | Best pick | Runner-up | Why |
|---|---|---|---|
| A | **DPDFNet-8** (Ceva), 48 kHz-HR variant preferred, 16 kHz variant otherwise | MP-SENet (16 kHz, MIT per secondary repack) | DPDFNet-8 has the best published numbers of the causal DFN-family candidates against DFN3 (16 kHz, authors' multilingual low-SNR set). Code and HF weights are Apache-2.0. The 48 kHz-HR variants have no published quality numbers yet (unverified). |
| B | **DPDFNet-8** (48 kHz-HR), same licence | DPDFNet-4 (48 kHz-HR) if DPDFNet-8 misses the CPU budget | Cost is 4.37 GMACs/s for DPDFNet-8 at 16 kHz vs 0.35 GMACs/s for DFN3, so it should fit a 1x-real-time, 4-thread budget (my estimate, not benchmarked). |
| C | **Keep DeepFilterNet3** | DPDFNet-2 (48 kHz-HR), only if it benchmarks within the 0.11-core budget | No verified candidate beats DFN3 within the C budget. DPDFNet-2 costs about 3.9x DFN3's MACs at 16 kHz, so it will probably exceed 0.11 core unless it is optimised (estimate). |

### 1b. All-in-one restoration (noise + reverb + bandwidth + clipping/codec)

| Product | Pick | Status |
|---|---|---|
| A | **UniPASE** (conditional) | Best published numbers found, but the weights licence is not stated. Use only after the licence is confirmed. Runner-up: none qualifies (see Section 4). |
| B | **None verified** | UniPASE (code MIT, weights unknown, 545.7M params, 79.2 GMACs/s) is the closest candidate, but CPU real-time is unbenchmarked and the weights licence is unknown. Sidon (MIT, 250M params) is a second candidate, but its own tables show restored WER above noisy WER. |
| C | **None** | No candidate is causal, within 40 ms and within CPU budget with a verified licence. |

### 1c. Neural acoustic echo cancellation (AEC)

- **C (live):** Keep WebRTC AEC3. No neural AEC I checked has a published head-to-head against AEC3 on a shared benchmark, so I cannot say a neural model beats it. The strongest candidate to bake off as a residual stage is **LocalVQE v1.4-AEC** (code Apache-2.0, 203K params, 16 ms hop, published CPU timings). Its weights licence is not stated, so it cannot be used commercially until that is clarified.
- **A and B (files with a reference track):**
  - A: **LocalVQE v1.3 (joint AEC + NS, 4.8M params)** is a candidate. Its weights licence is unstated, so it is pending the licence check.
  - B: blocked until the LocalVQE weights licence is confirmed.
  - Original DeepVQE weights were not found. NKF-AEC (linear, 5.3K params) has no stated licence for its code or weights.

---

## 2. Neural AEC: is anything worth adding on top of AEC3?

**Verdict: not verified. Nothing I checked has a shared-benchmark result against WebRTC AEC3, so I cannot recommend adding a neural stage to C.**

Evidence:
- **WebRTC AEC3 licence is verified.** The LICENSE file is 3-clause BSD ("Copyright (c) 2011, The WebRTC project authors"). Source: https://webrtc.googlesource.com/src/+/refs/heads/main/LICENSE
- **AEC3 ERLE figures in the literature are not comparable.** They come from different test sets. One paper reports about 17.4 dB ERLE for AEC3 on its own blind set, vs about 32 dB for DTLN-AEC (https://arxiv.org/pdf/2106.07577). Another reports about 30 dB for AEC3 on far-end speech (https://arxiv.org/pdf/2009.13931). A third reports 40 dB for WebRTC (https://arxiv.org/pdf/2005.09237). These numbers were not measured on a shared benchmark, so they cannot support a ranking.
- **The ICASSP 2023 AEC challenge paper has no AEC3 baseline.** Its only baseline is a GRU model (1.3M params). In its challenge table, the Microsoft entry MS-1 scored 0.856 vs 0.854 for the 1st-place entry (ByteAudio). Its loopback-based recording setup is relevant to C (https://arxiv.org/html/2309.12553). The challenge repository's dataset licences are mixed (CC BY 4.0, CC0, CC BY-SA 3.0, public domain) (https://github.com/microsoft/AEC-Challenge).

Neural AEC candidates (all need the far-end reference):

| Model | Task | Params / cost | Published result | Licence (code / weights) | Verdict |
|---|---|---|---|---|---|
| **LocalVQE v1.4-AEC** | AEC (causal, 16 kHz) | 203K params; 0.83 ms per 16 ms hop on 4 threads (19x RT, Ryzen 9 7900); single-thread 2.7K front-end at 0.36 ms (44x RT) | ICASSP 2022 AEC blind set (real recordings): far-end single-talk ERLE 14.6 / 11.1 dB; near-end echo MOS 4.99 vs 5.00 unprocessed. Not compared with AEC3 | Apache-2.0 / **not stated** | C: bake-off candidate. A/B: pending weights licence |
| **LocalVQE v1.3 (joint)** | AEC + NS (causal, 16 kHz) | 4.8M params; 3.2 ms per 16 ms hop on 4 threads (5x RT) | ERLE 50.9 / 49.9 dB (far-end single-talk, ICASSP 2022 blind set) | Apache-2.0 / **not stated** | A: candidate, pending weights licence. B: blocked. C: too heavy (estimate) |
| **NKF-AEC** (Tencent) | Linear AEC (causal, 16 kHz) | 5.3K params; RTF 0.09 on one CPU core | Echo MOS 4.02 on the AEC challenge blind set; beats linear baselines by at least 1.13 AECMOS-echo | Not stated / weights in `src/` | Cheap linear baseline; not a neural residual stage. Licence unverified |
| **EchoFree** (NWPU) | Neural post-filter after linear AEC | 0.28M params, 30M MACs/s | AECMOS on ICASSP 2023 blind set (table partly garbled in the extract) | no code or weights stated | Excluded: no release found |
| **MSA-EchoLite** (Aug 2026) | AEC with PBFDAF reference | 0.20M params; RTF 1.41% on one Xeon thread | EchoMOS higher than EchoFree in all four scenarios; DegMOS lower in all four | no code or weights stated | Excluded: no release found |
| **DeepVQE** (Microsoft, 2023) | AEC + NS + DR | Reported as real-time; Teams deployment per authors | SOTA on ICASSP 2023 AEC/DNS non-personalised tracks (per the paper) | No official weights found | Only LocalVQE's derivative is available |
| **Align-CRUSE** (2022) | Cross-attention alignment AEC | Not checked in detail | Deployed in Microsoft Teams per paper | Not checked | Excluded: no released weights found. (No "Align-CRUN" paper found; the name in the brief appears to be Align-CRUSE.) |

Sources: LocalVQE: https://github.com/localai-org/LocalVQE (weights listed as https://huggingface.co/LocalAI-io/LocalVQE). LocalVQE's README states a far-end reference is required. The weights page on huggingface.co/ResembleAI/LocalVQE returned 401, so the weights licence is unverified. NKF-AEC: https://arxiv.org/html/2207.11388 and https://github.com/fjiang9/NKF-AEC (README gives 16 kHz, no RTF numbers). EchoFree: https://arxiv.org/html/2508.06271v1. MSA-EchoLite: https://arxiv.org/html/2608.03650. DeepVQE: https://arxiv.org/abs/2306.03177. Align-CRUSE: https://arxiv.org/abs/2208.11308. Align-ULCNet (0.69M params, 0.10 GMACS; no code found): https://arxiv.org/abs/2410.13620.

The E2E-AEC preprint (Jan 2026) also appeared in search, but I could not retrieve a URL or code for it, so it is not assessed.

---

## 3. Candidate table

Key: P = published, U = unverified. "Causal" means verified from the paper or repo; otherwise U.

### 3a. Noise suppression and speech enhancement

| Model | Date | Tasks | Quality numbers (dataset) | SR | Causal / latency | Params / cost | Licence code / weights | Links |
|---|---|---|---|---|---|---|---|---|
| **DeepFilterNet3** (baseline) | 2023 | NS | VoiceBank+DEMAND: PESQ 3.17, CSIG 4.34, CBAK 3.61, COVL 3.77, STOI 0.944 (P). Low-SNR multilingual (DPDFNet authors' set): PESQ 2.76, STOI 90.5, DNSMOS-OVL 2.958 on DNS4 blind (P) | 48 kHz (P) | Causal; 40 ms algorithmic latency (2-frame lookahead) (P) | 2.14M params, 0.35 GMACs/s (P, DPDFNet table); RTF 0.19 single-thread i5-8250U (P, DFN paper) | MIT OR Apache-2.0 / pretrained-model licence not stated on repo (U) | https://arxiv.org/html/2305.08227 ; https://github.com/Rikorose/DeepFilterNet |
| **DPDFNet-8** (Ceva) | Dec 2025, v2 Jan 2026 | NS | Low-SNR multilingual set (12 languages, 0/5/10 dB, 324 clips): PESQ 3.20, STOI 93.4, SI-SNR 14.47, NISQA 4.21, PRISM 1.00 (P). DNS4 blind DNSMOS-OVL 3.122 vs DFN3 2.958 (P) | 16 kHz in paper; 48 kHz-HR variants in repo (P repo, no numbers) | Causal (P); StreamEnhancer ~20 ms initial latency + ~10 ms per block (repo/PyPI page) | 3.54M params, 4.37 GMACs/s (P, 16 kHz) | Apache-2.0 / Apache-2.0 on HF (P) | https://arxiv.org/html/2512.16420v2 ; https://github.com/ceva-ip/DPDFNet ; https://huggingface.co/Ceva-IP/DPDFNet |
| **DPDFNet-4 / -2** | same | NS | PESQ 3.18 / 3.14 (same set) (P) | 16 kHz (P) | Causal (P) | 2.84M / 2.49M params; 2.36 / 1.35 GMACs/s (P) | Apache-2.0 / Apache-2.0 (P) | same as above |
| **Sidon v0.1** (SaruLab) | Sep 2025, v3 Jan 2026 | Speech restoration (noisy in-the-wild to studio) | LibriTTS test-clean WER: noisy 0.040, Miipher 0.047, Sidon 0.045. test-other: noisy 0.079, Miipher 0.090, Sidon 0.095 (P). FLEURS avg CER: noisy 0.084, Sidon 0.090 (P) | Input 16 kHz, output 48 kHz (P) | Not streaming (U; clip-level per GGUF notes) | 250M params (198M predictor + 52.4M vocoder); RTF 0.002 on H200 GPU; no CPU speed (P) | MIT per HF card (P); paper CC BY 4.0 | https://arxiv.org/abs/2509.17052 ; https://arxiv.org/html/2509.17052v3 ; https://huggingface.co/sarulab-speech/sidon-v0.1 |
| **UniPASE** (NJU / Horizon) | Apr 2026 | Universal restoration (SE, PLC, GSR, USE; 8-48 kHz) | DNS2020 with-reverb: DNSMOS 3.33, UTMOS 3.62, dWER 8.16 (noisy 10.23, TF-GridNet 8.86, LLaSE-G1 41.66). VoiceFixer GSR: UTMOS 3.89. URGENT 2025 non-blind: UTMOS 2.97 (P, from paper) | Core 16 kHz; PostNet to 48 kHz (P) | Generative; not stated (U). GPU latency 44 ms for 1 s at 48 kHz on RTX 4090 (P) | 545.7M params; 79.2 GMACs/s (P) | Code MIT (P); checkpoints licence not stated (P) | https://arxiv.org/html/2604.14606v2 ; https://github.com/xiaobin-rong/unipase |
| **LLaSE-G1** | Mar 2025 | Multi-task SE incl. AEC | DNS with-reverb dWER 41.66 (noisy 10.23) (P, UniPASE table). Hallucination risk is high | Not stated (U) | Not causal (U) | LLaMA-based (P) | Apache-2.0 on HF metadata (P) | https://arxiv.org/abs/2503.00493 ; https://huggingface.co/ASLP-lab/LLaSE-G1 |
| **MossFormer2_SE_48K** (ClearerVoice) | toolkit Jun 2025 | NS, 48 kHz | No metric numbers found (U) | 48 kHz (config: 1920 FFT, 384 hop) (P) | U | 55.26M params (P, secondary port) | Apache-2.0 per secondary ports (U); toolkit Apache-2.0 (P) | https://github.com/modelscope/ClearerVoice-Studio ; https://arxiv.org/abs/2506.19398 ; https://huggingface.co/MigoXV/mossformer2-se-48k |
| **MP-SENet** | 2023 | NS | VoiceBank+DEMAND PESQ 3.50 (paper abstract; a secondary page says 3.60, so use 3.50). DNS WB-PESQ 3.62 (secondary summary of paper table, U) | 16 kHz (P) | U | 2.26M params (P) | MIT per secondary ONNX repack (U); upstream repo not fetched | https://arxiv.org/abs/2305.13686 ; https://huggingface.co/TigreGotico/audiosronnx-mpsenet |
| **SEMamba** (vd.pth) | 2024 | NS | PESQ: DNS-2020 3.66, VCTK-Demand 3.56 (P) | U | U | U | Licence not found (U) | https://arxiv.org/abs/2405.06573 ; https://github.com/RoyChao19477/SEMamba |
| **GTCRN** | ICASSP 2024 | NS | Low-SNR set (DPDFNet table): PESQ 2.21, STOI 87.2, OVL 2.25, P.808 3.00 (P) | U (trained on DNS3, VCTK-DEMAND) | Streaming version exists (P) | 48.2K params; 33.0 MMACs/s; RTF 0.07 on i5-12400 (P) | MIT (P, repo page) | https://github.com/Xiaobin-Rong/gtcrn |
| **Resemble Enhance** | 2024 | Denoiser + enhancer (restoration, BWE) | No metrics found (U) | Trained on 44.1 kHz (P) | U | U | Code MIT (P); HF model MIT (P) | https://github.com/resemble-ai/resemble-enhance ; https://huggingface.co/ResembleAI/resemble-enhance |
| **RE-USE** (NVIDIA) | 2026 (arXiv 2603.02641; date U) | Universal SE | No metrics on card (P) | 8-48 kHz in (P) | Not causal (U) | 9.6M params (P) | **NVIDIA One-Way Noncommercial (NSCLv1), "research and development only"** (P) | https://huggingface.co/nvidia/RE-USE |
| **Real-time RE-USE** | 2026 | Universal SE, streaming | No metrics found (U) | 8-48 kHz (P) | Streaming mode with look-ahead 0-2 (P) | 3.7M / 13.5M (P, conflicting) | NSCLv1, R&D only (P) | https://huggingface.co/nvidia/Real-time_RE-USE |
| **"One Model, Many Latencies"** (NVIDIA-linked) | Jun 2026 (SLT 2026) | Universal SE, configurable latency | Not extracted (U) | U | Configurable look-ahead (P) | U | Weights link to NSCLv1 model (P) | https://arxiv.org/abs/2606.25621 |
| **LIFT-SE** | 7 Oct 2026 | Generative NS + dereverb (QRes-Codec) | DNS1 with-reverb dWER 8.85 (noisy 10.99); URGENT 2025 WER 23.11 vs noisy 27.99 (P) | 16 kHz (P) | Not causal; AR decoding (P) | ~193M trainable (P) | Code "will be released" (P) | https://arxiv.org/abs/2610.09963 |
| **CoFi-Lite** | Jul 2026 | Ultra-light SE | No PESQ/DNSMOS in abstract (U) | U | U | 83.12k params; 12.87M MACs/s (P) | No code or weights stated (U) | https://arxiv.org/abs/2607.10142 |
| **Configurable-bandwidth TF model** | 24 Sep 2026 | Full-band SE across rates | No numbers in abstract (U) | 16 and 48 kHz training (P) | U | U | No code or weights stated (U) | https://arxiv.org/abs/2609.29463 |
| **FlowSE** (Honee-W) | May 2025 | Generative (flow) SE | U | U | Not causal (U) | U | Code and checkpoints claimed (P); licence unverified (U) | https://arxiv.org/abs/2505.19476 ; https://github.com/Honee-W/FlowSE |
| **BSRNN-Flow** (URGENT 2026 Track 1 baseline) | 2026 | Discriminative and generative baselines | Not extracted (U) | U | U | U | Repo Apache-2.0; checkpoint licence not stated (P) | https://github.com/urgent-challenge/urgent2026_challenge_track1 |
| **MeanFlowSE** (Xiamen) | Sep 2025 | One-step generative SE | VoiceBank-DEMAND claimed (U) | U | Not causal (U) | U | Code link claimed; not verified (U) | https://arxiv.org/abs/2509.14858 ; https://github.com/liduojia1/MeanFlowSE |
| **GenSE** | ICLR 2025 | Generative SE via LM | U | U | U | U | Code link is an anonymous mirror; licence not found (U) | https://arxiv.org/abs/2502.02942 |
| **FINALLY** (Samsung) | NeurIPS 2024 | Universal SE, 48 kHz | Not extracted (U) | 48 kHz output (P) | U | U | No code or weights found (U); project page returned 404 | https://neurips.cc/virtual/2024/poster/96882 |
| **AnyEnhance** | Jan 2025 (rev. Nov 2025) | Universal speech and singing enhancement | Claims SOTA on objective and subjective tests (P, abstract) | U | U | U | No code or weights found (U) | https://arxiv.org/abs/2501.15417 |
| **URGENT 2026 Track 1 winner (SoftBank)** | Feb 2026 | Universal SE | Press release says 1st in Track 1 (P; single Japanese-language source) | U | U | U | No code found | https://www.softbank.jp/corp/news/info/2026/20260219_01/ |
| **TS-URGENet** (NJU / Horizon) | May 2025 | Universal SE, 2nd in URGENT 2025 Track 1 | Not extracted (U) | U | U | U | U | https://arxiv.org/abs/2505.18533 |

Note on the URGENT 2026 result: UniPASE's abstract says its backbone won 1st place in URGENT 2026 objective evaluation (P, https://arxiv.org/abs/2604.14606). A SoftBank press release says it won Track 1. These two claims are not reconciled. The challenge overview says top systems were mainly hybrid generative/discriminative, and it does not name open-source top entries (https://arxiv.org/html/2601.13531v1).

### 3b. Acoustic echo cancellation

See the table in Section 2. Additional items:

| Model | Note | Link |
|---|---|---|
| WebRTC AEC3 | BSD-3 licence verified. Used as the default in Chrome-based calls (secondary source) | https://webrtc.googlesource.com/src/+/refs/heads/main/LICENSE |
| Microsoft AEC challenge baseline | MIT repo; baseline results not reported in the repo page (U) | https://github.com/microsoft/AEC-Challenge |

---

## 4. Excluded, and why

| Candidate | Reason |
|---|---|
| **RE-USE, Real-time RE-USE** (NVIDIA) | NSCLv1 is non-commercial and "for research and development only." That fails the A rule (no research-only), and it fails B and C. Source: https://huggingface.co/nvidia/RE-USE ; https://huggingface.co/nvidia/Real-time_RE-USE |
| **"One Model, Many Latencies"** | Weights are the NVIDIA NSCLv1 model. Same exclusion. https://arxiv.org/abs/2606.25621 |
| **LLaSE-G1** | Apache-2.0 metadata, but heavy word-change risk: dWER 41.66 vs 10.23 for noisy input on DNS with-reverb. Not causal. Excluded for B and C. For A it is excluded on quality. Sources: https://arxiv.org/html/2604.14606v2 (table V); https://huggingface.co/ASLP-lab/LLaSE-G1 |
| **Sidon** (as a restoration default) | MIT, but restored WER is above noisy WER on both LibriTTS subsets, and CER is above noisy on FLEURS average. Word-change risk. CPU speed unpublished (GPU only). Possible only as an opt-in for A. https://arxiv.org/html/2509.17052v3 |
| **UniPASE** (as default) | Code MIT, but checkpoint licence not stated. 545.7M params and 79.2 GMACs/s. CPU real time for B unbenchmarked (estimate: likely too heavy). Conditional pick for A only. https://github.com/xiaobin-rong/unipase |
| **LIFT-SE** | Code not released ("will be released"). Not causal. https://arxiv.org/abs/2610.09963 |
| **AnyEnhance, FINALLY, EchoFree, MSA-EchoLite, GenSE, E2E-AEC** | No code or weights found in the sources checked. |
| **DeepVQE original** | No official weights found. Only LocalVQE's derivative is available. |
| **ULCNet / Align-ULCNet** | No code or licence found. |
| **CoFi-Lite, Configurable-bandwidth TF (Sep 2026)** | No quality numbers and no code or weights in the abstracts I could read. |
| **SEMamba** | Released checkpoint, but licence not found. Re-check before use. |
| **FlowSE, MeanFlowSE, BSRNN-Flow** | Generative; not causal; latency unsuitable for C. Licences unverified. |

---

## 5. Open items to verify before any product decision

1. DPDFNet 48 kHz-HR variants: quality numbers and CPU cost at 48 kHz. Repo: https://github.com/ceva-ip/DPDFNet
2. DPDFNet algorithmic latency vs the 40 ms C budget, and CPU time per stream on the target server.
3. LocalVQE checkpoint licence (the HF weights page returned 401): https://huggingface.co/LocalAI-io/LocalVQE
4. UniPASE checkpoint licence and CPU real-time factor.
5. DeepFilterNet pretrained-model licence (the repo states only code licences).
6. Resemble Enhance and MossFormer2_SE_48K quality numbers and causality.
7. Whether a neural AEC residual stage beats AEC3 on a shared set (for example, the ICASSP 2022 blind set with AECMOS). This is the only test that would answer Section 2.

Fetch failures (not checked): https://huggingface.co/JusperLee/MossFormer2_SE_48K (401), https://huggingface.co/alibabasiam/MossFormer2_SE_48K (401), https://huggingface.co/ResembleAI/LocalVQE (401), https://samsunglabs.github.io/FINALLY-page/ (404), the Honee-W FlowSE repo and the RoyChao19477/SEMamba licence file (licence not on page), and the ClearerVoice ModelScope model page (not reached).
