# CPU-offline speech denoising: candidates vs DeepFilterNet3 (research as of 2026-10-15)

Scope: models that could run offline on CPU in a free desktop app, full-band output, licence that allows free distribution. Every number below is cited; anything not verified is marked "unverified". No CPU real-time factors were found for any 2025-2026 candidate, so all CPU speed claims are "not measured".

## 1. Ranked shortlist

| Rank | Model (checkpoint) | Released | Quality (source) | Sample rate | CPU runtime / speed | Params / size | Licence code / weights | Links |
|---|---|---|---|---|---|---|---|---|
| 1 | **DPDFNet** (`dpdfnet8_48khz_hr`, `dpdfnet2_48khz_hr`; DFN2-derived, dual-path RNN) | arXiv 2512.16420 (Dec 2025) | Authors' 16 kHz eval only (324 clips, 12 languages, SNR 0/5/10 dB). DPDFNet-8 vs DeepFilterNet3 in same table: PESQ 2.85 vs 2.45; STOI 93.4 vs 90.6; SI-SNR 14.47 vs 12.10; NISQA 4.21 vs 3.84; PRISM 1.00 vs 0.82. DNS4 blind DNSMOS OVRL 3.122 vs DFN3 2.958. **No 48 kHz quality numbers published.** | 48 kHz HR variants supported (also 8/16 kHz) | ONNX CPU via PyPI package; RTF not reported for CPU. Authors measured RTF only on Ceva NPUs (DPDFNet-4 RT 0.97 on NPN32). | 48 kHz-HR: dpdfnet2 2.58 M params, 2.42 GMACs; dpdfnet8 3.63 M, 7.17 GMACs. ONNX ~10.0 / 14.2 MB | Code: Apache-2.0 (GitHub). Weights: model card metadata says apache-2.0; no separate weight text (unverified) | https://arxiv.org/html/2512.16420v1 ; https://github.com/ceva-ip/DPDFNet ; https://huggingface.co/ceva-ip/DPDFNet |
| 2 | **MossFormer2_SE_48K** (ClearerVoice-Studio, Alibaba) | arXiv 2506.19398 (Jun 2025) | Paper, VoiceBank+DEMAND 48 kHz (its Table 2): PESQ 3.15, SI-SDR 19.36, P.808 3.53. Same table, "DeepFilterNet" row: PESQ 3.03, SI-SDR 15.71, P.808 3.47. Our own benchmark: MossFormer2_SE_48K 3.08 / 0.955 / 3.19 vs DFN3 3.14 / 0.944 / 3.20 (**conflicts with paper; our number is the one to trust for our pipeline**). | 48 kHz in/out | PyTorch; no RTF published. Our CPU RTF not measured for this model. | Not stated; third-party card says ~55 M params (unverified) | HF card: `License: apache-2.0` (verified on alibabasglab/MossFormer2_SE_48K). Repo README: Apache-2.0 (per third-party mirror, unverified). Note the arXiv page carries CC BY-NC-SA 4.0 — that is the paper, not the weights. | https://huggingface.co/alibabasglab/MossFormer2_SE_48K ; https://arxiv.org/html/2506.19398v1 ; https://github.com/modelscope/ClearerVoice-Studio |
| 3 | **Sidon** (sarulab-speech/sidon-v0.1; w2v-BERT 2.0 feature predictor + HiFi-GAN-style vocoder) | arXiv 2509.17052 (Sep 2025) | Restoration of clean speech, not a pure denoiser. LibriTTS test-clean: DNSMOS 3.303 (noisy 3.179), NISQA 4.790 (noisy 4.093), **WER 0.045 (noisy 0.040)**. FLEURS (100 langs): DNSMOS 3.393 (noisy 2.910), **CER 0.090 (noisy 0.084)**. Content changes are measurable: word/character error rates go up, so hallucination/word-change risk is real. | 48 kHz out ("48 kHz full-fidelity speech"); input 16 kHz | GPU RTF only (H200, bf16: 0.0020–0.0023). ONNX int8/fp16 bundle ~286 MB exists (soniqo guide); **no CPU RTF published**. | ~250 M total (198 M predictor + 52.4 M vocoder) | HF card: `License: mit` (verified on sarulab-speech/sidon-v0.1). **This contradicts the "no licence" note in the brief; check the sarulab repo LICENSE before relying on it.** | https://arxiv.org/html/2509.17052v2 ; https://huggingface.co/sarulab-speech/sidon-v0.1 ; https://huggingface.co/cstr/Sidon-GGUF |
| 4 | **UniPASE** (xiaobin-rong/unipase; generative, universal) | arXiv 2604.14606 (Apr 2026; v2 dated 20 Jul 2026) | Own tables, DNS 2020 no-reverb: DNSMOS 3.40, UTMOS 4.06, PESQ 3.05, ESTOI 0.93, dWER 2.17 % (TF-GridNet 3.18 / 3.86 / 2.86 %). URGENT 2025 non-blind: BSRNN-FAN (blind rank 1) PESQ 2.95 vs UniPASE 2.12; UniPASE CER 12.90 vs 11.08. Self-reported: backbone of a 1st-place objective result in URGENT 2026. | Inputs resampled to 16 kHz internally; PostNet outputs 48 kHz then resamples to original rate | GPU only: RTX 4090, 1 s of 48 kHz audio ≈ 44 ms. **No CPU timing.** | 545.7 M params; 79.2 GMACs per s of audio (per paper) | Code: MIT (GitHub). Checkpoints (DeWavLM-Omni.pt, Adapter.pt, Vocoder_DWO-L1.pt, PostNet.pt, ...): **licence not stated** | https://arxiv.org/html/2604.14606 ; https://github.com/xiaobin-rong/unipase |

Notes on the table:
- Quality numbers come from authors' own evals on different sets, so they are not directly comparable across rows. The only head-to-head vs DFN3 with published 48 kHz numbers is MossFormer2 (paper table) and our own VoiceBank run.
- Estimated runtime is not given: UniPASE (79 GMACs/s, ~546 M params) and Sidon (~250 M params, generative) are likely far slower than DFN3 on a laptop CPU. We did not measure it, so this is an expectation, not a result.

### Other possibly relevant 2026 systems (not shortlisted)

- **GAP-URGENet** (arXiv 2604.01832, Apr 2026; ICASSP 2026 URGENT Track 1). Objective-validation PESQ 2.89, DNSMOS 3.31, UTMOS 3.22, NISQA 3.96. Self-reported 1st in blind objective evaluation. 567.76 M params, 472.84 GMACs/s. **No code or weights link found** (only a demo page). Excluded: not usable and too heavy. https://arxiv.org/html/2604.01832v1
- **URGENT 2026 official ranking**: the stage-1 leaderboard is at https://urgent-challenge.com/competitions/15#results and the stage-2 PDF could not be parsed (unverified). Challenge overview: https://arxiv.org/pdf/2601.13531 (29 valid submissions; Track 1 must handle 8–48 kHz).

## 2. Direct answer

**No candidate is clearly better than DeepFilterNet3 on CPU with a shippable licence, based on what we could verify.**

- The best-supported upgrade path is **DPDFNet**: same DFN2/3 lineage, Apache-2.0 code, 2.6–3.6 M params, 48 kHz-HR variants. Its gains over DFN3 are large on the authors' 16 kHz benchmark, but those are authors' numbers, with no independent verification and no 48 kHz quality numbers. Its weight licence and CPU speed still need checking. Recommended next step: run dpdfnet8_48khz_hr through our VoiceBank-DEMAND and DNSMOS harness.
- **MossFormer2_SE_48K** is Apache-2.0 and 48 kHz, but our own benchmark had it below DFN3 on PESQ (3.08 vs 3.14). Its paper shows a PESQ gain of +0.12 and SI-SDR +3.6 dB over a DFN row, which conflicts with our run, so it is not clearly better.
- **Sidon** has the strongest restoration claims among the open candidates but changes words (WER and CER higher than noisy input in its own tables), needs ~250 M params, and has no CPU measurements. Treat it as a "clean-up for TTS data" model, not a faithful voice-cleanup model, unless our own listening test says otherwise.
- **UniPASE** has the best published objective numbers in this set, but it is 0.5 B-class, GPU-tested only, and its checkpoint licence is unstated. Not usable for us until the licence is confirmed and CPU speed is measured.

## 3. Excluded or unusable

| Model | Reason |
|---|---|
| NVIDIA RE-USE | Research/development only; licence "other"; unofficial mirror says NSCLv1 non-commercial. https://huggingface.co/nvidia/RE-USE/blob/main/README.md |
| NVIDIA SE Denoising SB 16k Small | CC-BY-NC-SA-4.0; research use only. https://huggingface.co/nvidia/se_den_sb_16k_small |
| NVIDIA Maxine BNR / Studio Voice | Evaluation licence; "exclusively part of NVIDIA AI Enterprise for production". https://build.nvidia.com/nvidia/bnr/modelcard ; https://build.nvidia.com/nvidia/studiovoice/modelcard |
| LLaSE-G1 (ASLP-lab) | Apache-2.0 (verified on HF), but LLM-based with heavy content change: dWER 41.66 % on DNS reverb vs 10.23 % noisy, WER 31.46 vs 8.25 clean on PLC (third-party measurements in UniPASE paper). https://huggingface.co/ASLP-lab/LLaSE-G1 ; https://arxiv.org/html/2604.14606 |
| USEMamba (URGENT 2025, rank 2) | No code or checkpoint URL found; no licence. https://arxiv.org/abs/2505.21198 |
| AnyEnhance | Code repo is for the CCF AATC challenge; licence not stated; no weights link found (unverified). https://github.com/viewfinder-annn/anyenhance-v1-ccf-aatc |
| FlowSE (Honee-W/FlowSE; seongq/flowmse) | Code and checkpoints announced; licence and sample rate unverified. https://arxiv.org/html/2505.19476v2 ; https://arxiv.org/html/2508.06840v1 |
| ZipEnhancer (Alibaba) | 16 kHz; 2.04 M params; PESQ 3.69 (DNS 2020) and 3.63 (VB+DEMAND) per the paper; licence unverified; BWE needed. https://arxiv.org/html/2501.05183v1 |
| MossFormerGAN_SE_16K | Apache-2.0 but 16 kHz only; CPU RTF 7.5 in our benchmark. https://huggingface.co/alibabasglab/MossFormerGAN_SE_16K |
| Meta denoiser (facebookresearch/denoiser) | Older model (2020); licence not verified. Not a 2025-2026 release. |
| BBWEXNet | Bandwidth extension only, not a denoiser. |
| GAP-URGENet | No code or weights released; 568 M params. |

## 4. Not verified / open items

- Sidon licence: HF says MIT; the brief said no licence. Check the sarulab-speech GitHub LICENSE.
- DPDFNet 48 kHz-HR quality: no published numbers. Weight licence not stated separately.
- MossFormer2_SE_48K: parameter count and CPU RTF not published; the paper's DFN row conflicts with our benchmark.
- UniPASE checkpoint licence and CPU speed.
- URGENT 2026 final stage-2 ranking (PDF unreadable here).

Sources are linked inline above.

## Addendum (2026-10-15): DPDFNet-8 vs DeepFilterNet3 by noise level, and the automatic choice

Our own check on the 200-clip VoiceBank-DEMAND subset (48 kHz DPDFNet-8 "HR" ONNX vs DeepFilterNet3,
attenuation unlimited, SI-SDR against the clean speech; noise estimate = 90th minus 10th percentile of
10 ms frame energies, which reads ~9 dB above the true SNR):

| Noise estimate (dB) | Clips | DFN3 SI-SDR | DPDFNet-8 SI-SDR | DPDFNet-8 better on |
|---|---|---|---|---|
| 0-14 | 31 | 13.43 | 14.19 | 97 % |
| 14-17 | 38 | 18.10 | 18.49 | 71 % |
| 17-19 | 28 | 18.86 | 19.11 | 68 % |
| 19-21 | 34 | 19.22 | 19.24 | 44 % |
| 21-23 | 23 | 20.56 | 20.03 | 26 % |
| 23-30 | 45 | 21.48 | 20.94 | 36 % |

Speech level on clean input is unchanged by both (-0.3 / -0.2 dB). Decision: recordings (website Clean &
polish, desktop app) choose per file with `pipeline.HEAVY_BELOW_DB = 19` ("Background noise: Automatic"),
and users can force Light (DeepFilterNet3) or Heavy (DPDFNet-8). Live streams and the in-browser tool keep
DeepFilterNet3 (DPDFNet-8 costs ~7x the CPU per stream). CPU for files: DPDFNet-8 RTF 0.85 on 1 thread,
0.33 on 4 (60 s clip).
