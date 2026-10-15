#!/usr/bin/env bash
# Fetch third-party code, weights and datasets for the benchmark. Everything is untrusted: each item goes in its own
# new directory under ~/audio-cleaner/research/, nothing is executed from there, and Python that reads it runs with -I.
# Large files are rate-limited (shared home link). Re-run is safe (skips what exists).
set -euo pipefail
R=~/audio-cleaner/research
H=https://huggingface.co
dl() { [ -s "$2" ] || curl -fsSL --limit-rate 40M "$1" -o "$2"; }

# --- code (shallow clones, pinned by the commit the benchmark ran against) ---
mkdir -p $R/denoise-src && cd $R/denoise-src
clone() { [ -d "$2" ] || { GIT_LFS_SKIP_SMUDGE=1 git clone -q --depth 1 "https://github.com/$1.git" "$2"; }; }
clone modelscope/ClearerVoice-Studio ClearerVoice-Studio   # 6b3774dc79c4 (Apache-2.0)
clone resemble-ai/resemble-enhance resemble-enhance        # 8e978149bfe8 (MIT)
clone Xiaobin-Rong/gtcrn gtcrn                             # 502ebfab64da (MIT)
clone ceva-ip/DPDFNet DPDFNet                              # 9bd9844a227b (Apache-2.0)

# --- DNSMOS P.835 ONNX (Microsoft DNS-Challenge repo; internal evaluation only) ---
mkdir -p dnsmos_dl && cd dnsmos_dl
dl https://raw.githubusercontent.com/microsoft/DNS-Challenge/master/DNSMOS/DNSMOS/sig_bak_ovr.onnx DNSMOS_DNSMOS_sig_bak_ovr.onnx
dl https://raw.githubusercontent.com/microsoft/DNS-Challenge/master/LICENSE LICENSE
cd ..

# --- weights ---
M=$R/denoise-models
mkdir -p $M/mossformer2_se_48k $M/mossformergan_se_16k $M/resemble_enhance/ds/G/default $M/dpdfnet
dl $H/alibabasglab/MossFormer2_SE_48K/resolve/main/last_best_checkpoint $M/mossformer2_se_48k/last_best_checkpoint
dl $H/alibabasglab/MossFormer2_SE_48K/resolve/main/last_best_checkpoint.pt $M/mossformer2_se_48k/last_best_checkpoint.pt
dl $H/alibabasglab/MossFormerGAN_SE_16K/resolve/main/last_best_checkpoint $M/mossformergan_se_16k/last_best_checkpoint
dl $H/alibabasglab/MossFormerGAN_SE_16K/resolve/main/last_best_checkpoint.pt $M/mossformergan_se_16k/last_best_checkpoint.pt
RE=$H/ResembleAI/resemble-enhance/resolve/main/enhancer_stage2
dl $RE/hparams.yaml $M/resemble_enhance/hparams.yaml
dl $RE/ds/G/latest $M/resemble_enhance/ds/G/latest
dl $RE/ds/G/default/mp_rank_00_model_states.pt $M/resemble_enhance/ds/G/default/mp_rank_00_model_states.pt
for f in dpdfnet2_48khz_hr dpdfnet8_48khz_hr dpdfnet2 dpdfnet4 baseline; do dl $H/Ceva-IP/DPDFNet/resolve/main/onnx/$f.onnx $M/dpdfnet/$f.onnx; done

# --- VoiceBank-DEMAND (Edinburgh DataShare 10283/2791, CC BY 4.0): TEST SET ONLY, 48 kHz ---
D=$R/denoise-data/dl_vbd; mkdir -p $D && cd $D
B=https://datashare.ed.ac.uk/bitstreams
dl $B/7e7bc32f-e94a-436a-8e97-416a972c7e1a/download license.txt
dl $B/dec213d3-bf57-4777-9663-c24bdce92d5e/download clean_testset_wav.zip
dl $B/13c1bfbf-14a6-41db-9b41-8f7310f01ad5/download noisy_testset_wav.zip
echo "unzip into $R/denoise-data/vbd48_clean and vbd48_noisy (flat, wav only) - see docs/research/denoise.md"
