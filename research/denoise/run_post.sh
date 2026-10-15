#!/usr/bin/env bash
# After the speed runs: per-frame live benchmark (machine quiet), a 50-clip NFE16 quality slice, metrics for everything, listening samples.
set -uo pipefail
PY=~/audio-cleaner/research/se-venv/bin/python
LOCK=~/audio-cleaner/dev/gpu.lock
cd "$(dirname "$0")"
export TQDM_DISABLE=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
while pgrep -f run_speed.sh >/dev/null; do sleep 5; done
$PY -I bench_live.py dfn3_live dfn3_ll dpdfnet2_48khz_hr dpdfnet8_48khz_hr dpdfnet2 dpdfnet4 baseline gtcrn_dns3 --frames 3000 --reps 3 --core 7
flock "$LOCK" $PY -I run_offline.py resemble_enhance_nfe16 --device cuda --threads 2 --subset 200 --first 50
$PY -I eval_metrics.py dpdfnet4 baseline gtcrn_dns3 gtcrn_vctk dpdfnet8_48khz_hr resemble_enhance_nfe32 resemble_enhance_nfe16 --workers 6
$PY -I eval_metrics.py mf2_48k mfgan16 hybrid_mfgan_dfn3 resemble_denoise resemble_enhance_nfe32 --clarity --workers 6
flock "$LOCK" $PY -I make_samples.py --nfe 32
echo POST_DONE
