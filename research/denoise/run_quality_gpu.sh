#!/usr/bin/env bash
# GPU quality runs on the 200-clip subset. Every GPU command holds the shared GPU lock only for its own slice.
set -uo pipefail
PY=~/audio-cleaner/research/se-venv/bin/python
LOCK=~/audio-cleaner/dev/gpu.lock
cd "$(dirname "$0")"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True TQDM_DISABLE=1
run() { flock "$LOCK" nice -n 5 $PY -I run_offline.py "$@" --device cuda --threads 2 --subset 200; }
run mf2_48k
run mfgan16
run resemble_denoise
for off in 0 50 100 150; do run resemble_enhance_nfe32 --offset $off --first 50; done
