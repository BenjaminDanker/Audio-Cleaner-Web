#!/usr/bin/env bash
# Speed / memory runs on the 10-minute file (GPU runs hold the shared GPU lock for their own duration only).
# Slow models use a shorter excerpt of the same file; the RTF is then a linear extrapolation (chunked => linear) and is labelled in `note`.
set -uo pipefail
PY=~/audio-cleaner/research/se-venv/bin/python
LOCK=~/audio-cleaner/dev/gpu.lock
cd "$(dirname "$0")"
export TQDM_DISABLE=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
g() { flock "$LOCK" $PY -I bench_speed.py "$@" --device cuda --threads 4; }
c() { $PY -I bench_speed.py "$@" --device cpu; }
g mf2_48k --seconds 600
g resemble_denoise --seconds 600
g mfgan16_w6 --seconds 180 --note "180 s excerpt, 6 s windows"
g mfgan16 --seconds 60 --note "60 s excerpt, upstream 10 s windows"
g resemble_enhance_nfe8 --seconds 120 --note "120 s excerpt"
g resemble_enhance_nfe16 --seconds 120 --note "120 s excerpt"
g resemble_enhance_nfe32 --seconds 120 --note "120 s excerpt"
c dfn3 --threads 1 --seconds 600
c dfn3 --threads 4 --seconds 600
c dfn3 --threads 8 --seconds 600
c mf2_48k --threads 8 --seconds 60 --note "60 s excerpt"
c mf2_48k --threads 16 --seconds 60 --note "60 s excerpt"
c resemble_denoise --threads 8 --seconds 60 --note "60 s excerpt"
c mfgan16_w6 --threads 8 --seconds 30 --note "30 s excerpt, 6 s windows"
c resemble_enhance_nfe16 --threads 8 --seconds 15 --note "15 s excerpt"
