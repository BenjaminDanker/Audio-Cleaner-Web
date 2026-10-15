#!/usr/bin/env bash
# CPU-only quality runs on the fixed 200-clip VoiceBank-DEMAND subset (single-threaded models, nice'd).
set -euo pipefail
PY=~/audio-cleaner/research/se-venv/bin/python
cd "$(dirname "$0")"
for m in dfn3 dfn3_ll dpdfnet2_48khz_hr dpdfnet8_48khz_hr dpdfnet2 dpdfnet4 baseline gtcrn_dns3 gtcrn_vctk; do
  nice -n 10 $PY -I run_offline.py "$m" --device cpu --threads 1 --subset 200
done
