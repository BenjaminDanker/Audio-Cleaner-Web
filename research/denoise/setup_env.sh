#!/usr/bin/env bash
# Create ~/audio-cleaner/research/se-venv (idempotent). No sudo, no system packages besides gcc/python3-dev for `pesq`.
set -euo pipefail
V=~/audio-cleaner/research/se-venv
[ -d "$V" ] || python3 -m venv "$V"
"$V/bin/pip" install -U pip wheel
"$V/bin/pip" install torch==2.11.0 torchaudio==2.11.0 --index-url https://download.pytorch.org/whl/cu126
"$V/bin/pip" install -r "$(dirname "$0")/requirements.txt"
