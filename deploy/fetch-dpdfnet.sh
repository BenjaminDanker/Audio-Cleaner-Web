#!/usr/bin/env bash
# Download the DPDFNet-8 48 kHz "HR" ONNX model (Apache-2.0, https://huggingface.co/Ceva-IP/DPDFNet,
# code: https://github.com/ceva-ip/DPDFNet) used for the noise-removal engine for recorded files.
# Usage: deploy/fetch-dpdfnet.sh [dest-dir]   (default ~/audio-cleaner/models/dpdfnet)
# A thin wrapper (the desktop build calls it): the pinned revision and sha256 live in deploy/models.lock and
# the work is done by deploy/fetch-models.sh. Idempotent: does nothing when the file is already right.
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
DEST="${1:-${ACW_DPDFNET_DIR:-$HOME/audio-cleaner/models/dpdfnet}}"
exec bash "$HERE/fetch-models.sh" --only 'models/dpdfnet/*' --dest-dir "$DEST"
