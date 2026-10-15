#!/usr/bin/env bash
# Download the DPDFNet-8 48 kHz "HR" ONNX model (Apache-2.0, https://huggingface.co/Ceva-IP/DPDFNet,
# code: https://github.com/ceva-ip/DPDFNet) used for the noise-removal engine for recorded files.
# Pinned to one Hugging Face commit and verified by sha256. Usage: deploy/fetch-dpdfnet.sh [dest-dir]
# (default ~/audio-cleaner/models/dpdfnet). Idempotent: does nothing when the file is already right.
set -euo pipefail
REV="dd6818d00f50c836fed43a6243ebe49116de5964"
FILE="dpdfnet8_48khz_hr.onnx"
SHA256="7b3afbb260a08fe9af3d16e3bda992971be1e7e951d1dee7c2d235f5c43f5631"
URL="https://huggingface.co/Ceva-IP/DPDFNet/resolve/$REV/onnx/$FILE"
DEST="${1:-${ACW_DPDFNET_DIR:-$HOME/audio-cleaner/models/dpdfnet}}"

mkdir -p "$DEST"
if [ -f "$DEST/$FILE" ] && echo "$SHA256  $DEST/$FILE" | sha256sum -c --status -; then
  echo "dpdfnet: $DEST/$FILE already present"
  exit 0
fi
tmp="$(mktemp "$DEST/.$FILE.XXXXXX")"
trap 'rm -f "$tmp"' EXIT
curl -fL --retry 3 -o "$tmp" "$URL"
echo "$SHA256  $tmp" | sha256sum -c --status - || { echo "dpdfnet: sha256 mismatch for $URL" >&2; exit 1; }
chmod 644 "$tmp"
mv "$tmp" "$DEST/$FILE"
trap - EXIT
echo "dpdfnet: installed $DEST/$FILE"
