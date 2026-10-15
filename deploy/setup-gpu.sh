#!/usr/bin/env bash
# Optional: the GPU transcription worker's Python environment (~/audio-cleaner/venv-gpu).
# Needs an NVIDIA driver (nvidia-smi works). Then run deploy/install.sh, which
# enables acw-worker-gpu when this environment exists.
#
# Pins are for the GTX 960 (Maxwell, sm_52): onnxruntime-gpu 1.19.2 is the last
# release with sm_52 kernels, cuDNN 9.7 the last that runs on it (docs/research/asr.md).
# A newer card can use current onnxruntime-gpu / cuDNN: change the pins below.
set -euo pipefail
REPO="$(cd "$(dirname "$0")/.." && pwd)"
V="$HOME/audio-cleaner/venv-gpu"
command -v nvidia-smi >/dev/null || { echo "no NVIDIA driver (nvidia-smi) - nothing to do" >&2; exit 1; }
[ -x "$V/bin/python" ] || python3 -m venv "$V"
"$V/bin/pip" install -q --upgrade pip
# What the job runner needs, minus the CPU ONNX Runtime (replaced below) and live/web-only packages.
REQ="$(mktemp)"
trap 'rm -f "$REQ"' EXIT
grep -vE "^(onnxruntime|sherpa-onnx|sherpa-onnx-core|fastapi|uvicorn|starlette|stripe)==" "$REPO/server/requirements.lock" > "$REQ"
"$V/bin/pip" install -q --no-deps -r "$REQ"
"$V/bin/pip" install -q --no-deps \
  onnxruntime-gpu==1.19.2 \
  nvidia-cublas-cu12==12.9.2.10 nvidia-cuda-runtime-cu12==12.9.79 nvidia-cufft-cu12==11.4.1.4 \
  nvidia-curand-cu12==10.3.10.19 nvidia-cuda-nvrtc-cu12==12.9.86 nvidia-nvjitlink-cu12==12.9.86 \
  nvidia-cudnn-cu12==9.7.1.26 \
  coloredlogs humanfriendly flatbuffers protobuf sympy mpmath packaging
"$V/bin/python" -c "import onnxruntime as o; assert 'CUDAExecutionProvider' in o.get_available_providers(); print('ok: onnxruntime-gpu', o.__version__)"
