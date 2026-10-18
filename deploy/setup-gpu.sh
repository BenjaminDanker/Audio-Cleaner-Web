#!/usr/bin/env bash
# Optional: the GPU transcription worker's Python environment (acw's ~/audio-cleaner/venv-gpu; run by the owner it
# stages these files for acw and runs this script there as acw).
# Needs an NVIDIA driver (nvidia-smi works). Then run deploy/install.sh, which
# enables acw-worker-gpu when this environment exists.
#
# Pins are for the GTX 960 (Maxwell, sm_52): onnxruntime-gpu 1.19.2 is the last
# release with sm_52 kernels, cuDNN 9.7 the last that runs on it (docs/research/asr.md).
# A newer card can use current onnxruntime-gpu / cuDNN: change the pins in deploy/requirements-gpu.lock
# and run `python3 deploy/hash_lock.py deploy/requirements-gpu.lock`.
set -euo pipefail
REPO="$(cd "$(dirname "$0")/.." && pwd)"
# shellcheck source=deploy/lib.sh
. "$REPO/deploy/lib.sh"
if [ "$ACW_DELEGATE" = 1 ]; then
  STAGE="$(stage_for_acw "$(git -C "$REPO" rev-parse --short HEAD)-gpu" deploy)"
  rc=0
  as_acw bash "$STAGE/deploy/setup-gpu.sh" "$@" || rc=$?
  exit "$rc"
fi
V="$H/venv-gpu"
command -v nvidia-smi >/dev/null || { echo "no NVIDIA driver (nvidia-smi) - nothing to do" >&2; exit 1; }
[ -x "$V/bin/python" ] || python3 -m venv "$V"
"$V/bin/pip" install -q --upgrade pip
# Every package comes from deploy/requirements-gpu.lock, pinned with the sha256 of each file PyPI lists
# (--require-hashes): the job runner's packages (same versions as server/requirements.lock), ONNX Runtime with CUDA
# and the CUDA 12 / cuDNN 9 libraries. --no-deps because the lock already names every dependency.
"$V/bin/pip" install -q --require-hashes --no-deps -r "$REPO/deploy/requirements-gpu.lock"
"$V/bin/python" -c "import onnxruntime as o; assert 'CUDAExecutionProvider' in o.get_available_providers(); print('ok: onnxruntime-gpu', o.__version__)"
