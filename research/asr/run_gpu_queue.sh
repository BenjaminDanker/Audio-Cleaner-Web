#!/bin/bash
# Sequential GPU benchmark queue (GTX 960, sm_52). Every job is wrapped in flock on the shared GPU lock.
#  - Parakeet: venv-gpu = onnxruntime-gpu 1.19.2 (last PyPI build with sm_52 SASS) + cuDNN 9.7.1 (9.27 fails on Maxwell)
#  - Whisper:  venv-ct2old = ctranslate2 3.24.0 + faster-whisper 0.10.1 + CUDA 11.8/cuDNN 8.9 wheels
#    (ctranslate2 >= 4.0 wheels contain SASS only for sm_53/60/61/70/75/80/86 -> cannot run on sm_52)
set -u
cd "$(dirname "$0")"
R=$HOME/audio-cleaner/research
M=$R/models
PROD=$HOME/audio-cleaner/models/whisper
LOCK=$HOME/audio-cleaner/dev/gpu.lock
mkdir -p results/logs
rung() { env=$1; venv=$2; tag=$3; shift 3; [ -f results/$tag.json ] && return
  echo "$(date +%T) start $tag"; uptime
  ( source $R/$env; flock $LOCK $R/$venv/bin/python -I bench_offline.py --tag $tag "$@" > results/logs/$tag.log 2>&1 )
  echo "$(date +%T) done $tag"; }
rung env_gpu.sh venv-gpu pk2-fp32-gpu --engine onnxasr:$M/parakeet-tdt-0.6b-v2-onnx:none:cuda --threads 4 --long --ts
rung env_gpu.sh venv-gpu pk3-fp32-gpu --engine onnxasr:$M/parakeet-tdt-0.6b-v3-onnx:none:cuda --threads 4 --long --ts --sets clean,noisy5,noisy5_dfn3,fleurs_es,fleurs_de,fleurs_fr
rung env_ct2old.sh venv-ct2old fw-distil35-gpu-fp32 --engine fw:$M/distil-large-v3.5-ct2:cuda:float32 --threads 4 --long --limit 50
# large-v3-turbo float32 needs ~3.9 GB incl. CUDA context -> ctranslate2 3.24 segfaults at the first encode on the 4 GB card (verified):
# rung env_ct2old.sh venv-ct2old fw-turbo-gpu-fp32 --engine fw:$PROD/large-v3-turbo:cuda:float32 --threads 4 --long --limit 50
