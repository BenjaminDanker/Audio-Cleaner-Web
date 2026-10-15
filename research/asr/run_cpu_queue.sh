#!/bin/bash
# Sequential CPU benchmark queue (one job at a time so timings are not disturbed by our own load).
# Usage: ./run_cpu_queue.sh   (research venv python; results in results/*.json, logs in results/logs/)
set -u
cd "$(dirname "$0")"
PY=$HOME/audio-cleaner/research/venv/bin/python
M=$HOME/audio-cleaner/research/models
PROD=$HOME/audio-cleaner/models/whisper
mkdir -p results/logs
run() { tag=$1; shift; [ -f results/$tag.json ] && return; echo "$(date +%T) start $tag"; uptime; $PY -I bench_offline.py --tag $tag "$@" > results/logs/$tag.log 2>&1; echo "$(date +%T) done $tag"; }
run pk2-int8-cpu   --engine onnxasr:$M/parakeet-tdt-0.6b-v2-onnx:int8:cpu --threads 8 --long --ts
run fw-turbo-cpu-int8 --engine fw:$PROD/large-v3-turbo:cpu:int8 --threads 8 --long --limit 20
run pk3-int8-cpu   --engine onnxasr:$M/parakeet-tdt-0.6b-v3-onnx:int8:cpu --threads 8 --long --ts --sets clean,noisy5,noisy5_dfn3,fleurs_es,fleurs_de,fleurs_fr
run pk2-fp32-cpu   --engine onnxasr:$M/parakeet-tdt-0.6b-v2-onnx:none:cpu --threads 8 --long --ts
run fw-turbo-cpu-int8-fleurs --engine fw:$PROD/large-v3-turbo:cpu:int8 --threads 8 --sets fleurs_es,fleurs_de,fleurs_fr --limit 10
run fw-distil35-cpu-int8 --engine fw:$M/distil-large-v3.5-ct2:cpu:int8 --threads 8 --long --limit 20
run pk3-fp32-cpu   --engine onnxasr:$M/parakeet-tdt-0.6b-v3-onnx:none:cpu --threads 8 --long --ts --sets clean,fleurs_es,fleurs_de,fleurs_fr
# --- added after fixing word-timestamp grouping in onnx-asr results (re-run of the Parakeet jobs) ---
run sherpa-pk2-int8-cpu --engine sherpa:$M/sherpa/sherpa-onnx-nemo-parakeet-tdt-0.6b-v2-int8:int8:cpu --threads 8 --long --ts
run fw-turbo-cpu-int8-ts --engine fw:$PROD/large-v3-turbo:cpu:int8 --threads 8 --sets clean --limit 20 --ts
run fw-distil35-cpu-int8-ts --engine fw:$M/distil-large-v3.5-ct2:cpu:int8 --threads 8 --sets clean --limit 20 --ts
