#!/bin/bash
# Re-measure the headline long-form RTFs / concurrency when the box is otherwise idle (the other queues overlapped
# with a heavy job of another agent: loadavg 8-10). Results: results/*-quiet.json, results/concurrent_files_quiet.txt
set -u
cd "$(dirname "$0")"
PY=$HOME/audio-cleaner/research/venv/bin/python
M=$HOME/audio-cleaner/research/models
PROD=$HOME/audio-cleaner/models/whisper
mkdir -p results/logs
run() { tag=$1; shift; [ -f results/$tag.json ] && return; echo "$(date +%T) start $tag"; uptime; $PY -I bench_offline.py --tag $tag "$@" > results/logs/$tag.log 2>&1; echo "$(date +%T) done $tag"; }
run pk2-fp32-cpu-quiet --engine onnxasr:$M/parakeet-tdt-0.6b-v2-onnx:none:cpu --threads 8 --long --ts --sets clean --limit 20
run sherpa-pk2-int8-cpu-quiet --engine sherpa:$M/sherpa/sherpa-onnx-nemo-parakeet-tdt-0.6b-v2-int8:int8:cpu --threads 8 --long --ts --sets clean --limit 20
run fw-turbo-cpu-int8-quiet --engine fw:$PROD/large-v3-turbo:cpu:int8 --threads 8 --long --sets clean --limit 2
for cfg in "1 8" "2 4" "4 2"; do
  set -- $cfg
  echo "$(date +%T) concurrent files N=$1 threads=$2"; uptime
  $PY -I bench_concurrent_files.py $M/parakeet-tdt-0.6b-v2-onnx none $1 $2 1 2>&1 | grep -E "^jobs=" | tee -a results/concurrent_files_quiet.txt
done
echo "$(date +%T) quiet all done"
