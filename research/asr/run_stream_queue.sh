#!/bin/bash
# Streaming (sherpa-onnx, CPU int8) benchmark queue: real-time paced live runs, then WER. One job at a time.
set -u
cd "$(dirname "$0")"
PY=$HOME/audio-cleaner/research/venv/bin/python
mkdir -p results/logs
job() { name=$1; shift; [ -f results/$name.json ] && return; echo "$(date +%T) start $name"; uptime; $PY -I bench_stream.py "$@" > results/logs/$name.log 2>&1; echo "$(date +%T) done $name"; }
# live: N concurrent streams, T threads per stream
job stream-zip-en-live1-t2        --model zip-en --live 1 --threads 2
job stream-zip-en-live4-t1        --model zip-en --live 4 --threads 1
job stream-nemo-fc-480-live1-t2   --model nemo-fc-480 --live 1 --threads 2
job stream-nemo-fc-480-live4-t1   --model nemo-fc-480 --live 4 --threads 1
job stream-nemo-fc-80-live1-t2    --model nemo-fc-80 --live 1 --threads 2
job stream-nemotron-560-live1-t4  --model nemotron-560 --live 1 --threads 4
job stream-nemotron-560-live2-t2  --model nemotron-560 --live 2 --threads 2
job stream-nemotron-160-live1-t4  --model nemotron-160 --live 1 --threads 4
job stream-nemotron35-560-live1-t4 --model nemotron35-560 --live 1 --threads 4
# WER (first 40 utterances of the clean set; 30 of the noisy sets for the candidates), 2 threads
for m in zip-en nemo-fc-80 nemo-fc-480 nemotron-160 nemotron-560 nemotron35-160 nemotron35-560; do
  job stream-$m-wer-t2 --model $m --wer-limit 40 --sets clean --threads 2
done
for m in zip-en nemo-fc-480 nemotron-560; do
  job stream-$m-wer-noisy-t2 --model $m --wer-limit 30 --sets noisy5,noisy5_dfn3 --threads 2
done
