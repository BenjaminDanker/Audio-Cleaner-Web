#!/bin/bash
# Higher stream counts for the two cheap streaming models (capacity estimate); waits for the other queues.
cd "$(dirname "$0")"
while pgrep -f "run_stream_queue.sh|run_final_queue.sh" > /dev/null; do sleep 5; done
PY=$HOME/audio-cleaner/research/venv/bin/python
for cfg in "zip-en 12 1" "nemo-fc-480 8 1" "nemo-fc-80 4 1"; do
  set -- $cfg
  [ -f results/stream-$1-live$2-t$3.json ] && continue
  echo "$(date +%T) start $cfg"; uptime
  $PY -I bench_stream.py --model $1 --live $2 --threads $3 > results/logs/stream-$1-live$2-t$3.log 2>&1
  echo "$(date +%T) done $cfg"
done
echo "extra done"
