#!/bin/bash
# usage: label_split.sh <split> ; dedupe stats by id, 3 label shards (CPU, low priority), concatenate into labels.jsonl.
set -u
S=$1
B=$HOME/audio-cleaner/research/auto-settings
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 CUDA_VISIBLE_DEVICES=-1
NICE="nice -n 19 ionice -c3"
cd $B
venv/bin/python - "$S" <<'PY'
import json, os, sys
S = sys.argv[1]; B = os.path.expanduser('~/audio-cleaner/research/auto-settings')
seen, lines = set(), []
for l in open(f'{B}/work/{S}/feat/stats.jsonl'):
    if not l.strip(): continue
    i = json.loads(l)['id']
    if i in seen: continue
    seen.add(i); lines.append(l.rstrip('\n'))
open(f'{B}/work/{S}/feat/stats.jsonl', 'w').write('\n'.join(lines) + '\n')
for k in range(3):
    open(f'{B}/work/{S}/feat/shard{k}.jsonl', 'w').write('\n'.join(lines[k::3]) + '\n')
print(S, len(lines), 'unique clips in stats')
PY
for k in 0 1 2; do
  $NICE venv/bin/python src/label.py $S work/$S/feat/shard$k.jsonl work/$S/raw work/$S/feat/labels$k.jsonl > logs/label_${S}_$k.log 2>&1 &
done
wait
cat work/$S/feat/labels0.jsonl work/$S/feat/labels1.jsonl work/$S/feat/labels2.jsonl > work/$S/feat/labels.jsonl
cat work/$S/feat/labels*.jsonl.skipped.txt > work/$S/feat/labels.skipped.txt 2>/dev/null
echo DONE label $S
