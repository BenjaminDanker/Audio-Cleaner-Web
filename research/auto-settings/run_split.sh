#!/bin/bash
# usage: run_split.sh <split> <count> ; generate, measure (Node, 3 shards), label (3 shards). Low priority.
set -u
S=$1; N=$2
B=$HOME/audio-cleaner/research/auto-settings
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 UV_THREADPOOL_SIZE=1
export DFN_WASM=$B/models/df_bg.wasm DFN_MODEL=$B/models/DeepFilterNet3_onnx.tar.gz
NICE="nice -n 19 ionice -c3"
cd $B
mkdir -p work/$S/raw work/$S/feat clips/$S
# 1. generate: 3 seeds in parallel
for k in 1 2 3; do
  $NICE venv/bin/python src/gen.py $S $(( (N + 2) / 3 )) $((k + ${SEEDBASE:-0})) > logs/gen_${S}_$k.log 2>&1 &
done
wait
# 2. manifest, then 3 shards of the measuring
venv/bin/python - "$S" <<'PY'
import glob, json, os, sys
S = sys.argv[1]; B = os.path.expanduser('~/audio-cleaner/research/auto-settings')
ids = sorted(os.path.basename(p)[:-5] for p in glob.glob(f'{B}/clips/{S}/*.json'))
m = [{'id': i, 'path': f'{B}/clips/{S}/{i}.wav'} for i in ids]
for k in range(3):
    json.dump(m[k::3], open(f'{B}/work/{S}/feat/m{k}.json', 'w'))
print(S, len(m), 'clips')
PY
for k in 0 1 2; do
  $NICE node src/feat.mjs work/$S/feat/m$k.json work/$S/feat/stats$k.jsonl work/$S/raw > logs/feat_${S}_$k.log 2>&1 &
done
wait
cat work/$S/feat/stats0.jsonl work/$S/feat/stats1.jsonl work/$S/feat/stats2.jsonl > work/$S/feat/stats.jsonl
node src/rules.mjs work/$S/feat/stats.jsonl work/$S/feat/rules.jsonl
# 3. labels (PESQ), 3 shards, dedupe, missing raw outputs skipped and logged: label_split.sh
bash src/label_split.sh $S
