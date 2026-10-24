#!/bin/bash
# LibriTTS-R (CC BY 4.0) parquet mirror on Hugging Face: mythicinfinity/libritts_r (license cc-by-4.0 in its card)
cd ~/audio-cleaner/research/auto-settings/data
for f in data/test.clean/test.clean-0000{0,1,2}-of-00003.parquet data/dev.clean/dev.clean-0000{0,1,2,3}-of-00004.parquet; do
  out=$(basename $f)
  [ -f $out.done ] || { nice -n 19 curl -sSL -o $out https://huggingface.co/datasets/mythicinfinity/libritts_r/resolve/main/$f && touch $out.done; }
done
echo ALLDONE
