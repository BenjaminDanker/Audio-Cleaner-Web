#!/bin/bash
# background download of LibriTTS-R subsets (CC BY 4.0), low priority
cd ~/audio-cleaner/research/auto-settings/data
for s in test_clean dev_clean; do
  [ -f $s.tar.gz.done ] || { nice -n 19 ionice -c3 curl -sS -L -o $s.tar.gz https://www.openslr.org/resources/141/$s.tar.gz && touch $s.tar.gz.done; }
done
