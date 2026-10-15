#!/bin/bash
# Builds the media fixtures the e2e scripts expect in $E2E_FILES (default ~/audio-cleaner/dev/webtest-files).
# Source speech: DeepFilterNet's noisy_snr0.wav (CC BY-NC) - never copy it into the repo.
set -e
OUT=${E2E_FILES:-$HOME/audio-cleaner/dev/webtest-files}
SRC=${E2E_SRC_WAV:-$HOME/audio-cleaner/src/DeepFilterNet/assets/noisy_snr0.wav}
FF=${FFMPEG:-$HOME/audio-cleaner/bin/ffmpeg}
mkdir -p "$OUT" && cd "$OUT"
cp "$SRC" noisy.wav
$FF -v error -y -i "$SRC" -ar 44100 -ac 2 stereo44.wav
$FF -v error -y -i "$SRC" -c:a libmp3lame -b:a 128k noisy.mp3
$FF -v error -y -i "$SRC" -c:a wmav2 noisy.wma  # a format browsers cannot open (the server reads it)
$FF -v error -y -f lavfi -i testsrc2=size=640x360:rate=25 -i "$SRC" -map 0:v -map 1:a -c:v libx264 -preset ultrafast -c:a aac -shortest test.mp4
$FF -v error -y -f lavfi -i testsrc2=size=320x180:rate=25 -i "$SRC" -map 0:v -map 1:a -c:v libvpx-vp9 -deadline realtime -cpu-used 8 -c:a libopus -shortest test.webm
echo "not audio at all" > bad.txt
head -c 300000 /dev/urandom > garbage.mp3
$FF -v error -y -stream_loop 56 -i stereo44.wav -c:a libmp3lame -b:a 128k long10min.mp3
ls -la "$OUT"
