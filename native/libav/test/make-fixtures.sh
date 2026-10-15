#!/usr/bin/env bash
# Generate decode fixtures for test-decode.mjs from a source wav with a full-featured
# ffmpeg (the build in ~/audio-cleaner/bin/ffmpeg). Formats ffmpeg cannot encode
# (.ape, .mpc) are skipped.
#   make-fixtures.sh [source.wav] [out-dir] [seconds]
set -euo pipefail
FF=${FFMPEG:-$HOME/audio-cleaner/bin/ffmpeg}
SRC=${1:-$HOME/audio-cleaner/dev/live_opus.wav}
OUT=${2:-$HOME/audio-cleaner/src/libav-build/fixtures}
SEC=${3:-120}
mkdir -p "$OUT"
# loop the (short) source up to $SEC seconds, stereo, 48 kHz, s16
BASE=$OUT/_base.wav
[ -f "$BASE" ] || "$FF" -v error -y -stream_loop -1 -i "$SRC" -t "$SEC" -ac 2 -ar 48000 -c:a pcm_s16le "$BASE"
f() {
  local name=$1; shift
  [ -s "$OUT/$name" ] && return 0
  if "$FF" -v error -y "$@" "$OUT/$name" 2>/dev/null && [ -s "$OUT/$name" ]; then echo "made $name"
  else rm -f "$OUT/$name"; echo "SKIP $name (ffmpeg could not make it)"; fi
}
A="-i $BASE"
VID="-f lavfi -t $SEC -i testsrc=size=160x120:rate=25"

# --- the requested set --------------------------------------------------------
f wma.wma       $A -c:a wmav2 -b:a 128k -f asf
f wmv.wmv       $VID $A -map 0:v -map 1:a -c:v wmv2 -b:v 200k -c:a wmav2 -b:a 128k -f asf
f avi_mp3.avi   $VID $A -map 0:v -map 1:a -c:v mpeg4 -b:v 200k -c:a libmp3lame -b:a 128k -f avi
f flv_mp3.flv   $A -c:a libmp3lame -ar 44100 -b:a 128k -f flv
f mpg_mp2.mpg   $A -c:a mp2 -b:a 192k -f mpeg
f ac3.ac3       $A -c:a ac3 -b:a 192k -f ac3
f amr_nb.amr    $A -ac 1 -ar 8000 -c:a libopencore_amrnb -b:a 12.2k -f amr
f wv.wv         $A -c:a wavpack
f tta.tta       $A -c:a tta
f au.au         $A -c:a pcm_s16be -f au
f caf_pcm.caf   $A -c:a pcm_s16le -f caf
f aiff.aiff     $A -c:a pcm_s16be -f aiff
# ape / mpc: no encoder in ffmpeg -> skipped (nothing to generate)

# --- extra coverage -----------------------------------------------------------
f eac3.eac3     $A -c:a eac3 -b:a 192k -f eac3
f dts.dts       $A -c:a dca -strict -2 -b:a 768k -f dts
f mp3.mp3       $A -c:a libmp3lame -b:a 192k
f aac.m4a       $A -c:a aac -b:a 160k -f ipod
f aac_adts.aac  $A -c:a aac -b:a 160k -f adts
f alac.m4a      $A -c:a alac -f ipod
f alac.caf      $A -c:a alac -f caf
f mp4_vid.mp4   $VID $A -map 0:v -map 1:a -c:v mpeg4 -b:v 200k -c:a aac -b:a 128k -f mp4
f opus.mkv      $A -c:a libopus -b:a 96k -f matroska
f vorbis.ogg    $A -c:a vorbis -strict -2 -ac 2 -b:a 128k -f ogg
f opus.ogg      $A -c:a libopus -b:a 96k -f ogg
f flac.flac     $A -c:a flac
f adpcm_ms.wav  $A -c:a adpcm_ms -f wav
f adpcm_ima.wav $A -c:a adpcm_ima_wav -f wav
f w64.w64       $A -c:a pcm_s16le -f w64
f ts_aac.ts     $VID $A -map 0:v -map 1:a -c:v mpeg2video -b:v 200k -c:a aac -b:a 128k -f mpegts
f ts_mp2.ts     $A -c:a mp2 -b:a 192k -f mpegts
f mpg_vid.mpg   $VID $A -map 0:v -map 1:a -c:v mpeg1video -b:v 200k -c:a mp2 -b:a 192k -f mpeg
f amr_3gp.3gp   $A -ac 1 -ar 8000 -c:a libopencore_amrnb -b:a 12.2k -f 3gp
f wav_f32.wav   $A -c:a pcm_f32le
f wav_u8.wav    $A -c:a pcm_u8
ls -l "$OUT"
