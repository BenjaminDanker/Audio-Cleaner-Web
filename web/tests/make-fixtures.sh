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
$FF -v error -y -i "$SRC" -c:a wmav2 noisy.wma  # browsers cannot open it themselves: decoded by libav.js (src/media/libav.js)
$FF -v error -y -f lavfi -i testsrc=size=320x240:rate=25 -i "$SRC" -shortest -c:v mpeg4 -c:a ac3 -b:a 192k noisy.avi  # same, video
$FF -v error -y -f lavfi -i testsrc2=size=640x360:rate=25 -i "$SRC" -map 0:v -map 1:a -c:v libx264 -preset ultrafast -c:a aac -shortest test.mp4
$FF -v error -y -f lavfi -i testsrc2=size=320x180:rate=25 -i "$SRC" -map 0:v -map 1:a -c:v libvpx-vp9 -deadline realtime -cpu-used 8 -c:a libopus -shortest test.webm
# Two audio tracks, as OBS writes them: the mic (voice + the speakers heard back, 40 ms late) and the desktop sound the speakers played.
SPK=${E2E_SPEAKER_WAV:-$(dirname "$SRC")/clean_freesound_33711.wav}
$FF -v error -y -f lavfi -i testsrc2=size=320x180:rate=25 -i "$SRC" -i "$SPK" -filter_complex "[2:a]aresample=48000,asplit[s1][s2];[s1]adelay=40,volume=0.5[e];[1:a]aresample=48000[v];[v][e]amix=inputs=2:normalize=0:duration=first[mic]" -map 0:v -map "[mic]" -map "[s2]" -c:v libx264 -preset ultrafast -c:a aac -shortest echo2.mp4
$FF -v error -y -i echo2.mp4 -map 0:a -c:a flac echo2.mka  # audio only, same two tracks
# Clear speech (DeepFilterNet's clean sample, same licence) for the free in-browser subtitles: as it is, four times in a
# row with short pauses (longer than the 30 s the subtitle maker hears at once), and as a small video.
SPEECH=${E2E_SPEECH_WAV:-$(dirname "$SRC")/clean_freesound_33711.wav}
cp "$SPEECH" speech.wav
$FF -v error -y -i "$SPEECH" -af apad=pad_dur=1.5 -f wav "$OUT/.speech-pad.wav"
$FF -v error -y -stream_loop 3 -i "$OUT/.speech-pad.wav" -c:a pcm_s16le speech-long.wav && rm -f "$OUT/.speech-pad.wav"
$FF -v error -y -f lavfi -i testsrc2=size=640x360:rate=25 -i "$SPEECH" -map 0:v -map 1:a -c:v libx264 -preset ultrafast -c:a aac -shortest speech.mp4
echo "not audio at all" > bad.txt
head -c 300000 /dev/urandom > garbage.mp3
$FF -v error -y -stream_loop 56 -i stereo44.wav -c:a libmp3lame -b:a 128k long10min.mp3
ls -la "$OUT"
