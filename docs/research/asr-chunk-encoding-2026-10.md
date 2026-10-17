# Encoding of the speech chunks sent to the transcription API (2026-10-17)

Premium file transcription cuts the speech into VAD segments of at most 15 s (`runner._write_asr_chunks`) and
`acw-web` posts each to `gpt-transcribe`. Until now each chunk was `ffmpeg -c:a libmp3lame -b:a 256k` at 16 kHz mono
(libmp3lame tops out at 160 kbps at that rate: measured 163 kbps) and one ffmpeg process per chunk.

## Accuracy against the real API

70 LibriSpeech dev-clean utterances of at most 15 s (418 s, 1005 words), encoded exactly as the runner does (float32
pipe into ffmpeg), sent with `languages[]=en`. WER against the references after lower-casing, punctuation removal and
Mr/Mrs/Dr expansion (the remaining ~3.4% is mostly the model writing numbers and names differently from the
references, the same for every variant). Cost of the whole experiment: about $0.41 (13 runs of 418 s at $0.0045/min).

| encoding | bytes (418 s) | WER clean | WER with noise at 5 dB SNR |
|---|---|---|---|
| mp3 256k (-> 163 kbps, the old one) | 8.54 MB | 3.68% (3.68% on a repeat: the API is deterministic here) | 11.74% |
| mp3 64k | 3.42 MB | 3.48% | 11.44% |
| mp3 48k | 2.56 MB | 3.38% | 11.54% |
| mp3 32k | 1.71 MB | 3.18% | not run |
| opus (ogg, voip) 32k | 1.68 MB | 3.38% | not run |
| opus 24k | 1.27 MB | 3.28% | 12.54% |
| opus 16k | 0.85 MB | 3.38% | 14.83% |

The API accepts ogg/opus, but under noise Opus loses accuracy below 32 kbps, so MP3 stays (the server also insists
on `.mp3` chunk names). Every MP3 rate down to 32 kbps is within noise on clean speech (2-5 words fewer errors than the old files in
1005, which is noise, not a gain); with noise 64k and 48k are the same as 256k. **64 kbps** was chosen: the same accuracy as
the old files with a safety margin over 48 kbps, 2.5x fewer bytes to upload.

## Fewer processes

One ffmpeg process per chunk costs ~55 ms of CPU of which the encode itself is a small part. One ffmpeg run for all
chunks (`asplit` + `atrim` over only the speech samples) is not the best either: each trim filter sees the whole
run's audio, so the cost grows with the square of the chunk count. 504 chunks (70 min of audio, 42 min of speech,
the 10-minute LibriSpeech stream x6 with gaps, taskset 6,7,14,15, nice 10):

| | ffmpeg runs | CPU (children) | wall | bytes |
|---|---|---|---|---|
| before: 256k, one process per chunk | 504 | 30.1-34.9 s | 29-35 s | 52.2 MB |
| 64k, one process per chunk | 504 | 34.2 s | 34 s | 20.9 MB |
| 64k, all 504 in one run | 1 | 83.1 s | 59 s | 20.9 MB |
| 64k, 64 per run | 8 | 20.1 s | 14.2 s | 20.9 MB |
| **64k, 32 per run (implemented)** | 16 | **17.5 s** | **12.7 s** | 20.9 MB |
| 64k, 16 per run | 32 | 17.7 s | 13.3 s | 20.9 MB |

So `FFmpeg.write_audio_chunks` encodes 32 chunks per run: half the CPU and a third of the wall time of the per-chunk
processes, 60% fewer bytes. The files are byte-identical to the per-chunk 64k files (checked on 70 chunks), so the
accuracy figures above apply as measured; decoded length differs from the span by at most 15 samples and there is
no offset. (Times on this shared machine vary by +-15% from run to run; the ordering did not.)
