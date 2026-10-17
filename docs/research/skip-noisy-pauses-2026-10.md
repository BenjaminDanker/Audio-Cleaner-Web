# Skipping noisy pauses in file denoising: evaluated, not implemented (2026-10-17)

`pipeline.sound_spans` already leaves digital silence (>= 1.5 s, below -90 dBFS) out of the DeepFilterNet3 run.
Question: can long pauses that hold room noise be left out too, and filled so that the result is indistinguishable
from a full run? The premise was that the model's own output in such pauses is near silent (about -72 dBFS).

Answer: only for broadband hiss, and then the saving is small. With real noise it is negligible. The prototype is
in `research/pauses/` (not wired into the pipeline); this is why it is not.

## Method

* Test recordings (`research/pauses/make_corpus.py`, `make_demand_corpus.py`): the 73 LibriSpeech dev-clean
  utterances of `scratchpad/libri` in random order (8 min of speech), pauses between them (40% of them 1.5-6 s,
  the rest 0.25-1 s: far more long pauses than a typical recording), over continuous noise at a chosen level.
  Ground truth for "speech" comes from the clean utterances (10 ms frames within 28 dB of the speech peak, and a
  looser 38 dB version). 12 synthetic noises (hiss at -40/-50/-60/-70 dBFS, pink, hum, babble, clicks, HVAC
  rumble, soft and whispered speech) and DEMAND real noise (living room, office, kitchen at 15, 25, 35 dB SNR).
* Skipping: Silero VAD (the `silero_vad.onnx` of models/, one batched pass; threshold 0.1, 0.4 s pad) finds
  stretches without speech; of each, 0.5 s on each side is kept and the middle (a "hole", on the 10 ms hop grid)
  is cut out of the signal the model sees; the model runs on that one continuous signal; the hole is filled.
* Reference: the same file through `dfn.denoise_array` in one contiguous pass (threads=1). SNR is
  sum(ref^2) / sum((ref - test)^2) over the whole file, DFN3, `atten_lim_db` 100, no polish.

## 1. The model leaves hiss, not other noise, near silent

Pause level, input and model output (more than 0.5 s from speech), with speech at -23 dBFS (rms) unless noted:

| noise | in | out | removed |
|---|---|---|---|
| hiss, -40 / -50 / -60 dBFS | -40.0 / -50.0 / -60.0 | -70.4 / -77.2 / -84.8 | 30 / 27 / 25 dB |
| hiss -70 dBFS (speech -33) | -70.0 | -94.9 | 25 dB |
| HVAC rumble + hiss, -45 | -45.0 | -63.6 | 19 dB |
| mains hum, -33 | -33.0 | -53.0 | 20 dB |
| pink, -38 / -23 | -38.4 / -23.2 | -44.5 / -30.5 | 6 / 7 dB |
| keyboard clicks, -35 | -35.1 | -41.9 | 7 dB |
| babble, -28 | -27.9 | -37.7 | 10 dB |
| pink under soft speech (-45) | -64.6 | -69.9 | 5 dB |

So "-72 dBFS" holds for steady broadband hiss only; everything else leaves a residual 20-50 dB louder, which a
zero or scaled-original fill cannot reproduce sample for sample.

## 2. Accuracy of the cut-out

Against the contiguous run, the fraction skipped is 10-17% of these recordings:

| how the hole is cut | SNR (hiss -50 dBFS) |
|---|---|
| spans processed separately, as the digital-silence skip does (cold model after each hole) | 25.5 dB |
| holes spliced out, model runs on the spliced signal | 40.2 dB |
| ... and holes on the 10 ms hop grid (frames after a hole keep their phase) | **50.2 dB** |

Fill: zero and "original scaled by the attenuation the model applied just outside the hole (linear between the two
sides), with 40 ms fades" give the same SNR where the hole is quiet (hiss: 49.5-50.2 dB), and scaled is 3-6 dB
better where it is not (pink 27.1 -> 30.9, clicks 23.9 -> 27.0, whispered 15.9 -> 19.1). Level step at the joins
(20 ms before vs after, relative to the full run, floored at -120 dBFS): 8-13 dB with the scaled fill, 21-39 dB
(down to the floor) with zero; all of it below -75 dBFS in the quiet cases, so neither is audible. Scaled is also
the safer failure: speech the VAD missed would survive at the model's attenuation instead of being deleted.

Without a gate, per recording (same comparison, zero / scaled fill): hiss -50/-60/-70: 50.2 / 49.6 / 47.9 dB;
hiss -40: 44.8; HVAC: 41.9; hum: 36.6 / 34.4; soft speech under pink: 29.5 / 35.1; pink 15 dB: 27.1 / 30.9; clicks:
23.9 / 27.0; whisper at 5 dB SNR: 15.9 / 19.1; pink at 0 dB SNR: 14.5 / 18.6. Only the first three reach 45 dB.

A gate that predicts the pause residual before cutting (probes: the model run on 3 loud holes with 6 s of lead,
level in / out; predicted output must be 52 dB under the speech level; holes with an event in them stay) opened
exactly for those: hiss -50 / -60 / -70 skipped 13.0 / 11.5 / 9.6% at 51.5 / 51.7 / 49.5 dB (the last with the 60 s minimum of candidate holes relaxed), and stayed shut for
all the others (including hiss -40 at 44.8 and HVAC at 41.9). 

## 3. Speech is never skipped

Over the 12 recordings, truth-speech samples inside any candidate hole: 0 (looser 38 dB truth too). The VAD's
misses are real: at threshold 0.5 whispered speech was missed by up to 1.05 s, at 0.1 by at most 0.61 s; with
0.4 s of pad and 0.5 s of kept pause a hole starts at least 0.9 s from anything the VAD calls speech, and the
nearest truth speech to any hole was 0.35 s.

## 4. Why it is not implemented

1. **The saving is small.** On real noise (DEMAND living room, office, kitchen at 15, 25, 35 dB SNR) the gate
   allows at most 21 s (3%) of a 10-minute recording to be skipped (office at 15 dB), the other eight files
   between 0 and 8 s: the model does not remove those noises. On steady hiss with an unusually pause-rich recording
   (11-13% skippable) the CPU saving measured at 4 threads was 4.1-5.9% (79.2 -> 74.5 s, 77.4 -> 74.3 s, medians of
   three) because the VAD pass (4 s) and the probes (3 s) are paid back out of it.
2. **Wall time gets worse.** The VAD and the probes run on one thread while the model run is spread over four:
   23.2 -> 27.1 s (+17%) on the same recordings.
3. **The strict criterion is not met in the production configuration.** Against the full run as the pipeline
   makes it (4 threads, 60 s segments) the skipped result is 40.8 / 41.6 dB (against the contiguous run: 46.7 dB for
   the second recording). That is the size of the difference any segmentation change makes: the production run
   differs from a contiguous run by 41.5 dB, 30 s segments by 35.6 dB, 45 s by 43.6 dB, and merely delaying the
   input by 100 samples gives 42.0 dB (1 sample: 63.9, one hop: 51.4). It is not audible, but it is under 45 dB.
4. The gate predicts from a probe; a wrong prediction is silent, and the VAD is a new dependency of `acw_common`.

Reconsider if real recordings turn out to be hiss-dominated with a lot of idle time (an unattended meeting
recording): then the VAD and probe passes should run in the thread pool and the min-saving test should use the
measured hole total. `research/pauses/` has the prototype (`pauses.py`: VAD, holes, gate, splice view, fill) and the
scripts that produced the numbers.
