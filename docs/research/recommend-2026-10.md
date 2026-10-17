# Recommended settings: how the thresholds were chosen (2026-10-17)

What it is: right after a file is chosen (Cleanup page), and about ten seconds into a microphone test (Live page, OBS plugin), about a
minute of the sound is measured and the options that fit are suggested and, for files, pre-selected. People can change everything. The
aim is that nobody has to experiment, and that the expensive paths (the stronger noise remover, the voice rebuild, Extra accurate
subtitles) are only suggested when the sound needs them.

Code: `web/src/media/analysis.js` (measuring; one module for the page, a worker, Node and the tests), `web/src/media/analyse.js` (decode
about 60 s spread over the file, in a worker), `web/src/recommend.js` (measurements -> choices and reasons), `web/src/live/advice.js`,
`obs-plugin/core/src/input_advice.cpp` (the same for the plugin, C++, no OBS). Tests: `web/tests/reco-unit.mjs`,
`web/tests/reco-ui.mjs` (real browser, mocked API), `server/tests/test_reco_agreement.py` (browser code and server functions on the same
files), `obs-plugin/tests/core_unit_test.cpp` (`test_input_advice`). Calibration scripts: `tools/reco-calibration/`.

## What is measured, and where it agrees with the server

| measurement | how | agreement with the server |
|---|---|---|
| noise figure | 90th minus 10th percentile of the levels of 10 ms frames, digital silence left out: exactly `pipeline.noise_estimate_db` | 2234 files (1456 mixes, 660 degraded copies, 90 long files, 18 non-speech sounds, the 10 `effiles`): largest difference 0.000004 dB; the same "heavy or not" in 2234 of 2234 |
| bandwidth | long-term spectrum of the active frames, the cliff search of `bwe.effective_bandwidth`, ported line by line (including scipy's zero-padded median filter) | the same cut-off to the Hz on all 2234 files, so the same generator (`bwe.pick_model`) in 2234 of 2234 |
| channel layout | mono, stereo, same on both sides, one side silent: `pipeline.channel_plan` | identical on all 2234 files |
| loudness | BS.1770 integrated loudness | within 0.05 LU of pyloudnorm |
| clipping, speech | share of samples flattened against full scale; how much the level of the speech band (150 Hz to 3.5 kHz) moves | see below |

A long file is not decoded whole: six stretches of 10 s spread evenly over it (`planWindows`), seeking in the container, about a second
for a 6-minute file. Files only libav.js can read are measured from their first 30 s. The noise figure from the six stretches against
the server's figure for the whole file (90 files of 97 s, 45 noise conditions): mean difference 0.46 dB, largest 1.59 dB, 90 % within
0.83 dB.

## "Heavy" is only recommended where the server would pick it

The stronger noise remover is chosen by the server when the noise figure is below `HEAVY_BELOW_DB` = 19. The recommendation says "much
noise, Clean & polish" only below 19 - 1.5 = 17.5, so the sampling error cannot push it across the server's own cut. Results:

* 552 mixes (whole 8 s clips): recommended heavy for 148, the server picks heavy for 148 of them (100 %). The server picks heavy for 170
  mixes; the 22 between 17.5 and 19 get the free cleanup in the browser (which only has the lighter remover) with the strength for a
  noisy recording; if the person chooses Clean & polish instead, the server's "automatic" picks the stronger remover for them itself.
* 90 long files, sampled: heavy recommended for 25, the server agrees in 25 (100 %); it picks heavy for 27.
* First 10 s of the same files (the live window): heavy for 21, the server agrees in 21.

## Does the recommendation match the best choice?

Ground truth is the server's own measurement. 552 mixes: 12 LibriSpeech clips x 7 kinds of noise (white, pink, babble, traffic, cafe,
50 Hz hum, keyboard clicks) x SNR 0..30 dB, plus the 12 clean clips; clean reference known. For every mix, the server's noise removers
(DeepFilterNet3 at attenuation limits 12/24/40/100 dB, DPDFNet-8 at the same four) were run and measured: SI-SDR against the clean clip,
and the noise left in the pauses relative to the speech level.

The "best choice" per mix:

* off: neither remover gains at least 2 dB SI-SDR (1 and 3 dB are reported too);
* heavy: DPDFNet-8 gains more than DeepFilterNet3 by more than 0.3 dB; otherwise light.

| threshold for "worth removing" | 3-way agreement (off / light / heavy) | stationary noise only | remove or not | "always remove, server auto" |
|---|---|---|---|---|
| 1 dB | 51.8 % | 59.0 % | 74.8 % | 47.6 % |
| 2 dB | 51.4 % | 60.2 % | 72.8 % | 43.1 % |
| 3 dB | 53.6 % | 63.6 % | 73.0 % | 39.3 % |

Where it goes wrong, honestly: what the noise is made of is not visible in the levels. Babble (35 % agreement) is hard for the noise
removers themselves (a gain of 2 dB at best), so the best choice is often "off" while the figure says "noisy"; keyboard clicks (15 %)
leave the quiet parts quiet, so the figure says "clean" while the removers do gain 5 to 10 dB. White and pink noise: 51 % and 79 %.
Light against heavy is the server's own cut and no better than 55 to 65 % on these mixes; the recommendation follows it on purpose.

The cut for "clean" (nothing to remove) was swept from 22 to 44 dB: the remove-or-not accuracy is flat at 73 to 74 % from 24 to 30 dB and
falls on both sides; 29 was taken.

### Strength

The gentlest of Gentle / Balanced / Strong / Maximum (12 / 24 / 40 / 100 dB) that is within 0.75 dB SI-SDR and 4 dB of pause noise of
what Maximum achieves, on the remover the server would use for that mix. The mapping by noise figure that scored best on the 332 mixes
where noise removal is recommended: Maximum below 18, Strong below 21, Balanced below 27, Gentle from 27 up to 29 (clean above).

| | never under-cleans | exact | within one step | stronger than needed |
|---|---|---|---|---|
| recommended | 88.3 % | 41.6 % | 75.9 % | 46.7 % |
| always Maximum (what the form did) | 100 % | 22.9 % | 43.7 % | 77.1 % |
| always Balanced | 56.3 % | 28.0 % | 77.1 % | 28.3 % |

Gentler steps keep more of the room; the measurements say they are as good for most mixes at the light end. The choice of noise
tolerance matters (a looser one gives more "Gentle", a tighter one more "Maximum"); the numbers above use the one stated.

### Subtitles

Standard (Parakeet) against its own transcript of the clean clip, 16 clips x 28 conditions: stationary noise hardly matters (white
noise at 0 dB SNR changes 9.5 % of the words, traffic 2.4 %); babble and cafe noise at 10 dB or worse change 10 to 70 %; speaking 1.9
times faster changes 5.5 % (1.3x: 3.6 %). So speed is not a reason for the dearer kind and is not used. By noise figure, the share of
words that change: below 12: 21 %, 12 to 15: 23 %, 15 to 18: 9 %, 18 to 21: 5 %, 21 to 30: 1.5 to 7 % (babble and cafe noise about
10 % there, stationary noise 1 to 2 %). Extra accurate is recommended below 18.
Of 288 noisy clips, Standard changes more than 10 % of the words in 75; Extra accurate is recommended for 109: recall 65 %, precision
45 %. The Extra accurate service itself could not be measured here; "it copes better with noise" is the product's claim, not a number
from this study.

Languages: Standard (Parakeet v2/v3) covers English and 24 European languages; for other languages the form recommends Extra accurate
once the person has picked the language (the language of a file is not guessed from its levels).

## Highs, echo, muffled voices, speech

* Restore highs / phone quality: the server restores highs for anything its generator list covers (bandwidth up to 8.8 kHz, e.g. all
  16 kHz-sampled material). The recommendation only calls a file "phone-like" and moves it to Clean & polish below 6.6 kHz (the 8k and
  12k generators): 40/40 AMR phone recordings, 40/40 steep low-pass at 3.4, 4 and 5.5 kHz, 0/86 clean 16 kHz sources, 0/60 filtered at 7
  and 9.5 kHz, 0/60 G.722 / MP3 32 kbps / Opus 12 kbps (their bandwidth is 7.7 to 8.2 kHz).
* Echoey room (rebuild the voice): the 95th percentile of how far the level falls within 60 ms, from the loud parts. Close-up speech:
  about 20 dB and more, a room with a half second of ringing: 8 to 9 dB. Recommended below 10.5 dB when the background is at least 30 dB
  below the voice: reverb of RT60 0.15 s 0/40, 0.3 s 6/40, 0.5 s 32/40, 0.7 s 36/40, 1.0 s 36/40, 1.4 s 34/40; 0 false alarms on 86
  clean clips and 1 of 1370 noisy mixes. (Synthetic exponential-decay room responses; no real rooms were measured.)
* Muffled voice (rebuild): the top of the speech band (4 to 8 kHz) 19 dB or more below its middle, without a sharp cut-off. A one-pole
  low-pass at 1.2 kHz: 23/40, 2 kHz: 14/40, 3 kHz: 6/40; 0 false alarms on 86 clean clips. Weak on purpose.
* The rebuild is only pre-selected when one of these two fires, never for noise alone; the reasons say "listen before you publish".
  Sidon can change a word on very echoey recordings (docs/research/restore-sidon.md).
* Speech at all: the level of the 150 Hz to 3.5 kHz band moves by at least 6.8 dB (90th minus 10th percentile of 10 ms frames).
  1249 of 1252 speech clips (SNR 0 to 30 dB and clean) pass; of 18 sounds that are not speech (white, pink and brown noise, a fan, air
  conditioning, 50 Hz hum, a tone, a synthetic melody, silence) one fails: a series of loud bangs. Real music was not available.

## Live (website and OBS plugin)

After 10 s of microphone the same noise figure gives the strength (or "leave noise removal off" in a quiet room), and the caption level.
Against the whole files of the long-file set, the first 10 s give the same class (off / light / heavy) in 91.1 % of cases and the same
strength in 86.7 %; never "heavy" where the server would not pick it. The plugin's C++ analyser gives the website's figures to 0.00005
dB on the same audio; a test (`reco-unit.mjs`) checks that its thresholds are the website's.

Echo (speakers heard in the microphone): the share of the microphone's energy that the played sound explains, best over delays up to
400 ms, on a 2 kHz copy. Another talker's recorded voice mixed in at 0.1 / 0.2 / 0.4 of the amplitude after 150 ms: explained 0.005 to
0.014 / 0.027 to 0.052 / 0.115 to 0.176; unrelated speech pairs: at most 0.008; the threshold is 0.03 (an echo about 14 dB below the
voice). In the plugin it compares the microphone with the echo reference (the desktop sound); without a reference nothing can be said.
In the browser the played sound is our own cleaned copy of the microphone, so only delays of 80 ms and more are searched and only
when that copy was really played (a hum or a tone in the room would otherwise "explain" itself): the browser already cancels echo, so
this mainly catches speakers that its own canceller does not cope with.

## Not verified

* The noise is synthetic or babble made of the same read speech; no recordings of real cafes, streets, fans or rooms. Real noise is
  usually less stationary than the stationary kinds here and less speech-like than babble.
* The source speech is LibriSpeech (read English, 16 kHz source, upsampled): everything is "band-limited at 8 kHz" for the server's own
  bandwidth rule. Full-band speech was only available as two synthetic voices and two long clips.
* "Best choice" is SI-SDR and pause noise, not listening. Preference for gentler or stronger removal is a matter of taste.
* The Extra accurate subtitles and the voice rebuild were not run. Sidon was not measured against the thresholds above.
* `core_loopback_test` has two timing-dependent checks about service warnings (`translation_unavailable`, `captions_unavailable`
  arriving right after `ready`, around lines 977 and 998): they fail on the unmodified branch too when that group is repeated (12
  failures in 6 repeats), and in full runs they pass or fail with the machine's load. The warning probably arrives before the network
  thread has finished handling `ready`, which then clears it. Not touched here.
