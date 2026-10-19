# Automatic settings model

The trees behind "Automatic best settings" for files: the website (`web/src/reco-model.js`, data in
`web/src/reco-model-data.js`) and the desktop app (`common/acw_common/reco_model.py`, data in
`common/acw_common/reco_model.json`, the same trees). Gradient-boosted trees (LightGBM) on the 14 measurements
`web/src/media/analysis.js` already takes (the desktop takes them with `common/acw_common/analysis.py`, a port that follows
the browser code step by step), so nothing extra is downloaded and a decision takes well under a millisecond.

## What it decides, and from what

| Decision | Trained on (label) | In the product |
|---|---|---|
| Noise strength (gentle, balanced, strong, max) | the site's own noise remover run at all four strengths; label = the gentlest one whose PESQ-WB against the clean original is within 0.05 of the best | the model |
| Voice rebuild | recordings damaged on purpose: echo (RT60 0.5 s or more), very small files (Opus 8 kbit/s or less, MP3 16 kbit/s or less), clipping | the model **or** the echoey/muffled rules: rebuild sounded better on nearly every recording in the listening test, so it is chosen more often than the labels say |
| Restore highs | the high sounds cut off at 7 kHz or lower | **not used any more**: the product has no Restore highs step (a recording that is missing its highs gets the voice rebuilt, a trigger of its own: phone-quality sound). The head is still trained here, and `export_web.py` leaves it out of the product's files (format version 3) |
| Heavy noise remover | (only the server's 19 dB rule, so the model could only copy it) | the rule (`web/src/recommend-room.js`, `pipeline.HEAVY_BELOW_DB`) |

Live ("Test my room") stays on the rules in `web/src/recommend-room.js`, which the OBS plugin shares.

The desktop app decides the same way (`reco_model.recommend`, the Python version of `web/src/recommend.js`): the strength, the
voice rebuild (or echoey, muffled, phone-quality or distorted) from the trees when the settings are left on automatic, the heavy
noise remover by the rule. Settings the user chose themselves are never changed.

## Results (test split, 1002 clips, speakers not seen in training)

| Decision | Model | Rules before |
|---|---|---|
| Strength exactly right | 73.5% | 15.7% |
| Strength within one step | 93.2% | 31.0% |
| PESQ lost against the best strength | 0.065 | 0.642 |
| Voice rebuild right | 80.6% | 44.5% |
| Restore highs right (rebuilt when not needed) | 90.0% (2.2%) | 85.7% (8.2%) |

The rules sent 537 of the 672 clips where Gentle was as good as anything to Maximum.

Note: the product no longer uses the restore-highs head, so `decide.mjs`, `rules.mjs`, `check_js.py` and `listen_build.py` (which compare the old web code's
`restoreHighs` with this training run) describe the earlier version of the website and no longer run against the current `web/src`.

## Data and licences

- Speech: LibriTTS-R dev-clean/test-clean (OpenSLR 141 says CC BY 4.0; the Hugging Face mirror used, mythicinfinity/libritts_r,
  says cc-by-4.0 in its card but its citation points at a BY-NC-ND licence: confirm before redistributing anything made from it).
  Splits: train 2400, val 402, test 1002 clips, speakers disjoint.
- Real recordings for checks: Common Voice 17.0 English test (CC0).
- DNSMOS P.835 (MIT) for the real-recording check.
- Simulation (`gen.py`): muffled microphone (30%), room echo (70%, RT60 0.15 to 1.2 s), codecs (low-pass, G.711, Opus, MP3, AAC),
  noise at -3 to 30 dB SNR (75%), clipping (12%).

## Running it

The scripts expect the working tree at `~/audio-cleaner-train` (venv with lightgbm, pesq, soundfile, scipy, onnxruntime) and this
repository at `~/repos/acw-desktop-merge` (they import the web measurement code from it). CPU only.

    bash run_split.sh <split> <count>          # generate, measure, current rules, labels (train / val / test)
    venv/bin/python train.py                   # out/reco-model.json, out/test_report.json
    venv/bin/python check_js.py                # the JS evaluator agrees with LightGBM
    python3 export_web.py out/reco-model.json      # writes web/src/reco-model-data.js and common/acw_common/reco_model.json
    python3 export_web.py --check                  # exit status 1 when either file is not what the export would write
    venv/bin/python listen_build.py choose; ...; venv/bin/python listen_build.py render   # the listening page

`rules.mjs` runs the repository's `recommendForFile`, which now uses this model: to compare against the old rules, run it on a
checkout from before the model went in.
