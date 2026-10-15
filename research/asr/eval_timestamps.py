#!/usr/bin/env python
"""Word-timestamp accuracy vs. Montreal-Forced-Aligner ground truth (LibriSpeech Alignments, CC BY 4.0).

Usage: python -I eval_timestamps.py results/words/<tag>_clean.json [...]   (produced by bench_offline.py --ts)
Hypothesis words are lower-cased / stripped of punctuation and matched to the reference words with difflib
(only exactly matching words are scored). Reports, over the utterances present in ALL given files: matched words,
mean/median absolute error of word start and end (ms), signed bias of start, and % of words within 100 / 200 ms.
"""
import difflib, json, re, sys
from pathlib import Path
import numpy as np

ALI = json.load(open(Path.home() / "audio-cleaner/research/data/eval/align_clean.json"))


def clean(w):
    return re.sub(r"[^a-z0-9']", "", w.lower())


def score(words_by_utt, ids):
    ds, de = [], []
    n_ref = 0
    for u in ids:
        ref = [(clean(w), s, e) for w, s, e in ALI[u] if clean(w)]
        hyp = [(clean(w), s, e) for w, s, e in words_by_utt[u] if clean(w)]
        n_ref += len(ref)
        sm = difflib.SequenceMatcher(None, [r[0] for r in ref], [h[0] for h in hyp], autojunk=False)
        for tag, i1, i2, j1, j2 in sm.get_opcodes():
            if tag == "equal":
                for i, j in zip(range(i1, i2), range(j1, j2)):
                    ds.append(hyp[j][1] - ref[i][1]); de.append(hyp[j][2] - ref[i][2])
    ds, de = np.array(ds) * 1000, np.array(de) * 1000
    return dict(n_ref=n_ref, matched=len(ds), start_mae=np.abs(ds).mean(), start_med=np.median(np.abs(ds)),
                start_bias=ds.mean(), end_mae=np.abs(de).mean(), end_med=np.median(np.abs(de)),
                s100=100 * (np.abs(ds) <= 100).mean(), s200=100 * (np.abs(ds) <= 200).mean(),
                e100=100 * (np.abs(de) <= 100).mean(), e200=100 * (np.abs(de) <= 200).mean())


if __name__ == "__main__":
    data = {Path(p).name.replace("_clean.json", ""): json.load(open(p)) for p in sys.argv[1:]}
    ids = sorted(set.intersection(*[set(d) for d in data.values()]))
    print(f"{len(ids)} common utterances")
    print(f"{'engine':28s} {'match':>6s} {'start MAE':>9s} {'med':>5s} {'bias':>6s} {'<100ms':>6s} {'<200ms':>6s} | {'end MAE':>7s} {'med':>5s} {'<100ms':>6s} {'<200ms':>6s}")
    for tag, d in data.items():
        r = score(d, ids)
        print(f"{tag:28s} {r['matched']:6d} {r['start_mae']:9.0f} {r['start_med']:5.0f} {r['start_bias']:6.0f} {r['s100']:6.1f} {r['s200']:6.1f} | {r['end_mae']:7.0f} {r['end_med']:5.0f} {r['e100']:6.1f} {r['e200']:6.1f}")
