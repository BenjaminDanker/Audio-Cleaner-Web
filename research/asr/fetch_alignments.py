#!/usr/bin/env python
"""Fetch word-level forced alignments (Montreal Forced Aligner TextGrids) for our 150 LibriSpeech test-clean
utterances from 'LibriSpeech Alignments' (Zenodo 2619474, CC BY 4.0) by range-reading single members of the
623 MB zip. Output: eval/align_clean.json = {utt_id: [[word, start, end], ...]} (ground truth for word-timestamp
accuracy; MFA boundaries are themselves only ~10-20 ms accurate). Raw TextGrids -> data/raw/alignments/.
Run: ~/audio-cleaner/research/venv/bin/python -I fetch_alignments.py
"""
import io, json, re, sys, zipfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from fetch_demand import RangeFile
from common import EVAL, ROOT, load_manifest

RAW = ROOT / "data/raw/alignments"


def parse_words(tg: str):
    """Words tier of a Praat long-format TextGrid -> [[word, xmin, xmax]] (silences '' skipped)."""
    tier = tg.split('name = "words"', 1)[1].split('name = "phones"', 1)[0]
    out = []
    for m in re.finditer(r'xmin = ([\d.]+)\s+xmax = ([\d.]+)\s+text = "([^"]*)"', tier):
        a, b, w = float(m.group(1)), float(m.group(2)), m.group(3)
        if w.strip():
            out.append([w.lower(), a, b])
    return out


def main():
    RAW.mkdir(parents=True, exist_ok=True)
    ids = [r["id"] for r in load_manifest("clean")]
    rf = io.BufferedReader(RangeFile("https://zenodo.org/api/records/2619474/files/librispeech_alignments.zip/content"),
                           buffer_size=1 << 16)
    out = {}
    with zipfile.ZipFile(rf) as zf:
        for uid in ids:
            spk, chap, _ = uid.split("-")
            name = f"test-clean/{spk}/{chap}/{uid}.TextGrid"
            cached = RAW / f"{uid}.TextGrid"
            if cached.exists():
                txt = cached.read_text()
            else:
                txt = zf.read(name).decode()
                cached.write_text(txt)
            out[uid] = parse_words(txt)
    json.dump(out, open(EVAL / "align_clean.json", "w"))
    print("alignments for", len(out), "utterances;", sum(len(v) for v in out.values()), "words")


if __name__ == "__main__":
    main()
