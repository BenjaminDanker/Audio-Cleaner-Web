"""Writes the fixtures that tests/test_analysis.py and tests/test_reco_model.py compare the Python code with: what the
website's own code (web/src/media/analysis.js, web/src/reco-model.js, web/src/recommend.js) says about
  - the made-up recordings of analysis_signals.py                      -> fixtures/analysis_js.json
  - a few hundred sets of measurements (real and made up)              -> fixtures/reco_model_js.json
Needs node and the website's noise remover files (see web/tests/reco-cli.mjs). Run it again only when the website's code
or the trees change:

    python common/tests/make_reco_fixtures.py [measurements.jsonl ...]

The jsonl files are lines of Analyser.result() (web/tests/reco-cli.mjs prints them), the real part of the second fixture; given
none, the real rows of the fixture already there are kept (and decided again by the website's code as it is now). The other rows
are made up: random ones across the ranges, and "edge" ones on both sides of every threshold of the decision."""
from __future__ import annotations

import json
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(HERE))
import analysis_signals as sig  # noqa: E402

FIELDS = ("seconds", "noiseDb", "floorDb", "levelDb", "peakDb", "silentFraction", "speechFraction", "bandDynDb", "hasSpeech",
          "bandwidthHz", "tiltDb", "dropDb", "clipFraction", "peak", "loudnessLufs")

JS_DECIDE = """
import { readFileSync } from 'node:fs';
import { modelDecision } from '%(web)s/src/reco-model.js';
import { recommendForFile } from '%(web)s/src/recommend.js';
const rows = JSON.parse(readFileSync(process.argv[2], 'utf8'));
// Why the voice is rebuilt: recommendForFile says every cause it heard, in this order, in its own sentence (src/recommend.js).
const WHY = [['echoey', /^The voice sounds echoey/], ['muffled', /^The voice sounds muffled/], ['phone', /^It sounds like a phone call/],
  ['distorted', /^The sound distorts/], ['unclear', /^The voice has lost some of its clarity/]];
const whyOf = (r) => (r.rebuild ? WHY.filter(([, re]) => r.reasons.some((t) => re.test(t))).map(([k]) => k) : []);
const out = rows.map((stats) => {
  const m = modelDecision(stats);
  const r = recommendForFile(stats, { hasAudio: true, duration: stats.seconds, channels: 1 });
  const why = r ? whyOf(r) : [];
  return { model: { strength: m.strength, rebuild: m.rebuild, p: m.p },
    reco: r && { strength: r.strength, rebuild: r.rebuild, heavy: r.heavy, noSpeech: r.noSpeech,
      rebuildReason: why[0] ?? null, rebuildWhy: why } };
});
console.log(JSON.stringify(out));
"""


def js_decide(rows: list[dict]) -> list[dict]:
    with tempfile.TemporaryDirectory() as d:
        (Path(d) / "rows.json").write_text(json.dumps(rows))
        (Path(d) / "decide.mjs").write_text(JS_DECIDE % {"web": str(ROOT / "web")})
        out = subprocess.run(["node", str(Path(d) / "decide.mjs"), str(Path(d) / "rows.json")], capture_output=True, text=True, check=True)
    return json.loads(out.stdout)


def analysis_fixture() -> dict:
    with tempfile.TemporaryDirectory() as d:
        files = {}
        for name, x in sig.signals().items():
            files[name] = Path(d) / f"{name}.wav"
            sig.write_wav(files[name], x)
        out = subprocess.run(["node", str(ROOT / "web" / "tests" / "reco-cli.mjs"), "--sampled", *map(str, files.values())],
                             capture_output=True, text=True, cwd=ROOT / "web", check=True)
    rows = {Path(json.loads(line)["file"]).stem: json.loads(line) for line in out.stdout.splitlines()}
    stats = {name: {k: rows[name].get(k) for k in FIELDS} | {"segments": rows[name]["segments"]} for name in files}
    decided = js_decide(list(stats.values()))
    return {name: {"stats": stats[name], **decided[i]} for i, name in enumerate(files)}


def random_stats(n: int, seed: int = 5) -> list[dict]:
    """Made-up measurements across (and a little beyond) the ranges the trees were trained on, some of them missing."""
    rng = np.random.default_rng(seed)
    rows = []
    for _ in range(n):
        floor = rng.uniform(-90, -20)
        peak_db = floor + rng.uniform(5, 60)
        row = {
            "seconds": float(rng.choice([1.2, 1.9, 2.0, 5, 10, 59.9])), "noiseDb": float(rng.uniform(-25, 45)),
            "floorDb": floor, "levelDb": floor + (peak_db - floor) * rng.uniform(0.5, 1), "peakDb": peak_db,
            "silentFraction": float(rng.choice([0, 0, 0.02, 0.3, 0.8])), "speechFraction": float(rng.uniform(0, 1)),
            "bandDynDb": float(rng.uniform(0, 40)), "hasSpeech": bool(rng.random() < 0.85),
            "bandwidthHz": float(rng.choice([3800, 4400, 5600, 7000, 7800, 9000, 11000, 24000, rng.uniform(3000, 24000)])),
            "tiltDb": float(rng.uniform(-60, 5)), "dropDb": float(rng.uniform(0, 40)),
            "clipFraction": float(rng.choice([0, 0, 0, 0.0005, 0.003, 0.02])), "peak": float(rng.choice([1.0, 0.99, rng.uniform(0.05, 1)])),
            "loudnessLufs": float(rng.uniform(-50, -5)),
        }
        for k in ("tiltDb", "dropDb", "loudnessLufs"):  # missing sometimes, as the browser's are
            if rng.random() < 0.15:
                row[k] = None
        if rng.random() < 0.05:
            row["noiseDb"] = None
        if rng.random() < 0.05:
            row["floorDb"] = row["levelDb"] = row["peakDb"] = None
        rows.append(row)
    return rows


def edge_stats() -> list[dict]:
    """Measurements on both sides of every threshold the decision has of its own (src/recommend-room.js T): a phone call is a
    bandwidth of 6600 Hz or lower, an echo a fall under 10.5 dB, a muffled voice a tilt of -19 dB or lower (not for a phone call),
    distortion more than 0.002 clipped; and the tilt and the fall missing."""
    base = {"seconds": 20.0, "noiseDb": 30.0, "floorDb": -60.0, "levelDb": -30.0, "peakDb": -26.0, "silentFraction": 0.05,
            "speechFraction": 0.5, "bandDynDb": 20.0, "hasSpeech": True, "bandwidthHz": 24000.0, "tiltDb": -12.0, "dropDb": 30.0,
            "clipFraction": 0.0, "peak": 0.7, "loudnessLufs": -22.0}
    return [{**base, "bandwidthHz": bw, "clipFraction": clip, "dropDb": drop, "tiltDb": tilt}
            for bw in (6599.9, 6600.0, 6600.1, 24000.0)
            for clip in (0.0, 0.002, 0.0020001)
            for drop in (None, 10.4999, 10.5, 30.0)
            for tilt in (None, -19.0001, -19.0, -18.9999, -12.0)]


def model_fixture(real_files: list[str]) -> list[dict]:
    """Rows are {"kind": real | made-up | edge, "stats", "model", "reco"}."""
    real = []
    for f in real_files:
        for line in Path(f).read_text().splitlines():
            if line.strip():
                r = json.loads(line)
                real.append({k: r.get(k) for k in FIELDS})
    rng = np.random.default_rng(3)
    if len(real) > 200:
        real = [real[i] for i in sorted(rng.choice(len(real), 200, replace=False))]
    if not real_files:  # keep the real rows of the fixture there (before it had "kind", its last 300 rows were the made-up ones)
        old_path = HERE / "fixtures" / "reco_model_js.json"
        old = json.loads(old_path.read_text()) if old_path.is_file() else []
        real = [{k: r["stats"].get(k) for k in FIELDS} for r in (old[:-300] if old and "kind" not in old[0] else
                                                                 [r for r in old if r["kind"] == "real"])]
    kinds = ["real"] * len(real)
    rows = real + random_stats(300)
    kinds += ["made-up"] * 300
    edges = edge_stats()
    rows += edges
    kinds += ["edge"] * len(edges)
    decided = js_decide(rows)
    return [{"kind": k, "stats": s, **d} for k, s, d in zip(kinds, rows, decided)]


if __name__ == "__main__":
    fx = HERE / "fixtures"
    fx.mkdir(exist_ok=True)
    (fx / "analysis_js.json").write_text(json.dumps(analysis_fixture(), indent=1) + "\n")
    (fx / "reco_model_js.json").write_text(json.dumps(model_fixture(sys.argv[1:]), separators=(",", ":")) + "\n")
    print("wrote", fx)
