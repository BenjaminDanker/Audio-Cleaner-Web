"""The automatic settings model, and what the desktop app does with it: "Automatic best settings" for a file, decided the
same way as on the website (web/src/reco-model.js and web/src/recommend.js).

Gradient-boosted trees (trained in research/auto-settings) over the 14 measurements of analysis.py: the gentlest noise
strength whose result scores as well as the strongest, whether the voice needs rebuilding, whether the missing high sounds
need restoring. The trees are in reco_model.json next to this file (the same trees as the website's reco-model-data.js;
research/auto-settings/export_web.py writes both), and a decision takes well under a millisecond.

    stats = analysis.measure(x, libdf_path, model_path)
    rec = reco_model.recommend(stats)
    rec.strength        # "gentle" | "balanced" | "strong" | "max"
    rec.rebuild         # rebuild the voice (and rec.rebuild_reason: "echoey", "muffled" or "unclear")
    rec.restore_highs   # restore the missing highs
    rec.heavy_remover   # the stronger noise remover (the rule: noise figure below pipeline.HEAVY_BELOW_DB)

Like the website: the strength, restoring the highs and part of the voice rebuild come from the trees. The voice is also
rebuilt where it measurably sounds echoey or muffled (more often than the trees alone would: rebuilding makes such a voice
fuller and clearer nearly every time). The heavy noise remover stays the rule. A recording nobody speaks in (music, noise)
gets the middle strength and nothing else.
"""
from __future__ import annotations

import functools
import json
import math
from dataclasses import dataclass
from pathlib import Path

from .analysis import FEATURES
from .pipeline import HEAVY_BELOW_DB

MODEL_FILE = Path(__file__).with_name("reco_model.json")
VERSION = 2

# The numbers of web/src/recommend-room.js (T) that the file decision uses.
ECHO_DROP_DB = 10.5  # the level can't fall faster than this within 60 ms (dB): the voice rings on, as in a large or bare room
DULL_TILT_DB = -19.0  # the top of the speech band sits this far below its middle (dB) or more: a muffled voice
PHONE_HZ = 6600.0  # the speech stops at this frequency or lower: a phone call or a very small file
MIN_SECONDS = 2.0  # the website recommends nothing for less sound than this

STRENGTH_CLASSES = ("gentle", "balanced", "strong", "max")
NO_SPEECH_STRENGTH = "balanced"


@functools.lru_cache(maxsize=2)
def load(path: str | Path | None = None) -> dict:
    """The trees (reco_model.json), read once."""
    model = json.loads(Path(path or MODEL_FILE).read_text(encoding="utf-8"))
    if model.get("version") != VERSION or tuple(model.get("features", ())) != FEATURES:
        raise ValueError("reco_model.json does not match analysis.FEATURES")
    return model


def feature_vector(stats: dict) -> list[float]:
    """The measurements in the trees' order (hasSpeech as 0/1; a missing one is NaN)."""
    out = []
    for f in FEATURES:
        v = (1 if stats.get("hasSpeech") else 0) if f == "hasSpeech" else stats.get(f)
        out.append(math.nan if v is None else float(v))
    return out


def _walk(node: list, x: list[float]) -> float:
    """One tree: [feature, threshold, missingGoesLeft, left, right]; a leaf is [value]."""
    while len(node) > 1:
        v = x[node[0]]
        node = node[3] if (node[2] == 1 if v != v else v <= node[1]) else node[4]
    return node[0]


def _scores(ensemble: dict, x: list[float]) -> list[float]:
    """The summed scores of an ensemble, one per class (the trees take turns by class)."""
    k = ensemble["num_class"]
    out = [0.0] * k
    for i, tree in enumerate(ensemble["trees"]):
        out[i % k] += _walk(tree, x)
    return out


def _sigmoid(v: float) -> float:
    return 1 / (1 + math.exp(-v)) if v >= 0 else math.exp(v) / (1 + math.exp(v))


@dataclass(frozen=True)
class ModelDecision:
    strength: str
    rebuild: bool
    restore_highs: bool
    p_strength: tuple[float, ...]
    p_rebuild: float
    p_restore_highs: float


def model_decision(stats: dict, model: dict | None = None) -> ModelDecision:
    """What the trees say for these measurements (web/src/reco-model.js modelDecision)."""
    model = model or load()
    x = feature_vector(stats)
    s = _scores(model["noise"], x)
    top = max(s)
    e = [math.exp(v - top) for v in s]
    total = sum(e)
    p = tuple(v / total for v in e)
    best = p.index(max(p))
    rebuild = _sigmoid(_scores(model["rebuild"], x)[0])
    restore = _sigmoid(_scores(model["restore"], x)[0])
    return ModelDecision(model["strength_classes"][best], rebuild > 0.5, restore > 0.5, p, rebuild, restore)


@dataclass(frozen=True)
class Recommendation:
    strength: str  # "gentle" | "balanced" | "strong" | "max"
    rebuild: bool
    rebuild_reason: str | None  # why: "echoey", "muffled", or "unclear" (the trees only); None when not rebuilt
    restore_highs: bool
    heavy_remover: bool  # the stronger noise remover (noise figure below HEAVY_BELOW_DB)
    no_speech: bool  # nobody speaks: middle strength, nothing else
    noise_db: float | None


def recommend(stats: dict, min_seconds: float = MIN_SECONDS, model: dict | None = None) -> Recommendation | None:
    """The file recommendation of web/src/recommend.js for the measurements `stats` (analysis.measure). None where the
    website recommends nothing: less than `min_seconds` of sound, or no noise figure."""
    if not stats or not (stats.get("seconds", 0) >= min_seconds) or stats.get("noiseDb") is None:
        return None
    noise_db = float(stats["noiseDb"])
    heavy = noise_db < HEAVY_BELOW_DB
    if not stats.get("hasSpeech"):
        return Recommendation(NO_SPEECH_STRENGTH, False, None, False, heavy, True, noise_db)
    m = model_decision(stats, model)
    drop, tilt = stats.get("dropDb"), stats.get("tiltDb")
    phone = stats["bandwidthHz"] <= PHONE_HZ
    echoey = drop is not None and drop < ECHO_DROP_DB
    dull = not phone and tilt is not None and tilt <= DULL_TILT_DB
    rebuild = echoey or dull or m.rebuild
    reason = "echoey" if echoey else "muffled" if dull else "unclear" if rebuild else None
    return Recommendation(m.strength, rebuild, reason, m.restore_highs, heavy, False, noise_db)
