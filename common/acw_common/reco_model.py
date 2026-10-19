"""The automatic settings model, and what the desktop app does with it: "Automatic best settings" for a file, decided the
same way as on the website (web/src/reco-model.js and web/src/recommend.js).

Gradient-boosted trees (trained in research/auto-settings) over the 14 measurements of analysis.py: the gentlest noise
strength whose result scores as well as the strongest, and whether the voice needs rebuilding. The trees are in reco_model.json next to this file (the same trees as the website's reco-model-data.js;
research/auto-settings/export_web.py writes both), and a decision takes well under a millisecond.

    stats = analysis.measure(x, libdf_path, model_path)
    rec = reco_model.recommend(stats)
    rec.strength        # "gentle" | "balanced" | "strong" | "max"
    rec.rebuild         # rebuild the voice (and rec.rebuild_reason: "echoey", "muffled", "phone", "distorted" or "unclear")
    rec.heavy_remover   # the stronger noise remover (the rule: noise figure below pipeline.HEAVY_BELOW_DB)

    choices = reco_model.choose(rec, rebuild_available=True)    # the settings that follow from it: the one function the desktop app
    choices.atten_lim_db, choices.noise, choices.rebuild, choices.reasons   # and the server's API jobs share

Like the website: the strength and part of the voice rebuild come from the trees. The voice is also
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
VERSION = 3  # (2 had a third head, which the products no longer use)

# The numbers of web/src/recommend-room.js (T) that the file decision uses.
ECHO_DROP_DB = 10.5  # the level can't fall faster than this within 60 ms (dB): the voice rings on, as in a large or bare room
DULL_TILT_DB = -19.0  # the top of the speech band sits this far below its middle (dB) or more: a muffled voice
PHONE_HZ = 6600.0  # the speech stops at this frequency or lower: a phone call or a very small file
CLIP_FRACTION = 0.002  # more of the samples than this are clipped: the sound distorts from being recorded too loud
# How each rebuild reason is said after "it sounded ...".
REBUILD_WORDS = {"echoey": "echoey", "muffled": "muffled", "phone": "like a phone call", "distorted": "distorted",
                 "unclear": "unclear"}
MIN_SECONDS = 2.0  # the website recommends nothing for less sound than this

STRENGTH_CLASSES = ("gentle", "balanced", "strong", "max")
NO_SPEECH_STRENGTH = "balanced"
# The four strengths and the attenuation limit (dB) of each: the same levels everywhere (the website's strengths.js, the desktop app,
# the API). They are the steps of pipeline.strength_for.
STRENGTH_DB = {"gentle": 12.0, "balanced": 24.0, "strong": 40.0, "max": 100.0}
STRENGTH_NAMES = {"gentle": "Gentle", "balanced": "Balanced", "strong": "Strong", "max": "Maximum"}
CLEAN_AT_LEAST_DB = 24.5  # a noise figure from which the sound counts as clean (web/src/recommend-room.js T.CLEAN_AT_LEAST)


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
    p_strength: tuple[float, ...]
    p_rebuild: float


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
    return ModelDecision(model["strength_classes"][best], rebuild > 0.5, p, rebuild)


@dataclass(frozen=True)
class Recommendation:
    strength: str  # "gentle" | "balanced" | "strong" | "max"
    rebuild: bool
    # why, the first that applies: "echoey", "muffled", "phone" (a phone call or a very small file), "distorted" (recorded too loud),
    # or "unclear" (the trees only); None when not rebuilt
    rebuild_reason: str | None
    heavy_remover: bool  # the stronger noise remover (noise figure below HEAVY_BELOW_DB)
    no_speech: bool  # nobody speaks: middle strength, nothing else
    noise_db: float | None
    rebuild_reasons: tuple[str, ...] = ()  # every cause heard, in the order of REBUILD_WORDS ("unclear" only alone); rebuild_reason is the first


def recommend(stats: dict, min_seconds: float = MIN_SECONDS, model: dict | None = None) -> Recommendation | None:
    """The file recommendation of web/src/recommend.js for the measurements `stats` (analysis.measure). None where the
    website recommends nothing: less than `min_seconds` of sound, or no noise figure."""
    if not stats or not (stats.get("seconds", 0) >= min_seconds) or stats.get("noiseDb") is None:
        return None
    noise_db = float(stats["noiseDb"])
    heavy = noise_db < HEAVY_BELOW_DB
    if not stats.get("hasSpeech"):
        return Recommendation(NO_SPEECH_STRENGTH, False, None, heavy, True, noise_db)
    m = model_decision(stats, model)
    drop, tilt = stats.get("dropDb"), stats.get("tiltDb")
    phone = stats["bandwidthHz"] <= PHONE_HZ
    echoey = drop is not None and drop < ECHO_DROP_DB
    dull = not phone and tilt is not None and tilt <= DULL_TILT_DB
    clipped = (stats.get("clipFraction") or 0.0) > CLIP_FRACTION
    reasons = [r for r, on in (("echoey", echoey), ("muffled", dull), ("phone", phone), ("distorted", clipped), ("unclear", m.rebuild)) if on]
    rebuild = bool(reasons)
    reason = reasons[0] if reasons else None
    heard = tuple(r for r in reasons if r != "unclear" or len(reasons) == 1)  # "unclear" only where nothing else is heard (as the website)
    return Recommendation(m.strength, rebuild, reason, heavy, False, noise_db, heard)


# ---- what the recommendation means for the settings of a job ---------------------------------------------------------
# The desktop app and the server's API jobs both turn a recommendation into settings here, so that they cannot drift apart.
# The caller says whether a voice rebuild is possible for this file (the desktop app: its model is installed and the user left it
# automatic; the server: the card is up and the file is short enough) and applies what it needs of the answer.
_PROBLEMS = {
    "echoey": "The voice sounds echoey, as if it was recorded in a big or bare room.",
    "muffled": "The voice sounds muffled.",
    "phone": "It sounds like a phone call or a very small file: the high sounds are missing.",
    "distorted": "The sound distorts in places because it was recorded too loud.",
    "unclear": "The voice has lost some of its clarity, from echo or from being saved as a very small file.",
}


@dataclass(frozen=True)
class Choices:
    """What "Automatic best settings" settle for one recording."""
    strength: str  # "gentle" | "balanced" | "strong" | "max" (only used when the voice is not rebuilt)
    atten_lim_db: float  # STRENGTH_DB of the strength
    noise: str  # the noise remover: "light" | "heavy"
    heavy: bool  # noise == "heavy"
    rebuild: bool  # the voice is rebuilt (instead of the noise removal)
    rebuild_wanted: bool  # the recording needs it, whether or not it can be done
    rebuild_reason: str | None  # why: the first of REBUILD_WORDS' keys that applies (None when not wanted)
    leveling: bool  # always
    no_speech: bool  # nobody speaks: the middle strength and nothing else
    noise_db: float | None
    reasons: tuple[str, ...]  # plain sentences: what was heard and what is done about it


def choose(rec: Recommendation, *, rebuild_available: bool = True, rebuild_unavailable: str = "voice rebuilding is not available") -> Choices:
    """The settings that follow from a recommendation (recommend). `rebuild_available`: the voice can be rebuilt for this file;
    where it cannot, the recording is cleaned with the noise removal instead, which `reasons`
    says, naming `rebuild_unavailable` (plain words, as in "... because <rebuild_unavailable>")."""
    wanted = rec.rebuild
    rebuild = wanted and rebuild_available
    noise = "heavy" if rec.heavy_remover else "light"
    out: list[str] = []
    name = STRENGTH_NAMES[rec.strength]
    if rec.no_speech:
        out.append("We could hardly hear anyone speaking in this file, so it gets Denoising at Balanced and Leveling, and nothing else.")
    elif rebuild:
        out.extend(_PROBLEMS[r] for r in (rec.rebuild_reasons or (rec.rebuild_reason,)) if r in _PROBLEMS)
        out.append("Rebuilding the voice fixes this: it is rebuilt instead of just taking the noise out.")
    else:
        soft = rec.strength in ("gentle", "balanced")
        if rec.noise_db is not None and rec.noise_db >= CLEAN_AT_LEAST_DB and rec.strength == "gentle":
            out.append("It is clean and quiet: Denoising takes out what little there is, gently.")
        elif rec.heavy_remover:
            out.append("There is a lot of background noise, nearly as loud as the voice: Denoising uses its stronger noise remover for this.")
            out.append(f"With it, {name} is enough{' and keeps the voice natural' if soft else ''}.")
        else:
            out.append("There is some background noise: Denoising takes it out.")
            out.append(f"{name} is enough to take it out{' while keeping the voice natural' if soft else ''}.")
        if wanted:
            out.extend(_PROBLEMS[r] for r in rec.rebuild_reasons if r in _PROBLEMS)
            out.append(f"Rebuilding the voice would fix this, but {rebuild_unavailable}, so Denoising is used instead.")
    out.append("Leveling evens out the voice and sets the loudness.")
    return Choices(rec.strength, STRENGTH_DB[rec.strength], noise, noise == "heavy", rebuild, wanted, rec.rebuild_reason if wanted else None,
                   True, rec.no_speech, rec.noise_db, tuple(out))


def automatic(stats: dict, *, rebuild_available: bool = True, rebuild_unavailable: str = "voice rebuilding is not available",
              min_seconds: float = MIN_SECONDS, model: dict | None = None) -> Choices | None:
    """recommend, then choose: the settings for a recording from its measurements (analysis.measure); None where recommend says
    nothing (too little sound, no noise figure)."""
    rec = recommend(stats, min_seconds, model)
    return None if rec is None else choose(rec, rebuild_available=rebuild_available, rebuild_unavailable=rebuild_unavailable)
