"""Label scheme of the automatic-settings model, shared by label.py (train split) and relabel.py (val/test).

noise strength (4 classes, the product's levels; the product never turns noise removal off any more):
  0 gentle, 1 balanced, 2 strong, 3 max = the gentlest level whose PESQ-WB (against the clean reference) is within MARGIN of
  the best of the four. Gentle when there is no noise or the input already scores as well as the best output.
heavy (1 = the heavy noise remover is needed): the server's own rule (pipeline.HEAVY_BELOW_DB = 19 dB on the noise figure,
  the same number as web/src/media/analysis.js HEAVY_BELOW_DB), applied to the browser measurement (stats.noiseDb). This is
  the rule's ground truth, not an independent listening test: the server's heavy remover is not in this project.
"""
import math

ATTEN = [12, 24, 40, 100]          # gentle, balanced, strong, max (web/src/strengths.js)
MARGIN = 0.05                      # PESQ-WB points: "as good as the strongest"
HEAVY_BELOW_DB = 19.0              # pipeline.HEAVY_BELOW_DB
STRENGTH_NAMES = ["gentle", "balanced", "strong", "max"]


def strength_label(snr_none, s_in, scores):
    """scores: {atten: pesq}. Returns 0..3 (index into STRENGTH_NAMES)."""
    fin = [scores[a] for a in ATTEN if math.isfinite(scores[a])]
    if snr_none or not fin:
        return 0
    best = max(fin)
    if math.isfinite(s_in) and s_in >= best - MARGIN:
        return 0
    for i, a in enumerate(ATTEN):
        if math.isfinite(scores[a]) and scores[a] >= best - MARGIN:
            return i
    return 0


def heavy_label(noise_db):
    """1 / 0, or -1 when the figure is missing (such rows are not used for the heavy head)."""
    if noise_db is None or not math.isfinite(noise_db):
        return -1
    return int(noise_db < HEAVY_BELOW_DB)
