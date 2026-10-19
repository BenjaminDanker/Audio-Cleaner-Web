// The automatic settings model: gradient-boosted trees (research/auto-settings) evaluated on the measurements that
// src/media/analysis.js (Analyser.result()) already takes, so nothing is downloaded and it takes well under a millisecond.
// It was trained on what actually came out: the noise strength is the gentlest one whose result scores as well as the
// strongest, and voice rebuild is what the damaged recordings needed. src/recommend.js decides with it.
import MODEL from './reco-model-data.js';

/** The measurements, in the order the trees use them (hasSpeech as 0/1; a missing one is NaN). */
export const FEATURES = ['floorDb', 'levelDb', 'peakDb', 'silentFraction', 'speechFraction', 'bandDynDb', 'hasSpeech',
  'bandwidthHz', 'tiltDb', 'dropDb', 'clipFraction', 'peak', 'loudnessLufs', 'noiseDb'];

if (MODEL.version !== 3 || MODEL.features.join() !== FEATURES.join()) throw new Error('reco-model-data.js does not match reco-model.js');

function featureVector(stats) {
  return FEATURES.map((f) => {
    const v = f === 'hasSpeech' ? (stats.hasSpeech ? 1 : 0) : stats[f];
    return v == null ? NaN : Number(v);
  });
}

/** One tree: [feature, threshold, missingGoesLeft, left, right]; a leaf is [value]. */
function walk(node, x) {
  let n = node;
  while (n.length > 1) {
    const v = x[n[0]];
    n = (Number.isNaN(v) ? n[2] === 1 : v <= n[1]) ? n[3] : n[4];
  }
  return n[0];
}

/** The summed scores of an ensemble, one per class (the trees take turns by class). */
function scores(ens, x) {
  const out = new Array(ens.num_class).fill(0);
  ens.trees.forEach((t, i) => { out[i % ens.num_class] += walk(t, x); });
  return out;
}

const sigmoid = (v) => 1 / (1 + Math.exp(-v));

/**
 * @param {object} stats Analyser.result()
 * @returns {{strength: 'gentle'|'balanced'|'strong'|'max', rebuild: boolean, p: {strength: number[], rebuild: number}}}
 */
export function modelDecision(stats) {
  const x = featureVector(stats);
  const s = scores(MODEL.noise, x);
  const top = Math.max(...s);
  const e = s.map((v) => Math.exp(v - top));
  const sum = e.reduce((a, b) => a + b, 0);
  const strength = e.map((v) => v / sum);
  const best = strength.indexOf(Math.max(...strength));
  const rebuild = sigmoid(scores(MODEL.rebuild, x)[0]);
  return {
    strength: MODEL.strength_classes[best],
    rebuild: rebuild > 0.5,
    p: { strength, rebuild },
  };
}
