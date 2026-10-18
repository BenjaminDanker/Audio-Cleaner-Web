// Checks of the automatic settings model (src/reco-model.js, generated data src/reco-model-data.js). No browser, no backend:
//   node tests/reco-model.mjs
// Always: modelDecision gives valid ids on synthetic stats, missing measurements do not throw, FEATURES are the measurements that
// Analyser.result() returns (src/media/analysis.js). When the training output exists (default ~/audio-cleaner-train, or
// RECO_TRAIN_DIR), also: the rounded numbers of the web copy decide the same as the original reco-model.js + reco-model.json on
// every row of work/test/feat/stats.jsonl (skipped with a message otherwise, e.g. in CI).
import assert from 'node:assert/strict';
import { existsSync, readFileSync } from 'node:fs';
import { homedir } from 'node:os';
import { dirname, join } from 'node:path';
import { fileURLToPath, pathToFileURL } from 'node:url';
import { FEATURES, modelDecision } from '../src/reco-model.js';

let failed = 0;
function check(name, fn) {
  try {
    fn();
    console.log(`PASS ${name}`);
  } catch (e) {
    failed++;
    console.log(`FAIL ${name} -- ${String(e.message).split('\n')[0]}`);
  }
}

const HERE = dirname(fileURLToPath(import.meta.url));
const STRENGTHS = ['gentle', 'balanced', 'strong', 'max'];
const stats = (o = {}) => ({
  noiseDb: 12, floorDb: -50, levelDb: -20, peakDb: -15, silentFraction: 0.05, speechFraction: 0.5, bandDynDb: 15, hasSpeech: true,
  bandwidthHz: 12000, tiltDb: -30, dropDb: 5, clipFraction: 0, peak: 0.8, loudnessLufs: -22, ...o,
});

function valid(d) {
  assert.ok(STRENGTHS.includes(d.strength), `strength ${d.strength}`);
  assert.equal(typeof d.rebuild, 'boolean');
  assert.equal(typeof d.restoreHighs, 'boolean');
  assert.equal(d.p.strength.length, 4);
  for (const v of [...d.p.strength, d.p.rebuild, d.p.restoreHighs]) assert.ok(Number.isFinite(v) && v >= 0 && v <= 1, `probability ${v}`);
  assert.ok(Math.abs(d.p.strength.reduce((a, b) => a + b, 0) - 1) < 1e-9, 'strength probabilities sum to 1');
  assert.equal(d.strength, STRENGTHS[d.p.strength.indexOf(Math.max(...d.p.strength))]);
  assert.equal(d.rebuild, d.p.rebuild > 0.5);
  assert.equal(d.restoreHighs, d.p.restoreHighs > 0.5);
}

check('modelDecision: valid ids on synthetic stats', () => {
  const cases = [
    stats(),
    stats({ noiseDb: -5, floorDb: -25, levelDb: -12, peakDb: -9, bandwidthHz: 3800, tiltDb: -45, clipFraction: 0.01, peak: 1 }),
    stats({ noiseDb: 40, floorDb: -80, levelDb: -30, peakDb: -26, bandwidthHz: 20000, tiltDb: -20 }),
    stats({ hasSpeech: false, speechFraction: 0, bandDynDb: 0 }),
    stats({ noiseDb: 0, floorDb: -60, levelDb: -45, peakDb: -40, silentFraction: 0.9, loudnessLufs: -50 }),
  ];
  for (const s of cases) valid(modelDecision(s));
});

check('modelDecision: deterministic and does not change its input', () => {
  const s = stats();
  const copy = JSON.stringify(s);
  assert.deepEqual(modelDecision(s), modelDecision(s));
  assert.equal(JSON.stringify(s), copy);
});

check('modelDecision: missing measurements (null/undefined) do not throw', () => {
  for (const f of FEATURES) {
    valid(modelDecision(stats({ [f]: null })));
    const s = stats();
    delete s[f];
    valid(modelDecision(s));
  }
  valid(modelDecision(stats({ noiseDb: null, tiltDb: null, dropDb: null })));
  valid(modelDecision({}));
  valid(modelDecision(Object.fromEntries(FEATURES.map((f) => [f, null]))));
});

check('FEATURES are the measurements Analyser.result() returns', () => {
  const src = readFileSync(join(HERE, '..', 'src', 'media', 'analysis.js'), 'utf8');
  const body = src.slice(src.indexOf('  result() {'));
  const ret = body.slice(body.indexOf('return {'));
  const keys = [...ret.slice(0, ret.indexOf('\n    };')).matchAll(/^ {6}([A-Za-z]+)[:,]/gm)].map((m) => m[1]);
  assert.ok(keys.length > 10, `parsed ${keys.length} keys`);
  const missing = FEATURES.filter((f) => !keys.includes(f));
  assert.deepEqual(missing, [], `features Analyser.result() does not return: ${missing}`);
  assert.equal(new Set(FEATURES).size, FEATURES.length, 'FEATURES has duplicates');
});

// ---- the rounding check against the unrounded original -------------------------------------------------------------------
const TRAIN = process.env.RECO_TRAIN_DIR || join(homedir(), 'audio-cleaner-train');
const ORIG_JS = join(TRAIN, 'out', 'reco-model.js');
const ORIG_JSON = join(TRAIN, 'out', 'reco-model.json');
const ROWS = join(TRAIN, 'work', 'test', 'feat', 'stats.jsonl');
if (![ORIG_JS, ORIG_JSON, ROWS].every(existsSync)) {
  console.log(`SKIP rounding check: the training output (${TRAIN}: out/reco-model.js, out/reco-model.json, work/test/feat/stats.jsonl) is not here`);
} else {
  const orig = await import(pathToFileURL(ORIG_JS).href);
  const model = orig.loadModel(readFileSync(ORIG_JSON, 'utf8'));
  const rows = readFileSync(ROWS, 'utf8').split('\n').filter(Boolean).map((l) => JSON.parse(l));
  let n = 0;
  const bad = { strength: 0, rebuild: 0, restoreHighs: 0 };
  let maxDp = 0;
  let shifted = 0; // rows where a split flipped (a value within 6 significant digits of a threshold) and moved a probability
  const examples = [];
  for (const r of rows) {
    const a = orig.decideFromModel(r, model, null);
    const b = modelDecision(r);
    n++;
    for (const k of Object.keys(bad)) if (a[k] !== b[k]) { bad[k]++; if (examples.length < 5) examples.push(`${r.id}: ${k} ${a[k]} vs ${b[k]}`); }
    const p = orig.predict(model, r);
    const dp = Math.max(Math.abs(p.rebuild - b.p.rebuild), Math.abs(p.restore - b.p.restoreHighs), ...p.strength.map((v, i) => Math.abs(v - b.p.strength[i])));
    maxDp = Math.max(maxDp, dp);
    if (dp > 1e-3) shifted++;
  }
  const agree = (k) => `${k} ${(100 * (1 - bad[k] / n)).toFixed(3)}% (${n - bad[k]}/${n})`;
  console.log(`rounding check on ${n} rows: ${agree('strength')}, ${agree('rebuild')}, ${agree('restoreHighs')}; largest probability difference ${maxDp.toFixed(3)}, ${shifted} rows differ by more than 0.001`);
  if (examples.length) console.log('  disagreements: ' + examples.join('; '));
  check('rounded web model agrees with the original on the test rows (>= 99.9% per decision, probabilities within 1e-3 on all but 1% of the rows)', () => {
    assert.ok(n > 0);
    for (const k of Object.keys(bad)) assert.ok(bad[k] / n <= 0.001, `${k}: ${bad[k]} of ${n} differ (${examples.join('; ')})`);
    assert.ok(shifted / n <= 0.01, `${shifted} of ${n} rows have a probability that differs by more than 0.001 (up to ${maxDp})`);
    assert.ok(maxDp < 0.15, `probabilities differ by up to ${maxDp}`);
  });
}

if (failed) { console.log(`${failed} FAILED`); process.exit(1); }
console.log('reco-model: all passed');
