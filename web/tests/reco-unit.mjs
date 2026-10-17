// Offline checks of "Recommended for this file / for your room": the measuring (src/media/analysis.js) and the advice
// (src/recommend.js) on generated audio, the wording rules of tests/copy.mjs, and that the thresholds the OBS plugin uses
// (obs-plugin/core/src/input_advice.cpp) are the website's. Needs no browser and no backend:  node tests/reco-unit.mjs
// (The agreement with the server's own functions: server/tests/test_reco_agreement.py.)
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { dirname, join } from 'node:path';
import { fileURLToPath } from 'node:url';
import { HEAVY_BELOW_DB, planWindows, RATE } from '../src/media/analysis.js';
import { recommendForFile, roomChanges, strengthFor, T } from '../src/recommend.js';
import { gauss, noisy, rng, voice } from './reco-audio.mjs';
import { analysePlanes, loadDfn } from './reco-cli.mjs';

await loadDfn(); // the site's own noise remover (public/dfn), which measures the noise

const ROOT = join(dirname(fileURLToPath(import.meta.url)), '..', '..');
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

// ---- generated material (tests/reco-audio.mjs), measured as the browser does (tests/reco-cli.mjs) ---------------------------
const analyse = (planes, sampled = false) => analysePlanes(planes, { sampled });

// ---- measuring -------------------------------------------------------------------------------------------------------
check('the noise figure falls as the noise rises, and 99 for pure silence', () => {
  const figs = [45, 30, 20, 10, 3].map((snr) => analyse([noisy(20, snr)]).noiseDb);
  for (let i = 1; i < figs.length; i++) assert.ok(figs[i] < figs[i - 1], figs.join(' '));
  assert.equal(analyse([new Float32Array(5 * RATE)]).noiseDb, 99);
});

check('noise that comes and goes counts: clicks between the words lower the figure as much as steady noise does', () => {
  const v = noisy(20, 50, 5);
  const r = rng(6);
  const clicks = Float32Array.from(v);
  for (let k = 0; k < 120; k++) { // keyboard-like: 120 sharp clicks of 15 ms, quiet in between
    const at = Math.floor(r() * (v.length - 720));
    for (let i = 0; i < 720; i++) clicks[at + i] += 0.25 * gauss(r) * Math.exp(-i / 150);
  }
  const quiet = analyse([v]).noiseDb;
  const clicky = analyse([clicks]).noiseDb;
  assert.ok(clicky < quiet - 8, `${quiet} -> ${clicky}`);
  assert.ok(clicky < T.CLEAN_AT_LEAST, String(clicky));
});

check('speech is told from steady sounds: a hum, a fan, and a tone have none', () => {
  assert.equal(analyse([voice(15)]).hasSpeech, true);
  assert.equal(analyse([noisy(15, 5)]).hasSpeech, true);
  const hum = Float32Array.from({ length: 12 * RATE }, (_, i) => 0.02 * Math.sin((2 * Math.PI * 50 * i) / RATE));
  assert.equal(analyse([hum]).hasSpeech, false);
  const r = rng(4);
  assert.equal(analyse([Float32Array.from({ length: 12 * RATE }, () => 0.02 * gauss(r))]).hasSpeech, false);
  assert.equal(analyse([new Float32Array(5 * RATE)]).hasSpeech, false);
});

check('clipping is counted, a loud but clean recording is not', () => {
  const v = noisy(12, 40);
  assert.equal(analyse([v]).clipFraction, 0);
  const hot = v.map((a) => Math.max(-0.9995, Math.min(0.9995, a * 8)));
  assert.ok(analyse([hot]).clipFraction > 0.02);
});

check('channel layout: stereo, the same on both sides, one side silent', () => {
  const a = noisy(10, 30, 1);
  const b = noisy(10, 20, 2);
  assert.equal(analyse([a, b]).layout, 'stereo');
  assert.equal(analyse([a, a]).layout, 'dual-mono');
  assert.equal(analyse([a, new Float32Array(a.length)]).layout, 'one-side');
  assert.equal(analyse([a]).layout, 'mono');
});

check('loudness follows the level: +6 dB of gain is +6 LU', () => {
  const v = noisy(15, 40);
  const l0 = analyse([v]).loudnessLufs;
  const l1 = analyse([v.map((a) => a * 2)]).loudnessLufs;
  assert.ok(Math.abs(l1 - l0 - 6.02) < 0.05, `${l0} ${l1}`);
});

check('a long file is judged from about a minute spread over it (six stretches), the same figure within 3 dB', () => {
  const v = noisy(100, 12, 3);
  const whole = analyse([v]);
  const part = analyse([v], true);
  assert.equal(part.segments, 6);
  assert.ok(part.seconds > 55 && part.seconds < 61, String(part.seconds));
  assert.ok(Math.abs(part.noiseDb - whole.noiseDb) < 3, `${part.noiseDb} ${whole.noiseDb}`);
  const w = planWindows(100);
  assert.ok(w[0].start >= 0 && w[5].start + w[5].length <= 100);
  assert.deepEqual(planWindows(40), [{ start: 0, length: 40 }]);
});

// ---- the advice ----------------------------------------------------------------------------------------------------------
const REASONS = [];
const rec = (x, ctx) => { const r = recommendForFile(analyse([x]), ctx); if (r) REASONS.push(...r.reasons, r.line); return r; };

check('a clean recording needs nothing; steady noise gets the free cleanup; a lot of noise gets the paid one with a stronger remover', () => {
  const clean = rec(noisy(20, 55));
  assert.equal(clean.noise, 'off');
  const mild = rec(noisy(20, 22));
  assert.equal(mild.noise, 'browser');
  assert.equal(mild.heavy, false);
  const heavy = rec(noisy(20, 3));
  assert.equal(heavy.noise, 'denoise');
  assert.equal(heavy.heavy, true);
  assert.equal(heavy.strength, 'max');
  assert.equal(strengthFor(T.CLEAN_AT_LEAST - 0.1), 'gentle');
});

check('"heavy" only where the server would pick the stronger remover (and not within the margin of its cut)', () => {
  for (let snr = -2; snr <= 40; snr += 2) {
    const s = analyse([noisy(20, snr, 7)]);
    const r = recommendForFile(s, {});
    if (r.heavy) assert.ok(s.noiseDb < HEAVY_BELOW_DB - T.HEAVY_MARGIN_DB + 1e-9, `${snr} dB: ${s.noiseDb}`);
    if (s.noiseDb >= HEAVY_BELOW_DB) assert.equal(r.heavy, false);
  }
});

check('the strength steps get gentler as the noise gets lighter', () => {
  const order = ['max', 'strong', 'balanced', 'gentle'];
  let last = 0;
  for (const nd of [5, T.MAX_BELOW - 0.1, T.MAX_BELOW + 0.1, T.STRONG_BELOW + 0.1, T.BALANCED_BELOW + 0.1, T.CLEAN_AT_LEAST - 0.1]) {
    const i = order.indexOf(strengthFor(nd));
    assert.ok(i >= last, `${nd}`);
    last = i;
  }
});

check('no speech: nothing to clean or write down', () => {
  const hum = Float32Array.from({ length: 12 * RATE }, (_, i) => 0.02 * Math.sin((2 * Math.PI * 50 * i) / RATE));
  const r = rec(hum);
  assert.equal(r.noSpeech, true);
  assert.equal(r.nothing, true);
  assert.equal(r.subs, null);
});

check('subtitles: Standard for a clear voice, Extra accurate for noise or a language Standard is weaker in', () => {
  assert.equal(rec(noisy(20, 55), { language: '' }).subs, 'standard');
  assert.equal(rec(noisy(20, 55), { language: 'es' }).subs, 'standard');
  assert.equal(rec(noisy(20, 55), { language: 'ja' }).subs, 'accurate');
  assert.equal(rec(noisy(20, 2)).subs, 'accurate');
});

check('very short or unreadable audio gives no advice', () => {
  assert.equal(recommendForFile(null), null);
  assert.equal(recommendForFile(analyse([voice(1)])), null);
});

check('live room test: noise removal stays on (a quiet room gets Gentle), the strength follows the noise, captions go up for noise', () => {
  const room = (x, now) => { const r = roomChanges(analyse([x]), now); REASONS.push(...r.changes.map((c) => c.text), ...r.notes); return r; };
  const quiet = room(noisy(20, 55), { denoise: true, strength: 'max', captions: '' });
  assert.equal(quiet.settings.denoise, true);
  assert.deepEqual(quiet.changes.map((c) => c.text), ['Noise removal strength: Maximum → Gentle']);
  const loud = room(noisy(20, 2), { denoise: true, strength: 'gentle', captions: 'local' });
  assert.equal(loud.settings.strength, 'max');
  assert.equal(loud.settings.captions, 'premium');
  assert.deepEqual(loud.changes.map((c) => c.key), ['strength', 'captions']);
  const off = room(noisy(20, 6), { denoise: false, strength: 'max', captions: '' });
  assert.equal(off.settings.denoise, true);
  assert.match(off.changes[0].text, /off → on, Maximum/);
  assert.deepEqual(room(noisy(20, 2), { denoise: true, strength: 'max', captions: 'premium' }).changes, []); // already right
  const silent = roomChanges(analyse([new Float32Array(20 * RATE)]), { denoise: false, strength: 'max', captions: '' });
  assert.equal(silent.heardSpeech, false);
  assert.equal(silent.settings.denoise, false); // nothing heard: nothing changes
});

// ---- wording and the plugin's copy of the numbers -----------------------------------------------------------------------
check('every sentence follows the copy rules (tests/copy.mjs): no model names, no corporate words', () => {
  const copy = readFileSync(join(ROOT, 'web', 'tests', 'copy.mjs'), 'utf8');
  const list = copy.slice(copy.indexOf('const BANNED = ['), copy.indexOf('];', copy.indexOf('const BANNED = [')) + 1);
  const banned = [...list.matchAll(/\/((?:\\\/|[^/])+)\/(i?)/g)].map((m) => new RegExp(m[1], m[2]));
  assert.ok(banned.length > 10);
  assert.ok(REASONS.length > 15, `only ${REASONS.length} sentences were produced`);
  const texts = [...new Set(REASONS)];
  for (const t of texts) for (const re of banned) assert.ok(!re.test(t), `"${t}" matches ${re}`);
  for (const t of texts) assert.ok(!/\bheavy\b|\blight\b|DFN|model|AI\b/i.test(t), `"${t}" talks about the machinery`);
});

check('the plugin uses the same thresholds as the website', () => {
  const cpp = readFileSync(join(ROOT, 'obs-plugin', 'core', 'src', 'input_advice.cpp'), 'utf8');
  const num = (name) => Number(new RegExp(`${name}\\s*=\\s*(-?[\\d.]+)`).exec(cpp)?.[1]);
  const web = readFileSync(join(ROOT, 'web', 'src', 'media', 'analysis.js'), 'utf8');
  assert.equal(num('kCleanAtLeast'), T.CLEAN_AT_LEAST);
  assert.equal(num('kMaxBelow'), T.MAX_BELOW);
  assert.equal(num('kStrongBelow'), T.STRONG_BELOW);
  assert.equal(num('kBalancedBelow'), T.BALANCED_BELOW);
  assert.equal(num('kCaptionsNoisyBelow'), T.SUBS_NOISY_BELOW);
  assert.equal(num('kClipFraction'), T.CLIP_FRACTION);
  assert.equal(num('kSpeechBandDyn'), Number(/SPEECH_BAND_DYN_DB\s*=\s*([\d.]+)/.exec(web)[1]));
});

console.log(failed ? `\n${failed} failed` : '\nall passed');
process.exit(failed ? 1 : 0);
