// Offline checks of "Recommended for this file / for your room": the measuring (src/media/analysis.js) and the advice
// (src/recommend.js) on generated audio, the wording rules of tests/copy.mjs, and that the thresholds the OBS plugin uses
// (obs-plugin/core/src/input_advice.cpp) are the website's. Needs no browser and no backend:  node tests/reco-unit.mjs
// (The agreement with the server's own functions: server/tests/test_reco_agreement.py.)
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { dirname, join } from 'node:path';
import { fileURLToPath } from 'node:url';
import { Analyser, EchoProbe, HEAVY_BELOW_DB, planWindows, RATE } from '../src/media/analysis.js';
import { recommendForFile, recommendForLive, strengthFor, T } from '../src/recommend.js';
import { gauss, noisy, rng, voice } from './reco-audio.mjs';

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

// ---- generated material (tests/reco-audio.mjs) -----------------------------------------------------------------------------
function analyse(planes, sampled = false) {
  const ch = planes.length;
  const a = new Analyser({ channels: ch });
  const frames = planes[0].length;
  const wins = sampled ? planWindows(frames / RATE) : [{ start: 0, length: frames / RATE }];
  wins.forEach((w, k) => {
    if (k) a.segment();
    const from = Math.round(w.start * RATE);
    const to = Math.min(frames, from + Math.round(w.length * RATE));
    for (let p = from; p < to; p += 24000) a.push(planes.map((x) => x.subarray(p, Math.min(to, p + 24000))));
  });
  return a.result();
}

// ---- measuring -------------------------------------------------------------------------------------------------------
check('the noise figure falls as the noise rises, and 99 for pure silence', () => {
  const figs = [45, 30, 20, 10, 3].map((snr) => analyse([noisy(20, snr)]).noiseDb);
  for (let i = 1; i < figs.length; i++) assert.ok(figs[i] < figs[i - 1], figs.join(' '));
  assert.equal(analyse([new Float32Array(5 * RATE)]).noiseDb, 99);
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

check('the speakers are found in the microphone when they are there, not when they are not', () => {
  const played = voice(12, 5);
  const mic = noisy(12, 40, 9);
  const echoed = mic.map((a, i) => a + (i > 7200 ? 0.5 * played[i - 7200] : 0));
  const probe = (m, ref) => {
    const p = new EchoProbe({ minLagMs: 80, seconds: 12 });
    for (let i = 0; i + 960 <= m.length; i += 960) { p.pushMic(m.subarray(i, i + 960)); p.pushRef(ref ? ref.subarray(i, i + 960) : null, 960); }
    return p.result();
  };
  assert.ok(probe(echoed, played).explained > T.ECHO_SHARE);
  assert.ok(Math.abs(probe(echoed, played).lagMs - 150) <= 10);
  assert.ok(probe(mic, played).explained < T.ECHO_SHARE);
  assert.ok(probe(mic, null).explained < T.ECHO_SHARE);
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
  assert.equal(heavy.noise, 'polish');
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
  assert.equal(rec(noisy(20, 5)).subs, 'accurate');
});

check('very short or unreadable audio gives no advice', () => {
  assert.equal(recommendForFile(null), null);
  assert.equal(recommendForFile(analyse([voice(1)])), null);
});

check('live: a quiet room can leave noise removal off; a noisy one is told how strong; echo and clipping are mentioned', () => {
  const live = (x, echo, ctx) => { const r = recommendForLive(analyse([x]), echo, ctx); if (r) REASONS.push(...r.notes); return r; };
  const quiet = live(noisy(12, 55), null, { mode: 'free', denoise: true, strength: 'max' });
  assert.equal(quiet.denoise, false);
  assert.match(quiet.line, /quiet/);
  const loud = live(noisy(12, 6), null, { mode: 'free', denoise: true, strength: 'gentle' });
  assert.equal(loud.denoise, true);
  assert.equal(loud.strength, 'max');
  assert.match(loud.line, /Maximum/);
  const off = live(noisy(12, 6), null, { mode: 'free', denoise: false });
  assert.match(off.line, /turn on noise removal/);
  const echo = live(noisy(12, 55), { explained: 0.2, known: true }, { mode: 'paid', denoise: true, strength: 'max', captions: 'local' });
  assert.equal(echo.echo, true);
  assert.match(echo.line, /speakers/);
  assert.equal(live(noisy(12, 55), { explained: 0, known: true }, { mode: 'free' }).echo, false);
  const captions = live(noisy(12, 5), null, { mode: 'paid', denoise: true, strength: 'max', captions: 'local' });
  assert.match(captions.line, /Extra accurate/);
  const fine = live(noisy(12, 55), null, { mode: 'paid', denoise: false, captions: 'premium' });
  assert.match(fine.line, /Standard captions are enough/);
  const silent = recommendForLive(analyse([new Float32Array(12 * RATE)]), null, {});
  assert.equal(silent.pending, true);
  assert.equal(recommendForLive(analyse([new Float32Array(12 * RATE)]), null, { final: true }), null); // nothing more will be heard: stop asking
  assert.equal(recommendForLive(analyse([voice(3)]), null, {}), null);
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
  assert.equal(num('kHeavyBelow'), HEAVY_BELOW_DB);
  assert.equal(num('kStrongBelow'), T.STRONG_BELOW);
  assert.equal(num('kBalancedBelow'), T.BALANCED_BELOW);
  assert.equal(num('kCaptionsNoisyBelow'), T.SUBS_NOISY_BELOW);
  assert.equal(num('kClipFraction'), T.CLIP_FRACTION);
  assert.equal(num('kEchoShare'), T.ECHO_SHARE);
  assert.equal(num('kSpeechBandDyn'), Number(/SPEECH_BAND_DYN_DB\s*=\s*([\d.]+)/.exec(web)[1]));
});

console.log(failed ? `\n${failed} failed` : '\nall passed');
process.exit(failed ? 1 : 0);
