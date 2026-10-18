// Offline checks of "Recommended for this file / for your room": the measuring (src/media/analysis.js) and the advice
// (src/recommend.js) on generated audio, the wording rules of tests/copy.mjs, and that the thresholds the OBS plugin uses
// (obs-plugin/core/src/input_advice.cpp) are the website's. Needs no browser and no backend:  node tests/reco-unit.mjs
// (The agreement with the server's own functions: server/tests/test_reco_agreement.py.)
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { dirname, join } from 'node:path';
import { fileURLToPath } from 'node:url';
import { HEAVY_BELOW_DB, planWindows, RATE } from '../src/media/analysis.js';
import { modelDecision } from '../src/reco-model.js';
import { recommendForFile, roomChanges, strengthFor, T } from '../src/recommend.js';
import { noiseVerdict } from '../src/recommend-room.js';
import { gauss, noisy, rng, voice } from './reco-audio.mjs';
import { analysePlanes, loadDfn } from './reco-cli.mjs';

await loadDfn(); // the noise remover (df.js from public/dfn; the wasm and model from ~/audio-cleaner, as the browser gets them from the model host), which measures the noise

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
  const figs = [45, 30, 20, 10, 3].map((snr) => analyse([noisy(10, snr)]).noiseDb);
  for (let i = 1; i < figs.length; i++) assert.ok(figs[i] < figs[i - 1], figs.join(' '));
  assert.equal(analyse([new Float32Array(5 * RATE)]).noiseDb, 99);
});

check('noise that comes and goes counts: clicks between the words lower the figure as much as steady noise does', () => {
  const v = noisy(10, 50, 5);
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
  assert.equal(analyse([voice(10)]).hasSpeech, true);
  assert.equal(analyse([noisy(10, 5)]).hasSpeech, true);
  const hum = Float32Array.from({ length: 12 * RATE }, (_, i) => 0.02 * Math.sin((2 * Math.PI * 50 * i) / RATE));
  assert.equal(analyse([hum]).hasSpeech, false);
  const r = rng(4);
  assert.equal(analyse([Float32Array.from({ length: 12 * RATE }, () => 0.02 * gauss(r))]).hasSpeech, false);
  assert.equal(analyse([new Float32Array(5 * RATE)]).hasSpeech, false);
});

check('clipping is counted, a loud but clean recording is not', () => {
  const v = noisy(10, 40);
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

check('a long file is judged from about a minute spread over it (six stretches), the same figure within 3 dB', () => {
  const v = noisy(80, 12, 3);
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

check('automatic settings always pick Denoising (never Off or the free cleanup) with Leveling: the strength is the trained model\'s (Gentle for a clean recording, Max for a lot of noise), the stronger remover is still the server\'s rule', () => {
  const clean = rec(noisy(10, 55));
  assert.deepEqual([clean.noise, clean.strength, clean.leveling, clean.nothing], ['denoise', 'gentle', true, false]);
  assert.equal(clean.heavy, false);
  const mild = analyse([noisy(10, 22)]);
  const mildRec = rec(noisy(10, 22));
  assert.equal(mildRec.noise, 'denoise');
  assert.equal(mildRec.heavy, false);
  assert.ok(mildRec.leveling);
  assert.equal(mildRec.strength, modelDecision(mild).strength); // wiring: from the model, not from the noise-figure steps
  const heavy = rec(noisy(10, 3));
  assert.equal(heavy.noise, 'denoise');
  assert.equal(heavy.heavy, true);
  assert.equal(heavy.strength, 'max');
  assert.equal(strengthFor(T.CLEAN_AT_LEAST - 0.1), 'gentle'); // the live room's rule is unchanged
});

// A grid of measurements (not audio): what the file advice makes of each must be the model's decisions, plus the rules it kept.
const gridStats = () => {
  const out = [];
  for (const noiseDb of [-3, 8, 14, 18, 20, 23, 26, 32, 45]) {
    for (const bandwidthHz of [3800, 6500, 8000, 12000, 24000]) {
      for (const tiltDb of [-45, -30, -20, -12]) {
        for (const dropDb of [null, 6, 14, 30]) {
          for (const floorDb of [-35, -60, -80]) {
            out.push({ seconds: 20, segments: 1, noiseDb, floorDb, levelDb: floorDb + 30, peakDb: floorDb + 34, silentFraction: 0.05, speechFraction: 0.5,
              bandDynDb: 20, hasSpeech: true, bandwidthHz, tiltDb, dropDb, clipFraction: 0, peak: 0.7, layout: 'mono', loudnessLufs: -22 });
          }
        }
      }
    }
  }
  return out;
};

check('files: strength, rebuild and restore highs follow the model; rebuild is also chosen for an echo or a muffled voice; heavy stays the rule', () => {
  const seen = { rebuildModel: 0, restore: 0, rebuildRule: 0, strengths: new Set() };
  for (const s of gridStats()) {
    const r = recommendForFile(s, {});
    const m = modelDecision(s);
    const v = noiseVerdict(s);
    const echoey = s.dropDb != null && s.dropDb < T.ECHO_DROP_DB;
    const dull = s.bandwidthHz > T.PHONE_HZ && s.tiltDb <= T.DULL_TILT_DB;
    const where = JSON.stringify(s);
    assert.equal(r.strength, m.strength, where);
    assert.equal(r.restoreHighs, m.restoreHighs, where); // no longer "phone => restore"
    assert.equal(r.rebuild, echoey || dull || m.rebuild, where);
    assert.equal(r.heavy, v.heavy && !v.clean, where);
    assert.deepEqual([r.noise, r.leveling, r.nothing, r.noSpeech], ['denoise', true, false, false]);
    assert.equal(r.line.startsWith(r.rebuild ? 'Voice rebuild' : 'Denoising'), true, r.line);
    if (m.rebuild) seen.rebuildModel++;
    if (m.restoreHighs) seen.restore++;
    if (!m.rebuild && (echoey || dull)) seen.rebuildRule++;
    seen.strengths.add(r.strength);
    REASONS.push(...r.reasons, r.line);
  }
  // the grid reaches every branch, or the checks above prove nothing
  assert.ok(seen.rebuildModel > 0 && seen.restore > 0 && seen.rebuildRule > 0, JSON.stringify({ ...seen, strengths: [...seen.strengths] }));
  assert.ok(seen.strengths.size >= 3, [...seen.strengths].join());
});

check('files: the reasons say what was chosen (echo, muffled, restore highs for a phone call or not, quiet and gentle)', () => {
  const grid = gridStats();
  const find = (f) => grid.find(f);
  const text = (s) => recommendForFile(s, {}).reasons.join(' ');
  const rb = (s) => modelDecision(s).rebuild;
  const rs = (s) => modelDecision(s).restoreHighs;
  const echo = find((s) => s.dropDb === 6 && s.bandwidthHz > T.PHONE_HZ);
  assert.match(text(echo), /echoey/);
  assert.equal(recommendForFile(echo, {}).rebuild, true);
  const dull = find((s) => s.dropDb == null && s.bandwidthHz > T.PHONE_HZ && s.tiltDb <= T.DULL_TILT_DB);
  assert.match(text(dull), /muffled|lost some of its clarity/);
  assert.equal(recommendForFile(dull, {}).rebuild, true);
  const phoneRestore = find((s) => s.bandwidthHz <= T.PHONE_HZ && rs(s));
  const otherRestore = find((s) => s.bandwidthHz > T.PHONE_HZ && rs(s));
  const phoneNo = find((s) => s.bandwidthHz <= T.PHONE_HZ && !rs(s));
  assert.ok(phoneRestore && otherRestore && phoneNo, 'the grid has no phone-bandwidth / wide-band examples for the restore head');
  assert.match(text(phoneRestore), /phone call or a very small file/);
  assert.match(text(otherRestore), /Some of the high sounds are missing/);
  assert.doesNotMatch(text(phoneNo), /high sounds are missing/); // a phone-bandwidth file the model leaves alone gets no restore
  assert.equal(recommendForFile(phoneNo, {}).restoreHighs, false);
  const modelOnly = find((s) => rb(s) && !(s.dropDb != null && s.dropDb < T.ECHO_DROP_DB) && !(s.bandwidthHz > T.PHONE_HZ && s.tiltDb <= T.DULL_TILT_DB));
  assert.ok(modelOnly, 'the grid has no case where only the model asks for the voice rebuild');
  assert.match(text(modelOnly), /lost some of its clarity/);
  const gentleClean = find((s) => s.noiseDb >= T.CLEAN_AT_LEAST && modelDecision(s).strength === 'gentle');
  assert.ok(gentleClean);
  assert.match(text(gentleClean), /clean and quiet/);
  const notGentle = find((s) => modelDecision(s).strength === 'max');
  assert.match(text(notGentle), /Maximum is enough/);
});

check('files: the strength sentence says "With it, ..." where the stronger remover is used, and the usual sentence where it is not', () => {
  const seen = new Set();
  for (const s of gridStats()) {
    const r = recommendForFile(s, {});
    const name = { gentle: 'Gentle', balanced: 'Balanced', strong: 'Strong', max: 'Maximum' }[r.strength];
    const soft = r.strength === 'gentle' || r.strength === 'balanced';
    const sentence = r.reasons.find((t) => / is enough/.test(t));
    if (r.heavy) {
      assert.equal(sentence, `With it, ${name} is enough${soft ? ' and keeps the voice natural' : ''}.`, JSON.stringify(s));
      assert.ok(r.reasons.some((t) => /stronger noise remover/.test(t)));
      seen.add(`heavy/${r.strength}`);
    } else if (sentence) { // (a clean file at Gentle has no such sentence: "clean and quiet" says it)
      assert.equal(sentence, `${name} is enough to take it out${soft ? ' while keeping the voice natural' : ''}.`, JSON.stringify(s));
      assert.ok(!r.reasons.some((t) => /stronger noise remover|With it,/.test(t)));
      seen.add(`plain/${r.strength}`);
    }
  }
  for (const k of ['heavy/gentle', 'heavy/strong', 'plain/gentle', 'plain/balanced', 'plain/strong']) assert.ok(seen.has(k), `the grid never reaches ${k}: ${[...seen]}`);
});

check('files: a file with no speech, or a short one, or without a noise figure, is not sent to the model', () => {
  const s = gridStats()[0];
  const none = recommendForFile({ ...s, hasSpeech: false }, {});
  assert.deepEqual([none.noSpeech, none.strength, none.rebuild, none.restoreHighs, none.heavy, none.subs], [true, 'balanced', false, false, false, null]);
  assert.equal(recommendForFile({ ...s, seconds: 1 }, {}), null);
  assert.equal(recommendForFile({ ...s, noiseDb: null }, {}), null);
  assert.equal(recommendForFile({ ...s, tiltDb: null, dropDb: null }, {}).noise, 'denoise'); // missing measurements still give advice
});

check('"heavy" only where the server would pick the stronger remover (and not within the margin of its cut)', () => {
  for (let snr = -2; snr <= 40; snr += 6) {
    const s = analyse([noisy(10, snr, 7)]);
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

check('no speech: still Denoising (Balanced) and Leveling, but no subtitles', () => {
  const hum = Float32Array.from({ length: 12 * RATE }, (_, i) => 0.02 * Math.sin((2 * Math.PI * 50 * i) / RATE));
  const r = rec(hum);
  assert.equal(r.noSpeech, true);
  assert.deepEqual([r.noise, r.strength, r.leveling, r.nothing, r.subs], ['denoise', 'balanced', true, false, null]);
});

check('subtitles: Standard for a clear voice, Extra accurate for noise or a language Standard is weaker in', () => {
  assert.equal(rec(noisy(10, 55), { language: '' }).subs, 'standard');
  assert.equal(rec(noisy(10, 55), { language: 'es' }).subs, 'standard');
  assert.equal(rec(noisy(10, 55), { language: 'ja' }).subs, 'accurate');
  assert.equal(rec(noisy(10, 2)).subs, 'accurate');
});

check('very short or unreadable audio gives no advice', () => {
  assert.equal(recommendForFile(null), null);
  assert.equal(recommendForFile(analyse([voice(1)])), null);
});

check('live room test: noise removal stays on (a quiet room gets Gentle), the strength follows the noise, captions go up for noise', () => {
  const room = (x, now) => { const r = roomChanges(analyse([x]), now); REASONS.push(...r.changes.map((c) => c.text), ...r.notes); return r; };
  const quiet = room(noisy(10, 55), { denoise: true, strength: 'max', captions: '' });
  assert.equal(quiet.settings.denoise, true);
  assert.deepEqual(quiet.changes.map((c) => c.text), ['Noise removal strength: Maximum → Gentle']);
  const loud = room(noisy(10, 2), { denoise: true, strength: 'gentle', captions: 'local' });
  assert.equal(loud.settings.strength, 'max');
  assert.equal(loud.settings.captions, 'premium');
  assert.deepEqual(loud.changes.map((c) => c.key), ['strength', 'captions']);
  const off = room(noisy(10, 6), { denoise: false, strength: 'max', captions: '' });
  assert.equal(off.settings.denoise, true);
  assert.match(off.changes[0].text, /off → on, Maximum/);
  assert.deepEqual(room(noisy(10, 2), { denoise: true, strength: 'max', captions: 'premium' }).changes, []); // already right
  const silent = roomChanges(analyse([new Float32Array(10 * RATE)]), { denoise: false, strength: 'max', captions: '' });
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
