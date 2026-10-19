// Offline checks of "Recommended for this file / for your room": the measuring (src/media/analysis.js) and the advice
// (src/recommend.js) on generated audio, the wording rules of tests/copy.mjs, and that the thresholds the OBS plugin uses
// (obs-plugin/core/src/input_advice.cpp) are the website's. Needs no browser and no backend:  node tests/reco-unit.mjs
// (The agreement with the server's own functions: server/tests/test_reco_agreement.py.)
import assert from 'node:assert/strict';
import { readdirSync, readFileSync } from 'node:fs';
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
  assert.equal(clean.rebuild, false); // a clean, close, full-band voice is not rebuilt
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
// The values around every threshold of the rules (6600 Hz, 10.5 dB, -19 dB, 0.002) are on it, on both sides.
const gridStats = () => {
  const out = [];
  for (const noiseDb of [-3, 8, 14, 18, 20, 23, 26, 32, 45]) {
    for (const bandwidthHz of [3800, 6500, 6600, 6601, 8000, 12000, 24000]) {
      for (const tiltDb of [-45, -30, -19, -17, -12, null]) { // (null: not measured)
        for (const dropDb of [null, 6, 10.5, 14, 30]) {
          for (const floorDb of [-35, -60, -80]) {
            for (const clipFraction of [0, 0.002, 0.003, 0.05]) {
              out.push({ seconds: 20, segments: 1, noiseDb, floorDb, levelDb: floorDb + 30, peakDb: floorDb + 34, silentFraction: 0.05, speechFraction: 0.5,
                bandDynDb: 20, hasSpeech: true, bandwidthHz, tiltDb, dropDb, clipFraction, peak: 0.7, layout: 'mono', loudnessLufs: -22 });
            }
          }
        }
      }
    }
  }
  return out;
};

// Why the voice is rebuilt, from the measurements alone: the rules written out here with their numbers, not read from T.
const causesOf = (s) => ({
  echo: s.dropDb != null && s.dropDb < 10.5, // the level falls less than 10.5 dB within 60 ms
  dull: s.bandwidthHz > 6600 && s.tiltDb != null && s.tiltDb <= -19, // muffled: a phone call is not "muffled", it is a phone call
  phone: s.bandwidthHz <= 6600, // the speech stops at 6600 Hz or lower
  clipped: s.clipFraction > 0.002, // more than 0.2 % of the samples are clipped
});
const CAUSES = ['echo', 'dull', 'phone', 'clipped'];
const onlyCauses = (...names) => (s) => { const c = causesOf(s); return CAUSES.every((k) => c[k] === names.includes(k)); };
const anyCause = (s) => Object.values(causesOf(s)).some(Boolean);

// Every sentence of the rebuild advice, word for word. (Restoring the highs is not advised any more: that sentence is never said.)
const SAY = {
  echo: 'The voice sounds echoey, as if it was recorded in a big or bare room.',
  dull: 'The voice sounds muffled.',
  phone: 'It sounds like a phone call or a very small file: the high sounds are missing.',
  clipped: 'The sound distorts in places because it was recorded too loud.',
  clarity: 'The voice has lost some of its clarity, from echo or from being saved as a very small file.',
  fix: 'Rebuilding the voice fixes this. Listen to the result before you publish.',
  next: 'Next time, turn the microphone level down.',
};
const RESTORE = 'Some of the high sounds are missing. Restoring them makes the voice clearer.'; // gone: no step restores them any more
const SAY_ALL = new Set(Object.values(SAY));
/** the rebuild/restore sentences among the reasons, in the order given */
const said = (r) => r.reasons.filter((t) => SAY_ALL.has(t));
/** what they must be for these measurements (and what the model says) */
function expectedSaid(s, m) {
  const c = causesOf(s);
  if (c.echo || c.dull || c.phone || c.clipped || m.rebuild) {
    const out = CAUSES.filter((k) => c[k]).map((k) => SAY[k]);
    if (!out.length) out.push(SAY.clarity); // only the model asked
    out.push(SAY.fix);
    if (c.clipped) out.push(SAY.next);
    return out;
  }
  return [];
}
const GONE = /Cleaning can.t fix|microphone level down next time/i; // the old advice: now said as "Next time, turn the microphone level down."

check('the rebuild rules use these numbers: a phone call is 6600 Hz or lower, echo is a fall under 10.5 dB, muffled is -19 dB or lower, distortion is more than 0.2 % clipped', () => {
  assert.deepEqual([T.PHONE_HZ, T.ECHO_DROP_DB, T.DULL_TILT_DB, T.CLIP_FRACTION], [6600, 10.5, -19, 0.002]);
});

check('files: rebuild = echoey || muffled || phone || clipped || the model, on the whole grid; the strength stays the model\'s; heavy stays the rule; every grid row says exactly the sentences its causes call for', () => {
  const seen = { rebuildModel: 0, rebuildRule: 0, strengths: new Set(), rebuilt: 0, kept: 0, only: {}, edge: {}, multi: 0 };
  const bump = (o, k) => { o[k] = (o[k] || 0) + 1; };
  for (const s of gridStats()) {
    const r = recommendForFile(s, {});
    const m = modelDecision(s);
    const v = noiseVerdict(s);
    const c = causesOf(s);
    const where = JSON.stringify(s);
    assert.equal(r.strength, m.strength, where);
    assert.ok(!('restoreHighs' in r) && !('restoreHighs' in m), where); // not a decision any more
    assert.equal(r.rebuild, c.echo || c.dull || c.phone || c.clipped || m.rebuild, where);
    assert.equal(r.heavy, v.heavy && !v.clean, where);
    assert.deepEqual([r.noise, r.leveling, r.nothing, r.noSpeech], ['denoise', true, false, false]);
    // the sentences: each cause once, in the order echoey, muffled, phone, distorts; then the one fix; "Next time" only for distortion
    assert.deepEqual(said(r), expectedSaid(s, m), where);
    assert.equal(r.reasons.filter((t) => t === SAY.fix).length, r.rebuild ? 1 : 0, where);
    assert.equal(r.reasons.includes(SAY.next), c.clipped, where);
    assert.ok(!r.reasons.some((t) => GONE.test(t) || t === RESTORE || /Restoring them|highs/i.test(t)), where);
    // the line: Voice rebuild replaces Denoising
    if (r.rebuild) assert.match(r.line, /^Voice rebuild, leveling, (Standard|Extra accurate) subtitles$/, r.line);
    else assert.equal(r.line.startsWith('Denoising ('), true, r.line);
    assert.ok(!/highs/i.test(r.line), r.line);
    // reachability counters
    if (m.rebuild) seen.rebuildModel++;
    if (!m.rebuild && anyCause(s)) seen.rebuildRule++;
    seen.strengths.add(r.strength);
    if (r.rebuild) seen.rebuilt++; else seen.kept++;
    if (CAUSES.filter((k) => c[k]).length > 1) seen.multi++;
    for (const k of CAUSES) if (onlyCauses(k)(s) && !m.rebuild) bump(seen.only, k);
    if (!anyCause(s) && m.rebuild) bump(seen.only, 'model');
    // both sides of each threshold
    if (!m.rebuild) {
      if (s.bandwidthHz === 6600 && onlyCauses('phone')(s)) bump(seen.edge, 'phone 6600 Hz is a phone call');
      if (s.bandwidthHz === 6601 && !r.rebuild) bump(seen.edge, 'phone 6601 Hz is not');
      if (s.clipFraction === 0.002 && !r.rebuild) bump(seen.edge, 'exactly 0.002 clipped is not distortion');
      if (s.clipFraction === 0.003 && onlyCauses('clipped')(s)) bump(seen.edge, '0.003 clipped is');
      if (s.dropDb === 10.5 && !r.rebuild) bump(seen.edge, 'a fall of exactly 10.5 dB is not an echo');
      if (s.dropDb === 6 && onlyCauses('echo')(s)) bump(seen.edge, 'a fall of 6 dB is an echo');
      if (s.tiltDb === -19 && onlyCauses('dull')(s)) bump(seen.edge, 'exactly -19 dB is muffled');
      if (s.tiltDb === -17 && s.bandwidthHz > 6600 && !r.rebuild) bump(seen.edge, '-17 dB is not');
    }
    REASONS.push(...r.reasons, r.line);
  }
  // the grid reaches every branch, or the checks above prove nothing
  const shown = JSON.stringify({ ...seen, strengths: [...seen.strengths] });
  assert.ok(seen.rebuildModel > 0 && seen.rebuildRule > 0 && seen.rebuilt > 0 && seen.kept > 0 && seen.multi > 0, shown);
  for (const k of [...CAUSES, 'model']) assert.ok(seen.only[k] > 0, `nothing but "${k}" is on the grid: ${shown}`);
  for (const k of ['phone 6600 Hz is a phone call', 'phone 6601 Hz is not', 'exactly 0.002 clipped is not distortion', '0.003 clipped is',
    'a fall of exactly 10.5 dB is not an echo', 'a fall of 6 dB is an echo', 'exactly -19 dB is muffled', '-17 dB is not']) assert.ok(seen.edge[k] > 0, `${k}: ${shown}`);
  assert.ok(seen.strengths.size >= 3, shown);
});

check('files: each cause has its sentence, and they come in the order echoey, muffled, phone, distorts, then "Rebuilding the voice fixes this", then "Next time" for distortion', () => {
  const grid = gridStats();
  const find = (f, what) => { const s = grid.find(f); assert.ok(s, `the grid has no ${what}`); return s; };
  const text = (s) => recommendForFile(s, {}).reasons;
  const noModel = (s) => !modelDecision(s).rebuild;
  const cases = [
    ['an echo', onlyCauses('echo'), [SAY.echo, SAY.fix]],
    ['a muffled voice', onlyCauses('dull'), [SAY.dull, SAY.fix]],
    ['a phone call', onlyCauses('phone'), [SAY.phone, SAY.fix]],
    ['distortion', onlyCauses('clipped'), [SAY.clipped, SAY.fix, SAY.next]],
    ['echo and a phone call', onlyCauses('echo', 'phone'), [SAY.echo, SAY.phone, SAY.fix]],
    ['echo and a muffled voice', onlyCauses('echo', 'dull'), [SAY.echo, SAY.dull, SAY.fix]],
    ['a phone call that distorts', onlyCauses('phone', 'clipped'), [SAY.phone, SAY.clipped, SAY.fix, SAY.next]],
    ['everything but a phone call', onlyCauses('echo', 'dull', 'clipped'), [SAY.echo, SAY.dull, SAY.clipped, SAY.fix, SAY.next]],
    ['echo, a phone call and distortion', onlyCauses('echo', 'phone', 'clipped'), [SAY.echo, SAY.phone, SAY.clipped, SAY.fix, SAY.next]],
  ];
  for (const [what, f, want] of cases) {
    const s = find((x) => f(x) && noModel(x), `file with ${what} that the model leaves alone`);
    const r = recommendForFile(s, {});
    assert.equal(r.rebuild, true, what);
    assert.deepEqual(said(r), want, what);
    // the sentences follow the strength sentence and come before the loudness and subtitle ones
    const at = (t) => r.reasons.indexOf(t);
    assert.ok(at(want[0]) > r.reasons.findIndex((t) => / is enough|clean and quiet/.test(t)), what);
    assert.ok(at(want[want.length - 1]) < r.reasons.findIndex((t) => /^For subtitles/.test(t)), what);
  }
  // a muffled voice is never also "a phone call": below 6600 Hz only the phone sentence is said
  assert.ok(grid.filter((s) => s.bandwidthHz <= 6600).every((s) => !text(s).includes(SAY.dull)));
  // only the model asked: the plain sentence, no cause named
  const modelOnly = find((s) => !anyCause(s) && modelDecision(s).rebuild, 'file that only the model asks to rebuild');
  assert.deepEqual(said(recommendForFile(modelOnly, {})), [SAY.clarity, SAY.fix]);
  assert.match(text(modelOnly).join(' '), /lost some of its clarity/);
  // and a cause is named, the plain sentence is not added
  assert.ok(grid.every((s) => !(anyCause(s) && text(s).includes(SAY.clarity))));
  // a file the voice is not rebuilt for gets no advice about missing high sounds: nothing restores them any more
  const noRestore = find((s) => !anyCause(s) && !modelDecision(s).rebuild, 'file that is left alone');
  assert.deepEqual(said(recommendForFile(noRestore, {})), []);
  assert.equal(recommendForFile(noRestore, {}).rebuild, false);
  assert.ok(grid.every((s) => !text(s).includes(RESTORE) && !/highs/i.test(recommendForFile(s, {}).line)));
  const phoneAny = grid.filter((s) => s.bandwidthHz <= 6600 && s.clipFraction <= 0.002);
  assert.ok(phoneAny.length > 100 && phoneAny.every((s) => recommendForFile(s, {}).rebuild && text(s).includes(SAY.phone) && !text(s).includes(RESTORE)));
  const gentleClean = find((s) => s.noiseDb >= T.CLEAN_AT_LEAST && modelDecision(s).strength === 'gentle', 'clean file at Gentle');
  assert.match(text(gentleClean).join(' '), /clean and quiet/);
  const notGentle = find((s) => modelDecision(s).strength === 'max', 'file at Maximum');
  assert.match(text(notGentle).join(' '), /Maximum is enough/);
});

check('files: a measurement that is missing is not a cause (no clipping figure, no tilt, no fall)', () => {
  const grid = gridStats();
  const calm = grid.find((s) => !anyCause(s) && !modelDecision(s).rebuild && s.tiltDb === -12 && s.dropDb === 30 && s.clipFraction === 0);
  assert.ok(calm);
  const base = recommendForFile(calm, {});
  assert.equal(base.rebuild, false);
  for (const gone of [{ clipFraction: null }, { clipFraction: undefined }, { clipFraction: NaN }]) {
    const r = recommendForFile({ ...calm, ...gone }, {});
    assert.deepEqual([r.rebuild, said(r)], [base.rebuild, said(base)], JSON.stringify(gone));
  }
  const dropless = grid.find((s) => !anyCause(s) && s.dropDb == null && s.clipFraction === 0 && s.tiltDb === -12 && !modelDecision(s).rebuild);
  assert.ok(dropless);
  assert.equal(recommendForFile(dropless, {}).rebuild, false);
  assert.equal(recommendForFile({ ...dropless, tiltDb: null }, {}).reasons.includes(SAY.dull), false);
  assert.equal(recommendForFile({ ...dropless, dropDb: null }, {}).reasons.includes(SAY.echo), false);
});

// ---- the causes heard in generated recordings (the grid above is made-up numbers) ------------------------------------------
const hotOf = (x) => x.map((a) => Math.max(-0.9995, Math.min(0.9995, a * 8))); // recorded too loud: clipped at the top
/** k times slower, so a voice that stops at 8 kHz stops below 2.7 kHz: a phone call, or a very small file */
const slowedBy = (x, k) => Float32Array.from({ length: x.length * k }, (_, i) => { const p = i / k; const j = Math.floor(p); const f = p - j; return x[j] * (1 - f) + (x[j + 1] ?? x[j]) * f; });
/** the voice rings on: four feedback delays of 21 to 51 ms (a big, bare room) */
const echoedBy = (x) => {
  const y = Float32Array.from(x);
  for (let i = 0; i < y.length; i++) for (const d of [1013, 1423, 1801, 2251]) if (i >= d) y[i] += (0.55 * 2 * y[i - d]) / 4;
  const peak = y.reduce((m, a) => Math.max(m, Math.abs(a)), 0);
  return y.map((a) => (a * 0.5) / peak);
};
const heard = (x) => { const st = analyse([x]); const r = recommendForFile(st, {}); REASONS.push(...r.reasons, r.line); return { st, r }; };

check('recordings: a phone-quality, an echoey and a too-loud recording get their voice rebuilt, each with its own sentence and the one fix; a clean one is left alone', () => {
  const clean = noisy(10, 55);
  const base = heard(clean);
  assert.ok(base.st.bandwidthHz > 6600 && base.st.tiltDb > -19 && base.st.dropDb > 10.5 && base.st.clipFraction === 0, JSON.stringify(base.st)); // nothing wrong with it
  assert.deepEqual([base.r.rebuild, said(base.r)], [false, []]);

  const loud = heard(hotOf(clean));
  assert.ok(loud.st.clipFraction > 0.002, String(loud.st.clipFraction));
  assert.equal(loud.r.rebuild, true);
  assert.ok(said(loud.r).includes(SAY.clipped) && said(loud.r).at(-2) === SAY.fix && said(loud.r).at(-1) === SAY.next, said(loud.r).join(' | '));
  assert.match(loud.r.line, /^Voice rebuild, leveling/);

  const phone = heard(slowedBy(noisy(4, 40), 3));
  assert.ok(phone.st.bandwidthHz <= 6600 && phone.st.clipFraction === 0, JSON.stringify(phone.st));
  assert.equal(phone.r.rebuild, true);
  assert.ok(said(phone.r).includes(SAY.phone) && !said(phone.r).includes(SAY.dull) && !said(phone.r).includes(SAY.next) && said(phone.r).at(-1) === SAY.fix, said(phone.r).join(' | '));
  assert.ok(!phone.r.reasons.includes(RESTORE) && !/highs/i.test(phone.r.line)); // rebuilt

  const room = heard(echoedBy(noisy(8, 40)));
  assert.ok(room.st.dropDb < 10.5 && room.st.clipFraction === 0, JSON.stringify(room.st));
  assert.equal(room.r.rebuild, true);
  assert.ok(said(room.r).includes(SAY.echo) && !said(room.r).includes(SAY.next) && said(room.r).at(-1) === SAY.fix, said(room.r).join(' | '));

  for (const h of [loud, phone, room]) assert.deepEqual(said(h.r), expectedSaid(h.st, modelDecision(h.st))); // the same as the rules say for these numbers
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
  assert.deepEqual([none.noSpeech, none.strength, none.rebuild, none.heavy, none.subs], [true, 'balanced', false, false, null]);
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

check('live room test is unchanged: a too-loud microphone is told "turn it down" and nothing else, and the rebuild causes (echo, a phone-quality sound, a muffled voice) say nothing there', () => {
  const NOW = { denoise: true, strength: 'gentle', captions: 'local' };
  const clean = noisy(10, 55);
  const loud = roomChanges(analyse([hotOf(clean)]), NOW);
  assert.deepEqual(loud.notes, ['Your microphone is too loud and distorts: turn it down.']);
  assert.deepEqual(roomChanges(analyse([clean]), NOW).notes, []);
  for (const x of [echoedBy(noisy(8, 40)), slowedBy(noisy(4, 40), 3)]) {
    const r = roomChanges(analyse([x]), NOW);
    assert.deepEqual(r.notes, []);
    assert.deepEqual(Object.keys(r.settings).sort(), ['captions', 'denoise', 'strength']); // no rebuild among the live settings
    assert.ok(r.changes.every((c) => c.key === 'strength' || c.key === 'captions'));
  }
  for (const s of gridStats()) { // the note is about the clipping figure alone
    const r = roomChanges(s, NOW);
    assert.deepEqual(r.notes, s.clipFraction > 0.002 ? ['Your microphone is too loud and distorts: turn it down.'] : [], JSON.stringify(s));
    assert.ok(![...r.changes.map((c) => c.text), ...r.notes].some((t) => /rebuil|Next time|\bphone\b|\becho|muffled/i.test(t)), JSON.stringify(s));
  }
  const src = readFileSync(join(ROOT, 'web', 'src', 'recommend-room.js'), 'utf8');
  assert.ok(src.includes("'Your microphone is too loud and distorts: turn it down.'"));
  assert.ok(!/rebuil/i.test(src), 'the live advice does not talk about rebuilding the voice'); // that is for files (src/recommend.js)
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
  for (const t of texts) assert.ok(!GONE.test(t), `"${t}" is the old advice`); // "Cleaning can’t fix that, so turn the microphone level down next time."
  for (const t of Object.values(SAY)) assert.ok(texts.includes(t), `the advice never said: ${t}`); // every sentence above was produced, and so was checked
});

check('the old sentence "Cleaning can’t fix that" is nowhere in the website or its docs', () => {
  const found = [];
  const walk = (dir) => {
    for (const e of readdirSync(dir, { withFileTypes: true })) {
      const p = join(dir, e.name);
      if (e.isDirectory()) { if (e.name !== 'node_modules') walk(p); } else if (/\.(jsx?|mjs|css|html|md)$/.test(e.name) && GONE.test(readFileSync(p, 'utf8'))) found.push(p.replace(ROOT, ''));
    }
  };
  walk(join(ROOT, 'web', 'src'));
  walk(join(ROOT, 'docs'));
  assert.deepEqual(found, []);
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
