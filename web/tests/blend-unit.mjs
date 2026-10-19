// The numbers behind the strength slider and the time-left line, no browser and no backend: src/media/blend.js (the share of the
// recording that stays in the result, the slider's positions, the gains, the loudness match, the blended arrays) and src/eta.js
// (the words and the estimate of the time left). The two files are bundled with esbuild (the site's own build tool) so that
// `import.meta.env` in src/api.js, which plain node does not know, is defined.  Usage: node tests/blend-unit.mjs
import assert from 'node:assert/strict';
import { dirname } from 'node:path';
import { fileURLToPath } from 'node:url';
import { build } from 'esbuild';

const HERE = dirname(fileURLToPath(import.meta.url));
const bundle = await build({
  stdin: {
    contents: "export * from '../src/media/blend.js'; export * from '../src/eta.js'; export { STRENGTHS } from '../src/strengths.js';",
    resolveDir: HERE,
    loader: 'js',
  },
  bundle: true,
  format: 'esm',
  platform: 'node',
  write: false,
  logLevel: 'silent',
  define: { 'import.meta.env.BASE_URL': '"/"' },
});
const m = await import(`data:text/javascript;base64,${Buffer.from(bundle.outputFiles[0].text).toString('base64')}`);
const { blendChannels, blendToPcm, dbAt, energySums, etaSeconds, etaText, forgetReadings, gainsFor, matchGain, mixOf, noteJobs, posOf, splitParts, strengthName, STRENGTHS } = m;

let failed = 0;
async function check(name, fn) {
  try {
    const detail = await fn();
    console.log(`PASS ${name}${detail ? ` -- ${detail}` : ''}`);
  } catch (e) {
    failed++;
    console.log(`FAIL ${name} -- ${String(e.message).split('\n')[0]}`);
  }
}

const near = (a, b, tol, what = '') => assert.ok(Math.abs(a - b) <= tol, `${what} ${a} vs ${b} (tolerance ${tol})`);
let seed = 7;
const rnd = () => { seed = (seed * 1103515245 + 12345) & 0x7fffffff; return seed / 0x7fffffff; };
/** Two correlated "signals" like the kept pair: the cleaned one is the recording with part of it removed, plus a little of its own. */
function pair(channels = 1, frames = 4000, loud = 1) {
  const full = [];
  const input = [];
  for (let c = 0; c < channels; c++) {
    const f = new Float32Array(frames);
    const x = new Float32Array(frames);
    for (let i = 0; i < frames; i++) {
      const voice = Math.sin(i / (13 + c)) * 0.4 * loud;
      const noise = (rnd() - 0.5) * 0.3 * loud;
      x[i] = voice + noise;
      f[i] = voice * 0.9 + noise * 0.1 + (rnd() - 0.5) * 0.01 * loud;
    }
    full.push(f);
    input.push(x);
  }
  return { full, input };
}
const energy = (arrs) => arrs.reduce((s, a) => s + a.reduce((t, v) => t + v * v, 0), 0);

// ---------------------------------------------------------------------------------------------------- the share of the recording

await check('mixOf: the share of the recording is 10^(-dB/20), nothing at 100 dB and above, all of it at 0 dB', () => {
  near(mixOf(24), 10 ** (-24 / 20), 1e-12);
  near(mixOf(24), 0.0630957, 1e-6);
  near(mixOf(12), 0.2511886, 1e-6);
  near(mixOf(40), 0.01, 1e-12);
  assert.equal(mixOf(100), 0);
  assert.equal(mixOf(120), 0);
  assert.equal(mixOf(0), 1);
  assert.equal(mixOf(-24), mixOf(24), 'the sign does not matter');
  assert.equal(mixOf('24'), mixOf(24), 'a number given as text');
  for (let db = 0; db < 99; db += 3) assert.ok(mixOf(db + 3) < mixOf(db), `falls with the strength at ${db}`);
  return 'a = 0.2512, 0.0631, 0.01, 0 for Gentle, Balanced, Strong, Maximum';
});

await check('the four names: Gentle 12 dB, Balanced 24, Strong 40, Maximum 100, in that order', () => {
  assert.deepEqual(STRENGTHS.map((s) => [s.label, s.db]), [['Gentle', 12], ['Balanced', 24], ['Strong', 40], ['Maximum', 100]]);
  assert.deepEqual(STRENGTHS.map((s) => s.id), ['gentle', 'balanced', 'strong', 'max']);
});

// ---------------------------------------------------------------------------------------------------- the slider's positions

await check('dbAt and posOf: the slider runs 0 to 3 over the four names, equally spaced, and the two are inverse over the whole range', () => {
  [[0, 12], [1, 24], [2, 40], [3, 100]].forEach(([pos, db]) => {
    near(dbAt(pos), db, 1e-9, `dbAt(${pos})`);
    near(posOf(db), pos, 1e-9, `posOf(${db})`);
  });
  near(dbAt(0.5), 18, 1e-9);
  near(dbAt(1.5), 32, 1e-9);
  near(dbAt(2.5), 70, 1e-9);
  near(posOf(18), 0.5, 1e-9);
  near(posOf(70), 2.5, 1e-9);
  assert.equal(dbAt(-1), 12);
  assert.equal(dbAt(7), 100);
  assert.equal(posOf(3), 0, 'below Gentle sticks to the first name');
  assert.equal(posOf(500), 3, 'above Maximum sticks to the last name');
  let worst = 0;
  for (let p = 0; p <= 3; p += 0.01) worst = Math.max(worst, Math.abs(posOf(dbAt(p)) - p));
  for (let db = 12; db <= 100; db += 0.37) worst = Math.max(worst, Math.abs(dbAt(posOf(db)) - db) / 29);
  assert.ok(worst < 1e-9, `round trip off by ${worst}`);
  for (let p = 0; p < 3; p += 0.05) assert.ok(dbAt(p + 0.05) > dbAt(p), 'the strength only rises along the slider');
  return 'round trip exact to 1e-9';
});

await check('strengthName: a named strength shows its name (within half a dB), anything between shows the two names around it', () => {
  assert.equal(strengthName(12), 'Gentle');
  assert.equal(strengthName(24.3), 'Balanced');
  assert.equal(strengthName(40), 'Strong');
  assert.equal(strengthName(100), 'Maximum');
  assert.equal(strengthName(18), 'Between Gentle and Balanced');
  assert.equal(strengthName(32), 'Between Balanced and Strong');
  assert.equal(strengthName(70), 'Between Strong and Maximum');
});

// ---------------------------------------------------------------------------------------------------- the gains

await check('gainsFor without a loudness match: full = (1 - a), input = a (24 dB gives 0.93690 and 0.06310, 100 dB gives 1 and 0)', () => {
  const g = gainsFor(24);
  assert.equal(g.g, 1);
  assert.equal(g.full.toFixed(5), '0.93690');
  assert.equal(g.input.toFixed(5), '0.06310');
  const max = gainsFor(100);
  assert.deepEqual([max.full, max.input, max.a], [1, 0, 0]);
  const gentle = gainsFor(12);
  assert.equal(gentle.full.toFixed(5), '0.74881');
  assert.equal(gentle.input.toFixed(5), '0.25119');
  for (const db of [12, 18, 24, 32, 40, 70, 100]) {
    const x = gainsFor(db);
    near(x.full + x.input, 1, 1e-12, `the two gains of ${db} dB add up to g`);
  }
});

await check('gainsFor with a loudness match: both gains are scaled by g, the gain that gives the blend the loudness of the full-strength sound', () => {
  const { full, input } = pair(2, 6000);
  const sums = energySums(full, input);
  for (const db of [12, 24, 40, 100]) {
    const a = mixOf(db);
    const x = gainsFor(db, { sums });
    near(x.g, matchGain(sums, a), 1e-12);
    near(x.full, (1 - a) * x.g, 1e-12);
    near(x.input, a * x.g, 1e-12);
  }
  assert.equal(gainsFor(100, { sums }).g, 1, 'full strength is the reference: no change');
});

// ---------------------------------------------------------------------------------------------------- the loudness match

await check('energySums are the sums of squares and of the products, over every channel', () => {
  const f = [Float32Array.from([1, 2, 3]), Float32Array.from([0.5, -0.5, 1])];
  const x = [Float32Array.from([1, 0, -1]), Float32Array.from([2, 2, 2])];
  const s = energySums(f, x);
  near(s.ef, 1 + 4 + 9 + 0.25 + 0.25 + 1, 1e-9);
  near(s.ei, 1 + 0 + 1 + 4 + 4 + 4, 1e-9);
  near(s.cfi, 1 + 0 - 3 + 1 - 1 + 2, 1e-9);
});

await check('matchGain: the closed form gives the same loudness as blending the arrays and measuring them (stereo, many strengths)', () => {
  const { full, input } = pair(2, 8000);
  const sums = energySums(full, input);
  const ef = energy(full);
  let worst = 0;
  let tried = 0;
  for (let db = 0; db <= 100; db += 4) {
    const a = mixOf(db);
    // brute force: blend sample by sample, measure the energy of the result
    const blended = full.map((f, c) => Float32Array.from(f, (v, i) => (1 - a) * v + a * input[c][i]));
    const brute = Math.sqrt(ef / energy(blended));
    const clamped = Math.max(0.1, Math.min(3, brute));
    const closed = matchGain(sums, a);
    worst = Math.max(worst, Math.abs(closed - clamped) / clamped);
    tried++;
    if (clamped === brute) near(Math.sqrt(energy(blended.map((b) => b.map((v) => v * closed))) / ef), 1, 1e-4, `the matched blend at ${db} dB has the loudness of the full sound`);
  }
  assert.ok(worst < 1e-5, `closed form off by ${(worst * 100).toExponential(2)} %`);
  return `${tried} strengths, worst relative difference ${worst.toExponential(1)}`;
});

await check('matchGain: a blend that comes out louder is turned down, one that comes out quieter is turned up, at most x3 up and x0.1 down; nothing to measure gives 1', () => {
  const sig = Float32Array.from({ length: 2000 }, (_, i) => Math.sin(i / 9) * 0.3);
  // the cleaned sound is the recording at half the level, so mixing the recording back in makes it louder
  const quiet = energySums([Float32Array.from(sig, (v) => v * 0.5)], [sig]);
  assert.ok(matchGain(quiet, 0.5) < 1, 'louder blend: turned down');
  const loud = energySums([sig], [Float32Array.from(sig, (v) => v * 0.5)]);
  assert.ok(matchGain(loud, 0.5) > 1, 'quieter blend: turned up');
  // nearly opposite signals: the blend of equal parts is almost silent, so the gain hits its ceiling
  const opp = energySums([sig], [Float32Array.from(sig, (v) => -0.98 * v)]);
  assert.equal(matchGain(opp, 0.5), 3);
  // a recording far louder than the cleaned sound: floor
  const huge = energySums([Float32Array.from(sig, (v) => v * 0.001)], [Float32Array.from(sig, (v) => v * 1000)]);
  assert.equal(matchGain(huge, 0.9), 0.1);
  assert.equal(matchGain({ ef: 0, ei: 1, cfi: 0 }, 0.5), 1, 'a silent cleaned sound');
  assert.equal(matchGain({ ef: 1, ei: 0, cfi: 0 }, 1), 1, 'nothing left to measure');
  assert.equal(matchGain({ ef: 1, ei: 1, cfi: 1 }, 0), 1, 'the full sound itself needs no change');
});

// ---------------------------------------------------------------------------------------------------- the blended sound

await check('blendChannels: every sample is (1 - a) * cleaned + a * recording, per channel; at 100 dB the cleaned channels come back as they are', () => {
  const { full, input } = pair(2, 3000);
  for (const db of [12, 24, 40, 70]) {
    const a = mixOf(db);
    const out = blendChannels(full, input, db);
    assert.equal(out.length, 2);
    let worst = 0;
    out.forEach((ch, c) => {
      assert.equal(ch.length, 3000);
      assert.ok(ch instanceof Float32Array);
      for (let i = 0; i < ch.length; i++) worst = Math.max(worst, Math.abs(ch[i] - ((1 - a) * full[c][i] + a * input[c][i])));
    });
    assert.ok(worst < 1e-6, `${db} dB: off by ${worst}`); // float32 storage
  }
  const same = blendChannels(full, input, 100);
  same.forEach((ch, c) => assert.equal(ch, full[c], 'the same array, nothing is copied or changed'));
  const zero = blendChannels(full, input, 0);
  zero.forEach((ch, c) => { for (let i = 0; i < ch.length; i += 97) assert.equal(ch[i], input[c][i]); });
  return 'float32 exact to 1e-6';
});

await check('blendToPcm: the same blend as parts of interleaved float32 with the right length, peak and loudness; a plan can copy a channel or leave one silent', async () => {
  const { full, input } = pair(1, 5000);
  const db = 24;
  const pcm = blendToPcm(full, input, db, [0, 0]);
  assert.equal(pcm.channels, 2);
  assert.equal(pcm.frames, 5000);
  assert.equal(pcm.sampleRate, 48000);
  const [l, r] = await splitParts(pcm.parts, 2);
  const want = blendChannels(full, input, db)[0];
  for (let i = 0; i < 5000; i += 13) { assert.equal(l[i], want[i]); assert.equal(r[i], want[i]); }
  near(pcm.rms, Math.sqrt(energy([want]) / 5000), 1e-6);
  near(pcm.peak, want.reduce((p, v) => Math.max(p, Math.abs(v)), 0), 1e-9);
  const silent = blendToPcm(full, input, db, [0, null]);
  const [a, b] = await splitParts(silent.parts, 2);
  assert.equal(b.every((v) => v === 0), true);
  assert.equal(a[100], want[100]);
});

// ---------------------------------------------------------------------------------------------------- the time left, in words

await check('etaText: "about 5 min left", "less than a minute left", "about 1 h 20 min left", and nothing when there is nothing to say', () => {
  assert.equal(etaText(300), 'about 5 min left');
  assert.equal(etaText(20), 'less than a minute left');
  assert.equal(etaText(0), 'less than a minute left');
  assert.equal(etaText(29), 'less than a minute left');
  assert.equal(etaText(30), 'about 1 min left');
  assert.equal(etaText(89), 'about 1 min left');
  assert.equal(etaText(4800), 'about 1 h 20 min left');
  assert.equal(etaText(3600), 'about 1 h 00 min left');
  assert.equal(etaText(3 * 3600 + 5 * 60), 'about 3 h 05 min left');
  assert.equal(etaText(59 * 60), 'about 59 min left');
  assert.equal(etaText(null), '');
  assert.equal(etaText(undefined), '');
  assert.equal(etaText(NaN), '');
  assert.equal(etaText(Infinity), '');
  for (const s of [0, 20, 300, 4800, 9000]) assert.ok(!/premium|tier|plan|polish|server/i.test(etaText(s)), 'plain words only');
});

// ---------------------------------------------------------------------------------------------------- the time left, in seconds

const T0 = 1_000_000_000_000;
const run = (id, progress, extra = {}) => ({ id, status: 'running', progress, ...extra });

await check('etaSeconds without any reading: the server\'s own estimate scaled by what is left of the bar; a queued job is its wait plus its estimate; uploading, done and unknown give nothing', () => {
  forgetReadings();
  near(etaSeconds(run('a', 0, { estimateSeconds: 100 }), T0), 100, 1e-9);
  near(etaSeconds(run('a', 25, { estimateSeconds: 100 }), T0), 75, 1e-9);
  near(etaSeconds(run('a', 100, { estimateSeconds: 100 }), T0), 0, 1e-9);
  near(etaSeconds(run('a', 250, { estimateSeconds: 100 }), T0), 0, 1e-9, 'a bar past 100 is clamped');
  assert.equal(etaSeconds(run('a', 50), T0), null, 'no estimate, no readings');
  assert.equal(etaSeconds({ id: 'q', status: 'queued', queueWaitSeconds: 120, estimateSeconds: 60 }, T0), 180);
  assert.equal(etaSeconds({ id: 'q', status: 'queued', queueWaitSeconds: 120 }, T0), 120);
  assert.equal(etaSeconds({ id: 'q', status: 'queued', estimateSeconds: 60 }, T0), 60);
  assert.equal(etaSeconds({ id: 'q', status: 'queued' }, T0), null);
  for (const status of ['uploading', 'done', 'failed', 'canceled', 'expired']) assert.equal(etaSeconds({ id: 'x', status, estimateSeconds: 60, progress: 10 }, T0), null, status);
  near(etaSeconds({ id: 'c', status: 'cloud_asr', progress: 50, estimateSeconds: 40 }, T0), 20, 1e-9, 'the cloud stage counts as running');
});

await check('etaSeconds from progress readings: the speed since the bar began to move (1 % a second, 60 % left: a minute), once the bar has moved enough and long enough', () => {
  forgetReadings();
  noteJobs([run('r', 10)], T0); // the first reading: nothing moved yet
  assert.equal(etaSeconds(run('r', 10), T0), null);
  noteJobs([run('r', 20)], T0 + 10_000); // it moved: this is where the speed is measured from
  assert.equal(etaSeconds(run('r', 20), T0 + 10_000), null, 'moved, but no time has passed since');
  noteJobs([run('r', 40)], T0 + 30_000); // 20 % in 20 s
  near(etaSeconds(run('r', 40, { estimateSeconds: 300 }), T0 + 30_000), 60, 1e-9, 'fully trusted after 15 % of movement, whatever the estimate says');
  near(etaSeconds(run('r', 40), T0 + 30_000), 60, 1e-9, 'no estimate: the measured speed alone');
  noteJobs([run('r', 70)], T0 + 50_000); // faster: 50 % in 40 s
  near(etaSeconds(run('r', 70), T0 + 50_000), 30 / (50 / 40), 1e-9);
});

await check('etaSeconds: before 15 % of movement the measured speed and the estimate are blended; too little movement or too little time falls back to the estimate', () => {
  forgetReadings();
  noteJobs([run('b', 10)], T0);
  noteJobs([run('b', 20)], T0 + 1000);
  noteJobs([run('b', 27.5)], T0 + 6000); // 7.5 % in 5 s = 1.5 % a second; half of the 15 % that is fully trusted
  const byRate = (100 - 27.5) / 1.5;
  const byEst = 200 * (1 - 0.275);
  near(etaSeconds(run('b', 27.5, { estimateSeconds: 200 }), T0 + 6000), 0.5 * byRate + 0.5 * byEst, 1e-9);
  forgetReadings();
  noteJobs([run('s', 10)], T0);
  noteJobs([run('s', 11)], T0 + 1000);
  noteJobs([run('s', 11.5)], T0 + 20_000); // 0.5 % since it moved: under the 2 % needed
  near(etaSeconds(run('s', 11.5, { estimateSeconds: 100 }), T0 + 20_000), 100 * 0.885, 1e-9, 'too little movement');
  forgetReadings();
  noteJobs([run('t', 10)], T0);
  noteJobs([run('t', 20)], T0 + 1000);
  noteJobs([run('t', 30)], T0 + 3000); // 10 % but only 2 s since it moved: under the 5 s needed
  near(etaSeconds(run('t', 30, { estimateSeconds: 100 }), T0 + 3000), 70, 1e-9, 'too little time');
  assert.equal(etaSeconds(run('t', 30), T0 + 3000), null, 'and with no estimate nothing is claimed');
});

await check('etaSeconds keeps up with the clock between two readings, starts over when a restart puts the bar back, and forgets jobs that are no longer running', () => {
  forgetReadings();
  noteJobs([run('k', 10)], T0);
  noteJobs([run('k', 20)], T0 + 5000);
  noteJobs([run('k', 30)], T0 + 15_000); // 1 % a second since it moved
  near(etaSeconds(run('k', 30), T0 + 15_000), 70, 1e-9);
  near(etaSeconds(run('k', 30), T0 + 25_000), 140, 1e-9, 'ten seconds later with no new reading: the bar is late, so the figure grows, it does not stand still');
  noteJobs([run('k', 5)], T0 + 40_000); // put back in the queue and started again
  assert.equal(etaSeconds(run('k', 5), T0 + 40_000), null, 'the old speed is not used for the new run');
  noteJobs([{ id: 'k', status: 'done', progress: 100 }], T0 + 50_000);
  noteJobs([run('k', 8)], T0 + 60_000); // a fresh start: the old readings were dropped when it was last seen done
  assert.equal(etaSeconds(run('k', 8), T0 + 60_000), null);
  noteJobs([run('k2', 10), { id: 'q1', status: 'queued', progress: 0 }], T0);
  noteJobs([run('k2', 20)], T0 + 5000);
  noteJobs([], T0 + 6000);
  noteJobs([run('k2', 90)], T0 + 7000);
  assert.equal(etaSeconds(run('k2', 90), T0 + 7000), null, 'a job missing from one list starts over');
});

console.log(failed ? `${failed} FAILED` : 'all passed');
process.exit(failed ? 1 : 0);
