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
    contents: "export * from '../src/media/blend.js'; export * from '../src/eta.js'; export { STRENGTHS } from '../src/strengths.js'; export { keptUrl } from '../src/api.js';",
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
const { BlendPlayer, blendChannels, blendToPcm, dbAt, energySums, etaSeconds, etaText, forgetReadings, gainsFor, keptSource, keptUrl, matchGain, memorySource, mixOf, noteJobs, posOf, previewFits, splitParts, strengthName, STRENGTHS } = m;

let failed = 0;
async function check(name, fn) {
  try {
    const detail = await fn();
    console.log(`PASS ${name}${detail ? ` -- ${detail}` : ''}`);
  } catch (e) {
    failed++;
    console.log(`FAIL ${name} -- ${String(e.message).split("\n").slice(0, 4).join(" / ")}`);
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


// ---------------------------------------------------------------------------------------------------- the player and its sources
// BlendPlayer plays from stretches of the two signals (a source hands them out): these checks run it on a fake audio context with a clock
// of their own, so what is scheduled, when, how much is held at a time, what a seek or a late stretch does, can be read off exactly.

const SR = 48000;
const sleep = (ms) => new Promise((r) => setTimeout(r, ms));
const settle = () => sleep(8);
const f32 = Math.fround;

class FakeCtx {
  constructor() {
    this.currentTime = 0;
    this.destination = {};
    this.nodes = [];
    this.buffers = 0;
    this.closed = false;
  }
  createGain() { const g = { gain: { value: 1, calls: [], setTargetAtTime(v, t) { this.value = v; this.calls.push([v, t]); } }, connect() {} }; return g; }
  createBuffer(channels, frames, rate) {
    this.buffers++;
    const light = this.light; // a very long run: the sound itself is not kept
    return { numberOfChannels: channels, length: frames, sampleRate: rate, duration: frames / rate, data: Array.from({ length: channels }, () => new Float32Array(light ? 0 : frames)), copyToChannel(a, c) { if (!light) this.data[c].set(a); } };
  }
  createBufferSource() {
    const ctx = this;
    const n = { buffer: null, at: null, off: null, stopped: false, connect() {}, disconnect() {}, start(at, off) { this.at = at; this.off = off ?? 0; ctx.nodes.push(this); }, stop() { this.stopped = true; } };
    return n;
  }
  resume() { return Promise.resolve(); }
  close() { this.closed = true; return Promise.resolve(); }
}

/** A source of `seconds` of one or two channels whose samples say where they are: full[i] = i / 1e7, input[i] = -i / 1e7 (exact to a float). */
function fakeSource(seconds, { channels = 1, plan = null, delay = null, cheap = false } = {}) {
  const samples = Math.round(seconds * SR);
  const log = [];
  const shared = cheap ? Array.from({ length: channels }, () => new Float32Array(10 * SR)) : null; // every stretch is the same silence
  const make = (sign, first, count) => (cheap ? shared.map((a) => a.subarray(0, count)) : Array.from({ length: channels }, (_, c) => {
    const a = new Float32Array(count);
    for (let i = 0; i < count; i++) a[i] = sign * (first + i + c * 0.5) / 1e7;
    return a;
  }));
  return {
    kind: 'fake',
    samples,
    plan: plan || Array.from({ length: channels }, (_, i) => i),
    maxSeconds: 30,
    sums: () => null,
    log,
    async read(first, count, signal) {
      const entry = { first, count, aborted: false, done: false };
      log.push(entry);
      signal?.addEventListener('abort', () => { entry.aborted = true; });
      if (delay) await delay(first / SR, entry);
      if (signal?.aborted) throw new DOMException('Canceled', 'AbortError');
      entry.done = true;
      return { full: make(1, first, count), input: make(-1, first, count), bytes: count * 8 };
    },
  };
}
const player = (source, opts = {}) => new BlendPlayer(source, { tickMs: 0, seekDelayMs: 1, chunkSeconds: 1, context: () => new FakeCtx(), ...opts });
const startsOf = (p) => p.ctx.nodes.filter((n) => !n.stopped).map((n) => [Number(n.at.toFixed(6)), Number(n.off.toFixed(6)), n.buffer.length]);

await check('BlendPlayer: Play schedules the first stretch at once and the next ones straight after it, both signals at the same instants, gapless (each starts where the one before ends)', async () => {
  const src = fakeSource(30, { channels: 2 });
  const p = player(src);
  await p.play();
  await settle();
  const full = p.ctx.nodes.filter((n) => n.buffer.data[0][0] === 0 || true);
  // two nodes (cleaned, recording) for each of the stretches held: 0, 1, 2 (the one that plays and CHUNKS_AHEAD after it)
  assert.equal(p.ctx.nodes.length, 6, 'three stretches, two signals each');
  const at = [...new Set(p.ctx.nodes.map((n) => n.at.toFixed(6)))].map(Number);
  assert.deepEqual(at.map((t) => Number((t - at[0]).toFixed(6))), [0, 1, 2], 'one second apart, no gap and no overlap');
  assert.ok(Math.abs(at[0] - 0.05) < 1e-9, 'a moment after Play');
  for (let i = 0; i < 6; i += 2) assert.equal(p.ctx.nodes[i].at, p.ctx.nodes[i + 1].at, 'the two signals start together');
  // the sound in the nodes is the sound of that stretch of the recording: sample for sample
  const f1 = p.ctx.nodes[2].buffer.data[0];
  const x1 = p.ctx.nodes[3].buffer.data[0];
  assert.equal(f1[0], f32(SR / 1e7));
  assert.equal(x1[100], f32(-(SR + 100) / 1e7));
  assert.equal(p.ctx.nodes[2].buffer.numberOfChannels, 2);
  p.close();
});

await check('BlendPlayer: only a few stretches are ever held, however long the recording is (a 3-hour recording played through: at most 4 held, every stretch asked for once, in order, none twice)', async () => {
  const src = fakeSource(3 * 3600, { channels: 2, cheap: true });
  const p = new BlendPlayer(src, { tickMs: 0, chunkSeconds: 10, context: () => Object.assign(new FakeCtx(), { light: true }) });
  assert.equal(p.chunkCount, 1080);
  await p.play();
  let most = 0;
  let nodesMost = 0;
  const clock = p.ctx;
  for (let t = 0; t < 3 * 3600 - 5; t += 3) {
    clock.currentTime += 3;
    p.tick();
    await Promise.resolve();
    most = Math.max(most, p.held.size);
    nodesMost = Math.max(nodesMost, p.ctx.nodes.filter((n) => !n.stopped).length);
    if (t % 600 === 0) await sleep(0);
    if (p.error) throw new Error(p.error);
  }
  assert.ok(most <= 4, `held ${most}`);
  assert.ok(nodesMost <= 8, `${nodesMost} sources playing or waiting at once (4 stretches, two signals each)`);
  const firsts = src.log.map((e) => e.first / (10 * SR));
  assert.equal(new Set(firsts).size, firsts.length, 'no stretch was asked for twice');
  assert.deepEqual(firsts, [...firsts].sort((a, b) => a - b), 'in order');
  assert.ok(firsts.length >= 1070, `${firsts.length} stretches asked for on the way`);
  assert.ok(p.snapshot().held <= 4);
  p.close();
  assert.equal(p.held.size, 0);
  return `at most ${most} held, ${nodesMost} sources at once, ${firsts.length} stretches fetched`;
});

await check('BlendPlayer: the next stretches are fetched ahead (the one playing and two more), the one behind is kept for a step back, older ones are let go', async () => {
  const src = fakeSource(60, { channels: 1 });
  const p = player(src);
  await p.play();
  await settle();
  assert.deepEqual([...p.held.keys()].sort((a, b) => a - b), [0, 1, 2]);
  p.ctx.currentTime = 1.5; // playing stretch 1
  p.tick(); await settle();
  assert.deepEqual([...p.held.keys()].sort((a, b) => a - b), [0, 1, 2, 3]);
  p.ctx.currentTime = 3.2; // playing stretch 3: 2 is kept behind, 0 and 1 are gone
  p.tick(); await settle();
  assert.deepEqual([...p.held.keys()].sort((a, b) => a - b), [2, 3, 4, 5]);
  const live = startsOf(p);
  assert.ok(live.every(([, , len]) => len === SR));
  assert.equal(p.ctx.nodes.filter((n) => n.stopped).length, 4, 'the sources of the two let go were stopped (two signals each)');
  assert.equal(Number(p.time.toFixed(2)), 3.15, 'the time is the context clock less the 0.05 s head start');
  p.close();
});

await check('BlendPlayer: a seek leaves the held stretches, drops them, fetches the ones at the new place (after a short pause, not for every position on the way) and plays from the exact second', async () => {
  const src = fakeSource(120, { channels: 1 });
  const p = player(src, { seekDelayMs: 20 });
  await p.play();
  await settle();
  const before = src.log.length;
  p.seek(30.5); p.seek(31.25); p.seek(80.4); // dragging the range
  assert.equal(p.snapshot().buffering, true);
  assert.equal(p.ctx.nodes.filter((n) => !n.stopped).length, 0, 'the old sound stopped at once');
  await sleep(5);
  assert.equal(src.log.length, before, 'nothing is asked for while the range is moving');
  await sleep(60);
  const asked = src.log.slice(before).map((e) => e.first / SR);
  assert.ok(asked.includes(80) && asked.every((t) => t >= 79 && t <= 82), `asked for ${asked}`);
  assert.deepEqual([...p.held.keys()].filter((k) => k < 79 || k > 83), [], 'nothing from the old place is held');
  assert.equal(p.snapshot().buffering, false);
  const first = p.ctx.nodes.filter((n) => !n.stopped)[0];
  assert.ok(Math.abs(first.off - 0.4) < 1e-9, `starts ${first.off} s into the stretch`);
  assert.equal(first.buffer.data[0][0], f32(80 * SR / 1e7), 'it is the stretch that starts at 80 s');
  p.ctx.currentTime += 2;
  assert.ok(Math.abs(p.time - (80.4 + 2 - 0.0)) < 0.06, `time ${p.time}`);
  p.close();
});

await check('BlendPlayer: a seek while paused moves the position, asks for nothing until Play (but the stretches there are fetched ahead after a pause), and Play starts from there', async () => {
  const src = fakeSource(60, { channels: 1 });
  const p = player(src, { seekDelayMs: 5 });
  p.prepare(); await settle();
  assert.equal(src.log.length, 3, 'the first stretch and two ahead are asked for before Play');
  p.seek(40.5);
  assert.equal(p.time, 40.5);
  await sleep(30);
  assert.deepEqual([...p.held.keys()].sort((a, b) => a - b), [40, 41, 42]);
  await p.play();
  await settle();
  const first = p.ctx.nodes[0];
  assert.ok(Math.abs(first.off - 0.5) < 1e-9);
  assert.equal(first.buffer.data[0][0], f32(40 * SR / 1e7));
  p.pause();
  assert.ok(Math.abs(p.time - 40.5) < 1e-6, 'paused where it was');
  assert.equal(p.ctx.nodes.filter((n) => !n.stopped).length, 0);
  p.close();
});

await check('BlendPlayer: a stretch that is late makes the sound wait for it (buffering, the time stands still) and go on from the end of the last one, with no gap or skip', async () => {
  let release;
  const gate = new Promise((r) => { release = r; });
  const src = fakeSource(30, { channels: 1, delay: async (t) => { if (t === 3) await gate; } });
  const p = player(src);
  await p.play();
  await settle();
  assert.equal(p.scheduledTo, 2, 'stretches 0 to 2 are scheduled; 3 is on its way');
  p.ctx.currentTime = 2.9;
  p.tick();
  assert.equal(p.snapshot().buffering, false);
  p.ctx.currentTime = 3.06; // the end of stretch 2 (head start 0.05)
  p.tick();
  assert.equal(p.snapshot().buffering, true, 'waiting');
  assert.equal(p.playing, true, 'still wants sound');
  assert.equal(p.time, 3, 'the time stands still at the end of what played');
  p.ctx.currentTime = 7; // a long time passes
  assert.equal(p.time, 3);
  release();
  await settle();
  assert.equal(p.snapshot().buffering, false);
  const next = p.ctx.nodes.filter((n) => !n.stopped);
  assert.ok(next.length >= 2 && next[0].buffer.data[0][0] === f32(3 * SR / 1e7) && next[0].off === 0, 'it goes on with stretch 3 from its first sample');
  assert.ok(Math.abs(next[0].at - 7.05) < 0.02 || next[0].at >= 7, `starts now, not in the past: ${next[0].at}`);
  assert.ok(Math.abs(p.time - 3) < 0.1, `time ${p.time}`);
  p.close();
});

await check('BlendPlayer: Pause during buffering stops the wait; the end of the recording stops by itself and calls onEnd; Play after the end starts from the beginning', async () => {
  let ended = 0;
  const src = fakeSource(3, { channels: 1 });
  const p = player(src, { onEnd: () => { ended++; } });
  await p.play();
  await settle();
  p.ctx.currentTime = 3.04;
  p.tick();
  assert.equal(ended, 1);
  assert.equal(p.playing, false);
  assert.equal(p.time, 0);
  assert.equal(p.ctx.nodes.filter((n) => !n.stopped).length, 0);
  await p.play();
  await settle();
  assert.equal(p.ctx.nodes.filter((n) => !n.stopped)[0].buffer.data[0][0], 0, 'from the beginning');
  p.pause();
  // Pause while the first stretch is still on its way
  let release;
  const slow = fakeSource(10, { channels: 1, delay: async () => { await new Promise((r) => { release = r; }); } });
  const q = player(slow);
  const playing = q.play();
  await settle();
  assert.equal(q.snapshot().buffering, true);
  q.pause();
  assert.equal(q.snapshot().buffering, false);
  release();
  await playing;
  await settle();
  assert.equal(q.ctx.nodes.length, 0, 'nothing was scheduled after the pause');
  assert.equal(q.playing, false);
  p.close(); q.close();
});

await check('BlendPlayer: a stretch that can not be fetched stops the sound with a message (onEnd is called), asks for it again on "Try again" and on the next Play, and ahead-of-time failures do not stop the sound that has it', async () => {
  let broken = true;
  const base = fakeSource(20, { channels: 1 });
  const src = { ...base, read: async (first, count, signal) => { if (broken && first >= 5 * SR) throw new Error('Couldn’t get the sound to try other strengths with. Please try again.'); return base.read(first, count, signal); } };
  let ended = 0;
  const p = player(src, { onEnd: () => { ended++; } });
  await p.play();
  await settle();
  assert.equal(p.error, '', 'stretches 0 to 2 came; 3 and later are not needed yet');
  for (const t of [1.1, 2.1, 3.1, 4.1]) { p.ctx.currentTime = t; p.tick(); await settle(); } // a second at a time; stretch 5 fails to come, ahead of the sound
  assert.equal(p.playing, true, 'a stretch that failed ahead of time does not stop the sound that has it');
  assert.equal(p.held.get(5).state, 'failed');
  p.ctx.currentTime = 5.06; p.tick(); await settle(); // and now it is needed
  assert.equal(p.playing, false);
  assert.match(p.error, /Couldn’t get the sound/);
  assert.equal(ended, 1);
  broken = false;
  await p.retry();
  assert.equal(p.error, '');
  assert.equal(p.held.get(Math.floor(p.offset)).state, 'ready');
  await p.play();
  await settle();
  assert.equal(p.playing, true);
  p.close();
});

await check('BlendPlayer: gains are set on the two inputs (a ramp, no click) and keep their value across stretches; a recording whose sides are the same plays as one channel; a silent side is zeros', async () => {
  const same = player(fakeSource(10, { channels: 1, plan: [0, 0] }));
  await same.play(); await settle();
  assert.equal(same.ctx.nodes[0].buffer.numberOfChannels, 1);
  same.setGains(0.9369, 0.0631);
  assert.deepEqual([same.gains.full.gain.value, same.gains.input.gain.value], [0.9369, 0.0631]);
  same.close();
  const two = player(fakeSource(10, { channels: 2, plan: [1, 0] }));
  await two.play(); await settle();
  const buf = two.ctx.nodes[0].buffer;
  assert.equal(buf.numberOfChannels, 2);
  assert.equal(buf.data[0][0], f32(0.5 / 1e7), 'the left side is the second kept channel');
  assert.equal(buf.data[1][0], 0);
  two.close();
  const silent = player(fakeSource(10, { channels: 2, plan: [0, null] }));
  await silent.play(); await settle();
  assert.equal(silent.ctx.nodes[0].buffer.data[1][5], 0);
  assert.equal(silent.ctx.nodes[0].buffer.data[0][5], f32(5 / 1e7));
  silent.close();
});

await check('BlendPlayer: a recording shorter than a stretch, and a last stretch cut short by the end, play whole (their buffers are exactly as long as the sound)', async () => {
  const p = player(fakeSource(2.5, { channels: 1 }));
  assert.equal(p.chunkCount, 3);
  await p.play(); await settle();
  const lens = [...new Set(p.ctx.nodes.map((n) => n.buffer.length))].sort((a, b) => a - b);
  assert.deepEqual(lens, [SR / 2, SR]);
  assert.equal(p.duration, 2.5);
  p.close();
  const short = new BlendPlayer(fakeSource(4), { tickMs: 0, context: () => new FakeCtx() }); // default 10 s stretches
  assert.equal(short.chunkCount, 1);
  assert.equal(short.chunkSeconds, 10);
  short.close();
  const capped = new BlendPlayer({ ...fakeSource(100), maxSeconds: 5 }, { tickMs: 0, context: () => new FakeCtx() }); // the server gives at most 5 s
  assert.equal(capped.chunkSeconds, 5);
  capped.close();
});

await check('memorySource: hands out the same stretch of both signals from arrays already here (views, no copy), energy sums from the whole', async () => {
  const { full, input } = pair(2, 5000);
  const src = memorySource({ full, input, plan: [0, 1] });
  assert.equal(src.samples, 5000);
  assert.equal(src.maxSeconds, Infinity);
  const got = await src.read(1000, 800);
  assert.equal(got.full.length, 2);
  assert.equal(got.full[1].length, 800);
  assert.equal(got.input[0][0], input[0][1000]);
  assert.equal(got.full[1][799], full[1][1799]);
  assert.equal((await src.read(4900, 800)).full[0].length, 100, 'cut short by the end');
  const s = src.sums();
  assert.deepEqual(s, energySums(full, input));
  assert.equal(src.sums(), s, 'worked out once');
});

// keptSource and the URLs: fetch and decodeAudioData are replaced by fakes that record what was asked
const urls = [];
globalThis.window = globalThis;
globalThis.OfflineAudioContext = class { decodeAudioData(bytes) { const n = new Float32Array(bytes).length; return Promise.resolve({ numberOfChannels: 1, getChannelData: () => Float32Array.from({ length: n }, (_, i) => (i + 1) / 1000) }); } };
let answer = null;
globalThis.fetch = async (url) => {
  urls.push(url);
  const q = new URL(url, 'http://x');
  const first = Math.round(Number(q.searchParams.get('start')) * SR);
  const count = Math.round(Number(q.searchParams.get('seconds')) * SR);
  const headers = new Map(answer?.headers ?? [['X-Stretch-Start', String(first)], ['X-Stretch-Samples', String(count)]]);
  return { ok: answer?.ok ?? true, headers: { get: (k) => headers.get(k) ?? null, has: (k) => headers.has(k) }, arrayBuffer: async () => new Float32Array(count).buffer };
};

await check('keptUrl: the whole copy has no query; a stretch names its first sample and length with six decimals that give the exact samples back (checked on 2000 positions)', () => {
  assert.equal(keptUrl('abc', 'noise'), '/api/jobs/abc/kept/noise');
  assert.equal(keptUrl('abc', 'input', { first: 480000, count: 480000 }), '/api/jobs/abc/kept/input?start=10.000000&seconds=10.000000');
  let seed2 = 11;
  for (let i = 0; i < 2000; i++) {
    seed2 = (seed2 * 1103515245 + 12345) & 0x7fffffff;
    const first = seed2 % (4 * 3600 * SR);
    const count = 1 + (seed2 >> 3) % (30 * SR);
    const q = new URL(keptUrl('j', 'noise', { first, count }), 'http://x').searchParams;
    assert.equal(Math.round(Number(q.get('start')) * SR), first);
    assert.equal(Math.round(Number(q.get('seconds')) * SR), count);
  }
});

await check('keptSource: asks for the same stretch of both copies, scales a loud one back up, numbers the channels from the first kept one and passes on the energy sums; a wrong stretch or a refusal is an error', async () => {
  const preview = { samples: 100 * SR, channels: [1], plan: [1, 1], scale: { noise: 0.5, input: 1 }, bytes: {}, stretchMaxSeconds: 30, energy: { full: 4, input: 5, cross: 3 } };
  const src = keptSource('J1', preview);
  assert.deepEqual(src.plan, [0, 0]);
  assert.equal(src.maxSeconds, 30);
  assert.deepEqual(src.sums(), { ef: 4, ei: 5, cfi: 3 });
  const got = await src.read(10 * SR, 4);
  assert.deepEqual(urls, ['/api/jobs/J1/kept/noise?start=10.000000&seconds=0.000083', '/api/jobs/J1/kept/input?start=10.000000&seconds=0.000083']);
  assert.equal(got.full.length, 1);
  assert.deepEqual(Array.from(got.full[0]), [1, 2, 3, 4].map((v) => f32(f32(v / 1000) * 2)), 'the loud copy is brought back up by its scale (1 / 0.5)');
  assert.deepEqual(Array.from(got.input[0]), [1, 2, 3, 4].map((v) => f32(v / 1000)), 'the other is as it is');
  assert.equal(keptSource('J1', { ...preview, energy: undefined }).sums(), null, 'no sums known');
  assert.equal(keptSource('J1', { ...preview, energy: { full: 1, input: 'x', cross: 0 } }).sums(), null);
  answer = { headers: [['X-Stretch-Start', '5'], ['X-Stretch-Samples', '4']] };
  await assert.rejects(src.read(10 * SR, 4), /wrong part/);
  answer = { ok: false };
  await assert.rejects(src.read(10 * SR, 4), /Couldn’t get the sound/);
  answer = null;
});

await check('previewFits: the whole recording is only ever held for 20 minutes of mono or 10 of stereo (the free run here, and saving a strength on this device)', () => {
  assert.ok(previewFits(1200, 1) && !previewFits(1200.5, 1));
  assert.ok(previewFits(600, 2) && !previewFits(601, 2));
  assert.ok(previewFits(600) && !previewFits(601));
});

console.log(failed ? `${failed} FAILED` : 'all passed');
process.exit(failed ? 1 : 0);
