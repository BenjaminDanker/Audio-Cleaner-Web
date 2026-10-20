// The editor's logic with no browser and no backend: the timeline (src/editor/timeline.js: the same rules as the server's), the gains the
// mix plays (src/editor/mix.js: strength as a blend, layer switches, equal-power crossfades at the edges of rebuilt parts, the closest
// version that exists, the stretches MixPlayer fetches and the automation it schedules, checked on a stand-in for the Web Audio
// context), the browser's suggestions (src/media/suggest-core.js: change points on audio with two parts, cancel, progress), the words of
// "Which one should I pick?" and of the suggestions. The suggestions against the server's: tests/suggest-agreement.mjs.
// Bundled with esbuild (the site's own build tool) so that `import.meta.env` in src/api.js is defined.  Usage: node tests/editor-unit.mjs
import assert from 'node:assert/strict';
import { dirname } from 'node:path';
import { fileURLToPath } from 'node:url';
import { build } from 'esbuild';

const HERE = dirname(fileURLToPath(import.meta.url));
const bundle = await build({
  stdin: {
    contents: `
      export * from '../src/editor/timeline.js';
      export * from '../src/editor/mix.js';
      export { sayAsk, sayWhat, suggestionOps, keyOf } from '../src/editor/suggest.js';
      export * from '../src/media/suggest-core.js';
      export { quoteDetails } from '../src/components/editor/FixAMoment.jsx';
      export { DENOISING_POINTS, REBUILD_POINTS } from '../src/components/WhichOne.jsx';
      export { recommendForFile } from '../src/recommend.js';
    `,
    resolveDir: HERE,
    loader: 'js',
  },
  bundle: true,
  format: 'esm',
  platform: 'node',
  write: false,
  logLevel: 'silent',
  jsx: 'automatic',
  loader: { '.jsx': 'jsx' },
  define: { 'import.meta.env.BASE_URL': '"/"', __DFN_VERSION__: '""', __DFN_JS_SIZE__: '0', __MODEL_HOST__: '""', __LIBAV_DIR__: '""', __ORT_DIR__: '""' },
});
const m = await import(`data:text/javascript;base64,${Buffer.from(bundle.outputFiles[0].text).toString('base64')}`);
const {
  Timeline, replay, restoreOps, differences, partLabel, levelOf, mergeRanges, subtract, intersect, rangeText, TimelineError,
  prepare, weightsAt, envelope, breakpoints, describe, needsProcessing, levelFactor, MixPlayer, HEAR_ALL, FADE_REBUILD,
  sayAsk, sayWhat, suggestionOps, quoteDetails, DENOISING_POINTS, REBUILD_POINTS,
  recommendPart, reasonsFor, clippedBursts, measureWindows, findSuggestions, recommendForFile,
} = m;

let failed = 0;
async function check(name, fn) {
  try {
    const detail = await fn();
    console.log(`PASS ${name}${detail ? ` -- ${detail}` : ''}`);
  } catch (e) {
    failed++;
    console.log(`FAIL ${name} -- ${String(e.stack || e.message).split('\n').slice(0, 5).join(' / ')}`);
  }
}
const near = (a, b, tol = 1e-6, what = '') => assert.ok(Math.abs(a - b) <= tol, `${what} ${a} vs ${b} (tolerance ${tol})`);

/* ------------------------------------------------------------------------------------------------ the timeline */

await check('timeline: a part painted over others replaces them, neighbours with the same settings join, times are clamped to the millisecond', () => {
  const t = new Timeline(60);
  t.paint('noise', 0, 60, { strength_db: 40, remover: 'auto' });
  t.paint('noise', 10, 20, { strength_db: 12, remover: 'auto' });
  t.paint('noise', 15, 25, { strength_db: 24, remover: 'auto' });
  assert.deepEqual(t.parts('noise').map((p) => [p.start, p.end, p.settings.strength_db]), [[0, 10, 40], [10, 15, 12], [15, 25, 24], [25, 60, 40]]);
  t.paint('noise', 15, 25, { strength_db: 40, remover: 'auto' }); // the same as its neighbours again: one part
  t.paint('noise', 10, 15, { strength_db: 40, remover: 'auto' });
  assert.deepEqual(t.parts('noise').map((p) => [p.start, p.end]), [[0, 60]]);
  t.paint('rebuild', -5, 1000);
  assert.deepEqual(t.parts('rebuild').map((p) => [p.start, p.end]), [[0, 60]]);
  t.paint('rebuild', 10.0004, 10.0006); // less than a millisecond: nothing
  t.erase('rebuild', 20, 30);
  assert.deepEqual(t.parts('rebuild').map((p) => [p.start, p.end]), [[0, 20], [30, 60]]);
  t.paint('rebuild', 20.12345, 30);
  assert.deepEqual(t.parts('rebuild').map((p) => p.start), [0, 20.123]);
});

await check('timeline: the JSON is the server\'s (version 1, layers, settings with their defaults), reading it back is the same timeline, bad input is refused in plain words', () => {
  const t = Timeline.wholeFile(123.4, { echo: true, noise: { strength_db: 40, remover: 'heavy' }, rebuild: false, leveling: { loudness_lufs: -14 } });
  t.paint('rebuild', 30.5, 60);
  const j = t.toJSON();
  assert.deepEqual(j, { version: 1, seconds: 123.4, layers: {
    echo: [{ start: 0, end: 123.4 }],
    noise: [{ start: 0, end: 123.4, strength_db: 40, remover: 'heavy' }],
    rebuild: [{ start: 30.5, end: 60 }],
    leveling: [{ start: 0, end: 123.4, loudness_lufs: -14 }],
  } });
  assert.ok(Timeline.fromJSON(j).equals(t));
  assert.ok(Timeline.fromJSON({ version: 1, seconds: 10, layers: { noise: [{ start: 0, end: 5, strength: 'gentle' }, { start: 3, end: 8, strength_db: 24 }] } }).parts('noise').length === 2); // (the later one wins)
  for (const bad of [{ strength_db: 0 }, { strength_db: 101 }, { remover: 'fast' }, { strength: 'loud' }]) assert.throws(() => new Timeline(10).paint('noise', 0, 5, bad), TimelineError, JSON.stringify(bad));
  assert.throws(() => new Timeline(10).paint('leveling', 0, 5, { loudness_lufs: 5 }), TimelineError);
  assert.throws(() => new Timeline(10).paint('mystery', 0, 5), TimelineError);
  assert.throws(() => Timeline.fromJSON({ version: 2, seconds: 1, layers: {} }), TimelineError);
  assert.throws(() => new Timeline(10).paint('noise', 5, 2), TimelineError);
});

await check('timeline: the changes are a list over a base (replay), any of them can be taken back, and a part goes back to what it was (restoreOps)', () => {
  const base = Timeline.wholeFile(60, { noise: { strength_db: 24, remover: 'auto' }, leveling: true });
  const ops = [
    { layer: 'noise', start: 10, end: 20, settings: { strength_db: 100, remover: 'auto' } },
    { layer: 'rebuild', start: 15, end: 25, settings: {}, tag: 'x' },
    { layer: 'leveling', start: 12, end: 18, settings: null },
  ];
  const t = replay(base, ops);
  assert.ok(!t.equals(base));
  assert.deepEqual(t.parts('leveling').map((p) => [p.start, p.end]), [[0, 12], [18, 60]]);
  assert.ok(replay(base, ops.filter((o) => o.tag !== 'x')).parts('rebuild').length === 0);
  const back = replay(base, [...ops, ...restoreOps(base, 10, 25)]);
  assert.ok(back.equals(base), JSON.stringify(back.toJSON()));
  assert.deepEqual(differences(t, base), [[10, 25]]);
  assert.deepEqual(differences(base, base), []);
  assert.deepEqual(mergeRanges([[5, 8], [0, 2], [1, 3]]), [[0, 3], [5, 8]]);
  assert.deepEqual(subtract([[0, 10]], [[2, 3], [5, 20]]), [[0, 2], [3, 5]]);
  assert.deepEqual(intersect([[0, 10]], [[5, 20]]), [[5, 10]]);
});

await check('timeline: names for the chips ("Gentle · 0:40–0:45", "whole file") and the strength levels', () => {
  const t = new Timeline(180);
  t.paint('noise', 40, 45, { strength_db: 12, remover: 'auto' });
  t.paint('noise', 0, 40, { strength_db: 100, remover: 'auto' });
  t.paint('rebuild', 0, 180);
  assert.equal(partLabel('noise', t.parts('noise')[1], 180), 'Gentle · 0:40–0:45');
  assert.equal(partLabel('noise', t.parts('noise')[0], 180), 'Maximum · 0:00–0:40');
  assert.equal(partLabel('rebuild', t.parts('rebuild')[0], 180), 'whole file');
  assert.deepEqual([12, 24, 40, 100, 33, 0].map(levelOf), [1, 2, 3, 4, 3, 0]);
  assert.equal(rangeText(75, 3725), '1:15–1:02:05');
});

/* ------------------------------------------------------------------------------------------------ the gains */

const whole = (extra = {}) => {
  const t = Timeline.wholeFile(60, { noise: { strength_db: 12, remover: 'light' }, leveling: true, ...extra });
  return t;
};
const ALL = { noise: [[0, 60]], rebuild: [[0, 60]], echo: [[0, 60]] };
const sum2 = (w) => w.source ** 2 + w.echo ** 2 + w.noise ** 2 + w.rebuild ** 2;

await check('mix: strength is a blend: (1 - a) of the noise removal and a of the recording, a = 10^(-dB/20), for each of the four names', () => {
  for (const [db, a] of [[12, 10 ** (-12 / 20)], [24, 10 ** (-24 / 20)], [40, 0.01], [100, 0]]) {
    const t = Timeline.wholeFile(60, { noise: { strength_db: db, remover: 'auto' }, leveling: false });
    const w = weightsAt(prepare({ tl: t, cached: ALL }), 30);
    near(w.noise, 1 - a, 1e-9, `noise at ${db} dB`);
    near(w.source, a, 1e-9, `source at ${db} dB`);
    assert.equal(w.rebuild, 0);
  }
});

await check('mix: a layer switched off is not heard (noise removal off: the recording alone), and nothing is heard of a layer with no part there', () => {
  const t = whole();
  const off = weightsAt(prepare({ tl: t, hear: { ...HEAR_ALL, noise: false }, cached: ALL }), 30);
  assert.deepEqual([off.source, off.noise, off.rebuild, off.echo], [1, 0, 0, 0]);
  const none = new Timeline(60);
  const w = weightsAt(prepare({ tl: none, cached: ALL }), 10);
  assert.equal(w.source, 1);
  assert.equal(w.noise + w.rebuild + w.echo, 0);
  // Leveling is not a stretch: it is a gain on all of them, and its switch decides whether it applies
  assert.equal(describe(prepare({ tl: t, cached: ALL }), 30, 2).level, 2);
  assert.equal(describe(prepare({ tl: t, hear: { ...HEAR_ALL, leveling: false }, cached: ALL }), 30, 2).level, 1);
});

await check('mix: the echo-free voice is the dry sound where echo removal covers the moment (and only if the file has a second track and the switch is on)', () => {
  const t = whole();
  t.paint('echo', 20, 40);
  const prep = prepare({ tl: t, cached: ALL, secondTrack: true });
  const inside = weightsAt(prep, 30);
  const outside = weightsAt(prep, 50);
  near(inside.echo, 10 ** (-12 / 20), 1e-9);
  assert.equal(inside.source, 0);
  assert.equal(outside.echo, 0);
  assert.equal(weightsAt(prepare({ tl: t, cached: ALL, secondTrack: false }), 30).echo, 0);
  assert.equal(weightsAt(prepare({ tl: t, hear: { ...HEAR_ALL, echo: false }, cached: ALL, secondTrack: true }), 30).echo, 0);
});

await check('mix: Voice rebuild wins where both are on, and its edges are an equal-power crossfade of about 250 ms (sin and cos: the powers add to one)', () => {
  const t = whole();
  t.paint('rebuild', 20, 40);
  const prep = prepare({ tl: t, cached: ALL });
  const mid = weightsAt(prep, 30);
  assert.equal(mid.rebuild, 1);
  near(mid.noise, 0, 1e-12);
  near(mid.source, 0, 1e-12);
  const out = weightsAt(prep, 10);
  assert.equal(out.rebuild, 0);
  near(out.noise, 1 - 10 ** (-12 / 20), 1e-9);
  const at = weightsAt(prep, 20); // exactly on the edge: half and half in power
  near(at.rebuild ** 2, 0.5, 1e-9);
  let worst = 0;
  for (let k = -12; k <= 12; k++) {
    const x = 20 + (k / 12) * FADE_REBUILD * 1.0;
    const w = weightsAt(prep, x);
    const rest = 1 - 10 ** (-12 / 20);
    // the rest (noise removal + the little dry sound the strength mixes back) goes down as cos, the rebuilt voice up as sin
    const cos2 = (w.noise / rest) ** 2;
    worst = Math.max(worst, Math.abs(cos2 + w.rebuild ** 2 - 1));
  }
  assert.ok(worst < 1e-9, `powers add to one: ${worst}`);
  near(FADE_REBUILD * 2, 0.25, 1e-9);
  // the same crossfade at the far edge, and none before 19.8 s or after 40.2 s
  assert.equal(weightsAt(prep, 19.8).rebuild, 0);
  assert.equal(weightsAt(prep, 40.2).rebuild, 0);
  assert.ok(weightsAt(prep, 40).rebuild > 0.7 && weightsAt(prep, 40).rebuild < 0.71);
  // at the start and the end of the file there is no edge to fade
  const all = Timeline.wholeFile(60, { noise: false, rebuild: true, leveling: false });
  assert.equal(weightsAt(prepare({ tl: all, cached: ALL }), 0).rebuild, 1);
  assert.equal(weightsAt(prepare({ tl: all, cached: ALL }), 59.99).rebuild, 1);
});

await check('mix: what is not made yet is skipped and the closest version that exists is heard (rebuild -> noise removal -> the recording), and the part is marked as needing processing', () => {
  const t = whole();
  t.paint('rebuild', 20, 40);
  const noRebuild = prepare({ tl: t, cached: { noise: [[0, 60]], rebuild: [] } });
  const w = weightsAt(noRebuild, 30);
  assert.equal(w.rebuild, 0);
  near(w.noise, 1 - 10 ** (-12 / 20), 1e-9);
  const nothing = weightsAt(prepare({ tl: t, cached: {} }), 30);
  assert.deepEqual([nothing.source, nothing.noise, nothing.rebuild], [1, 0, 0]);
  // half of the rebuilt part exists: the voice is rebuilt there and not after
  const half = prepare({ tl: t, cached: { noise: [[0, 60]], rebuild: [[0, 30]] } });
  assert.ok(weightsAt(half, 25).rebuild === 1 && weightsAt(half, 35).rebuild === 0);
  const need = needsProcessing(t, { noise: [[0, 60]], rebuild: [[0, 30]] }, false);
  assert.deepEqual(need.map((n) => [n.layer, n.start, n.end]), [['rebuild', 30, 40]]);
  assert.deepEqual(needsProcessing(whole(), ALL, false), []);
  assert.deepEqual(needsProcessing(whole({ echo: true }), { noise: [[0, 60]] }, false), []); // (no second track: no echo removal to make)
  // a stretch that came back with silent samples (X-Stretch-Missing) counts as missing there
  const miss = prepare({ tl: t, cached: ALL, missing: { rebuild: [[30, 35]] } });
  assert.ok(weightsAt(miss, 32).rebuild < 0.01 && weightsAt(miss, 25).rebuild === 1);
});

await check('mix: breakpoints and envelope: every gain at every breakpoint of a chunk, in time order, with the crossfade steps for a rebuilt part; Leveling\'s match is a clamped gain', () => {
  const t = whole();
  t.paint('rebuild', 4, 8);
  const prep = prepare({ tl: t, cached: ALL });
  const env = envelope(prep, 0, 10);
  assert.ok(env.times[0] === 0 && env.times[env.times.length - 1] === 10);
  assert.ok(env.times.every((x, i) => i === 0 || x > env.times[i - 1]));
  for (const k of ['source', 'echo', 'noise', 'rebuild', 'level']) assert.equal(env[k].length, env.times.length, k);
  assert.ok(env.times.filter((x) => x > 3.8 && x < 4.2).length >= 8, 'the crossfade steps');
  assert.ok(Math.max(...env.rebuild) === 1 && Math.min(...env.rebuild) === 0);
  assert.equal(levelFactor(-30, -16), 4); // (clamped at +12 dB)
  assert.equal(levelFactor(-10, -30), 0.25);
  near(levelFactor(-20, -14), 10 ** (6 / 20), 1e-9);
  assert.equal(levelFactor(-20, null), 1);
});

/** A stand-in for the Web Audio context: records what is scheduled on every gain. */
function fakeContext() {
  const made = { gains: [], sources: [] };
  const param = () => ({ value: 0, events: [], setValueAtTime(v, t) { this.events.push(['set', v, t]); }, linearRampToValueAtTime(v, t) { this.events.push(['ramp', v, t]); }, cancelScheduledValues() { this.events = []; }, setTargetAtTime() {} });
  const ctx = {
    currentTime: 0, destination: {}, async resume() {}, close: async () => {},
    createGain() { const g = { gain: param(), connect() {}, disconnect() {} }; made.gains.push(g); return g; },
    createDynamicsCompressor() { return { threshold: param(), knee: param(), ratio: param(), attack: param(), release: param(), connect() {}, disconnect() {} }; },
    createBuffer(ch, n) { return { numberOfChannels: ch, length: n, duration: n / 48000, copyToChannel() {} }; },
    createBufferSource() { const s = { buffer: null, started: null, connect() {}, disconnect() {}, start(at, off) { this.started = [at, off]; }, stop() {} }; made.sources.push(s); return s; },
  };
  return { ctx, made };
}

await check('mix: MixPlayer fetches only the layers a chunk needs (the recording, the noise removal where a part uses it, the rebuilt voice only where one exists), schedules the gains of every stretch, and follows a change at once', async () => {
  const { ctx, made } = fakeContext();
  const asked = [];
  const fetcher = async (job, layer, start, seconds) => { asked.push(`${layer}@${start}`); return { channels: [new Float32Array(Math.round(seconds * 48000))], missing: [], bytes: 1 }; };
  const t = whole();
  t.paint('rebuild', 12, 18);
  const model = { tl: t, hear: { ...HEAR_ALL }, cached: { noise: [[0, 60]], rebuild: [[0, 60]] }, secondTrack: false };
  const player = new MixPlayer({ jobId: 'j', preview: { samples: 60 * 48000, channels: 1 }, getModel: () => model, context: () => ctx, tickMs: 0, fetcher });
  await player.prepare();
  assert.deepEqual(asked.filter((a) => a.endsWith('@0')).sort(), ['noise@0', 'source@0']); // chunk 0 (0-10 s): no rebuilt voice there
  assert.ok(asked.includes('rebuild@10') && asked.includes('noise@10') && asked.includes('source@10'), asked.join(' '));
  await player.play();
  await new Promise((r) => setTimeout(r, 50));
  assert.ok(made.sources.length >= 5, `${made.sources.length} sources`);
  // the gains of the chunk with the rebuilt part: the rebuilt voice ramps 0 -> 1 -> 0 around 12 s and 18 s (with the 0.05 s head start of the context)
  const rebuildGain = () => player.held.get(1).gain.rebuild.gain.events; // chunk 1 (10-20 s) holds the rebuilt part
  const values = rebuildGain().map((e) => e[1]);
  assert.ok(values.length > 8 && Math.max(...values) === 1 && Math.min(...values) === 0, values.join(' '));
  assert.ok(values.some((v) => Math.abs(v - Math.SQRT1_2) < 0.03), 'the edge is an equal-power crossfade (sin and cos pass the half-power point)');
  // the noise removal goes down as the rebuilt voice comes up (cos)
  const noiseValues = player.held.get(1).gain.noise.gain.events.map((e) => e[1]);
  assert.ok(Math.min(...noiseValues) < 0.01 && Math.max(...noiseValues) > 0.7);
  // Voice rebuild switched off: the next remix gives the rebuilt voice no gain at all
  model.hear = { ...HEAR_ALL, rebuild: false };
  player.remix();
  assert.ok(rebuildGain().every((e) => e[1] === 0), 'the rebuilt voice is silent when its switch is off');
  assert.ok(player.held.get(1).gain.noise.gain.events.every((e) => e[1] > 0.7), 'and the noise removal plays through the part');
  const g = player.gainsNow();
  assert.equal(typeof g.noise, 'number');
  // a part painted over a chunk that is held (a layer it needs is fetched then)
  const before = asked.length;
  model.tl = (() => { const x = whole(); x.paint('rebuild', 40, 50); return x; })();
  model.hear = { ...HEAR_ALL };
  player.remix();
  await new Promise((r) => setTimeout(r, 30));
  player.close();
  return `${asked.length} stretches asked for, ${made.gains.length} gains`;
});

/* ------------------------------------------------------------------------------------------------ the words */

await check('words: a suggestion says what it heard and asks the question that goes with it; its operations are the server\'s `apply`', () => {
  const rebuild = { start: 41, end: 44, kind: 'clipping', rebuild: true, noise: null, reasons: ['sounds distorted here (clipped)'] };
  const noisy = { start: 68, end: 90, kind: 'part', rebuild: false, noise: { strength_db: 40, remover: 'heavy' }, reasons: ['is noisier', 'needs the stronger noise remover'] };
  assert.equal(sayWhat(rebuild), 'sounds distorted here (clipped).');
  assert.equal(sayWhat(noisy), 'is noisier and needs the stronger noise remover.');
  assert.deepEqual(sayAsk(rebuild), { ask: 'Rebuild just this part?', act: 'Rebuild this part', does: 'Voice rebuild' });
  assert.equal(sayAsk(noisy).act, 'Use Strong here');
  const base = Timeline.wholeFile(120, { noise: { strength_db: 12, remover: 'light' }, rebuild: true, leveling: false });
  const t = replay(base, suggestionOps(noisy, 'k')); // (the part does not rebuild: Voice rebuild comes off there, its own noise removal goes on)
  assert.deepEqual(t.parts('rebuild').map((p) => [p.start, p.end]), [[0, 68], [90, 120]]);
  assert.deepEqual(t.partAt('noise', 70).settings, { strength_db: 40, remover: 'heavy' });
  const t2 = replay(Timeline.wholeFile(120, { noise: true, leveling: false }), suggestionOps(rebuild, 'r'));
  assert.deepEqual(t2.parts('rebuild').map((p) => [p.start, p.end]), [[41, 44]]);
});

await check('words: "Which one should I pick?" and the automatic reasons use the terms of the owner\'s comparison, and no model names, no "premium", never "polish"', () => {
  const all = [...DENOISING_POINTS, ...REBUILD_POINTS].join(' ');
  for (const need of [/exactly as recorded/, /never changes words/, /Laughs, shouts, whispers and singing stay as they were/, /any length/, /Costs less/, /game sounds/, /gentler strengths keep some room sound/i,
    /from scratch/, /echo, a muffled or phone-quality voice and distortion/, /slightly too clean/, /keeps only speech/i, /music and room sound are all gone/, /come out oddly/, /can change a word, so listen before you publish/, /90 minutes/, /Costs more/, /many languages/]) {
    assert.match(all, need);
  }
  assert.doesNotMatch(all, /premium|polish|DeepFilter|Sidon|\bmodel\b|our servers?/i);
  // the automatic settings say which one they picked and why, in the same terms
  const grid = (o) => ({ seconds: 60, noiseDb: 10, floorDb: -50, levelDb: -24, peakDb: -12, silentFraction: 0.1, speechFraction: 0.5, bandDynDb: 12, hasSpeech: true, bandwidthHz: 20000, tiltDb: -5, dropDb: 20, clipFraction: 0, peak: 0.5, loudnessLufs: -22, ...o });
  const echoey = recommendForFile(grid({ dropDb: 5 }), {});
  assert.equal(echoey.rebuild, true);
  assert.equal(echoey.pick, 'Voice rebuild: the voice sounds echoey.');
  assert.equal(echoey.reasons[0], echoey.pick);
  const phone = recommendForFile(grid({ bandwidthHz: 4000, clipFraction: 0.01 }), {});
  assert.equal(phone.pick, 'Voice rebuild: it sounds like a phone call or a very small file and the sound distorts from being recorded too loud.');
  const clear = recommendForFile(grid({}), {});
  if (!clear.rebuild) assert.equal(clear.pick, 'Denoising: the voice is already clear, so it is kept exactly as recorded and only the background sound is taken out.');
  for (const r of [echoey, phone, clear]) assert.doesNotMatch(r.pick, /premium|polish|DeepFilter|Sidon|\bmodel\b|our servers?/i);
});

await check('quote details: per step of the quote, the parts that have to be made and the stretch each is made from, with what the step also listens to', () => {
  const t = whole();
  t.paint('noise', 20, 30, { strength_db: 40, remover: 'light' });
  t.paint('rebuild', 20, 30);
  const cached = { noise: [[0, 60]], rebuild: [] };
  const quote = { free: false, unchanged: false, lines: [{ item: 'rebuild', label: 'Voice rebuild', seconds: 10, micros: 5000 }, { item: 'leveling', label: 'Leveling', seconds: 49, micros: 2000 }], steps: { rebuild: { seconds: 10 }, leveling: { seconds: 49 } } };
  const rendered = whole();
  const d = quoteDetails(quote, t, cached, rendered, 60);
  assert.deepEqual(d[0].parts[0].core, [20, 30]);
  assert.deepEqual(d[0].parts[0].read, [6, 44]); // (a rebuild block is made from the stretch before it and the one after)
  assert.deepEqual(d[1].parts[0].core, [20, 30]); // Leveling is made again where the sound under it changed
  assert.deepEqual(quoteDetails({ ...quote, unchanged: true }, t, cached, rendered, 60), []);
});

/* ------------------------------------------------------------------------------------------------ the suggestions in the browser */

await check('suggestions: the decision of a part (reco_model.recommend: strength from the model, the stronger remover below 19 dB, every cause for a rebuild) and the reasons in plain words', () => {
  const stats = (o) => ({ seconds: 5, noiseDb: 25, floorDb: -60, levelDb: -24, peakDb: -12, silentFraction: 0.1, speechFraction: 0.5, bandDynDb: 12, hasSpeech: true, bandwidthHz: 20000, tiltDb: -5, dropDb: 20, clipFraction: 0, peak: 0.5, loudnessLufs: -22, ...o });
  assert.equal(recommendPart(stats({ noiseDb: 18.9 })).heavy, true);
  assert.equal(recommendPart(stats({ noiseDb: 19 })).heavy, false);
  assert.deepEqual(recommendPart(stats({ bandwidthHz: 6000, dropDb: 5 })).reasons, ['echoey', 'phone']);
  assert.deepEqual(recommendPart(stats({ tiltDb: -25 })).reasons, ['muffled']);
  assert.deepEqual(recommendPart(stats({ tiltDb: -25, bandwidthHz: 6000 })).reasons, ['phone']); // (never "muffled" as well as a phone call)
  assert.equal(recommendPart(stats({ hasSpeech: false })).noSpeech, true);
  assert.equal(recommendPart({ ...stats(), noiseDb: null }), null);
  const part = { rebuild: true, remover: 'light', strength: 'balanced', noiseDb: 5, levelDb: -40, reasons: ['phone', 'distorted'] };
  const file = { rebuild: false, remover: 'light', strength: 'balanced', noiseDb: 20, levelDb: -24, reasons: [] };
  assert.deepEqual(reasonsFor(part, file), ['sounds like a phone call', 'sounds distorted', 'is noisier', 'is quiet']);
  // the reason follows the proposed setting (the server's reasons_for): "is noisier" only for a stronger setting where the figure is lower
  assert.deepEqual(reasonsFor({ ...file, remover: 'heavy', strength: 'max', noiseDb: 20 }, file), ['needs the stronger noise remover', 'has noise that needs a stronger setting']);
  assert.deepEqual(reasonsFor({ ...file, strength: 'max', noiseDb: 10 }, file), ['is noisier']);
  assert.deepEqual(reasonsFor({ ...file, strength: 'gentle', noiseDb: 20 }, file), ['needs a gentler noise removal']);
  assert.deepEqual(reasonsFor({ ...file, strength: 'gentle', noiseDb: 25 }, file), ['is cleaner']);
  assert.deepEqual(reasonsFor({ ...file, noiseDb: 25 }, file), []); // (no standalone "is cleaner" / "is noisier")
  assert.deepEqual(reasonsFor({ ...file, noiseDb: 10 }, file), []);
});

await check('suggestions: clipped bursts are found in a speech-like signal (the pile-up at the top of the amplitude histogram) and not in the untouched one; the windows to measure are the server\'s', () => {
  const n = 48000 * 20;
  const x = new Float32Array(n);
  let s = 1;
  for (let i = 0; i < n; i++) { s = (s * 1103515245 + 12345) & 0x7fffffff; x[i] = 0.25 * Math.sin(i * 0.05) * Math.sin(i * 0.0007) + 0.01 * (s / 0x7fffffff - 0.5); }
  assert.deepEqual(clippedBursts(x), []);
  for (let i = 48000 * 7; i < 48000 * 7.5; i++) x[i] = Math.max(-0.4, Math.min(0.4, x[i] * 40));
  const bursts = clippedBursts(x);
  assert.ok(bursts.length >= 1 && bursts.every(([a, b]) => a >= 48000 * 6.9 && b <= 48000 * 7.6), JSON.stringify(bursts.slice(0, 3)));
  assert.deepEqual(measureWindows(60 * 48000), [[0, 2880000]]);
  assert.deepEqual(measureWindows(120 * 48000).map(([a]) => a / 48000), [5, 25, 45, 65, 85, 105]);
  assert.deepEqual(measureWindows(0), []);
});

await check('suggestions: audio with two different parts: the change is found (to a second), the parts are at least 20 s, progress goes up to one, the peaks of the waveform come out, and a cancel stops it', async () => {
  const sr = 48000;
  const n = 60 * sr;
  const x = new Float32Array(n);
  let seed = 5;
  const rnd = () => { seed = (seed * 1103515245 + 12345) & 0x7fffffff; return seed / 0x7fffffff - 0.5; };
  for (let i = 0; i < n; i++) {
    const t = i / sr;
    const syl = Math.abs(Math.sin(t * 3.1)) * Math.abs(Math.sin(t * 1.3 + 1));
    const voice = 0.2 * syl * (Math.sin(2 * Math.PI * 140 * t) + 0.5 * Math.sin(2 * Math.PI * 280 * t) + 0.3 * Math.sin(2 * Math.PI * 1500 * t));
    x[i] = t < 30 ? voice + 0.001 * rnd() : voice * 0.4 + 0.25 * rnd(); // the second half: quieter voice in loud broadband noise
  }
  // a stand-in for the noise remover: what is smooth is the voice, what is left over is the noise
  const noiseSums = async (mono) => {
    let voice = 0;
    let removed = 0;
    let lp = 0;
    for (let i = 0; i < mono.length; i++) { lp += 0.05 * (mono[i] - lp); voice += lp * lp; const d = mono[i] - lp; removed += d * d; }
    return { voice, removed };
  };
  const progress = [];
  const peaks = [];
  const out = await findSuggestions({ seconds: 60, read: async (a, b) => [x.slice(a, b)], noiseSums, onProgress: (f) => progress.push(f), onPeaks: (from, p) => peaks.push([from, p.length]), concurrency: 3 });
  assert.equal(out.seconds, 60);
  assert.deepEqual(out.parts.length, 2, JSON.stringify(out.parts.map((p) => [p.start, p.end])));
  assert.ok(Math.abs(out.parts[1].start - 30) <= 1.01, `cut at ${out.parts[1].start}`);
  assert.ok(out.parts.every((p) => p.end - p.start >= 20));
  assert.equal(progress[progress.length - 1], 1);
  assert.ok(Math.max(...progress) === 1 && progress.every((v) => v >= 0 && v <= 1));
  assert.ok(peaks.length >= 1 && peaks[0][0] === 0 && peaks[0][1] === 240); // a peak per quarter second of the first minute
  for (const s of out.suggestions) assert.ok(s.start < s.end && ['part', 'clipping'].includes(s.kind) && Array.isArray(s.reasons));
  const ctl = new AbortController();
  let reads = 0;
  await assert.rejects(findSuggestions({ seconds: 60, signal: ctl.signal, noiseSums, read: async (a, b) => { if (++reads === 2) ctl.abort(); return [x.slice(a, b)]; } }), (e) => e.name === 'AbortError');
  const short = await findSuggestions({ seconds: 20, read: async (a, b) => [x.slice(a, b)], noiseSums });
  assert.deepEqual(short.suggestions, []); // (nothing to compare it with: no change points under 40 s)
  return `cut at ${out.parts[1].start} s, ${out.suggestions.length} suggestions`;
});

process.exit(failed ? 1 : 0);
