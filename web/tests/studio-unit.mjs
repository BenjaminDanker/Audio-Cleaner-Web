// The editor's logic with no browser and no backend: the four tracks and their rules (src/editor/tracks.js: edit, split, auto-merge, the
// selection, prices from the price list, saving changes), the mapping to the server's timeline version 2 (src/editor/timeline.js), the
// waveform of what is heard (src/editor/peaks.js), what the choices mean for the files and what stops a start (src/studio/plan.js), the
// browser's share of a job (src/studio/runs.js) and the tooltips' words (src/studio/copy.js).
// Bundled with esbuild so that `import.meta.env` in src/api.js is defined.  Usage: node tests/studio-unit.mjs
import assert from 'node:assert/strict';
import { dirname } from 'node:path';
import { fileURLToPath } from 'node:url';
import { build } from 'esbuild';

const HERE = dirname(fileURLToPath(import.meta.url));
const bundle = await build({
  stdin: {
    contents: `
      export * from '../src/editor/timeline.js';
      export * from '../src/editor/tracks.js';
      export { heardSeries, layerPeaks, roughDb } from '../src/editor/peaks.js';
      export { prepare } from '../src/editor/mix.js';
      export { outputPlan, problemsFor, jobKind, wantsOf, browserVerdict, subsVerdict, inGigabytes } from '../src/studio/plan.js';
      export { fitPcm, samplesOf, piecesIn, rangeFor } from '../src/studio/runs.js';
      export { tipsFor, priceWords } from '../src/studio/copy.js';
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
  Timeline, TimelineError, makeState, edit, splitAll, setSel, toggleRange, selTitle, cost, delta, valSig, sameTracks, ratesFrom, money, toTimeline, fromTimeline,
  freshTracks, vkey, uni, overl, applyToSel, freePatch, echoBlocked, undo, redo, pushUndo, recoPatches, applyReco, toSuggestion, autoDone, needsServer, serverSound,
  changesSound, freeNoiseRanges, freeSubRanges, fmt, lenTxt, perMinute, heardSeries, layerPeaks, prepare, outputPlan, problemsFor, jobKind, wantsOf, subsVerdict,
  browserVerdict, inGigabytes, fitPcm, samplesOf, piecesIn, rangeFor, tipsFor, priceWords, dbOf, lvlOfLufs,
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
const near = (a, b, tol = 1e-6, what = '') => assert.ok(Math.abs(a - b) <= tol, `${what} ${a} vs ${b}`);

// the price list the way GET /api/billing/pricing says it (USD a minute)
const PRICING = {
  rebuildAvailable: true,
  rates: [
    { product: 'denoise', tier: 'standard', usdPerMinute: 0.009 }, { product: 'denoise', tier: 'rebuild', usdPerMinute: 0.03 },
    { product: 'echo', tier: 'standard', usdPerMinute: 0.0026 }, { product: 'leveling', tier: 'standard', usdPerMinute: 0.001 },
    { product: 'transcribe', tier: 'local', usdPerMinute: 0.004 }, { product: 'transcribe', tier: 'premium', usdPerMinute: 0.015 },
    { product: 'transcribe_translate', tier: 'standard', usdPerMinute: 0.004 },
  ],
};
const R = ratesFrom(PRICING);
const state = (T = 180, o = {}) => makeState(T, { second: false, auto: true, ...o });

/* ------------------------------------------------------------------------------------------------ timeline version 2 */

await check('timeline v2: the subtitles layer (quality, language, translations) and noise parts the browser cleaned; version 2 only when it is used', () => {
  const t = new Timeline(100);
  t.paint('noise', 0, 100, { strength_db: 24, remover: 'auto' });
  assert.equal(t.toJSON().version, 1); // nothing new: written as before
  t.paint('subtitles', 0, 50, { quality: 'standard', language: 'en', translate: ['es', 'ja'] });
  t.paint('subtitles', 50, 100, { quality: 'free', language: null });
  const j = t.toJSON();
  assert.equal(j.version, 2);
  assert.deepEqual(j.layers.subtitles, [
    { start: 0, end: 50, quality: 'standard', language: 'en', translate: ['es', 'ja'] },
    { start: 50, end: 100, quality: 'free', language: null, translate: [] },
  ]);
  const b = new Timeline(100);
  b.paint('noise', 10, 20, { browser: true, strength_db: 12 }); // (strength and remover are not kept for a browser part)
  assert.deepEqual(b.toJSON(), { version: 2, seconds: 100, layers: { noise: [{ start: 10, end: 20, browser: true }] } });
  assert.deepEqual(Timeline.fromJSON(j).toJSON(), j);
  assert.deepEqual(Timeline.fromJSON(b.toJSON()).toJSON(), b.toJSON());
});

await check('timeline v2: neighbours with the same subtitle settings are one part, different translations are two; bad subtitles are refused in plain words', () => {
  const t = new Timeline(60);
  t.paint('subtitles', 0, 30, { quality: 'standard', language: 'en', translate: ['es'] });
  t.paint('subtitles', 30, 60, { quality: 'standard', language: 'en', translate: ['es'] });
  assert.equal(t.parts('subtitles').length, 1);
  t.paint('subtitles', 30, 60, { quality: 'standard', language: 'en', translate: ['de'] });
  assert.equal(t.parts('subtitles').length, 2);
  assert.throws(() => t.paint('subtitles', 0, 10, { quality: 'premium' }), TimelineError);
  assert.throws(() => t.paint('subtitles', 0, 10, { quality: 'free', language: 'en', translate: ['es'] }), /free subtitles are made in the browser/);
  assert.throws(() => t.paint('subtitles', 0, 10, { quality: 'standard', language: 'en', translate: ['en'] }), /same as the spoken language/);
  assert.throws(() => t.paint('subtitles', 0, 10, { quality: 'standard', language: 'en', translate: ['a', 'b', 'c', 'd'] }), /at most 3/);
  assert.throws(() => Timeline.fromJSON({ version: 1, seconds: 10, layers: { subtitles: [{ start: 0, end: 5, quality: 'free' }] } }), /version 2/);
  assert.throws(() => Timeline.fromJSON({ version: 1, seconds: 10, layers: { noise: [{ start: 0, end: 5, browser: true }] } }), /version 2/);
  const c = t.copy();
  c.parts('subtitles')[0].settings.translate.push('fr');
  assert.deepEqual(t.parts('subtitles')[0].settings.translate, ['es']); // (a copy shares nothing)
});

/* ------------------------------------------------------------------------------------------------ the tracks */

await check('tracks: a change applies to the selection and cuts at its edges; neighbours that end up the same join by themselves', () => {
  const S = state();
  setSel(S, [{ s: 15, e: 25 }]);
  applyToSel(S, 'voice', { str: 3 });
  assert.deepEqual(S.tracks.voice.map((x) => [x.s, x.e, x.str]), [[0, 15, 2], [15, 25, 3], [25, 180, 2]]);
  assert.equal(S.tracks.voice[1].auto, false);
  assert.equal(S.tracks.voice[0].auto, true);
  applyToSel(S, 'voice', { str: 2 }); // the same as its neighbours again: one section
  assert.deepEqual(S.tracks.voice.map((x) => [x.s, x.e]), [[0, 180]]);
  assert.equal(selTitle(S), '0:15–0:25 · 10 s');
  setSel(S, [{ s: 0, e: 180 }]);
  assert.equal(S.sel, null); // the whole file is no selection: the default
  assert.equal(selTitle(S), 'Whole file · 3:00');
});

await check('tracks: a split stays until a change touches it; Select more adds and takes out ranges', () => {
  const S = state();
  assert.equal(splitAll(S, 60), true);
  assert.equal(S.tracks.voice.length, 2);
  assert.equal(S.tracks.echo.length, 2);
  assert.equal(S.tracks.voice[1].hard, true);
  assert.equal(splitAll(S, 0), false);
  assert.equal(splitAll(S, 180), false);
  setSel(S, [{ s: 0, e: 60 }]);
  applyToSel(S, 'voice', { str: 1 }); // a change at the split: the split is a plain edge now
  assert.deepEqual(S.tracks.voice.map((x) => [x.s, x.e, x.hard]), [[0, 60, false], [60, 180, false]]);
  applyToSel(S, 'voice', { str: 2 });
  assert.equal(S.tracks.voice.length, 1); // ... and joins again with its neighbour
  S.sel = null;
  toggleRange(S, 0, 60);
  toggleRange(S, 120, 180);
  assert.equal(selTitle(S), '2 ranges · 2:00 total');
  toggleRange(S, 10, 20); // inside a selected range: it comes out
  assert.deepEqual(S.sel, [{ s: 0, e: 10 }, { s: 20, e: 60 }, { s: 120, e: 180 }]);
});

await check('tracks: Mixed selections, uni(), undo and redo (a hundred steps), a suggestion marked done when its change is all there', () => {
  const S = state();
  setSel(S, [{ s: 0, e: 60 }]);
  pushUndo(S); applyToSel(S, 'voice', { str: 4 });
  setSel(S, [{ s: 0, e: 180 }]);
  assert.equal(uni(S, 'voice', 'str'), null);
  S.sel = [{ s: 0, e: 60 }];
  assert.equal(uni(S, 'voice', 'str'), 4);
  assert.equal(undo(S), true);
  assert.equal(S.tracks.voice.length, 1);
  assert.equal(redo(S), true);
  assert.equal(S.tracks.voice.length, 2);
  S.sugg = [{ id: 1, s: 10, e: 20, why: 'x', todo: 'Voice rebuild', patch: { mode: 2 }, st: 'new' }];
  edit(S, 'voice', 10, 20, { mode: 2 });
  autoDone(S);
  assert.equal(S.sugg[0].st, 'applied');
});

await check('tracks: Free is Gentle to Strong and has no choice of remover; echo can not be where the noise removal is Free', () => {
  const S = state(180, { second: true });
  assert.equal(S.tracks.echo[0].on, 1);
  assert.deepEqual(freePatch({ str: 4, db: null }), { kind: 'free', str: 3, db: null });
  assert.deepEqual(freePatch({ str: 4, db: 80 }), { kind: 'free', str: 3, db: 40 });
  setSel(S, [{ s: 30, e: 60 }]);
  assert.equal(echoBlocked(S), false);
  applyToSel(S, 'voice', freePatch);
  assert.equal(echoBlocked(S), true);
  S.sel = [{ s: 100, e: 120 }];
  assert.equal(echoBlocked(S), false);
});

/* ------------------------------------------------------------------------------------------------ prices */

await check('prices: from the price list, per second of audio, no minimum; the words are the prototype\'s ("Free", "<$0.001", "$0.024")', () => {
  assert.equal(R.noise, 150); // micros a second: 9000 a minute
  assert.equal(perMinute(R.noise), 9000);
  const S = state(600);
  let c = cost(S.tracks, R);
  assert.equal(c.nPaid, 600);
  assert.equal(c.total, 90000 + 10000); // noise removal 10 minutes + Leveling (YouTube)
  assert.equal(c.paidAny, true);
  edit(S, 'subs', 0, 600, { q: 2, lang: 'en', tr: ['es', 'ja'] });
  c = cost(S.tracks, R);
  assert.equal(c.trLangs.length, 2);
  assert.equal(c.total, 90000 + 10000 + 40000 + 2 * 40000);
  edit(S, 'subs', 0, 600, { q: 1 });
  c = cost(S.tracks, R);
  assert.equal(c.sFree, 600);
  assert.equal(c.trSec, 0); // (translations come with Standard or Extra accurate only)
  assert.equal(money(0), 'Free');
  assert.equal(money(400), '<$0.001');
  assert.equal(money(23700), '$0.024');
  assert.equal(money(1500000), '$1.50');
  assert.equal(ratesFrom(null), null);
  const none = cost(state().tracks, null);
  assert.equal(none.total, 0);
});

await check('prices: only what is new is charged for a finished recording (delta), and the signature tells equal tracks from different ones', () => {
  const S = state(180);
  const old = JSON.parse(JSON.stringify(S.tracks));
  assert.equal(sameTracks(S.tracks, old, 180), true);
  assert.equal(delta(S.tracks, old, 180, R), 0);
  edit(S, 'voice', 90, 180, { str: 4 }); // another strength: free
  assert.equal(sameTracks(S.tracks, old, 180), false);
  assert.equal(delta(S.tracks, old, 180, R), 0);
  edit(S, 'voice', 0, 60, { mode: 2 }); // Voice rebuild for a minute that had none
  assert.equal(delta(S.tracks, old, 180, R), 30000);
  edit(S, 'subs', 0, 30, { q: 3, lang: 'auto' });
  assert.equal(delta(S.tracks, old, 180, R), 30000 + 7500);
  edit(S, 'voice', 0, 60, { mode: 1 });
  edit(S, 'level', 0, 180, { lvl: 0 });
  assert.equal(delta(S.tracks, old, 180, R), 7500); // taking a step away never costs
});

await check('prices: a fractional length works (the last section ends at the recording\'s end)', () => {
  const S = state(47.3);
  edit(S, 'voice', 10, 47.3, { mode: 2 });
  near(cost(S.tracks, R).rSec, 37.3);
  assert.equal(S.tracks.voice[S.tracks.voice.length - 1].e, 47.3);
});

/* ------------------------------------------------------------------------------------------------ the mapping to the server's timeline */

await check('mapping: Standard noise removal -> noise {strength_db, remover}, Voice rebuild -> rebuild, Free -> a part the browser cleaned, Echo, Leveling, Subtitles', () => {
  const S = state(100, { second: true });
  edit(S, 'voice', 0, 20, { mode: 1, kind: 'paid', str: 1, rem: 0 });
  edit(S, 'voice', 20, 40, { str: 3, rem: 2 });
  edit(S, 'voice', 40, 60, { mode: 2 });
  edit(S, 'voice', 60, 80, { mode: 1, kind: 'free', str: 3, rem: 0 });
  edit(S, 'voice', 80, 100, { mode: 0 });
  edit(S, 'level', 0, 50, { lvl: 1 });
  edit(S, 'level', 50, 80, { lvl: 3 });
  edit(S, 'level', 80, 100, { lvl: 4 });
  edit(S, 'subs', 0, 100, { q: 2, lang: 'en', tr: ['es'] });
  edit(S, 'subs', 10, 20, { q: 3, lang: 'auto', tr: [] });
  edit(S, 'subs', 70, 90, { q: 1, lang: 'de' });
  const j = toTimeline(S.tracks, 100).toJSON();
  assert.equal(j.version, 2);
  assert.deepEqual(j.layers.noise, [
    { start: 0, end: 20, strength_db: 12, remover: 'auto' },
    { start: 20, end: 40, strength_db: 40, remover: 'heavy' },
    { start: 60, end: 80, browser: true },
  ]);
  assert.deepEqual(j.layers.rebuild, [{ start: 40, end: 60 }]);
  assert.deepEqual(j.layers.echo, [{ start: 0, end: 100 }]);
  assert.deepEqual(j.layers.leveling, [
    { start: 0, end: 50, loudness_lufs: -16 }, { start: 50, end: 80, loudness_lufs: -23 }, { start: 80, end: 100, loudness_lufs: null },
  ]);
  assert.deepEqual(j.layers.subtitles.map((p) => [p.start, p.end, p.quality, p.language, p.translate]), [
    [0, 10, 'standard', 'en', ['es']], [10, 20, 'accurate', null, []], [20, 70, 'standard', 'en', ['es']], [70, 90, 'free', 'de', []], [90, 100, 'standard', 'en', ['es']],
  ]);
  assert.deepEqual(Timeline.fromJSON(j).toJSON(), j);
});

await check('mapping: a custom strength from the slider is the noise part\'s strength_db; the last part ends at the recording\'s length; Echo off and Leveling Off make no part', () => {
  const S = state(47.3);
  S.tracks.voice[0].db = 30;
  S.tracks.level[0].lvl = 0;
  const j = toTimeline(S.tracks, 47.3).toJSON();
  assert.deepEqual(j.layers.noise, [{ start: 0, end: 47.3, strength_db: 30, remover: 'auto' }]);
  assert.equal(j.layers.echo, undefined);
  assert.equal(j.layers.leveling, undefined);
  assert.equal(j.version, 1);
});

await check('mapping: a timeline read back from a job becomes the tracks it was made from (named strengths, a strength of its own, Voice rebuild over noise, Keep, a loudness of its own)', () => {
  const t = new Timeline(100);
  t.paint('noise', 0, 100, { strength_db: 24, remover: 'light' });
  t.paint('noise', 10, 20, { strength_db: 30, remover: 'auto' });
  t.paint('rebuild', 40, 50);
  t.paint('noise', 60, 70, { browser: true });
  t.paint('leveling', 0, 50, { loudness_lufs: -14 });
  t.paint('leveling', 50, 80, { loudness_lufs: null });
  t.paint('leveling', 80, 100, { loudness_lufs: -20 });
  t.paint('subtitles', 0, 100, { quality: 'accurate', language: 'fr', translate: ['en'] });
  const tr = fromTimeline(t, 100);
  const v = tr.voice.map((x) => [x.s, x.e, x.mode, x.kind, x.str, x.db, x.rem]);
  assert.deepEqual(v, [[0, 10, 1, 'paid', 2, null, 1], [10, 20, 1, 'paid', 2, 30, 0], [20, 40, 1, 'paid', 2, null, 1], [40, 50, 2, 'paid', 2, null, 0],
    [50, 60, 1, 'paid', 2, null, 1], [60, 70, 1, 'free', 3, null, 0], [70, 100, 1, 'paid', 2, null, 1]]);
  assert.deepEqual(tr.level.map((x) => [x.s, x.e, x.lvl, x.lufs]), [[0, 50, 2, null], [50, 80, 4, null], [80, 100, 9, -20]]);
  assert.deepEqual(tr.subs.map((x) => [x.q, x.lang, x.tr]), [[3, 'fr', ['en']]]);
  assert.equal(lvlOfLufs(-16), 1);
  assert.deepEqual(toTimeline(tr, 100).toJSON().layers.leveling.map((p) => p.loudness_lufs), [-14, null, -20]);
  // a job made without a timeline has no subtitles layer: its own choice is the whole-file part
  const bare = fromTimeline(Timeline.wholeFile(60), 60, { subs: { quality: 'standard', language: 'en', translate: ['es'] } });
  assert.deepEqual(bare.subs.map((x) => [x.s, x.e, x.q, x.lang, x.tr]), [[0, 60, 2, 'en', ['es']]]);
});

await check('needs: what runs on our side, what changes the sound, and the browser\'s ranges and strengths', () => {
  const S = state(100);
  edit(S, 'voice', 0, 100, { kind: 'free', str: 3 });
  edit(S, 'level', 0, 100, { lvl: 0 });
  assert.equal(needsServer(S.tracks), false);
  assert.equal(jobKind(S.tracks), null);
  assert.equal(changesSound(S.tracks), true);
  assert.equal(serverSound(S.tracks), false);
  edit(S, 'subs', 0, 100, { q: 1 });
  assert.equal(needsServer(S.tracks), false);
  edit(S, 'voice', 40, 60, { str: 1 });
  assert.deepEqual(freeNoiseRanges(S.tracks), [{ start: 0, end: 40, db: 40 }, { start: 40, end: 60, db: 12 }, { start: 60, end: 100, db: 40 }]);
  assert.deepEqual(freeSubRanges(S.tracks), [{ start: 0, end: 100, language: '' }]);
  edit(S, 'subs', 0, 30, { q: 2 });
  assert.equal(needsServer(S.tracks), true);
  assert.equal(jobKind(S.tracks), 'denoise'); // a job made with parts: the subtitles are parts too
  assert.equal(serverSound(S.tracks), false);
  edit(S, 'level', 0, 10, { lvl: 2 });
  assert.equal(serverSound(S.tracks), true);
});

/* ------------------------------------------------------------------------------------------------ Automatic and suggestions */

await check('Automatic: the choices go on the parts that are still automatic and stay marked so; Use them again takes every part', () => {
  const S = state(180);
  edit(S, 'voice', 60, 90, { str: 4 }); // changed by hand
  const p = recoPatches({ strength: 'strong', rebuild: false, leveling: true, subs: 'accurate' });
  applyReco(S, p);
  assert.deepEqual(S.tracks.voice.map((x) => [x.s, x.e, x.str, x.auto]), [[0, 60, 3, true], [60, 90, 4, false], [90, 180, 3, true]]);
  assert.equal(S.tracks.subs[0].q, 3);
  applyReco(S, p, { all: true });
  assert.equal(S.tracks.voice.length, 1);
  assert.equal(S.tracks.voice[0].auto, true);
  assert.deepEqual(recoPatches({ strength: 'gentle', rebuild: true, leveling: false, subs: null }).voice, { mode: 2 });
  assert.equal(recoPatches({ strength: 'gentle', rebuild: true, leveling: false, subs: null }, { rebuildOk: false }).voice.mode, 1);
  assert.deepEqual(recoPatches({ strength: 'gentle', rebuild: false, leveling: false, subs: null }).level, { lvl: 0, lufs: null });
});

await check('suggestions: a server or browser suggestion becomes the prototype\'s (whole seconds, a plain why, what it would do), with the patch the Apply button makes', () => {
  const a = toSuggestion({ start: 41.3, end: 43.2, kind: 'part', rebuild: true, noise: null, reasons: ['sounds distorted'], key: 'part:413-432' }, 180, 7);
  assert.deepEqual([a.id, a.s, a.e, a.why, a.todo, a.st], [7, 41, 44, 'Sounds distorted', 'Voice rebuild', 'new']);
  assert.deepEqual(a.patch, { mode: 2 });
  const b = toSuggestion({ start: 68, end: 90, kind: 'part', rebuild: false, noise: { strength_db: 40, remover: 'heavy' }, reasons: ['noisier', 'quiet'] }, 180, 8);
  assert.equal(b.why, 'Noisier, quiet');
  assert.equal(b.todo, 'Strong');
  assert.deepEqual(b.patch, { mode: 1, str: 3, kind: 'paid', rem: 2, db: null });
  const c = toSuggestion({ start: 179.9, end: 180, reasons: [] }, 180, 9);
  assert.ok(c.s < c.e && c.e <= 180);
});

/* ------------------------------------------------------------------------------------------------ the waveform of what is heard */

await check('waveform: what is heard is each layer\'s peaks at the gains the mix plays; the speaker buttons change it; a layer not made yet falls back on the recording', () => {
  const T = 8;
  const mk = (v) => ({ rate: 4, a: Float32Array.from({ length: 32 }, () => v) });
  const layers = { source: mk(0.8), noise: mk(0.2), rebuild: null, echo: null };
  const tl = new Timeline(T);
  tl.paint('noise', 0, 8, { strength_db: 100, remover: 'auto' });
  const heard = (hear, cached = { noise: [[0, 8]] }) => heardSeries({ N: 16, T, prep: prepare({ tl, hear, cached, secondTrack: false }), layers, top: 0.8 });
  const all = { echo: true, noise: true, rebuild: true, leveling: true };
  near(heard(all)[3], 0.2 / 0.8, 1e-5); // Maximum: only the noise removal's sound
  near(heard({ ...all, noise: false })[3], 1, 1e-5); // the voice off: the recording
  near(heard(all, { noise: [] })[3], 1, 1e-5); // not made yet: the recording
  const half = new Timeline(T);
  half.paint('noise', 0, 8, { strength_db: 6, remover: 'auto' }); // 6 dB: about half the recording stays
  const h = heardSeries({ N: 16, T, prep: prepare({ tl: half, hear: all, cached: { noise: [[0, 8]] }, secondTrack: false }), layers, top: 0.8 });
  assert.ok(h[3] > 0.2 / 0.8 && h[3] < 1);
  const lp = layerPeaks({ rate: 4, seconds: 2, peaks: [0.1, null, 0.3] });
  assert.ok(Number.isNaN(lp.a[1]));
  assert.equal(layerPeaks(null), null);
});

await check('waveform: a part the browser cleaned is already at its strength: the noise removal is heard in full there', () => {
  const tl = new Timeline(8);
  tl.paint('noise', 0, 8, { browser: true });
  const prep = prepare({ tl, hear: { echo: true, noise: true, rebuild: true, leveling: true }, cached: { noise: [[0, 8]] }, secondTrack: false });
  const layers = { source: { rate: 4, a: new Float32Array(32).fill(0.8) }, noise: { rate: 4, a: new Float32Array(32).fill(0.1) } };
  near(heardSeries({ N: 8, T: 8, prep, layers, top: 0.8 })[2], 0.125, 1e-5);
});

/* ------------------------------------------------------------------------------------------------ the files and what stops a start */

const VIDEO = { hasVideo: true, hasAudio: true, viaMediabunny: true, duration: 180, audioTracks: 1, channels: 2 };
const AUDIO = { hasVideo: false, hasAudio: true, viaMediabunny: true, duration: 180, audioTracks: 1, channels: 1 };
const choiceOf = (format) => ({ format, name: `x.${format}`, isVideoFile: ['mp4', 'mkv', 'mov', 'webm'].includes(format), size: 'same', okIds: ['mp4', 'mkv', 'wav'] });
const out0 = { sfiles: ['srt'], burn: false };

await check('files: the subtitles go into the video (an MKV) only when there are subtitles and the result is a video; a video the browser can not open goes up whole and we put them in', () => {
  const S = state();
  edit(S, 'subs', 0, 180, { q: 2 });
  let p = outputPlan({ tracks: S.tracks, info: VIDEO, file: { size: 1e6 }, choice: choiceOf('mp4'), out: { ...out0, burn: true } });
  assert.deepEqual([p.cleaning, p.wantsText, p.serverText, p.videoResult, p.canEmbed, p.embedNow, p.embedHere, p.embedOnServer], [true, true, true, true, true, true, true, false]);
  p = outputPlan({ tracks: S.tracks, info: AUDIO, file: { size: 1e6 }, choice: choiceOf('wav'), out: out0 });
  assert.equal(p.videoResult, false);
  assert.equal(p.canEmbed, false);
  p = outputPlan({ tracks: S.tracks, info: { ...VIDEO, viaMediabunny: false }, file: { size: 1e6 }, choice: choiceOf('mp4'), out: { ...out0, burn: true } });
  assert.deepEqual([p.embedOnServer, p.embedHere, p.embedNow], [true, false, true]);
  p = outputPlan({ tracks: S.tracks, info: VIDEO, file: { size: 5e9 }, choice: choiceOf('mp4'), out: { ...out0, burn: true }, limit: 1.5 * 1024 ** 3 });
  assert.deepEqual([p.tooBigHere, p.canEmbed], [true, false]);
  // only subtitles: no cleaned sound, the video keeps its own type
  const T2 = state();
  edit(T2, 'voice', 0, 180, { mode: 0 }); edit(T2, 'level', 0, 180, { lvl: 0 }); edit(T2, 'subs', 0, 180, { q: 2 });
  p = outputPlan({ tracks: T2.tracks, info: VIDEO, file: { name: 'a.webm', size: 1e6 }, choice: choiceOf('mp4'), out: { ...out0, burn: true } });
  assert.deepEqual([p.cleaning, p.videoResult, p.embedNow], [false, true, true]);
});

await check('files: wantsOf is what the finishing step (finish.js) takes: the sound and the subtitles made here only for a run with nothing on our side', () => {
  const S = state();
  edit(S, 'voice', 0, 180, { kind: 'free', str: 3 }); edit(S, 'level', 0, 180, { lvl: 0 }); edit(S, 'subs', 0, 180, { q: 1 });
  const plan = outputPlan({ tracks: S.tracks, info: AUDIO, file: { size: 1e6 }, choice: choiceOf('wav'), out: { sfiles: ['srt', 'txt'], burn: false } });
  const w = wantsOf({ tracks: S.tracks, plan, choice: { ...choiceOf('wav'), name: ' x.wav ' }, out: { sfiles: ['srt', 'txt'], burn: false }, file: null, local: true });
  assert.deepEqual(w, { sound: true, format: 'wav', name: 'x.wav', size: 'same', subs: true, caps: { srt: true, vtt: false, txt: true }, embed: false, soundHere: true, subsHere: true });
  assert.equal(wantsOf({ tracks: S.tracks, plan, choice: choiceOf('wav'), out: out0, file: null, local: false }).soundHere, false);
});

await check('start: the messages that stop it are the Cleanup form\'s, word for word (no sound, too long, too big, too many files, prices, credit, nothing chosen, no format, MKV)', () => {
  const S = state();
  const base = { info: AUDIO, file: { size: 1e6 }, tracks: S.tracks, est: 1000, duration: 180, maxJob: 4 * 3600, pricing: PRICING, pricingError: null, balance: { available_micros: 5e6 }, active: 0, maxActive: 3,
    choice: choiceOf('wav'), out: out0 };
  const run = (o) => problemsFor({ ...base, plan: outputPlan({ tracks: (o.tracks || base.tracks), info: o.info || base.info, file: base.file, choice: base.choice, out: base.out }), ...o }).map((p) => p.text);
  assert.deepEqual(run({}), []);
  assert.deepEqual(run({ info: { ...AUDIO, hasAudio: false } }), ['This file has no audio track.']);
  assert.deepEqual(run({ duration: 5 * 3600 }), ['Files can be at most 4 hours long.']);
  assert.deepEqual(run({ file: { size: 5 * 1024 ** 3 } }), ['This file is 5 GB. The most we can take for upload is 4 GB. Cleaning in your browser doesn’t upload the file, so this limit doesn’t apply there.']);
  assert.deepEqual(run({ active: 3 }), ['You have 3 files being processed. Start this one when one of them is done.']);
  assert.deepEqual(problemsFor({ ...base, est: null, pricingError: 'x', plan: outputPlan({ tracks: S.tracks, info: AUDIO, file: base.file, choice: base.choice, out: base.out }) }).map((p) => [p.text, p.kind]),
    [['We couldn’t load the prices just now, so this can’t start yet.', 'prices']]);
  const poor = problemsFor({ ...base, est: 2e6, balance: { available_micros: 1e6 }, plan: outputPlan({ tracks: S.tracks, info: AUDIO, file: base.file, choice: base.choice, out: base.out }) });
  assert.equal(poor[0].text, 'This costs about $2.00, but your balance is $1.00. You can also choose the free options.');
  assert.equal(poor[0].kind, 'credit');
  const none = state(); edit(none, 'voice', 0, 180, { mode: 0 }); edit(none, 'level', 0, 180, { lvl: 0 });
  assert.deepEqual(run({ tracks: none.tracks }), ['Choose at least one thing to do: noise removal, subtitles, or both.']);
  const subs = state(); edit(subs, 'subs', 0, 180, { q: 2 });
  assert.deepEqual(run({ tracks: subs.tracks, out: { sfiles: [], burn: false } }), ['Choose at least one caption or transcript format.']);
  const burn = problemsFor({ ...base, tracks: subs.tracks, choice: choiceOf('mp4'), out: { sfiles: ['srt'], burn: true }, info: VIDEO,
    plan: outputPlan({ tracks: subs.tracks, info: VIDEO, file: base.file, choice: choiceOf('mp4'), out: { sfiles: ['srt'], burn: true } }) });
  assert.deepEqual(burn.map((p) => p.text), ['Subtitles put into the video need an MKV file (most players can’t show them from MP4). Pick MKV, or untick “Put the subtitles into the video”.']);
  assert.equal(inGigabytes('Files must be under 4096 MB.'), 'Files must be under 4 GB.');
});

await check('start: the free options say why they can not be used (no sound, a type the browser can not open, too long)', () => {
  assert.equal(browserVerdict(AUDIO, { ok: true }).ok, true);
  assert.equal(browserVerdict({ ...AUDIO, hasAudio: false }, { ok: true }).why, 'This file has no audio track.');
  assert.equal(browserVerdict(AUDIO, { ok: false }).why, 'This browser can’t run the noise remover. Paid Denoising can do it.');
  assert.match(browserVerdict({ ...AUDIO, canDecode: false, viaMediabunny: false, duration: null }, { ok: true }).why, /can’t open this file type/);
  assert.match(browserVerdict({ ...AUDIO, duration: 7 * 3600 }, { ok: true }).why, /longer than 6 hours/);
  assert.equal(subsVerdict(AUDIO, true).ok, true);
  assert.equal(subsVerdict(AUDIO, false).why, 'This browser can’t run the subtitle maker. Standard can (paid).');
});

/* ------------------------------------------------------------------------------------------------ the browser's share */

await check('browser share: a cleaned range is exactly the samples the server takes (cut or padded), and the Free pieces are cut to the range', () => {
  assert.equal(samplesOf(10, 20), 480000);
  assert.equal(samplesOf(0.5, 47.3), Math.round(47.3 * 48000) - 24000);
  const blob = (n) => new Blob([new Float32Array(n * 2)]); // stereo
  const pcm = { parts: [blob(100), blob(100)], channels: 2, frames: 200, seconds: 200 / 48000 };
  assert.equal(fitPcm(pcm, 200), pcm);
  const cut = fitPcm(pcm, 150);
  assert.equal(cut.frames, 150);
  assert.equal(cut.parts.reduce((n, b) => n + b.size, 0), 150 * 8);
  const pad = fitPcm(pcm, 230);
  assert.equal(pad.frames, 230);
  assert.equal(pad.parts.reduce((n, b) => n + b.size, 0), 230 * 8);
  assert.deepEqual(piecesIn([{ start: 0, end: 40, db: 40 }, { start: 40, end: 60, db: 12 }, { start: 90, end: 100, db: 40 }], 30, 95), [
    { start: 30, end: 40, db: 40 }, { start: 40, end: 60, db: 12 }, { start: 90, end: 95, db: 40 }]);
  assert.deepEqual(rangeFor({ start: 0, end: 47 }, 47.3), { start: 0, end: Infinity }); // (to the very end of the sound)
  assert.deepEqual(rangeFor({ start: 10, end: 20 }, 47.3), { start: 10, end: 20 });
});

/* ------------------------------------------------------------------------------------------------ the words */

await check('words: tooltips carry the live prices (and leave them out while the list is not here); the good-to-know lines are alternatives.jsx\'s; no internals', () => {
  const t = tipsFor(PRICING);
  assert.match(t['k.paid'], /\$0\.009 a minute\./);
  assert.match(t['v.rebuild'], /\$0\.03 a minute, files up to 90 minutes/);
  assert.match(t.echo, /\$0\.0026 a minute\.$/);
  assert.match(t.level, /\$0\.001 a minute\./);
  assert.match(t['q.2'], /\$0\.004 a minute\./);
  assert.match(t['q.3'], /\$0\.015 a minute\./);
  assert.match(t.tr, /\$0\.004 a minute each\./);
  assert.match(t['k.free'], /Good to know: Just a little hum\?/);
  assert.match(t['q.1'], /No translations\./);
  const none = tipsFor(null);
  assert.doesNotMatch(Object.values(none).filter((v) => typeof v === 'string').join(' '), /\$\d/);
  assert.match(none['k.paid'], /a choice of remover\. Good to know/);
  assert.equal(priceWords(PRICING).noise, '$0.009 a minute');
  const BANNED = [/DeepFilterNet/i, /Whisper/, /OpenAI/i, /WebAssembly/i, /\bour servers?\b/i, /premium/i, /\bStudio\b/i, /\bpolish/i];
  for (const v of Object.values(t)) if (typeof v === 'string') for (const re of BANNED) assert.doesNotMatch(v, re);
  assert.equal(fmt(3723), '1:02:03');
  assert.equal(lenTxt(125), '2 min 5 s');
});

console.log(failed ? `\n${failed} check(s) failed` : '\nall passed');
process.exit(failed ? 1 : 0);
