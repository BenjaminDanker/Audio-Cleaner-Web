// Suggestions per part, in the browser: where a recording changes, and what fits each part better than the whole file. The same steps and
// numbers as common/acw_common/suggest.py (the server makes its own after the first run; this is for a file that has not been
// uploaded yet), on the measurements of analysis.js and the trained decision of reco-model.js:
//
//   1. CHANGE POINTS: 5 s windows that do not overlap, each with the features of analysis.js and a noise figure measured on a 3 s stretch
//      of its middle (the noise remover runs over it); a candidate boundary is the distance between neighbouring windows; local peaks
//      from 1.5 up, the strongest first, no two closer than 20 s.
//   2. REFINED on a 1 s grid around each candidate.
//   3. ONE DECISION per part (analysis + the model, as for a whole file); only the parts whose decision differs from the file's are
//      suggestions, with the reasons in plain words.
//   4. CLIPPING SPOTS: short bursts of clipped samples become Voice rebuild suggestions with half a second around them.
//
// Pure: no browser APIs. `read(a, b)` answers the audio from sample a to sample b (48 kHz) as one Float32Array per channel (one or two;
// fewer at the end of the file); `noiseSums(mono)` runs the noise remover (full strength) over a stretch and answers {voice, removed}.
// A worker (suggest.worker.js) and the Node tests (tests/suggest-agreement.mjs) both call findSuggestions.

import { Analyser, HEAVY_BELOW_DB, RATE } from './analysis.js';
import { modelDecision } from '../reco-model.js';

export const COARSE_S = 5;
export const NF_STRETCH_S = 3;
export const TAU = 1.5;
export const MIN_PART_S = 20;
export const REFINE_S = 5;
export const MAX_SECONDS = 30 * 60; // longer files get theirs from the first run
const SCALE = { floorDb: 6, levelDb: 6, noiseDb: 6, tiltDb: 6, dropDb: 4, speechFraction: 0.25, clipFraction: 0.002, bwLog2: 0.6, hasSpeech: 0.5 };
const QUIET_DB = 8;
const NOISY_DB = 3;
const CLIP_WINDOW_S = 60;
const CLIP_OVERLAP_S = 1;
const CLIP_CONTEXT_S = 0.5;
const CLIP_MIN_SAMPLES = 8;
const STRENGTH_CLASSES = ['gentle', 'balanced', 'strong', 'max'];
const STRENGTH_DB = { gentle: 12, balanced: 24, strong: 40, max: 100 };
const REASON_WORDS = {
  echoey: 'sounds echoey', muffled: 'sounds muffled', phone: 'sounds like a phone call', distorted: 'sounds distorted', unclear: 'has lost some of its clarity',
};
// the numbers of recommend-room.js T that the file decision uses
const ECHO_DROP_DB = 10.5;
const DULL_TILT_DB = -19;
const PHONE_HZ = 6600;
const CLIP_FRACTION = 0.002;

const abortError = () => new DOMException('Canceled', 'AbortError');

/** Runs fn over items with at most `limit` at once, answering in order. */
async function mapLimit(items, limit, fn) {
  const out = new Array(items.length);
  let next = 0;
  const run = async () => {
    for (;;) {
      const i = next++;
      if (i >= items.length) return;
      out[i] = await fn(items[i], i);
    }
  };
  await Promise.all(Array.from({ length: Math.max(1, Math.min(limit, items.length)) }, run));
  return out;
}
const pyround = (x) => { const r = Math.round(x); return Math.abs(x % 1) === 0.5 && r % 2 ? r - 1 : r; }; // (Python's round: halves to even)

/* ------------------------------------------------------------------------------------------------ the decision of a stretch */

/** reco_model.recommend with no minimum of sound: strength, whether to rebuild and why, the stronger remover. Null: nothing to say. */
export function recommendPart(stats) {
  if (!stats || stats.noiseDb == null) return null;
  const noiseDb = Number(stats.noiseDb);
  const heavy = noiseDb < HEAVY_BELOW_DB;
  if (!stats.hasSpeech) return { strength: 'balanced', rebuild: false, heavy, noSpeech: true, noiseDb, reasons: [] };
  const m = modelDecision(stats);
  const phone = stats.bandwidthHz <= PHONE_HZ;
  const echoey = stats.dropDb != null && stats.dropDb < ECHO_DROP_DB;
  const dull = !phone && stats.tiltDb != null && stats.tiltDb <= DULL_TILT_DB;
  const clipped = (stats.clipFraction || 0) > CLIP_FRACTION;
  const all = [['echoey', echoey], ['muffled', dull], ['phone', phone], ['distorted', clipped], ['unclear', m.rebuild]].filter(([, on]) => on).map(([r]) => r);
  const reasons = all.filter((r) => r !== 'unclear' || all.length === 1);
  return { strength: m.strength, rebuild: all.length > 0, heavy, noSpeech: false, noiseDb, reasons };
}

/** The decision for a stretch with these measurements: {rebuild, remover, strength, noiseDb, levelDb, reasons} (suggest._decision). */
export function decisionOf(stats) {
  const rec = recommendPart(stats);
  if (!rec) return { rebuild: false, remover: 'light', strength: 'balanced', noiseDb: null, levelDb: null, reasons: [] };
  return { rebuild: rec.rebuild, remover: rec.heavy ? 'heavy' : 'light', strength: rec.strength, noiseDb: stats.noiseDb, levelDb: stats.levelDb, reasons: rec.reasons };
}

const keyOf = (d) => `${d.rebuild}/${d.remover}/${d.strength}`;

/** Plain words for how a part differs from the whole file (suggest.reasons_for). */
export function reasonsFor(part, whole) {
  const out = [];
  if (part.rebuild && !whole.rebuild) {
    for (const r of part.reasons) if (REASON_WORDS[r]) out.push(REASON_WORDS[r]);
    if (!out.length) out.push('sounds like it needs rebuilding');
  } else if (whole.rebuild && !part.rebuild) {
    out.push('sounds fine here, no rebuild needed');
  }
  let noisier = false;
  let cleaner = false;
  if (part.noiseDb != null && whole.noiseDb != null) {
    noisier = part.noiseDb <= whole.noiseDb - NOISY_DB;
    cleaner = part.noiseDb >= whole.noiseDb + NOISY_DB;
  }
  if (part.rebuild && noisier) out.push('is noisier');
  const ok = (v) => v != null && !Number.isNaN(v);
  if (ok(part.levelDb) && ok(whole.levelDb)) {
    if (part.levelDb <= whole.levelDb - QUIET_DB) out.push('is quiet');
    else if (part.levelDb >= whole.levelDb + QUIET_DB) out.push('is loud');
  }
  if (!part.rebuild && part.remover !== whole.remover) out.push(part.remover === 'heavy' ? 'needs the stronger noise remover' : 'needs only the lighter noise remover');
  if (!part.rebuild && part.strength !== whole.strength) {
    // the reason follows the setting proposed: a stronger setting is never explained by "is cleaner", nor a gentler one by "is noisier"
    // (the noise figure and the trained choice can disagree)
    if (STRENGTH_CLASSES.indexOf(part.strength) > STRENGTH_CLASSES.indexOf(whole.strength)) out.push(noisier ? 'is noisier' : 'has noise that needs a stronger setting');
    else out.push(cleaner ? 'is cleaner' : 'needs a gentler noise removal');
  }
  return out;
}

/* ------------------------------------------------------------------------------------------------ measuring a stretch */

/** pipeline.measure_windows: all of a short stretch, otherwise six 10 s stretches spread evenly over it, as [from, to) sample pairs. */
export function measureWindows(n, totalSeconds = 60, count = 6) {
  const duration = n / RATE;
  if (n <= 0) return [];
  if (duration <= totalSeconds * 1.25) return [[0, n]];
  const length = totalSeconds / count;
  const out = [];
  for (let i = 0; i < count; i++) {
    const mid = ((i + 0.5) * duration) / count;
    const start = Math.min(duration - length, Math.max(0, mid - length / 2));
    const a = pyround(start * RATE);
    out.push([a, Math.min(n, a + pyround(length * RATE))]);
  }
  return out;
}

const monoOf = (planes) => {
  if (planes.length === 1) return planes[0];
  const m = new Float32Array(planes[0].length);
  for (let i = 0; i < m.length; i++) m[i] = (planes[0][i] + planes[1][i]) * 0.5;
  return m;
};

const figureOf = (voice, removed) => (voice + removed <= 0 ? 99 : Math.min(99, 10 * Math.log10((voice + 1e-12) / (removed + 1e-12))));

/** The noise figure of the stretch [A, B) of the file (pipeline.noise_figure_db: the noise remover over the measured windows) and its features. */
async function stretchStats(io, A, B) {
  const windows = measureWindows(B - A);
  const got = await mapLimit(windows, io.concurrency, async ([a0, b0]) => {
    io.check();
    const planes = await io.read(A + a0, A + b0);
    if (!planes[0] || !planes[0].length) return null;
    return { planes, sums: await io.noiseSums(monoOf(planes)) };
  });
  const an = new Analyser({ channels: 1 });
  let voice = 0;
  let removed = 0;
  let first = true;
  for (const g of got) {
    if (!g) continue;
    if (first) { an.channels = g.planes.length >= 2 ? 2 : 1; first = false; } else an.segment();
    voice += g.sums.voice;
    removed += g.sums.removed;
    an.push(g.planes.length >= 2 ? g.planes.slice(0, 2) : [g.planes[0]]);
  }
  const stats = an.result();
  stats.noiseDb = figureOf(voice, removed);
  return stats;
}

/** Features of one 5 s window and the noise figure of the 3 s from its middle. `planes`: the window's audio. */
async function windowVector(io, planes) {
  const mono = monoOf(planes);
  const mid = Math.floor(mono.length / 2);
  const half = Math.floor((NF_STRETCH_S * RATE) / 2);
  const sums = await io.noiseSums(mono.subarray(Math.max(0, mid - half), mid + half));
  const nf = figureOf(sums.voice, sums.removed);
  const an = new Analyser({ channels: planes.length >= 2 ? 2 : 1 });
  an.push(planes.length >= 2 ? planes.slice(0, 2) : [planes[0]]);
  return fvec(an.result(), nf);
}

function fvec(st, nf) {
  const num = (v) => (v == null ? NaN : v);
  return {
    floorDb: num(st.floorDb), levelDb: num(st.levelDb), tiltDb: num(st.tiltDb), dropDb: num(st.dropDb), speechFraction: num(st.speechFraction),
    clipFraction: num(st.clipFraction), noiseDb: nf, bwLog2: Math.log2(Math.max(st.bandwidthHz || 1, 1)), hasSpeech: st.hasSpeech ? 1 : 0,
  };
}

function dist(a, b) {
  let best = 0;
  let any = false;
  for (const k of Object.keys(SCALE)) {
    if (Number.isNaN(a[k]) || Number.isNaN(b[k])) continue;
    const d = Math.abs(a[k] - b[k]) / SCALE[k];
    if (!any || d > best) best = d;
    any = true;
  }
  return any ? best : 0;
}

/* ------------------------------------------------------------------------------------------------ clipping spots */

function clipLevel(a, { bw = 0.0125, minCount = 40, minRatio = 2.5, keep = 0.3, floor = 0.45 } = {}) {
  const nb = 80;
  const h = new Int32Array(nb);
  for (let i = 0; i < a.length; i++) h[Math.min(nb - 1, Math.floor(a[i] / bw))]++;
  const b0 = pyround(0.1 / bw);
  const b1 = pyround(0.2 / bw);
  const start = Math.trunc(floor / bw);
  let best = null;
  const median = (arr) => { const s = Float64Array.from(arr).sort(); const m = s.length >> 1; return s.length % 2 ? s[m] : (s[m - 1] + s[m]) / 2; };
  for (let i = start; i < nb; i++) {
    const lo = i - b1;
    const hi = i - b0;
    if (lo < 0) continue;
    const base = median(h.subarray(lo, hi)) + 1;
    if (h[i] >= minCount && h[i] >= minRatio * base) {
      const sc = h[i] / base;
      if (best === null || sc > best.sc) best = { sc, i, base };
    }
  }
  if (!best) return null;
  let j = best.i;
  while (j - 1 >= start && h[j - 1] >= Math.max(keep * h[best.i], 1.8 * best.base)) j--;
  return j * bw; // (the histogram's edges are multiples of bw)
}

/** Bursts of clipped samples in a mono stretch: [[first, end, clipped samples], ...] (suggest.clipped_bursts). */
export function clippedBursts(x, { minRun = 2, burstGap = 0.1, minCount = 40, minRatio = 1.6, minLevel = 0.25 } = {}) {
  const n = x.length;
  let peak = 0;
  for (let i = 0; i < n; i++) { const v = x[i] < 0 ? -x[i] : x[i]; if (v > peak) peak = v; }
  if (peak < 0.05) return [];
  const a = new Float32Array(n);
  for (let i = 0; i < n; i++) a[i] = Math.abs(x[i]) / peak;
  const thr = clipLevel(a, { minCount, minRatio });
  if (thr === null || thr * peak < minLevel) return [];
  const events = [];
  let i = 0;
  while (i < n) {
    if (a[i] < thr) { i++; continue; }
    const st = i;
    while (i < n && a[i] >= thr) i++;
    // split where the sign changes
    let cut = st;
    for (let j = st + 1; j <= i; j++) {
      if (j === i || Math.sign(x[j]) !== Math.sign(x[j - 1])) {
        if (j - cut >= minRun) events.push([cut, j]);
        cut = j;
      }
    }
  }
  if (!events.length) return [];
  const gap = Math.trunc(burstGap * RATE);
  const bursts = [[events[0][0], events[0][1], events[0][1] - events[0][0]]];
  for (const [st, en] of events.slice(1)) {
    const last = bursts[bursts.length - 1];
    if (st - last[1] <= gap) { last[1] = en; last[2] += en - st; } else bursts.push([st, en, en - st]);
  }
  return bursts;
}

/** The clipped spots found in the audio of one window of the file starting at sample `a0` (of `n` samples in all): spans in samples. */
function clipSpans(mono, a0, n) {
  const bursts = clippedBursts(mono);
  if (!bursts.length) return [];
  if (bursts.reduce((s, b) => s + b[2], 0) > 0.08 * mono.length) return []; // heavily clipped throughout: not a spot
  const ctx = Math.trunc(CLIP_CONTEXT_S * RATE);
  return bursts.filter((b) => b[2] >= CLIP_MIN_SAMPLES).map(([s0, e0]) => [Math.max(0, a0 + s0 - ctx), Math.min(n, a0 + e0 + ctx)]);
}

function mergeSpans(spans) {
  const sorted = [...spans].sort((p, q) => p[0] - q[0] || p[1] - q[1]);
  const out = [];
  for (const [s0, e0] of sorted) {
    const last = out[out.length - 1];
    if (last && s0 - last[1] <= Math.trunc(0.25 * RATE)) last[1] = Math.max(last[1], e0);
    else out.push([s0, e0]);
  }
  return out.map(([s0, e0]) => [s0 / RATE, e0 / RATE]);
}

/* ------------------------------------------------------------------------------------------------ the whole thing */

/**
 * @param {{seconds: number, read: (a: number, b: number) => Promise<Float32Array[]>, noiseSums: (mono: Float32Array) => Promise<{voice: number, removed: number}>,
 *   onProgress?: (fraction: number) => void, onPeaks?: (from: number, peaks: Float32Array) => void, signal?: AbortSignal, clipping?: boolean,
 *   concurrency?: number}} o  (concurrency: how many noise remover runs may be going at once; the answer does not depend on it)
 * @returns {Promise<{seconds, whole: {rebuild, remover, strength}, parts, suggestions: {start, end, kind, rebuild, noise, reasons}[]}>}
 */
export async function findSuggestions({ seconds, read, noiseSums, onProgress, onPeaks, signal, clipping = true, concurrency = 1 }) {
  const n = Math.round(seconds * RATE);
  const total = n / RATE;
  const check = () => { if (signal?.aborted) throw abortError(); };
  const io = { read: async (a, b) => read(a, Math.min(n, b)), noiseSums, check, concurrency };
  const progress = (f) => onProgress?.(Math.max(0, Math.min(1, f)));
  const cache = new Map(); // window start (s) -> feature vector
  const k = Math.floor(total / COARSE_S);
  const planWindows = k >= 2 && total >= 2 * MIN_PART_S;
  const spans = [];

  // pass 1: the windows, in blocks of the clipping detector's own size so that the file is read once
  const rest = [];
  const hop = Math.trunc((CLIP_WINDOW_S - CLIP_OVERLAP_S) * RATE);
  const size = Math.trunc(CLIP_WINDOW_S * RATE);
  const blocks = [];
  for (let a = 0; a < Math.max(1, n - Math.trunc(CLIP_OVERLAP_S * RATE)); a += hop) { blocks.push(a); if (Math.min(n, a + size) >= n) break; }
  const peaksPerSecond = 4;
  for (let bi = 0; bi < blocks.length; bi++) {
    check();
    const a0 = blocks[bi];
    const b0 = Math.min(n, a0 + size);
    const planes = await io.read(a0, b0);
    const mono = monoOf(planes);
    // the waveform, for the page
    const per = RATE / peaksPerSecond;
    const pk = new Float32Array(Math.ceil(mono.length / per));
    for (let p = 0; p < pk.length; p++) {
      let m = 0;
      for (let i = p * per, e = Math.min(mono.length, (p + 1) * per); i < e; i += 4) { const v = mono[i] < 0 ? -mono[i] : mono[i]; if (v > m) m = v; }
      pk[p] = m;
    }
    onPeaks?.(Math.round(a0 / per), pk);
    if (clipping) spans.push(...clipSpans(mono, a0, n));
    if (planWindows) {
      const todo = [];
      for (let w = 0; w < k; w++) {
        const s = w * COARSE_S;
        const wa = Math.round(s * RATE);
        const wb = wa + Math.round(COARSE_S * RATE);
        if (!cache.has(s) && wa >= a0 && wb <= b0) todo.push([s, wa, wb]);
      }
      await mapLimit(todo, concurrency, async ([s, wa, wb]) => {
        check();
        cache.set(s, await windowVector(io, planes.map((p) => p.subarray(wa - a0, wb - a0))));
        progress(0.7 * (cache.size / k));
      });
    }
    progress(planWindows ? 0.7 * (cache.size / k) : 0.7 * ((bi + 1) / blocks.length));
  }
  if (planWindows) {
    for (let w = 0; w < k; w++) {
      const s = w * COARSE_S;
      if (cache.has(s)) continue;
      rest.push(s);
    }
    await mapLimit(rest, concurrency, async (s) => {
      check();
      const wa = Math.round(s * RATE);
      cache.set(s, await windowVector(io, await io.read(wa, wa + Math.round(COARSE_S * RATE))));
    });
  }
  progress(0.7);

  // the file's own decision
  const wholeStats = n ? await stretchStats(io, 0, n) : { noiseDb: null };
  const whole = decisionOf(wholeStats);
  progress(0.75);

  // (the windows around a candidate are cut from one read of the stretch)
  const fill = async (starts) => {
    const need = [...new Set(starts)].filter((t) => !cache.has(t)).sort((p, q) => p - q);
    if (!need.length) return;
    const from = Math.round(need[0] * RATE);
    const to = Math.min(n, Math.round((need[need.length - 1] + COARSE_S) * RATE));
    const block = await io.read(from, to);
    await mapLimit(need, concurrency, async (t) => {
      check();
      const wa = Math.round(t * RATE);
      cache.set(t, await windowVector(io, block.map((p) => p.subarray(wa - from, wa - from + Math.round(COARSE_S * RATE)))));
    });
  };

  // change points
  let cuts = [];
  if (planWindows) {
    const starts = Array.from({ length: k }, (_, i) => i * COARSE_S);
    const cand = [];
    for (let i = 0; i < k - 1; i++) cand.push([(i + 1) * COARSE_S, dist(cache.get(starts[i]), cache.get(starts[i + 1]))]);
    const peaks = cand.filter(([, d], j) => d >= TAU && (j === 0 || d >= cand[j - 1][1]) && (j === cand.length - 1 || d >= cand[j + 1][1]));
    peaks.sort((p, q) => q[1] - p[1]);
    const kept = [];
    for (const [t] of peaks) if (t >= MIN_PART_S && t <= total - MIN_PART_S && kept.every((u) => Math.abs(t - u) >= MIN_PART_S)) kept.push(t);
    const refined = [];
    let done = 0;
    for (const t of [...kept].sort((p, q) => p - q)) {
      const lo = pyround(t) - REFINE_S;
      const hi = pyround(t) + REFINE_S;
      const cs = [];
      for (let c = lo; c <= hi; c++) if (c - COARSE_S >= 0 && c + COARSE_S <= k * COARSE_S + 1e-9 && (c + COARSE_S) * RATE <= n) cs.push(c);
      await fill(cs.flatMap((c) => [c - COARSE_S, c]));
      let best = -1;
      let bt = t;
      for (const c of cs) {
        const d = dist(cache.get(c - COARSE_S), cache.get(c));
        if (d > best) { best = d; bt = c; }
      }
      refined.push(bt);
      progress(0.75 + 0.1 * (++done / Math.max(1, kept.length)));
    }
    for (const t of refined.sort((p, q) => p - q)) {
      if ((!cuts.length || t - cuts[cuts.length - 1] >= MIN_PART_S) && total - t >= MIN_PART_S && t >= MIN_PART_S) cuts.push(t);
    }
  }
  progress(0.85);

  // one decision per part
  const edges = [0, ...cuts, total];
  const parts = [];
  const suggestions = [];
  for (let i = 0; i < edges.length - 1; i++) {
    check();
    const a = edges[i];
    const b = edges[i + 1];
    const d = cuts.length === 0 ? whole : decisionOf(await stretchStats(io, Math.round(a * RATE), Math.min(n, Math.round(b * RATE))));
    parts.push([a, b, d]);
    if (keyOf(d) !== keyOf(whole)) {
      suggestions.push({ start: a, end: b, kind: 'part', rebuild: d.rebuild, noise: { strength_db: STRENGTH_DB[d.strength], remover: d.remover }, reasons: reasonsFor(d, whole) });
    }
    progress(0.85 + 0.1 * ((i + 1) / (edges.length - 1)));
  }
  if (clipping && n) {
    const rebuiltAt = (t) => { const p = parts.find(([a, b]) => a <= t && t < b); return p ? p[2].rebuild : whole.rebuild; };
    for (const [a, b] of mergeSpans(spans)) {
      if (rebuiltAt((a + b) / 2) && rebuiltAt(a) && rebuiltAt(Math.min(b, total - 1e-3))) continue; // the voice is rebuilt there anyway
      suggestions.push({ start: a, end: b, kind: 'clipping', rebuild: true, noise: null, reasons: ['sounds distorted here (clipped)'] });
    }
  }
  suggestions.sort((p, q) => p.start - q.start || p.end - q.end);
  progress(1);
  const r3 = (t) => Math.round(t * 1000) / 1000;
  return {
    seconds: total,
    whole: { rebuild: whole.rebuild, remover: whole.remover, strength: whole.strength },
    parts: parts.map(([a, b, d]) => ({ start: a, end: b, rebuild: d.rebuild, remover: d.remover, strength: d.strength })),
    suggestions: suggestions.map((s) => ({ ...s, start: r3(s.start), end: r3(s.end) })),
  };
}
