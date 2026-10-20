// Listening to the timeline in the browser. The server hands out short stretches of what the job holds (source, echo, noise at full
// strength, rebuild; GET /api/jobs/{id}/preview/{layer}) that line up sample for sample; this page mixes them with the gains the
// timeline asks for, so a change of strength, a layer switched off or a part painted is heard at once:
//
//   strength a (share of the dry sound that stays): noise gain (1 - a), dry gain a        (as the noise remover itself blends)
//   dry sound: the echo-free voice where echo removal covers the moment, else the recording
//   Voice rebuild wins where both it and noise removal cover a moment; its edges are an equal-power crossfade (~250 ms) with
//   whatever is there instead (the noise removal, or the dry sound)
//   a layer that is not made yet for a moment (not in `cached`) is skipped there: the closest version that exists is heard
//   Leveling is previewed as loudness-matched (a gain toward the target), and applied for real when you save
//
// weightsAt / envelope / describe are pure (tests/editor-unit.mjs); MixPlayer plays the stretches with Web Audio.

import { decodeAudio, channelsOf, mixOf, SAMPLE_RATE } from '../media/blend.js';
import { API } from '../api.js';
import { intersect, mergeRanges, subtract } from './timeline.js';

export const HEAR_ALL = { echo: true, noise: true, rebuild: true, leveling: true };
export const FADE_REBUILD = 0.125; // half of the equal-power crossfade at the edges of a rebuilt part (seconds)
const FADE_SHORT = 0.01; // the edges of the other parts: only enough to avoid a click
const FADE_LEVEL = 0.05;
const REBUILD_STEPS = 8;

/** Coverage of a time by sorted ranges, with a linear ramp of `half` seconds each side of every edge (no ramp at the ends of the file). */
function cover(ranges, t, half, seconds) {
  let c = 0;
  for (const [s, e] of ranges) {
    if (t < s - half || t > e + half) continue;
    const up = s <= 0 ? 1 : Math.min(1, Math.max(0, (t - (s - half)) / (2 * half)));
    const down = e >= seconds - 1e-6 ? 1 : Math.min(1, Math.max(0, (e + half - t) / (2 * half)));
    c += Math.min(up, down);
  }
  return Math.min(1, c);
}

/**
 * Everything the gains depend on, worked out once per change: model = {tl, hear, cached, secondTrack, missing?}. `cached`: seconds ranges
 * per layer ({echo, noise, rebuild}) that exist on the server; `missing`: ranges a fetched stretch said are silent (X-Stretch-Missing).
 * `wanted` ignores `hear` and the cache: which layers' stretches are worth fetching.
 */
export function prepare(model) {
  const { tl, hear = HEAR_ALL, cached = {}, secondTrack = false, missing = {} } = model;
  const have = (layer) => subtract(mergeRanges(cached[layer] || []), missing[layer] || []);
  const ranges = (layer) => mergeRanges(tl.parts(layer).map((p) => [p.start, p.end]));
  const wanted = {
    rebuild: intersect(ranges('rebuild'), have('rebuild')),
    echo: secondTrack ? intersect(ranges('echo'), have('echo')) : [],
    noise: tl.parts('noise').map((p) => ({ ranges: intersect([[p.start, p.end]], have('noise')), a: p.settings.browser ? 0 : mixOf(p.settings.strength_db) })), // (a part the browser cleaned is already at its strength)
  };
  const lv = tl.parts('leveling');
  return {
    seconds: tl.seconds,
    wanted,
    R: hear.rebuild ? wanted.rebuild : [],
    E: hear.echo ? wanted.echo : [],
    N: hear.noise ? wanted.noise.map((n) => ({ ...n, a: Math.min(1, n.a) })) : [],
    L: hear.leveling ? lv.map((p) => [p.start, p.end]) : [],
    lufs: lv.length ? lv[0].settings.loudness_lufs : null,
  };
}

/** The gain of each stretch at time t: {source, echo, noise, rebuild} (equal power between the rebuilt voice and the rest). */
export function weightsAt(prep, t) {
  const sec = prep.seconds;
  const x = cover(prep.R, t, FADE_REBUILD, sec);
  const rest = Math.cos((x * Math.PI) / 2);
  const reb = Math.sin((x * Math.PI) / 2);
  let nShare = 0;
  for (const n of prep.N) nShare += cover(n.ranges, t, FADE_SHORT, sec) * (1 - n.a);
  nShare = Math.min(1, nShare);
  const e = cover(prep.E, t, FADE_SHORT, sec);
  const dry = rest * (1 - nShare);
  return { source: dry * (1 - e), echo: dry * e, noise: rest * nShare, rebuild: reb };
}

/** How much of Leveling's gain applies at t (0 to 1). */
export const levelCoverAt = (prep, t) => cover(prep.L, t, FADE_LEVEL, prep.seconds);

/** The moments in [t0, t1] where a gain can change direction: the edges of the parts, with the steps of the crossfades. */
export function breakpoints(prep, t0, t1) {
  const pts = new Set([t0, t1]);
  const add = (t) => { if (t > t0 && t < t1) pts.add(Math.round(t * 1e4) / 1e4); };
  for (const [s, e] of prep.R) {
    for (const edge of [s, e]) for (let i = 0; i <= REBUILD_STEPS; i++) add(edge - FADE_REBUILD + (2 * FADE_REBUILD * i) / REBUILD_STEPS);
  }
  const short = (ranges) => { for (const [s, e] of ranges) for (const edge of [s, e]) { add(edge - FADE_SHORT); add(edge + FADE_SHORT); } };
  short(prep.E);
  for (const n of prep.N) short(n.ranges);
  for (const [s, e] of prep.L) for (const edge of [s, e]) { add(edge - FADE_LEVEL); add(edge + FADE_LEVEL); }
  return [...pts].sort((a, b) => a - b);
}

/** Gain values of every stretch at the breakpoints of [t0, t1]: {times, source[], echo[], noise[], rebuild[], level[]}. */
export function envelope(prep, t0, t1) {
  const times = breakpoints(prep, t0, t1);
  const out = { times, source: [], echo: [], noise: [], rebuild: [], level: [] };
  for (const t of times) {
    const w = weightsAt(prep, t);
    for (const k of ['source', 'echo', 'noise', 'rebuild']) out[k].push(w[k]);
    out.level.push(levelCoverAt(prep, t));
  }
  return out;
}

/** What is heard at t, for the page and its tests. */
export function describe(prep, t, levelFactor = 1) {
  const w = weightsAt(prep, t);
  const lc = levelCoverAt(prep, t);
  return { ...w, level: 1 + (levelFactor - 1) * lc, levelCover: lc };
}

/** Which layers of a part are not made yet (so the preview plays something else there): [{layer, start, end}]. */
export function needsProcessing(tl, cached, secondTrack) {
  const out = [];
  for (const layer of ['echo', 'noise', 'rebuild']) {
    if (layer === 'echo' && !secondTrack) continue;
    const have = mergeRanges(cached[layer] || []);
    for (const p of tl.parts(layer)) for (const [s, e] of subtract([[p.start, p.end]], have)) if (e - s > 0.05) out.push({ layer, start: s, end: e });
  }
  return out;
}

/** The gain that brings sound at `measuredDb` (RMS, dBFS) to the target (LUFS), kept within 12 dB: Leveling's preview. */
export function levelFactor(measuredDb, lufs) {
  if (lufs == null || !Number.isFinite(measuredDb)) return 1;
  return Math.min(4, Math.max(0.25, 10 ** ((lufs - measuredDb) / 20)));
}

/* ------------------------------------------------------------------------------------------------ the stretches */

export const CHUNK_SECONDS = 10;
const AHEAD = 2;
const BEHIND = 1;
const RETRY_MS = 2000;

/** One stretch of a layer from the server: {channels: Float32Array[], missing: [[s, e]] (absolute seconds), bytes} or null when none of it is made. */
export async function fetchStretch(jobId, layer, start, seconds, spec, signal) {
  const url = `${API}jobs/${encodeURIComponent(jobId)}/preview/${layer}?start=${start.toFixed(6)}&seconds=${seconds.toFixed(6)}`;
  const res = await fetch(url, { credentials: 'same-origin', signal });
  if (res.status === 404) return null;
  if (!res.ok) throw new Error('Couldn’t get the sound to listen with. Please try again.');
  const bytes = await res.arrayBuffer();
  const scale = Number(res.headers.get('X-Stretch-Scale')) || 1;
  const samples = Number(res.headers.get('X-Stretch-Samples')) || Math.round(seconds * SAMPLE_RATE);
  let missing = [];
  try { missing = JSON.parse(res.headers.get('X-Stretch-Missing') || '[]').map(([a, b]) => [start + a, start + b]); } catch { /* none */ }
  const decoded = await decodeAudio(bytes);
  return { channels: channelsOf(decoded, { samples, channels: spec.channels, scale }), missing, bytes: bytes.byteLength, samples };
}

const rmsDb = (arrays) => {
  let s = 0;
  let n = 0;
  for (const a of arrays) { for (let i = 0; i < a.length; i += 4) { s += a[i] * a[i]; n++; } }
  return n ? 10 * Math.log10(s / n + 1e-12) : -120;
};

/**
 * Plays the timeline from the job's stretches, 10 s at a time (the one that plays, two ahead, one behind). `getModel()` gives the
 * current model of prepare(); call remix() whenever it changed. `preview`: the job's timeline answer's `preview` ({samples, channels}).
 * options: onState(snapshot), onEnd(), context (a function that makes the AudioContext: for tests), tickMs.
 */
export class MixPlayer {
  constructor({ jobId, preview, getModel, onState, onEnd, context = null, tickMs = 100, fetcher = fetchStretch }) {
    this.jobId = jobId;
    this.spec = { channels: Math.max(1, Math.min(2, preview.channels?.length ?? preview.channels ?? 2)) };
    this.samples = preview.samples;
    this.duration = preview.samples / SAMPLE_RATE;
    this.getModel = getModel;
    this.onState = onState;
    this.onEnd = onEnd;
    this.fetcher = fetcher;
    this.ctx = context ? context() : new (window.AudioContext || window.webkitAudioContext)({ sampleRate: SAMPLE_RATE });
    this.master = this.ctx.createGain();
    this.limiter = this.ctx.createDynamicsCompressor();
    this.limiter.threshold.value = -2; this.limiter.knee.value = 0; this.limiter.ratio.value = 20; this.limiter.attack.value = 0.003; this.limiter.release.value = 0.1;
    this.master.connect(this.limiter);
    this.limiter.connect(this.ctx.destination);
    this.chunkCount = Math.max(1, Math.ceil(this.duration / CHUNK_SECONDS));
    this.held = new Map();
    this.offset = 0; this.startedAt = 0; this.offsetAtStart = 0;
    this.running = false; this.playing = false; this.buffering = false;
    this.scheduledTo = -1; this.epoch = 0; this.error = ''; this.requests = 0;
    this.refEnergy = 0; this.refLen = 0; // the level of what was heard so far, for the loudness match of Leveling
    this.peaks = new Map(); // chunk -> Float32Array of peaks per 0.25 s, for the waveform
    this.closed = false;
    this.timer = tickMs ? setInterval(() => this.tick(), tickMs) : null;
  }

  snapshot() {
    let ready = 0;
    for (const e of this.held.values()) if (e.state === 'ready') ready++;
    return { buffering: this.buffering, error: this.error, held: ready, requests: this.requests };
  }

  _emit() { if (!this.closed) this.onState?.(this.snapshot()); }

  get time() {
    if (!this.playing || !this.running) return this.offset;
    return Math.min(this.duration, Math.max(this.offsetAtStart, this.ctx.currentTime - this.startedAt));
  }

  _chunkAt(t) { return Math.max(0, Math.min(this.chunkCount - 1, Math.floor(t / CHUNK_SECONDS))); }

  _span(k) { const a = k * CHUNK_SECONDS; return { start: a, seconds: Math.min(CHUNK_SECONDS, this.duration - a) }; }

  /** The layers of chunk k worth fetching: the recording, and what the timeline asks for there that exists. */
  _layersFor(k) {
    const { start, seconds } = this._span(k);
    const prep = prepare(this.getModel());
    const hits = (ranges) => ranges.some(([s, e]) => e > start && s < start + seconds);
    const out = ['source'];
    if (prep.wanted.noise.some((n) => hits(n.ranges))) out.push('noise');
    if (hits(prep.wanted.rebuild)) out.push('rebuild');
    if (hits(prep.wanted.echo)) out.push('echo');
    return out;
  }

  _load(k, force = false) {
    let e = this.held.get(k);
    if (e && (e.state !== 'failed' || (!force && Date.now() < e.retryAt))) return e.promise;
    if (e) this._drop(k);
    const { start, seconds } = this._span(k);
    const ctl = new AbortController();
    e = { state: 'loading', ctl, bufs: {}, missing: {}, nodes: [], retryAt: 0, promise: null, layers: new Set() };
    this.held.set(k, e);
    this.requests++;
    e.promise = this._fetchLayers(e, k, start, seconds, this._layersFor(k)).then(
      () => {
        if (this.closed || this.held.get(k) !== e) return e;
        e.state = 'ready';
        this._emit();
        if (this.playing && this.running && !this.buffering) this._fill();
        return e;
      },
      (err) => {
        if (this.closed || this.held.get(k) !== e) return e;
        e.state = 'failed'; e.retryAt = Date.now() + RETRY_MS; e.error = err;
        this._emit();
        throw err;
      },
    );
    e.promise.catch(() => {});
    this._emit();
    return e.promise;
  }

  async _fetchLayers(e, k, start, seconds, layers) {
    await Promise.all(layers.map(async (layer) => {
      if (e.layers.has(layer)) return;
      const got = await this.fetcher(this.jobId, layer, start, seconds, this.spec, e.ctl.signal);
      e.layers.add(layer);
      if (!got) { e.missing[layer] = [[start, start + seconds]]; return; }
      e.missing[layer] = got.missing;
      e.bufs[layer] = this._buffer(got.channels);
      if (layer === 'source') this._measure(k, got.channels);
    }));
  }

  _buffer(arrays) {
    const frames = Math.max(1, arrays[0]?.length || 1);
    const buf = this.ctx.createBuffer(arrays.length, frames, SAMPLE_RATE);
    arrays.forEach((a, c) => buf.copyToChannel(a, c));
    return buf;
  }

  _measure(k, arrays) {
    const db = rmsDb(arrays);
    if (db > -60) { this.refEnergy += 10 ** (db / 10) * arrays[0].length; this.refLen += arrays[0].length; }
    const n = Math.ceil(arrays[0].length / (SAMPLE_RATE / 4));
    const peaks = new Float32Array(n);
    for (let i = 0; i < n; i++) {
      let m = 0;
      for (const a of arrays) for (let j = i * (SAMPLE_RATE / 4); j < Math.min(a.length, (i + 1) * (SAMPLE_RATE / 4)); j += 8) m = Math.max(m, Math.abs(a[j]));
      peaks[i] = m;
    }
    this.peaks.set(k, peaks);
    this.onPeaks?.(k, peaks);
  }

  _refDb() { return this.refLen ? 10 * Math.log10(this.refEnergy / this.refLen) : NaN; }

  /** Layers that the model now wants in chunk k but that were not fetched (a part was painted over a stretch already held). */
  _topUp(k, e) {
    const more = this._layersFor(k).filter((l) => !e.layers.has(l));
    if (!more.length || e.state !== 'ready') return;
    const { start, seconds } = this._span(k);
    e.state = 'loading';
    e.promise = this._fetchLayers(e, k, start, seconds, more).then(() => {
      if (this.closed || this.held.get(k) !== e) return e;
      e.state = 'ready';
      this._remix(k, e);
      this._emit();
      return e;
    }, () => { if (!this.closed && this.held.get(k) === e) { e.state = 'ready'; this._emit(); } return e; });
  }

  _drop(k) {
    const e = this.held.get(k);
    if (!e) return;
    this.held.delete(k);
    e.ctl.abort();
    this._stopNodes(e);
    e.bufs = {};
  }

  _stopNodes(e) {
    for (const n of e.nodes) { n.onended = null; try { n.stop?.(); } catch { /* not started */ } try { n.disconnect(); } catch { /* gone */ } }
    e.nodes = []; e.gain = null; e.bus = null;
  }

  _keepAround(k) {
    let changed = false;
    for (const j of [...this.held.keys()]) if (j < k - BEHIND || j > k + AHEAD) { this._drop(j); changed = true; }
    if (changed) this._emit();
  }

  _prefetch(k) { for (let j = k; j <= Math.min(this.chunkCount - 1, k + AHEAD); j++) this._load(j).catch(() => {}); }

  /** The gain automation of one held chunk, from the current model. */
  _remix(k, e) {
    if (!e.gain || !this.running) return;
    const model = this.getModel();
    const prep = prepare({ ...model, missing: missingOf(e) });
    const { start, seconds } = this._span(k);
    const env = envelope(prep, start, start + seconds);
    const at = (t) => this.startedAt + t;
    const f = prep.lufs == null ? 1 : levelFactor(this._refDb(), prep.lufs);
    const sched = (param, values, map = (v) => v) => {
      param.cancelScheduledValues(0);
      param.setValueAtTime(map(values[0]), at(env.times[0]));
      for (let i = 1; i < env.times.length; i++) param.linearRampToValueAtTime(map(values[i]), at(env.times[i]));
    };
    for (const layer of Object.keys(e.gain)) sched(e.gain[layer].gain, env[layer]);
    sched(e.bus.gain, env.level, (c) => 1 + (f - 1) * c);
  }

  _fill() {
    if (!this.running || this.closed) return;
    const minAt = this.startedAt + this.offsetAtStart;
    for (let j = this.scheduledTo + 1; j < this.chunkCount; j++) {
      const e = this.held.get(j);
      if (!e || e.state !== 'ready') break;
      const tStart = this.startedAt + j * CHUNK_SECONDS;
      const at = Math.max(tStart, minAt, this.ctx.currentTime + 0.01);
      const off = at - tStart;
      this.scheduledTo = j;
      const src = e.bufs.source;
      if (!src || off >= src.duration) continue;
      e.bus = this.ctx.createGain();
      e.bus.connect(this.master);
      e.gain = {};
      e.nodes = [e.bus];
      for (const layer of ['source', 'echo', 'noise', 'rebuild']) {
        const buf = e.bufs[layer];
        if (!buf) continue;
        const g = this.ctx.createGain();
        g.gain.value = layer === 'source' ? 1 : 0;
        const node = this.ctx.createBufferSource();
        node.buffer = buf;
        node.connect(g);
        g.connect(e.bus);
        node.start(at, off);
        e.gain[layer] = g;
        e.nodes.push(node, g);
      }
      this._remix(j, e);
    }
  }

  /** The model changed (a part painted, a layer switched, the cache grew): the gains of what is held follow at once. */
  remix() {
    for (const [k, e] of this.held) { this._remix(k, e); this._topUp(k, e); }
    this._emit();
  }

  /** The gains that apply now, for the page. */
  gainsNow() {
    const model = this.getModel();
    const e = this.held.get(this._chunkAt(this.time));
    const prep = prepare({ ...model, missing: e ? missingOf(e) : {} });
    const f = prep.lufs == null ? 1 : levelFactor(this._refDb(), prep.lufs);
    return describe(prep, this.time, f);
  }

  async play() {
    if (this.playing || this.closed) return;
    this.playing = true; this.error = '';
    const epoch = ++this.epoch;
    await this.ctx.resume();
    if (epoch !== this.epoch) return;
    if (this.offset >= this.duration - 0.01) this.offset = 0;
    await this._begin(epoch);
  }

  async _begin(epoch) {
    const k = this._chunkAt(this.offset);
    this.buffering = true; this._emit();
    this._keepAround(k);
    try { await this._load(k, true); } catch (err) { if (epoch === this.epoch && !this.closed) this._fail(err); return; }
    if (epoch !== this.epoch || this.closed) return;
    this.buffering = false;
    this.offsetAtStart = this.offset;
    this.startedAt = this.ctx.currentTime + 0.05 - this.offset;
    this.scheduledTo = k - 1;
    this.running = true;
    this._fill();
    this._prefetch(k);
    this._emit();
  }

  _fail(err) {
    this._silence(); this.playing = false; this.buffering = false;
    this.error = err?.message || 'Couldn’t get the sound to listen with. Please try again.';
    this.onEnd?.(); this._emit();
  }

  _silence() {
    this.epoch++;
    clearTimeout(this.seekTimer);
    this.running = false;
    for (const e of this.held.values()) this._stopNodes(e);
    this.scheduledTo = -1;
  }

  pause() {
    if (!this.playing) return;
    this.offset = this.time;
    this._silence(); this.playing = false; this.buffering = false; this._emit();
  }

  tick() {
    if (this.closed || !this.playing || !this.running) return;
    const now = this.ctx.currentTime;
    const time = Math.min(this.duration, Math.max(0, now - this.startedAt));
    if (time >= this.duration - 0.02) {
      this._silence(); this.playing = false; this.offset = 0; this.buffering = false; this._emit(); this.onEnd?.();
      return;
    }
    const k = this._chunkAt(time);
    this._keepAround(k); this._prefetch(k); this._fill();
    const scheduledEnd = this.startedAt + (this.scheduledTo + 1) * CHUNK_SECONDS;
    if (this.scheduledTo < this.chunkCount - 1 && now >= scheduledEnd - 0.02) {
      this.offset = Math.min(this.duration, (this.scheduledTo + 1) * CHUNK_SECONDS);
      this._silence();
      this._begin(this.epoch);
    }
  }

  seek(seconds) {
    if (this.closed) return Promise.resolve();
    const was = this.playing;
    this._silence();
    this.offset = Math.max(0, Math.min(this.duration, seconds));
    this.error = '';
    const k = this._chunkAt(this.offset);
    this._keepAround(k);
    if (!was) {
      this.buffering = false; this._emit();
      this.seekTimer = setTimeout(() => this._prefetch(k), 150);
      return Promise.resolve();
    }
    this.buffering = true; this._emit();
    const epoch = this.epoch;
    return new Promise((resolve) => { this.seekTimer = setTimeout(() => { this._begin(epoch).then(resolve, resolve); }, 150); });
  }

  prepare() {
    const k = this._chunkAt(this.offset);
    return this._load(k).then(() => this._prefetch(k), (e) => { this.error = e?.message || 'Couldn’t get the sound to listen with.'; this._emit(); });
  }

  retry() { this.error = ''; this._emit(); return this._load(this._chunkAt(this.offset), true).catch(() => {}); }

  close() {
    if (this.closed) return;
    this._silence(); this.playing = false; this.closed = true;
    clearInterval(this.timer); clearTimeout(this.seekTimer);
    for (const k of [...this.held.keys()]) this._drop(k);
    this.ctx.close?.().catch?.(() => {});
  }
}

/** What a held chunk cannot play, per layer: the ranges its stretch said are silent, and every layer that was not fetched for it (yet). */
function missingOf(e) {
  const out = { ...e.missing };
  for (const layer of ['echo', 'noise', 'rebuild']) if (!e.layers.has(layer)) out[layer] = [[0, 1e9]];
  return out;
}
