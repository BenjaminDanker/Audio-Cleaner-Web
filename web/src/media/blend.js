// Strength is a blend. The noise remover always runs at full strength; what a person calls "strength" is how much of the
// recording is mixed back into its output: result = (1 - a) * full + a * recording, a = 10 ^ (-dB / 20) (100 dB or more: nothing
// is mixed back). That is exactly what the remover itself does at a limit of `dB` (checked to float precision, server/tests/test_blend.py),
// so with the two signals kept, the strength can be changed afterwards: live while listening (Web Audio, two aligned sources with
// a gain each) and in the saved file (free, on this device).
//
//   full, input: Float32Array[] (one per kept channel, 48 kHz, the same length, sample for sample the same grid)
//   plan: for every channel of the result, the kept channel it is (null: silent), e.g. [0, 0] = both sides are one recording
//
// Listening does not need the whole recording: a source (memorySource: arrays already here; keptSource: a job's two kept copies on
// our side, fetched in short stretches) hands out the same stretch of both signals, and BlendPlayer plays them from a few
// stretches held at a time (the one playing, the next ones fetched ahead, one behind).

import { keptUrl } from '../api.js';
import { STRENGTHS } from '../strengths.js';

export const SAMPLE_RATE = 48000;
/**
 * The most the page holds in memory when it needs a whole recording at once (the free noise removal here keeps its two signals,
 * and a strength is saved on this device from the whole copies): channel-seconds of audio (20 min of mono, 10 of stereo: about 460 MB
 * for both signals). The slider itself holds only a few seconds, whatever the length (BlendPlayer), and longer recordings are
 * saved with a free re-run on our side.
 */
export const PREVIEW_MAX_CHANNEL_SECONDS = 1200;

export const previewFits = (seconds, channels) => seconds * Math.max(1, Math.min(2, channels || 2)) <= PREVIEW_MAX_CHANNEL_SECONDS;

/* ------------------------------------------------------------------------------------------------ the numbers */

/** The share `a` of the recording that stays in the result (0 at 100 dB and above). */
export function mixOf(db) {
  const lim = Math.abs(Number(db));
  return lim >= 100 ? 0 : Math.min(10 ** (-lim / 20), 1);
}

const POINTS = STRENGTHS.map((s) => s.db); // 12, 24, 40, 100: Gentle, Balanced, Strong, Maximum

/** The slider runs over the four names, equally spaced; between two names the strength is the dB in between. Position 0 to 3. */
export function dbAt(pos) {
  const p = Math.max(0, Math.min(POINTS.length - 1, pos));
  const i = Math.min(Math.floor(p), POINTS.length - 2);
  return POINTS[i] + (POINTS[i + 1] - POINTS[i]) * (p - i);
}

export function posOf(db) {
  const d = Math.max(POINTS[0], Math.min(POINTS[POINTS.length - 1], db));
  let i = 0;
  while (i < POINTS.length - 2 && d > POINTS[i + 1]) i++;
  return i + (d - POINTS[i]) / (POINTS[i + 1] - POINTS[i]);
}

/** The name for a strength: a named one, or "Between Balanced and Strong". */
export function strengthName(db) {
  const named = STRENGTHS.find((s) => Math.abs(s.db - db) < 0.5);
  if (named) return named.label;
  const p = posOf(db);
  const i = Math.floor(p);
  return `Between ${STRENGTHS[i].label} and ${STRENGTHS[Math.min(i + 1, STRENGTHS.length - 1)].label}`;
}

/** Sums that give the loudness of any blend in closed form: E(a) = (1 - a)^2 ef + a^2 ei + 2 a (1 - a) cfi (over all channels). */
export function energySums(full, input) {
  let ef = 0;
  let ei = 0;
  let cfi = 0;
  for (let c = 0; c < full.length; c++) {
    const f = full[c];
    const x = input[c];
    for (let i = 0; i < f.length; i++) {
      const a = f[i];
      const b = x[i];
      ef += a * a;
      ei += b * b;
      cfi += a * b;
    }
  }
  return { ef, ei, cfi };
}

/** The gain that gives a blend with share `a` the loudness of the full-strength sound (the preview of a step that is not the last). */
export function matchGain(sums, a) {
  const e = (1 - a) ** 2 * sums.ef + a * a * sums.ei + 2 * a * (1 - a) * sums.cfi;
  if (!(e > 0) || !(sums.ef > 0)) return 1;
  return Math.max(0.1, Math.min(3, Math.sqrt(sums.ef / e)));
}

/** What the two sources play at, as gains: (1 - a) * g for the noise removal and a * g for the recording. */
export function gainsFor(db, { sums = null } = {}) {
  const a = mixOf(db);
  const g = sums ? matchGain(sums, a) : 1;
  return { a, g, full: (1 - a) * g, input: a * g };
}

/* ------------------------------------------------------------------------------------------------ the blended sound */

/** The channels of the blend: [(1 - a) * full + a * input] for each kept channel (float64 arithmetic, stored as float32). */
export function blendChannels(full, input, db) {
  const a = mixOf(db);
  return full.map((f, c) => {
    if (a === 0) return f;
    const x = input[c];
    const out = new Float32Array(f.length);
    for (let i = 0; i < f.length; i++) out[i] = (1 - a) * f[i] + a * x[i];
    return out;
  });
}

const PART_FRAMES = SAMPLE_RATE * 10;

/**
 * The blend as the finished cleaned sound the rest of the page works with ({parts, channels, frames, sampleRate, seconds, peak,
 * rms}: interleaved float32 parts, like media/pipeline.js denoiseFile gives), for the channels of `plan`.
 */
export function blendToPcm(full, input, db, plan = null) {
  const mixed = blendChannels(full, input, db);
  const out = plan || mixed.map((_, i) => i);
  const frames = mixed[0]?.length || 0;
  const channels = out.length;
  const parts = [];
  let peak = 0;
  let sumSq = 0;
  for (let s = 0; s < frames; s += PART_FRAMES) {
    const e = Math.min(frames, s + PART_FRAMES);
    const buf = new Float32Array((e - s) * channels);
    for (let c = 0; c < channels; c++) {
      const src = out[c] == null ? null : mixed[out[c]];
      if (!src) continue;
      for (let i = s, o = c; i < e; i++, o += channels) {
        const v = src[i];
        buf[o] = v;
        const m = v < 0 ? -v : v;
        if (m > peak) peak = m;
        sumSq += v * v;
      }
    }
    parts.push(new Blob([buf], { type: 'application/octet-stream' }));
  }
  return { parts, channels, frames, sampleRate: SAMPLE_RATE, seconds: frames / SAMPLE_RATE, peak, rms: Math.sqrt(sumSq / Math.max(1, frames * channels)), srcChannels: channels, sourceRate: SAMPLE_RATE, elapsed: 0 };
}

/** Interleaved float32 parts (Blobs) -> one Float32Array per channel. */
export async function splitParts(parts, channels) {
  const frames = parts.reduce((n, p) => n + p.size, 0) / 4 / channels;
  const out = Array.from({ length: channels }, () => new Float32Array(frames));
  let pos = 0;
  for (const part of parts) {
    const f = new Float32Array(await part.arrayBuffer());
    const n = f.length / channels;
    for (let c = 0; c < channels; c++) for (let i = 0, o = c; i < n; i++, o += channels) out[c][pos + i] = f[o];
    pos += n;
  }
  return out;
}

/* ------------------------------------------------------------------------------------------------ the two copies of a job on our side */

let decoder = null;
export function decodeAudio(bytes) {
  const Ctx = window.OfflineAudioContext || window.webkitOfflineAudioContext;
  if (!Ctx) return Promise.reject(new Error('This browser can’t play the strengths side by side.'));
  decoder ||= new Ctx(1, 1, SAMPLE_RATE);
  return decoder.decodeAudioData(bytes).catch(() => { throw new Error('This browser can’t open the sound needed to try other strengths.'); });
}

async function fetchBytes(url, { signal, onBytes }) {
  const res = await fetch(url, { credentials: 'same-origin', signal });
  if (!res.ok) throw new Error('Couldn’t get the sound to try other strengths with. Please try again.');
  if (!res.body || !onBytes) return res.arrayBuffer();
  const reader = res.body.getReader();
  const chunks = [];
  let total = 0;
  for (;;) {
    const { done, value } = await reader.read();
    if (done) break;
    chunks.push(value);
    total += value.length;
    onBytes(total);
  }
  const all = new Uint8Array(total);
  let o = 0;
  for (const c of chunks) { all.set(c, o); o += c.length; }
  return all.buffer;
}

/** Decoded audio -> one Float32Array of exactly `samples` per kept channel, brought back up by `scale` (a loud sample was brought down when kept, acw_common/kept.py). */
export function channelsOf(decoded, { samples, channels, scale }) {
  const inv = 1 / scale;
  const out = [];
  for (let c = 0; c < channels; c++) {
    const src = decoded.getChannelData(Math.min(c, decoded.numberOfChannels - 1));
    if (src.length === samples) { // the decoder's own array: scaled where it is
      if (inv !== 1) for (let i = 0; i < samples; i++) src[i] *= inv;
      out.push(src);
      continue;
    }
    const a = new Float32Array(samples); // a decoder may pad or trim the last block
    const n = Math.min(samples, src.length);
    if (inv === 1) a.set(src.subarray(0, n));
    else for (let i = 0; i < n; i++) a[i] = src[i] * inv;
    out.push(a);
  }
  return out;
}

async function decodeKept(bytes, spec) {
  return channelsOf(await decodeAudio(bytes), spec);
}

/**
 * The two copies a job kept for the slider, whole and decoded: GET /api/jobs/{id}/kept/noise and /input (24-bit FLAC, sample for sample
 * the same grid). `preview` is the job's redo.preview. Resolves {full, input, plan}. For recordings that fit in memory (previewFits):
 * saving a strength on this device blends these; listening uses keptSource's stretches instead.
 */
export async function loadKeptPair(jobId, preview, { signal, onProgress } = {}) {
  const total = (preview.bytes?.noise || 0) + (preview.bytes?.input || 0);
  const got = { noise: 0, input: 0 };
  const report = () => onProgress?.(total ? Math.min(1, (got.noise + got.input) / total) : 0);
  const [a, b] = await Promise.all(['noise', 'input'].map((which) => fetchBytes(keptUrl(jobId, which), { signal, onBytes: (n) => { got[which] = n; report(); } })));
  const spec = (which) => ({ samples: preview.samples, channels: preview.channels.length, scale: preview.scale[which] });
  const full = await decodeKept(a, spec('noise'));
  const input = await decodeKept(b, spec('input'));
  onProgress?.(1);
  // the kept channels are numbered as the channels of the recording ([1] alone: only the right side); the plan counts from the first kept one
  return { full, input, plan: keptPlan(preview) };
}

const keptPlan = (preview) => preview.plan.map((c) => (c == null ? null : c - preview.channels[0]));

/** One stretch of a kept copy: `count` samples from sample `first`, decoded (GET .../kept/{which}?start=&seconds=, exactly those samples). */
async function fetchStretch(jobId, which, { first, count }, scale, channels, signal) {
  const res = await fetch(keptUrl(jobId, which, { first, count }), { credentials: 'same-origin', signal });
  if (!res.ok) throw new Error('Couldn’t get the sound to try other strengths with. Please try again.');
  const bytes = await res.arrayBuffer();
  const start = Number(res.headers.get('X-Stretch-Start'));
  const got = Number(res.headers.get('X-Stretch-Samples'));
  if (res.headers.has('X-Stretch-Start') && (start !== first || got !== count)) throw new Error('Got the wrong part of the sound. Please try again.');
  return { channels: channelsOf(await decodeAudio(bytes), { samples: count, channels, scale }), bytes: bytes.byteLength };
}

/* ------------------------------------------------------------------------------------------------ where the player's signals come from */

/**
 * A source of the two signals for BlendPlayer: {samples, plan, kind, maxSeconds, sums(), read(first, count, signal)}.
 * read resolves {full, input, bytes}: Float32Array[] per kept channel, `count` samples (fewer at the very end) from sample `first`.
 * sums(): the loudness sums of the whole recording (energySums), or null when not known.
 */
export function memorySource(pair) {
  let sums = null;
  return {
    kind: 'memory',
    samples: pair.full[0]?.length || 0,
    plan: pair.plan,
    maxSeconds: Infinity,
    sums: () => (sums ||= energySums(pair.full, pair.input)),
    read: async (first, count) => ({ full: pair.full.map((a) => a.subarray(first, first + count)), input: pair.input.map((a) => a.subarray(first, first + count)), bytes: 0 }),
  };
}

/** A job's two kept copies on our side, fetched in stretches: only a few seconds of either are ever here. `preview` is the job's redo.preview. */
export function keptSource(jobId, preview) {
  const channels = preview.channels.length;
  const e = preview.energy;
  const sums = e && Number.isFinite(e.full) && Number.isFinite(e.input) && Number.isFinite(e.cross) ? { ef: e.full, ei: e.input, cfi: e.cross } : null;
  return {
    kind: 'server',
    samples: preview.samples,
    plan: keptPlan(preview),
    maxSeconds: preview.stretchMaxSeconds || 30,
    sums: () => sums,
    async read(first, count, signal) {
      const [a, b] = await Promise.all(['noise', 'input'].map((which) => fetchStretch(jobId, which, { first, count }, preview.scale[which], channels, signal)));
      return { full: a.channels, input: b.channels, bytes: a.bytes + b.bytes };
    },
  };
}

/* ------------------------------------------------------------------------------------------------ playing the blend */

export const CHUNK_SECONDS = 10; // one stretch
export const CHUNKS_AHEAD = 2; // fetched before they are needed, after the one that plays
export const CHUNKS_BEHIND = 1; // kept after they played, for a step back
const RETRY_MS = 2000; // a stretch that failed to come is asked for again after this, when it is needed or ahead of time

/**
 * Plays the two signals together, started at the same instant with a gain each, so a change of strength is a change of two gains
 * and is heard at once, without a gap. The signals come from a `source` in stretches of CHUNK_SECONDS: the stretch around the play
 * position and the next CHUNKS_AHEAD are held (fetched ahead, so the sound goes on without a gap), one behind it, and nothing else, so the
 * memory used does not depend on the length of the recording. A seek that leaves the held stretches drops them and fetches the new ones
 * (after a short pause, so dragging the range does not ask for every position on the way). If the next stretch is late the sound waits
 * for it ("buffering") instead of playing on into silence. `channels` buffers: a recording whose sides are the same plays as one channel.
 *
 * options: onEnd(), onState(snapshot) (buffering / error / what is held, whenever it changes), chunkSeconds, tickMs (0: no timer, call
 * tick() yourself), seekDelayMs, context (a function that makes the AudioContext: for tests).
 */
export class BlendPlayer {
  constructor(source, { onEnd, onState, chunkSeconds = CHUNK_SECONDS, tickMs = 100, seekDelayMs = 150, context = null } = {}) {
    this.ctx = context ? context() : new (window.AudioContext || window.webkitAudioContext)({ sampleRate: SAMPLE_RATE });
    this.source = source;
    this.chunkSeconds = Math.max(0.5, Math.min(chunkSeconds, source.maxSeconds || chunkSeconds));
    this.chunkFrames = Math.round(this.chunkSeconds * SAMPLE_RATE);
    this.chunkCount = Math.max(1, Math.ceil(source.samples / this.chunkFrames));
    this.duration = source.samples / SAMPLE_RATE;
    const plan = source.plan;
    this.same = plan.length === 2 && plan[0] === 0 && plan[1] === 0; // both sides are one recording: one channel plays
    this.gains = { full: this.ctx.createGain(), input: this.ctx.createGain() };
    this.gains.full.connect(this.ctx.destination);
    this.gains.input.connect(this.ctx.destination);
    this.held = new Map(); // chunk index -> {state: 'loading' | 'ready' | 'failed', promise, ctl, buffers, nodes, retryAt}
    this.offset = 0; // seconds into the recording where it is paused / started
    this.startedAt = 0; // context time that second 0 would have been at
    this.offsetAtStart = 0; // where the sound was started from (nothing is scheduled before it)
    this.running = false; // sources are scheduled
    this.scheduledTo = -1; // the last chunk handed to the context
    this.epoch = 0; // a pause, seek or stall makes older async steps stop
    this.playing = false; // the person wants sound (also while it waits for a stretch)
    this.buffering = false;
    this.error = '';
    this.requests = 0;
    this.bytesLoaded = 0;
    this.onEnd = onEnd;
    this.onState = onState;
    this.seekDelayMs = seekDelayMs;
    this.target = { full: 1, input: 0 };
    this.closed = false;
    this.timer = tickMs ? setInterval(() => this.tick(), tickMs) : null;
  }

  /** What is going on, for the page: whether it waits for sound, a failure, and what is held in memory. */
  snapshot() {
    let ready = 0;
    for (const e of this.held.values()) if (e.state === 'ready') ready++;
    return { buffering: this.buffering, error: this.error, held: ready, requests: this.requests, heldSeconds: ready * this.chunkSeconds, chunkSeconds: this.chunkSeconds };
  }

  _emit() { if (!this.closed) this.onState?.(this.snapshot()); }

  /** Set the two gains (a short ramp, so there is no click). */
  setGains(full, input) {
    this.target = { full, input };
    const t = this.ctx.currentTime;
    this.gains.full.gain.setTargetAtTime(full, t, 0.015);
    this.gains.input.gain.setTargetAtTime(input, t, 0.015);
  }

  get time() {
    if (!this.playing || !this.running) return this.offset;
    return Math.min(this.duration, Math.max(this.offsetAtStart, this.ctx.currentTime - this.startedAt)); // (not before where it started: the head start)
  }

  _chunkAt(seconds) { return Math.max(0, Math.min(this.chunkCount - 1, Math.floor(seconds / this.chunkSeconds))); }

  _span(k) {
    const first = k * this.chunkFrames;
    return { first, count: Math.min(this.chunkFrames, this.source.samples - first) };
  }

  /** The stretch k (asked for if it is not held): resolves its entry when it is ready. `force`: ask again even if it failed a moment ago. */
  _load(k, force = false) {
    let e = this.held.get(k);
    if (e && (e.state !== 'failed' || (!force && Date.now() < e.retryAt))) return e.promise;
    if (e) this._drop(k);
    const { first, count } = this._span(k);
    const ctl = new AbortController();
    e = { state: 'loading', ctl, buffers: null, nodes: [], retryAt: 0, promise: null };
    this.held.set(k, e);
    this.requests++;
    e.promise = this.source.read(first, count, ctl.signal).then(
      (got) => {
        if (this.closed || this.held.get(k) !== e) return e;
        e.buffers = { full: this._buffer(got.full, count), input: this._buffer(got.input, count) };
        e.state = 'ready';
        this.bytesLoaded += got.bytes || 0;
        this._emit();
        if (this.playing && this.running && !this.buffering) this._fill();
        return e;
      },
      (err) => {
        if (this.closed || this.held.get(k) !== e) return e;
        e.state = 'failed';
        e.retryAt = Date.now() + RETRY_MS;
        e.error = err;
        this._emit();
        throw err;
      },
    );
    e.promise.catch(() => {}); // a stretch fetched ahead that fails is asked for again when it is needed (or after RETRY_MS)
    this._emit();
    return e.promise;
  }

  _buffer(arrays, count) {
    const frames = Math.max(1, Math.min(count, arrays[0]?.length || 0));
    const use = this.same || this.source.plan.length === 1 ? [arrays[0]] : this.source.plan.map((c) => (c == null ? null : arrays[c]));
    const buf = this.ctx.createBuffer(use.length, frames, SAMPLE_RATE);
    use.forEach((a, c) => { if (a) buf.copyToChannel(a.length === frames ? a : a.subarray(0, frames), c); });
    return buf;
  }

  _drop(k) {
    const e = this.held.get(k);
    if (!e) return;
    this.held.delete(k);
    e.ctl.abort();
    for (const n of e.nodes) { n.onended = null; try { n.stop(); } catch { /* not started */ } try { n.disconnect(); } catch { /* gone */ } }
    e.nodes = [];
    e.buffers = null;
  }

  /** Keep the stretches around `k` (CHUNKS_BEHIND before, CHUNKS_AHEAD after) and let go of the rest. */
  _keepAround(k) {
    let changed = false;
    for (const j of [...this.held.keys()]) {
      if (j < k - CHUNKS_BEHIND || j > k + CHUNKS_AHEAD) { this._drop(j); changed = true; }
    }
    if (changed) this._emit();
  }

  /** Ask for the stretches ahead of `k` that are not held. */
  _prefetch(k) {
    for (let j = k; j <= Math.min(this.chunkCount - 1, k + CHUNKS_AHEAD); j++) this._load(j).catch(() => {});
  }

  /** Hand every held, ready stretch after the last scheduled one to the context, one right after the other. */
  _fill() {
    if (!this.running || this.closed) return;
    const minAt = this.startedAt + this.offsetAtStart;
    for (let j = this.scheduledTo + 1; j < this.chunkCount; j++) {
      const e = this.held.get(j);
      if (!e || e.state !== 'ready') break;
      const tStart = this.startedAt + j * this.chunkSeconds;
      const now = this.ctx.currentTime + 0.01;
      const at = Math.max(tStart, minAt, now);
      const off = at - tStart;
      this.scheduledTo = j;
      if (off >= e.buffers.full.duration) continue; // already past
      e.nodes = ['full', 'input'].map((key) => {
        const node = this.ctx.createBufferSource();
        node.buffer = e.buffers[key];
        node.connect(this.gains[key]);
        node.start(at, off);
        return node;
      });
    }
  }

  async play() {
    if (this.playing || this.closed) return;
    this.playing = true;
    this.error = '';
    const epoch = ++this.epoch;
    await this.ctx.resume();
    if (epoch !== this.epoch) return;
    if (this.offset >= this.duration - 0.01) this.offset = 0;
    await this._begin(epoch);
  }

  /** Start the sound at this.offset: wait for the stretch there, then schedule it and the ones after it. */
  async _begin(epoch) {
    const k = this._chunkAt(this.offset);
    this.buffering = true;
    this._emit();
    this._keepAround(k);
    try {
      await this._load(k, true);
    } catch (e) {
      if (epoch === this.epoch && !this.closed) this._fail(e);
      return;
    }
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

  _fail(e) {
    this._silence();
    this.playing = false;
    this.buffering = false;
    this.error = e?.message || 'Couldn’t get the sound to try other strengths with. Please try again.';
    this.onEnd?.();
    this._emit();
  }

  /** Stop every scheduled source and forget the schedule (the stretches stay held). */
  _silence() {
    this.epoch++;
    clearTimeout(this.seekTimer);
    this.running = false;
    for (const e of this.held.values()) {
      for (const n of e.nodes) { n.onended = null; try { n.stop(); } catch { /* not started */ } try { n.disconnect(); } catch { /* gone */ } }
      e.nodes = [];
    }
    this.scheduledTo = -1;
  }

  pause() {
    if (!this.playing) return;
    this.offset = this.time;
    this._silence();
    this.playing = false;
    this.buffering = false;
    this._emit();
  }

  /** Every 100 ms while it plays: the end, a stretch that is late, what to fetch next and what to let go. */
  tick() {
    if (this.closed || !this.playing || !this.running) return;
    const now = this.ctx.currentTime;
    const time = Math.min(this.duration, Math.max(0, now - this.startedAt));
    if (time >= this.duration - 0.02) {
      this._silence();
      this.playing = false;
      this.offset = 0;
      this.buffering = false;
      this._emit();
      this.onEnd?.();
      return;
    }
    const k = this._chunkAt(time);
    this._keepAround(k);
    this._prefetch(k);
    this._fill();
    // the sound that was handed over has all played and the next stretch has not come: wait for it instead of playing on into silence
    const scheduledEnd = this.startedAt + (this.scheduledTo + 1) * this.chunkSeconds;
    if (this.scheduledTo < this.chunkCount - 1 && now >= scheduledEnd - 0.02) {
      this.offset = Math.min(this.duration, (this.scheduledTo + 1) * this.chunkSeconds);
      this._silence();
      this._begin(this.epoch);
    }
  }

  /** Go to `seconds`. While it plays the sound starts again from there after a short pause (the stretches there are fetched then). */
  seek(seconds) {
    if (this.closed) return Promise.resolve();
    const was = this.playing;
    this._silence();
    this.offset = Math.max(0, Math.min(this.duration, seconds));
    this.error = '';
    const k = this._chunkAt(this.offset);
    this._keepAround(k);
    if (!was) {
      this.buffering = false;
      this._emit();
      this.seekTimer = setTimeout(() => this._prefetch(k), this.seekDelayMs); // so Play starts at once from there
      return Promise.resolve();
    }
    this.buffering = true;
    this._emit();
    const epoch = this.epoch;
    return new Promise((resolve) => {
      this.seekTimer = setTimeout(() => { this._begin(epoch).then(resolve, resolve); }, this.seekDelayMs);
    });
  }

  /** Get the first stretch ready before Play is pressed (the sound starts at once then). */
  prepare(force = false) {
    const k = this._chunkAt(this.offset);
    return this._load(k, force).then(() => this._prefetch(k), (e) => { this.error = e?.message || 'Couldn’t get the sound to try other strengths with.'; this._emit(); });
  }

  /** After a failure: try again from where it was. */
  retry() {
    this.error = '';
    this._emit();
    return this.prepare(true);
  }

  close() {
    if (this.closed) return;
    this._silence();
    this.playing = false;
    this.closed = true;
    clearInterval(this.timer);
    clearTimeout(this.seekTimer);
    for (const k of [...this.held.keys()]) this._drop(k);
    this.ctx.close?.().catch?.(() => {});
  }
}
