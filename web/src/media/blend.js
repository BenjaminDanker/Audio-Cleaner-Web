// Strength is a blend. The noise remover always runs at full strength; what a person calls "strength" is how much of the
// recording is mixed back into its output: result = (1 - a) * full + a * recording, a = 10 ^ (-dB / 20) (100 dB or more: nothing
// is mixed back). That is exactly what the remover itself does at a limit of `dB` (checked to float precision, server/tests/test_blend.py),
// so with the two signals kept, the strength can be changed afterwards: live while listening (Web Audio, two aligned sources with
// a gain each) and in the saved file (free, on this device).
//
//   full, input: Float32Array[] (one per kept channel, 48 kHz, the same length, sample for sample the same grid)
//   plan: for every channel of the result, the kept channel it is (null: silent), e.g. [0, 0] = both sides are one recording

import { keptUrl } from '../api.js';
import { STRENGTHS } from '../strengths.js';

export const SAMPLE_RATE = 48000;
/** The most the page holds in memory for the slider: channel-seconds of audio (20 min of mono, 10 of stereo: about 460 MB for both signals). */
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

async function decodeKept(bytes, { samples, channels, scale }) {
  const Ctx = window.OfflineAudioContext || window.webkitOfflineAudioContext;
  if (!Ctx) throw new Error('This browser can’t play the strengths side by side.');
  let decoded;
  try {
    decoded = await new Ctx(1, 1, SAMPLE_RATE).decodeAudioData(bytes);
  } catch {
    throw new Error('This browser can’t open the sound needed to try other strengths.');
  }
  const inv = 1 / scale; // a loud sample was brought down by `scale` when it was kept (acw_common/kept.py)
  const out = [];
  for (let c = 0; c < channels; c++) {
    const src = decoded.getChannelData(Math.min(c, decoded.numberOfChannels - 1));
    const a = new Float32Array(samples); // exactly the length of the job (a decoder may pad or trim the last block)
    const n = Math.min(samples, src.length);
    if (inv === 1) a.set(src.subarray(0, n));
    else for (let i = 0; i < n; i++) a[i] = src[i] * inv;
    out.push(a);
  }
  return out;
}

/**
 * The two copies a job kept for the slider, decoded: GET /api/jobs/{id}/kept/noise and /input (24-bit FLAC, sample for sample
 * the same grid). `preview` is the job's redo.preview. Resolves {full, input, plan}.
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
  const first = preview.channels[0];
  return { full, input, plan: preview.plan.map((c) => (c == null ? null : c - first)) };
}

/* ------------------------------------------------------------------------------------------------ playing the blend */

/**
 * Plays the two signals together, started at the same instant with a gain each, so a change of strength is a change of two gains
 * and is heard at once, without a gap. `channels` buffers: a recording whose sides are the same plays as one channel.
 */
export class BlendPlayer {
  constructor(pair, { onEnd } = {}) {
    const Ctx = window.AudioContext || window.webkitAudioContext;
    this.ctx = new Ctx({ sampleRate: SAMPLE_RATE });
    const make = (arrays) => {
      const same = pair.plan.length === 2 && pair.plan[0] === 0 && pair.plan[1] === 0;
      const use = same || pair.plan.length === 1 ? [arrays[0]] : pair.plan.map((c) => (c == null ? new Float32Array(arrays[0].length) : arrays[c]));
      const buf = this.ctx.createBuffer(use.length, use[0].length, SAMPLE_RATE);
      use.forEach((a, c) => buf.copyToChannel(a, c));
      return buf;
    };
    this.buffers = { full: make(pair.full), input: make(pair.input) };
    this.duration = this.buffers.full.duration;
    this.gains = { full: this.ctx.createGain(), input: this.ctx.createGain() };
    this.gains.full.connect(this.ctx.destination);
    this.gains.input.connect(this.ctx.destination);
    this.sources = null;
    this.offset = 0; // seconds into the recording where it is paused / started
    this.startedAt = 0; // context time that second 0 would have been at
    this.onEnd = onEnd;
    this.playing = false;
    this.target = { full: 1, input: 0 };
  }

  /** Set the two gains (a short ramp, so there is no click). */
  setGains(full, input) {
    this.target = { full, input };
    const t = this.ctx.currentTime;
    this.gains.full.gain.setTargetAtTime(full, t, 0.015);
    this.gains.input.gain.setTargetAtTime(input, t, 0.015);
  }

  get time() {
    if (!this.playing) return this.offset;
    return Math.min(this.duration, Math.max(0, this.ctx.currentTime - this.startedAt));
  }

  async play() {
    if (this.playing) return;
    await this.ctx.resume();
    if (this.offset >= this.duration - 0.01) this.offset = 0;
    const when = this.ctx.currentTime + 0.03;
    const make = (key) => {
      const s = this.ctx.createBufferSource();
      s.buffer = this.buffers[key];
      s.connect(this.gains[key]);
      s.start(when, this.offset);
      return s;
    };
    this.sources = { full: make('full'), input: make('input') };
    this.sources.full.onended = () => {
      if (this.sources && this.playing && this.time >= this.duration - 0.05) {
        this.playing = false;
        this.offset = 0;
        this.sources = null;
        this.onEnd?.();
      }
    };
    this.startedAt = when - this.offset;
    this.playing = true;
  }

  pause() {
    if (!this.playing) return;
    this.offset = this.time;
    this.playing = false;
    const s = this.sources;
    this.sources = null;
    if (s) for (const n of Object.values(s)) { n.onended = null; try { n.stop(); } catch { /* already stopped */ } }
  }

  async seek(seconds) {
    const was = this.playing;
    if (was) this.pause();
    this.offset = Math.max(0, Math.min(this.duration, seconds));
    if (was) await this.play();
  }

  close() {
    this.pause();
    this.ctx.close().catch(() => {});
  }
}
