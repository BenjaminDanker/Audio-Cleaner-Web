// The waveform of the editor.
//   Before anything is processed: the file's real waveform, decoded cheaply in the browser. The sound is decoded straight to 8 kHz (an
//   OfflineAudioContext at 8000 Hz resamples while decoding: about 1/6 of the work and memory of 48 kHz), then reduced to 2,000 peaks and
//   levels, yielding to the page every few hundred thousand samples so it never stalls. A file the browser's own decoder can't open (or
//   one too big to hold whole) goes through the site's streaming decoder instead (media/decode.js).
//   After processing: what you hear. The server hands out the peaks of each layer (GET /api/jobs/{id}/peaks/{source|echo|noise|rebuild},
//   4 a second) and the page combines them with the gains the mix plays (editor/mix.js weightsAt) for the speaker buttons that are on.

import { levelCoverAt, levelFactor, weightsAt } from './mix.js';

export const PEAK_COUNT = 2000;
const WEB_DECODE_MAX_BYTES = 400 * 1024 ** 2; // a bigger file is read in pieces, never whole
const YIELD_EVERY = 400000; // samples between two looks back at the page

const tick = () => new Promise((r) => setTimeout(r));

/** 2,000 peaks and levels from decoded channels: {dur, N, o (peaks, 0 to 1), m (levels, 0 to 1), d (1 where the sound clips)}. */
async function reduce(chans, len, dur, N = PEAK_COUNT, signal) {
  const pk = new Float32Array(N);
  const ms = new Float32Array(N);
  const clip = new Uint8Array(N);
  let n = 0;
  for (let i = 0; i < N; i++) {
    const a = Math.floor((i * len) / N);
    const b = Math.max(a + 1, Math.floor(((i + 1) * len) / N));
    let m = 0;
    let q = 0;
    chans.forEach((ch) => { for (let j = a; j < b && j < len; j++) { const v = Math.abs(ch[j]); q += v * v; if (v > m) m = v; } });
    pk[i] = m;
    ms[i] = Math.sqrt(q / Math.max(1, (b - a) * chans.length));
    if (m >= 0.985) clip[i] = 1;
    n += b - a;
    if (n > YIELD_EVERY) { n = 0; await tick(); if (signal?.aborted) throw new DOMException('Canceled', 'AbortError'); }
  }
  return normalize({ dur, N, pk, ms, clip });
}

function normalize({ dur, N, pk, ms, clip }) {
  const top = Math.max(1e-6, ...pk);
  const rtop = Math.max(1e-6, ...ms);
  return { dur, N, top, o: Array.from(pk, (v) => v / top), m: Array.from(ms, (v) => (0.8 * v) / rtop), d: Array.from(clip) };
}

/** The peaks of a file by the browser's own decoder (small files). */
async function viaWebAudio(file, signal) {
  const Ctx = window.OfflineAudioContext || window.webkitOfflineAudioContext;
  if (!Ctx) throw new Error('This browser cannot decode audio files.');
  const ab = await new Ctx(1, 1, 8000).decodeAudioData(await file.arrayBuffer());
  if (signal?.aborted) throw new DOMException('Canceled', 'AbortError');
  const chans = [];
  for (let k = 0; k < ab.numberOfChannels; k++) chans.push(ab.getChannelData(k));
  return reduce(chans, ab.length, ab.duration, PEAK_COUNT, signal);
}

/** The peaks of a file by the site's streaming decoder (any type it can read, any length): 20 readings a second, then 2,000 peaks. */
async function viaStream(file, signal) {
  const { decodeToChunks, TARGET_RATE } = await import('../media/decode.js');
  const PER_SEC = 20;
  const step = TARGET_RATE / PER_SEC;
  const peak = [];
  const sq = [];
  const cnt = [];
  let at = 0; // frames seen
  for await (const ev of decodeToChunks(file, { signal })) {
    if (ev.type !== 'chunk') continue;
    const len = ev.chunk[0].length;
    for (let i = 0; i < len; i++) {
      const bin = Math.floor((at + i) / step);
      let m = 0;
      let q = 0;
      for (const c of ev.chunk) { const v = Math.abs(c[i]); if (v > m) m = v; q += v * v; }
      if (bin >= peak.length) { peak[bin] = 0; sq[bin] = 0; cnt[bin] = 0; }
      if (m > peak[bin]) peak[bin] = m;
      sq[bin] += q / ev.chunk.length;
      cnt[bin] += 1;
    }
    at += len;
    if (signal?.aborted) throw new DOMException('Canceled', 'AbortError');
    await tick();
  }
  const dur = at / TARGET_RATE;
  if (!peak.length) throw new Error('No audio could be read from this file.');
  const N = PEAK_COUNT;
  const pk = new Float32Array(N);
  const ms = new Float32Array(N);
  const clip = new Uint8Array(N);
  for (let i = 0; i < N; i++) {
    const a = Math.floor((i * peak.length) / N);
    const b = Math.max(a + 1, Math.floor(((i + 1) * peak.length) / N));
    let m = 0;
    let q = 0;
    let n = 0;
    for (let j = a; j < b && j < peak.length; j++) { if (peak[j] > m) m = peak[j]; q += sq[j]; n += cnt[j]; }
    pk[i] = m;
    ms[i] = Math.sqrt(q / Math.max(1, n));
    if (m >= 0.985) clip[i] = 1;
  }
  return normalize({ dur, N, pk, ms, clip });
}

/**
 * The real waveform of a file. Resolves {dur, N, top, o, m, d}: o the peaks and m the levels (both 0 to 1), d 1 where the sound clips.
 * `info`: what probeMedia found (a video the browser's decoder can't open whole goes the streaming way).
 */
export async function filePeaks(file, { signal, info } = {}) {
  const small = file.size <= WEB_DECODE_MAX_BYTES;
  if (small) {
    try { return await viaWebAudio(file, signal); } catch (e) { if (e?.name === 'AbortError') throw e; /* the streaming decoder reads more types */ }
  }
  if (info && info.hasAudio === false) throw new Error('This file has no audio track.');
  return viaStream(file, signal);
}

/** The peaks of sound made in this browser (the parts of media/pipeline.js: interleaved float32 blobs). Same shape as filePeaks; `top` is the recording's. */
export async function pcmPeaks(pcm, top, N = PEAK_COUNT) {
  const ch = Math.max(1, pcm.channels || 1);
  const frames = Math.max(1, pcm.frames);
  const pk = new Float32Array(N);
  const ms = new Float32Array(N);
  const cnt = new Float64Array(N);
  let at = 0;
  for (const blob of pcm.parts) {
    const a = new Float32Array(await blob.arrayBuffer());
    const n = Math.floor(a.length / ch);
    for (let i = 0; i < n; i++) {
      const bin = Math.min(N - 1, Math.floor(((at + i) * N) / frames));
      for (let c = 0; c < ch; c++) { const v = Math.abs(a[i * ch + c]); if (v > pk[bin]) pk[bin] = v; ms[bin] += v * v; cnt[bin] += 1; }
    }
    at += n;
    await tick();
  }
  const rtop = Math.max(1e-6, ...ms.map((q, i) => Math.sqrt(q / Math.max(1, cnt[i]))));
  return { N, o: Array.from(pk, (v) => v / (top || 1)), m: Array.from(ms, (q, i) => (0.8 * Math.sqrt(q / Math.max(1, cnt[i]))) / rtop) };
}

/* ------------------------------------------------------------------------------------------------ what is heard */

/** A server answer {rate, seconds, peaks: [number|null]} as an array of numbers (NaN where it is not made). */
export function layerPeaks(answer) {
  if (!answer || !Array.isArray(answer.peaks)) return null;
  return { rate: answer.rate || 4, a: Float32Array.from(answer.peaks, (v) => (v == null ? NaN : v)) };
}

const val = (layer, t) => {
  if (!layer) return NaN;
  const i = Math.min(layer.a.length - 1, Math.max(0, Math.floor(t * layer.rate)));
  return layer.a[i];
};

/** The recording's loudness (dBFS, rough: from the peaks) for the loudness match of Leveling. */
export function roughDb(source) {
  if (!source) return NaN;
  let s = 0;
  let n = 0;
  for (const v of source.a) if (v > 0.003) { s += (0.35 * v) ** 2; n++; }
  return n ? 10 * Math.log10(s / n + 1e-12) : NaN;
}

/**
 * What is heard at N points along the recording, as the same scale as the recording's peaks: each layer's peaks times the gain the mix
 * plays it at (editor/mix.js), a layer that is not made there yet falling back on the closest one that is. `prep`: mix.prepare(...) for
 * the timeline and the speaker buttons; `layers`: {source, echo, noise, rebuild} from layerPeaks(); `top`: the recording's highest peak.
 */
export function heardSeries({ N, T, prep, layers, top }) {
  const out = new Float32Array(N);
  const f = prep.lufs == null ? 1 : levelFactor(roughDb(layers.source), prep.lufs);
  for (let k = 0; k < N; k++) {
    const t = Math.min(T - 0.001, ((k + 0.5) / N) * T);
    const w = weightsAt(prep, t);
    const s = val(layers.source, t);
    const pick = (v, ...alt) => { for (const x of [v, ...alt]) if (Number.isFinite(x)) return x; return Number.isFinite(s) ? s : 0; };
    const e = pick(val(layers.echo, t), s);
    const n = pick(val(layers.noise, t), s);
    const r = pick(val(layers.rebuild, t), val(layers.noise, t), s);
    let v = w.source * pick(s) + w.echo * e + w.noise * n + w.rebuild * r;
    const lc = levelCoverAt(prep, t);
    v *= 1 + (f - 1) * lc;
    out[k] = Math.min(1, v / (top || 1));
  }
  return out;
}
