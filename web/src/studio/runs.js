// What the browser does itself for the Free parts of an edit: clean a range of the recording, give it back as FLAC to upload, transcribe a
// range, or put the cleaned ranges back into the whole recording (a run with nothing on our side). The pipeline (media/pipeline.js,
// media/speech.js) is the one the free run always used; a part is read with a few seconds of lead-in so the remover has settled when it starts.

import { decodeToChunks, TARGET_RATE } from '../media/decode.js';
import { encodeFlac, sniffPcm } from '../media/flac.js';
import { denoiseFile, PREROLL_SECONDS } from '../media/pipeline.js';
import { transcribeFile } from '../media/speech.js';
import { buildWav, interleave } from '../media/wav.js';

const EPS = 0.0005;
const ms = (t) => Math.round(t * 1000) / 1000;
const tick = () => new Promise((r) => setTimeout(r));

/**
 * A range as the decoder takes it: a part that starts at the beginning, or ends within half a second of the end of the recording, is
 * read from the very beginning / to the very end (the length the browser reports may differ a little from the sound itself).
 */
export function rangeFor(range, T) {
  return { start: range.start <= EPS ? 0 : range.start, end: T && range.end >= T - 0.6 ? Infinity : range.end };
}

/** The Free pieces (each {start, end, db}) inside [start, end]. */
export function piecesIn(freeRanges, start, end) {
  return freeRanges
    .map((r) => ({ start: Math.max(r.start, start), end: Math.min(r.end, end), db: r.db }))
    .filter((p) => p.end - p.start > EPS);
}

/**
 * Clean a range in the browser: each piece of it at its own strength. Resolves a pcm like denoiseFile's ({parts, channels, frames, ...}) that
 * is just as long as the range. `onProgress({done, total, rate, eta, elapsed})` counts the seconds of the range.
 */
export async function cleanRange(file, range, freeRanges, { signal, onProgress, onStage, keepPair = false, T = 0 } = {}) {
  const pieces = piecesIn(freeRanges, range.start, range.end);
  if (!pieces.length) throw new Error('Nothing to clean in this range.');
  const total = pieces.reduce((n, p) => n + (p.end - p.start), 0);
  let before = 0;
  const outs = [];
  for (const p of pieces) {
    const len = p.end - p.start;
    const pcm = await denoiseFile(file, {
      attenDb: p.db, range: rangeFor(p, T), durationHint: len + Math.min(PREROLL_SECONDS, p.start), signal, onStage, keepPair,
      onProgress: onProgress && ((q) => onProgress({ ...q, done: before + Math.max(0, Math.min(len, q.done - Math.min(PREROLL_SECONDS, p.start))), total })),
    });
    before += len;
    outs.push(pcm);
  }
  if (outs.length === 1) return outs[0];
  const first = outs[0];
  return {
    ...first,
    parts: outs.flatMap((o) => o.parts), frames: outs.reduce((n, o) => n + o.frames, 0), seconds: outs.reduce((n, o) => n + o.seconds, 0),
    peak: Math.max(...outs.map((o) => o.peak)), rms: Math.sqrt(outs.reduce((n, o) => n + o.rms * o.rms * o.frames, 0) / Math.max(1, outs.reduce((n, o) => n + o.frames, 0))),
    pair: null,
  };
}

/** The pcm made exactly `frames` long: cut at the end, or padded with silence (the server takes exactly the samples of the part). */
export function fitPcm(pcm, frames) {
  if (pcm.frames === frames) return pcm;
  const bytesPer = 4 * pcm.channels;
  if (pcm.frames > frames) {
    const parts = [];
    let left = frames;
    for (const b of pcm.parts) {
      if (left <= 0) break;
      const n = b.size / bytesPer;
      if (n <= left) { parts.push(b); left -= n; } else { parts.push(b.slice(0, left * bytesPer)); left = 0; }
    }
    return { ...pcm, parts, frames, seconds: frames / TARGET_RATE };
  }
  return { ...pcm, parts: [...pcm.parts, new Blob([new Float32Array((frames - pcm.frames) * pcm.channels)])], frames, seconds: frames / TARGET_RATE };
}

/** How many samples (48 kHz) the server takes for the part [start, end]: exactly the part's range. */
export const samplesOf = (start, end) => Math.round(end * TARGET_RATE) - Math.round(start * TARGET_RATE);

/** A cleaned sound as the FLAC the server takes for a browser part: 48 kHz, 24-bit, as many channels as the sound has. */
export async function pcmToFlac(pcm, { signal } = {}) {
  const wav = await buildWav(pcm, 24);
  const info = await sniffPcm(wav);
  if (!info) throw new Error('Couldn’t pack the cleaned sound for upload.');
  return encodeFlac(wav, info, { signal });
}

/** The words of a range, written in the browser: [{start, end, text}] in seconds from the start of the recording, and the language heard. */
export async function transcribeRange(file, range, language, { signal, onProgress, onStage, T = 0 } = {}) {
  const r = await transcribeFile(file, { language: language || null, durationHint: range.end - range.start, signal, onProgress, onStage, range: rangeFor(range, T) });
  return { cues: r.segments.map((s) => ({ start: ms(s.start), end: ms(s.end), text: s.text })), language: r.language, segments: r.segments };
}

/** All the parts of a pcm as one interleaved Float32Array. */
async function readAll(pcm) {
  const out = new Float32Array(pcm.frames * pcm.channels);
  let o = 0;
  for (const part of pcm.parts) {
    const a = new Float32Array(await part.arrayBuffer());
    out.set(a.subarray(0, Math.min(a.length, out.length - o)), o);
    o += a.length;
    await tick();
  }
  return out;
}

/**
 * The whole recording with the cleaned pieces put back in (a run with nothing on our side): the recording is read again and every piece
 * replaces its stretch. `pieces`: [{start, end, pcm}]. Resolves a pcm like denoiseFile's, as long as the recording.
 */
export async function splicePcm(file, pieces, { signal, onProgress } = {}) {
  const loaded = [];
  for (const p of pieces) loaded.push({ at: Math.round(p.start * TARGET_RATE), n: p.pcm.frames, ch: p.pcm.channels, data: await readAll(p.pcm) });
  const parts = [];
  let channels = 1;
  let srcChannels = 1;
  let sourceRate = TARGET_RATE;
  let frames = 0;
  let peak = 0;
  let sumSq = 0;
  const started = performance.now();
  for await (const ev of decodeToChunks(file, { signal })) {
    if (signal?.aborted) throw new DOMException('Canceled', 'AbortError');
    if (ev.type === 'info') { channels = ev.channels; srcChannels = ev.srcChannels; sourceRate = ev.sourceRate; continue; }
    const n = ev.chunk[0].length;
    const out = ev.chunk.map((c) => c.slice());
    for (const l of loaded) {
      const a = Math.max(frames, l.at);
      const b = Math.min(frames + n, l.at + l.n);
      for (let i = a; i < b; i++) for (let c = 0; c < channels; c++) out[c][i - frames] = l.data[(i - l.at) * l.ch + Math.min(c, l.ch - 1)];
    }
    for (const ch of out) for (let i = 0; i < n; i++) { const v = Math.abs(ch[i]); if (v > peak) peak = v; sumSq += ch[i] * ch[i]; }
    parts.push(new Blob([interleave(out)], { type: 'application/octet-stream' }));
    frames += n;
    onProgress?.({ done: frames / TARGET_RATE, total: null, rate: 0, eta: null, elapsed: (performance.now() - started) / 1000 });
  }
  if (!frames) throw new Error('No audio could be read from this file.');
  return {
    parts, channels, frames, sampleRate: TARGET_RATE, peak, rms: Math.sqrt(sumSq / Math.max(1, frames * channels)), seconds: frames / TARGET_RATE,
    srcChannels, sourceRate, elapsed: (performance.now() - started) / 1000, pair: null,
  };
}

/** The subtitle files' segments of several ranges, in time order (a range's own words may touch the next: they are not merged). */
export const joinSegments = (lists) => lists.flat().sort((a, b) => a.start - b.start);
