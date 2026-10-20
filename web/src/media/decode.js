// Decode any audio/video file to 48 kHz float32 PCM chunks (mono or stereo), streaming.
//   1. mediabunny (WebCodecs): streams MP4/MOV/WebM/MKV/MP3/WAV/OGG/FLAC/AAC, never loads the whole file.
//   2. libav.js (FFmpeg in WebAssembly, lazy-loaded): WMA, AVI, FLV, MPEG, AC-3, DTS, AMR, WavPack, ... streamed.
//   3. Last resort: AudioContext.decodeAudioData + OfflineAudioContext (whole file in memory).

import { libavAvailable, libavFrames } from './libav.js';
import { openInput } from './mb.js';
import { Resampler } from './resample.js';

export const TARGET_RATE = 48000;
export const CHUNK_FRAMES = 48000; // ~1 s

export class DecodeError extends Error {}

/** Collects arbitrary-length per-channel samples and emits fixed-size chunks. */
class Chunker {
  constructor(channels, size = CHUNK_FRAMES) {
    this.channels = channels;
    this.size = size;
    this.acc = Array.from({ length: channels }, () => []);
    this.len = 0;
  }

  push(perChannel) {
    for (let c = 0; c < this.channels; c++) this.acc[c].push(perChannel[c]);
    this.len += perChannel[0].length;
  }

  *drain(final = false) {
    while (this.len >= this.size || (final && this.len > 0)) {
      const n = Math.min(this.size, this.len);
      const out = [];
      for (let c = 0; c < this.channels; c++) out.push(take(this.acc[c], n));
      this.len -= n;
      yield out;
    }
  }
}

function take(list, n) {
  const out = new Float32Array(n);
  let o = 0;
  while (o < n) {
    const head = list[0];
    const need = n - o;
    if (head.length <= need) {
      out.set(head, o);
      o += head.length;
      list.shift();
    } else {
      out.set(head.subarray(0, need), o);
      list[0] = head.subarray(need);
      o = n;
    }
  }
  return out;
}

async function* viaMediabunny(file, { signal, from = 0, to = Infinity }) {
  const { mb, input } = await openInput(file);
  try {
    const track = await input.getPrimaryAudioTrack();
    if (!track) throw new DecodeError('This file has no audio track.');
    if (!(await track.canDecode())) throw new DecodeError('This browser cannot decode the audio format of this file.');
    const srcChannels = await track.getNumberOfChannels();
    const channels = srcChannels >= 2 ? (srcChannels === 2 ? 2 : 1) : 1;
    let resamplers = null;
    let rate = 0;
    const chunker = new Chunker(channels);
    const sink = new mb.AudioSampleSink(track);
    const firstRate = await track.getSampleRate();
    yield { type: 'info', channels, sourceRate: firstRate, srcChannels, base: from };
    // Respect sample timestamps: containers carry encoder priming (AAC: 1024 frames, MP3, Opus) as negative start
    // times or edit lists, and audio may start after the video. Trim negative time, pad gaps with silence, so the
    // cleaned audio stays in sync with the video.
    let pos = Math.round(from * firstRate); // source frames emitted so far (the timeline starts at t = 0; a range starts at `from`)
    for await (const sample of from > 0 || to < Infinity ? sink.samples(from, to) : sink.samples()) {
      if (signal?.aborted) throw new DOMException('Canceled', 'AbortError');
      try {
        if (!resamplers || rate !== sample.sampleRate) {
          if (resamplers) chunker.push(resamplers.map((r) => r.finish().slice()));
          rate = sample.sampleRate;
          resamplers = Array.from({ length: channels }, () => new Resampler(rate, TARGET_RATE));
          pos = Math.max(pos, 0);
        }
        const frames = sample.numberOfFrames;
        const startFrame = Math.round(sample.timestamp * rate);
        let skip = 0;
        if (startFrame < pos) skip = Math.min(frames, pos - startFrame); // overlap or negative (priming) time
        const gap = startFrame - pos;
        if (gap > Math.max(2, rate * 0.0005) && gap < rate * 3600) {
          const silence = new Float32Array(gap);
          chunker.push(resamplers.map((r) => r.process(silence).slice()));
          pos += gap;
        }
        if (skip < frames) {
          const planes = [];
          for (let ch = 0; ch < sample.numberOfChannels; ch++) {
            const buf = new Float32Array(frames);
            sample.copyTo(buf, { planeIndex: ch, format: 'f32-planar' });
            planes.push(skip ? buf.subarray(skip) : buf);
          }
          const n = frames - skip;
          let mixed;
          if (channels === 1 && planes.length > 1) {
            const m = new Float32Array(n);
            for (const p of planes) for (let i = 0; i < n; i++) m[i] += p[i];
            const g = 1 / planes.length;
            for (let i = 0; i < n; i++) m[i] *= g;
            mixed = [m];
          } else {
            mixed = planes.slice(0, channels);
          }
          chunker.push(mixed.map((p, c) => resamplers[c].process(p).slice()));
          pos += n;
        }
      } finally {
        sample.close();
      }
      yield* [...chunker.drain()].map((chunk) => ({ type: 'chunk', chunk }));
    }
    if (resamplers) chunker.push(resamplers.map((r) => r.finish().slice()));
    yield* [...chunker.drain(true)].map((chunk) => ({ type: 'chunk', chunk }));
  } finally {
    input.dispose();
  }
}

async function* viaLibav(file, opts) {
  let channels = 1;
  let rate = 0;
  let resamplers = null;
  let chunker = null;
  for await (const ev of libavFrames(file, opts)) {
    if (ev.type === 'info') {
      channels = ev.srcChannels === 2 ? 2 : 1;
      chunker = new Chunker(channels);
      yield { type: 'info', channels, sourceRate: ev.sourceRate, srcChannels: ev.srcChannels, base: 0 };
      continue;
    }
    if (!resamplers || rate !== ev.rate) {
      if (resamplers) chunker.push(resamplers.map((r) => r.finish().slice()));
      rate = ev.rate;
      resamplers = Array.from({ length: channels }, () => new Resampler(rate, TARGET_RATE));
    }
    let mixed = ev.planes.slice(0, channels);
    if (channels === 1 && ev.planes.length > 1) {
      const n = ev.planes[0].length;
      const m = new Float32Array(n);
      for (const p of ev.planes) for (let i = 0; i < n; i++) m[i] += p[i];
      const g = 1 / ev.planes.length;
      for (let i = 0; i < n; i++) m[i] *= g;
      mixed = [m];
    }
    chunker.push(mixed.map((p, c) => resamplers[c].process(p).slice()));
    yield* [...chunker.drain()].map((chunk) => ({ type: 'chunk', chunk }));
  }
  if (resamplers) chunker.push(resamplers.map((r) => r.finish().slice()));
  if (chunker) yield* [...chunker.drain(true)].map((chunk) => ({ type: 'chunk', chunk }));
}

async function* viaWebAudio(file, { signal }) {
  if (file.size > 1.5 * 1024 ** 3) throw new DecodeError('This file is too large to decode in this browser. Try a smaller file or extract the audio first.');
  const ab = await file.arrayBuffer();
  const Ctx = window.OfflineAudioContext || window.webkitOfflineAudioContext;
  if (!Ctx) throw new DecodeError('This browser cannot decode audio files.');
  let decoded;
  try {
    decoded = await new Ctx(1, 1, TARGET_RATE).decodeAudioData(ab);
  } catch {
    throw new DecodeError('Could not read this file. It may be damaged or in a format this browser does not support.');
  }
  const n = decoded.numberOfChannels;
  const channels = n === 2 ? 2 : 1;
  yield { type: 'info', channels, sourceRate: decoded.sampleRate, srcChannels: n, base: 0 };
  const total = decoded.length;
  const planes = Array.from({ length: n }, (_, i) => decoded.getChannelData(i));
  for (let pos = 0; pos < total; pos += CHUNK_FRAMES) {
    if (signal?.aborted) throw new DOMException('Canceled', 'AbortError');
    const end = Math.min(total, pos + CHUNK_FRAMES);
    if (channels === 1 && n > 1) {
      const m = new Float32Array(end - pos);
      for (const p of planes) for (let i = pos; i < end; i++) m[i - pos] += p[i];
      for (let i = 0; i < m.length; i++) m[i] /= n;
      yield { type: 'chunk', chunk: [m] };
    } else {
      yield { type: 'chunk', chunk: planes.slice(0, channels).map((p) => p.slice(pos, end)) };
    }
  }
}

/** Keeps only the part of the stream between `start` and `end` seconds (the decoders that can't start in the middle read from the top). */
async function* trimmed(gen, start, end) {
  const S0 = Math.round(start * TARGET_RATE);
  const S1 = end === Infinity ? Infinity : Math.round(end * TARGET_RATE);
  let at = 0; // the frame (48 kHz, from the start of the file) the next chunk begins at
  for await (const ev of gen) {
    if (ev.type === 'info') { at = Math.round((ev.base || 0) * TARGET_RATE); yield { ...ev, base: start }; continue; }
    const n = ev.chunk[0].length;
    const a = at;
    at += n;
    if (a >= S1) return;
    if (a + n <= S0) continue;
    const from = Math.max(0, S0 - a);
    const to = Math.min(n, S1 - a);
    yield { type: 'chunk', chunk: from === 0 && to === n ? ev.chunk : ev.chunk.map((c) => c.slice(from, to)) };
    if (a + n >= S1) return;
  }
}

/**
 * Async generator of events: first {type:'info', channels, sourceRate, srcChannels}, then
 * {type:'chunk', chunk: Float32Array[]} (48 kHz, one array per output channel).
 * `start` / `end` (seconds): only that part of the recording; mediabunny starts right there, the other decoders read up to it.
 */
export async function* decodeToChunks(file, opts = {}) {
  const { start = 0, end = Infinity } = opts;
  if (start > 0 || end < Infinity) {
    yield* trimmed(decodeAll(file, { ...opts, from: Math.max(0, start), to: end === Infinity ? Infinity : end + 0.5 }), Math.max(0, start), end);
    return;
  }
  yield* decodeAll(file, opts);
}

async function* decodeAll(file, opts) {
  let produced = false;
  try {
    for await (const ev of viaMediabunny(file, opts)) {
      if (ev.type === 'chunk') produced = true;
      yield ev;
    }
    return;
  } catch (e) {
    if (e?.name === 'AbortError' || produced) throw e;
    // Nothing decoded yet: try libav.js, then the browser's own decoder.
  }
  if (libavAvailable()) {
    try {
      for await (const ev of viaLibav(file, opts)) {
        if (ev.type === 'chunk') produced = true;
        yield ev;
      }
      return;
    } catch (e) {
      if (e?.name === 'AbortError' || produced) throw e;
    }
  }
  yield* viaWebAudio(file, opts);
}
