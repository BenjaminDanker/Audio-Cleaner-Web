// Main-thread side of the free in-browser subtitles: one shared model download and one worker per page, then
// subtitles for a whole file (transcribeFile) or for short pieces of live speech (transcribeSpeech).
//
// The model (Whisper base, about 80 MB) and the runtime's WebAssembly sit next to the noise remover's files (the same
// sign-in gate, scripts/sync-dfn.mjs) and stay in the browser's cache, so they are downloaded once. Everything runs
// on the visitor's device; nothing is uploaded.

import { decodeToChunks, TARGET_RATE } from './decode.js';
import { Resampler } from './resample.js';

const MODEL = 'whisper-base-1846881'; // the folder scripts/sync-dfn.mjs puts it in (SPEECH_ID)
const MODEL_BYTES = 79664191; // the seven files together, for the progress readout
const RATE = 16000;
const WINDOW = 30 * RATE; // the model hears at most 30 s at a time
const CUT_FROM = 20 * RATE; // a window ends at the quietest moment between 20 and 29.5 s
const CUT_TO = 29.5 * RATE;
const SILENT_RMS = 0.001; // about -60 dB: nothing to transcribe (the model would make words up)

export const STAGE_SPEECH_LOADING = 'Getting the subtitle maker ready…';

/* ------------------------------------------------------------------------------------------------ the shared engine */

let engine = null; // { worker, ready, waiting, seq }
const progress = new Map(); // file -> loaded bytes
const listeners = new Set();
let snapshot = { loaded: 0, total: MODEL_BYTES, done: false, started: false };

function notify(patch) {
  snapshot = { ...snapshot, ...patch };
  listeners.forEach((fn) => fn());
}

/** Download progress of the model: { loaded, total, done, started } (bytes). For useSyncExternalStore. */
export const speechLoadState = () => snapshot;
export function subscribeSpeechLoad(fn) {
  listeners.add(fn);
  return () => listeners.delete(fn);
}

function plainWasmOnly() {
  // Safari before 26 without WebGPU can't run the runtime's WebGPU-capable build (the library makes the same choice).
  const ua = navigator.userAgent;
  const safari = /Safari\//.test(ua) && !/Chrome|Chromium|Edg\//.test(ua);
  const major = Number((ua.match(/Version\/(\d+)/) || [])[1] || 0);
  return safari && major < 26 && !navigator.gpu;
}

/** Starts the worker and the model download once per page (later calls share it). Resolves when it can work. */
export function preloadSpeech() {
  if (engine) return engine.ready;
  const worker = new Worker(new URL('./transcribe.worker.js', import.meta.url), { type: 'module' });
  const e = { worker, waiting: new Map(), seq: 0, ready: null };
  engine = e;
  notify({ started: true });
  e.ready = new Promise((resolve, reject) => {
    worker.onmessage = (ev) => {
      const m = ev.data;
      if (m.type === 'progress') {
        progress.set(m.file, m.loaded || 0);
        notify({ loaded: Math.min(MODEL_BYTES, [...progress.values()].reduce((a, b) => a + b, 0)) });
      } else if (m.type === 'ready') {
        notify({ loaded: MODEL_BYTES, done: true });
        resolve();
      } else if (m.id != null && e.waiting.has(m.id)) {
        const w = e.waiting.get(m.id);
        e.waiting.delete(m.id);
        if (m.type === 'error') w.reject(new Error(m.message));
        else w.resolve(m);
      } else if (m.type === 'error') {
        reject(new Error('The subtitle maker couldn’t start in this browser. Please reload the page and try again.'));
      }
    };
    worker.onerror = () => reject(new Error('The subtitle maker couldn’t start in this browser.'));
  });
  e.ready.catch(() => { if (engine === e) stopSpeech(); });
  const origin = location.origin;
  worker.postMessage({
    type: 'init', model: MODEL, plainWasm: plainWasmOnly(),
    modelHost: `${origin}${import.meta.env.BASE_URL}dfn/models/`,
    ortBase: `${origin}${import.meta.env.BASE_URL}dfn/ort/${__ORT_DIR__}/`,
  });
  return e.ready;
}

/** Ends the worker (a cancel in the middle of a long file). The next use starts it again; the model is in the cache. */
export function stopSpeech() {
  const e = engine;
  if (!e) return;
  engine = null;
  e.worker.terminate();
  for (const w of e.waiting.values()) w.reject(new DOMException('Canceled', 'AbortError'));
  progress.clear();
  notify({ loaded: 0, done: false, started: false });
}

async function ask(type, payload, transfer = []) {
  await preloadSpeech();
  const e = engine;
  const id = ++e.seq;
  return new Promise((resolve, reject) => {
    e.waiting.set(id, { resolve, reject });
    e.worker.postMessage({ type, id, ...payload }, transfer);
  });
}

const rms = (a) => {
  let s = 0;
  for (let i = 0; i < a.length; i++) s += a[i] * a[i];
  return a.length ? Math.sqrt(s / a.length) : 0;
};

/** The language spoken in this piece (16 kHz mono, up to 30 s): a two-letter code like 'en'. */
export async function detectLanguage(audio) {
  return (await ask('detect', { audio })).language;
}

/** Subtitles for one piece of speech (16 kHz mono, up to 30 s): [{start, end, text}] in seconds from its start. */
export async function transcribeSpeech(audio, language) {
  if (rms(audio) < SILENT_RMS) return [];
  return (await ask('transcribe', { audio, language })).segments;
}

/* ------------------------------------------------------------------------------------------------ a whole file */

/** Where to end a window: the quietest 100 ms between CUT_FROM and CUT_TO, so a word is rarely cut in two. */
function quietestCut(buf) {
  const step = RATE / 10;
  let best = CUT_TO;
  let bestE = Infinity;
  for (let i = CUT_FROM; i + step <= Math.min(CUT_TO, buf.length); i += step / 2) {
    let e = 0;
    for (let j = i; j < i + step; j++) e += buf[j] * buf[j];
    if (e < bestE) { bestE = e; best = i + step / 2; }
  }
  return Math.round(best);
}

/**
 * Subtitles for a whole recording, made in this browser.
 * @param {File} file
 * @param {{language?:string|null, durationHint?:number|null, signal?:AbortSignal,
 *   onProgress?:(p:{done:number,total:number|null,eta:number|null})=>void, onStage?:(s:string)=>void}} opts
 * @returns {Promise<{segments:{start:number,end:number,text:string}[], language:string, seconds:number}>}
 */
export async function transcribeFile(file, { language = null, durationHint = null, signal, onProgress, onStage } = {}) {
  const abort = () => stopSpeech();
  signal?.addEventListener('abort', abort, { once: true });
  const started = performance.now();
  try {
    if (!speechLoadState().done) onStage?.(STAGE_SPEECH_LOADING);
    await preloadSpeech();
    onStage?.('Writing the subtitles…');
    let lang = language || null;
    const segments = [];
    let buf = new Float32Array(WINDOW * 2);
    let have = 0;
    let offset = 0; // seconds before buf[0]
    let resampler = null;
    let workStarted = 0;
    let workDone = 0;

    const report = () => {
      if (!onProgress) return;
      const total = durationHint || null;
      const elapsed = (performance.now() - (workStarted || started)) / 1000;
      const rate = workDone > 0 && elapsed > 0 ? workDone / elapsed : 0;
      onProgress({ done: offset, total, eta: total && rate ? Math.max(0, (total - offset) / rate) : null });
    };
    const run = async (piece) => {
      if (signal?.aborted) throw new DOMException('Canceled', 'AbortError');
      if (!workStarted) workStarted = performance.now();
      if (rms(piece) >= SILENT_RMS) {
        if (!lang) lang = await detectLanguage(piece.slice(0, WINDOW));
        for (const s of await transcribeSpeech(piece, lang)) segments.push({ start: s.start + offset, end: s.end + offset, text: s.text });
      }
      workDone += piece.length / RATE;
      offset += piece.length / RATE;
      report();
    };
    const push = (mono) => {
      if (have + mono.length > buf.length) {
        const bigger = new Float32Array(Math.max(buf.length * 2, have + mono.length));
        bigger.set(buf.subarray(0, have));
        buf = bigger;
      }
      buf.set(mono, have);
      have += mono.length;
    };

    for await (const ev of decodeToChunks(file, { signal })) {
      if (ev.type !== 'chunk') continue;
      const ch = ev.chunk;
      let mono = ch[0];
      if (ch.length > 1) {
        mono = new Float32Array(ch[0].length);
        for (const c of ch) for (let i = 0; i < mono.length; i++) mono[i] += c[i] / ch.length;
      }
      resampler ||= new Resampler(TARGET_RATE, RATE);
      push(resampler.process(mono));
      while (have >= WINDOW) {
        const cut = quietestCut(buf.subarray(0, have));
        await run(buf.slice(0, cut));
        buf.copyWithin(0, cut, have);
        have -= cut;
      }
    }
    if (resampler) push(resampler.finish ? resampler.finish() : new Float32Array(0));
    if (have > RATE * 0.3) await run(buf.slice(0, have));
    else offset += have / RATE;
    return { segments, language: lang || language || 'en', seconds: offset };
  } finally {
    signal?.removeEventListener('abort', abort);
  }
}

/* ------------------------------------------------------------------------------------------------ subtitle files */

const MAX_LINE = 42;
const MAX_LINES = 2;
const MAX_CUE_S = 7;

function wrap(text) {
  const lines = [];
  let cur = '';
  for (const w of text.split(/\s+/).filter(Boolean)) {
    if (cur && (cur + ' ' + w).length > MAX_LINE) {
      lines.push(cur);
      cur = w;
    } else {
      cur = cur ? `${cur} ${w}` : w;
    }
  }
  if (cur) lines.push(cur);
  return lines;
}

/** Subtitle cues people can read: at most two lines of about 42 characters and 7 seconds each (a long segment is
 *  split at word boundaries, its time shared out by the length of the text). */
export function buildCues(segments) {
  const cues = [];
  for (const seg of segments) {
    const lines = wrap(seg.text);
    const groups = [];
    for (let i = 0; i < lines.length; i += MAX_LINES) groups.push(lines.slice(i, i + MAX_LINES));
    const dur = Math.max(0.5, seg.end - seg.start);
    let pieces = groups;
    if (groups.length === 1 && dur > MAX_CUE_S && lines.length > 1) pieces = lines.map((l) => [l]);
    const chars = pieces.reduce((n, g) => n + g.join(' ').length, 0) || 1;
    let t = seg.start;
    for (const g of pieces) {
      const share = (dur * g.join(' ').length) / chars;
      cues.push({ start: t, end: Math.min(seg.start + dur, t + share), lines: g });
      t += share;
    }
  }
  // never overlapping, never shorter than a moment
  for (let i = 0; i < cues.length; i++) {
    if (i + 1 < cues.length && cues[i].end > cues[i + 1].start) cues[i].end = cues[i + 1].start;
    if (cues[i].end - cues[i].start < 0.3) cues[i].end = cues[i].start + 0.3;
  }
  return cues;
}

function stamp(sec, sep) {
  const ms = Math.max(0, Math.round(sec * 1000));
  const h = Math.floor(ms / 3600000);
  const m = Math.floor((ms % 3600000) / 60000);
  const s = Math.floor((ms % 60000) / 1000);
  const pad = (n, w = 2) => String(n).padStart(w, '0');
  return `${pad(h)}:${pad(m)}:${pad(s)}${sep}${pad(ms % 1000, 3)}`;
}

export const toSrt = (cues) => cues.map((c, i) => `${i + 1}\n${stamp(c.start, ',')} --> ${stamp(c.end, ',')}\n${c.lines.join('\n')}\n`).join('\n');
export const toVtt = (cues) => `WEBVTT\n\n${cues.map((c) => `${stamp(c.start, '.')} --> ${stamp(c.end, '.')}\n${c.lines.join('\n')}\n`).join('\n')}`;

/** A plain transcript: the segments as paragraphs, a new one after a pause of two seconds or more. */
export function toText(segments) {
  const paras = [];
  let cur = [];
  let lastEnd = null;
  for (const s of segments) {
    if (lastEnd != null && s.start - lastEnd >= 2 && cur.length) {
      paras.push(cur.join(' '));
      cur = [];
    }
    cur.push(s.text);
    lastEnd = s.end;
  }
  if (cur.length) paras.push(cur.join(' '));
  return `${paras.join('\n\n')}\n`;
}
