// Main-thread side of the in-browser noise remover: one shared download, then one module worker per channel.

import { DENOISER, fetchPinned } from './models.js';

const BASE = import.meta.env.BASE_URL;

export { STRENGTHS } from '../strengths.js';

function dfnBase() {
  return new URL(`${BASE}dfn/`, location.href).href;
}

// ---- one shared download ---------------------------------------------------------------------------------------
// The noise remover is three files: df.js (the glue code, on this site) and the WebAssembly build and the model (about
// 22 MB together, from Hugging Face, pinned and checked: models.js). They are fetched ONCE per page load, in the
// background, as soon as a file has been chosen (the free in-browser cleanup, and the quick listen behind "Automatic best
// settings": analyse.js), or when "Test my room" starts on the Live page. Nothing is downloaded before.
// The wasm is compiled once, the compiled Module and the model bytes are handed to every worker, so a stereo file (two
// workers) never downloads or compiles anything twice. df.js is only warmed into the HTTP cache here:
// it is a module, so each worker imports it by URL (a cache hit, about 15 kB).
//
// Repeat visits: the wasm and the model are kept in the browser (models.js). df.js carries ?v=<hash> (__DFN_VERSION__),
// so a changed file is a new URL; the front door (deploy/vps/sites/00-audio-cleaner-snippets.caddy) lets the browser keep it for a week.
const V = `?v=${__DFN_VERSION__}`;
// Sizes of the three files; only the denominator of the progress readout.
const TOTAL = __DFN_JS_SIZE__ + DENOISER.wasm.size + DENOISER.model.size;

let load = null; // the download in progress or finished: { promise, startedAt, loaded, done }
const loadListeners = new Set();
let snapshot = { loaded: 0, total: TOTAL, done: false, started: false, secondsLeft: null };
let lastNotify = 0;

function notify(force = false) {
  const now = performance.now();
  if (!force && now - lastNotify < 100) return;
  lastNotify = now;
  if (load) {
    const loaded = Math.min(load.loaded, TOTAL);
    const elapsed = (now - load.startedAt) / 1000;
    const speed = elapsed > 0.2 && loaded > 0 ? loaded / elapsed : 0;
    const secondsLeft = load.done ? 0 : speed > 0 ? (TOTAL - loaded) / speed : null;
    snapshot = { loaded, total: TOTAL, done: load.done, started: true, secondsLeft };
  } else {
    snapshot = { loaded: 0, total: TOTAL, done: false, started: false, secondsLeft: null };
  }
  loadListeners.forEach((fn) => fn());
}

/** Progress of the shared download as an immutable snapshot: { loaded, total, done, started, secondsLeft } (bytes). */
export const denoiserLoadState = () => snapshot;
export function subscribeDenoiserLoad(fn) {
  loadListeners.add(fn);
  return () => loadListeners.delete(fn);
}

async function download(state) {
  const count = (n) => { state.loaded += n; notify(); };
  const js = fetch(`${dfnBase()}df.js${V}`, { priority: 'low', credentials: 'same-origin' }).then((res) => {
    if (!res.ok) throw new Error(`Could not get the noise remover (${res.status}).`);
    return res.arrayBuffer();
  }).then((b) => count(b.byteLength));
  const wasm = fetchPinned(DENOISER.wasm, { onBytes: count }).then((bytes) => WebAssembly.compile(bytes));
  const model = fetchPinned(DENOISER.model, { onBytes: count });
  const [, module, bytes] = await Promise.all([js, wasm, model]);
  return { module, model: bytes };
}

/**
 * Starts the one background download (idempotent: later calls get the same promise, so there is never a second
 * download) and resolves with { module: WebAssembly.Module, model: Uint8Array }. If it fails, calling this again tries again.
 */
export function preloadDenoiser() {
  if (load) return load.promise;
  const state = { loaded: 0, done: false, startedAt: performance.now(), promise: null };
  load = state;
  // The readout also has to move when no bytes arrive for a while (slow or stalled connection): "about N left" keeps growing.
  const tick = setInterval(() => notify(true), 500);
  state.promise = download(state).then(
    (assets) => {
      clearInterval(tick);
      state.done = true;
      notify(true);
      return assets;
    },
    (err) => {
      clearInterval(tick);
      if (load === state) load = null; // allow a retry
      notify(true);
      throw err;
    },
  );
  state.promise.catch(() => {}); // whoever needs the result handles the error; this just avoids an unhandled rejection
  notify(true);
  return state.promise;
}

/** What a worker needs to run the noise remover itself (the measuring worker, analyse.js): the shared download plus where df.js is. */
export function denoiserAssets() {
  return preloadDenoiser().then(({ module, model }) => ({ module, model, dfnBase: dfnBase(), version: __DFN_VERSION__ }));
}

class ChannelWorker {
  constructor(attenDb, assets) {
    this.worker = new Worker(new URL('./denoise.worker.js', import.meta.url), { type: 'module' });
    this.seq = 0;
    this.waiting = new Map();
    this.readyP = new Promise((resolve, reject) => {
      this.onReady = resolve;
      this.onFail = reject;
    });
    this.readyP.catch(() => {});
    this.worker.onmessage = (e) => this.handle(e.data);
    this.worker.onerror = (e) => this.fail(new Error(e.message || 'The noise remover stopped unexpectedly.'));
    // The compiled wasm Module and the model bytes come from the shared download: workers never fetch them.
    assets.then(
      ({ module, model }) => this.worker.postMessage({ type: 'init', dfnBase: dfnBase(), attenDb, version: __DFN_VERSION__, module, model }),
      (err) => this.fail(err),
    );
  }

  handle(m) {
    if (m.type === 'ready') return this.onReady();
    if (m.type === 'out') {
      const w = this.waiting.get(m.seq);
      if (w) {
        this.waiting.delete(m.seq);
        w.resolve(m.samples);
      }
      return undefined;
    }
    if (m.type === 'error') {
      const w = this.waiting.get(m.seq);
      const err = new Error(m.message);
      if (w) {
        this.waiting.delete(m.seq);
        return w.reject(err);
      }
      return this.fail(err);
    }
    return undefined;
  }

  fail(err) {
    this.onFail(err);
    for (const w of this.waiting.values()) w.reject(err);
    this.waiting.clear();
  }

  request(type, samples) {
    const seq = ++this.seq;
    return new Promise((resolve, reject) => {
      this.waiting.set(seq, { resolve, reject });
      if (samples) this.worker.postMessage({ type, seq, samples }, [samples.buffer]);
      else this.worker.postMessage({ type, seq });
    });
  }

  terminate() {
    this.worker.terminate();
    this.fail(new Error('Canceled'));
  }
}

export class DenoiseEngine {
  constructor({ channels, attenDb }) {
    const assets = preloadDenoiser();
    this.workers = Array.from({ length: channels }, () => new ChannelWorker(attenDb, assets));
  }

  ready() {
    return Promise.all(this.workers.map((w) => w.readyP));
  }

  /** chunk: array of Float32Array (one per channel, equal length). Takes ownership of the buffers. */
  process(chunk) {
    return Promise.all(this.workers.map((w, i) => w.request('process', chunk[i])));
  }

  flush() {
    return Promise.all(this.workers.map((w) => w.request('flush')));
  }

  /** Change how strong the noise removal is, from the next frame on (live use). */
  setAtten(db) {
    for (const w of this.workers) w.worker.postMessage({ type: 'atten', db });
  }

  terminate() {
    for (const w of this.workers) w.terminate();
  }
}
