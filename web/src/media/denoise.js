// Main-thread side of the in-browser noise remover: one shared download, then one module worker per channel.

const BASE = import.meta.env.BASE_URL;

export const STRENGTHS = [
  { id: 'natural', label: 'Natural', db: 18, hint: 'Keeps some room tone. Gentlest.' },
  { id: 'strong', label: 'Strong', db: 40, hint: 'Best for most voices.' },
  { id: 'maximum', label: 'Maximum', db: 100, hint: 'Removes as much noise as possible.' },
];

function dfnBase() {
  return new URL(`${BASE}dfn/`, location.href).href;
}

// ---- one shared download ---------------------------------------------------------------------------------------
// The noise remover is three files (df.js, df_bg.wasm, the model; about 22 MB together). They are fetched ONCE per page
// load, in the background, as soon as the /free page opens (nothing is downloaded on any other page). The wasm is
// compiled once, the compiled Module and the model bytes are handed to every worker, so a stereo file (two workers)
// never downloads or compiles anything twice. df.js is only warmed into the HTTP cache here: it is a module, so each
// worker imports it by URL (a cache hit, about 15 kB).
//
// Repeat visits: every URL carries ?v=<hash of the files> (__DFN_VERSION__), so the server can (and does) answer them
// with `Cache-Control: public, max-age=31536000, immutable`, and a returning visitor gets all of it from the browser
// cache without a network round trip.
const V = `?v=${__DFN_VERSION__}`;
// Decoded sizes of the three files; only the denominator of the progress readout (see vite.config.js).
const SIZES = __DFN_SIZES__;
const TOTAL = SIZES.js + SIZES.wasm + SIZES.model;

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

/** Counts every received byte of a response body into the shared progress. */
function counted(body, state) {
  return body.pipeThrough(new TransformStream({
    transform(chunk, ctl) {
      state.loaded += chunk.byteLength;
      notify();
      ctl.enqueue(chunk);
    },
  }));
}

function get(url) {
  // `priority: 'low'` keeps this behind the page itself and whatever the visitor is doing (ignored where unsupported).
  return fetch(url, { priority: 'low', credentials: 'same-origin' }).then((res) => {
    if (!res.ok) throw new Error(`Could not get the noise remover (${res.status}).`);
    return res;
  });
}

async function download(state) {
  const base = dfnBase();
  const js = get(`${base}df.js${V}`).then((r) => r.arrayBuffer()).then((b) => { state.loaded += b.byteLength; });
  const wasm = get(`${base}df_bg.wasm${V}`).then((res) => {
    // Compile while the bytes arrive. The MIME type is set here, so a server that labels .wasm wrongly still works.
    const headers = new Headers({ 'Content-Type': 'application/wasm' });
    return WebAssembly.compileStreaming(new Response(counted(res.body, state), { status: 200, headers }));
  });
  const model = get(`${base}DeepFilterNet3_onnx.tar.gz${V}`).then(async (res) => {
    const chunks = [];
    const reader = counted(res.body, state).getReader();
    for (;;) {
      const { done, value } = await reader.read();
      if (done) break;
      chunks.push(value);
    }
    return new Uint8Array(await new Blob(chunks).arrayBuffer());
  });
  const [, module, bytes] = await Promise.all([js, wasm, model]);
  let gz = bytes;
  // Some static servers (and the Vite dev server) label *.tar.gz as Content-Encoding: gzip, so the browser hands us
  // the already-decompressed tar. The model loader wants the gzip, so compress it again in that case.
  if (!(gz[0] === 0x1f && gz[1] === 0x8b)) {
    const stream = new Blob([gz]).stream().pipeThrough(new CompressionStream('gzip'));
    gz = new Uint8Array(await new Response(stream).arrayBuffer());
  }
  return { module, model: gz };
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

  terminate() {
    for (const w of this.workers) w.terminate();
  }
}
