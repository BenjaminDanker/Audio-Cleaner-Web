// Main-thread side of the in-browser DeepFilterNet3 denoiser: one module worker per channel.

const BASE = import.meta.env.BASE_URL;

export const STRENGTHS = [
  { id: 'natural', label: 'Natural', db: 18, hint: 'Keeps some room tone. Gentlest.' },
  { id: 'strong', label: 'Strong', db: 40, hint: 'Best for most voices.' },
  { id: 'maximum', label: 'Maximum', db: 100, hint: 'Removes as much noise as possible.' },
];

function dfnBase() {
  return new URL(`${BASE}dfn/`, location.href).href;
}

class ChannelWorker {
  constructor(attenDb) {
    this.worker = new Worker(new URL('./denoise.worker.js', import.meta.url), { type: 'module' });
    this.seq = 0;
    this.waiting = new Map();
    this.readyP = new Promise((resolve, reject) => {
      this.onReady = resolve;
      this.onFail = reject;
    });
    this.worker.onmessage = (e) => this.handle(e.data);
    this.worker.onerror = (e) => this.fail(new Error(e.message || 'The noise-reduction worker crashed.'));
    this.worker.postMessage({ type: 'init', dfnBase: dfnBase(), attenDb, version: __DFN_VERSION__ });
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
    this.workers = Array.from({ length: channels }, () => new ChannelWorker(attenDb));
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
