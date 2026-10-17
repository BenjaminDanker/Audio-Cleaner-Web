// The noise remover (df.js, the libDF WebAssembly build) on one stream of 48 kHz mono audio, in pieces of any length.
// The model has a fixed algorithmic delay of 1440 samples: the first 1440 output samples are dropped and flush() pads with
// zeros, so the output lines up with the input sample for sample and is just as long. Used by the cleanup worker
// (denoise.worker.js), the measuring worker (analyse.worker.js) and the Node tools; `mod` is the loaded df.js module.

export const HOP = 480;
export const DELAY = 1440;

export class DfnStream {
  /** @param {object} mod df.js, initialised  @param {Uint8Array} model the model (.tar.gz bytes)  @param {number} attenDb */
  constructor(mod, model, attenDb = 100) {
    this.mod = mod;
    this.state = mod.df_create(model, attenDb);
    if (mod.df_get_frame_length(this.state) !== HOP) throw new Error('Unexpected model frame size.');
    this.pending = new Float32Array(0); // input not yet forming a whole 480-sample frame
    this.totalIn = 0;
    this.producedOut = 0; // model output samples so far (including the delay)
    this.emitted = 0; // samples handed back (after dropping the delay)
  }

  _run(frames) {
    if (frames.length === 0) return new Float32Array(0);
    const out = this.mod.df_process_frames(this.state, frames);
    const startAt = this.producedOut;
    this.producedOut += out.length;
    const dropUntil = Math.max(0, DELAY - startAt);
    if (dropUntil >= out.length) return new Float32Array(0);
    const kept = out.subarray(dropUntil);
    this.emitted += kept.length;
    return new Float32Array(kept); // own, transferable copy
  }

  /** Cleaned audio for what came in so far (whole frames only; the rest waits for the next call or flush). */
  process(samples) {
    this.totalIn += samples.length;
    let input = samples;
    if (this.pending.length) {
      input = new Float32Array(this.pending.length + samples.length);
      input.set(this.pending, 0);
      input.set(samples, this.pending.length);
    }
    const whole = Math.floor(input.length / HOP) * HOP;
    this.pending = input.slice(whole);
    return this._run(input.subarray(0, whole));
  }

  /** The rest: zero-padded until every sample that went in has come out. */
  flush() {
    if (this.totalIn === 0) return new Float32Array(0);
    const padLen = (HOP - (this.pending.length % HOP)) % HOP;
    const have = this.producedOut + this.pending.length + padLen;
    const extraFrames = Math.max(0, Math.ceil((this.totalIn + DELAY - have) / HOP));
    const input = new Float32Array(this.pending.length + padLen + extraFrames * HOP);
    input.set(this.pending, 0);
    this.pending = new Float32Array(0);
    let out = this._run(input);
    const over = this.emitted - this.totalIn;
    if (over > 0) {
      out = out.subarray(0, out.length - over);
      this.emitted = this.totalIn;
    }
    return new Float32Array(out);
  }

  setAtten(db) {
    if (this.state) this.mod.df_set_atten_lim(this.state, db);
  }

  free() {
    if (this.state) this.mod.df_free(this.state);
    this.state = 0;
  }
}

/** df.js loaded in a worker or a page from the site's /dfn/ folder, initialised with an already compiled module. */
export async function loadDfnModule(dfnBase, version, module) {
  const v = version ? `?v=${version}` : '';
  const mod = await import(/* @vite-ignore */ `${dfnBase}df.js${v}`);
  await mod.default({ module_or_path: module || `${dfnBase}df_bg.wasm${v}` });
  return mod;
}

/** Pairs audio with the noise remover's answers for it (which come a little behind, in pieces of their own) and hands each
 *  lined-up piece to `fn(x, y)`. */
export class Pairer {
  constructor(fn) {
    this.fn = fn;
    this.waiting = []; // audio not answered for yet
  }

  /** @param {Float32Array|null} x audio that went into the noise remover  @param {Float32Array} y what came out this time */
  feed(x, y) {
    if (x) this.waiting.push(x);
    let k = 0;
    while (k < y.length) {
      const w = this.waiting[0];
      const n = Math.min(w.length, y.length - k);
      this.fn(w.subarray(0, n), y.subarray(k, k + n));
      k += n;
      if (n === w.length) this.waiting.shift();
      else this.waiting[0] = w.subarray(n);
    }
  }
}
