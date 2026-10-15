// Noise-remover worker. One instance cleans one audio channel.
//
// Protocol (main -> worker):
//   { type: 'init', dfnBase, attenDb, version, module, model }
//        module: the compiled WebAssembly.Module, model: the model bytes (both from the page's one shared download, so
//        the worker fetches nothing but the small df.js glue module, which is in the HTTP cache by then; without them
//        the worker falls back to fetching everything itself)  -> { type: 'ready' } | { type: 'error', message }
//   { type: 'process', seq, samples: Float32Array }  -> { type: 'out', seq, samples }   (arbitrary length, 48 kHz mono)
//   { type: 'flush', seq }                           -> { type: 'out', seq, samples, final: true }
//   { type: 'atten', db }
//   { type: 'close' }
// The model has a fixed algorithmic delay of 1440 samples: the first 1440 output samples are dropped and
// the stream is flushed with zeros, so total output length equals total input length.

const HOP = 480;
const DELAY = 1440;

let mod = null;
let state = 0;
let pending = new Float32Array(0); // input not yet forming a whole 480-sample frame
let totalIn = 0;
let producedOut = 0; // model output samples produced so far (including the delay)
let emitted = 0; // samples handed back (after dropping the delay)

async function init({ dfnBase, attenDb, version, module, model }) {
  const v = version ? `?v=${version}` : '';
  mod = await import(/* @vite-ignore */ `${dfnBase}df.js${v}`);
  await mod.default({ module_or_path: module || `${dfnBase}df_bg.wasm${v}` });
  let bytes = model;
  if (!bytes) {
    const res = await fetch(`${dfnBase}DeepFilterNet3_onnx.tar.gz${v}`);
    if (!res.ok) throw new Error(`Could not load the noise remover (${res.status}).`);
    bytes = new Uint8Array(await res.arrayBuffer());
    // Some static servers label *.tar.gz as Content-Encoding: gzip, so the browser hands us the decompressed tar; the
    // model loader wants the gzip, so compress it again in that case.
    if (!(bytes[0] === 0x1f && bytes[1] === 0x8b)) {
      const gz = new Blob([bytes]).stream().pipeThrough(new CompressionStream('gzip'));
      bytes = new Uint8Array(await new Response(gz).arrayBuffer());
    }
  }
  state = mod.df_create(bytes, attenDb);
  if (mod.df_get_frame_length(state) !== HOP) throw new Error('Unexpected model frame size.');
}

function run(frames) {
  // frames: Float32Array with a whole number of 480-sample frames
  if (frames.length === 0) return new Float32Array(0);
  const out = mod.df_process_frames(state, frames);
  const startAt = producedOut;
  producedOut += out.length;
  // drop the first DELAY samples of the model output
  const dropUntil = Math.max(0, DELAY - startAt);
  if (dropUntil >= out.length) return new Float32Array(0);
  const kept = out.subarray(dropUntil);
  emitted += kept.length;
  return new Float32Array(kept); // own, transferable copy
}

function process(samples) {
  totalIn += samples.length;
  let input = samples;
  if (pending.length) {
    input = new Float32Array(pending.length + samples.length);
    input.set(pending, 0);
    input.set(samples, pending.length);
  }
  const whole = Math.floor(input.length / HOP) * HOP;
  pending = input.slice(whole);
  return run(input.subarray(0, whole));
}

function flush() {
  // Pad the last partial frame with zeros, then push enough zero frames for every real sample to come out.
  if (totalIn === 0) return new Float32Array(0);
  const padLen = (HOP - (pending.length % HOP)) % HOP;
  const have = producedOut + pending.length + padLen;
  const extraFrames = Math.max(0, Math.ceil((totalIn + DELAY - have) / HOP));
  const input = new Float32Array(pending.length + padLen + extraFrames * HOP);
  input.set(pending, 0);
  pending = new Float32Array(0);
  let out = run(input);
  const over = emitted - totalIn;
  if (over > 0) {
    out = out.subarray(0, out.length - over);
    emitted = totalIn;
  }
  return new Float32Array(out);
}

self.onmessage = async (e) => {
  const m = e.data;
  try {
    if (m.type === 'init') {
      await init(m);
      self.postMessage({ type: 'ready' });
    } else if (m.type === 'process') {
      const samples = process(m.samples);
      self.postMessage({ type: 'out', seq: m.seq, samples }, [samples.buffer]);
    } else if (m.type === 'flush') {
      const samples = flush();
      self.postMessage({ type: 'out', seq: m.seq, samples, final: true }, [samples.buffer]);
    } else if (m.type === 'atten') {
      if (state) mod.df_set_atten_lim(state, m.db);
    } else if (m.type === 'close') {
      if (state) mod.df_free(state);
      state = 0;
      self.close();
    }
  } catch (err) {
    self.postMessage({ type: 'error', seq: m.seq, message: String(err?.message || err) });
  }
};
