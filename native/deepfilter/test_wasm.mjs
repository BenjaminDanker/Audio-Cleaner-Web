#!/usr/bin/env node
// Smoke test + benchmark for the libDF wasm build (Node >= 18, wasm SIMD).
//
//   node native/deepfilter/test_wasm.mjs <dir with df.js + df_bg.wasm> <model.tar.gz> \
//        [--input in.f32 [--ref native-out.f32]] [--seconds 20]
//
// Loads df_bg.wasm with initSync, creates the state from the model tarball bytes via
// df_create_with_params (always-on stages), processes N seconds of 48 kHz audio in one
// df_process_frames call and again frame by frame with df_process_frame, and prints ms/frame.
// With --input (raw little-endian f32, e.g. written by test_native.py --save) the same signal as
// the native test is used, and with --ref the output is compared to the native library's output.
import { readFileSync } from 'node:fs';
import { resolve } from 'node:path';
import { pathToFileURL } from 'node:url';
import { performance } from 'node:perf_hooks';

const SR = 48000;
const args = process.argv.slice(2);
const opt = (name, def) => {
  const i = args.indexOf(name);
  if (i < 0) return def;
  return args.splice(i, 2)[1];
};
const inputPath = opt('--input', null);
const refPath = opt('--ref', null);
const seconds = Number(opt('--seconds', 20));
const [dir, modelPath] = args;
if (!dir || !modelPath) {
  console.error('usage: node test_wasm.mjs <wasm dir> <model.tar.gz> [--input in.f32 [--ref out.f32]] [--seconds 20]');
  process.exit(2);
}

const readF32 = (p) => {
  const b = readFileSync(p);
  return new Float32Array(b.buffer.slice(b.byteOffset, b.byteOffset + b.byteLength));
};

// Deterministic synthetic noisy speech-like signal (harmonic voice + syllabic envelope + noise).
function synth(sec) {
  const n = Math.floor(sec * SR);
  const x = new Float32Array(n);
  let seed = 12345;
  const rnd = () => ((seed = (Math.imul(seed, 1664525) + 1013904223) >>> 0) / 4294967296) - 0.5;
  let phase = 0;
  let lp = 0;
  for (let i = 0; i < n; i++) {
    const t = i / SR;
    const f0 = 120 + 40 * Math.sin(2 * Math.PI * 0.3 * t) + 15 * Math.sin(2 * Math.PI * 2.1 * t);
    phase += (2 * Math.PI * f0) / SR;
    let v = 0;
    for (let k = 1; k <= 24; k++) v += Math.sin(k * phase) / k;
    const env = Math.pow(Math.max(Math.sin(2 * Math.PI * 3.5 * t) + 0.3, 0), 1.5) *
      (Math.sin(2 * Math.PI * 0.4 * t) > -0.3 ? 1 : 0);
    lp = 0.95 * lp + 0.05 * rnd() * 12; // coloured noise
    x[i] = Math.max(-0.98, Math.min(0.98, 0.07 * v * env + 0.05 * lp + 0.01 * rnd()));
  }
  return x;
}

// df.js is an ES module; without the wasm-pack package.json ("type": "module") next to it Node
// would treat it as CommonJS, so fall back to importing it from a data: URL.
let df;
try {
  df = await import(pathToFileURL(resolve(dir, 'df.js')).href);
} catch (e) {
  if (!(e instanceof SyntaxError)) throw e;
  const src = readFileSync(resolve(dir, 'df.js'), 'utf8');
  df = await import('data:text/javascript;base64,' + Buffer.from(src).toString('base64'));
}
const wasmBytes = readFileSync(resolve(dir, 'df_bg.wasm'));
const model = new Uint8Array(readFileSync(modelPath));

let t0 = performance.now();
try {
  df.initSync({ module: wasmBytes });
} catch (e) {
  df.initSync(wasmBytes); // older wasm-bindgen signature
}
console.log(`initSync (compile+instantiate ${(wasmBytes.length / 1e6).toFixed(1)} MB wasm): ${(performance.now() - t0).toFixed(0)} ms`);

const INF = Infinity;
// error paths must throw JS errors, not trap
let threw = false;
try { df.df_create_with_params(new Uint8Array([1, 2, 3]), 100, -INF, INF, INF, 0); } catch (e) { threw = true; console.log('bad model bytes -> throws:', String(e.message ?? e).slice(0, 100)); }
if (!threw) throw new Error('expected an exception for an invalid model');
threw = false;
try { df.df_create_with_params(model, 100, -INF, INF, INF, -1); } catch (e) { threw = true; }
if (!threw) throw new Error('expected an exception for a negative post filter beta');

t0 = performance.now();
const st = df.df_create_with_params(model, 100, -INF, INF, INF, 0);
console.log(`df_create_with_params (load model): ${(performance.now() - t0).toFixed(0)} ms`);
const hop = df.df_get_frame_length(st);

let x = inputPath ? readF32(inputPath) : synth(seconds);
const nfr = Math.floor(x.length / hop);
x = x.subarray(0, nfr * hop);
console.log(`input: ${(x.length / SR).toFixed(1)} s, ${nfr} frames of ${hop} samples${inputPath ? ' (from file)' : ' (synthetic)'}`);

threw = false;
try { df.df_process_frames(st, x.subarray(0, hop + 1)); } catch (e) { threw = true; }
if (!threw) throw new Error('expected an exception for a non-multiple-of-hop input');

// warm-up on a throw-away state so JIT/tier-up does not skew the timing
{
  const w = df.df_create_with_params(model, 100, -INF, INF, INF, 0);
  df.df_process_frames(w, x.subarray(0, 50 * hop));
  df.df_free(w);
}

// 1) batch: whole signal in one call
t0 = performance.now();
const yBatch = df.df_process_frames(st, x);
const batchMs = performance.now() - t0;
df.df_free(st);
const bPer = batchMs / nfr;
console.log(`df_process_frames (1 call, ${nfr} frames): ${bPer.toFixed(3)} ms/frame, RTF ${(bPer / (1e3 * hop / SR)).toFixed(4)}`);

// 2) frame by frame
const st2 = df.df_create_with_params(model, 100, -INF, INF, INF, 0);
const yFrame = new Float32Array(x.length);
t0 = performance.now();
for (let i = 0; i < nfr; i++) yFrame.set(df.df_process_frame(st2, x.subarray(i * hop, (i + 1) * hop)), i * hop);
const fMs = performance.now() - t0;
df.df_free(st2);
const fPer = fMs / nfr;
console.log(`df_process_frame  (per frame):          ${fPer.toFixed(3)} ms/frame, RTF ${(fPer / (1e3 * hop / SR)).toFixed(4)}`);

// 3) df_reset: a used state, reset, gives exactly the output of a fresh one
{
  const a = df.df_create_with_params(model, 100, -INF, INF, INF, 0);
  const n = Math.min(nfr, 300) * hop;
  df.df_process_frames(a, x.subarray(0, 100 * hop));
  df.df_reset(a);
  const yReset = df.df_process_frames(a, x.subarray(0, n));
  df.df_free(a);
  const b = df.df_create_with_params(model, 100, -INF, INF, INF, 0);
  const yFresh = df.df_process_frames(b, x.subarray(0, n));
  df.df_free(b);
  for (let i = 0; i < n; i++) if (yReset[i] !== yFresh[i]) throw new Error('df_reset output differs from a fresh state at ' + i);
  console.log(`df_reset: ${n / hop} frames after a reset are bit-identical to a fresh state`);
}

const finite = (a) => a.every(Number.isFinite);
if (!finite(yBatch) || !finite(yFrame)) throw new Error('non-finite output');
let maxd = 0, e = 0;
for (let i = 0; i < yBatch.length; i++) { maxd = Math.max(maxd, Math.abs(yBatch[i] - yFrame[i])); e += yBatch[i] ** 2; }
const rms = Math.sqrt(e / yBatch.length);
console.log(`output finite, rms ${rms.toFixed(4)}; batch vs per-frame max |diff| ${maxd.toExponential(2)}`);
if (!(rms > 0)) throw new Error('silent output');
if (maxd > 1e-6) throw new Error('batch and per-frame outputs differ');

if (refPath) {
  const ref = readF32(refPath).subarray(0, yBatch.length);
  let num = 0, den = 0, md = 0;
  for (let i = 0; i < ref.length; i++) { const d = yBatch[i] - ref[i]; num += ref[i] ** 2; den += d * d; md = Math.max(md, Math.abs(d)); }
  console.log(`wasm vs native output: SNR ${(10 * Math.log10(num / Math.max(den, 1e-30))).toFixed(1)} dB, max |diff| ${md.toExponential(2)}`);
}
console.log('PASS');
