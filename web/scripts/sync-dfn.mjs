// Copies the DeepFilterNet3 WebAssembly build + model into public/dfn/, and the libav.js decoder
// (native/libav/build.sh) into public/dfn/libav/<content hash>/.
//   npm run sync-dfn                  always copy
//   node scripts/sync-dfn.mjs --if-missing   copy only when public/dfn is incomplete (used by `npm run build`)
// Sources (override with env vars):
//   DFN_WASM_DIR   default ~/audio-cleaner/wasm   (df.js, df_bg.wasm, LICENSE-*)
//   DFN_MODEL      default ~/audio-cleaner/models/DeepFilterNet3_onnx.tar.gz
//   LIBAV_DIST     default ~/audio-cleaner/src/libav-build/dist (optional: without it, formats the browser
//                  cannot read itself simply are not offered in the browser)
// libav.js builds its own file URLs from the folder it is loaded from, so the folder name carries the
// version (the files are cached for a week); vite.config.js reads the folder name.
import { createHash } from 'node:crypto';
import { copyFileSync, existsSync, mkdirSync, readFileSync, readdirSync, rmSync, statSync } from 'node:fs';
import { homedir } from 'node:os';
import { dirname, join, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';

const root = resolve(dirname(fileURLToPath(import.meta.url)), '..');
const out = join(root, 'public', 'dfn');
const wasmDir = process.env.DFN_WASM_DIR || join(homedir(), 'audio-cleaner', 'wasm');
const model = process.env.DFN_MODEL || join(homedir(), 'audio-cleaner', 'models', 'DeepFilterNet3_onnx.tar.gz');

const files = [
  [join(wasmDir, 'df.js'), 'df.js'],
  [join(wasmDir, 'df_bg.wasm'), 'df_bg.wasm'],
  [join(wasmDir, 'LICENSE-MIT'), 'LICENSE-MIT'],
  [join(wasmDir, 'LICENSE-APACHE'), 'LICENSE-APACHE'],
  [model, 'DeepFilterNet3_onnx.tar.gz'],
];

const libavDist = process.env.LIBAV_DIST || join(homedir(), 'audio-cleaner', 'src', 'libav-build', 'dist');
const libavFiles = ['libav-acw-audio.mjs', 'libav-6.10.9.0-acw-audio.wasm.mjs', 'libav-6.10.9.0-acw-audio.wasm.wasm'];
const libavOut = join(out, 'libav');

function syncLibav() {
  if (!libavFiles.every((f) => existsSync(join(libavDist, f)))) {
    console.warn(`sync-dfn: no libav.js build in ${libavDist} (native/libav/build.sh); skipping it`);
    return;
  }
  const h = createHash('sha1');
  for (const f of libavFiles) h.update(readFileSync(join(libavDist, f)));
  const dir = join(libavOut, h.digest('hex').slice(0, 10));
  if (existsSync(libavOut)) for (const d of readdirSync(libavOut)) if (join(libavOut, d) !== dir) rmSync(join(libavOut, d), { recursive: true });
  mkdirSync(dir, { recursive: true });
  for (const f of libavFiles) copyFileSync(join(libavDist, f), join(dir, f));
  console.log(`sync-dfn: libav.js -> public/dfn/libav/${dir.split('/').pop()}/`);
}

// The speech runtime for the free in-browser subtitles (ONNX Runtime Web, MIT, from node_modules): its WebAssembly files
// are loaded at run time, so they are copied next to the noise remover's (same sign-in gate), in a folder named after
// the version (cached for a week). Which of the two builds a browser takes: src/media/transcribe.worker.js.
const ortPkg = JSON.parse(readFileSync(join(root, 'node_modules', 'onnxruntime-web', 'package.json'), 'utf8'));
const ortOut = join(out, 'ort');
const ortDir = join(ortOut, ortPkg.version.replace(/[^A-Za-z0-9.-]/g, '-'));
const ortFiles = ['ort-wasm-simd-threaded.asyncify.mjs', 'ort-wasm-simd-threaded.asyncify.wasm', 'ort-wasm-simd-threaded.mjs', 'ort-wasm-simd-threaded.wasm'];
function syncOrt() {
  if (ortFiles.every((f) => existsSync(join(ortDir, f)))) return;
  if (existsSync(ortOut)) for (const d of readdirSync(ortOut)) if (join(ortOut, d) !== ortDir) rmSync(join(ortOut, d), { recursive: true });
  mkdirSync(ortDir, { recursive: true });
  for (const f of ortFiles) copyFileSync(join(root, 'node_modules', 'onnxruntime-web', 'dist', f), join(ortDir, f));
  console.log(`sync-dfn: onnxruntime-web -> public/dfn/ort/${ortDir.split('/').pop()}/`);
}

// The free subtitles' model (Whisper base, ~80 MB): deploy/fetch-models.sh downloads it (pinned in deploy/models.lock)
// to ~/audio-cleaner/models/browser/whisper-base; it is copied next to the noise remover (same sign-in gate on the VPS),
// into a folder named after the pin, after its sha256 is checked against models.lock. The model id must match
// src/media/speech.js (MODEL).
const SPEECH_ID = 'whisper-base-1846881';
const speechSrc = process.env.SPEECH_MODEL_DIR || join(homedir(), 'audio-cleaner', 'models', 'browser', 'whisper-base');
const speechOut = join(out, 'models', SPEECH_ID);
function speechPins() {
  const lock = readFileSync(join(root, '..', 'deploy', 'models.lock'), 'utf8');
  const pins = [];
  for (const line of lock.split('\n')) {
    const p = line.trim().split(/\s+/);
    if (p[0] === 'file' && p[4]?.startsWith('models/browser/whisper-base/')) pins.push({ name: p[4].slice('models/browser/whisper-base/'.length), sha: p[2], size: Number(p[3]) });
  }
  return pins;
}
function syncSpeech() {
  const pins = speechPins();
  if (!pins.length) throw new Error('sync-dfn: no models/browser/whisper-base pins in deploy/models.lock');
  const ok = (dir) => pins.every((f) => existsSync(join(dir, f.name)) && statSync(join(dir, f.name)).size === f.size);
  if (ok(speechOut)) return;
  if (!ok(speechSrc)) {
    console.warn(`sync-dfn: the subtitle model is not in ${speechSrc} (run deploy/fetch-models.sh); the free subtitles will not load`);
    return;
  }
  for (const f of pins) {
    const got = createHash('sha256').update(readFileSync(join(speechSrc, f.name))).digest('hex');
    if (got !== f.sha) throw new Error(`sync-dfn: ${f.name} does not match its pin in deploy/models.lock`);
  }
  const modelsOut = join(out, 'models');
  if (existsSync(modelsOut)) for (const d of readdirSync(modelsOut)) if (d !== SPEECH_ID) rmSync(join(modelsOut, d), { recursive: true });
  for (const f of pins) {
    mkdirSync(dirname(join(speechOut, f.name)), { recursive: true });
    copyFileSync(join(speechSrc, f.name), join(speechOut, f.name));
  }
  console.log(`sync-dfn: subtitle model -> public/dfn/models/${SPEECH_ID}/`);
}

const libavPresent = () => existsSync(libavOut) && readdirSync(libavOut).length === 1;
const complete = () => files.every(([, name]) => existsSync(join(out, name))) && (libavPresent() || !existsSync(libavDist));
if (process.argv.includes('--if-missing') && complete()) {
  syncOrt();
  syncSpeech();
  console.log('sync-dfn: public/dfn already complete');
  process.exit(0);
}

mkdirSync(out, { recursive: true });
let total = 0;
for (const [src, name] of files) {
  if (!existsSync(src)) {
    console.error(`sync-dfn: missing ${src}\nSet DFN_WASM_DIR / DFN_MODEL to point at the DeepFilterNet wasm build and model.`);
    process.exit(1);
  }
  copyFileSync(src, join(out, name));
  total += statSync(src).size;
  console.log(`sync-dfn: ${name}`);
}
console.log(`sync-dfn: copied ${(total / 1048576).toFixed(1)} MB to public/dfn/`);
syncLibav();
syncOrt();
syncSpeech();
