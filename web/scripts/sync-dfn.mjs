// Copies the noise remover's glue code (df.js of the DeepFilterNet3 WebAssembly build) and its licences into public/dfn/,
// the libav.js decoder (native/libav/build.sh) into public/dfn/libav/<content hash>/ and the speech runtime into
// public/dfn/ort/<version>/. The WebAssembly build itself and the models are downloaded by the browser from Hugging Face
// (src/media/models.js), so df.js must belong to the df_bg.wasm pinned there: this script stops when the local build is
// a different one (upload the new build there and update the pin first).
//   npm run sync-dfn                  always copy
//   node scripts/sync-dfn.mjs --if-missing   copy only when public/dfn is incomplete (used by `npm run build`)
// Sources (override with env vars):
//   DFN_WASM_DIR   default ~/audio-cleaner/wasm   (df.js, df_bg.wasm, LICENSE-*)
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

const files = [
  [join(wasmDir, 'df.js'), 'df.js'],
  [join(wasmDir, 'LICENSE-MIT'), 'LICENSE-MIT'],
  [join(wasmDir, 'LICENSE-APACHE'), 'LICENSE-APACHE'],
];

/** df.js and the pinned df_bg.wasm come from the same build. */
function checkWasmPin() {
  const pin = /path: 'deepfilternet3\/df_bg\.wasm', size: \d+, sha256: '([0-9a-f]{64})'/.exec(readFileSync(join(root, 'src', 'media', 'models.js'), 'utf8'));
  if (!pin) throw new Error('sync-dfn: no df_bg.wasm pin in src/media/models.js');
  const local = join(wasmDir, 'df_bg.wasm');
  if (!existsSync(local)) throw new Error(`sync-dfn: missing ${local} (native/deepfilter/build.sh)`);
  if (createHash('sha256').update(readFileSync(local)).digest('hex') !== pin[1]) {
    throw new Error(`sync-dfn: ${local} is not the df_bg.wasm pinned in src/media/models.js: upload the new build to Hugging Face and update the pin, or rebuild the pinned one`);
  }
}

/** Files earlier versions of this script put here, now downloaded from Hugging Face. */
function removeOld() {
  for (const f of ['df_bg.wasm', 'DeepFilterNet3_onnx.tar.gz']) rmSync(join(out, f), { force: true });
  rmSync(join(out, 'models'), { recursive: true, force: true });
}

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
// the version (cached for a week). Only the plain build (no WebGPU) is used: src/media/transcribe.worker.js.
const ortPkg = JSON.parse(readFileSync(join(root, 'node_modules', 'onnxruntime-web', 'package.json'), 'utf8'));
const ortOut = join(out, 'ort');
const ortDir = join(ortOut, ortPkg.version.replace(/[^A-Za-z0-9.-]/g, '-'));
const ortFiles = ['ort-wasm-simd-threaded.mjs', 'ort-wasm-simd-threaded.wasm'];
function syncOrt() {
  // files of an earlier release of this script (the WebGPU build) are not published any more
  if (existsSync(ortDir)) for (const f of readdirSync(ortDir)) if (!ortFiles.includes(f)) rmSync(join(ortDir, f));
  if (ortFiles.every((f) => existsSync(join(ortDir, f)))) return;
  if (existsSync(ortOut)) for (const d of readdirSync(ortOut)) if (join(ortOut, d) !== ortDir) rmSync(join(ortOut, d), { recursive: true });
  mkdirSync(ortDir, { recursive: true });
  for (const f of ortFiles) copyFileSync(join(root, 'node_modules', 'onnxruntime-web', 'dist', f), join(ortDir, f));
  console.log(`sync-dfn: onnxruntime-web -> public/dfn/ort/${ortDir.split('/').pop()}/`);
}

const libavPresent = () => existsSync(libavOut) && readdirSync(libavOut).length === 1;
const complete = () => files.every(([, name]) => existsSync(join(out, name))) && (libavPresent() || !existsSync(libavDist));
checkWasmPin();
if (process.argv.includes('--if-missing') && complete()) {
  removeOld();
  syncOrt();
  console.log('sync-dfn: public/dfn already complete');
  process.exit(0);
}

mkdirSync(out, { recursive: true });
let total = 0;
for (const [src, name] of files) {
  if (!existsSync(src)) {
    console.error(`sync-dfn: missing ${src}\nSet DFN_WASM_DIR to point at the DeepFilterNet wasm build.`);
    process.exit(1);
  }
  copyFileSync(src, join(out, name));
  total += statSync(src).size;
  console.log(`sync-dfn: ${name}`);
}
console.log(`sync-dfn: copied ${(total / 1048576).toFixed(1)} MB to public/dfn/`);
removeOld();
syncLibav();
syncOrt();
