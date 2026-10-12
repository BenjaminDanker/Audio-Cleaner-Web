// Copies the DeepFilterNet3 WebAssembly build + model into public/dfn/.
//   npm run sync-dfn                  always copy
//   node scripts/sync-dfn.mjs --if-missing   copy only when public/dfn is incomplete (used by `npm run build`)
// Sources (override with env vars):
//   DFN_WASM_DIR   default ~/audio-cleaner/wasm   (df.js, df_bg.wasm, LICENSE-*)
//   DFN_MODEL      default ~/audio-cleaner/models/DeepFilterNet3_onnx.tar.gz
import { copyFileSync, existsSync, mkdirSync, statSync } from 'node:fs';
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

const complete = () => files.every(([, name]) => existsSync(join(out, name)));
if (process.argv.includes('--if-missing') && complete()) {
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
