// Copies the licence texts of everything we ship into public/licenses/ so /licenses can link to them.
// Run automatically by `npm run build` (and `npm run dev`); the list lives in scripts/license-list.mjs.
// Fails (exit 1) if a source file is missing, so a build never ships a licence page with dead links.
import { copyFileSync, existsSync, mkdirSync, readdirSync, rmSync, statSync } from 'node:fs';
import { dirname, join, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';
import { COMPONENTS } from './license-list.mjs';

const root = resolve(dirname(fileURLToPath(import.meta.url)), '..');
const out = join(root, 'public', 'licenses');

const wanted = new Map();
for (const c of COMPONENTS) for (const t of c.texts) if (t.file) wanted.set(t.file, t.from);

const missing = [...wanted].filter(([, from]) => !existsSync(join(root, from)));
if (missing.length) {
  console.error(`sync-licenses: missing source file(s):\n${missing.map(([f, from]) => `  ${from} (for ${f})`).join('\n')}\nRun \`npm ci\` (node_modules) and \`npm run sync-dfn\` (public/dfn) first.`);
  process.exit(1);
}

mkdirSync(out, { recursive: true });
for (const f of readdirSync(out)) if (!wanted.has(f)) rmSync(join(out, f), { force: true }); // stale
let total = 0;
for (const [file, from] of wanted) {
  copyFileSync(join(root, from), join(out, file));
  total += statSync(join(out, file)).size;
}
console.log(`sync-licenses: ${wanted.size} files (${(total / 1024).toFixed(0)} kB) in public/licenses/`);
