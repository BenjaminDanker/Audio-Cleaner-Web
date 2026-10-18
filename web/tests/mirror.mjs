// A local copy of the Hugging Face files the free tools download (src/media/models.js), for the browser suites.
// The site is built with ACW_MODEL_HOST=http://127.0.0.1:<port>/ (default port 8798, E2E_MODEL_PORT), and this serves
// <host><repo>/resolve/<revision>/<path> from the real files, found by the pins in models.js (size and SHA-256), so a model
// change there is followed without touching this file. CORS is open, the bytes are served raw (no Content-Encoding) and
// never cached by the browser's HTTP cache, so every request that reaches the server is a real download.
// `mirror.hits` counts the downloads per path; `mirror.tamper(path)` serves that file with one byte changed (same size)
// until `mirror.tamper(null)`, to prove the product refuses a file that is not the pinned one.
import { createHash } from 'node:crypto';
import { createServer } from 'node:http';
import { existsSync, readdirSync, readFileSync, statSync } from 'node:fs';
import { homedir } from 'node:os';
import { basename, join } from 'node:path';

export const MIRROR_PORT = Number(process.env.E2E_MODEL_PORT || 8798);
export const MIRROR_ORIGIN = `http://127.0.0.1:${MIRROR_PORT}`;

const AC = join(homedir(), 'audio-cleaner');
const ROOTS = [
  ...(process.env.ACW_MODEL_SRC ? process.env.ACW_MODEL_SRC.split(':') : []),
  process.env.DFN_WASM_DIR || join(AC, 'wasm'), join(AC, 'models'), join(AC, 'dev', 'testmodels'),
];

/** Every pinned file of models.js: { path: '<repo>/resolve/<revision>/<path>', size, sha256 }. */
async function pins() {
  globalThis.__MODEL_HOST__ = `${MIRROR_ORIGIN}/`; // models.js reads the build-time define
  const m = await import('../src/media/models.js');
  const out = [...Object.values(m.DENOISER)];
  for (const [path, f] of Object.entries(m.SPEECH.files)) out.push({ repo: m.SPEECH.repo, revision: m.SPEECH.revision, path, ...f });
  return out.map((p) => ({ url: `/${p.repo}/resolve/${p.revision}/${p.path}`, name: basename(p.path), size: p.size, sha256: p.sha256 }));
}

/** A local file with this name and size (found under the model folders, a few levels deep). */
function find(name, size) {
  const walk = (dir, depth) => {
    if (!existsSync(dir)) return null;
    for (const e of readdirSync(dir, { withFileTypes: true })) {
      const p = join(dir, e.name);
      if (e.isFile() && e.name === name && statSync(p).size === size) return p;
      if (e.isDirectory() && depth < 5 && !['venv', 'node_modules'].includes(e.name)) { const r = walk(p, depth + 1); if (r) return r; }
    }
    return null;
  };
  for (const r of ROOTS) { const p = walk(r, 0); if (p) return p; }
  return null;
}

let started = null;
/** Starts the mirror once per process and returns it. */
export function startMirror() {
  started ??= (async () => {
    const files = new Map();
    for (const p of await pins()) {
      const local = find(p.name, p.size);
      if (!local) throw new Error(`mirror: no local copy of ${p.name} (${p.size} bytes); set ACW_MODEL_SRC to the folder that has it`);
      files.set(p.url, { ...p, local, bytes: null });
    }
    const hits = {};
    let tampered = null;
    const server = createServer((req, res) => {
      const cors = { 'access-control-allow-origin': '*', 'access-control-allow-headers': '*', 'cache-control': 'no-store' };
      if (req.method === 'OPTIONS') { res.writeHead(204, cors); return res.end(); }
      const url = new URL(req.url, MIRROR_ORIGIN).pathname;
      const f = files.get(url);
      if (!f) { res.writeHead(404, cors); return res.end('not found'); }
      hits[url] = (hits[url] || 0) + 1;
      if (!f.bytes) {
        f.bytes = readFileSync(f.local);
        if (createHash('sha256').update(f.bytes).digest('hex') !== f.sha256) console.error(`MIRROR: ${f.local} is not the file pinned in models.js`);
      }
      let body = f.bytes;
      if (tampered && url.endsWith(tampered)) { body = Buffer.from(body); body[body.length >> 1] ^= 0xff; }
      res.writeHead(200, { ...cors, 'content-type': 'application/octet-stream', 'content-length': body.length });
      res.end(body);
    });
    await new Promise((ok, no) => { server.once('error', no); server.listen(MIRROR_PORT, '127.0.0.1', ok); });
    server.unref();
    return {
      origin: MIRROR_ORIGIN,
      hits,
      /** Downloads so far, as { 'df_bg.wasm': 1, ... } by file name. */
      byName: () => Object.fromEntries(Object.entries(hits).map(([u, n]) => [basename(u), n])),
      reset: () => { for (const k of Object.keys(hits)) delete hits[k]; },
      tamper: (suffix) => { tampered = suffix; },
      close: () => server.close(),
    };
  })();
  return started;
}
