// The browser's suggestions (src/media/suggest-core.js: the same steps as common/acw_common/suggest.py, on the website's measurements and
// decision) against the server's, on the same generated files: clean then noisy then phone-like speech (mono and stereo), clipped
// bursts, a short file and a rearranged one. tests/suggest-py.py makes the files and the server's answers (needs the libDF build, the
// model and a speech clip under ~/audio-cleaner, as the Python suites do); this runs the browser code in Node on each and compares:
// the file's decision, the change points (to a second), and every suggestion (its range to 1.5 s, rebuild or not, strength, remover,
// the reasons). No browser, no backend. Usage: node tests/suggest-agreement.mjs
import assert from 'node:assert/strict';
import { execFileSync } from 'node:child_process';
import { readFileSync } from 'node:fs';
import { homedir, tmpdir } from 'node:os';
import { dirname, join } from 'node:path';
import { fileURLToPath } from 'node:url';
import { DfnStream } from '../src/media/dfnstream.js';
import { findSuggestions } from '../src/media/suggest-core.js';
import { loadDfn, readWav } from './reco-cli.mjs';

const HERE = dirname(fileURLToPath(import.meta.url));
const PY = process.env.ACW_PYTHON || join(homedir(), 'audio-cleaner', 'venv', 'bin', 'python');
const OUT = process.env.SUGGEST_DIR || join(process.env.ACW_TEST_STATE || tmpdir(), `suggest-agreement-${process.getuid()}`);
let failed = 0;
const lines = [];
async function check(name, fn) {
  const t0 = Date.now();
  try {
    const detail = await fn();
    console.log(`PASS ${name}${detail ? ` -- ${detail}` : ''} (${Date.now() - t0} ms)`);
  } catch (e) {
    failed++;
    console.log(`FAIL ${name} -- ${String(e.message).split('\n').slice(0, 4).join(' / ')}`);
  }
}

if (!process.env.SUGGEST_REUSE) console.log(execFileSync(PY, [join(HERE, 'suggest-py.py'), OUT], { encoding: 'utf8', maxBuffer: 1 << 24 }).trim());
const dfn = await loadDfn();

function noiseSums(mono) {
  const s = new DfnStream(dfn.mod, dfn.model, 100);
  const parts = [s.process(mono), s.flush()];
  let voice = 0;
  let removed = 0;
  let at = 0;
  for (const y of parts) {
    for (let i = 0; i < y.length && at + i < mono.length; i++) { voice += y[i] * y[i]; const d = mono[at + i] - y[i]; removed += d * d; }
    at += y.length;
  }
  s.free();
  return Promise.resolve({ voice, removed });
}

async function browser(name) {
  const { planes } = readWav(join(OUT, `${name}.wav`));
  const ch = planes.length;
  const t0 = Date.now();
  const r = await findSuggestions({
    seconds: planes[0].length / 48000,
    read: async (a, b) => planes.slice(0, Math.min(ch, 2)).map((p) => p.slice(a, b)),
    noiseSums,
  });
  return { ...r, ms: Date.now() - t0 };
}

const near = (a, b, tol) => Math.abs(a - b) <= tol;
const sameSet = (a, b) => a.length === b.length && a.every((x) => b.includes(x));
let matched = 0;
let total = 0;

for (const name of ['parts-mono', 'parts-stereo', 'clipping', 'short', 'noisy-then-clean']) {
  await check(`${name}: the browser finds what the server finds`, async () => {
    const py = JSON.parse(readFileSync(join(OUT, `${name}.py.json`), 'utf8'));
    const js = await browser(name);
    const problems = [];
    const note = (ok, what) => { total++; if (ok) matched++; else problems.push(what); };
    note(JSON.stringify(js.whole) === JSON.stringify(py.whole), `whole file: browser ${JSON.stringify(js.whole)} vs server ${JSON.stringify(py.whole)}`);
    const pc = py.parts.slice(1).map((p) => p.start);
    const jc = js.parts.slice(1).map((p) => p.start);
    note(pc.length === jc.length && pc.every((t, i) => near(t, jc[i], 1.01)), `change points: browser ${jc} vs server ${pc}`);
    note(py.suggestions.length === js.suggestions.length, `${js.suggestions.length} suggestions vs ${py.suggestions.length}`);
    py.suggestions.forEach((s, i) => {
      const t = js.suggestions[i];
      if (!t) { note(false, `missing ${JSON.stringify(s)}`); return; }
      note(near(s.start, t.start, 1.5) && near(s.end, t.end, 1.5) && s.kind === t.kind, `range ${s.start}-${s.end} (${s.kind}) vs ${t.start}-${t.end} (${t.kind})`);
      note(s.rebuild === t.rebuild && JSON.stringify(s.noise) === JSON.stringify(t.noise), `settings ${JSON.stringify(s)} vs ${JSON.stringify(t)}`);
      note(sameSet(s.reasons, t.reasons), `reasons ${s.reasons} vs ${t.reasons}`);
    });
    lines.push(`${name}: ${py.suggestions.length} server / ${js.suggestions.length} browser suggestions, cuts ${pc} / ${jc}, ${(js.ms / 1000).toFixed(1)} s in Node`);
    if (problems.length) throw new Error(problems.join(' ; '));
    return `${js.suggestions.length} suggestions, cuts ${jc.join(', ') || 'none'}`;
  });
}

await check('cancel: a canceled run stops with an abort and nothing is returned', async () => {
  const { planes } = readWav(join(OUT, 'parts-mono.wav'));
  const ctl = new AbortController();
  let reads = 0;
  const p = findSuggestions({
    seconds: planes[0].length / 48000, signal: ctl.signal, noiseSums,
    read: async (a, b) => { if (++reads === 3) ctl.abort(); return planes.map((x) => x.slice(a, b)); },
  });
  await assert.rejects(p, (e) => e.name === 'AbortError');
});

console.log(`AGREEMENT ${matched}/${total} checks equal between the browser and the server`);
for (const l of lines) console.log(`  ${l}`);
process.exit(failed ? 1 : 0);
