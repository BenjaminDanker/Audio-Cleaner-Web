// Node: the browser's own measuring code (web/src/media/analysis.js, the same as tests/reco-cli.mjs) plus the site's
// noise remover (df.js + DeepFilterNet3) at four attenuation limits, so the outcome of each noise option can be scored.
// usage: node feat.mjs <manifest.json> <out.jsonl> <rawdir>
// manifest: [{id, path}] 48 kHz mono WAV. Output per clip: one JSON line with the Analyser stats (the 100 dB run is the one
// the browser's measuring worker uses) and the four cleaned signals written as float32 48 kHz to <rawdir>/<id>.f32
// (order: 12, 24, 40, 100 dB, each of the clip's length).
import { readFileSync, writeFileSync, appendFileSync } from 'node:fs';
import { RATE, Analyser } from '/home/silver/repos/acw-desktop-merge/web/src/media/analysis.js';
import { DfnStream, Pairer } from '/home/silver/repos/acw-desktop-merge/web/src/media/dfnstream.js';
import { readWav, loadDfn } from '/home/silver/repos/acw-desktop-merge/web/tests/reco-cli.mjs';

const [manifestPath, outPath, rawDir] = process.argv.slice(2);
const ATTEN = [12, 24, 40, 100];
const manifest = JSON.parse(readFileSync(manifestPath, 'utf8'));
const dfn = await loadDfn();

for (const { id, path } of manifest) {
  const { planes } = readWav(path);
  const x = planes[0];
  const frames = x.length;
  const a = new Analyser({ channels: 1 });
  const outs = [];
  for (const att of ATTEN) {
    const y = new Float32Array(frames);
    const cleaner = new DfnStream(dfn.mod, dfn.model, att);
    let got = 0;
    const pairs = new Pairer((px, py) => {
      if (att === 100) a.addCleaned(px, py);
      y.set(py.subarray(0, Math.min(py.length, frames - got)), got);
      got += py.length;
    });
    for (let p = 0; p < frames; p += RATE) {
      const piece = x.subarray(p, Math.min(frames, p + RATE));
      if (att === 100) a.push([piece]);
      pairs.feed(piece, cleaner.process(piece));
    }
    pairs.feed(null, cleaner.flush());
    cleaner.free();
    outs.push(y);
  }
  const stats = a.result();
  const buf = new Float32Array(frames * ATTEN.length);
  outs.forEach((y, k) => buf.set(y, k * frames));
  writeFileSync(`${rawDir}/${id}.f32`, Buffer.from(buf.buffer));
  appendFileSync(outPath, JSON.stringify({ id, ...stats }) + '\n');
}
