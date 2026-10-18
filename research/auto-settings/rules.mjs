// Node: the CURRENT rules, i.e. the product's own recommendForFile (imported from the repo at run time, so it is always the
// code in web/src/recommend.js as it is now) on each measured clip. usage: node rules.mjs <stats.jsonl> <out.jsonl>
// Output per clip: {id, rules: {noise, strength, heavy, rebuild, restoreHighs, leveling, nothing, noSpeech} | null}.
// Duplicate ids in the stats file are read once.
import { readFileSync, writeFileSync } from 'node:fs';
import { recommendForFile } from '/home/silver/repos/acw-desktop-merge/web/src/recommend.js';
const lines = readFileSync(process.argv[2], 'utf8').trim().split('\n');
const out = [];
const seen = new Set();
for (const l of lines) {
  const s = JSON.parse(l);
  if (seen.has(s.id)) continue;
  seen.add(s.id);
  const r = recommendForFile(s, {});
  out.push(JSON.stringify({ id: s.id, rules: r ? { noise: r.noise, strength: r.strength, heavy: r.heavy, rebuild: r.rebuild, restoreHighs: r.restoreHighs, leveling: r.leveling, nothing: r.nothing, noSpeech: r.noSpeech } : null }));
}
writeFileSync(process.argv[3], out.join('\n') + '\n');
