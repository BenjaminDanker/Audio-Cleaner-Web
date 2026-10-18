// Node: the new model's decisions (out/reco-model.js, the JSON trees) for each measured clip. The yes/no heads use the current
// rules inside the model's uncertain band (0.4-0.6), as the product would. usage: node decide.mjs <model.json> <stats.jsonl> <out.jsonl> [rules.jsonl]
// Output per clip: {id, strength, heavy, rebuild, restoreHighs, uncertain, strengthProbs, probs, rawRebuild, rawRestore}
import { readFileSync, writeFileSync } from 'node:fs';
import { loadModel, decideFromModel, predict } from '/home/silver/audio-cleaner-train/out/reco-model.js';
const [modelPath, statsPath, outPath, rulesPath] = process.argv.slice(2);
const model = loadModel(readFileSync(modelPath, 'utf8'));
const rules = rulesPath ? Object.fromEntries(readFileSync(rulesPath, 'utf8').trim().split('\n').map((l) => { const r = JSON.parse(l); return [r.id, r.rules]; })) : {};
const out = [];
const seen = new Set();
for (const line of readFileSync(statsPath, 'utf8').trim().split('\n')) {
  const s = JSON.parse(line);
  if (seen.has(s.id)) continue;
  seen.add(s.id);
  const fb = () => (rules[s.id] ? { rebuild: !!rules[s.id].rebuild, restoreHighs: !!rules[s.id].restoreHighs } : { rebuild: false, restoreHighs: false });
  const d = decideFromModel(s, model, fb);
  const p = predict(model, s);
  out.push(JSON.stringify({ id: s.id, strength: d.strength, heavy: d.heavy, rebuild: d.rebuild, restoreHighs: d.restoreHighs,
    uncertain: d.uncertain, strengthProbs: d.strengthProbs, probs: d.probs, rawRebuild: p.rebuild > 0.5, rawRestore: p.restore > 0.5 }));
}
writeFileSync(outPath, out.join('\n') + '\n');
