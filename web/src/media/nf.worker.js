// Runs the noise remover (full strength) over stretches of audio and answers what it keeps and what it takes out, as sums of squares: the
// noise figure of suggest-core.js is made of these. Several of these work side by side (suggest-run.js). Messages in:
//   {type:'start', assets:{module, model, dfnBase, version}}   {type:'run', id, mono}      out: {type:'sums', id, voice, removed}  {type:'error', message}
import { DfnStream, loadDfnModule } from './dfnstream.js';

let mod = null;
let model = null;
let ready = null;

export function sumsFor(mod2, model2, mono) {
  const s = new DfnStream(mod2, model2, 100);
  try {
    const first = s.process(mono);
    const rest = s.flush();
    let voice = 0;
    let removed = 0;
    let at = 0;
    for (const y of [first, rest]) {
      for (let i = 0; i < y.length && at + i < mono.length; i++) {
        const v = y[i];
        const d = mono[at + i] - v;
        voice += v * v;
        removed += d * d;
      }
      at += y.length;
    }
    return { voice, removed };
  } finally {
    s.free();
  }
}

self.onmessage = async (e) => {
  const m = e.data;
  try {
    if (m.type === 'start') {
      model = m.assets.model;
      ready = loadDfnModule(m.assets.dfnBase, m.assets.version, m.assets.module).then((x) => { mod = x; });
    }
    await ready;
    if (m.type === 'run') self.postMessage({ type: 'sums', id: m.id, ...sumsFor(mod, model, m.mono) });
  } catch (err) {
    self.postMessage({ type: 'error', id: m.id, message: String(err?.message || err) });
  }
};
