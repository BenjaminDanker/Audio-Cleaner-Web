// Measures audio off the page's main thread (see analysis.js), the noise remover included. Messages in:
//   {type:'start', channels, assets:{module, model, dfnBase, version}}   {type:'segment'}
//   {type:'planes', planes:[Float32Array], rate}   {type:'finish'}
// Message out: {type:'result', stats} (or {type:'error', message}).
import { Analyser, RATE } from './analysis.js';
import { DfnStream, loadDfnModule, Pairer } from './dfnstream.js';
import { Resampler } from './resample.js';

let analyser = null;
let resamplers = null;
let rate = 0;
let mod = null;
let model = null;
let cleaner = null; // the noise remover on the current segment: a fresh one per segment, as pipeline.noise_figure_db runs it
let pairs = null;

function endSegment() {
  if (!cleaner) return;
  pairs.feed(null, cleaner.flush());
  cleaner.free();
  cleaner = null;
}

let ready = null;

self.onmessage = async (e) => {
  const m = e.data;
  try {
    if (m.type === 'start') {
      analyser = new Analyser({ channels: m.channels });
      model = m.assets.model;
      ready = loadDfnModule(m.assets.dfnBase, m.assets.version, m.assets.module).then((x) => { mod = x; });
    }
    await ready; // messages wait for the noise remover; their order is kept (each awaits the same promise)
    if (m.type === 'segment') {
      endSegment();
      resamplers = null;
      analyser.segment();
    } else if (m.type === 'planes') {
      if (!resamplers || rate !== m.rate) {
        rate = m.rate;
        resamplers = m.planes.map(() => new Resampler(rate, RATE));
      }
      const planes = m.planes.map((p, c) => resamplers[c].process(p));
      analyser.push(planes);
      let mono = planes[0];
      if (planes.length > 1) {
        mono = new Float32Array(planes[0].length);
        for (let i = 0; i < mono.length; i++) mono[i] = (planes[0][i] + planes[1][i]) * 0.5;
      }
      if (!cleaner) {
        cleaner = new DfnStream(mod, model, 100);
        pairs = new Pairer((x, y) => analyser.addCleaned(x, y));
      }
      pairs.feed(mono, cleaner.process(mono));
    } else if (m.type === 'finish') {
      endSegment();
      self.postMessage({ type: 'result', stats: analyser.result() });
    }
  } catch (err) {
    self.postMessage({ type: 'error', message: String(err?.message || err) });
  }
};
