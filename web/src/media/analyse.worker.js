// Measures audio off the page's main thread (see analysis.js). Messages in:
//   {type:'start', channels}   {type:'segment'}   {type:'planes', planes:[Float32Array], rate}   {type:'finish'}
// Message out: {type:'result', stats} (or {type:'error', message}).
import { Analyser, RATE } from './analysis.js';
import { Resampler } from './resample.js';

let analyser = null;
let resamplers = null;
let rate = 0;

self.onmessage = (e) => {
  const m = e.data;
  try {
    if (m.type === 'start') {
      analyser = new Analyser({ channels: m.channels });
    } else if (m.type === 'segment') {
      resamplers = null;
      analyser.segment();
    } else if (m.type === 'planes') {
      if (!resamplers || rate !== m.rate) {
        rate = m.rate;
        resamplers = m.planes.map(() => new Resampler(rate, RATE));
      }
      analyser.push(m.planes.map((p, c) => resamplers[c].process(p)));
    } else if (m.type === 'finish') {
      self.postMessage({ type: 'result', stats: analyser.result() });
    }
  } catch (err) {
    self.postMessage({ type: 'error', message: String(err?.message || err) });
  }
};
