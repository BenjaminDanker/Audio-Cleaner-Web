// Finds suggestions off the page's main thread (suggest-core.js), the noise remover included. The page reads the file (it has the
// decoders) and hands the audio over on request. Messages in:
//   {type:'start', seconds, concurrency}   {type:'audio', id, planes}   {type:'sums', id, voice, removed}   {type:'cancel'}
// Messages out: {type:'read', id, a, b}  {type:'nf', id, mono}  {type:'progress', f}  {type:'peaks', from, peaks}  {type:'result', result}  {type:'canceled'}  {type:'error', message}
import { findSuggestions } from './suggest-core.js';

let ctl = null;
let nextId = 1;
const pending = new Map();

const read = (a, b) => new Promise((resolve, reject) => {
  const id = nextId++;
  pending.set(id, { resolve, reject });
  self.postMessage({ type: 'read', id, a, b });
});

/** The noise remover over one stretch runs in one of the page's helper workers (suggest-run.js): the page answers with the sums. */
const sums = (mono) => new Promise((resolve, reject) => {
  const id = nextId++;
  pending.set(id, { resolve, reject });
  const copy = mono.slice();
  self.postMessage({ type: 'nf', id, mono: copy }, [copy.buffer]);
});

self.onmessage = async (e) => {
  const m = e.data;
  if (m.type === 'audio') {
    pending.get(m.id)?.resolve(m.planes);
    pending.delete(m.id);
  } else if (m.type === 'sums') {
    pending.get(m.id)?.resolve({ voice: m.voice, removed: m.removed });
    pending.delete(m.id);
  } else if (m.type === 'cancel') {
    ctl?.abort();
    for (const p of pending.values()) p.reject(new DOMException('Canceled', 'AbortError'));
    pending.clear();
  } else if (m.type === 'start') {
    ctl = new AbortController();
    try {
      const result = await findSuggestions({
        seconds: m.seconds,
        read,
        noiseSums: sums,
        concurrency: m.concurrency || 1,
        signal: ctl.signal,
        onProgress: (f) => self.postMessage({ type: 'progress', f }),
        onPeaks: (from, peaks) => self.postMessage({ type: 'peaks', from, peaks }, [peaks.buffer]),
      });
      self.postMessage({ type: 'result', result });
    } catch (err) {
      self.postMessage(err?.name === 'AbortError' ? { type: 'canceled' } : { type: 'error', message: String(err?.message || err) });
    }
  }
};
