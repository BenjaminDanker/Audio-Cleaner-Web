// Pulls a video's sound out off the page's main thread (extract.js). Message in: {file, tracks}.
// Messages out: {type:'progress', frac} (at most 10 a second), then {type:'done', result} (extractAudioHere's result, null when the
// sound can't be pulled out) or {type:'error', message}.
import { extractAudioHere } from './extract.js';

self.onmessage = async (e) => {
  let last = 0;
  const onProgress = (frac) => {
    const now = performance.now();
    if (now - last < 100 && frac < 1) return;
    last = now;
    self.postMessage({ type: 'progress', frac });
  };
  try {
    const result = await extractAudioHere(e.data.file, { tracks: e.data.tracks, onProgress });
    self.postMessage({ type: 'done', result });
  } catch (err) {
    self.postMessage({ type: 'error', message: err?.message || String(err) });
  }
};
