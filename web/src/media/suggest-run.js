// Suggestions for a file that has not been uploaded yet, found in the browser (suggest-core.js in a worker): the page opens the file
// and answers the worker's requests for audio. Cancelable, with progress. Files up to MAX_SECONDS that this browser can open itself.

import { denoiserAssets } from './denoise.js';
import { openInput } from './mb.js';
import { Resampler } from './resample.js';
import { MAX_SECONDS } from './suggest-core.js';

export { MAX_SECONDS };
const RATE = 48000;
const PAD = 0.05; // seconds read on each side of a stretch so the resampler's edges fall outside it

/** Can this browser find suggestions for this file itself? `why`: the plain words when it can't. */
export function suggestVerdict(info, seconds) {
  if (typeof Worker === 'undefined' || typeof WebAssembly === 'undefined') return { ok: false, why: 'This browser can’t look for suggestions.' };
  if (info?.hasAudio === false) return { ok: false, why: 'This file has no audio track.' };
  if (!info?.viaMediabunny || info.canDecode === false) return { ok: false, why: 'Suggestions appear after the first run.' };
  if (!(seconds > 0)) return { ok: false, why: 'Suggestions appear after the first run.' };
  if (seconds > MAX_SECONDS) return { ok: false, why: 'Suggestions appear after the first run.' };
  if (seconds < 4) return { ok: false, why: 'This is too short for suggestions.' };
  return { ok: true, why: '' };
}

/** A reader of the file: read(a, b) answers samples a to b (48 kHz) as one Float32Array per channel (mono or stereo). */
async function openReader(file) {
  const { mb, input } = await openInput(file);
  const track = await input.getPrimaryAudioTrack();
  if (!track || !(await track.canDecode())) { input.dispose(); throw new Error('This browser can’t open the sound of this file.'); }
  const rate = await track.getSampleRate();
  const src = await track.getNumberOfChannels();
  const channels = src === 2 ? 2 : 1;
  const sink = new mb.AudioSampleSink(track);
  return {
    channels,
    dispose: () => input.dispose(),
    async read(a, b) {
      const t0 = a / RATE;
      const t1 = b / RATE;
      const from = Math.max(0, t0 - PAD);
      const frames = Math.ceil((t1 + PAD - from) * rate) + 2;
      const bufs = Array.from({ length: channels }, () => new Float32Array(frames));
      for await (const sample of sink.samples(from, t1 + PAD)) {
        try {
          const at = Math.round((sample.timestamp - from) * rate);
          const n = sample.numberOfFrames;
          const planes = [];
          for (let c = 0; c < sample.numberOfChannels; c++) {
            const p = new Float32Array(n);
            sample.copyTo(p, { planeIndex: c, format: 'f32-planar' });
            planes.push(p);
          }
          for (let i = Math.max(0, -at); i < n && at + i < frames; i++) {
            if (channels === 1) {
              let v = 0;
              for (const p of planes) v += p[i];
              bufs[0][at + i] += v / planes.length;
            } else {
              bufs[0][at + i] = planes[0][i];
              bufs[1][at + i] = planes[1][i];
            }
          }
        } finally {
          sample.close();
        }
      }
      const skip = Math.round((t0 - from) * RATE);
      const want = b - a;
      return bufs.map((buf) => {
        const rs = new Resampler(rate, RATE);
        const out = rs.process(buf);
        const tail = rs.finish();
        const all = new Float32Array(out.length + tail.length);
        all.set(out, 0);
        all.set(tail, out.length);
        const cut = new Float32Array(want);
        cut.set(all.subarray(skip, skip + want));
        return cut;
      });
    },
  };
}

/**
 * Starts looking. Returns {done: Promise<result>, cancel()}. result: {seconds, whole, parts, suggestions} as the server's suggestions
 * are (docs/api.md section 8). `onProgress(0..1)`, `onPeaks(from, peaks)` (per quarter second, for the waveform).
 */
export function startSuggestions(file, info, { seconds, onProgress, onPeaks } = {}) {
  let worker = null;
  let reader = null;
  let pool = [];
  let settle = null;
  let canceled = false;
  let ended = false;
  const done = new Promise((resolve, reject) => { settle = { resolve, reject }; });
  const finish = () => { ended = true; worker?.terminate(); for (const p of pool) p.worker.terminate(); reader?.dispose(); };
  const fail = (err) => { if (!ended) { settle.reject(err); finish(); } };
  // the noise remover runs in a few helper workers side by side; each stretch goes to the next free one
  const queue = [];
  const give = () => {
    while (queue.length) {
      const free = pool.find((p) => !p.busy);
      if (!free) return;
      const job = queue.shift();
      free.busy = job;
      free.worker.postMessage({ type: 'run', id: job.id, mono: job.mono }, [job.mono.buffer]);
    }
  };
  (async () => {
    try {
      const assets = await denoiserAssets();
      reader = await openReader(file);
      if (canceled) throw new DOMException('Canceled', 'AbortError');
      const size = Math.max(1, Math.min(4, (navigator.hardwareConcurrency || 2) - 1));
      pool = Array.from({ length: size }, () => {
        const w = new Worker(new URL('./nf.worker.js', import.meta.url), { type: 'module' });
        const p = { worker: w, busy: null };
        w.onmessage = (ev) => {
          const m = ev.data;
          if (m.type === 'error') { fail(new Error(m.message)); return; }
          const job = p.busy;
          p.busy = null;
          if (job) worker?.postMessage({ type: 'sums', id: job.id, voice: m.voice, removed: m.removed });
          give();
        };
        w.onerror = (ev) => fail(new Error(ev.message || 'The suggestion worker failed.'));
        w.postMessage({ type: 'start', assets });
        return p;
      });
      worker = new Worker(new URL('./suggest.worker.js', import.meta.url), { type: 'module' });
      worker.onmessage = async (e) => {
        const m = e.data;
        if (m.type === 'read') {
          try {
            const planes = await reader.read(m.a, m.b);
            worker.postMessage({ type: 'audio', id: m.id, planes }, planes.map((p) => p.buffer));
          } catch (err) {
            fail(err);
          }
        } else if (m.type === 'nf') { queue.push({ id: m.id, mono: m.mono }); give(); }
        else if (m.type === 'progress') onProgress?.(m.f);
        else if (m.type === 'peaks') onPeaks?.(m.from, m.peaks);
        else if (m.type === 'result') { settle.resolve(m.result); finish(); }
        else if (m.type === 'canceled') fail(new DOMException('Canceled', 'AbortError'));
        else if (m.type === 'error') fail(new Error(m.message));
      };
      worker.onerror = (ev) => fail(new Error(ev.message || 'The suggestion worker failed.'));
      worker.postMessage({ type: 'start', seconds: seconds ?? info?.duration ?? 0, concurrency: size });
    } catch (err) {
      fail(err);
    }
  })();
  return {
    done,
    cancel() {
      canceled = true;
      queue.length = 0;
      if (worker) worker.postMessage({ type: 'cancel' });
      else fail(new DOMException('Canceled', 'AbortError'));
    },
  };
}
