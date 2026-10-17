// Looks at a chosen file for "Recommended for this file": about 60 s of its sound, spread over the whole file, measured
// (analysis.js) in a worker so the page stays smooth, with the noise remover the free cleanup uses (fetched once; nothing is
// uploaded). Never throws: a file that can't be read, or a noise remover that can't be fetched, just gets no recommendation.

import { denoiserAssets } from './denoise.js';
import { planWindows } from './analysis.js';
import { libavAvailable, libavFrames } from './libav.js';
import { openInput } from './mb.js';

const TOTAL_SECONDS = 60;
const WINDOWS = 6;
const LIBAV_SECONDS = 30; // libav.js can't jump around in a file: only its beginning is looked at

/** One measuring job in a worker: send audio in, get the numbers out (the noise remover runs in there too). */
function runner(channels, assets) {
  const worker = new Worker(new URL('./analyse.worker.js', import.meta.url), { type: 'module' });
  let settle = null;
  const done = new Promise((resolve, reject) => { settle = { resolve, reject }; });
  worker.onmessage = (e) => {
    if (e.data.type === 'result') settle.resolve(e.data.stats);
    else if (e.data.type === 'error') settle.reject(new Error(e.data.message));
  };
  worker.onerror = (e) => settle.reject(new Error(e.message || 'worker failed'));
  worker.postMessage({ type: 'start', channels, assets });
  return {
    push: (planes, rate) => worker.postMessage({ type: 'planes', planes, rate }, planes.map((p) => p.buffer)),
    segment: () => worker.postMessage({ type: 'segment' }),
    finish: () => { worker.postMessage({ type: 'finish' }); return done.finally(() => worker.terminate()); },
    cancel: () => worker.terminate(),
  };
}

const slice = () => new Promise((r) => setTimeout(r, 0));

async function viaMediabunny(file, info, assets, signal) {
  const { mb, input } = await openInput(file);
  try {
    const track = await input.getPrimaryAudioTrack();
    if (!track || !(await track.canDecode())) return null;
    const src = await track.getNumberOfChannels();
    const channels = src === 2 ? 2 : 1;
    const run = runner(channels, assets);
    try {
      const sink = new mb.AudioSampleSink(track);
      const duration = info?.duration || (await input.computeDuration());
      const windows = planWindows(duration, TOTAL_SECONDS, WINDOWS);
      let first = true;
      for (const w of windows) {
        if (!first) run.segment();
        first = false;
        for await (const sample of sink.samples(w.start, w.start + w.length)) {
          if (signal?.aborted) throw new DOMException('Canceled', 'AbortError');
          try {
            const n = sample.numberOfFrames;
            const planes = [];
            for (let c = 0; c < sample.numberOfChannels; c++) {
              const buf = new Float32Array(n);
              sample.copyTo(buf, { planeIndex: c, format: 'f32-planar' });
              planes.push(buf);
            }
            let out = planes;
            if (channels === 1 && planes.length > 1) {
              const m = new Float32Array(n);
              for (const p of planes) for (let i = 0; i < n; i++) m[i] += p[i];
              for (let i = 0; i < n; i++) m[i] /= planes.length;
              out = [m];
            } else out = planes.slice(0, channels);
            run.push(out, sample.sampleRate);
          } finally {
            sample.close();
          }
        }
        await slice();
      }
      return await run.finish();
    } catch (e) {
      run.cancel();
      throw e;
    }
  } finally {
    input.dispose();
  }
}

async function viaLibav(file, assets, signal) {
  let run = null;
  let seconds = 0;
  try {
    for await (const ev of libavFrames(file, { signal })) {
      if (ev.type === 'info') { run = runner(ev.srcChannels === 2 ? 2 : 1, assets); continue; }
      const channels = run ? (ev.planes.length === 2 ? 2 : 1) : 1;
      const planes = channels === 1 && ev.planes.length > 1
        ? [ev.planes[0].map((_, i) => ev.planes.reduce((s, p) => s + p[i], 0) / ev.planes.length)]
        : ev.planes.slice(0, channels);
      run.push(planes, ev.rate);
      seconds += ev.planes[0].length / ev.rate;
      if (seconds >= LIBAV_SECONDS) break;
      if (signal?.aborted) throw new DOMException('Canceled', 'AbortError');
    }
    return run ? await run.finish() : null;
  } catch (e) {
    run?.cancel();
    throw e;
  }
}

/**
 * @param {File} file
 * @param {object} info what probeMedia() found
 * @param {{signal?: AbortSignal}} opts
 * @returns {Promise<object|null>} the measurements (analysis.js: Analyser.result), or null when the file couldn't be read
 */
export async function analyseFile(file, info, { signal } = {}) {
  try {
    if (info?.hasAudio === false) return null;
    let stats = null;
    const mediabunny = info?.viaMediabunny && info.canDecode !== false;
    if (!mediabunny && !(info?.viaLibav && libavAvailable())) return null;
    const assets = await denoiserAssets();
    if (mediabunny) stats = await viaMediabunny(file, info, assets, signal);
    else stats = await viaLibav(file, assets, signal);
    if (stats && !(stats.seconds > 0.5)) return null;
    return stats;
  } catch (e) {
    if (e?.name === 'AbortError') throw e;
    return null;
  }
}
