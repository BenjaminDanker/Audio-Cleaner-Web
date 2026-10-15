// Decode -> denoise (workers) -> collect. Used by the Cleanup page.

import { decodeToChunks, TARGET_RATE } from './decode.js';
import { DenoiseEngine, denoiserLoadState } from './denoise.js';
import { interleave } from './wav.js';

const MAX_IN_FLIGHT = 3;

/** Stage text while the (one shared) noise-remover download is still running; the Cleanup page shows its progress instead. */
export const STAGE_LOADING = 'Getting the noise remover ready…';

/**
 * @param {File} file
 * @param {{attenDb:number, durationHint:number|null, signal?:AbortSignal,
 *   onProgress?:(p:{done:number,total:number|null,rate:number,eta:number|null,elapsed:number})=>void,
 *   onStage?:(s:string)=>void}} opts
 * @returns {Promise<{parts:Blob[], channels:number, frames:number, sampleRate:number, peak:number, rms:number,
 *   seconds:number, srcChannels:number, sourceRate:number, elapsed:number}>}
 */
export async function denoiseFile(file, { attenDb, durationHint, signal, onProgress, onStage }) {
  const started = performance.now();
  let engine = null;
  const abort = () => engine?.terminate();
  signal?.addEventListener('abort', abort, { once: true });
  try {
    const parts = [];
    let frames = 0;
    let peak = 0;
    let sumSq = 0;
    let channels = 1;
    let srcChannels = 1;
    let sourceRate = TARGET_RATE;
    let processed = 0; // input frames sent
    const inflight = [];
    let firstDone = 0; // time when first chunk completed (start of steady state)
    let firstDoneFrames = 0;

    const collect = (outs) => {
      const n = outs[0].length;
      if (n === 0) return;
      for (const ch of outs) {
        for (let i = 0; i < n; i++) {
          const v = ch[i];
          const a = v < 0 ? -v : v;
          if (a > peak) peak = a;
          sumSq += v * v;
        }
      }
      const inter = interleave(outs);
      parts.push(new Blob([inter], { type: 'application/octet-stream' }));
      frames += n;
    };

    const report = (done) => {
      if (!onProgress) return;
      const now = performance.now();
      const elapsed = (now - started) / 1000;
      const total = durationHint && durationHint > 0 ? durationHint : null;
      const sec = done / TARGET_RATE;
      let rate = 0;
      if (firstDone && done > firstDoneFrames) rate = (done - firstDoneFrames) / TARGET_RATE / ((now - firstDone) / 1000);
      const eta = total && rate > 0 ? Math.max(0, (total - sec) / rate) : null;
      onProgress({ done: sec, total, rate, eta, elapsed });
    };

    const settle = async (limit) => {
      while (inflight.length > limit) {
        const { promise, n } = inflight.shift();
        const outs = await promise;
        collect(outs);
        if (!firstDone) {
          firstDone = performance.now();
          firstDoneFrames = n;
        }
        report(n);
      }
    };

    onStage?.('Reading the file…');
    for await (const ev of decodeToChunks(file, { signal })) {
      if (signal?.aborted) throw new DOMException('Canceled', 'AbortError');
      if (ev.type === 'info') {
        channels = ev.channels;
        srcChannels = ev.srcChannels;
        sourceRate = ev.sourceRate;
        continue;
      }
      if (!engine) {
        // Normally the download started when the page opened and is long finished. If not, say so (never a second download:
        // the engine joins the one in progress).
        onStage?.(denoiserLoadState().done ? 'Warming up…' : STAGE_LOADING);
        engine = new DenoiseEngine({ channels, attenDb });
        if (signal?.aborted) throw new DOMException('Canceled', 'AbortError');
        await engine.ready();
        onStage?.('Cleaning…');
      }
      processed += ev.chunk[0].length;
      inflight.push({ promise: engine.process(ev.chunk), n: processed });
      await settle(MAX_IN_FLIGHT - 1);
    }
    if (!engine) throw new Error('No audio could be read from this file.');
    await settle(0);
    collect(await engine.flush());
    report(processed);
    if (signal?.aborted) throw new DOMException('Canceled', 'AbortError');
    const rms = Math.sqrt(sumSq / Math.max(1, frames * channels));
    return {
      parts, channels, frames, sampleRate: TARGET_RATE, peak, rms,
      seconds: frames / TARGET_RATE, srcChannels, sourceRate, elapsed: (performance.now() - started) / 1000,
    };
  } catch (e) {
    if (signal?.aborted) throw new DOMException('Canceled', 'AbortError');
    throw e;
  } finally {
    signal?.removeEventListener('abort', abort);
    engine?.terminate();
  }
}
