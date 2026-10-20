// Decode -> denoise (workers) -> collect. Used by the Cleanup page.

import { mixOf } from './blend.js';
import { decodeToChunks, TARGET_RATE } from './decode.js';
import { DenoiseEngine, denoiserLoadState } from './denoise.js';
import { Pairer } from './dfnstream.js';
import { interleave } from './wav.js';

const MAX_IN_FLIGHT = 3;
export const PREROLL_SECONDS = 5; // a part of a recording is cleaned from this much earlier, so the remover has settled when the part starts

function concat(list) {
  const out = new Float32Array(list.reduce((n, a) => n + a.length, 0));
  let o = 0;
  for (const a of list) { out.set(a, o); o += a.length; }
  return out;
}

/** Stage text while the (one shared) noise-remover download is still running; the Cleanup page shows its progress instead. */
export const STAGE_LOADING = 'Getting the noise remover ready…';

/**
 * The noise remover always runs at full strength; the strength `attenDb` is the blend of its output with the recording
 * (media/blend.js: exactly what the remover does at that limit). That way the full-strength output and the recording can be
 * kept (`keepPair`) and the strength changed afterwards, free, on this device.
 * A `range` ({start, end} seconds) cleans only that part: it is read from PREROLL_SECONDS earlier (the remover settles on the sound before
 * it), and what comes out is trimmed to the part, so the result is just as long as the part.
 * @param {File} file
 * @param {{attenDb:number, durationHint:number|null, signal?:AbortSignal, keepPair?:boolean, range?:{start:number,end:number},
 *   onProgress?:(p:{done:number,total:number|null,rate:number,eta:number|null,elapsed:number})=>void,
 *   onStage?:(s:string)=>void}} opts
 * @returns {Promise<{parts:Blob[], channels:number, frames:number, sampleRate:number, peak:number, rms:number,
 *   seconds:number, srcChannels:number, sourceRate:number, elapsed:number, pair:{full:Blob[], input:Blob[]}|null}>}
 *   `pair`: the full-strength output and the recording, interleaved float32 parts of the same length (only with keepPair)
 */
export async function denoiseFile(file, { attenDb, durationHint, signal, onProgress, onStage, keepPair = false, range = null }) {
  const started = performance.now();
  let engine = null;
  const abort = () => engine?.terminate();
  signal?.addEventListener('abort', abort, { once: true });
  try {
    const parts = [];
    const fullParts = [];
    const inputParts = [];
    const share = mixOf(attenDb); // the part of the recording that stays in the result
    let pairers = null; // per channel: lines the remover's answers up with the audio that went in
    let frames = 0;
    let peak = 0;
    let sumSq = 0;
    let channels = 1;
    let srcChannels = 1;
    let sourceRate = TARGET_RATE;
    let processed = 0; // input frames sent
    const from = range ? Math.max(0, range.start - PREROLL_SECONDS) : 0;
    let lead = range ? Math.round((range.start - from) * TARGET_RATE) : 0; // frames of the sound before the part still to drop from the answers
    const inflight = [];
    let firstDone = 0; // time when first chunk completed (start of steady state)
    let firstDoneFrames = 0;

    /** Line the answers `outs` up with the audio `sent` that went in (null at the end: the rest of it is already waiting). */
    const align = (outs, sent) => {
      const ys = [];
      const xs = [];
      for (let c = 0; c < outs.length; c++) {
        const y = [];
        const x = [];
        pairers[c].fn = (px, py) => { x.push(px.slice()); y.push(py.slice()); };
        pairers[c].feed(sent ? sent[c] : null, outs[c]);
        ys.push(y.length === 1 ? y[0] : concat(y));
        xs.push(x.length === 1 ? x[0] : concat(x));
      }
      return { ys, xs };
    };

    const collect = (aligned) => {
      let { ys, xs } = aligned;
      if (lead > 0) {
        const k = Math.min(lead, ys[0].length);
        lead -= k;
        if (k) { ys = ys.map((y) => y.subarray(k)); xs = xs.map((x) => x.subarray(k)); }
      }
      const n = ys[0].length;
      if (n === 0) return;
      const outs = share > 0 ? ys.map((y, c) => {
        const mix = new Float32Array(n);
        const x = xs[c];
        for (let i = 0; i < n; i++) mix[i] = (1 - share) * y[i] + share * x[i];
        return mix;
      }) : ys;
      for (const ch of outs) {
        for (let i = 0; i < n; i++) {
          const v = ch[i];
          const a = v < 0 ? -v : v;
          if (a > peak) peak = a;
          sumSq += v * v;
        }
      }
      parts.push(new Blob([interleave(outs)], { type: 'application/octet-stream' }));
      if (keepPair) {
        fullParts.push(new Blob([interleave(ys)], { type: 'application/octet-stream' }));
        inputParts.push(new Blob([interleave(xs)], { type: 'application/octet-stream' }));
      }
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
        const { promise, n, sent } = inflight.shift();
        const outs = await promise;
        collect(align(outs, sent));
        if (!firstDone) {
          firstDone = performance.now();
          firstDoneFrames = n;
        }
        report(n);
      }
    };

    onStage?.('Reading the file…');
    for await (const ev of decodeToChunks(file, range ? { signal, start: from, end: range.end } : { signal })) {
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
        engine = new DenoiseEngine({ channels, attenDb: 100 });
        pairers = Array.from({ length: channels }, () => new Pairer(() => {}));
        if (signal?.aborted) throw new DOMException('Canceled', 'AbortError');
        await engine.ready();
        onStage?.('Cleaning…');
      }
      processed += ev.chunk[0].length;
      const sent = ev.chunk.map((c) => c.slice()); // the worker takes the buffers: keep the recording for the blend
      inflight.push({ promise: engine.process(ev.chunk), n: processed, sent });
      await settle(MAX_IN_FLIGHT - 1);
    }
    if (!engine) throw new Error('No audio could be read from this file.');
    await settle(0);
    collect(align(await engine.flush(), null));
    report(processed);
    if (signal?.aborted) throw new DOMException('Canceled', 'AbortError');
    const rms = Math.sqrt(sumSq / Math.max(1, frames * channels));
    return {
      parts, channels, frames, sampleRate: TARGET_RATE, peak, rms,
      seconds: frames / TARGET_RATE, srcChannels, sourceRate, elapsed: (performance.now() - started) / 1000,
      pair: keepPair ? { full: fullParts, input: inputParts } : null,
    };
  } catch (e) {
    if (signal?.aborted) throw new DOMException('Canceled', 'AbortError');
    throw e;
  } finally {
    signal?.removeEventListener('abort', abort);
    engine?.terminate();
  }
}
