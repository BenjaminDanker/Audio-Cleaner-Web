// "Test my room": listens to the microphone for ROOM_TEST_SECONDS while the person talks the way they do on stream (their game,
// music and fans as usual), then hands back what it heard; recommend.js roomChanges turns that into the settings that fit.
// The noise is measured by the noise remover itself (the free in-browser one, at full strength: analysis.js NoiseFigure), so
// wind, keyboards and chatter count, not just a steady hum. Nothing is sent anywhere and nothing is billed.
import { Analyser } from '../media/analysis.js';
import { DenoiseEngine } from '../media/denoise.js';
import captureUrl from './capture.worklet.js?url';

export const ROOM_TEST_SECONDS = 20;

const join = (parts) => {
  const out = new Float32Array(parts.reduce((n, p) => n + p.length, 0));
  let at = 0;
  for (const p of parts) { out.set(p, at); at += p.length; }
  return out;
};

/**
 * @param {{seconds?: number, onListening?: () => void, onProgress?: (fraction: number) => void, signal?: AbortSignal}} opts
 *   onListening: the noise remover is ready (fetched once, the first time) and the microphone is being heard
 * @returns {Promise<object>} Analyser.result() of what the microphone heard
 */
export async function testRoom({ seconds = ROOM_TEST_SECONDS, onListening, onProgress, signal } = {}) {
  const engine = new DenoiseEngine({ channels: 1, attenDb: 100 });
  let stream = null;
  let ctx = null;
  const stop = () => {
    engine.terminate();
    stream?.getTracks().forEach((t) => t.stop());
    ctx?.close().catch(() => {});
  };
  const aborted = () => new Promise((_, reject) => {
    if (signal?.aborted) reject(new DOMException('The room test was stopped.', 'AbortError'));
    signal?.addEventListener('abort', () => reject(new DOMException('The room test was stopped.', 'AbortError')), { once: true });
  });
  try {
    await Promise.race([engine.ready(), aborted()]);
    // The same microphone settings as the live test: what is measured is what live would get.
    stream = await navigator.mediaDevices.getUserMedia({
      audio: { channelCount: 1, sampleRate: 48000, echoCancellation: true, noiseSuppression: false, autoGainControl: false },
    });
    ctx = new AudioContext({ sampleRate: 48000 });
    if (ctx.sampleRate !== 48000) throw new Error('Your browser can’t run audio at 48 kHz, which live needs.');
    await ctx.audioWorklet.addModule(captureUrl);
    await ctx.resume();
    const capture = new AudioWorkletNode(ctx, 'acw-capture', { numberOfInputs: 1, numberOfOutputs: 1, outputChannelCount: [1] });
    const mute = ctx.createGain();
    mute.gain.value = 0;
    ctx.createMediaStreamSource(stream).connect(capture);
    capture.connect(mute);
    mute.connect(ctx.destination);
    onListening?.();
    const analyser = new Analyser({ channels: 1 });
    const heardParts = [];
    const cleanedParts = []; // the noise remover's answers, in order
    let heard = 0;
    await Promise.race([aborted(), new Promise((resolve, reject) => {
      capture.port.onmessage = (e) => {
        const x = e.data;
        analyser.push([x]);
        heardParts.push(x);
        engine.process([x.slice()]).then(([y]) => cleanedParts.push(y), reject);
        heard += x.length / 48000;
        if (heard < seconds) {
          onProgress?.(heard / seconds);
          return;
        }
        capture.port.onmessage = null; // frames still on their way are not wanted
        resolve();
      };
    })]);
    const [rest] = await Promise.race([engine.flush(), aborted()]);
    cleanedParts.push(rest);
    analyser.addCleaned(join(heardParts), join(cleanedParts));
    return analyser.result();
  } finally {
    stop();
  }
}
