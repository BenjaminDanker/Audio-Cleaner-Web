// "Test my room": listens to the microphone for ROOM_TEST_SECONDS while the person talks the way they do on stream (their game,
// music and fans as usual), then hands back what it heard; recommend.js roomChanges turns that into the settings that fit.
// Nothing is sent anywhere and nothing is billed.
import { Analyser } from '../media/analysis.js';
import captureUrl from './capture.worklet.js?url';

export const ROOM_TEST_SECONDS = 20;

/**
 * @param {{seconds?: number, onProgress?: (fraction: number) => void, signal?: AbortSignal}} opts
 * @returns {Promise<object>} Analyser.result() of what the microphone heard
 */
export async function testRoom({ seconds = ROOM_TEST_SECONDS, onProgress, signal } = {}) {
  // The same microphone settings as the live test: what is measured is what live would get.
  const stream = await navigator.mediaDevices.getUserMedia({
    audio: { channelCount: 1, sampleRate: 48000, echoCancellation: true, noiseSuppression: false, autoGainControl: false },
  });
  const ctx = new AudioContext({ sampleRate: 48000 });
  const stop = () => {
    stream.getTracks().forEach((t) => t.stop());
    ctx.close().catch(() => {});
  };
  try {
    if (ctx.sampleRate !== 48000) throw new Error('Your browser can’t run audio at 48 kHz, which live needs.');
    await ctx.audioWorklet.addModule(captureUrl);
    await ctx.resume();
    const capture = new AudioWorkletNode(ctx, 'acw-capture', { numberOfInputs: 1, numberOfOutputs: 1, outputChannelCount: [1] });
    const mute = ctx.createGain();
    mute.gain.value = 0;
    ctx.createMediaStreamSource(stream).connect(capture);
    capture.connect(mute);
    mute.connect(ctx.destination);
    const analyser = new Analyser({ channels: 1 });
    let heard = 0;
    return await new Promise((resolve, reject) => {
      signal?.addEventListener('abort', () => reject(new DOMException('The room test was stopped.', 'AbortError')), { once: true });
      capture.port.onmessage = (e) => {
        analyser.push([e.data]);
        heard += e.data.length / 48000;
        if (heard < seconds) {
          onProgress?.(heard / seconds);
          return;
        }
        capture.port.onmessage = null; // frames still on their way are not wanted
        resolve(analyser.result());
      };
    });
  } finally {
    stop();
  }
}
