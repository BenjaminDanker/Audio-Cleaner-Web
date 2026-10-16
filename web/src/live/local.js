// The free live test: the same as startLive (session.js), but everything runs in this browser, nothing is sent anywhere.
// mic (48 kHz mono) -> capture worklet (20 ms frames) -> noise remover worker (the Cleanup page's, streaming) ->
// playback worklet; and the original mic, 16 kHz, cut into utterances at pauses -> the subtitle maker (speech.js).
// Events and controls match startLive, so the Live page shows both the same way:
//   onEvent({type:'ready', modelLatencyMs}) | {type:'caption', text, final, id} | {type:'error', message} | {type:'closed', clean}
//   onStats({rttAvg (processing delay, ms), sent, recv, bufferMs, underruns, dropped, inLevel, outLevel, elapsed})

import { DenoiseEngine } from '../media/denoise.js';
import { Resampler } from '../media/resample.js';
import { detectLanguage, preloadSpeech, transcribeSpeech } from '../media/speech.js';
import captureUrl from './capture.worklet.js?url';
import playbackUrl from './playback.worklet.js?url';

const RATE16 = 16000;
const VAD_FRAME = 320; // 20 ms at 16 kHz
const END_SILENCE_MS = 600; // a pause this long ends an utterance
const MAX_UTTERANCE_S = 10; // a long stretch of speech is cut here
const PARTIAL_EVERY_MS = 1500; // while someone talks, the line so far is written again this often (when the maker is free)

function peak(f32) {
  let p = 0;
  for (let i = 0; i < f32.length; i++) {
    const a = f32[i] < 0 ? -f32[i] : f32[i];
    if (a > p) p = a;
  }
  return p;
}

/** Cuts the mic into utterances at pauses and has each written as a caption (partials while it goes on). */
class Captioner {
  constructor(language, emit) {
    // 'mixed': someone who switches languages. Each finished line is detected again (one more pass of the model per
    // line, on this device), partial lines use the last language heard.
    this.mixed = language === 'mixed';
    this.language = this.mixed ? null : language || null;
    this.emit = emit;
    this.resampler = new Resampler(48000, RATE16);
    this.frame = new Float32Array(0);
    this.utt = []; // Float32Array pieces of the current utterance
    this.uttLen = 0;
    this.silentMs = 0;
    this.speaking = false;
    this.floor = 0.003; // running estimate of the room's level
    this.lastPartial = 0;
    this.busy = false;
    this.id = 0;
    this.queue = Promise.resolve();
    this.closed = false;
  }

  push(f48) {
    const f16 = this.resampler.process(f48);
    const merged = new Float32Array(this.frame.length + f16.length);
    merged.set(this.frame);
    merged.set(f16, this.frame.length);
    let i = 0;
    for (; i + VAD_FRAME <= merged.length; i += VAD_FRAME) this.vad(merged.subarray(i, i + VAD_FRAME));
    this.frame = merged.slice(i);
  }

  vad(f) {
    let s = 0;
    for (let i = 0; i < f.length; i++) s += f[i] * f[i];
    const rms = Math.sqrt(s / f.length);
    const voice = rms > Math.max(0.008, this.floor * 3);
    if (!voice) this.floor = this.floor * 0.98 + rms * 0.02;
    if (voice) {
      this.speaking = true;
      this.silentMs = 0;
    } else if (this.speaking) {
      this.silentMs += 20;
    }
    if (!this.speaking) return;
    this.utt.push(f.slice());
    this.uttLen += f.length;
    const seconds = this.uttLen / RATE16;
    if (this.silentMs >= END_SILENCE_MS || seconds >= MAX_UTTERANCE_S) {
      this.finish();
    } else if (!this.busy && seconds >= 1 && performance.now() - this.lastPartial >= PARTIAL_EVERY_MS) {
      this.lastPartial = performance.now();
      this.partial();
    }
  }

  audio() {
    const a = new Float32Array(this.uttLen);
    let o = 0;
    for (const p of this.utt) { a.set(p, o); o += p.length; }
    return a;
  }

  async lang(a) {
    if (!this.language) this.language = await detectLanguage(a);
    return this.language;
  }

  partial() {
    const a = this.audio();
    const id = this.id;
    this.busy = true;
    this.queue = this.queue.then(async () => {
      try {
        const segs = await transcribeSpeech(a, await this.lang(a));
        const text = segs.map((s) => s.text).join(' ').trim();
        if (!this.closed && id === this.id && text) this.emit({ type: 'caption', id: `u${id}`, text, final: false });
      } catch { /* a partial that fails is simply skipped */ } finally {
        this.busy = false;
      }
    });
  }

  finish() {
    const a = this.audio();
    const id = this.id++;
    this.utt = [];
    this.uttLen = 0;
    this.speaking = false;
    this.silentMs = 0;
    if (a.length < RATE16 * 0.3) return this.queue; // a click, not speech
    this.queue = this.queue.then(async () => {
      try {
        const language = this.mixed ? (this.language = await detectLanguage(a)) : await this.lang(a);
        const segs = await transcribeSpeech(a, language);
        const text = segs.map((s) => s.text).join(' ').trim();
        this.emit({ type: 'caption', id: `u${id}`, text, final: true }); // an empty final removes the partial
      } catch (e) {
        if (!this.closed) this.emit({ type: 'error', message: e?.message || 'The captions stopped working.' });
      }
    });
    return this.queue;
  }

  async close() {
    if (this.speaking) await this.finish();
    await this.queue;
    this.closed = true;
  }
}

/**
 * @param {{denoise: boolean, captions: boolean, language: string|null, attenDb: number, monitor: 'off'|'original'|'processed',
 *   onEvent: (ev:object)=>void, onStats:(s:object)=>void}} opts
 */
export async function startLocalLive(opts) {
  const { onEvent, onStats } = opts;
  let monitor = opts.monitor;
  let stream = null;
  let ctx = null;
  let capture = null;
  let playback = null;
  let engine = null;
  let captioner = null;
  let closed = false;
  let stopping = false;
  let sent = 0;
  let recv = 0;
  let inLevel = 0;
  let outLevel = 0;
  const delays = [];
  let bufferMs = 0;
  let underruns = 0;
  let dropped = 0;
  const startedAt = performance.now();
  let statTimer = null;

  const emitStats = () => {
    const avg = delays.length ? delays.reduce((a, b) => a + b, 0) / delays.length : 0;
    onStats({ rttAvg: avg, rttLast: delays.at(-1) || 0, sent, recv, bufferMs, underruns, dropped, inLevel, outLevel, elapsed: (performance.now() - startedAt) / 1000 });
  };
  const cleanup = () => {
    if (closed) return;
    closed = true;
    clearInterval(statTimer);
    try { capture && (capture.port.onmessage = null); } catch { /* ignore */ }
    stream?.getTracks().forEach((t) => t.stop());
    try { capture?.disconnect(); } catch { /* ignore */ }
    try { playback?.disconnect(); } catch { /* ignore */ }
    ctx?.close().catch(() => {});
    engine?.terminate();
  };

  try {
    // Both engines start downloading (once; then from the browser's cache) while the microphone is asked for.
    const ready = [];
    if (opts.denoise) {
      engine = new DenoiseEngine({ channels: 1, attenDb: opts.attenDb });
      ready.push(engine.ready());
    }
    if (opts.captions) ready.push(preloadSpeech());
    stream = await navigator.mediaDevices.getUserMedia({
      audio: { channelCount: 1, sampleRate: 48000, echoCancellation: true, noiseSuppression: false, autoGainControl: false },
    });
    await Promise.all(ready);
    ctx = new AudioContext({ sampleRate: 48000, latencyHint: 'interactive' });
    if (ctx.sampleRate !== 48000) throw new Error('Your browser can\'t run audio at 48 kHz, which live needs.');
    await ctx.audioWorklet.addModule(captureUrl);
    await ctx.audioWorklet.addModule(playbackUrl);
    await ctx.resume();
    const src = ctx.createMediaStreamSource(stream);
    capture = new AudioWorkletNode(ctx, 'acw-capture', { numberOfInputs: 1, numberOfOutputs: 1, outputChannelCount: [1] });
    playback = new AudioWorkletNode(ctx, 'acw-playback', { numberOfInputs: 0, numberOfOutputs: 1, outputChannelCount: [1], processorOptions: { prefill: 2880 } });
    const mute = ctx.createGain();
    mute.gain.value = 0;
    src.connect(capture);
    capture.connect(mute);
    mute.connect(ctx.destination);
    playback.connect(ctx.destination);
    if (opts.captions) captioner = new Captioner(opts.language, (ev) => { if (!closed || ev.final) onEvent(ev); });

    playback.port.onmessage = (e) => {
      if (e.data.type === 'stats') {
        bufferMs = (e.data.queuedSamples / 48000) * 1000;
        underruns = e.data.underruns;
        dropped = e.data.dropped;
      }
    };
    capture.port.onmessage = (e) => {
      if (stopping) return;
      const f32 = e.data;
      inLevel = Math.max(inLevel * 0.85, peak(f32));
      sent++;
      if (monitor === 'original') playback.port.postMessage({ type: 'frame', samples: f32.slice() });
      captioner?.push(f32);
      if (engine) {
        const t0 = performance.now();
        engine.process([f32.slice()]).then(([out]) => {
          if (closed || !out.length) return;
          delays.push(performance.now() - t0);
          if (delays.length > 50) delays.shift();
          recv++;
          outLevel = Math.max(outLevel * 0.85, peak(out));
          if (monitor === 'processed') playback.port.postMessage({ type: 'frame', samples: out }, [out.buffer]);
        }, () => {});
      } else {
        recv++;
        outLevel = inLevel;
        if (monitor === 'processed') playback.port.postMessage({ type: 'frame', samples: f32.slice() });
      }
    };
    statTimer = setInterval(emitStats, 250);
    onEvent({ type: 'ready', modelLatencyMs: engine ? 30 : 0, local: true });
  } catch (e) {
    cleanup();
    throw e;
  }

  return {
    setMonitor(mode) {
      monitor = mode;
      playback?.port.postMessage({ type: 'clear' });
    },
    setAtten(db) {
      engine?.setAtten(db);
    },
    stop() {
      if (closed || stopping) return;
      stopping = true;
      stream?.getTracks().forEach((t) => t.stop());
      const done = captioner ? captioner.close() : Promise.resolve();
      // the last words are written before the test ends (not longer than a few seconds)
      Promise.race([done, new Promise((r) => setTimeout(r, 6000))]).finally(() => {
        cleanup();
        onEvent({ type: 'closed', clean: true });
      });
    },
  };
}
