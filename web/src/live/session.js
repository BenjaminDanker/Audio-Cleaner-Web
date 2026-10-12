// Browser live-denoise demo client for GET /api/stream/ws (docs/streaming-protocol.md).
// mic (48 kHz mono, all browser DSP off) -> capture worklet (20 ms frames) -> WebSocket (pcm_s16le + 16-byte header)
// <- processed frames -> playback worklet (jitter buffer) -> speakers

import { streamUrl } from '../api.js';
import captureUrl from './capture.worklet.js?url';
import playbackUrl from './playback.worklet.js?url';

const FRAME = 960;
const HEADER = 16;

function peak(f32) {
  let p = 0;
  for (let i = 0; i < f32.length; i++) {
    const a = f32[i] < 0 ? -f32[i] : f32[i];
    if (a > p) p = a;
  }
  return p;
}

/**
 * @param {{denoise: boolean, captions: 'local'|null, language: string|null, attenDb: number, monitor: 'off'|'original'|'processed',
 *   onEvent: (ev:object)=>void, onStats:(s:object)=>void}} opts
 */
export async function startLive(opts) {
  const { onEvent, onStats } = opts;
  let monitor = opts.monitor;
  let stream = null;
  let ctx = null;
  let ws = null;
  let capture = null;
  let playback = null;
  let closed = false;
  let stopping = false;
  let ready = false;
  let seq = 0;
  let sent = 0;
  let recv = 0;
  let inLevel = 0;
  let outLevel = 0;
  const rtts = [];
  let lastRtt = 0;
  let bufferMs = 0;
  let underruns = 0;
  let dropped = 0;
  let lastStats = 0;
  const startedAt = performance.now();
  let statTimer = null;

  const cleanup = () => {
    if (closed) return;
    closed = true;
    clearInterval(statTimer);
    try { capture?.port && (capture.port.onmessage = null); } catch { /* ignore */ }
    stream?.getTracks().forEach((t) => t.stop());
    try { capture?.disconnect(); } catch { /* ignore */ }
    try { playback?.disconnect(); } catch { /* ignore */ }
    ctx?.close().catch(() => {});
  };

  const emitStats = () => {
    const avg = rtts.length ? rtts.reduce((a, b) => a + b, 0) / rtts.length : 0;
    onStats({
      rttAvg: avg, rttLast: lastRtt, sent, recv, bufferMs, underruns, dropped,
      inLevel, outLevel, elapsed: (performance.now() - startedAt) / 1000,
    });
  };

  try {
    stream = await navigator.mediaDevices.getUserMedia({
      audio: { channelCount: 1, sampleRate: 48000, echoCancellation: false, noiseSuppression: false, autoGainControl: false },
    });
    ctx = new AudioContext({ sampleRate: 48000, latencyHint: 'interactive' });
    if (ctx.sampleRate !== 48000) throw new Error('Your browser cannot run audio at 48 kHz, which live processing needs.');
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

    ws = new WebSocket(streamUrl());
    ws.binaryType = 'arraybuffer';

    await new Promise((resolve, reject) => {
      const fail = (msg) => reject(new Error(msg));
      const timer = setTimeout(() => fail('The server did not answer in time.'), 15000);
      ws.onopen = () => {
        ws.send(JSON.stringify({
          type: 'start', version: 1, denoise: opts.denoise ? 'light' : null, captions: opts.captions || null,
          language: opts.language || null, sampleRate: 48000, attenLimDb: opts.attenDb, client: 'acw-web-demo/1.0',
        }));
      };
      ws.onerror = () => { clearTimeout(timer); fail('Could not connect to the live service.'); };
      ws.onclose = (e) => { clearTimeout(timer); fail(e.code === 1008 ? 'The live service refused the connection. Are you signed in?' : 'The connection closed before it started.'); };
      ws.onmessage = (e) => {
        if (typeof e.data !== 'string') return;
        const m = JSON.parse(e.data);
        if (m.type === 'ready') {
          clearTimeout(timer);
          ready = true;
          resolve(m);
          onEvent(m);
        } else if (m.type === 'error') {
          clearTimeout(timer);
          const err = new Error(m.message || 'The live service reported an error.');
          err.code = m.code;
          reject(err);
        }
      };
    });

    // From here on: steady-state handlers.
    ws.onerror = () => onEvent({ type: 'error', code: 'socket', message: 'The connection to the live service failed.' });
    ws.onclose = () => {
      cleanup();
      onEvent({ type: 'closed', clean: stopping });
    };
    ws.onmessage = (e) => {
      if (typeof e.data === 'string') {
        try { onEvent(JSON.parse(e.data)); } catch { /* ignore */ }
        return;
      }
      const buf = e.data;
      if (buf.byteLength <= HEADER) return;
      const dv = new DataView(buf);
      const codec = dv.getUint8(1);
      const ts = Number(dv.getBigUint64(8, true));
      lastRtt = performance.now() - ts / 1000;
      rtts.push(lastRtt);
      if (rtts.length > 50) rtts.shift();
      recv++;
      let f32;
      if (codec === 1) {
        const i16 = new Int16Array(buf, HEADER, (buf.byteLength - HEADER) >> 1);
        f32 = new Float32Array(i16.length);
        for (let i = 0; i < i16.length; i++) f32[i] = i16[i] / 32768;
      } else {
        f32 = new Float32Array(buf.slice(HEADER));
      }
      outLevel = Math.max(outLevel * 0.85, peak(f32));
      if (monitor === 'processed') playback.port.postMessage({ type: 'frame', samples: f32 }, [f32.buffer]);
    };

    playback.port.onmessage = (e) => {
      if (e.data.type === 'stats') {
        bufferMs = (e.data.queuedSamples / 48000) * 1000;
        underruns = e.data.underruns;
        dropped = e.data.dropped;
      }
    };

    capture.port.onmessage = (e) => {
      const f32 = e.data;
      inLevel = Math.max(inLevel * 0.85, peak(f32));
      if (monitor === 'original') playback.port.postMessage({ type: 'frame', samples: f32.slice() });
      if (!ready || stopping || ws.readyState !== WebSocket.OPEN) return;
      const out = new ArrayBuffer(HEADER + FRAME * 2);
      const dv = new DataView(out);
      dv.setUint8(0, 1);
      dv.setUint8(1, 1); // pcm_s16le
      dv.setUint16(2, 0, true);
      dv.setUint32(4, seq++ >>> 0, true);
      dv.setBigUint64(8, BigInt(Math.round(performance.now() * 1000)), true);
      const i16 = new Int16Array(out, HEADER, FRAME);
      for (let i = 0; i < FRAME; i++) {
        const s = Math.max(-1, Math.min(1, f32[i]));
        i16[i] = s < 0 ? s * 32768 : s * 32767;
      }
      ws.send(out);
      sent++;
      const now = performance.now();
      if (now - lastStats > 200) {
        lastStats = now;
        emitStats();
      }
    };
    statTimer = setInterval(emitStats, 500);
  } catch (e) {
    try { ws?.close(); } catch { /* ignore */ }
    cleanup();
    throw e;
  }

  return {
    setMonitor(mode) {
      monitor = mode;
      playback?.port.postMessage({ type: 'clear' });
    },
    setAtten(db) {
      if (ws?.readyState === WebSocket.OPEN) ws.send(JSON.stringify({ type: 'config', attenLimDb: db }));
    },
    stop() {
      if (closed || stopping) return;
      stopping = true;
      stream?.getTracks().forEach((t) => t.stop()); // no more audio
      if (ws?.readyState === WebSocket.OPEN) {
        ws.send(JSON.stringify({ type: 'stop' }));
        // The server flushes outstanding captions and closes; don't wait forever.
        setTimeout(() => { try { ws.close(); } catch { /* ignore */ } cleanup(); }, 4000);
      } else {
        cleanup();
      }
    },
  };
}
