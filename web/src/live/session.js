// Browser live-denoise demo client for GET /api/stream/ws (docs/streaming-protocol.md).
// mic (48 kHz mono, the browser removes speaker echo; its noise suppression and auto gain are off) -> capture worklet (20 ms frames) -> WebSocket (pcm_s16le + 16-byte header)
// <- processed frames -> playback worklet (jitter buffer) -> speakers

import { streamUrl } from '../api.js';
import { LiveAdvisor } from './advice.js';
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
 * Folds one `{"type":"caption", text, final, id?}` event from the server into the list of caption lines.
 * - Extra accurate (premium) captions stream partial captions with an `id`, then a final one with the same `id`: the line with
 *   that id is updated in place (partial -> final), so text never jumps or duplicates.
 * - Standard captions have no `id`: there is at most one id-less partial at the end of the list, replaced by the next
 *   partial and dropped when the id-less final arrives (which is appended).
 * Returns a new array of `{key, id, text, final}` (at most `max` lines, oldest dropped); `key` is stable for React.
 */
let captionKey = 0;
export function applyCaption(items, ev, max = 200) {
  const text = String(ev.text ?? '').trim();
  const final = !!ev.final;
  const id = ev.id != null && ev.id !== '' ? `id:${ev.id}` : null;
  const idlessPartial = (x) => !x.id && !x.final;
  let next;
  if (id) {
    const i = items.findIndex((x) => x.id === id);
    if (!text) {
      if (!final || i < 0) return items; // empty partial: nothing to show; empty final: closes (removes) its partial
      next = items.filter((x) => x.id !== id);
    } else if (i >= 0) {
      if (items[i].final && !final) return items; // a late partial must not reopen a finished line
      next = items.slice();
      next[i] = { ...items[i], text, final };
    } else {
      next = [...items, { key: ++captionKey, id, text, final }];
    }
  } else if (final) {
    const rest = items.filter((x) => !idlessPartial(x));
    next = text ? [...rest, { key: ++captionKey, id: null, text, final: true }] : rest;
  } else {
    const i = items.findIndex(idlessPartial);
    if (!text) return i < 0 ? items : items.filter((x) => !idlessPartial(x));
    if (i >= 0) {
      next = items.slice();
      next[i] = { ...items[i], text };
    } else {
      next = [...items, { key: ++captionKey, id: null, text, final: false }];
    }
  }
  return next.length > max ? next.slice(-max) : next;
}

/**
 * @param {{denoise: 'light'|null (the noise-removal tier id), captions: 'local'|'premium'|null, language: string|null, attenDb: number, monitor: 'off'|'original'|'processed',
 *   onEvent: (ev:object)=>void, onStats:(s:object)=>void, onAdvice?: (found:{stats:object, echo:object|null})=>void}} opts
 *   Server events reach `onEvent` as parsed JSON ('ready', 'caption', 'warning', 'stopped', 'error', plus a local 'closed');
 *   feed 'caption' events to applyCaption().
 */
export async function startLive(opts) {
  const { onEvent, onStats } = opts;
  let monitor = opts.monitor;
  let stream = null;
  let ctx = null;
  let ws = null;
  let capture = null;
  let playback = null;
  const advisor = opts.onAdvice ? new LiveAdvisor({ onAdvice: opts.onAdvice }) : null; // "Recommended for your room", ~10 s in
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
  const pendingChanges = new Map(); // config id -> resolve(answer)
  let changeId = 0;

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
      audio: { channelCount: 1, sampleRate: 48000, echoCancellation: true, noiseSuppression: false, autoGainControl: false },
    });
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

    ws = new WebSocket(streamUrl());
    ws.binaryType = 'arraybuffer';

    await new Promise((resolve, reject) => {
      const fail = (msg) => reject(new Error(msg));
      const timer = setTimeout(() => fail('Live didn\'t answer in time.'), 15000);
      ws.onopen = () => {
        ws.send(JSON.stringify({
          type: 'start', version: 1, denoise: opts.denoise || null, captions: opts.captions || null,
          language: opts.language || null, sampleRate: 48000, attenLimDb: opts.attenDb, client: 'acw-web-demo/1.0',
          ...(opts.translate?.length ? { translate: opts.translate } : {}),
        }));
      };
      ws.onerror = () => { clearTimeout(timer); fail('Couldn’t connect to live.'); };
      ws.onclose = (e) => { clearTimeout(timer); fail(e.code === 1008 ? 'Live said no to the connection. Are you signed in?' : 'The connection closed before live could start.'); };
      ws.onmessage = (e) => {
        if (typeof e.data !== 'string') return;
        let m;
        try {
          m = JSON.parse(e.data);
        } catch {
          return; // not one of ours
        }
        if (m.type === 'ready') {
          clearTimeout(timer);
          ready = true;
          resolve(m);
          onEvent(m);
        } else if (m.type === 'error') {
          clearTimeout(timer);
          const err = new Error(m.message || 'Live ran into a problem.');
          err.code = m.code; // e.g. 'full', 'bad_config', 'insufficient_funds', 'tos'
          if (Array.isArray(m.available)) err.available = m.available; // with 'full': what it can still take, e.g. [{denoise:'light'},{captions:'premium'}]
          reject(err);
        }
      };
    });

    // From here on: steady-state handlers.
    ws.onerror = () => onEvent({ type: 'error', code: 'socket', message: 'The connection to live dropped.' });
    ws.onclose = () => {
      cleanup();
      onEvent({ type: 'closed', clean: stopping });
    };
    ws.onmessage = (e) => {
      if (typeof e.data === 'string') {
        let m;
        try { m = JSON.parse(e.data); } catch { return; }
        if (m.type === 'config' && pendingChanges.has(m.id)) {
          pendingChanges.get(m.id)(m);
          pendingChanges.delete(m.id);
          return;
        }
        onEvent(m);
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
      if (monitor === 'processed') {
        advisor?.pushPlayed(f32);
        playback.port.postMessage({ type: 'frame', samples: f32 }, [f32.buffer]);
      }
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
      if (ready && !stopping) advisor?.pushMic(f32);
      if (monitor === 'original') {
        if (ready && !stopping) advisor?.pushPlayed(null, f32.length); // the microphone itself: no use for telling an echo from the voice
        playback.port.postMessage({ type: 'frame', samples: f32.slice() });
      } else if (monitor === 'off' && ready && !stopping) advisor?.pushPlayed(null, f32.length);
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
    /** Change the session's settings while it runs ({denoise, captions, language, translate, echoReference}); resolves
     * with the service's answer: {ok: true, usdPerMinute, ...} or {ok: false, code, message} (then nothing changed). */
    reconfigure(changes) {
      return new Promise((resolve) => {
        if (ws?.readyState !== WebSocket.OPEN) {
          resolve({ ok: false, message: 'Live isn’t connected.' });
          return;
        }
        const id = `c${++changeId}`;
        pendingChanges.set(id, resolve);
        ws.send(JSON.stringify({ type: 'config', id, ...changes }));
        setTimeout(() => {
          if (!pendingChanges.has(id)) return;
          pendingChanges.delete(id);
          resolve({ ok: false, message: 'Live didn’t answer the change in time. Your settings stay as they were.' });
        }, 8000);
      });
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
