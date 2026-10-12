// Interleaved float32 parts (Blobs) -> PCM WAV Blob, streamed so long files never need one giant buffer.

export const SAMPLE_RATE = 48000;

function header({ channels, sampleRate, bits, frames }) {
  const bytesPerSample = bits / 8;
  const dataLen = frames * channels * bytesPerSample;
  if (dataLen > 0xfffffff0 - 44) throw new Error('This recording is too long for a single WAV file. Choose a shorter file.');
  const buf = new ArrayBuffer(44);
  const v = new DataView(buf);
  const w = (o, s) => { for (let i = 0; i < s.length; i++) v.setUint8(o + i, s.charCodeAt(i)); };
  w(0, 'RIFF');
  v.setUint32(4, 36 + dataLen, true);
  w(8, 'WAVE');
  w(12, 'fmt ');
  v.setUint32(16, 16, true);
  v.setUint16(20, 1, true); // PCM
  v.setUint16(22, channels, true);
  v.setUint32(24, sampleRate, true);
  v.setUint32(28, sampleRate * channels * bytesPerSample, true);
  v.setUint16(32, channels * bytesPerSample, true);
  v.setUint16(34, bits, true);
  w(36, 'data');
  v.setUint32(40, dataLen, true);
  return new Uint8Array(buf);
}

function toPcm16(f32) {
  const out = new Int16Array(f32.length);
  for (let i = 0; i < f32.length; i++) {
    // TPDF dither at +-1 LSB avoids quantisation distortion in quiet passages
    const d = (Math.random() - Math.random()) / 32768;
    let s = f32[i] + d;
    s = s > 1 ? 1 : s < -1 ? -1 : s;
    out[i] = s < 0 ? Math.round(s * 32768) : Math.round(s * 32767);
  }
  return out;
}

function toPcm24(f32) {
  const out = new Uint8Array(f32.length * 3);
  for (let i = 0, o = 0; i < f32.length; i++, o += 3) {
    let s = f32[i];
    s = s > 1 ? 1 : s < -1 ? -1 : s;
    const n = Math.round(s < 0 ? s * 8388608 : s * 8388607);
    out[o] = n & 0xff;
    out[o + 1] = (n >> 8) & 0xff;
    out[o + 2] = (n >> 16) & 0xff;
  }
  return out;
}

/**
 * @param {{parts: Blob[], channels: number, frames: number}} pcm interleaved float32 parts
 * @param {16|24} bits
 */
export async function buildWav(pcm, bits = 16, { sampleRate = SAMPLE_RATE, onProgress } = {}) {
  const out = [header({ channels: pcm.channels, sampleRate, bits, frames: pcm.frames })];
  let done = 0;
  for (const part of pcm.parts) {
    const f32 = new Float32Array(await part.arrayBuffer());
    out.push(bits === 24 ? toPcm24(f32) : toPcm16(f32));
    done++;
    if (onProgress && done % 8 === 0) onProgress(done / pcm.parts.length);
  }
  onProgress?.(1);
  return new Blob(out, { type: 'audio/wav' });
}

/** Interleave per-channel arrays into one Float32Array. */
export function interleave(chans) {
  if (chans.length === 1) return chans[0];
  const n = chans[0].length;
  const out = new Float32Array(n * chans.length);
  for (let c = 0; c < chans.length; c++) {
    const src = chans[c];
    for (let i = 0, o = c; i < n; i++, o += chans.length) out[o] = src[i];
  }
  return out;
}
