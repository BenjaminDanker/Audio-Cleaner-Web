// Generated speech-like audio for the checks of "Recommended for this file" (tests/reco-unit.mjs, tests/reco-ui.mjs): no files needed.
import { RATE } from '../src/media/analysis.js';
export { RATE };

export function rng(seed) {
  let s = seed >>> 0 || 1;
  return () => { s ^= s << 13; s >>>= 0; s ^= s >>> 17; s ^= s << 5; s >>>= 0; return s / 4294967296; };
}
export const gauss = (r) => Math.sqrt(-2 * Math.log(r() + 1e-12)) * Math.cos(2 * Math.PI * r());

/** Voice-like: harmonics up to 8 kHz falling 6 dB per octave on a gliding pitch, shaped by three formants that move from one
 *  vowel to another within each word and three fixed higher ones, words 0.2-0.6 s long with pauses of 0.1-0.5 s. The noise remover keeps this as a voice
 *  (a plain buzz of harmonics it partly takes out as noise, which would read as noise in a clean recording). */
export function voice(seconds, seed = 1) {
  const HIGH = [[3500, 250], [4500, 300], [6000, 400]];
  const VOWELS = [[730, 1090, 2440], [270, 2290, 3010], [300, 870, 2240], [530, 1840, 2480], [640, 1190, 2390], [490, 1350, 1690]];
  const r = rng(seed);
  const n = Math.round(seconds * RATE);
  const out = new Float32Array(n);
  const gains = new Float32Array(80);
  let phase = 0;
  let word = 0;
  let talking = false;
  let left = 0;
  let from = VOWELS[0];
  let to = VOWELS[1];
  let base = 120;
  for (let i = 0; i < n; i++) {
    if (left <= 0) {
      talking = !talking;
      left = Math.round((talking ? 0.2 + 0.4 * r() : 0.1 + 0.4 * r()) * RATE);
      word = left;
      from = VOWELS[Math.floor(r() * VOWELS.length)];
      to = VOWELS[Math.floor(r() * VOWELS.length)];
      base = 105 + 40 * r();
    }
    left--;
    if (!talking) continue;
    const p = (word - left) / word;
    const f0 = base * (1 + 0.15 * Math.sin(Math.PI * p));
    phase += (2 * Math.PI * f0) / RATE;
    if ((word - left) % 480 === 1) { // the formants move every 10 ms
      for (let k = 1; k < gains.length; k++) {
        let g = 0;
        for (let j = 0; j < 3; j++) g += 1 / (1 + ((k * f0 - (from[j] + (to[j] - from[j]) * p)) / (80 + 40 * j)) ** 2) / (j + 1);
        for (const [fc, bw] of HIGH) g += 0.45 / (1 + ((k * f0 - fc) / bw) ** 2);
        gains[k] = k * f0 < 8000 ? g / k : 0;
      }
    }
    let v = 0;
    for (let k = 1; k < gains.length && gains[k]; k++) v += Math.sin(k * phase) * gains[k];
    out[i] = v * Math.sin(Math.PI * p);
  }
  return out;
}
export const rms = (x) => Math.sqrt(x.reduce((s, v) => s + v * v, 0) / x.length);

/** voice + steady noise `snr` dB under it, peak 0.5 */
export function noisy(seconds, snr, seed = 1, kind = 'white') {
  const v = voice(seconds, seed);
  const r = rng(seed + 100);
  const noise = new Float32Array(v.length);
  let lp = 0;
  for (let i = 0; i < noise.length; i++) {
    const g = gauss(r);
    if (kind === 'pink') { lp = 0.97 * lp + 0.03 * g; noise[i] = lp * 6; } else noise[i] = g;
  }
  const gain = (rms(v) / rms(noise)) * 10 ** (-snr / 20);
  const y = v.map((a, i) => a + noise[i] * gain);
  const peak = y.reduce((m, a) => Math.max(m, Math.abs(a)), 0);
  return y.map((a) => (a * 0.5) / peak);
}


/** 16-bit mono WAV file bytes for Float32 samples (at `rate`; the audio is just played at that rate). */
export function wavBytes(samples, rate = RATE) {
  const n = samples.length;
  const b = Buffer.alloc(44 + n * 2);
  b.write('RIFF', 0); b.writeUInt32LE(36 + n * 2, 4); b.write('WAVEfmt ', 8);
  b.writeUInt32LE(16, 16); b.writeUInt16LE(1, 20); b.writeUInt16LE(1, 22); b.writeUInt32LE(rate, 24); b.writeUInt32LE(rate * 2, 28);
  b.writeUInt16LE(2, 32); b.writeUInt16LE(16, 34); b.write('data', 36); b.writeUInt32LE(n * 2, 40);
  for (let i = 0; i < n; i++) b.writeInt16LE(Math.max(-32768, Math.min(32767, Math.round(samples[i] * 32767))), 44 + i * 2);
  return b;
}
