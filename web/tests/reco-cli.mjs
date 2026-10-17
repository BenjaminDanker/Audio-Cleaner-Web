// Measures WAV files with the same code the browser uses (src/media/analysis.js) and prints one JSON object per file.
// Used by the Python agreement test (server/tests/test_reco_agreement.py) and the calibration scripts.
//   node tests/reco-cli.mjs [--sampled] [--first=SECONDS] [--reco] file.wav...   (48 kHz, PCM16 or float32, mono or stereo)
// --sampled: only the stretches the browser would decode (planWindows), as it does for a long file.
// --first=10: only the first 10 seconds, as the live test hears them.   --reco: also print the recommendation.
import { readFileSync } from 'node:fs';
import { Analyser, planWindows, RATE } from '../src/media/analysis.js';

export function readWav(path) {
  const b = readFileSync(path);
  if (b.toString('ascii', 0, 4) !== 'RIFF') throw new Error(`${path}: not a WAV file`);
  let off = 12;
  let fmt = null;
  let data = null;
  while (off + 8 <= b.length) {
    const id = b.toString('ascii', off, off + 4);
    let size = b.readUInt32LE(off + 4);
    if (id === 'data' && off + 8 + size > b.length) size = b.length - off - 8;
    if (id === 'fmt ') fmt = { tag: b.readUInt16LE(off + 8), ch: b.readUInt16LE(off + 10), rate: b.readUInt32LE(off + 12), bits: b.readUInt16LE(off + 22) };
    if (id === 'data') { data = b.subarray(off + 8, off + 8 + size); break; }
    off += 8 + size + (size & 1);
  }
  if (!fmt || !data) throw new Error(`${path}: no audio`);
  if (fmt.rate !== RATE) throw new Error(`${path}: ${fmt.rate} Hz (48000 needed)`);
  const bytes = fmt.bits / 8;
  const frames = Math.floor(data.length / (bytes * fmt.ch));
  const planes = Array.from({ length: fmt.ch }, () => new Float32Array(frames));
  const view = new DataView(data.buffer, data.byteOffset, data.length);
  for (let i = 0; i < frames; i++) {
    for (let c = 0; c < fmt.ch; c++) {
      const o = (i * fmt.ch + c) * bytes;
      let v;
      if (fmt.tag === 3 || (fmt.tag === 0xfffe && fmt.bits === 32)) v = fmt.bits === 64 ? view.getFloat64(o, true) : view.getFloat32(o, true);
      else if (fmt.bits === 16) v = view.getInt16(o, true) / 32768;
      else if (fmt.bits === 24) v = ((view.getUint8(o) | (view.getUint8(o + 1) << 8) | (view.getInt8(o + 2) << 16)) / 8388608);
      else v = view.getInt32(o, true) / 2147483648;
      planes[c][i] = v;
    }
  }
  return { planes, rate: fmt.rate };
}

/** Feeds `planes` to an Analyser in 1 s pieces, the whole file or just the sampled stretches. */
export function analysePlanes(planes, { sampled = false } = {}) {
  const ch = Math.min(planes.length, 2);
  const a = new Analyser({ channels: ch });
  const frames = planes[0].length;
  const wins = sampled ? planWindows(frames / RATE) : [{ start: 0, length: frames / RATE }];
  wins.forEach((w, k) => {
    if (k) a.segment();
    const from = Math.round(w.start * RATE);
    const to = Math.min(frames, from + Math.round(w.length * RATE));
    for (let p = from; p < to; p += RATE) a.push(planes.slice(0, ch).map((x) => x.subarray(p, Math.min(to, p + RATE))));
  });
  return a.result();
}

if (process.argv[1] && process.argv[1].endsWith('reco-cli.mjs')) {
  const args = process.argv.slice(2);
  const sampled = args.includes('--sampled');
  const withReco = args.includes('--reco');
  const first = Number((args.find((a) => a.startsWith('--first=')) || '').slice(8)) || 0;
  const { recommendForFile } = withReco ? await import('../src/recommend.js') : {};
  for (const f of args.filter((x) => !x.startsWith('--'))) {
    let { planes } = readWav(f);
    if (first) planes = planes.map((p) => p.subarray(0, Math.round(first * RATE)));
    const stats = analysePlanes(planes, { sampled });
    const out = { file: f, ...stats };
    if (withReco) out.reco = recommendForFile(stats, { hasAudio: true, duration: planes[0].length / RATE, channels: planes.length });
    console.log(JSON.stringify(out));
  }
}
