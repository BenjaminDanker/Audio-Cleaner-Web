// Lossless-FLAC check for src/media/flac.js (the browser's WAV/AIFF -> FLAC upload compression), no browser needed.
// Generates WAV/AIFF files (16/24 bit, mono/stereo, silence, noise, clipping, odd lengths, extra chunks, ...), encodes
// them with the production code, and verifies the result three ways: ffmpeg decode is bit-identical (with frame CRC
// checks), libsndfile/libFLAC decode is identical, and the STREAMINFO MD5 equals the MD5 of the original PCM.
//   node tests/flac.mjs        (uses ~/audio-cleaner/bin/ffmpeg and ~/audio-cleaner/venv/bin/python, see helpers.mjs)
//   FORCE=1 node tests/flac.mjs   also checks files that would not shrink (noise) by bypassing the size check
import { mkdtempSync, readFileSync, rmSync, writeFileSync } from 'node:fs';
import { execFileSync } from 'node:child_process';
import { createHash } from 'node:crypto';
import { homedir, tmpdir } from 'node:os';
import { join } from 'node:path';
import { compressForUpload, encodeFlac, sniffPcm } from '../src/media/flac.js';

const FF = process.env.FFMPEG || join(homedir(), 'audio-cleaner', 'bin', 'ffmpeg');
const PY = process.env.ACW_PYTHON || join(homedir(), 'audio-cleaner', 'venv', 'bin', 'python');
const out = mkdtempSync(join(tmpdir(), 'acw-flac-'));

let seed = 12345;
const rnd = () => { seed = (seed * 1103515245 + 12345) & 0x7fffffff; return seed / 0x7fffffff; };

function wav({ nch, bits, rate, frames, gen, extensible = false, pad = false, trailing = false }) {
  const bps = bits / 8;
  const data = Buffer.alloc(frames * nch * bps);
  for (let i = 0; i < frames; i++) for (let c = 0; c < nch; c++) {
    const v = gen(i, c);
    const o = (i * nch + c) * bps;
    if (bits === 16) data.writeInt16LE(Math.max(-32768, Math.min(32767, v)), o);
    else { const x = Math.max(-8388608, Math.min(8388607, v)); data.writeIntLE(x, o, 3); }
  }
  const fmt = Buffer.alloc(extensible ? 40 : 16);
  fmt.writeUInt16LE(extensible ? 0xfffe : 1, 0); fmt.writeUInt16LE(nch, 2); fmt.writeUInt32LE(rate, 4);
  fmt.writeUInt32LE(rate * nch * bps, 8); fmt.writeUInt16LE(nch * bps, 12); fmt.writeUInt16LE(bits, 14);
  if (extensible) { fmt.writeUInt16LE(22, 16); fmt.writeUInt16LE(bits, 18); fmt.writeUInt32LE(nch === 1 ? 4 : 3, 20); fmt.writeUInt16LE(1, 24); Buffer.from('000000001000800000aa00389b71', 'hex').copy(fmt, 26); }
  const chunk = (id, b) => Buffer.concat([Buffer.from(id), Buffer.from(new Uint32Array([b.length]).buffer), b, b.length & 1 ? Buffer.alloc(1) : Buffer.alloc(0)]);
  const parts = [chunk('fmt ', fmt)];
  if (pad) parts.push(chunk('LIST', Buffer.from('INFOISFT\x05\x00\x00\x00test\x00')));
  parts.push(chunk('data', data));
  if (trailing) parts.push(chunk('LIST', Buffer.from('INFOabc')));
  const body = Buffer.concat([Buffer.from('WAVE'), ...parts]);
  return Buffer.concat([Buffer.from('RIFF'), Buffer.from(new Uint32Array([body.length]).buffer), body]);
}

function ext80(rate) { // 80-bit extended
  const b = Buffer.alloc(10);
  let exp = 16383 + Math.floor(Math.log2(rate));
  const mant = BigInt(Math.round(rate * 2 ** (63 - Math.floor(Math.log2(rate)))));
  b.writeUInt16BE(exp, 0); b.writeBigUInt64BE(mant, 2);
  return b;
}
function aiff({ nch, bits, rate, frames, gen }) {
  const bps = bits / 8;
  const data = Buffer.alloc(frames * nch * bps);
  for (let i = 0; i < frames; i++) for (let c = 0; c < nch; c++) {
    const v = gen(i, c); const o = (i * nch + c) * bps;
    if (bits === 16) data.writeInt16BE(Math.max(-32768, Math.min(32767, v)), o); else data.writeIntBE(Math.max(-8388608, Math.min(8388607, v)), o, 3);
  }
  const comm = Buffer.alloc(18); comm.writeInt16BE(nch, 0); comm.writeUInt32BE(frames, 2); comm.writeInt16BE(bits, 6); ext80(rate).copy(comm, 8);
  const ssnd = Buffer.concat([Buffer.alloc(8), data]);
  const chunk = (id, b) => { const h = Buffer.alloc(8); h.write(id, 0); h.writeUInt32BE(b.length, 4); return Buffer.concat([h, b, b.length & 1 ? Buffer.alloc(1) : Buffer.alloc(0)]); };
  const body = Buffer.concat([Buffer.from('AIFF'), chunk('COMM', comm), chunk('SSND', ssnd)]);
  const h = Buffer.alloc(8); h.write('FORM', 0); h.writeUInt32BE(body.length, 4);
  return Buffer.concat([h, body]);
}

const speechLike = (amp) => (i, c) => Math.round(amp * (0.5 * Math.sin(i / 17 + c) + 0.3 * Math.sin(i / 3.1) * Math.sin(i / 2500) + 0.05 * (rnd() - 0.5)));
const cases = [
  ['mono16_speechlike', wav({ nch: 1, bits: 16, rate: 48000, frames: 48000 * 3 + 123, gen: speechLike(20000) })],
  ['stereo16_44k', wav({ nch: 2, bits: 16, rate: 44100, frames: 44100 * 3 + 777, gen: speechLike(12000) })],
  ['mono24', wav({ nch: 1, bits: 24, rate: 48000, frames: 48000 * 2 + 5, gen: speechLike(6000000) })],
  ['stereo24_ext', wav({ nch: 2, bits: 24, rate: 96000, frames: 96000 + 4097, gen: speechLike(5000000), extensible: true })],
  ['noise_fullscale', wav({ nch: 2, bits: 16, rate: 48000, frames: 20000, gen: () => Math.round((rnd() * 2 - 1) * 32767) })],
  ['clipping_square', wav({ nch: 1, bits: 16, rate: 48000, frames: 30000, gen: (i) => (Math.floor(i / 50) & 1 ? 32767 : -32768) })],
  ['tiny_1frame', wav({ nch: 1, bits: 16, rate: 48000, frames: 1, gen: () => 5, pad: true })],
  ['block_plus1', wav({ nch: 1, bits: 16, rate: 48000, frames: 4096 * 2 + 1, gen: speechLike(3000) })],
  ['list_chunks', wav({ nch: 1, bits: 16, rate: 48000, frames: 10000, gen: speechLike(3000), pad: true, trailing: true })],
  ['aiff_stereo16', aiff({ nch: 2, bits: 16, rate: 44100, frames: 44100 + 31, gen: speechLike(10000) })],
];

const sha = (b) => createHash('md5').update(b).digest('hex');
let fail = 0;
for (const [name, buf] of cases) {
  const isAiff = name.startsWith('aiff');
  const src = `${out}/${name}.${isAiff ? 'aiff' : 'wav'}`;
  writeFileSync(src, buf);
  const blob = Object.assign(new Blob([buf]), { name: `${name}.${isAiff ? 'aiff' : 'wav'}` });
  const t0 = performance.now();
  let r = await compressForUpload(blob);
  if (!r && (process.env.FORCE ?? '1')) { const pcm = await sniffPcm(blob); if (pcm) r = { blob: await encodeFlac(blob, pcm), filename: 'forced.flac' }; }
  const ms = (performance.now() - t0).toFixed(0);
  if (!r) { console.log(`${name}: null (not compressed) sniff=${JSON.stringify(await sniffPcm(blob))}`); if (!['tiny_1frame','tiny_100','silence_x'].includes(name)) { /* null allowed only when not smaller */ } continue; }
  const dst = `${out}/${name}.flac`;
  writeFileSync(dst, Buffer.from(await r.blob.arrayBuffer()));
  // decode both with ffmpeg to raw s32le and compare md5; CRC checked
  const dec = (f) => execFileSync(FF, ['-v', 'error', '-err_detect', 'crccheck+compliant', '-i', f, '-f', 's32le', '-acodec', 'pcm_s32le', '-'], { maxBuffer: 1 << 30 });
  const a = dec(src); let b;
  try { b = dec(dst); } catch (e) { console.log(`${name}: FFMPEG DECODE FAILED ${e.message.slice(0, 200)}`); fail++; continue; }
  const same = a.length === b.length && sha(a) === sha(b);
  // libsndfile (libFLAC) decode + STREAMINFO md5 check
  const py = execFileSync(PY, ['-I', '-c', `
import soundfile as sf, numpy as np, hashlib, sys
d, sr = sf.read(sys.argv[1], dtype='int32', always_2d=True)
d2, sr2 = sf.read(sys.argv[2], dtype='int32', always_2d=True)
print('SF', 'same' if (sr == sr2 and d.shape == d2.shape and (d == d2).all()) else 'DIFF', d2.shape, sr2)
`, src, dst], { encoding: 'utf8' }).trim();
  const flacBytes = readFileSync(dst);
  const streamMd5 = flacBytes.subarray(26, 42).toString('hex');
  // md5 of signed LE interleaved PCM at native width
  const pcm = execFileSync(FF, ['-v', 'error', '-i', src, '-f', r.blob.size ? (name.includes('24') ? 's24le' : 's16le') : 's16le', '-acodec', name.includes('24') ? 'pcm_s24le' : 'pcm_s16le', '-'], { maxBuffer: 1 << 30 });
  const md5ok = sha(pcm) === streamMd5;
  if (!same || !md5ok || !py.includes('same')) fail++;
  console.log(`${name}: ${buf.length} -> ${r.blob.size} (${(100 * r.blob.size / buf.length).toFixed(1)}%) ${r.filename} ffmpeg:${same ? 'bit-exact' : 'DIFF'} md5:${md5ok ? 'ok' : 'BAD'} ${py} ${ms}ms`);
}
console.log(fail ? `FAILURES: ${fail}` : 'ALL OK');
if (process.env.KEEP) console.log(`(files in ${out})`);
else rmSync(out, { recursive: true, force: true });
process.exit(fail ? 1 : 0);
