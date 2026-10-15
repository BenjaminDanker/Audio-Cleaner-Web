// Lossless FLAC encoder for uncompressed WAV / AIFF input, used to shrink file-job uploads.
//
// Why our own encoder: WebCodecs has no FLAC encoder in Chromium (AudioEncoder.isConfigSupported({codec: 'flac'}) is
// false, verified on Chromium 145), and mediabunny only delegates encoding to WebCodecs. FLAC is simple enough to
// write directly: fixed predictors (orders 0-4), Rice-coded residuals with partitioning, stereo decorrelation
// (left/side, side/right, mid/side), constant and verbatim subframes. The output is bit-exact lossless and carries the
// STREAMINFO MD5 of the original samples, so any decoder can verify it.
//
// Supported input: PCM WAV (RIFF, incl. WAVE_FORMAT_EXTENSIBLE with a PCM sub-format) and plain AIFF, 16 or 24 bit,
// 1 or 2 channels, up to 1 GB. Anything else (float, 8/32 bit, >2 channels, RF64, AIFF-C, ...) returns null and the
// caller uploads the original file.

const BLOCK = 4096; // samples per frame (FLAC "streamable subset" for rates up to 48 kHz)
const MAX_INPUT_BYTES = 1024 ** 3;
const MAX_RATE = 655350;

/* ------------------------------------------------------------------------------------------------ input sniffing */

async function readBytes(file, offset, length) {
  return new Uint8Array(await file.slice(offset, offset + length).arrayBuffer());
}

const ascii = (b, o) => String.fromCharCode(b[o], b[o + 1], b[o + 2], b[o + 3]);

/** 80-bit IEEE extended (AIFF COMM sample rate) -> number. */
function extended80(b, o) {
  const exp = ((b[o] & 0x7f) << 8) | b[o + 1];
  const hi = ((b[o + 2] << 24) | (b[o + 3] << 16) | (b[o + 4] << 8) | b[o + 5]) >>> 0;
  const lo = ((b[o + 6] << 24) | (b[o + 7] << 16) | (b[o + 8] << 8) | b[o + 9]) >>> 0;
  if (exp === 0 && hi === 0 && lo === 0) return 0;
  return (hi * 4294967296 + lo) * 2 ** (exp - 16383 - 63);
}

/**
 * Looks at the header of a file. Resolves to null unless it is integer PCM we can encode.
 * @returns {Promise<null | {container:'wav'|'aiff', bigEndian:boolean, channels:number, bits:number, rate:number,
 *   frames:number, dataOffset:number}>}
 */
export async function sniffPcm(file) {
  try {
    if (!file || file.size < 44 || file.size > MAX_INPUT_BYTES) return null;
    const head = await readBytes(file, 0, 12);
    const form = ascii(head, 0);
    const kind = ascii(head, 8);
    let container;
    if (form === 'RIFF' && kind === 'WAVE') container = 'wav';
    else if (form === 'FORM' && kind === 'AIFF') container = 'aiff';
    else return null;
    const be = container === 'aiff';

    let fmt = null;
    let data = null;
    let offset = 12;
    for (let i = 0; i < 256 && offset + 8 <= file.size && !(fmt && data); i++) {
      const h = await readBytes(file, offset, 8);
      const id = ascii(h, 0);
      const dv = new DataView(h.buffer);
      const size = dv.getUint32(4, !be); // DataView takes a little-endian flag
      const body = offset + 8;
      if (container === 'wav' && id === 'fmt ') {
        if (size < 16) return null;
        const f = new DataView((await readBytes(file, body, Math.min(size, 40))).buffer);
        let tag = f.getUint16(0, true);
        const channels = f.getUint16(2, true);
        const rate = f.getUint32(4, true);
        const blockAlign = f.getUint16(12, true);
        const bits = f.getUint16(14, true);
        if (tag === 0xfffe) {
          if (size < 40) return null;
          if (f.getUint16(18, true) !== bits) return null; // valid bits != container bits
          tag = f.getUint16(24, true); // first two bytes of the sub-format GUID
        }
        if (tag !== 1 || blockAlign !== channels * (bits >> 3)) return null;
        fmt = { channels, rate, bits };
      } else if (container === 'wav' && id === 'data') {
        let len = size;
        if (len === 0 || len === 0xffffffff || body + len > file.size) len = file.size - body; // streamed / truncated file
        data = { offset: body, length: len };
      } else if (container === 'aiff' && id === 'COMM') {
        if (size < 18) return null;
        const c = await readBytes(file, body, 18);
        const cv = new DataView(c.buffer);
        fmt = { channels: cv.getInt16(0, false), rate: Math.round(extended80(c, 8)), bits: cv.getInt16(6, false), frames: cv.getUint32(2, false) };
      } else if (container === 'aiff' && id === 'SSND') {
        if (size < 8) return null;
        const s = new DataView((await readBytes(file, body, 8)).buffer);
        const skip = s.getUint32(0, false);
        const start = body + 8 + skip;
        let len = size - 8 - skip;
        if (start + len > file.size) len = file.size - start;
        data = { offset: start, length: len };
      }
      offset = body + size + (size & 1);
    }
    if (!fmt || !data) return null;
    const { channels, rate, bits } = fmt;
    if (![1, 2].includes(channels) || ![16, 24].includes(bits) || !(rate >= 1 && rate <= MAX_RATE)) return null;
    const blockAlign = channels * (bits >> 3);
    let frames = Math.floor(data.length / blockAlign);
    if (container === 'aiff') frames = Math.min(frames, fmt.frames);
    if (frames < 16) return null; // STREAMINFO block sizes must be at least 16, and there is nothing to gain
    return { container, bigEndian: be, channels, bits, rate, frames, dataOffset: data.offset };
  } catch {
    return null;
  }
}

/* ------------------------------------------------------------------------------------------------------- MD5 */
// FLAC stores the MD5 of the unencoded samples (signed little-endian, interleaved, ceil(bits/8) bytes per sample).

const MD5_K = new Int32Array(64);
for (let i = 0; i < 64; i++) MD5_K[i] = Math.floor(Math.abs(Math.sin(i + 1)) * 4294967296) | 0;

const MD5_S = new Uint8Array([
  7, 12, 17, 22, 7, 12, 17, 22, 7, 12, 17, 22, 7, 12, 17, 22,
  5, 9, 14, 20, 5, 9, 14, 20, 5, 9, 14, 20, 5, 9, 14, 20,
  4, 11, 16, 23, 4, 11, 16, 23, 4, 11, 16, 23, 4, 11, 16, 23,
  6, 10, 15, 21, 6, 10, 15, 21, 6, 10, 15, 21, 6, 10, 15, 21,
]);

class Md5 {
  constructor() {
    this.h = new Int32Array([0x67452301, 0xefcdab89 | 0, 0x98badcfe | 0, 0x10325476]);
    this.buf = new Uint8Array(64);
    this.fill = 0;
    this.length = 0;
    this.w = new Int32Array(16);
  }

  block(u8, off) {
    const w = this.w;
    for (let i = 0, j = off; i < 16; i++, j += 4) w[i] = u8[j] | (u8[j + 1] << 8) | (u8[j + 2] << 16) | (u8[j + 3] << 24);
    const h = this.h;
    let a = h[0];
    let b = h[1];
    let c = h[2];
    let d = h[3];
    let t;
    let f;
    for (let i = 0; i < 16; i++) {
      f = (((b & c) | (~b & d)) + a + MD5_K[i] + w[i]) | 0;
      a = d; d = c; c = b;
      t = MD5_S[i];
      b = (b + ((f << t) | (f >>> (32 - t)))) | 0;
    }
    for (let i = 16; i < 32; i++) {
      f = (((d & b) | (~d & c)) + a + MD5_K[i] + w[(5 * i + 1) & 15]) | 0;
      a = d; d = c; c = b;
      t = MD5_S[i];
      b = (b + ((f << t) | (f >>> (32 - t)))) | 0;
    }
    for (let i = 32; i < 48; i++) {
      f = ((b ^ c ^ d) + a + MD5_K[i] + w[(3 * i + 5) & 15]) | 0;
      a = d; d = c; c = b;
      t = MD5_S[i];
      b = (b + ((f << t) | (f >>> (32 - t)))) | 0;
    }
    for (let i = 48; i < 64; i++) {
      f = ((c ^ (b | ~d)) + a + MD5_K[i] + w[(7 * i) & 15]) | 0;
      a = d; d = c; c = b;
      t = MD5_S[i];
      b = (b + ((f << t) | (f >>> (32 - t)))) | 0;
    }
    h[0] = (h[0] + a) | 0;
    h[1] = (h[1] + b) | 0;
    h[2] = (h[2] + c) | 0;
    h[3] = (h[3] + d) | 0;
  }

  update(u8) {
    let i = 0;
    const n = u8.length;
    this.length += n;
    if (this.fill) {
      const take = Math.min(64 - this.fill, n);
      this.buf.set(u8.subarray(0, take), this.fill);
      this.fill += take;
      i = take;
      if (this.fill < 64) return;
      this.block(this.buf, 0);
      this.fill = 0;
    }
    for (; i + 64 <= n; i += 64) this.block(u8, i);
    if (i < n) {
      this.buf.set(u8.subarray(i), 0);
      this.fill = n - i;
    }
  }

  digest() {
    const bitLen = this.length * 8;
    const padLen = this.fill < 56 ? 56 - this.fill : 120 - this.fill;
    const tail = new Uint8Array(padLen + 8);
    tail[0] = 0x80;
    const dv = new DataView(tail.buffer);
    dv.setUint32(padLen, bitLen >>> 0, true);
    dv.setUint32(padLen + 4, Math.floor(bitLen / 4294967296), true);
    const keep = this.length; // update() would change it
    this.update(tail);
    this.length = keep;
    const out = new Uint8Array(16);
    const ov = new DataView(out.buffer);
    for (let i = 0; i < 4; i++) ov.setInt32(i * 4, this.h[i], true);
    return out;
  }
}

/* ------------------------------------------------------------------------------------------------ bit writer */

const CRC8 = new Uint8Array(256);
const CRC16 = new Uint16Array(256);
for (let i = 0; i < 256; i++) {
  let c8 = i;
  let c16 = i << 8;
  for (let k = 0; k < 8; k++) {
    c8 = c8 & 0x80 ? ((c8 << 1) ^ 0x07) & 0xff : (c8 << 1) & 0xff;
    c16 = c16 & 0x8000 ? ((c16 << 1) ^ 0x8005) & 0xffff : (c16 << 1) & 0xffff;
  }
  CRC8[i] = c8;
  CRC16[i] = c16;
}

class BitWriter {
  constructor(cap = 1 << 20) {
    this.buf = new Uint8Array(cap);
    this.pos = 0; // whole bytes written
    this.acc = 0; // pending bits (fewer than 8)
    this.n = 0;
  }

  grow(extra) {
    if (this.pos + extra <= this.buf.length) return;
    const bigger = new Uint8Array(Math.max(this.buf.length * 2, this.pos + extra));
    bigger.set(this.buf.subarray(0, this.pos));
    this.buf = bigger;
  }

  /** Writes the low `n` bits (0..32) of unsigned/two's-complement `v`. */
  bits(v, n) {
    if (n > 16) {
      this.bits((v >>> 16) & ((1 << (n - 16)) - 1), n - 16);
      this.bits(v & 0xffff, 16);
      return;
    }
    if (n === 0) return;
    if (this.pos + 4 > this.buf.length) this.grow(1 << 16);
    this.acc = (this.acc << n) | (v & ((1 << n) - 1));
    this.n += n;
    while (this.n >= 8) {
      this.n -= 8;
      this.buf[this.pos++] = (this.acc >>> this.n) & 255;
    }
    this.acc &= (1 << this.n) - 1;
  }

  /** `q` zero bits followed by a one bit. */
  unary(q) {
    while (q >= 16) {
      this.bits(0, 16);
      q -= 16;
    }
    this.bits(1, q + 1);
  }

  align() {
    if (this.n) this.bits(0, 8 - this.n);
  }

  crc8(from) {
    let c = 0;
    for (let i = from; i < this.pos; i++) c = CRC8[c ^ this.buf[i]];
    return c;
  }

  crc16(from) {
    let c = 0;
    for (let i = from; i < this.pos; i++) c = ((c << 8) & 0xffff) ^ CRC16[(c >>> 8) ^ this.buf[i]];
    return c;
  }

  take() {
    const out = this.buf.slice(0, this.pos);
    this.pos = 0;
    return out;
  }
}

/* ---------------------------------------------------------------------------------------------- subframe plan */

const MAX_PART_ORDER = 6; // up to 64 Rice partitions per subframe
const MAX_ORDER = 4;

/** Fills `res` with the residual of fixed predictor `order` for x[0..n); returns nothing (warm-up samples are not part of it). */
function fixedResidual(x, n, order, res) {
  switch (order) {
    case 0:
      for (let i = 0; i < n; i++) res[i] = x[i];
      break;
    case 1:
      for (let i = 1; i < n; i++) res[i] = x[i] - x[i - 1];
      break;
    case 2:
      for (let i = 2; i < n; i++) res[i] = x[i] - 2 * x[i - 1] + x[i - 2];
      break;
    case 3:
      for (let i = 3; i < n; i++) res[i] = x[i] - 3 * x[i - 1] + 3 * x[i - 2] - x[i - 3];
      break;
    default:
      for (let i = 4; i < n; i++) res[i] = x[i] - 4 * x[i - 1] + 6 * x[i - 2] - 4 * x[i - 3] + x[i - 4];
  }
}

const zig = (e) => ((e << 1) ^ (e >> 31)) >>> 0;
const sumsScratch = Array.from({ length: MAX_PART_ORDER + 1 }, (_, o) => new Float64Array(1 << o));

const INV_POW2 = Float64Array.from({ length: 32 }, (_, k) => 1 / 2 ** k);

/** Rice parameter for a partition holding `cnt` values whose zigzag sum is `sum`, and its approximate size in bits. */
function riceFor(cnt, sum) {
  if (cnt === 0) return { k: 0, bits: 0 };
  const m = sum / cnt >= 1 ? (sum / cnt) | 0 : 0;
  const k = m ? 31 - Math.clz32(m) : 0; // floor(log2(mean)); the best parameter is this or one less
  let best = k;
  let bestBits = cnt * (k + 1) + Math.floor(sum * INV_POW2[k]);
  if (k > 0) {
    const bits = cnt * k + Math.floor(sum * INV_POW2[k - 1]);
    if (bits < bestBits) { bestBits = bits; best = k - 1; }
  }
  return { k: best, bits: bestBits };
}

/**
 * Fixed-predictor order with the smallest total absolute residual over x[4..n), as {order, sum}.
 * (Same samples for every order, so the sums are comparable; this is also the cheap estimate used for stereo modes.)
 */
function bestOrder(x, n) {
  let s0 = 0;
  let s1 = 0;
  let s2 = 0;
  let s3 = 0;
  let s4 = 0;
  let p1 = 0;
  let p2 = 0;
  let p3 = 0;
  for (let i = 1; i < n; i++) {
    // e_k = k-th difference; the recurrence is only valid from i >= k, hence the sums start at i = 4
    const e1 = x[i] - x[i - 1];
    const e2 = e1 - p1;
    const e3 = e2 - p2;
    const e4 = e3 - p3;
    p1 = e1; p2 = e2; p3 = e3;
    if (i >= MAX_ORDER) {
      const e0 = x[i];
      s0 += e0 < 0 ? -e0 : e0;
      s1 += e1 < 0 ? -e1 : e1;
      s2 += e2 < 0 ? -e2 : e2;
      s3 += e3 < 0 ? -e3 : e3;
      s4 += e4 < 0 ? -e4 : e4;
    }
  }
  let order = 0;
  let sum = s0;
  if (s1 < sum) { order = 1; sum = s1; }
  if (s2 < sum) { order = 2; sum = s2; }
  if (s3 < sum) { order = 3; sum = s3; }
  if (s4 < sum) { order = 4; sum = s4; }
  return { order, sum };
}

/**
 * Chooses the cheapest representation of one channel's block, using fixed-predictor `order`.
 * `res` receives the residual (reused when writing).
 * @returns {{kind:'const'|'verbatim'|'fixed', order:number, po:number, ks:number[], bits:number, res:Int32Array}}
 */
function analyze(x, n, bps, res, order) {
  const first = x[0];
  let constant = true;
  for (let i = 1; i < n; i++) if (x[i] !== first) { constant = false; break; }
  if (constant) return { kind: 'const', order: 0, po: 0, ks: [], bits: 8 + bps, res };

  const verbatimBits = 8 + n * bps;
  if (n < 16) return { kind: 'verbatim', order: 0, po: 0, ks: [], bits: verbatimBits, res };

  fixedResidual(x, n, order, res);
  // Partition search over the residual res[order..n).
  let maxPo = 0;
  while (maxPo < MAX_PART_ORDER && n % (1 << (maxPo + 1)) === 0 && (n >> (maxPo + 1)) > order) maxPo++;
  const fine = sumsScratch[maxPo];
  const per = n >> maxPo;
  for (let p = 0, i = order; p < 1 << maxPo; p++) {
    const end = (p + 1) * per;
    let s = 0;
    for (; i < end; i++) s += zig(res[i]);
    fine[p] = s;
  }
  let bestPo = 0;
  let bestBits = Infinity;
  let bestKs = null;
  let cur = new Float64Array(fine); // zigzag sums per partition, starting at the finest order
  for (let po = maxPo; po >= 0; po--) {
    const parts = 1 << po;
    let total = 4 * parts;
    const ks = new Array(parts);
    for (let p = 0; p < parts; p++) {
      const cnt = (n >> po) - (p === 0 ? order : 0);
      const r = riceFor(cnt, cur[p]);
      ks[p] = r.k;
      total += r.bits;
    }
    if (total < bestBits) { bestBits = total; bestPo = po; bestKs = ks; }
    if (po > 0) {
      const next = new Float64Array(parts >> 1);
      for (let p = 0; p < next.length; p++) next[p] = cur[2 * p] + cur[2 * p + 1];
      cur = next;
    }
  }
  const total = 8 + order * bps + 2 + 4 + bestBits;
  if (total >= verbatimBits) return { kind: 'verbatim', order: 0, po: 0, ks: [], bits: verbatimBits, res };
  return { kind: 'fixed', order, po: bestPo, ks: bestKs, bits: total, res };
}

function writeSubframe(w, x, n, bps, plan) {
  const mask = bps >= 32 ? 0xffffffff : (1 << bps) - 1;
  if (plan.kind === 'const') {
    w.bits(0, 8); // pad, type CONSTANT (000000), no wasted bits
    w.bits(x[0] & mask, bps);
    return;
  }
  if (plan.kind === 'verbatim') {
    w.bits(1 << 1, 8); // type 000001
    for (let i = 0; i < n; i++) w.bits(x[i] & mask, bps);
    return;
  }
  const { order, po, ks, res } = plan;
  w.bits((8 + order) << 1, 8); // 0 | 001ooo | 0
  for (let i = 0; i < order; i++) w.bits(x[i] & mask, bps);
  const wide = ks.some((k) => k > 14);
  w.bits(wide ? 1 : 0, 2); // residual coding method: Rice (4-bit parameters) or Rice2 (5-bit)
  w.bits(po, 4);
  const parts = 1 << po;
  const per = n >> po;
  let i = order;
  for (let p = 0; p < parts; p++) {
    const end = (p + 1) * per;
    const k = ks[p];
    w.bits(k, wide ? 5 : 4);
    const kmask = (1 << k) - 1;
    for (; i < end; i++) {
      const u = zig(res[i]);
      const q = u >>> k;
      if (q + 1 + k <= 32) w.bits(((1 << k) | (u & kmask)) >>> 0, q + 1 + k); // q zeros, a one, k low bits
      else { w.unary(q); if (k) w.bits(u & kmask, k); }
    }
  }
}

function utf8Number(w, v) {
  if (v < 0x80) { w.bits(v, 8); return; }
  const len = v < 0x800 ? 2 : v < 0x10000 ? 3 : v < 0x200000 ? 4 : v < 0x4000000 ? 5 : 6;
  w.bits(((0xff00 >> len) & 0xff) | (v >> (6 * (len - 1))), 8);
  for (let s = len - 2; s >= 0; s--) w.bits(0x80 | ((v >> (6 * s)) & 0x3f), 8);
}

/* ---------------------------------------------------------------------------------------------------- encoder */

/**
 * Encodes the PCM described by `pcm` (from sniffPcm) to a FLAC Blob.
 * @param {Blob} file
 * @param {{onProgress?:(frac:number)=>void, signal?:AbortSignal}} opts
 */
export async function encodeFlac(file, pcm, { onProgress, signal } = {}) {
  const { channels: nch, bits: bps, rate, frames, dataOffset, bigEndian } = pcm;
  const bytesPer = bps >> 3;
  const align = nch * bytesPer;
  const readFrames = BLOCK * Math.max(1, Math.floor((1 << 20) / (BLOCK * align)));
  const w = new BitWriter();
  const md5 = new Md5();
  const parts = [new Uint8Array(42)]; // 'fLaC' + STREAMINFO block, patched at the end
  const sig = {
    L: new Int32Array(BLOCK), R: new Int32Array(BLOCK), M: new Int32Array(BLOCK), S: new Int32Array(BLOCK),
  };
  const res = { L: new Int32Array(BLOCK), R: new Int32Array(BLOCK), M: new Int32Array(BLOCK), S: new Int32Array(BLOCK) };
  let maxBs = 0;
  let minFs = Infinity;
  let maxFs = 0;
  let frameNo = 0;
  let done = 0;
  let lastBs = 0;

  while (done < frames) {
    if (signal?.aborted) throw new DOMException('Canceled', 'AbortError');
    const nf = Math.min(readFrames, frames - done);
    const buf = await readBytes(file, dataOffset + done * align, nf * align);
    if (buf.length !== nf * align) throw new Error('short read');
    if (bigEndian) {
      // FLAC's MD5 is over little-endian samples
      const le = new Uint8Array(buf);
      if (bytesPer === 2) for (let i = 0; i < le.length; i += 2) { const t = le[i]; le[i] = le[i + 1]; le[i + 1] = t; }
      else for (let i = 0; i < le.length; i += 3) { const t = le[i]; le[i] = le[i + 2]; le[i + 2] = t; }
      md5.update(le);
    } else {
      md5.update(buf);
    }
    const dv = new DataView(buf.buffer, buf.byteOffset, buf.byteLength);
    for (let off = 0; off < nf; off += BLOCK) {
      const n = Math.min(BLOCK, nf - off);
      for (let c = 0; c < nch; c++) {
        const dst = c === 0 ? sig.L : sig.R;
        let p = (off * nch + c) * bytesPer;
        const step = nch * bytesPer;
        if (bps === 16) for (let i = 0; i < n; i++, p += step) dst[i] = dv.getInt16(p, !bigEndian);
        else if (bigEndian) for (let i = 0; i < n; i++, p += step) dst[i] = ((buf[p] << 24) | (buf[p + 1] << 16) | (buf[p + 2] << 8)) >> 8;
        else for (let i = 0; i < n; i++, p += step) dst[i] = ((buf[p + 2] << 24) | (buf[p + 1] << 16) | (buf[p] << 8)) >> 8;
      }

      // Pick the representation (and for stereo the channel assignment).
      let assignment = 0; // 0 = mono
      let plans;
      let chans;
      let widths;
      if (nch === 1) {
        chans = [sig.L];
        widths = [bps];
        plans = [analyze(sig.L, n, bps, res.L, bestOrder(sig.L, n).order)];
      } else {
        for (let i = 0; i < n; i++) {
          sig.S[i] = sig.L[i] - sig.R[i];
          sig.M[i] = (sig.L[i] + sig.R[i]) >> 1;
        }
        // Cheap size estimate per signal (log2 of the mean absolute residual of its best fixed predictor), then
        // the full analysis only for the two signals of the winning channel assignment.
        const est = {};
        for (const k of ['L', 'R', 'M', 'S']) {
          const b = bestOrder(sig[k], n);
          est[k] = { order: b.order, bits: n * Math.log2(Math.max(b.sum / n, 0.5)) };
        }
        const choices = [
          [est.L.bits + est.R.bits, 1, ['L', 'R']],
          [est.L.bits + est.S.bits + 1, 8, ['L', 'S']],
          [est.S.bits + est.R.bits + 1, 9, ['S', 'R']],
          [est.M.bits + est.S.bits + 1, 10, ['M', 'S']],
        ];
        const best = choices.reduce((a, b) => (b[0] < a[0] ? b : a));
        assignment = best[1];
        chans = best[2].map((k) => sig[k]);
        widths = best[2].map((k) => (k === 'S' ? bps + 1 : bps));
        plans = best[2].map((k, i) => analyze(sig[k], n, widths[i], res[k], est[k].order));
      }

      // Frame header
      const start = w.pos;
      w.bits(0xfff8, 16); // sync + fixed block size strategy
      const bsCode = n === 4096 ? 12 : n <= 256 ? 6 : 7;
      w.bits(bsCode, 4);
      w.bits(0, 4); // sample rate: take it from STREAMINFO
      w.bits(assignment, 4);
      w.bits(bps === 16 ? 4 : 6, 3);
      w.bits(0, 1);
      utf8Number(w, frameNo);
      if (bsCode === 6) w.bits(n - 1, 8);
      else if (bsCode === 7) w.bits(n - 1, 16);
      w.bits(w.crc8(start), 8);
      for (let c = 0; c < chans.length; c++) writeSubframe(w, chans[c], n, widths[c], plans[c]);
      w.align();
      w.bits(w.crc16(start), 16);

      const size = w.pos - start;
      if (size < minFs) minFs = size;
      if (size > maxFs) maxFs = size;
      if (n > maxBs) maxBs = n;
      lastBs = n;
      frameNo++;
    }
    parts.push(w.take());
    done += nf;
    onProgress?.(done / frames);
  }

  // STREAMINFO: the minimum block size excludes the last frame (unless it is the only one).
  const fullBlocks = Math.floor(frames / BLOCK);
  const minBlock = fullBlocks > 0 ? BLOCK : lastBs;
  const info = parts[0];
  const iv = new DataView(info.buffer);
  info.set([0x66, 0x4c, 0x61, 0x43]); // 'fLaC'
  iv.setUint8(4, 0x80); // last metadata block, type STREAMINFO
  iv.setUint8(5, 0);
  iv.setUint16(6, 34, false);
  iv.setUint16(8, minBlock, false);
  iv.setUint16(10, maxBs, false);
  iv.setUint8(12, (minFs >> 16) & 255); iv.setUint16(13, minFs & 0xffff, false);
  iv.setUint8(15, (maxFs >> 16) & 255); iv.setUint16(16, maxFs & 0xffff, false);
  iv.setUint32(18, (((rate << 12) | ((nch - 1) << 9) | ((bps - 1) << 4) | Math.floor(frames / 4294967296)) >>> 0), false);
  iv.setUint32(22, frames % 4294967296, false);
  info.set(md5.digest(), 26);
  return new Blob(parts, { type: 'audio/flac' });
}

/**
 * Shrinks an uncompressed WAV/AIFF for upload. Resolves to null when the file is not eligible, encoding fails, or the
 * result would not be smaller; the caller then uploads the original. Rejects only with AbortError.
 * `onStart` is called once the file is known to be eligible (before the first progress report); `name` overrides file.name.
 * @returns {Promise<null | {blob: Blob, filename: string, bytesIn: number, bytesOut: number}>}
 */
export async function compressForUpload(file, { name = file?.name, onStart, onProgress, signal } = {}) {
  const pcm = await sniffPcm(file);
  if (!pcm) return null;
  onStart?.();
  let blob;
  try {
    blob = await encodeFlac(file, pcm, { onProgress, signal });
  } catch (e) {
    if (e?.name === 'AbortError' || signal?.aborted) throw e;
    return null;
  }
  if (blob.size >= file.size) return null;
  const base = String(name || 'audio').split(/[\\/]/).pop().replace(/\.[^.]+$/, '') || 'audio';
  return { blob, filename: `${base}.flac`, bytesIn: file.size, bytesOut: blob.size };
}
