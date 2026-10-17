// Measuring a recording, quickly: the numbers behind "Recommended for this file" and "Test my room" (src/recommend.js).
// Pure functions and classes, no browser APIs, so a worker, the Live page and the Node tests all run the same code.
//
// The noise figure is measured by the noise remover itself: it runs at full strength over the audio (in a worker, or Node), and
// what it keeps (the voice) is compared with what it takes out (NoiseFigure), so wind, keyboards and chatter count as much as a
// steady hiss. The server measures the same way to choose its noise remover (common/acw_common/pipeline.py: noise_figure_db,
// measure_windows; bandwidth: bwe.effective_bandwidth). tests/test_reco_agreement.py runs this module and the Python functions on
// the same files and fails when they drift apart. Keep the two in step.
//
//   const a = new Analyser({ channels: 2 });
//   a.push([left, right]);            // 48 kHz Float32Arrays of equal length; call as often as you like
//   a.addCleaned(mono, cleaned);      // the mono mix and what the noise remover (100 dB) made of it, lined up
//   a.segment();                      // the next push does not continue the last one (a different part of the file)
//   const stats = a.result();

export const RATE = 48000;
/** The server's "heavy noise" cut: a noise figure below this goes to the stronger noise remover (pipeline.HEAVY_BELOW_DB). */
export const HEAVY_BELOW_DB = 19.0;
/** Frame energy (dBFS, 10 ms frames) below which audio counts as digital silence (pipeline.SILENCE_DB). */
export const SILENCE_DB = -90.0;
export const DEAD_CHANNEL_PEAK = 1e-4;
export const DUAL_MONO_SIDE_DB = -40.0;

const FRAME = RATE / 100; // 10 ms
const NFFT = 2048;
const HOP = 1024;
const BINS = NFFT / 2 + 1;
const BIN_HZ = RATE / NFFT;
const MAX_FFT_FRAMES = 6000;
/** The speech band's level moves at least this much (90th minus 10th percentile, dB) when someone speaks; stationary noise, hum and tones stay below. */
export const SPEECH_BAND_DYN_DB = 6.8;
const CLIP_LEVEL = 0.999;
const CLIP_RUN = 3;

/** RBJ biquad (Butterworth, Q = 1/sqrt 2) at 48 kHz, as normalised [b0, b1, b2, a1, a2]. */
function biquad(kind, fc) {
  const w = (2 * Math.PI * fc) / RATE;
  const c = Math.cos(w);
  const alpha = Math.sin(w) / (2 * Math.SQRT1_2);
  const a0 = 1 + alpha;
  const b = kind === 'high' ? [(1 + c) / 2, -(1 + c), (1 + c) / 2] : [(1 - c) / 2, 1 - c, (1 - c) / 2];
  return [b[0] / a0, b[1] / a0, b[2] / a0, (-2 * c) / a0, (1 - alpha) / a0];
}
// the speech band, 150 Hz to 3.5 kHz: what moves with words and pauses
const BAND_HP = biquad('high', 150);
const BAND_LP = biquad('low', 3500);

// BS.1770 K-weighting at 48 kHz.
const K1 = { b: [1.53512485958697, -2.69169618940638, 1.19839281085285], a: [-1.69065929318241, 0.73248077421585] };
const K2 = { b: [1, -2, 1], a: [-1.99004745483398, 0.99007225036621] };

/** numpy.percentile (linear interpolation) of an already sorted array, q in 0..100. */
export function percentile(sorted, q) {
  const n = sorted.length;
  if (!n) return NaN;
  const pos = (q / 100) * (n - 1);
  const lo = Math.floor(pos);
  const hi = Math.min(n - 1, lo + 1);
  return sorted[lo] + (sorted[hi] - sorted[lo]) * (pos - lo);
}

const sortedCopy = (arr) => Float64Array.from(arr).sort();

// ---- FFT (radix 2, in place) ------------------------------------------------------------------------------------
const HANN = (() => {
  const w = new Float64Array(NFFT);
  for (let i = 0; i < NFFT; i++) w[i] = 0.5 - 0.5 * Math.cos((2 * Math.PI * i) / (NFFT - 1)); // numpy.hanning
  return w;
})();
const COS = new Float64Array(NFFT / 2);
const SIN = new Float64Array(NFFT / 2);
for (let i = 0; i < NFFT / 2; i++) {
  COS[i] = Math.cos((2 * Math.PI * i) / NFFT);
  SIN[i] = -Math.sin((2 * Math.PI * i) / NFFT);
}
const REV = (() => {
  const r = new Uint16Array(NFFT);
  const bits = Math.log2(NFFT);
  for (let i = 0; i < NFFT; i++) {
    let x = i;
    let y = 0;
    for (let b = 0; b < bits; b++) { y = (y << 1) | (x & 1); x >>= 1; }
    r[i] = y;
  }
  return r;
})();

function fft(re, im) {
  for (let i = 0; i < NFFT; i++) {
    const j = REV[i];
    if (j > i) {
      let t = re[i]; re[i] = re[j]; re[j] = t;
      t = im[i]; im[i] = im[j]; im[j] = t;
    }
  }
  for (let size = 2; size <= NFFT; size <<= 1) {
    const half = size >> 1;
    const step = NFFT / size;
    for (let start = 0; start < NFFT; start += size) {
      for (let k = 0, t = 0; k < half; k++, t += step) {
        const a = start + k;
        const b = a + half;
        const xr = re[b] * COS[t] - im[b] * SIN[t];
        const xi = re[b] * SIN[t] + im[b] * COS[t];
        re[b] = re[a] - xr; im[b] = im[a] - xi;
        re[a] += xr; im[a] += xi;
      }
    }
  }
}

/** scipy.signal.medfilt (zero padded at both ends), odd kernel. */
export function medfilt(x, k) {
  const out = new Float64Array(x.length);
  const half = (k - 1) / 2;
  const win = new Float64Array(k);
  for (let i = 0; i < x.length; i++) {
    for (let j = 0; j < k; j++) {
      const s = i + j - half;
      win[j] = s >= 0 && s < x.length ? x[s] : 0;
    }
    win.sort();
    out[i] = win[half];
  }
  return out;
}

// ---- bandwidth (bwe.effective_bandwidth) ---------------------------------------------------------------------------
const EDGE_DB = 20.0;
const CLIFF_DB = 18.0;

/** Long-term spectrum (dB per bin) of the active frames, and the frame energies it was chosen with. */
function ltasDb(spectra, speechEnergy) {
  const n = spectra.length;
  if (!n) return null;
  let max = 0;
  for (const e of speechEnergy) if (e > max) max = e;
  if (max <= 0) return null;
  const thr = Math.max(percentile(sortedCopy(speechEnergy), 60), max * 1e-4);
  let act = [];
  for (let i = 0; i < n; i++) if (speechEnergy[i] >= thr) act.push(i);
  if (act.length < 8) return null;
  if (act.length > MAX_FFT_FRAMES) {
    const m = act.length;
    act = Array.from({ length: MAX_FFT_FRAMES }, (_, i) => act[Math.trunc((i * (m - 1)) / (MAX_FFT_FRAMES - 1))]);
  }
  const total = new Float64Array(BINS);
  for (const i of act) {
    const s = spectra[i];
    for (let k = 0; k < BINS; k++) total[k] += s[k];
  }
  const db = new Float64Array(BINS);
  for (let k = 0; k < BINS; k++) db[k] = 10 * Math.log10(total[k] / act.length + 1e-20);
  return db;
}

function bandMean(db, lo, hi) {
  let s = 0;
  let c = 0;
  for (let k = 0; k < BINS; k++) {
    const f = k * BIN_HZ;
    if (f >= lo && f < hi) { s += db[k]; c++; }
  }
  return c ? s / c : NaN;
}

/** Where the speech spectrum ends (Hz), or RATE / 2 when there is no sharp cut-off. A port of bwe.effective_bandwidth. */
export function bandwidthOf(spectra, speechEnergy, extra = null) {
  const nyq = RATE / 2;
  const raw = ltasDb(spectra, speechEnergy);
  if (!raw) return nyq;
  const db = medfilt(raw, Math.max(3, Math.round(200 / BIN_HZ) | 1));
  const mean = (lo, hi) => bandMean(db, lo, hi);
  const ref = mean(600, 2400);
  const top = Math.min(nyq * 0.95, 20000);
  if (extra) {
    // dullness without a cliff: how far the top of the speech band sits below its middle
    extra.tiltDb = mean(4000, 8000) - ref;
  }
  for (let i = 0; i < BINS; i++) {
    const f = i * BIN_HZ;
    if (!(f >= 2400 && f < top && db[i] < ref - EDGE_DB)) continue;
    const fe = f;
    if (mean(fe * 1.15, nyq) >= ref - EDGE_DB) continue; // does not stay down: a dip, not an edge
    if (mean(fe * 0.8, fe * 0.9) - mean(fe * 1.15, fe * 1.4) < CLIFF_DB) continue; // shallow slope: natural roll-off
    const local = mean(fe * 0.8, fe * 0.9);
    for (let j = 0; j < BINS; j++) {
      if (j * BIN_HZ >= fe * 0.9 && db[j] < local - 12.0) return j * BIN_HZ;
    }
  }
  return nyq;
}

// ---- how fast the sound can stop (a blind guide to an echoey room) --------------------------------------------------
/**
 * From the 10 ms frame levels (dB): the 95th percentile of how far the level drops within 60 ms, taken from the loud parts.
 * Close-up speech stops abruptly (about 20 dB and more); in an echoey room the tail of each word fills the next moments, so the
 * level can only fall as fast as the room allows (a room that rings for half a second: 7 dB and less). Null when the background
 * is too close to the voice to see it (less than 30 dB between them) or there is too little loud sound to say.
 */
export function dropOf(levels, floorDb, peakDb) {
  if (peakDb - floorDb < 30) return null;
  const W = 6;
  const drops = [];
  for (let i = 0; i + W < levels.length; i++) if (levels[i] >= peakDb - 10) drops.push(levels[i] - levels[i + W]);
  if (drops.length < 30) return null;
  drops.sort((a, b) => a - b);
  return percentile(drops, 95);
}

/** Which parts of a file to look at: the whole of a short one, otherwise `count` stretches spread evenly over `total` seconds. */
export function planWindows(duration, total = 60, count = 6) {
  if (!(duration > 0)) return [];
  if (duration <= total * 1.25) return [{ start: 0, length: duration }];
  const length = total / count;
  return Array.from({ length: count }, (_, i) => {
    const mid = ((i + 0.5) * duration) / count;
    return { start: Math.min(duration - length, Math.max(0, mid - length / 2)), length };
  });
}

// ---- the noise figure (pipeline.noise_figure_db) --------------------------------------------------------------------
/** dB between the voice and the noise in it: what the noise remover (at full strength) keeps against what it takes out. */
export class NoiseFigure {
  constructor() {
    this.voice = 0;
    this.removed = 0;
    this.measured = false;
  }

  /** @param {Float32Array} x the audio  @param {Float32Array} y the noise remover's output for it, lined up, as long */
  add(x, y) {
    let v = 0;
    let r = 0;
    for (let i = 0; i < x.length; i++) {
      v += y[i] * y[i];
      const d = x[i] - y[i];
      r += d * d;
    }
    this.voice += v;
    this.removed += r;
    this.measured = true;
  }

  /** 99 when there was nothing to hear. */
  db() {
    if (this.voice + this.removed <= 0) return 99;
    return Math.min(99, 10 * Math.log10((this.voice + 1e-12) / (this.removed + 1e-12)));
  }
}

// ---- the analyser --------------------------------------------------------------------------------------------------
export class Analyser {
  /** @param {{channels?: number}} opts channels: 1 or 2 (what push() gets) */
  constructor({ channels = 1 } = {}) {
    this.channels = channels >= 2 ? 2 : 1;
    this.levels = new Float32Array(4096); // 10 ms frame energies, dB
    this.bandLevels = new Float32Array(4096); // the same for the speech band only
    this.nLevels = 0;
    this.sumSq = 0;
    this.bandSq = 0;
    this.bq = new Float64Array(8); // x1 x2 y1 y2 of the high-pass, then of the low-pass
    this.cnt = 0;
    this.samples = 0;
    this.carry = new Float32Array(NFFT);
    this.carryLen = 0;
    this.spectra = [];
    this.speechEnergy = [];
    this.re = new Float64Array(NFFT);
    this.im = new Float64Array(NFFT);
    // levels
    this.peak = [0, 0];
    this.clipped = [0, 0];
    this.run = [0, 0];
    this.exact = true;
    this.side = 0;
    this.mid = 0;
    // loudness
    this.kState = [0, 1].map(() => ({ x1: 0, x2: 0, y1: 0, y2: 0, z1: 0, z2: 0, w1: 0, w2: 0 }));
    this.sub = 0;
    this.subLen = 0;
    this.subs = [];
    this.blocks = [];
    this.segments = 1;
    this.noise = new NoiseFigure();
  }

  /** The mono mix of what was pushed and the noise remover's output for it (lined up, as long): the noise figure. */
  addCleaned(x, y) {
    this.noise.add(x, y);
  }

  /** The next push() is a different part of the file: nothing continues across the join. */
  segment() {
    this.sumSq = 0;
    this.bandSq = 0;
    this.bq.fill(0);
    this.cnt = 0;
    this.carryLen = 0;
    this.sub = 0;
    this.subLen = 0;
    this.subs = [];
    for (const s of this.kState) Object.assign(s, { x1: 0, x2: 0, y1: 0, y2: 0, z1: 0, z2: 0, w1: 0, w2: 0 });
    this.run = [0, 0];
    this.segments++;
  }

  /** @param {Float32Array[]} planes one array per channel (what the constructor was told), 48 kHz, equal lengths */
  push(planes) {
    const n = planes[0].length;
    if (!n) return;
    const ch = Math.min(planes.length, 2);
    const mono = ch === 1 ? planes[0] : new Float32Array(n);
    if (ch === 2) for (let i = 0; i < n; i++) mono[i] = (planes[0][i] + planes[1][i]) * 0.5;
    this.samples += n;
    this._levels(mono, n);
    this._spectra(mono, n);
    this._peaks(planes, ch, n);
    this._loudness(planes, ch, n);
  }

  _addLevel(v, band) {
    if (this.nLevels === this.levels.length) {
      const bigger = new Float32Array(this.levels.length * 2);
      bigger.set(this.levels);
      this.levels = bigger;
      const bigger2 = new Float32Array(this.bandLevels.length * 2);
      bigger2.set(this.bandLevels);
      this.bandLevels = bigger2;
    }
    this.bandLevels[this.nLevels] = band;
    this.levels[this.nLevels++] = v;
  }

  _levels(x, n) {
    let s = this.sumSq;
    let bs = this.bandSq;
    let c = this.cnt;
    const q = this.bq;
    const [h0, h1, h2, h3, h4] = BAND_HP;
    const [l0, l1, l2, l3, l4] = BAND_LP;
    for (let i = 0; i < n; i++) {
      const v = x[i];
      s += v * v;
      const hy = h0 * v + h1 * q[0] + h2 * q[1] - h3 * q[2] - h4 * q[3];
      q[1] = q[0]; q[0] = v; q[3] = q[2]; q[2] = hy;
      const ly = l0 * hy + l1 * q[4] + l2 * q[5] - l3 * q[6] - l4 * q[7];
      q[5] = q[4]; q[4] = hy; q[7] = q[6]; q[6] = ly;
      bs += ly * ly;
      if (++c === FRAME) {
        this._addLevel(10 * Math.log10(s / FRAME + 1e-12), 10 * Math.log10(bs / FRAME + 1e-12));
        s = 0;
        bs = 0;
        c = 0;
      }
    }
    this.sumSq = s;
    this.bandSq = bs;
    this.cnt = c;
  }

  _spectra(x, n) {
    let pos = 0;
    while (pos < n) {
      const take = Math.min(NFFT - this.carryLen, n - pos);
      this.carry.set(x.subarray(pos, pos + take), this.carryLen);
      this.carryLen += take;
      pos += take;
      if (this.carryLen < NFFT) break;
      this._frame();
      this.carry.copyWithin(0, HOP, NFFT);
      this.carryLen = NFFT - HOP;
    }
  }

  _frame() {
    const { re, im, carry } = this;
    for (let i = 0; i < NFFT; i++) { re[i] = carry[i] * HANN[i]; im[i] = 0; }
    fft(re, im);
    const p = new Float32Array(BINS);
    let e = 0;
    const lo = Math.ceil(150 / BIN_HZ);
    const hi = Math.floor(3500 / BIN_HZ);
    for (let k = 0; k < BINS; k++) {
      p[k] = re[k] * re[k] + im[k] * im[k];
      if (k >= lo && k <= hi) e += p[k];
    }
    if (this.spectra.length < MAX_FFT_FRAMES * 4) {
      this.spectra.push(p);
      this.speechEnergy.push(e);
    }
  }

  _peaks(planes, ch, n) {
    let side = 0;
    let mid = 0;
    for (let c = 0; c < ch; c++) {
      const x = planes[c];
      let pk = this.peak[c];
      let run = this.run[c];
      let clipped = this.clipped[c];
      for (let i = 0; i < n; i++) {
        const a = x[i] < 0 ? -x[i] : x[i];
        if (a > pk) pk = a;
        if (a >= CLIP_LEVEL) {
          run++;
          if (run === CLIP_RUN) clipped += CLIP_RUN;
          else if (run > CLIP_RUN) clipped++;
        } else run = 0;
      }
      this.peak[c] = pk;
      this.run[c] = run;
      this.clipped[c] = clipped;
    }
    if (ch === 2) {
      const l = planes[0];
      const r = planes[1];
      for (let i = 0; i < n; i++) {
        if (l[i] !== r[i]) this.exact = false;
        const d = l[i] - r[i];
        const m = l[i] + r[i];
        side += d * d;
        mid += m * m;
      }
      this.side += side;
      this.mid += mid;
    }
  }

  _loudness(planes, ch, n) {
    const SUB = RATE / 10; // 100 ms
    for (let i = 0; i < n; i++) {
      let power = 0;
      for (let c = 0; c < ch; c++) {
        const s = this.kState[c];
        const x = planes[c][i];
        // stage 1 (shelf) and stage 2 (high-pass), direct form I / II transposed mix kept simple: two biquads in series
        const y1 = K1.b[0] * x + K1.b[1] * s.x1 + K1.b[2] * s.x2 - K1.a[0] * s.y1 - K1.a[1] * s.y2;
        s.x2 = s.x1; s.x1 = x; s.y2 = s.y1; s.y1 = y1;
        const y2 = K2.b[0] * y1 + K2.b[1] * s.z1 + K2.b[2] * s.z2 - K2.a[0] * s.w1 - K2.a[1] * s.w2;
        s.z2 = s.z1; s.z1 = y1; s.w2 = s.w1; s.w1 = y2;
        power += y2 * y2;
      }
      this.sub += power;
      if (++this.subLen === SUB) {
        this.subs.push(this.sub / SUB);
        this.sub = 0;
        this.subLen = 0;
        if (this.subs.length >= 4) {
          const k = this.subs.length;
          this.blocks.push((this.subs[k - 1] + this.subs[k - 2] + this.subs[k - 3] + this.subs[k - 4]) / 4);
        }
      }
    }
  }

  /** Integrated loudness (LUFS, BS.1770 gating) of what was pushed, or null when there is not 0.4 s of it. */
  loudness() {
    const lk = (z) => -0.691 + 10 * Math.log10(z);
    const gated = this.blocks.filter((z) => z > 0 && lk(z) > -70);
    if (!gated.length) return null;
    const mean = (a) => a.reduce((s, v) => s + v, 0) / a.length;
    const rel = lk(mean(gated)) - 10;
    const kept = gated.filter((z) => lk(z) > rel);
    return kept.length ? lk(mean(kept)) : null;
  }

  /** Everything measured so far. Cheap enough to call once or twice; the FFT frames are kept, not recomputed. */
  result() {
    const seconds = this.samples / RATE;
    const lv = this.levels.subarray(0, this.nLevels);
    const loud = [];
    let silent = 0;
    for (let i = 0; i < lv.length; i++) {
      if (lv[i] > SILENCE_DB) loud.push(lv[i]);
      else silent++;
    }
    const sorted = Float64Array.from(loud).sort();
    const enough = loud.length >= 10;
    const p10 = enough ? percentile(sorted, 10) : NaN;
    const p90 = enough ? percentile(sorted, 90) : NaN;
    const p95 = enough ? percentile(sorted, 95) : NaN;
    // Share of the frames that are clearly above the room: speech, as far as a level alone can tell.
    const gate = enough ? Math.max(p10 + 12, p95 - 25) : Infinity;
    let above = 0;
    for (const v of loud) if (v >= gate) above++;
    const speechFraction = enough ? above / loud.length : 0;
    const extra = {};
    const bandwidthHz = this.spectra.length >= 3 ? bandwidthOf(this.spectra, this.speechEnergy, extra) : RATE / 2;
    const clipFraction = this.samples ? Math.max(...this.clipped.slice(0, this.channels)) / this.samples : 0;
    let layout = 'mono';
    if (this.channels === 2) {
      const dead = this.peak.map((p) => p < DEAD_CHANNEL_PEAK);
      if (dead[0] && dead[1]) layout = 'silent';
      else if (dead[0] || dead[1]) layout = 'one-side';
      else if (this.exact || (this.mid > 0 && 10 * Math.log10(this.side / this.mid + 1e-30) < DUAL_MONO_SIDE_DB)) layout = 'dual-mono';
      else layout = 'stereo';
    }
    const drop = enough ? dropOf(lv, p10, p95) : null;
    // how much the level of the speech band moves up and down: words and pauses move it, a fan, rumble or hum do not
    const band = [];
    for (let i = 0; i < lv.length; i++) if (lv[i] > SILENCE_DB) band.push(this.bandLevels[i]);
    band.sort((a, b) => a - b);
    const bandDyn = band.length >= 10 ? percentile(band, 90) - percentile(band, 10) : 0;
    const dynamic = enough ? p95 - p10 : 0;
    return {
      seconds,
      segments: this.segments,
      noiseDb: this.noise.measured ? this.noise.db() : null, // null: the noise remover didn't run (no recommendation then)
      floorDb: p10,
      levelDb: p90,
      peakDb: p95,
      silentFraction: lv.length ? silent / lv.length : 1,
      speechFraction,
      bandDynDb: bandDyn,
      hasSpeech: enough && bandDyn >= SPEECH_BAND_DYN_DB,
      bandwidthHz,
      tiltDb: extra.tiltDb ?? null,
      dropDb: drop,
      clipFraction,
      peak: Math.max(...this.peak.slice(0, this.channels)),
      layout,
      loudnessLufs: this.loudness(),
    };
  }
}
