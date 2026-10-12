// Streaming windowed-sinc resampler (one instance per channel). Chunk boundaries are seamless because
// the filter history is carried between calls. Quality: Lanczos-windowed sinc, 12 zero crossings,
// low-pass at the lower of the two Nyquist frequencies.

const TABLE_RES = 256; // table entries per input sample

export class Resampler {
  constructor(inRate, outRate = 48000, lobes = 12) {
    this.inRate = inRate;
    this.outRate = outRate;
    this.step = inRate / outRate; // input samples advanced per output sample
    this.passthrough = inRate === outRate;
    this.cutoff = Math.min(1, outRate / inRate);
    this.halfWidth = Math.ceil(lobes / this.cutoff); // in input samples
    const n = this.halfWidth * TABLE_RES + 2;
    this.table = new Float32Array(n);
    const fc = this.cutoff;
    for (let i = 0; i < n; i++) {
      const u = i / TABLE_RES;
      const x = Math.PI * fc * u;
      const sinc = u === 0 ? 1 : Math.sin(x) / x;
      const w = u >= this.halfWidth ? 0 : (() => { const y = Math.PI * u / this.halfWidth; return y === 0 ? 1 : Math.sin(y) / y; })();
      this.table[i] = fc * sinc * w;
    }
    this.buf = new Float32Array(this.halfWidth + 1); // zeros before the first sample
    this.base = -(this.halfWidth + 1); // global input index of buf[0]
    this.outIndex = 0; // number of output samples produced
    this.totalIn = 0;
  }

  /** Feed input samples, get all output samples that are fully determined so far. */
  process(input) {
    this.totalIn += input.length;
    if (this.passthrough) {
      this.outIndex += input.length;
      return input;
    }
    return this._run(input);
  }

  /** Flush the tail; total output length matches round(totalIn * outRate / inRate). */
  finish() {
    if (this.passthrough) return new Float32Array(0);
    const tail = this._run(new Float32Array(this.halfWidth + 2));
    const want = Math.round((this.totalIn * this.outRate) / this.inRate);
    const have = this.outIndex;
    const extra = want - (have - tail.length);
    return extra < tail.length ? tail.subarray(0, Math.max(0, extra)) : tail;
  }

  _run(input) {
    const hw = this.halfWidth;
    const merged = new Float32Array(this.buf.length + input.length);
    merged.set(this.buf, 0);
    merged.set(input, this.buf.length);
    const base = this.base;
    const lastIdx = base + merged.length - 1; // global index of last available sample
    const table = this.table;
    let t = this.outIndex * this.step;
    // an output at time t needs samples up to floor(t)+hw
    const maxOut = Math.ceil((lastIdx - hw + 1 - t) / this.step) + 2;
    const res = new Float32Array(Math.max(0, maxOut));
    let n = 0;
    while (Math.floor(t) + hw <= lastIdx) {
      const i0 = Math.floor(t);
      let acc = 0;
      let wsum = 0;
      for (let k = i0 - hw + 1; k <= i0 + hw; k++) {
        const u = Math.abs(t - k) * TABLE_RES;
        const ui = u | 0;
        const w = table[ui] + (table[ui + 1] - table[ui]) * (u - ui);
        const s = merged[k - base];
        acc += s * w;
        wsum += w;
      }
      res[n++] = wsum > 1e-6 ? acc / wsum : acc;
      this.outIndex++;
      t = this.outIndex * this.step;
    }
    // keep history needed for the next output
    const keepFrom = Math.max(base, Math.floor(t) - hw);
    this.buf = merged.slice(keepFrom - base);
    this.base = keepFrom;
    return res.subarray(0, n);
  }
}
