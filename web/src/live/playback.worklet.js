// Plays frames posted from the main thread through a small jitter buffer.
// Messages in:  {type:'frame', samples: Float32Array} | {type:'clear'} | {type:'prefill', samples:number}
// Messages out: {type:'stats', queuedSamples, underruns, dropped} about every 250 ms.
// Latency cap: if the buffer ever holds more than HIGH (clock drift, a network burst), drop the oldest frames
// down to TARGET so the monitor never lags noticeably behind the microphone.
const HIGH = 48000 * 0.18;
const TARGET = 48000 * 0.09;

class PlaybackProcessor extends AudioWorkletProcessor {
  constructor(options) {
    super();
    const o = (options && options.processorOptions) || {};
    this.prefill = o.prefill || 2880; // 60 ms
    this.queue = [];
    this.cur = null;
    this.pos = 0;
    this.queued = 0;
    this.started = false;
    this.underruns = 0;
    this.dropped = 0;
    this.sinceStats = 0;
    this.port.onmessage = (e) => {
      const m = e.data;
      if (m.type === 'frame') {
        this.queue.push(m.samples);
        this.queued += m.samples.length;
        if (this.queued > HIGH) {
          while (this.queued > TARGET && this.queue.length > 1) {
            const d = this.queue.shift();
            this.queued -= d.length;
            this.dropped++;
          }
        }
      } else if (m.type === 'clear') {
        this.queue = [];
        this.cur = null;
        this.pos = 0;
        this.queued = 0;
        this.started = false;
      } else if (m.type === 'prefill') {
        this.prefill = m.samples;
      }
    };
  }

  process(inputs, outputs) {
    const out = outputs[0][0];
    if (!this.started) {
      if (this.queued >= this.prefill) this.started = true;
      else out.fill(0);
    }
    if (this.started) {
      let i = 0;
      while (i < out.length) {
        if (!this.cur) {
          this.cur = this.queue.shift();
          this.pos = 0;
          if (!this.cur) break;
        }
        const n = Math.min(out.length - i, this.cur.length - this.pos);
        out.set(this.cur.subarray(this.pos, this.pos + n), i);
        i += n;
        this.pos += n;
        this.queued -= n;
        if (this.pos >= this.cur.length) this.cur = null;
      }
      if (i < out.length) {
        out.fill(0, i);
        this.started = false; // underrun: refill before resuming
        this.underruns++;
      }
    }
    this.sinceStats += out.length;
    if (this.sinceStats >= 12000) {
      this.sinceStats = 0;
      this.port.postMessage({ type: 'stats', queuedSamples: this.queued, underruns: this.underruns, dropped: this.dropped });
    }
    return true;
  }
}

registerProcessor('acw-playback', PlaybackProcessor);
