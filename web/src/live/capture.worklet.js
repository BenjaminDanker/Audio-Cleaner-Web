// Collects the microphone into 960-sample (20 ms at 48 kHz) mono frames and posts them to the main thread.
const FRAME = 960;

class CaptureProcessor extends AudioWorkletProcessor {
  constructor() {
    super();
    this.buf = new Float32Array(FRAME);
    this.n = 0;
  }

  process(inputs) {
    const ch = inputs[0] && inputs[0][0];
    if (!ch) return true;
    for (let i = 0; i < ch.length; i++) {
      this.buf[this.n++] = ch[i];
      if (this.n === FRAME) {
        this.port.postMessage(this.buf, [this.buf.buffer]);
        this.buf = new Float32Array(FRAME);
        this.n = 0;
      }
    }
    return true;
  }
}

registerProcessor('acw-capture', CaptureProcessor);
