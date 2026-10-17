// Noise-remover worker. One instance cleans one audio channel.
//
// Protocol (main -> worker):
//   { type: 'init', dfnBase, attenDb, version, module, model }
//        module: the compiled WebAssembly.Module, model: the model bytes (both from the page's one shared download, so
//        the worker fetches nothing but the small df.js glue module, which is in the HTTP cache by then)
//        -> { type: 'ready' } | { type: 'error', message }
//   { type: 'process', seq, samples: Float32Array }  -> { type: 'out', seq, samples }   (arbitrary length, 48 kHz mono)
//   { type: 'flush', seq }                           -> { type: 'out', seq, samples, final: true }
//   { type: 'atten', db }
//   { type: 'close' }
// The output lines up with the input and is just as long in total (dfnstream.js).
import { DfnStream, loadDfnModule } from './dfnstream.js';

let stream = null;

self.onmessage = async (e) => {
  const m = e.data;
  try {
    if (m.type === 'init') {
      const mod = await loadDfnModule(m.dfnBase, m.version, m.module);
      stream = new DfnStream(mod, m.model, m.attenDb);
      self.postMessage({ type: 'ready' });
    } else if (m.type === 'process') {
      const samples = stream.process(m.samples);
      self.postMessage({ type: 'out', seq: m.seq, samples }, [samples.buffer]);
    } else if (m.type === 'flush') {
      const samples = stream.flush();
      self.postMessage({ type: 'out', seq: m.seq, samples, final: true }, [samples.buffer]);
    } else if (m.type === 'atten') {
      stream?.setAtten(m.db);
    } else if (m.type === 'close') {
      stream?.free();
      stream = null;
      self.close();
    }
  } catch (err) {
    self.postMessage({ type: 'error', seq: m.seq, message: String(err?.message || err) });
  }
};
