// Speech-to-text worker for the free in-browser subtitles (Whisper base, 8-bit, about 80 MB, run by ONNX Runtime Web on
// the visitor's processor). One instance per page; it handles one request at a time, in order.
//
// Protocol (main -> worker):
//   { type: 'init', modelHost, model, ortBase, plainWasm }   -> progress* then { type: 'ready' } | { type: 'error', message }
//        modelHost: absolute URL the model folder hangs off (the site's dfn/models/), model: its id there;
//        ortBase: absolute URL of the runtime's folder; plainWasm: use the build without WebGPU support (older Safari)
//   { type: 'detect', id, audio: Float32Array (16 kHz mono, up to 30 s) }   -> { type: 'language', id, language }
//   { type: 'transcribe', id, audio, language }   -> { type: 'result', id, segments: [{start, end, text}] }  (seconds)
//   any request can answer { type: 'error', id, message }
// Progress while the model downloads: { type: 'progress', file, loaded, total } (bytes, per file).

import { env, pipeline, Tensor } from '@huggingface/transformers';

let asr = null;

async function init({ modelHost, model, ortBase, plainWasm }) {
  env.allowLocalModels = false;
  env.allowRemoteModels = true;
  env.remoteHost = modelHost;
  env.remotePathTemplate = '{model}/';
  env.useBrowserCache = true; // the files stay in the browser's cache: the model is downloaded once
  const build = plainWasm ? 'ort-wasm-simd-threaded' : 'ort-wasm-simd-threaded.asyncify';
  env.backends.onnx.wasm.wasmPaths = { mjs: `${ortBase}${build}.mjs`, wasm: `${ortBase}${build}.wasm` };
  // Load the runtime's glue code from our own address, not as a blob: the site's script policy allows only 'self'.
  // (The browser's HTTP cache keeps both files; the model files are what the library caches itself.)
  env.useWasmCache = false;
  env.backends.onnx.wasm.numThreads = 1; // threads need a cross-origin isolated page, which this site is not
  asr = await pipeline('automatic-speech-recognition', model, {
    device: 'wasm',
    dtype: { encoder_model: 'q8', decoder_model_merged: 'q8' },
    progress_callback: (p) => {
      if (p.status === 'progress') self.postMessage({ type: 'progress', file: p.file, loaded: p.loaded, total: p.total });
    },
  });
}

/** The language spoken, from one step of the decoder after <|startoftranscript|> (Whisper's own way of telling). */
async function detect(audio) {
  const gc = asr.model.generation_config;
  const { input_features } = await asr.processor(audio);
  const start = new Tensor('int64', BigInt64Array.from([BigInt(gc.decoder_start_token_id)]), [1, 1]);
  const out = await asr.model({ input_features, decoder_input_ids: start });
  const vocab = out.logits.dims.at(-1);
  const last = out.logits.data.length - vocab;
  let best = null;
  for (const [token, id] of Object.entries(gc.lang_to_id)) {
    const score = out.logits.data[last + id];
    if (!best || score > best[1]) best = [token, score];
  }
  return best ? best[0].replace(/[<|>]/g, '') : 'en';
}

async function transcribe(audio, language) {
  const seconds = audio.length / 16000;
  const r = await asr(audio, { return_timestamps: true, language: language || 'en', task: 'transcribe' });
  const chunks = r.chunks?.length ? r.chunks : [{ timestamp: [0, seconds], text: r.text }];
  return chunks
    .map((c) => ({
      start: Math.max(0, Math.min(seconds, c.timestamp?.[0] ?? 0)),
      end: Math.max(0, Math.min(seconds, c.timestamp?.[1] ?? seconds)),
      text: String(c.text || '').trim(),
    }))
    .filter((s) => s.text);
}

let chain = Promise.resolve();
self.onmessage = (e) => {
  const m = e.data;
  chain = chain.then(async () => {
    try {
      if (m.type === 'init') {
        await init(m);
        self.postMessage({ type: 'ready' });
      } else if (m.type === 'detect') {
        self.postMessage({ type: 'language', id: m.id, language: await detect(m.audio) });
      } else if (m.type === 'transcribe') {
        self.postMessage({ type: 'result', id: m.id, segments: await transcribe(m.audio, m.language) });
      }
    } catch (err) {
      self.postMessage({ type: 'error', id: m.id, message: String(err?.message || err) });
    }
  });
};
