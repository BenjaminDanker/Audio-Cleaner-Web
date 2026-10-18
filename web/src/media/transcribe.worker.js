// Speech-to-text worker for the free in-browser subtitles (Whisper base, 8-bit, about 80 MB, run by ONNX Runtime Web on
// the visitor's processor). One instance per page; it handles one request at a time, in order.
//
// Protocol (main -> worker):
//   { type: 'init', modelHost, speech, ortBase }   -> progress* then { type: 'ready' } | { type: 'error', message }
//        modelHost: https://huggingface.co/ (or the tests' copy), speech: models.js SPEECH (repo, revision, dtype, and
//        every file's size and SHA-256); ortBase: absolute URL of the runtime's folder
//   { type: 'detect', id, audio: Float32Array (16 kHz mono, up to 30 s) }   -> { type: 'language', id, language }
//   { type: 'transcribe', id, audio, language }   -> { type: 'result', id, segments: [{start, end, text}] }  (seconds)
//   any request can answer { type: 'error', id, message }
// Progress while the model downloads: { type: 'progress', file, loaded, total } (bytes, per file).

import { env, pipeline, Tensor } from '@huggingface/transformers';

let asr = null;

/**
 * Every model file goes through here: it must be one of the pinned files, it is always taken from the pinned commit,
 * its bytes must have the pinned size and SHA-256 (a file that does not is never used, nor kept in the browser's cache),
 * and its download progress is reported. The library asks for some files at "main" before it applies the revision (its
 * file list), and for the first byte of every file (a Range request) just to learn its size for a progress total: the
 * first go to the pinned commit too, the second are answered from the pins, and a file asked for again is answered
 * from the checked copy, so every file is downloaded exactly once.
 */
function checkedFetch(speech) {
  const realFetch = env.fetch;
  const base = `${env.remoteHost}${speech.repo}/resolve/`;
  const pinned = `${base}${speech.revision}/`;
  const checked = new Map(); // file -> its checked bytes
  const answer = (bytes) => new Response(bytes, { status: 200, headers: { 'content-type': 'application/octet-stream', 'content-length': String(bytes.length) } });
  env.fetch = async (input, init) => {
    const url = String(input?.url ?? input);
    const rest = url.startsWith(base) ? url.slice(base.length).split('?')[0] : '';
    const slash = rest.indexOf('/');
    const rev = rest.slice(0, slash);
    const name = slash > 0 && (rev === speech.revision || rev === 'main') ? rest.slice(slash + 1) : null;
    const pin = name ? speech.files[name] : null;
    if (!pin) throw new Error(`Not one of the subtitle maker's files: ${url}`);
    if (new Headers(init?.headers).get('range') === 'bytes=0-0') {
      return new Response(new Uint8Array(1), { status: 206, headers: { 'content-range': `bytes 0-0/${pin.size}`, 'content-type': 'application/octet-stream' } });
    }
    if (checked.has(name)) return answer(checked.get(name));
    const res = await realFetch(pinned + name, { credentials: 'omit', mode: 'cors' });
    if (!res.ok) return res;
    const bytes = new Uint8Array(pin.size);
    let have = 0;
    const reader = res.body.getReader();
    for (;;) {
      const { done, value } = await reader.read();
      if (done) break;
      if (have + value.length > pin.size) throw new Error(`${name} is not the expected file.`);
      bytes.set(value, have);
      have += value.length;
      self.postMessage({ type: 'progress', file: name, loaded: have, total: pin.size });
    }
    const digest = Array.from(new Uint8Array(await crypto.subtle.digest('SHA-256', bytes)), (b) => b.toString(16).padStart(2, '0')).join('');
    if (have !== pin.size || digest !== pin.sha256) throw new Error(`${name} is not the expected file.`);
    checked.set(name, bytes);
    return answer(bytes);
  };
}

async function init({ modelHost, speech, ortBase }) {
  env.allowLocalModels = false;
  env.allowRemoteModels = true;
  env.remoteHost = modelHost;
  env.remotePathTemplate = '{model}/resolve/{revision}/';
  env.useBrowserCache = true; // the checked files stay in the browser's cache: the model is downloaded once
  checkedFetch(speech);
  // The model only ever runs on the processor (device 'wasm'), so the plain build is enough: the one with WebGPU/asyncify
  // support is twice the size to download and about a fifth slower.
  env.backends.onnx.wasm.wasmPaths = { mjs: `${ortBase}ort-wasm-simd-threaded.mjs`, wasm: `${ortBase}ort-wasm-simd-threaded.wasm` };
  // Load the runtime's glue code from our own address, not as a blob: the site's script policy allows only 'self'.
  // (The browser's HTTP cache keeps both files; the model files are what the library caches itself.)
  env.useWasmCache = false;
  env.backends.onnx.wasm.numThreads = 1; // threads need a cross-origin isolated page, which this site is not
  asr = await pipeline('automatic-speech-recognition', speech.repo, { device: 'wasm', revision: speech.revision, dtype: speech.dtype });
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
  // no_repeat_ngram_size 10: stops the decoder repeating itself in a loop (one looping window cost 5x the time and filled
  // the subtitles with the same line), without touching phrases people really repeat: measured on FLEURS in 10
  // languages, sung lyrics and repeated sentences (smaller values damaged those).
  const r = await asr(audio, { return_timestamps: true, language: language || 'en', task: 'transcribe', no_repeat_ngram_size: 10 });
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
