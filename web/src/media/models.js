// Where the free tools' models come from: Hugging Face, every file pinned to one commit, with its size and SHA-256.
// A file is used only when its SHA-256 matches, and a checked copy is kept in the browser (Cache Storage), so it is
// downloaded once. The noise remover's glue code (df.js), the speech runtime and the decoder stay on the site itself.
//
// __MODEL_HOST__ is https://huggingface.co/ (vite.config.js; the test suites point it at a local copy of the same files).

export const MODEL_HOST = __MODEL_HOST__;

// Our own files: SilverJunk08/audio-cleaner-models (deploy/models.lock pins the same commit for the server).
const OURS = { repo: 'SilverJunk08/audio-cleaner-models', revision: 'PENDING-UPLOAD' };

/** The noise remover: the libDF WebAssembly build (it must be the build df.js on the site belongs to: scripts/sync-dfn.mjs
 *  checks that) and the DeepFilterNet3 model. */
export const DENOISER = {
  wasm: { ...OURS, path: 'deepfilternet3/df_bg.wasm', size: 14028025, sha256: '8ed0596104271ff3503d79ae1571b4e5674d20f32982d0781b80c70f1c4ac4c1' },
  model: { ...OURS, path: 'deepfilternet3/DeepFilterNet3_onnx.tar.gz', size: 7983136, sha256: 'c94d91f70911001c946e0fabb4aa9adc37045f45a03b56008cb0c8244cb63616' },
};

/** The free subtitles' model (Whisper base, 8-bit ONNX; OpenAI, MIT), as published by onnx-community. */
export const SPEECH = {
  repo: 'onnx-community/whisper-base',
  revision: '1846881b6b3a3024392c1eea3ad983695bc23925',
  dtype: { encoder_model: 'q8', decoder_model_merged: 'q8' },
  files: {
    'config.json': { size: 2243, sha256: 'f4d0608f7d918166da7edb3e188de5ef1bfe70d9802e785d271fd88111e9cf4b' },
    'generation_config.json': { size: 3832, sha256: '61070cf8de25b1e9256e8e102ded49d8d24a8369ed36ef84fdf21549e68125a0' },
    'preprocessor_config.json': { size: 339, sha256: 'a6a76d28c93edb273669eb9e0b0636a2bddbb1272c3261e47b7ca6dfdbac1b8d' },
    'tokenizer.json': { size: 2480466, sha256: '27fc476bfe7f17299480be2273fc0608e4d5a99aba2ab5dec5374b4482d1a566' },
    'tokenizer_config.json': { size: 282682, sha256: '2e036e4dbacfdeb7242c7d4ec4149f4a16e86026048f94d1637e3a8ee9c6a573' },
    'onnx/encoder_model_quantized.onnx': { size: 23201314, sha256: '5862993336bf33acd23736071aae2b32261d3b1b2f37780194460d4ef974dd46' },
    'onnx/decoder_model_merged_quantized.onnx': { size: 53693315, sha256: 'fa3ef9902734ce5ae6f9ef2bdb2ba9a6c4b5785b09f4f420ce036573dc9d090b' },
  },
};

export const fileUrl = ({ repo, revision, path }) => `${MODEL_HOST}${repo}/resolve/${revision}/${path}`;

const CACHE = 'acw-models';

async function sha256Hex(bytes) {
  const d = new Uint8Array(await crypto.subtle.digest('SHA-256', bytes));
  return Array.from(d, (b) => b.toString(16).padStart(2, '0')).join('');
}

async function openCache() {
  try {
    return typeof caches === 'undefined' ? null : await caches.open(CACHE);
  } catch {
    return null; // private window or storage blocked: downloaded each time
  }
}

/**
 * The bytes of one pinned file: from the browser's copy when it has one that still matches, else downloaded (each
 * received chunk's size goes to `onBytes`), checked and kept. Throws when the download fails or does not match.
 * @param {{repo:string, revision:string, path:string, size:number, sha256:string}} pin
 */
export async function fetchPinned(pin, { onBytes, priority = 'low' } = {}) {
  const url = fileUrl(pin);
  const cache = await openCache();
  const kept = await cache?.match(url).catch(() => null);
  if (kept) {
    const bytes = new Uint8Array(await kept.arrayBuffer());
    if (bytes.length === pin.size && (await sha256Hex(bytes)) === pin.sha256) {
      onBytes?.(bytes.length);
      return bytes;
    }
    await cache.delete(url).catch(() => {});
  }
  const res = await fetch(url, { priority, credentials: 'omit', mode: 'cors' });
  if (!res.ok) throw new Error(`Could not download ${pin.path.split('/').pop()} (${res.status}).`);
  const bytes = new Uint8Array(pin.size);
  let have = 0;
  const reader = res.body.getReader();
  for (;;) {
    const { done, value } = await reader.read();
    if (done) break;
    if (have + value.length > pin.size) throw new Error(`${pin.path.split('/').pop()} is not the expected file.`);
    bytes.set(value, have);
    have += value.length;
    onBytes?.(value.length);
  }
  if (have !== pin.size || (await sha256Hex(bytes)) !== pin.sha256) throw new Error(`${pin.path.split('/').pop()} is not the expected file.`);
  await cache?.put(url, new Response(bytes, { headers: { 'content-type': 'application/octet-stream', 'content-length': String(pin.size) } })).catch(() => {});
  return bytes;
}
