// src/media/convert.js bundled for node (esbuild) with mediabunny stood in for by a stub whose abilities a test sets, so the output list
// can be computed with no browser: `await loadConvert({ avc: true, vp9: true })`. Each call is a fresh copy of the module (what this browser can do
// is asked once per copy).
import { dirname } from 'node:path';
import { fileURLToPath } from 'node:url';
import { build } from 'esbuild';

const HERE = dirname(fileURLToPath(import.meta.url));

const STUB = `
  const can = () => globalThis.__ABILITIES || {};
  const fmt = (codecs) => class { getSupportedVideoCodecs() { return codecs; } };
  export const canEncodeAudio = async (codec) => !!can().audio?.[codec];
  export const canEncodeVideo = async (codec) => !!can()[codec === 'avc' ? 'avc' : codec === 'vp9' ? 'vp9' : 'vp8'];
  export const WavOutputFormat = fmt([]);
  export const Mp3OutputFormat = fmt([]);
  export const OggOutputFormat = fmt([]);
  export const Mp4OutputFormat = fmt(['avc', 'hevc', 'vp9', 'av1']);
  export const MovOutputFormat = fmt(['avc', 'hevc']);
  export const MkvOutputFormat = fmt(['avc', 'hevc', 'vp8', 'vp9', 'av1']);
  export const WebMOutputFormat = fmt(['vp8', 'vp9', 'av1']);
`;

let text = null;
let copies = 0;

export async function loadConvert(abilities = {}) {
  if (text === null) {
    const bundle = await build({
      stdin: { contents: "export * from '../src/media/convert.js';", resolveDir: HERE, loader: 'js' },
      bundle: true,
      format: 'esm',
      platform: 'node',
      write: false,
      logLevel: 'silent',
      define: { 'import.meta.env.BASE_URL': '"/"', __DFN_VERSION__: '""', __DFN_JS_SIZE__: '0', __MODEL_HOST__: '""', __LIBAV_DIR__: '""', __ORT_DIR__: '""' },
      plugins: [{
        name: 'stub-mediabunny',
        setup(b) {
          b.onResolve({ filter: /^(mediabunny|@mediabunny\/mp3-encoder)$/ }, (a) => ({ path: a.path, namespace: 'stub' }));
          b.onLoad({ filter: /.*/, namespace: 'stub' }, () => ({ contents: STUB, loader: 'js' }));
        },
      }],
    });
    text = bundle.outputFiles[0].text;
  }
  globalThis.__ABILITIES = abilities; // (read when the module asks the browser what it can do, once per copy)
  const m = await import(`data:text/javascript;base64,${Buffer.from(`${text}\n// copy ${copies++}`).toString('base64')}`);
  await m.outputCaps(); // (asked now, while the abilities are these)
  return m;
}
