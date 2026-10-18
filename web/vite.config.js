import { defaultClientConditions, defineConfig, loadEnv } from 'vite';
import react from '@vitejs/plugin-react';
import { createHash } from 'node:crypto';
import { existsSync, readFileSync, readdirSync, statSync } from 'node:fs';

// Version string for df.js (unhashed, in public/dfn), appended as ?v= so it can be cached for a long time.
function dfnVersion() {
  const p = new URL('./public/dfn/df.js', import.meta.url);
  return existsSync(p) ? createHash('sha1').update(readFileSync(p)).digest('hex').slice(0, 10) : '';
}

// Its size, for the progress readout while the noise remover downloads (src/media/denoise.js).
function dfnJsSize() {
  const p = new URL('./public/dfn/df.js', import.meta.url);
  return existsSync(p) ? statSync(p).size : 0;
}

// Where the free tools' models are downloaded from (src/media/models.js): Hugging Face. The test suites set
// ACW_MODEL_HOST to a local copy laid out the same way (<repo>/resolve/<revision>/<path>).
function modelHost() {
  const h = process.env.ACW_MODEL_HOST || 'https://huggingface.co/';
  return h.endsWith('/') ? h : `${h}/`;
}

// The versioned folder of the libav.js decoder (scripts/sync-dfn.mjs), or '' when it is not built.
function libavDir() {
  const p = new URL('./public/dfn/libav/', import.meta.url);
  const dirs = existsSync(p) ? readdirSync(p) : [];
  return dirs.length === 1 ? dirs[0] : '';
}

// The versioned folder of the speech runtime (scripts/sync-dfn.mjs), or '' when it is not there.
function ortDir() {
  const p = new URL('./public/dfn/ort/', import.meta.url);
  const dirs = existsSync(p) ? readdirSync(p) : [];
  return dirs.length === 1 ? dirs[0] : '';
}

// The public address of the site (no trailing slash), for the absolute links in index.html (canonical, social previews).
// Set VITE_SITE_ORIGIN when building for another domain.
function siteOrigin(env) {
  return (process.env.VITE_SITE_ORIGIN || env.VITE_SITE_ORIGIN || 'https://mc-multi-world.space').replace(/\/+$/, '');
}

// Fills %SITE_ORIGIN% in index.html and writes robots.txt next to the app (under the base path). Crawlers only read
// the robots.txt at the root of a domain, so the front door has to serve one there too (it is not part of this build).
function sitePlugin(origin, base) {
  return {
    name: 'acw-site',
    transformIndexHtml: (html) => html.replaceAll('%SITE_ORIGIN%', origin),
    generateBundle() {
      this.emitFile({
        type: 'asset',
        fileName: 'robots.txt',
        source: ['User-agent: *', `Allow: ${base}`, `Disallow: ${base}app`, `Disallow: ${base}login`, `Disallow: ${base}api/`, ''].join('\n'),
      });
    },
  };
}

// The app is served under VITE_BASE (default "/audio/", use "/" on a dedicated domain).
// API calls go to `${BASE_URL}api/` on the same origin, so no CORS and cookie auth just works.
export default defineConfig(({ mode }) => {
  const env = loadEnv(mode, process.cwd(), 'VITE_');
  let base = process.env.VITE_BASE || env.VITE_BASE || '/audio/';
  if (!base.startsWith('/')) base = `/${base}`;
  if (!base.endsWith('/')) base = `${base}/`;
  const backend = process.env.ACW_DEV_BACKEND || 'http://127.0.0.1:8791';

  return {
    base,
    plugins: [react(), sitePlugin(siteOrigin(env), base)],
    // The speech runtime's build that loads its WebAssembly from where we say (the gated public/dfn/ort/ folder,
    // src/media/transcribe.worker.js) instead of having a 26 MB copy of it put among the public assets.
    resolve: { conditions: ['onnxruntime-web-use-extern-wasm', ...defaultClientConditions] },
    define: { __DFN_VERSION__: JSON.stringify(dfnVersion()), __DFN_JS_SIZE__: JSON.stringify(dfnJsSize()), __MODEL_HOST__: JSON.stringify(modelHost()), __LIBAV_DIR__: JSON.stringify(libavDir()), __ORT_DIR__: JSON.stringify(ortDir()) },
    worker: { format: 'es' },
    build: {
      target: 'es2022',
      sourcemap: false,
      // Never inline assets as data: URLs - AudioWorklet modules and workers must load from same-origin URLs (strict CSP).
      assetsInlineLimit: 0,
      // mediabunny (~700 kB, lazy-loaded only after a file is chosen) is the one deliberately large chunk.
      chunkSizeWarningLimit: 800,
      rollupOptions: {
        output: {
          manualChunks(id) {
            if (id.includes('node_modules/mediabunny')) return 'mediabunny';
            if (id.includes('node_modules/react-router') || id.includes('node_modules/@remix-run')) return 'router';
            if (id.includes('node_modules/react-dom') || id.includes('node_modules/react/') || id.includes('node_modules/scheduler')) return 'react';
          },
        },
      },
    },
    server: {
      port: 5173,
      proxy: {
        [`${base}api`]: {
          target: backend,
          ws: true,
          changeOrigin: true,
          // The backend only accepts state-changing requests whose Origin equals its public origin.
          configure(proxy) {
            proxy.on('proxyReq', (req) => req.setHeader('origin', backend));
            proxy.on('proxyReqWs', (req) => req.setHeader('origin', backend));
          },
        },
      },
    },
  };
});
