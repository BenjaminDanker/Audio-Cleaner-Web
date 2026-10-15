import { defineConfig, loadEnv } from 'vite';
import react from '@vitejs/plugin-react';
import { createHash } from 'node:crypto';
import { existsSync, readFileSync, statSync } from 'node:fs';

// Version string for the (unhashed) public/dfn files, appended as ?v= so they can be cached for a long time.
function dfnVersion() {
  const h = createHash('sha1');
  for (const f of ['df.js', 'df_bg.wasm', 'DeepFilterNet3_onnx.tar.gz']) {
    const p = new URL(`./public/dfn/${f}`, import.meta.url);
    if (existsSync(p)) h.update(readFileSync(p));
  }
  return h.digest('hex').slice(0, 10);
}

// Decoded byte sizes of the three files, for the progress readout while they download (src/media/denoise.js).
function dfnSizes() {
  const size = (f) => { const p = new URL(`./public/dfn/${f}`, import.meta.url); return existsSync(p) ? statSync(p).size : 0; };
  return { js: size('df.js'), wasm: size('df_bg.wasm'), model: size('DeepFilterNet3_onnx.tar.gz') };
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
    plugins: [react()],
    define: { __DFN_VERSION__: JSON.stringify(dfnVersion()), __DFN_SIZES__: JSON.stringify(dfnSizes()) },
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
