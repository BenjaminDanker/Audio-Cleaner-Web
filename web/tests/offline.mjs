// The backend is down: the "taking a short break" banner, the calm panel instead of the paid tools, and both going away by
// themselves once /api/health answers again (the page's own 30 s re-check, run on a fake clock so nothing waits).
// Starts its own tiny static server (port 8797, override with OFFLINE_PORT) that serves web/dist at the root and forwards
// /api/* to a backend port it can switch at runtime: a dead port (what the proxy turns into HTTP 502), the front door's
// maintenance answer, or the real test backend on 8792 for the recovery.
// Usage: tests/run.sh offline   (builds the site with an empty base path and starts the backend on 8792), or by hand with
// DIST=<site built for the root, not web/dist's /audio/ base> and a running `tests/backend.sh start`: node tests/offline.mjs
import { createServer, request as httpRequest } from 'node:http';
import { existsSync, readFileSync, statSync } from 'node:fs';
import { dirname, extname, join, normalize, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';
import { SHOTS, check, launch, summary } from './helpers.mjs';

const DIST = process.env.DIST || resolve(dirname(fileURLToPath(import.meta.url)), '..', 'dist');
const PORT = Number(process.env.OFFLINE_PORT || 8797);
const REAL_BACKEND = Number(process.env.E2E_BACKEND_PORT || 8792);
const DEAD_PORT = 9; // nothing listens here
const OFF = `http://127.0.0.1:${PORT}`;
const TYPES = { '.html': 'text/html', '.js': 'text/javascript', '.css': 'text/css', '.wasm': 'application/wasm', '.json': 'application/json', '.png': 'image/png', '.ico': 'image/x-icon', '.webmanifest': 'application/manifest+json', '.txt': 'text/plain' };

let mode = 'dead-502'; // dead-502 | maint | real
const apiHits = [];

const server = createServer((req, res) => {
  const url = new URL(req.url, OFF);
  if (url.pathname.startsWith('/api/')) {
    apiHits.push(`${mode} ${req.method} ${url.pathname}`);
    if (mode === 'maint') { res.writeHead(503, { 'content-type': 'application/json' }); res.end(JSON.stringify({ detail: 'maintenance', message: 'Audio Cleaner is taking a short break.' })); return; }
    const port = mode === 'real' ? REAL_BACKEND : DEAD_PORT;
    const up = httpRequest({ host: '127.0.0.1', port, path: req.url, method: req.method, headers: req.headers }, (r) => { res.writeHead(r.statusCode, r.headers); r.pipe(res); });
    up.on('error', () => { res.writeHead(502, { 'content-type': 'text/plain' }); res.end('Bad Gateway'); }); // what a reverse proxy answers when the backend is stopped
    req.pipe(up);
    return;
  }
  const rel = normalize(decodeURIComponent(url.pathname.replace(/^\//, ''))).replace(/^(\.\.[/\\])+/, '');
  let file = join(DIST, rel);
  if (!rel || !existsSync(file) || !statSync(file).isFile()) file = join(DIST, 'index.html');
  res.writeHead(200, { 'content-type': TYPES[extname(file)] || 'application/octet-stream' });
  res.end(readFileSync(file));
});
await new Promise((r) => server.listen(PORT, '127.0.0.1', r));

const TEXT = 'Audio Cleaner is taking a short break. Signing in and everything that needs an account will be back shortly.';
const { browser, logs } = await launch();

await check('offline: the banner says the server is on a short break (the backend does not answer), and the Cleanup page shows the calm panel instead of the tool', async () => {
  mode = 'dead-502';
  const ctx = await browser.newContext();
  const page = await ctx.newPage();
  try {
    await page.goto(`${OFF}/`);
    const banner = page.getByTestId('api-offline');
    await banner.waitFor({ timeout: 10000 });
    const text = (await banner.innerText()).replace(/\s*Dismiss\s*$/, '').trim();
    if (text !== TEXT) throw new Error(text);
    await page.goto(`${OFF}/app`);
    await page.getByTestId('maintenance').waitFor({ timeout: 10000 });
    if (await page.locator('input[type=file]').count()) throw new Error('/app still offers a file picker while the server is down');
    await page.screenshot({ path: join(SHOTS, 'offline-banner.png') });
    return 'banner and panel shown';
  } finally {
    await ctx.close();
  }
});

await check('maintenance (the front door answers 503): Cleanup and Billing show "We\'ll be right back" with no forms and no claim that anything still works; the panel has a Check now button', async () => {
  mode = 'maint';
  const ctx = await browser.newContext();
  const page = await ctx.newPage();
  try {
    for (const p of ['/app', '/app/billing']) {
      await page.goto(`${OFF}${p}`);
      const panel = page.getByTestId('maintenance');
      await panel.waitFor({ timeout: 10000 });
      const text = await panel.innerText();
      if (!/We.ll be right back/.test(text) || /still work/i.test(text)) throw new Error(`${p}: ${text}`);
      if (await page.locator('main input, main select, main form').count()) throw new Error(`${p}: a form is still on the page`);
      if (!(await panel.getByRole('button', { name: 'Check now' }).count())) throw new Error(`${p}: no Check now button`);
    }
    return 'panel on Cleanup and Billing';
  } finally {
    await ctx.close();
  }
});

await check('recovery: the banner and the panel go away by themselves once /api/health answers again (the page re-checks every 30 s; the clock is run forward, not waited for)', async () => {
  const alive = await fetch(`http://127.0.0.1:${REAL_BACKEND}/api/health`).then((r) => r.ok, () => false);
  if (!alive) throw new Error(`precondition: the test backend is not answering on port ${REAL_BACKEND} (start it with tests/backend.sh start, or run this through tests/run.sh); without it the page can never recover`);
  const ctx = await browser.newContext();
  const page = await ctx.newPage();
  try {
    await page.clock.install();
    mode = 'dead-502';
    await page.goto(`${OFF}/`);
    await page.getByTestId('api-offline').waitFor({ timeout: 10000 });
    mode = 'real';
    await page.clock.runFor(31000); // the next /api/health check
    await page.getByTestId('api-offline').waitFor({ state: 'detached', timeout: 15000 });
    if (!apiHits.some((h) => h.startsWith('real') && h.endsWith('/api/health'))) throw new Error('recovery did not use /api/health: ' + apiHits.slice(-4).join(' ; '));
    return 'recovered through /api/health';
  } finally {
    await ctx.close();
  }
});

console.log(`API hits seen by the stub: ${apiHits.length}`);
console.log('LOGS:\n' + logs.filter((l) => !/Failed to load resource|ERR_|502|503|504/.test(l)).join('\n'));
await browser.close();
server.close();
process.exit(summary().some((r) => !r.ok) ? 1 : 0);
