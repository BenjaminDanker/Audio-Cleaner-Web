// "API offline" banner + the free tool with the paid backend down.
// Starts its own tiny static server (port 8794, override with OFFLINE_PORT) that serves web/dist under /audio/ and
// forwards /audio/api/* to a backend port it can switch at runtime: a dead port (what Caddy turns into HTTP 502, or a
// reset connection), a fixed 502/503/504, or the real test backend on 8792 (tests/e2e.mjs setup) for the recovery check.
// Usage: node tests/offline.mjs   (needs `npm run build`, fixtures, and the 8792 backend only for the recovery check)
import { createServer, request as httpRequest } from 'node:http';
import { existsSync, readFileSync, statSync } from 'node:fs';
import { dirname, extname, join, normalize, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';
import { FILES, SHOTS, check, launch, summary } from './helpers.mjs';

const DIST = resolve(dirname(fileURLToPath(import.meta.url)), '..', 'dist');
const PORT = Number(process.env.OFFLINE_PORT || 8794);
const REAL_BACKEND = Number(process.env.E2E_BACKEND_PORT || 8792);
const DEAD_PORT = 9; // nothing listens here
const OFF = `http://127.0.0.1:${PORT}/audio`;
const TYPES = { '.html': 'text/html', '.js': 'text/javascript', '.css': 'text/css', '.wasm': 'application/wasm', '.json': 'application/json', '.png': 'image/png', '.ico': 'image/x-icon', '.webmanifest': 'application/manifest+json', '.txt': 'text/plain' };

let mode = 'dead-502'; // dead-502 | dead-reset | s502 | s503 | s504 | maint (the real maintenance answer of the front door) | real
const apiHits = [];

const server = createServer((req, res) => {
  const url = new URL(req.url, OFF);
  if (url.pathname.startsWith('/audio/api/')) {
    apiHits.push(`${mode} ${req.method} ${url.pathname}`);
    if (mode === 'dead-reset') { req.socket.destroy(); return; }
    if (mode === 'maint') { res.writeHead(503, { 'content-type': 'application/json' }); res.end(JSON.stringify({ detail: 'maintenance', message: 'Audio Cleaner is taking a short break. Free tools in your browser still work.' })); return; }
    if (mode === 's502' || mode === 's503' || mode === 's504') { res.writeHead(Number(mode.slice(1)), { 'content-type': 'application/json' }); res.end('{"detail":"upstream"}'); return; }
    const port = mode === 'real' ? REAL_BACKEND : DEAD_PORT;
    const up = httpRequest({ host: '127.0.0.1', port, path: req.url, method: req.method, headers: req.headers }, (r) => { res.writeHead(r.statusCode, r.headers); r.pipe(res); });
    up.on('error', () => { res.writeHead(502, { 'content-type': 'text/plain' }); res.end('Bad Gateway'); }); // what a reverse proxy answers when the backend is stopped
    req.pipe(up);
    return;
  }
  let rel = normalize(decodeURIComponent(url.pathname.replace(/^\/audio\/?/, ''))).replace(/^(\.\.[/\\])+/, '');
  let file = join(DIST, rel);
  if (!rel || !existsSync(file) || !statSync(file).isFile()) file = join(DIST, 'index.html');
  res.writeHead(200, { 'content-type': TYPES[extname(file)] || 'application/octet-stream' });
  res.end(readFileSync(file));
});
await new Promise((r) => server.listen(PORT, '127.0.0.1', r));

const TEXT = 'Audio Cleaner is taking a short break. Free tools in your browser still work.';
const { browser, page, logs } = await launch();
const errors = [];
page.on('pageerror', (e) => errors.push(e.message));

for (const m of ['dead-502', 'dead-reset', 's502', 's503', 's504', 'maint']) {
  await check(`offline banner appears when the backend answers ${m}`, async () => {
    mode = m;
    await page.goto(`${OFF}/`);
    const banner = page.getByTestId('api-offline');
    await banner.waitFor({ timeout: 10000 });
    const text = (await banner.innerText()).replace(/\s*Dismiss\s*$/, '').trim();
    if (text !== TEXT) throw new Error(text);
    if ((await banner.getAttribute('role')) !== 'status') throw new Error('role');
    return `shown (${m})`;
  });
}

await check('offline banner is dismissible, stays dismissed while navigating, and is small', async () => {
  mode = 'dead-reset';
  await page.goto(`${OFF}/`);
  const banner = page.getByTestId('api-offline');
  await banner.waitFor();
  const h = (await banner.boundingBox()).height;
  if (h > 60) throw new Error(`banner is ${h}px tall`);
  await page.screenshot({ path: join(SHOTS, 'offline-banner.png') });
  await page.getByRole('button', { name: 'Dismiss' }).click();
  await banner.waitFor({ state: 'detached' });
  await page.getByRole('link', { name: 'Free denoise' }).first().click();
  await page.waitForURL(/\/audio\/free$/);
  await page.waitForTimeout(800);
  if (await page.getByTestId('api-offline').count()) throw new Error('banner came back after navigation');
  return `${Math.round(h)}px tall`;
});

await check('free tool works fully with the API down (denoise + download), banner visible', async () => {
  mode = 'dead-reset';
  await page.goto(`${OFF}/free`);
  await page.getByTestId('api-offline').waitFor();
  await page.setInputFiles('input[type=file]', join(FILES, 'noisy.wav'));
  await page.getByRole('button', { name: 'Clean audio' }).click();
  await page.waitForSelector('[data-testid=result]', { timeout: 120000 });
  const res = page.locator('[data-testid=result]');
  const rms = Number(await res.getAttribute('data-rms'));
  const frames = Number(await res.getAttribute('data-frames'));
  if (rms < 0.001) throw new Error('silent result');
  if (Math.abs(frames - 508591) > 2) throw new Error('frames ' + frames);
  const [dl] = await Promise.all([page.waitForEvent('download'), page.getByRole('link', { name: /Cleaned WAV/ }).click()]);
  if (!/\.wav$/i.test(dl.suggestedFilename())) throw new Error(dl.suggestedFilename());
  await page.screenshot({ path: join(SHOTS, 'offline-free-done.png'), fullPage: true });
  return `rms ${rms.toFixed(4)}, ${frames} frames, ${dl.suggestedFilename()}`;
});

await check('no banner while the API is up; paid pages degrade without crashing when it is down', async () => {
  mode = 'real';
  await page.goto(`${OFF}/`);
  await page.waitForSelector('table tbody tr');
  await page.waitForTimeout(500);
  if (await page.getByTestId('api-offline').count()) throw new Error('banner shown although the API answers');
  mode = 'dead-502';
  for (const p of ['/', '/transcribe', '/login', '/app', '/app/billing', '/app/live', '/terms']) {
    await page.goto(`${OFF}${p}`);
    await page.waitForSelector('main h1, .loading-screen', { timeout: 10000 });
    await page.waitForTimeout(250);
  }
  if (errors.length) throw new Error(errors.join(' | '));
  return 'ok';
});

await check('banner disappears by itself once the backend is back (re-checks /api/health)', async () => {
  mode = 'dead-502';
  await page.goto(`${OFF}/`);
  await page.getByTestId('api-offline').waitFor();
  mode = 'real';
  await page.getByTestId('api-offline').waitFor({ state: 'detached', timeout: 45000 });
  return 'recovered';
});

await check('maintenance: Studio, Transcribe, Live, Billing and Account show a calm "We\'ll be right back" panel instead of forms; free tools stay', async () => {
  mode = 'maint';
  for (const p of ['/app', '/transcribe', '/app/live', '/app/billing', '/app/account']) {
    await page.goto(`${OFF}${p}`);
    const panel = page.getByTestId('maintenance');
    await panel.waitFor({ timeout: 10000 });
    const text = await panel.innerText();
    if (!/We.ll be right back/.test(text) || !/free tools still work/i.test(text)) throw new Error(`${p}: ${text}`);
    if (await page.locator('main input, main select, main form').count()) throw new Error(`${p}: a form is still on the page`);
    if (page.url().includes('/login')) throw new Error(`${p}: bounced to the sign-in page`);
    if (!(await panel.getByRole('link', { name: /browser/ }).getAttribute('href'))?.endsWith('/free')) throw new Error(`${p}: no link to the free tool`);
  }
  await page.screenshot({ path: join(SHOTS, 'maintenance.png') });
  await page.goto(`${OFF}/free`);
  if (await page.getByTestId('maintenance').count()) throw new Error('/free must not show the panel');
  if (!(await page.locator('input[type=file]').count())) throw new Error('/free lost its file picker');
  return 'panel on 5 pages, none on /free';
});

await check('maintenance: the panel goes away by itself when /api/health answers again', async () => {
  mode = 'maint';
  await page.goto(`${OFF}/transcribe`);
  await page.getByTestId('maintenance').waitFor();
  mode = 'real';
  await page.getByTestId('maintenance').waitFor({ state: 'detached', timeout: 50000 });
  await page.locator('main h1').first().waitFor();
  if (await page.getByTestId('api-offline').count()) throw new Error('banner still shown');
  if (!apiHits.some((h) => h.startsWith('real') && h.endsWith('/audio/api/health'))) throw new Error('recovery did not use /api/health: ' + apiHits.slice(-5).join(' ; '));
  return 'recovered through /api/health';
});

console.log(`API hits seen by the stub: ${apiHits.length}`);
console.log('LOGS:\n' + logs.filter((l) => !/Failed to load resource|ERR_|502|503|504/.test(l)).join('\n'));
await browser.close();
server.close();
process.exit(summary().some((r) => !r.ok) ? 1 : 0);
