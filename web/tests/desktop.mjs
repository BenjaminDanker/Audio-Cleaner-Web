// The /desktop page: "Coming soon" vs download buttons (public/releases/desktop.json decides), OS highlighting,
// first-launch note (site.desktop.signed), landing/nav/footer links, privacy text, mobile layout, strict CSP.
// Self-contained: serves web/dist under /audio/ on an ephemeral port with the production CSP and a stubbed /api/me,
// so no backend or fixtures are needed. Usage: npm run build && node tests/desktop.mjs
// One extra in-memory build (signed: true, no public/ copy) checks that the first-launch note disappears.
import { createServer } from 'node:http';
import { existsSync, mkdtempSync, readFileSync, rmSync, statSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { dirname, extname, join, normalize, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';
import { build } from 'vite';
import { check, launch, summary } from './helpers.mjs';
import { detectMacChip, detectPlatform } from '../src/platform.js';
import { SITE } from '../src/site.js';

const WEB = resolve(dirname(fileURLToPath(import.meta.url)), '..');
const DIST = join(WEB, 'dist');
if (!existsSync(join(DIST, 'index.html'))) {
  console.error('web/dist is missing: run `npm run build` first');
  process.exit(2);
}

// Same policy as the production Caddy config (see tests/ui.mjs): any inline style/script would be a violation.
const CSP = "default-src 'none'; script-src 'self' 'wasm-unsafe-eval'; style-src 'self'; img-src 'self' data:; media-src 'self' blob:; worker-src 'self'; connect-src 'self'; font-src 'self'; manifest-src 'self'; base-uri 'none'; form-action 'self'; frame-ancestors 'none'";
const TYPES = { '.html': 'text/html', '.js': 'text/javascript', '.css': 'text/css', '.json': 'application/json', '.png': 'image/png', '.ico': 'image/x-icon', '.webmanifest': 'application/manifest+json', '.wasm': 'application/wasm', '.txt': 'text/plain' };

/** Static server for a build directory under /audio/ (SPA fallback like Caddy) with a signed-out /api/me stub. */
async function serve(dir) {
  const server = createServer((req, res) => {
    const url = new URL(req.url, 'http://x');
    if (url.pathname.startsWith('/audio/api/')) {
      if (url.pathname === '/audio/api/me') {
        res.writeHead(200, { 'content-type': 'application/json' });
        res.end(JSON.stringify({ signedIn: false, csrfToken: 'test' }));
      } else {
        res.writeHead(404, { 'content-type': 'application/json' });
        res.end('{"detail":"not found"}');
      }
      return;
    }
    const rel = normalize(decodeURIComponent(url.pathname.replace(/^\/audio\/?/, ''))).replace(/^(\.\.[/\\])+/, '');
    let file = join(dir, rel);
    if (!rel || !existsSync(file) || !statSync(file).isFile()) file = join(dir, 'index.html');
    res.writeHead(200, { 'content-type': TYPES[extname(file)] || 'application/octet-stream', 'content-security-policy': CSP, 'cache-control': 'no-cache' });
    res.end(readFileSync(file));
  });
  await new Promise((r) => server.listen(0, '127.0.0.1', r));
  return { server, origin: `http://127.0.0.1:${server.address().port}`, close: () => new Promise((r) => server.close(r)) };
}

const real = await serve(DIST);
const URL_ = `${real.origin}/audio/desktop`;
const D = SITE.desktop;
const RELEASE = { available: true, version: '9.8.7', url: 'https://example.invalid/audio/desktop', notes: 'Test release notes & <b>markup</b>.' };
const jsonRoute = (body, status = 200) => (r) => r.fulfill({ status, contentType: 'application/json', body: typeof body === 'string' ? body : JSON.stringify(body) });
const noiseOk = (l) => !/Failed to load resource/.test(l); // expected when a test makes the release file fail on purpose

const { browser } = await launch();

/** Fresh context + page, collecting console errors/warnings, page errors and CSP violations. */
async function scenario({ route, width = 1280, init, origin = real.origin } = {}, fn) {
  const context = await browser.newContext({ viewport: { width, height: 900 } });
  if (init) await context.addInitScript(init);
  const page = await context.newPage();
  const logs = [];
  page.on('console', (m) => { if (['error', 'warning'].includes(m.type())) logs.push(`[${m.type()}] ${m.text()}`); });
  page.on('pageerror', (e) => logs.push(`[pageerror] ${e.message}`));
  if (route !== undefined) await page.route('**/audio/releases/desktop.json', route);
  try {
    return await fn(page, logs, origin);
  } finally {
    await context.close();
  }
}

const open = async (page, origin = real.origin) => {
  await page.goto(`${origin}/audio/desktop`);
  await page.waitForSelector('main h1');
  await page.waitForFunction(() => !document.querySelector('main [aria-busy="true"]')); // release file has been read
};
const links = (page) => page.locator('main a[href*="AudioCleaner-"]');
const overflow = (page) => page.evaluate(() => ({ sw: document.documentElement.scrollWidth, iw: window.innerWidth }));
const fakeNav = (platform, ua, extra = '') => `Object.defineProperty(navigator,'platform',{get:()=>${JSON.stringify(platform)}});Object.defineProperty(navigator,'userAgent',{get:()=>${JSON.stringify(ua)}});${extra}`;
const UA = {
  win: 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/130.0 Safari/537.36',
  mac: 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/130.0 Safari/537.36',
  linux: 'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/130.0 Safari/537.36',
  iphone: 'Mozilla/5.0 (iPhone; CPU iPhone OS 17_0 like Mac OS X) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.0 Mobile/15E148 Safari/604.1',
};
const uaData = (platform, arch) => `Object.defineProperty(navigator,'userAgentData',{configurable:true,get:()=>({platform:${JSON.stringify(platform)},mobile:false,getHighEntropyValues:async()=>(${arch ? `{architecture:${JSON.stringify(arch)}}` : 'Promise.reject(new Error("blocked"))'})})});`;
const noUaData = "Object.defineProperty(navigator,'userAgentData',{configurable:true,get:()=>undefined});";

// ---- unit: platform detection ------------------------------------------------------------------------------------
await check('platform.js: detectPlatform / detectMacChip', async () => {
  const cases = [
    [{ platform: 'Win32', userAgent: UA.win }, 'windows'],
    [{ userAgentData: { platform: 'Windows', mobile: false }, platform: '', userAgent: '' }, 'windows'],
    [{ platform: 'MacIntel', userAgent: UA.mac, maxTouchPoints: 0 }, 'mac'],
    [{ userAgentData: { platform: 'macOS', mobile: false }, userAgent: '' }, 'mac'],
    [{ platform: 'MacIntel', userAgent: UA.mac, maxTouchPoints: 5 }, 'mobile'], // iPadOS pretends to be a Mac
    [{ platform: 'iPhone', userAgent: UA.iphone }, 'mobile'],
    [{ platform: 'Linux armv8l', userAgent: 'Mozilla/5.0 (Linux; Android 14; Pixel 8) Chrome/130.0 Mobile Safari/537.36' }, 'mobile'],
    [{ platform: 'Linux x86_64', userAgent: UA.linux }, 'other'],
    [null, 'other'],
  ];
  for (const [nav, want] of cases) {
    const got = detectPlatform(nav);
    if (got !== want) throw new Error(`${JSON.stringify(nav)} -> ${got}, want ${want}`);
  }
  const chip = async (arch) => detectMacChip({ userAgentData: { getHighEntropyValues: async () => ({ architecture: arch }) } });
  if ((await chip('arm')) !== 'arm' || (await chip('x86')) !== 'intel' || (await chip('')) !== null) throw new Error('chip mapping');
  if ((await detectMacChip({ userAgentData: { getHighEntropyValues: async () => { throw new Error('no'); } } })) !== null) throw new Error('rejected hint must be null');
  if ((await detectMacChip({})) !== null) throw new Error('no userAgentData must be null');
  return `${cases.length} platform cases`;
});

// ---- real build: release file says "not available" ----------------------------------------------------------------
await check('/desktop (built site): renders, "Coming soon", no download links, no console errors, strict CSP', () => scenario({}, async (page, logs) => {
  const reqs = [];
  page.on('request', (r) => { if (r.url().endsWith('releases/desktop.json')) reqs.push(r); });
  await open(page);
  const h1 = await page.locator('main h1').innerText();
  if (!/desktop|computer/i.test(h1)) throw new Error('h1: ' + h1);
  if (!/Desktop app/.test(await page.title())) throw new Error('title: ' + (await page.title()));
  await page.getByText('Coming soon', { exact: true }).first().waitFor();
  await page.getByTestId('desktop-coming-soon').waitFor();
  if (await links(page).count()) throw new Error('download links shown while unavailable');
  if (await page.getByTestId('desktop-downloads').count()) throw new Error('downloads grid shown while unavailable');
  if (await page.getByTestId('desktop-first-launch').count()) throw new Error('first-launch note shown while unavailable');
  const dead = await page.locator('main a:not([href])').count();
  if (dead) throw new Error(`${dead} anchors without href`);
  if (!reqs.length) throw new Error('release file was not requested');
  const cc = (await reqs[0].allHeaders())['cache-control'] || '';
  if (!/no-cache|max-age=0/.test(cc)) throw new Error('release file fetched without revalidation: cache-control=' + cc);
  await page.getByRole('heading', { name: 'System requirements' }).waitFor();
  await page.getByText('Windows 10 or Windows 11').waitFor();
  await page.getByText('macOS 11 (Big Sur) or newer').waitFor();
  await page.getByRole('link', { name: /open-source licences and third-party notices/ }).waitFor();
  if (logs.length) throw new Error(logs.join(' | '));
  return `${reqs.length} release request, cache-control: ${cc}`;
}));

await check('/desktop at 375px: no horizontal scroll (coming soon and with downloads)', () => scenario({ width: 375 }, async (page, logs) => {
  const bad = [];
  await open(page);
  let w = await overflow(page);
  if (w.sw > w.iw + 1) bad.push(`coming soon ${w.sw}>${w.iw}`);
  await page.route('**/audio/releases/desktop.json', jsonRoute(RELEASE));
  await open(page);
  await page.getByTestId('desktop-downloads').waitFor();
  w = await overflow(page);
  if (w.sw > w.iw + 1) {
    const culprit = await page.evaluate(() => [...document.querySelectorAll('body *')].filter((e) => e.getBoundingClientRect().right > window.innerWidth + 1).slice(0, 3).map((e) => `${e.tagName}.${e.className?.baseVal ?? e.className}`));
    bad.push(`downloads ${w.sw}>${w.iw} ${culprit.join(',')}`);
  }
  if (bad.length) throw new Error(bad.join(' | '));
  if (logs.length) throw new Error(logs.join(' | '));
  return 'ok';
}));

for (const [name, handler] of [
  ['HTTP 404', jsonRoute({ detail: 'nope' }, 404)],
  ['HTTP 500', jsonRoute('oops', 500)],
  ['network error', (r) => r.abort()],
  ['HTML instead of JSON (SPA fallback)', (r) => r.fulfill({ status: 200, contentType: 'text/html', body: '<!doctype html><title>x</title>' })],
  ['invalid JSON', jsonRoute('{not json')],
  ['JSON null', jsonRoute('null')],
  ['available: "true" (string)', jsonRoute({ available: 'true', version: '1.0.0' })],
  ['available: false', jsonRoute({ ...RELEASE, available: false })],
]) {
  await check(`/desktop: release file ${name} -> "Coming soon", no links`, () => scenario({ route: handler }, async (page, logs) => {
    await open(page);
    await page.getByTestId('desktop-coming-soon').waitFor();
    if (await links(page).count()) throw new Error('download links shown');
    if (await page.getByTestId('desktop-downloads').count()) throw new Error('downloads grid shown');
    const bad = logs.filter(noiseOk);
    if (bad.length) throw new Error(bad.join(' | '));
    return 'ok';
  }));
}

// ---- release file says "available" ---------------------------------------------------------------------------------
await check('/desktop: available -> three download links from site.js, version + notes (escaped), first-launch note, no "Coming soon"', () => scenario({ route: jsonRoute(RELEASE), init: fakeNav('Linux x86_64', UA.linux) + noUaData }, async (page, logs) => {
  await open(page);
  await page.getByTestId('desktop-downloads').waitFor();
  const want = { windows: D.downloads.windows, macArm: D.downloads.macArm, macIntel: D.downloads.macIntel };
  for (const [key, href] of Object.entries(want)) {
    const a = page.getByTestId(`dl-${key}`).getByRole('link');
    if ((await a.getAttribute('href')) !== href) throw new Error(`${key}: ${await a.getAttribute('href')}`);
    if (!/noopener/.test((await a.getAttribute('rel')) || '')) throw new Error(`${key}: missing rel=noopener`);
  }
  if (await page.getByText('Coming soon').count()) throw new Error('"Coming soon" shown while available');
  const text = await page.locator('main').innerText();
  if (!text.includes('9.8.7')) throw new Error('version missing');
  if (!text.includes('Test release notes & <b>markup</b>.')) throw new Error('notes missing or interpreted as HTML');
  if (await page.locator('main b', { hasText: 'markup' }).count()) throw new Error('notes rendered as HTML');
  const rn = await page.getByRole('link', { name: 'Release notes' }).getAttribute('href');
  if (rn !== D.releaseNotesUrl) throw new Error('release notes link: ' + rn);
  const note = page.getByTestId('desktop-first-launch');
  await note.waitFor();
  const t = await note.innerText();
  for (const s of ['More info', 'Run anyway', 'right-click', 'Open']) if (!t.includes(s)) throw new Error(`first-launch note lacks "${s}"`);
  if (D.signed) throw new Error('site.desktop.signed is true; this check assumes the default (false)');
  if (logs.length) throw new Error(logs.join(' | '));
  return 'ok';
}));

// The release file fetch is relative to the app base: BASE_URL + releases/desktop.json.
await check('/desktop: release file is requested at <base>releases/desktop.json', () => scenario({}, async (page) => {
  const seen = [];
  page.on('request', (r) => { if (/desktop\.json/.test(r.url())) seen.push(new URL(r.url()).pathname); });
  await open(page);
  if (seen.join() !== '/audio/releases/desktop.json') throw new Error(seen.join() || 'none');
  return seen[0];
}));

const dmgArm = D.downloads.macArm;
const dmgIntel = D.downloads.macIntel;
const heroLink = (page) => page.locator('.hero a.btn-primary');
for (const [name, init, expect] of [
  ['Windows', fakeNav('Win32', UA.win) + noUaData, { rec: ['windows'], hero: D.downloads.windows }],
  ['macOS, Apple silicon (client hints)', fakeNav('MacIntel', UA.mac) + uaData('macOS', 'arm'), { rec: ['macArm'], hero: dmgArm }],
  ['macOS, Intel (client hints)', fakeNav('MacIntel', UA.mac) + uaData('macOS', 'x86'), { rec: ['macIntel'], hero: dmgIntel }],
  ['macOS, chip unknown (Safari/Firefox)', fakeNav('MacIntel', UA.mac) + noUaData, { rec: ['macArm', 'macIntel'], hero: '#download' }],
  ['Linux', fakeNav('Linux x86_64', UA.linux) + noUaData, { rec: [], hero: '#download' }],
  ['iPhone', fakeNav('iPhone', UA.iphone) + noUaData, { rec: [], hero: '#download', note: /Windows and macOS computers/ }],
]) {
  await check(`/desktop: OS detection highlights the right download (${name})`, () => scenario({ route: jsonRoute(RELEASE), init }, async (page) => {
    await open(page);
    await page.getByTestId('desktop-downloads').waitFor();
    await page.waitForTimeout(150); // the Apple-silicon hint resolves asynchronously
    const got = [];
    for (const k of ['windows', 'macArm', 'macIntel']) if ((await page.getByTestId(`dl-${k}`).getAttribute('data-recommended')) === 'true') got.push(k);
    if (got.join() !== expect.rec.join()) throw new Error(`recommended ${got.join() || 'none'}, want ${expect.rec.join() || 'none'}`);
    if ((await page.getByTestId('desktop-downloads').getByRole('link').count()) !== 3) throw new Error('all three downloads must stay visible');
    const hero = await heroLink(page).getAttribute('href');
    if (hero !== expect.hero) throw new Error(`hero href ${hero}, want ${expect.hero}`);
    if (expect.note && !expect.note.test(await page.locator('.hero').innerText())) throw new Error('hero note missing');
    return `recommended: ${got.join(',') || 'none'}; hero -> ${hero.slice(0, 40)}`;
  }));
}

// ---- same site with signed: true -------------------------------------------------------------------------------------
{
  const out = mkdtempSync(join(tmpdir(), 'acw-desktop-signed-'));
  try {
    await build({
      configFile: join(WEB, 'vite.config.js'),
      root: WEB,
      logLevel: 'silent',
      publicDir: false,
      build: { outDir: out, emptyOutDir: true },
      plugins: [{
        name: 'test-signed-site',
        transform(code, id) {
          if (!id.endsWith('/src/site.js')) return null;
          if (!code.includes('signed: false')) throw new Error('site.js no longer contains `signed: false`');
          return code.replace('signed: false', 'signed: true');
        },
      }],
    });
    const signed = await serve(out);
    try {
      await check('/desktop: site.desktop.signed = true -> no first-launch note, downloads still shown', () => scenario({ route: jsonRoute(RELEASE), origin: signed.origin }, async (page, logs) => {
        await open(page, signed.origin);
        await page.getByTestId('desktop-downloads').waitFor();
        if (await page.getByTestId('desktop-first-launch').count()) throw new Error('first-launch note shown for signed builds');
        if (await page.getByText(/More info|Run anyway/).count()) throw new Error('SmartScreen text shown for signed builds');
        if (logs.length) throw new Error(logs.join(' | '));
        return 'ok';
      }));
    } finally {
      await signed.close();
    }
  } catch (e) {
    await check('/desktop: signed build', async () => { throw e; });
  } finally {
    rmSync(out, { recursive: true, force: true });
  }
}

// ---- navigation, landing, footer, privacy ------------------------------------------------------------------------------
await check('landing: desktop card links to /desktop; footer + header nav link; client-side navigation works', () => scenario({}, async (page, logs) => {
  await page.goto(`${real.origin}/audio/`);
  await page.waitForSelector('main h1');
  const card = page.locator('#desktop');
  await card.waitFor();
  if (!(await card.innerText()).match(/Windows and macOS/)) throw new Error('promo text');
  const nav = page.locator('#main-nav').getByRole('link', { name: 'Desktop app' });
  if ((await nav.getAttribute('href')) !== '/audio/desktop') throw new Error('nav href ' + (await nav.getAttribute('href')));
  const foot = page.locator('footer').getByRole('link', { name: 'Desktop app' });
  if ((await foot.getAttribute('href')) !== '/audio/desktop') throw new Error('footer href');
  await card.getByRole('link', { name: /desktop app/i }).click();
  await page.waitForURL(/\/audio\/desktop$/);
  await page.waitForSelector('main h1');
  if ((await nav.getAttribute('aria-current')) !== 'page') throw new Error('nav link not marked current');
  if (logs.filter(noiseOk).length) throw new Error(logs.join(' | '));
  return 'ok';
}));

await check('header: one row with the "Desktop app" link at 861px (signed in, long name) and at 375px (menu toggle)', async () => {
  const me = { signedIn: true, csrfToken: 't', user: { email: 'someone.with.a.long.address@example.com', displayName: 'Someone Long-Named', tosAccepted: true }, balance: { available_micros: 123456789, reserved_micros: 0 } };
  const rowCheck = (width) => scenario({ width }, async (page) => {
    await page.route('**/audio/api/me', jsonRoute(me));
    await page.goto(`${real.origin}/audio/desktop`);
    await page.waitForSelector('.header-right .balance-chip');
    return page.evaluate(() => {
      const box = (s) => document.querySelector(s)?.getBoundingClientRect();
      return { sw: document.documentElement.scrollWidth, iw: window.innerWidth, headerH: box('.header-inner').height, navRight: box('#main-nav')?.right, rightLeft: box('.header-right').left, navLines: [...document.querySelectorAll('#main-nav a')].map((a) => Math.round(a.getBoundingClientRect().height)) };
    });
  });
  const out = [];
  for (const width of [861, 900, 1000, 1280, 375]) {
    const r = await rowCheck(width);
    if (r.sw > r.iw + 1) out.push(`${width}px: horizontal overflow ${r.sw}>${r.iw}`);
    if (r.headerH > 64 || Math.max(...r.navLines, 0) > 45) out.push(`${width}px: header wrapped (height ${Math.round(r.headerH)}, nav link heights ${r.navLines.join(',')})`);
    if (width > 860 && r.navRight > r.rightLeft + 1) out.push(`${width}px: nav overlaps account controls (${Math.round(r.navRight)} > ${Math.round(r.rightLeft)})`);
  }
  if (out.length) throw new Error(out.join(' | '));
  return 'ok at 861/900/1000/1280/375';
});

await check('mobile menu at 375px lists "Desktop app"', () => scenario({ width: 375 }, async (page) => {
  await page.goto(`${real.origin}/audio/`);
  await page.getByRole('button', { name: 'Menu' }).click();
  await page.locator('#main-nav').getByRole('link', { name: 'Desktop app' }).waitFor({ state: 'visible' });
  return 'ok';
}));

await check('privacy policy has a "The desktop app" section (and the table of contents links to it)', () => scenario({}, async (page) => {
  await page.goto(`${real.origin}/audio/privacy`);
  await page.waitForSelector('main h1');
  await page.getByRole('heading', { name: /The desktop app/ }).waitFor();
  const t = await page.locator('#desktop').evaluate((h) => { let s = h.textContent; for (let n = h.nextElementSibling; n && n.tagName !== 'H2'; n = n.nextElementSibling) s += ' ' + n.textContent; return s; });
  for (const s of ['never uploaded', 'local log', 'once a day', 'IP address', 'no telemetry']) if (!t.toLowerCase().includes(s.toLowerCase())) throw new Error(`privacy text lacks "${s}"`);
  await page.locator('nav.toc a[href="#desktop"]').waitFor();
  return 'ok';
}));

await browser.close();
await real.close();
const results = summary();
console.log(`${results.filter((r) => r.ok).length}/${results.length} passed`);
process.exit(results.some((r) => !r.ok) ? 1 : 0);
