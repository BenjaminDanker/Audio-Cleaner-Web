// "Automatic best settings" on the Cleanup page and the paid "Test my room" on the Live page, in a real browser. The numbers
// behind the advice are checked in tests/reco-unit.mjs; here the page must show the same choices for the same audio.
// Needs no backend: the API is answered by this script (page.route), and the audio is generated here. Point it at any server that
// serves the built site at the root (the test backend on 8792 by default):
//   E2E_ORIGIN=http://127.0.0.1:8792 node tests/reco-ui.mjs
import { mkdtempSync, writeFileSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import { chromium } from 'playwright';
import { noisy, voice, wavBytes } from './reco-audio.mjs';
import { startMirror } from './mirror.mjs';

await startMirror(); // the built site downloads the noise remover from the tests' local model host

const ORIGIN = process.env.E2E_ORIGIN || 'http://127.0.0.1:8792';
const BASE = ORIGIN;
const dir = mkdtempSync(join(tmpdir(), 'reco-ui-'));

const rate = (product, tier, h) => ({ product, tier, usdPerHour: h, usdPerMinute: h / 60, available: true });
const PRICING = {
  maxJobSeconds: 36000, maxUploadBytes: 4 * 1024 ** 3, rebuildAvailable: true,
  rates: [
    rate('denoise', 'standard', 1.2), rate('denoise', 'rebuild', 3.0), rate('transcribe', 'local', 0.6), rate('transcribe', 'premium', 2.4),
    rate('transcribe_translate', 'standard', 0.6), rate('stream_denoise', 'light', 1.2), rate('stream_captions', 'local', 0.6),
    rate('stream_captions', 'premium', 2.4), rate('stream_translate', 'standard', 0.6),
  ],
};
const ME = { signedIn: true, user: { email: 'reco@example.com' }, balance: { available_micros: 9000000 }, csrfToken: 'x', trialLiveSecondsLeft: 0, contactForm: false };

const results = [];
async function check(name, fn) {
  try {
    const detail = await fn();
    results.push(true);
    console.log(`PASS ${name}${detail ? ` -- ${detail}` : ''}`);
  } catch (e) {
    results.push(false);
    console.log(`FAIL ${name} -- ${String(e.message).split('\n')[0]}`);
  }
}

function file(name, samples, rateHz = 44100) {
  const p = join(dir, name);
  writeFileSync(p, wavBytes(samples, rateHz));
  return p;
}
// The voice files are 48 kHz: the generated voice stops at 8 kHz, and labelled 44.1 kHz it would sound like a phone call.
const FILES = {
  clean: file('clean.wav', noisy(10, 60, 1), 48000),
  mild: file('mild.wav', noisy(10, 22, 2, 'pink'), 48000),
  loud: file('noisy.wav', noisy(10, 2, 3, 'pink'), 48000),
  hum: file('hum.wav', Float32Array.from({ length: 12 * 44100 }, (_, i) => 0.05 * Math.sin((2 * Math.PI * 50 * i) / 44100))),
  long: file('long.wav', noisy(80, 40, 4), 48000), // over the 75 s that are measured as six stretches
};
const MIC_NOISY = file('mic-noisy.wav', noisy(30, 2, 6, 'pink'), 48000);

const browser = await chromium.launch({ headless: true, args: ['--autoplay-policy=no-user-gesture-required'] });

async function newPage(extraArgs = {}) {
  const ctx = await browser.newContext({ viewport: { width: 1200, height: 1100 }, permissions: ['microphone'], ...extraArgs });
  const page = await ctx.newPage();
  const problems = [];
  page.on('pageerror', (e) => problems.push(`pageerror: ${e.message}`));
  await page.route('**/api/**', (route) => {
    const p = new URL(route.request().url()).pathname.replace('/api/', '');
    if (p === 'me') return route.fulfill({ json: ME });
    if (p === 'billing/pricing') return route.fulfill({ json: PRICING });
    if (p === 'health') return route.fulfill({ json: { ok: true } });
    return route.fulfill({ status: 404, json: { detail: 'not mocked' } });
  });
  return { ctx, page, problems };
}

const read = (page) => page.evaluate(() => ({
  noise: document.querySelector('input[name=noise]:checked')?.value,
  subs: document.querySelector('input[name=subs]:checked')?.value,
  strength: document.querySelector('input[name=strength]:checked')?.value,
  recommended: document.querySelectorAll('.reco-badge, [data-testid^=reco-noise], [data-testid^=reco-subs]').length > 0 || /Recommended/.test(document.querySelector('form')?.innerText || ''),
  leveling: !!document.querySelector('input[name=leveling]:checked, input[name=leveling-alone]:checked'),
  loudness: document.querySelector('#loud, #loud-alone')?.value,
  rebuild: !!document.querySelector('input[name=rebuild]:checked'),
  auto: document.querySelector('input[name=auto]')?.checked,
  status: document.querySelector('[data-testid=reco-status]')?.textContent || '',
  line: document.querySelector('[data-testid=reco-line]')?.textContent || '',
  why: [...document.querySelectorAll('[data-testid=reco-why] li')].map((l) => l.textContent),
}));

async function open(page, path) {
  await page.goto(`${BASE}/app`);
  await page.waitForSelector('input[type=file]', { timeout: 20000 });
  await page.setInputFiles('input[type=file]', path);
  await page.getByTestId('start').waitFor({ timeout: 20000 });
  await page.getByTestId('reco-line').waitFor({ timeout: 30000 });
  return read(page);
}

{
  const { ctx, page, problems } = await newPage();

  await check('clean recording: Denoising at Gentle is set, with the reasons in plain words; Standard subtitles and Leveling are on, the loudness target is YouTube; nothing is marked "Recommended"', async () => {
    const t0 = Date.now();
    const s = await open(page, FILES.clean);
    if (s.noise !== 'denoise' || s.strength !== 'gentle') throw new Error(`noise ${s.noise}/${s.strength}: ${s.line}`);
    if (s.recommended) throw new Error('something is still marked Recommended');
    if (!/clean and quiet/.test(s.why.join(' '))) throw new Error(s.why.join(' | '));
    if (s.subs !== 'standard') throw new Error(`subtitles ${s.subs}`);
    if (!s.leveling || s.rebuild) throw new Error(`leveling ${s.leveling} rebuild ${s.rebuild}`);
    if (s.loudness !== '-14') throw new Error(`loudness ${s.loudness}`);
    if (!/Set automatically for this file/.test(s.status)) throw new Error(`status: ${s.status}`);
    return `${Date.now() - t0} ms: ${s.line}`;
  });

  await check('steady noise sets Denoising with a strength below Maximum; a lot of noise sets Denoising at Maximum with Extra accurate subtitles and Leveling; a recording with no one speaking sets Denoising, Leveling and no subtitles', async () => {
    const mild = await open(page, FILES.mild);
    if (mild.noise !== 'denoise' || !mild.strength || mild.strength === 'max' || !mild.leveling) throw new Error(`mild: ${JSON.stringify(mild)}`);
    const loud = await open(page, FILES.loud);
    if (loud.noise !== 'denoise' || !/stronger noise remover/.test(loud.why.join(' '))) throw new Error(`loud: ${JSON.stringify(loud)}`);
    if (loud.subs !== 'accurate' || !loud.leveling || loud.strength !== 'max') throw new Error(`loud: ${JSON.stringify(loud)}`);
    const hum = await open(page, FILES.hum);
    if (hum.noise !== 'denoise' || hum.strength !== 'balanced' || !/no subtitles/.test(hum.line) || hum.subs !== 'off' || !hum.leveling) throw new Error(JSON.stringify(hum));
    return `mild -> ${mild.noise}/${mild.strength}; loud -> ${loud.noise}/${loud.strength}/${loud.subs}; hum -> ${hum.line}`;
  });

  await check('Leveling is a plain checkbox: picking Off or Denoising leaves it as it is', async () => {
    await open(page, FILES.clean);
    for (const v of ['off', 'denoise']) {
      await page.locator(`label.choice:has(input[name=noise][value=${v}])`).click();
      if (!(await read(page)).leveling) throw new Error(`Leveling switched off by ${v}`);
    }
    await page.locator('input[name=leveling]').uncheck();
    for (const v of ['off', 'denoise']) {
      await page.locator(`label.choice:has(input[name=noise][value=${v}])`).click();
      if ((await read(page)).leveling) throw new Error(`Leveling switched on by ${v}`);
    }
  });

  await check('Undo puts the choices back to the usual start and switches the automatic setting off (remembered for the next file)', async () => {
    const before = await open(page, FILES.loud);
    if (before.subs !== 'accurate') throw new Error(`not set: ${JSON.stringify(before)}`);
    await page.getByTestId('reco-undo').click();
    const s = await read(page);
    if (s.noise !== 'denoise' || s.subs !== 'standard' || !s.leveling || s.rebuild || s.auto) throw new Error(`after undo: ${JSON.stringify(s)}`);
    const next = await page.evaluate(() => localStorage.getItem('acw.autoSettings'));
    if (next !== 'off') throw new Error(`stored: ${next}`);
    await page.evaluate(() => localStorage.removeItem('acw.autoSettings'));
  });

  await check('a long file (over a minute and a quarter) is looked at in a few seconds, on a sample of it, and the recommendation is in', async () => {
    const t0 = Date.now();
    const s = await open(page, FILES.long);
    const ms = Date.now() - t0;
    if (!s.line || ms > 20000) throw new Error(`${ms} ms: ${s.line}`);
    return `${ms} ms: ${s.line}`;
  });

  await check('a choice you made yourself stays as you set it when the listen finishes later; the rest is set', async () => {
    const slow = await newPage();
    await slow.page.route('http://127.0.0.1:8798/**', async (route) => { await new Promise((r) => setTimeout(r, 4000)); await route.continue(); });
    await slow.page.goto(`${BASE}/app`);
    await slow.page.waitForSelector('input[type=file]', { timeout: 20000 });
    await slow.page.setInputFiles('input[type=file]', FILES.loud);
    await slow.page.getByTestId('reco-listening').waitFor({ timeout: 20000 });
    await slow.page.locator('label.choice:has(input[name=subs][value=off])').click();
    await slow.page.getByTestId('reco-line').waitFor({ timeout: 60000 });
    const s = await read(slow.page);
    await slow.ctx.close();
    if (s.subs !== 'off') throw new Error(`the person's subtitle choice was changed: ${s.subs}`);
    if (s.noise !== 'denoise' || !s.leveling) throw new Error(`the rest was not set: ${JSON.stringify(s)}`);
    if (!/changed by you/.test(s.status)) throw new Error(`status: ${s.status}`);
  });

  await check('with Automatic best settings off, nothing is listened to or changed; the starting choices are Denoising, Standard subtitles, Leveling, YouTube loudness, no rebuild', async () => {
    const off = await newPage({ storageState: { cookies: [], origins: [{ origin: BASE, localStorage: [{ name: 'acw.autoSettings', value: 'off' }] }] } });
    await off.page.goto(`${BASE}/app`);
    await off.page.waitForSelector('input[type=file]', { timeout: 20000 });
    await off.page.setInputFiles('input[type=file]', FILES.clean);
    await off.page.getByTestId('start').waitFor({ timeout: 20000 });
    await off.page.waitForTimeout(4000);
    const s = await read(off.page);
    const listening = await off.page.getByTestId('reco-listening').count();
    await off.ctx.close();
    if (listening || s.line) throw new Error('it listened anyway');
    if (s.auto !== false) throw new Error('the switch should be off');
    if (s.noise !== 'denoise' || s.subs !== 'standard' || !s.leveling || s.loudness !== '-14' || s.rebuild) throw new Error(JSON.stringify(s));
  });

  await check('no page errors on the Cleanup page', async () => {
    if (problems.length) throw new Error(problems.join(' | '));
  });
  await ctx.close();
}

async function roomTest(mic, setup) {
  const b = await chromium.launch({
    headless: true,
    args: ['--autoplay-policy=no-user-gesture-required', '--use-fake-ui-for-media-stream', '--use-fake-device-for-media-stream', `--use-file-for-fake-audio-capture=${mic}`],
  });
  const c = await b.newContext({ viewport: { width: 1200, height: 1100 }, permissions: ['microphone'] });
  const page = await c.newPage();
  const problems = [];
  page.on('pageerror', (e) => problems.push(e.message));
  await page.route('**/api/**', (route) => {
    const p = new URL(route.request().url()).pathname.replace('/api/', '');
    if (p === 'me') return route.fulfill({ json: ME });
    if (p === 'billing/pricing') return route.fulfill({ json: PRICING });
    return route.fulfill({ status: 404, json: { detail: 'not mocked' } });
  });
  await page.goto(`${BASE}/app/live`);
  await page.getByTestId('live-start').waitFor();
  await setup(page);
  const t0 = Date.now();
  await page.getByTestId('room-test').click();
  await page.getByTestId('room-listening').waitFor();
  await page.getByTestId('room-result').waitFor({ timeout: 40000 });
  const seconds = (Date.now() - t0) / 1000;
  const changes = await page.getByTestId('room-changes').locator('li').allInnerTexts().catch(() => []);
  const price = await page.getByTestId('room-price').count() ? await page.getByTestId('room-price').innerText() : '';
  let after = null;
  if (await page.getByTestId('room-apply').count()) {
    await page.getByTestId('room-apply').click();
    after = await page.evaluate(() => ({
      strength: document.querySelector('input[name=live-str]:checked')?.value,
      captions: document.querySelector('input[name=live-captions]:checked')?.value,
    }));
  }
  await b.close();
  return { seconds, changes, price, after, problems };
}

await check('Test my room (paid test in a noisy room, Gentle and Standard captions): after about 20 s it lists the changes with their price per minute, and applying them sets the choices', async () => {
  const r = await roomTest(MIC_NOISY, async (page) => {
    await page.getByTestId('live-mode').locator('label', { hasText: 'Paid, done on our side' }).click();
    await page.locator('label:has(input[name=live-str][value=gentle])').click();
  });
  if (r.problems.length) throw new Error(r.problems.join(' | '));
  if (r.seconds < 19 || r.seconds > 30) throw new Error(`took ${r.seconds} s`);
  if (!r.changes.some((t) => /strength: Gentle → Maximum/.test(t)) || !r.changes.some((t) => /Captions: Standard → Extra accurate/.test(t))) throw new Error(JSON.stringify(r.changes));
  if (!/a minute more/.test(r.price)) throw new Error('price: ' + r.price);
  if (r.after?.strength !== 'max' || r.after?.captions !== 'premium') throw new Error(JSON.stringify(r.after));
  return `${r.seconds.toFixed(1)} s: ${r.changes.join('; ')}; ${r.price}`;
});

await browser.close();
console.log(results.every(Boolean) ? '\nall passed' : '\nsome failed');
process.exit(results.every(Boolean) ? 0 : 1);
