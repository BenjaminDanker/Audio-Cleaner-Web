// "Recommended for this file" on the Cleanup page and "Recommended for your room" on the Live page, in a real browser.
// Needs no backend: the API is answered by this script (page.route), and the audio is generated here. Point it at any server that
// serves the built site under /audio/ (for example `npx vite preview --port 4399`):
//   E2E_ORIGIN=http://127.0.0.1:4399 node tests/reco-ui.mjs
import { mkdtempSync, writeFileSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import { chromium } from 'playwright';
import { noisy, voice, wavBytes } from './reco-audio.mjs';

const ORIGIN = process.env.E2E_ORIGIN || 'http://127.0.0.1:4399';
const BASE = `${ORIGIN}/audio`;
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
const FILES = {
  clean: file('clean.wav', noisy(20, 60, 1)),
  mild: file('mild.wav', noisy(20, 22, 2, 'pink')),
  loud: file('noisy.wav', noisy(20, 6, 3, 'pink')),
  hum: file('hum.wav', Float32Array.from({ length: 12 * 44100 }, (_, i) => 0.05 * Math.sin((2 * Math.PI * 50 * i) / 44100))),
  long: file('long.wav', noisy(200, 40, 4), 48000),
};
const MIC = file('mic-clean.wav', noisy(30, 55, 5), 48000);
const MIC_NOISY = file('mic-noisy.wav', noisy(30, 6, 6, 'pink'), 48000);

const browser = await chromium.launch({ headless: true, args: ['--autoplay-policy=no-user-gesture-required'] });

async function newPage(extraArgs = {}) {
  const ctx = await browser.newContext({ viewport: { width: 1200, height: 1100 }, permissions: ['microphone'], ...extraArgs });
  const page = await ctx.newPage();
  const problems = [];
  page.on('pageerror', (e) => problems.push(`pageerror: ${e.message}`));
  await page.route('**/audio/api/**', (route) => {
    const p = new URL(route.request().url()).pathname.replace('/audio/api/', '');
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
  lstrength: document.querySelector('input[name=lstrength]:checked')?.value,
  strength: document.querySelector('input[name=strength]:checked')?.value,
  rebuild: document.querySelector('input[name=rebuild]')?.checked,
  badges: [...document.querySelectorAll('.reco-badge')].map((b) => b.dataset.testid),
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

  await check('clean recording: noise removal Off is recommended and chosen, with the reasons in plain words', async () => {
    const t0 = Date.now();
    const s = await open(page, FILES.clean);
    if (s.noise !== 'off') throw new Error(`noise ${s.noise}: ${s.line}`);
    if (!s.badges.includes('reco-noise-off') || !s.badges.includes('reco-subs-standard')) throw new Error(`badges ${s.badges}`);
    if (!/clean and quiet/.test(s.why.join(' '))) throw new Error(s.why.join(' | '));
    if (s.subs !== 'off') throw new Error('subtitles must not be switched on for the person');
    return `${Date.now() - t0} ms: ${s.line}`;
  });

  await check('steady noise: the free cleanup in the browser, with a strength; much noise: Clean & polish', async () => {
    const mild = await open(page, FILES.mild);
    if (mild.noise !== 'browser' || !mild.lstrength || mild.lstrength === 'max') throw new Error(`mild: ${JSON.stringify(mild)}`);
    const loud = await open(page, FILES.loud);
    if (loud.noise !== 'polish' || !/stronger noise remover/.test(loud.why.join(' '))) throw new Error(`loud: ${JSON.stringify(loud)}`);
    if (!loud.badges.includes('reco-noise-polish') || !loud.badges.includes('reco-subs-accurate')) throw new Error(`badges ${loud.badges}`);
    return `mild -> ${mild.noise}/${mild.lstrength}; loud -> ${loud.noise}/${loud.strength}`;
  });

  await check('a recording with no one speaking: nothing needed, nothing offered', async () => {
    const s = await open(page, FILES.hum);
    if (s.noise !== 'off' || !/nothing needed/.test(s.line) || s.badges.some((b) => b.includes('subs'))) throw new Error(JSON.stringify(s));
    return s.line;
  });

  await check('a long file (3+ minutes) is looked at in about a second and the recommendation is in', async () => {
    const t0 = Date.now();
    const s = await open(page, FILES.long);
    const ms = Date.now() - t0;
    if (!s.line || ms > 8000) throw new Error(`${ms} ms: ${s.line}`);
    return `${ms} ms: ${s.line}`;
  });

  await check('the person can change everything, and "Use the recommended choices" brings the recommendation back', async () => {
    const s = await open(page, FILES.loud);
    await page.locator('label.choice:has(input[name=noise][value=off])').click();
    await page.getByTestId('reco-apply').waitFor({ timeout: 3000 });
    if ((await read(page)).noise !== 'off') throw new Error('did not change');
    await page.getByTestId('reco-apply').click();
    const back = await read(page);
    if (back.noise !== s.noise || back.strength !== s.strength) throw new Error(`${JSON.stringify(back)} vs ${JSON.stringify(s)}`);
    if (await page.getByTestId('reco-apply').count()) throw new Error('the button should be gone');
    return `back to ${back.noise}/${back.strength}`;
  });

  await check('"Automatic best settings" switched off: nothing listened to or picked, remembered; on again picks the settings', async () => {
    const s = await open(page, FILES.loud);
    await page.getByTestId('auto-settings').click();
    const off = await read(page);
    if (off.line || off.badges.length) throw new Error('recommendation still shown: ' + JSON.stringify(off));
    if (off.noise !== s.noise) throw new Error('the choices must stay as they were');
    await page.goto(`${BASE}/app`);
    await page.waitForSelector('input[type=file]', { timeout: 20000 });
    await page.setInputFiles('input[type=file]', FILES.loud);
    await page.getByTestId('start').waitFor({ timeout: 20000 });
    await page.waitForTimeout(1500);
    const fresh = await read(page);
    if (fresh.line || fresh.badges.length || await page.getByTestId('reco-listening').count()) throw new Error('listened while off: ' + JSON.stringify(fresh));
    if (await page.locator('input[name=auto]').isChecked()) throw new Error('the switch was not remembered');
    await page.getByTestId('auto-settings').click();
    await page.getByTestId('reco-line').waitFor({ timeout: 30000 });
    const on = await read(page);
    if (on.noise !== s.noise || on.strength !== s.strength) throw new Error(`${JSON.stringify(on)} vs ${JSON.stringify(s)}`);
    return `off: ${fresh.noise}; on again: ${on.noise}/${on.strength}`;
  });

  await check('picking the subtitle language moves the recommendation to Extra accurate for a language Standard is weaker in', async () => {
    await open(page, FILES.clean);
    if (!(await read(page)).badges.includes('reco-subs-standard')) throw new Error('standard not recommended first');
    await page.locator('label.choice:has(input[name=subs][value=standard])').click();
    await page.selectOption('#lang', 'ja');
    const s = await read(page);
    if (!s.badges.includes('reco-subs-accurate') || s.badges.includes('reco-subs-standard')) throw new Error(JSON.stringify(s.badges));
    return 'ok';
  });

  await check('no page errors on the Cleanup page', async () => {
    if (problems.length) throw new Error(problems.join(' | '));
  });
  await ctx.close();
}

async function liveReco(mic, startStrength) {
  const b = await chromium.launch({
    headless: true,
    args: ['--autoplay-policy=no-user-gesture-required', '--use-fake-ui-for-media-stream', '--use-fake-device-for-media-stream', `--use-file-for-fake-audio-capture=${mic}`],
  });
  const c = await b.newContext({ viewport: { width: 1200, height: 1100 }, permissions: ['microphone'] });
  const page = await c.newPage();
  const problems = [];
  page.on('pageerror', (e) => problems.push(e.message));
  await page.route('**/audio/api/**', (route) => {
    const p = new URL(route.request().url()).pathname.replace('/audio/api/', '');
    if (p === 'me') return route.fulfill({ json: ME });
    if (p === 'billing/pricing') return route.fulfill({ json: PRICING });
    return route.fulfill({ status: 404, json: { detail: 'not mocked' } });
  });
  await page.goto(`${BASE}/app/live`);
  await page.getByTestId('live-start').waitFor();
  await page.locator('label.choice:has(input[name=live-free-captions][value=off])').click(); // no caption download needed
  if (startStrength) await page.locator(`label:has(input[name=live-str][value=${startStrength}])`).click();
  await page.getByTestId('live-start').click();
  await page.getByTestId('live-stop').waitFor({ timeout: 60000 });
  if (await page.getByTestId('live-reco').count()) throw new Error('shown before there was anything to say');
  const t0 = Date.now();
  await page.getByTestId('live-reco-line').waitFor({ timeout: 40000 });
  const seconds = (Date.now() - t0) / 1000;
  const line = await page.getByTestId('live-reco-line').textContent();
  const apply = await page.getByTestId('live-reco-apply').count();
  const strengthBefore = await page.evaluate(() => document.querySelector('input[name=live-str]:checked')?.value);
  if (apply) await page.getByTestId('live-reco-apply').click();
  const strengthAfter = await page.evaluate(() => document.querySelector('input[name=live-str]:checked')?.value);
  await b.close();
  return { seconds, line, apply, strengthBefore, strengthAfter, problems };
}

await check('live: about ten seconds in, a quiet room is told noise removal is not needed', async () => {
  const r = await liveReco(MIC);
  if (r.seconds > 14 || !/quiet/.test(r.line) || r.problems.length) throw new Error(JSON.stringify(r));
  return `after ${r.seconds.toFixed(1)} s: ${r.line}`;
});

await check('live: a noisy room with the strength set too low is told which strength suits it, and the button applies it while live', async () => {
  const r = await liveReco(MIC_NOISY, 'gentle');
  if (r.problems.length) throw new Error(JSON.stringify(r));
  if (!/Maximum noise removal suits/.test(r.line) || !r.apply) throw new Error(JSON.stringify(r));
  if (r.strengthBefore !== 'gentle' || r.strengthAfter !== 'max') throw new Error(`${r.strengthBefore} -> ${r.strengthAfter}`);
  return `${r.line} (${r.strengthBefore} -> ${r.strengthAfter})`;
});

await check('live: the right strength already set: the settings fit, no button', async () => {
  const r = await liveReco(MIC_NOISY, 'max');
  if (r.problems.length || r.apply || !/fit your room/.test(r.line)) throw new Error(JSON.stringify(r));
  return r.line;
});

await browser.close();
console.log(results.every(Boolean) ? '\nall passed' : '\nsome failed');
process.exit(results.every(Boolean) ? 0 : 1);
