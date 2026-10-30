// Parity: every feature of today's file pages, one check per row of the owner's checklist (/home/silver/acw-editor-designs/e-editor/parity.html) that has
// something on screen, so nothing of the Cleanup form, the job page, Change and save again and Fix a moment is lost in the editor. Each check names its row.
// Rows that are only a message or a state ("AS TODAY") check the same words as before; rows that moved check where they live now.
// Needs the test backend (tests/run.sh parity) and the fixtures; a few checks make their own small media files with ffmpeg.
// Usage: node tests/parity.mjs
import { execFileSync } from 'node:child_process';
import { existsSync, writeFileSync } from 'node:fs';
import { join } from 'node:path';
import {
  BASE, FFMPEG, FILES, admin, apiCall, check, dragRange, dropFile, launch, mainButton, pick, selectionTitle, setAuto, setKind, setLeveling, setStrength, setSubs, setVoice,
  settleRecommendation, signIn, summary, waitResult,
} from './helpers.mjs';

const EMAIL = `parity-${Date.now()}@example.com`;
const { browser, context, page, logs } = await launch([], { acceptDownloads: true });
const downloads = [];
page.on('download', (d) => downloads.push(d.suggestedFilename()));
const posts = []; // every request that changes something: {m, p, body}
page.on('request', (r) => { const u = new URL(r.url()); if (u.pathname.startsWith('/api/') && r.method() !== 'GET') posts.push({ m: r.method(), p: u.pathname, body: r.postData() }); });
await signIn(page, EMAIL);
admin('grant', EMAIL, '5', 'parity test');
const now = () => Math.floor(Date.now() / 1000);
const sleep = (ms) => page.waitForTimeout(ms);
const txt = (loc) => loc.innerText().then((t) => t.replace(/\s+/g, ' ').trim());
const has = (text, re, what) => { if (!re.test(text)) throw new Error(`${what || 'text'}: ${text.slice(0, 300)}`); };
const tipOf = (loc) => loc.getAttribute('data-tip');
/** Opens a file in the editor with Automatic off (the usual start: Standard, Balanced, Leveling YouTube, no subtitles). */
async function open(file, { auto = false, path = '/app' } = {}) {
  await page.goto(`${BASE}${path}`);
  await page.evaluate((on) => localStorage.setItem('acw.autoSettings', on ? 'on' : 'off'), auto); // (this browser remembers the switch)
  await page.setInputFiles('input[type=file]', join(FILES, file));
  await page.getByTestId('start').waitFor({ timeout: 30000 });
  await settleRecommendation(page);
  await sleep(200);
}
const sections = (tr) => page.locator(`[data-testid=track-${tr}] [data-testid=section]`);
const sectionTexts = async (tr) => (await sections(tr).allInnerTexts()).map((t) => t.trim());
const hover = async (loc) => { await loc.hover(); await page.waitForFunction(() => !document.querySelector('.studio .tip')?.hidden, null, { timeout: 4000 }); return page.locator('.studio .tip').innerText(); };
const pricing = async () => (await apiCall(page, 'GET', 'billing/pricing')).body;

/** Stubs the API for a finished job the editor can open (a stand-in for what the server answers). */
async function stubJob(id, extra = {}, timeline = null, tl = {}) {
  const job = {
    id, kind: 'denoise', tier: 'standard', free: false, status: 'done', inputName: `${id}.wav`, displayName: `${id}.wav`, label: 'Denoising', createdAt: now() - 60, finishedAt: now() - 30,
    durationSeconds: 60, progress: 100, stage: '', costMicros: 2600, expiresAt: now() + 7000, queuePosition: null, queueWaitSeconds: null, options: {}, transcribeTier: null,
    outputs: [{ name: `${id}-cleaned.flac`, kind: 'audio', bytes: 123456 }], edit: { available: true, hasTimeline: true, run: null, lastRun: null, suggestions: false }, ...extra,
  };
  const state = { job };
  await page.route(/\/api\/jobs$/, (r) => (r.request().method() === 'GET' ? r.fulfill({ json: { jobs: [state.job], limits: { maxActiveJobs: 3, retentionSeconds: 7200 } } }) : r.continue()));
  await page.route(new RegExp(`/api/jobs/${id}$`), (r) => (r.request().method() === 'GET' ? r.fulfill({ json: state.job }) : r.continue()));
  await page.route(`**/api/jobs/${id}/timeline`, (r) => r.fulfill({ json: {
    timeline: timeline || { version: 1, seconds: 60, layers: { noise: [{ start: 0, end: 60, strength_db: 24, remover: 'auto' }], leveling: [{ start: 0, end: 60, loudness_lufs: -14 }] } },
    rendered: null, seconds: 60, channels: 1, secondTrack: false, cached: {}, suggestions: null, run: null, lastRun: null, preview: null,
    // what the page needs to work out the price of a change itself (editing.quote_inputs): nothing made yet, so every step of a change is charged
    quote: { facts: { n: 2880000, channels: 1, plan: [0], spans: { 0: [[0, 2880000]] }, rebuildMid: false, reference: false }, sourceId: 'stub', cached: [], resolved: {}, subtitles: {},
      rates: { echo: 2600, light: 9000, heavy: 9000, rebuild: 30000, leveling: 1000 }, subtitleRates: { transcribe_standard: 4000, transcribe_accurate: 15000, translate: 4000 },
      video: { makes: false, embedded: true, convertMicros: 0, convertLine: null } },
    ...tl,
  } }));
  await page.route(`**/api/jobs/${id}/peaks/**`, (r) => r.fulfill({ status: 404, json: { detail: 'Nothing is cached for this job yet.', code: 'not_cached' } }));
  await page.route(`**/api/jobs/${id}/suggestions`, (r) => r.fulfill({ json: { suggestions: null } }));
  await page.route(`**/api/jobs/${id}/outputs/**`, (r) => r.fulfill({ body: 'x', contentType: 'application/octet-stream' }));
  state.off = async () => { for (const p of [/\/api\/jobs$/, new RegExp(`/api/jobs/${id}$`), `**/api/jobs/${id}/timeline`, `**/api/jobs/${id}/peaks/**`, `**/api/jobs/${id}/suggestions`, `**/api/jobs/${id}/outputs/**`]) await page.unroute(p); };
  return state;
}
const gotoJob = async (id) => { await page.goto(`${BASE}/app/job/${id}`); await page.getByTestId('studio').waitFor({ timeout: 20000 }); await sleep(600); };

/* ================================================================================================ Opening a file */

await check('Opening a file: drop or pick a file; the long list of accepted types is the same text as today', async () => {
  await page.goto(`${BASE}/app`);
  await page.locator('input[type=file]').waitFor({ state: 'attached' });
  has(await page.locator('.dropzone').innerText(), /Drop an audio or video file here/);
  has(await page.locator('.dropzone').innerText(), /or click to pick one\. Almost any audio or video file: MP3, WAV, M4A, FLAC, WMA, AMR, MP4, MOV, MKV, AVI, WMV, MPEG and more\./);
  const accept = await page.locator('input[type=file]').getAttribute('accept');
  for (const e of ['.wma', '.amr', '.avi', '.wmv', '.mpeg', '.mkv']) if (!accept.includes(e)) throw new Error(`${e} is not accepted: ${accept}`);
  return 'drop zone text and accept list';
});

await check('Opening a file: file facts (length, video or audio, size) are the line under the file name', async () => {
  await open('noisy.wav');
  has(await txt(page.getByTestId('file-meta')), /^0:1[01] · Audio · \d+(\.\d)? KB · Different file$/);
  has(await txt(page.getByTestId('file-name')), /^noisy\.wav$/);
  await open('test.mp4');
  has(await txt(page.getByTestId('file-meta')), /· Video · /);
  return await txt(page.getByTestId('file-meta'));
});

await check('Opening a file: reading the file; an empty file and a file too small to hold sound are refused with today\'s words, no editor opens', async () => {
  await page.goto(`${BASE}/app`);
  await page.setInputFiles('input[type=file]', { name: 'empty.wav', mimeType: 'audio/wav', buffer: Buffer.alloc(0) });
  await page.getByTestId('job-error').waitFor();
  if ((await txt(page.getByTestId('job-error'))) !== 'That file is empty (0 bytes). Please pick another one.') throw new Error(await txt(page.getByTestId('job-error')));
  await page.setInputFiles('input[type=file]', { name: 'tiny.txt', mimeType: 'text/plain', buffer: Buffer.from('hello') });
  await page.waitForFunction(() => /too small/.test(document.querySelector('[data-testid=job-error]')?.textContent || ''));
  if ((await txt(page.getByTestId('job-error'))) !== 'That file is too small to hold any sound. Please pick another one.') throw new Error(await txt(page.getByTestId('job-error')));
  if (await page.getByTestId('studio').count()) throw new Error('an editor opened for a file with no sound');
  return 'both messages, no editor';
});

await check('Opening a file: Your jobs stays on the Cleanup page; opening a job opens the editor', async () => {
  const s = await stubJob('pj1', {}, null);
  try {
    await page.goto(`${BASE}/app`);
    await page.getByTestId('job-row').waitFor();
    has(await txt(page.getByTestId('jobs-note')), /Every file stays here, with its results, for 2 hours after your last change to it, and at most 6 hours after the upload\./);
    await page.getByTestId('job-open').click();
    await page.waitForURL(/\/app\/job\/pj1$/);
    await page.getByTestId('studio').waitFor();
    return 'row -> editor';
  } finally { await s.off(); }
});

await check('Opening a file: the Cleanup page is the editor before a file is chosen (Choose a file, the drop zone in the main pane, the (i) button with Good to know, a Start that is off, the balance, the jobs beside it, no Balance card); choosing a file fills it in without a page reload', async () => {
  const listRoute = (jobs) => page.route(/\/api\/jobs$/, (r) => (r.request().method() === 'GET' ? r.fulfill({ json: { jobs, limits: { maxActiveJobs: 3, retentionSeconds: 7200 } } }) : r.continue()));
  await listRoute([]);
  try {
    await page.goto(`${BASE}/app`);
    const empty = page.getByTestId('studio-empty');
    await empty.waitFor({ timeout: 20000 });
    has(await txt(page.locator('main h1')), /^Choose a file$/);
    if ((await page.locator('main h1').count()) !== 1) throw new Error('more than one h1');
    if ((await empty.locator('.pane.main input[type=file]').count()) !== 1) throw new Error('the drop zone is not in the main pane');
    if (!(await empty.getByTestId('start-empty').isDisabled())) throw new Error('Start is on with no file');
    has(await txt(empty.getByTestId('bar-balance')), /^Balance \$\d+\.\d\d$/);
    if (await page.getByTestId('available').count() || await page.getByTestId('held').count() || await page.getByRole('heading', { name: 'Add a file' }).count() || await page.getByRole('heading', { name: 'Balance', exact: true }).count()) throw new Error('the old Add a file / Balance cards are back');
    if (await page.getByRole('heading', { name: 'Good to know' }).count()) throw new Error('Good to know is on the page: it belongs to the (i) window');
    // the jobs are a column beside the editor
    await page.getByTestId('jobs-note').waitFor();
    has(await txt(page.locator('.jobs-col')), /Nothing here yet\. Your files show up here once they run\./);
    const ed = await empty.locator('.ed').boundingBox();
    const col = await page.locator('.jobs-col').boundingBox();
    if (!(col.x >= ed.x + ed.width - 1) || Math.abs(col.y - ed.y) > 80) throw new Error(`the jobs are not beside the editor: editor ${JSON.stringify(ed)}, jobs ${JSON.stringify(col)}`);
    // the (i) button opens the same window, with Good to know in it
    await page.getByTestId('info-open').click();
    await page.getByTestId('info-modal').waitFor();
    has(await txt(page.getByTestId('good-to-know')), /^Good to know Free noise removal happens in your browser/);
    await page.getByTestId('info-close').click();
    // the start row is a part of the page
    const pos = await page.getByTestId('studio-empty').locator('.bb').evaluate((e) => getComputedStyle(e).position);
    if (pos === 'fixed' || pos === 'sticky') throw new Error('the start row is ' + pos);
    // choosing a file swaps the empty editor for the real one in place: no reload, the jobs stay beside it
    await page.evaluate(() => { window.__sameDocument = 1; });
    await page.setInputFiles('input[type=file]', join(FILES, 'noisy.wav'));
    await page.getByTestId('start').waitFor({ timeout: 30000 });
    if (await page.getByTestId('studio-empty').count()) throw new Error('the empty editor is still there');
    if ((await page.evaluate(() => window.__sameDocument)) !== 1) throw new Error('the page was reloaded');
    await page.getByTestId('jobs-note').waitFor();
    const col2 = await page.locator('.jobs-col').boundingBox();
    const ed2 = await page.locator('.studio .ed').boundingBox();
    if (!(col2.x >= ed2.x + ed2.width - 1)) throw new Error('the jobs moved from beside the editor');
    return 'empty editor, jobs beside it, a file fills it in place';
  } finally { await page.unroute(/\/api\/jobs$/); }
});

await check('The editor\'s pages (the Cleanup page and a job\'s page) are 1440 px wide on a wide screen and the header and footer match them; other pages keep the narrower width; the header has a Billing link when signed in', async () => {
  const s = await stubJob('pw1', {}, null);
  await page.setViewportSize({ width: 1700, height: 900 });
  try {
    const widths = async () => page.evaluate(() => {
      const w = (sel) => document.querySelector(sel)?.getBoundingClientRect().width ?? null;
      return { head: w('.header-inner'), foot: w('.footer-grid'), body: w('.studio .container.ws') ?? w('main .container') };
    });
    await page.goto(`${BASE}/app`);
    await page.getByTestId('studio-empty').waitFor();
    const cleanup = await widths();
    await gotoJob('pw1');
    const job = await widths();
    await page.goto(`${BASE}/app/billing`);
    await page.getByTestId('available').waitFor();
    const billing = await widths();
    for (const [what, w] of [['Cleanup', cleanup], ['job', job]]) {
      if (Math.abs(w.body - 1440) > 2 || Math.abs(w.head - w.body) > 2 || Math.abs(w.foot - w.body) > 2) throw new Error(`${what}: ${JSON.stringify(w)}`);
    }
    if (!(billing.head < 1300) || Math.abs(billing.head - billing.foot) > 2) throw new Error('Billing: ' + JSON.stringify(billing));
    const nav = page.getByRole('navigation', { name: 'Main' });
    if ((await nav.getByRole('link', { name: 'Billing', exact: true }).getAttribute('href')) !== '/app/billing') throw new Error('no Billing link in the header');
    return `Cleanup ${Math.round(cleanup.body)}, job ${Math.round(job.body)}, Billing ${Math.round(billing.head)} px`;
  } finally { await page.setViewportSize({ width: 1280, height: 900 }); await s.off(); }
});

await check('Opening a file: the start row is under the editor and the settings, in the page (not fixed, not sticky), with Details / Your files, the balance link to Billing and the one button in that order; the output pane is above it', async () => {
  await open('noisy.wav');
  const bar = page.getByTestId('bar');
  const g = await bar.evaluate((b) => {
    const r = (sel) => document.querySelector(sel).getBoundingClientRect();
    const first = b.querySelector('[data-testid=bar-det]').getBoundingClientRect();
    const bal = b.querySelector('[data-testid=bar-balance]').getBoundingClientRect();
    const btn = b.querySelector('[data-testid=start]').getBoundingClientRect();
    return { pos: getComputedStyle(b).position, top: b.getBoundingClientRect().top + window.scrollY, insp: r('[data-testid=inspector]').bottom + window.scrollY, out: r('[data-testid=output-pane]').bottom + window.scrollY,
      det: first.left, bal: bal.left, btn: btn.left, href: b.querySelector('[data-testid=bar-balance]').getAttribute('href') };
  });
  if (g.pos === 'fixed' || g.pos === 'sticky') throw new Error('the start row is ' + g.pos);
  if (!(g.top >= g.insp - 1) || !(g.top >= g.out - 1)) throw new Error('the start row is not under the editor and the settings: ' + JSON.stringify(g));
  if (!(g.det < g.bal && g.bal < g.btn)) throw new Error('order in the start row: ' + JSON.stringify(g));
  if (g.href !== '/app/billing') throw new Error('the balance goes to ' + g.href);
  // it scrolls with the page: a fixed bar would stay where it is
  const y0 = (await bar.boundingBox()).y;
  await page.evaluate(() => window.scrollBy(0, 120));
  await sleep(200);
  const y1 = (await bar.boundingBox()).y;
  const moved = await page.evaluate(() => window.scrollY);
  if (moved > 0 && Math.abs(y0 - y1 - moved) > 2) throw new Error(`the start row did not scroll with the page: ${y0} -> ${y1} after ${moved} px`);
  return `under the editor; order Details, balance, button; scrolled ${moved} px with the page`;
});

await check('Opening a file: the Cleanup page opens without signing in (the free tools; a job\'s page and billing still ask for sign-in), and "taking a short break" shows when the server is on a break (the editor stays, with a notice that the paid options are unavailable; billing shows the calm panel)', async () => {
  const other = await browser.newContext();
  const p = await other.newPage();
  await p.goto(`${BASE}/app`);
  await p.getByTestId('free-no-account').waitFor({ timeout: 15000 });
  if (new URL(p.url()).pathname !== '/app') throw new Error('signed-out /app went to ' + p.url());
  await p.goto(`${BASE}/app/billing`);
  await p.waitForURL(/\/login/, { timeout: 15000 });
  await other.close();
  await page.route('**/api/me', (r) => r.fulfill({ status: 503, json: { detail: 'maintenance' } }));
  try {
    await page.goto(`${BASE}/app`);
    await page.getByTestId('paid-offline').waitFor({ timeout: 15000 });
    has(await txt(page.getByTestId('paid-offline')), /The paid options are unavailable right now\. The free ones still work\./);
    if (!(await page.locator('input[type=file]').count())) throw new Error('no file picker while the server is on a break');
    await page.goto(`${BASE}/app/billing`);
    await page.getByTestId('maintenance').waitFor({ timeout: 15000 });
    has(await txt(page.getByTestId('maintenance')), /We’ll be right back.*is taking a short break/);
  } finally { await page.unroute('**/api/me'); }
  return 'Cleanup opens; billing asks for sign-in; break: editor with a notice, billing panel';
});

/* ================================================================================================ Automatic settings */

await check('Automatic settings: the Automatic switch is under the file name, on at first; switching it off keeps the choices and stops the suggestions', async () => {
  await open('noisy.wav', { auto: true });
  const sw = page.locator('[data-testid=auto-settings] button[role=switch]');
  if ((await sw.getAttribute('aria-checked')) !== 'true') throw new Error('Automatic is not on');
  if (!(await page.locator('.tinfo').getByTestId('auto-settings').count())) throw new Error('the switch is not next to the file name');
  await sw.click();
  if ((await sw.getAttribute('aria-checked')) !== 'false') throw new Error('did not switch off');
  if (await page.getByTestId('sg-chip').count()) throw new Error('the suggestion count stayed with Automatic off');
  return 'on, then off';
});

await check('Automatic settings: suggestions for moments: the count chip, pins on the ruler and the card in the panel, with Apply and Dismiss', async () => {
  await open('suggest-two-parts.wav', { auto: true });
  await page.getByTestId('sg-chip').waitFor({ timeout: 120000 });
  const n = Number(((await txt(page.getByTestId('sg-chip'))) || '0').replace(/\D/g, ''));
  if (n < 1) throw new Error('count ' + n);
  if ((await page.getByTestId('pin').count()) < 1) throw new Error('no pin on the ruler');
  await page.getByTestId('sg-chip').click();
  await page.getByTestId('sg-card').waitFor();
  has(await txt(page.getByTestId('sg-card')), /suggested/);
  const before = await sectionTexts('voice');
  await page.getByTestId('sg-apply').click();
  const after = await sectionTexts('voice');
  if (JSON.stringify(before) === JSON.stringify(after)) throw new Error('Apply changed nothing: ' + after);
  if (await page.getByTestId('sg-chip').count()) await page.getByTestId('sg-chip').click();
  if (await page.getByTestId('sg-card').count()) {
    await page.getByTestId('sg-dismiss').click();
    if (await page.getByTestId('sg-card').count()) throw new Error('the card stayed after Dismiss');
  }
  return `${n} suggestion(s); Apply painted ${after.join(' | ')}`;
});

await check('Automatic settings: "Why these choices", "Set automatically… then changed by you", Undo / Use them again keep today\'s words', async () => {
  await open('noisy.wav', { auto: true });
  await page.getByTestId('reco-status').waitFor({ timeout: 60000 });
  if ((await txt(page.getByTestId('reco-status'))) !== 'Set automatically for this file.') throw new Error(await txt(page.getByTestId('reco-status')));
  await page.getByTestId('reco-details').locator('summary').click();
  has(await txt(page.getByTestId('reco-details').locator('summary')), /^Why these choices$/);
  if (!(await txt(page.getByTestId('reco-line'))) || !(await txt(page.getByTestId('reco-pick')))) throw new Error('the line and the pick are empty');
  await setStrength(page, 'strong');
  await page.getByTestId('reco-apply').waitFor();
  if ((await txt(page.getByTestId('reco-status'))) !== 'Set automatically, then changed by you.') throw new Error(await txt(page.getByTestId('reco-status')));
  if ((await txt(page.getByTestId('reco-apply'))) !== 'Use them again') throw new Error('button');
  await page.getByTestId('reco-apply').click();
  await page.getByTestId('reco-undo').waitFor();
  if ((await txt(page.getByTestId('reco-undo'))) !== 'Undo') throw new Error('button');
  await page.getByTestId('reco-undo').click();
  if ((await page.locator('[data-testid=auto-settings] button[role=switch]').getAttribute('aria-checked')) !== 'false') throw new Error('Undo should switch Automatic off');
  return 'status, Why these choices, Use them again, Undo';
});

/* ================================================================================================ Voice */

await check('Voice: Off · Free in your browser · Denoising (paid) are Voice: Off / Noise removal (Free | Standard) / Voice rebuild', async () => {
  await open('noisy.wav');
  const radios = await page.locator('.studio [role=radio]').allInnerTexts();
  has(radios.join(' | '), /Off.*Noise removal.*Voice rebuild/s);
  await setVoice(page, 'noise');
  const kinds = await page.locator('.studio button[data-act=v-kind]').allInnerTexts();
  if (kinds.join() !== 'Free,Standard') throw new Error('kinds ' + kinds);
  return radios.map((r) => r.split('\n')[0]).join(' | ') + '; ' + kinds.join(' | ');
});

await check('Voice: the price a minute is in each option\'s tooltip, from the live price list (changed here to prove it is not written in the page)', async () => {
  const real = await pricing();
  const fake = JSON.parse(JSON.stringify(real));
  for (const r of fake.rates) if (r.product === 'denoise' && r.tier === 'standard') r.usdPerMinute = 0.0123;
  await page.route('**/api/billing/pricing', (r) => r.fulfill({ json: fake }));
  try {
    await open('noisy.wav');
    await setVoice(page, 'noise');
    const tip = await hover(page.locator('.studio button[data-act=v-kind][data-v=paid]'));
    has(tip, /\$0\.0123 a minute\./);
    has(await tipOf(page.locator('.studio button[data-act=v-kind][data-v=free]')), /Done in your browser/);
    return tip.slice(0, 120);
  } finally { await page.unroute('**/api/billing/pricing'); }
});

await check('Voice: Strength Gentle / Balanced / Strong / Maximum, each with its line (its tooltip); Max is Standard only', async () => {
  await open('noisy.wav');
  await setVoice(page, 'noise');
  const want = { 1: /^Gentle: takes the edge off/, 2: /^Balanced: most of the noise goes/, 3: /^Strong: nearly all of the noise goes/, 4: /^Max: removes as much as it can/ };
  for (const [v, re] of Object.entries(want)) has(await tipOf(page.locator(`.studio button[data-act=v-str][data-v="${v}"]`)), re);
  await setKind(page, 'free');
  const max = page.locator('.studio button[data-act=v-str][data-v="4"]');
  if ((await max.getAttribute('aria-disabled')) !== 'true') throw new Error('Max is offered with Free');
  has(await tipOf(max), /^Max is Standard only\./);
  return 'four lines; Max off with Free';
});

await check('Voice: Background noise Automatic / Light / Heavy with their hints are the Remover buttons\' tooltips (Standard only)', async () => {
  await open('noisy.wav');
  await setVoice(page, 'noise');
  const want = [/^Auto: we measure how noisy/, /^Light noise: hiss, hum/, /^Heavy noise: a café, traffic/];
  for (let i = 0; i < 3; i++) has(await tipOf(page.locator(`.studio button[data-act=v-rem][data-v="${i}"]`)), want[i]);
  await setKind(page, 'free');
  if ((await page.locator('.studio button[data-act=v-rem][data-v="1"]').getAttribute('aria-disabled')) !== 'true') throw new Error('remover offered with Free');
  has(await txt(page.locator('.studio .pnly')), /^Standard only$/);
  return 'three hints; off with Free';
});

await check('Voice: Rebuild the voice has its long hint (with the price) as the Voice rebuild option\'s tooltip', async () => {
  const real = await pricing();
  const fake = { ...real, rebuildAvailable: true };
  if (!fake.rates.some((r) => r.product === 'denoise' && r.tier === 'rebuild')) throw new Error('the price list has no rebuild row');
  await page.route('**/api/billing/pricing', (r) => r.fulfill({ json: fake }));
  try {
    await open('noisy.wav');
    const tip = await tipOf(page.getByTestId('voice-rebuild'));
    has(tip, /^For an echoey room, a muffled voice, phone-quality or damaged sound, or distortion from recording too loud/);
    has(tip, /\$0\.03 a minute, files up to 90 minutes\./);
    return 'long hint with price';
  } finally { await page.unroute('**/api/billing/pricing'); }
});

await check('Voice: "Which one should I pick?" (Denoising vs Voice rebuild) is the first section of All the options', async () => {
  await open('noisy.wav');
  await page.getByTestId('info-open').click();
  await page.getByTestId('info-modal').waitFor();
  const first = await txt(page.getByTestId('info-modal').locator('section').first());
  has(first, /Voice: Noise removal or Voice rebuild\?/);
  has(first, /Keeps your voice exactly as recorded and never changes words\./);
  has(first, /Makes the voice again from scratch, so it also fixes echo/);
  await page.getByTestId('info-close').click();
  return 'comparison in the window';
});

await check('Voice: Rebuild paused (the machine is down) and rebuild up to 90 minutes: the option says "paused right now" and is not offered', async () => {
  const real = await pricing();
  const fake = { ...real, rebuildAvailable: false };
  await page.route('**/api/billing/pricing', (r) => r.fulfill({ json: fake }));
  try {
    await open('noisy.wav');
    const o = page.getByTestId('voice-rebuild');
    if ((await o.getAttribute('aria-disabled')) !== 'true') throw new Error('offered while paused');
    has(await txt(o), /paused right now/);
    has(await tipOf(o), /^Paused right now\. Try again later\.$/);
    return 'paused';
  } finally { await page.unroute('**/api/billing/pricing'); }
});

await check('Voice: Echo removal with the second audio track, and its OBS hint: only for a file that has a second track', async () => {
  await open('noisy.wav');
  if (await page.getByTestId('echo-switch').count() || await page.getByTestId('track-echo').count()) throw new Error('echo offered for one track');
  await open('echo2.mp4');
  await page.getByTestId('echo-switch').waitFor();
  has(await tipOf(page.getByTestId('echo-switch')), /^Removes your speaker echo using the file’s second audio track \(your desktop sound\)\. OBS can record your mic and your desktop audio on separate tracks: put the mic on track 1 and the desktop sound on track 2\. Done on our side, \$0\.0026 a minute\.$/);
  if ((await page.getByTestId('echo-switch').getAttribute('aria-checked')) !== 'true') throw new Error('echo is not on at first for a two-track file');
  if (!(await page.getByTestId('track-echo').count())) throw new Error('no echo row');
  return 'switch, tooltip and row';
});

await check('Voice: Free: one-time 22 MB download, keep the tab open — the Free tooltip says so', async () => {
  await open('noisy.wav');
  await setVoice(page, 'noise');
  const tip = await tipOf(page.locator('.studio button[data-act=v-kind][data-v=free]'));
  has(tip, /Done in your browser, on your own device: nothing is uploaded\. Gentle to Strong \(no Max\), and no choice of remover\. The first time, it downloads the noise remover \(about 22 MB\); how fast it goes depends on your device, and the tab has to stay open\./);
  return 'tooltip';
});

await check('Voice: Free: channels mixed to mono, time on this device, long-recording warning, a video the browser can\'t rebuild gives WAV — the notes are today\'s words', async () => {
  const six = join(FILES, 'parity-6ch.wav');
  const long = join(FILES, 'parity-3h.ogg');
  if (!existsSync(six)) execFileSync(FFMPEG, ['-v', 'error', '-y', '-f', 'lavfi', '-i', 'sine=frequency=300:duration=3', '-ac', '6', six]);
  if (!existsSync(long)) execFileSync(FFMPEG, ['-v', 'error', '-y', '-f', 'lavfi', '-i', 'anullsrc=r=8000:cl=mono', '-t', '7300', '-c:a', 'libopus', '-b:a', '6k', long]);
  await open('parity-6ch.wav');
  await setKind(page, 'free');
  const notes = await txt(page.getByTestId('free-notes'));
  has(notes, /This file has 6 channels; they will be mixed down to mono\./);
  has(notes, /Should take about .* on this device/);
  await open('parity-3h.ogg');
  await setKind(page, 'free');
  has(await txt(page.getByTestId('free-notes')), /This is a long recording \(2 h 01 min\)\. Cleaning it here could take about .* and needs a lot of memory\. Keep this tab open, or choose Denoising for long files\./);
  await open('noisy.avi');
  await setKind(page, 'free');
  has(await txt(page.getByTestId('audio-only-note')), /^Your browser can clean the sound of this video, but can’t rebuild this video type, so you’ll get the cleaned sound as a WAV file\. For the video, pick a video file type under Output: it’s made on our side, with the whole video uploaded\.$/);
  return 'channels, eta, long recording, audio-only';
});

await check('Voice: "Good to know" lines (OBS filter, YouTube Create, captions, auto-translate) are the last sentence of the matching tooltips and are in All the options', async () => {
  await open('noisy.wav');
  await setVoice(page, 'noise');
  has(await tipOf(page.locator('.studio button[data-act=v-kind][data-v=free]')), /Good to know: Just a little hum\? Your editor’s or OBS’s own noise filter may be enough\.$/);
  has(await tipOf(page.locator('.studio button[data-act=v-kind][data-v=paid]')), /Good to know: Light, steady noise from a phone recording\? YouTube Create’s free Audio cleanup may be enough\.$/);
  has(await tipOf(page.locator('.studio button[data-act=s-q][data-v="1"]')), /Good to know: English video for YouTube\? Its free automatic captions are fine for a first pass\.$/);
  await setSubs(page, 'standard');
  has(await tipOf(page.locator('.studio select[data-act=tr-add]')), /Good to know: On YouTube, viewers can turn on its free auto-translate\.$/);
  await page.getByTestId('info-open').click();
  const modal = await txt(page.getByTestId('info-modal'));
  for (const re of [/Just a little hum\?/, /YouTube Create’s free Audio cleanup/, /English video for YouTube\?/, /auto-translate/]) has(modal, re, 'All the options');
  await page.getByTestId('info-close').click();
  return 'tooltips and window';
});

/* ================================================================================================ Leveling */

await check('Leveling: Podcast −16 / YouTube −14 / Broadcast −23 LUFS: the buttons say the names, the tooltips say the LUFS, the timeline sent has the numbers', async () => {
  await open('noisy.wav');
  const names = await page.locator('.studio .bseg.lv button').allInnerTexts();
  if (names.join() !== 'Off,Podcast,YouTube,Broadcast,Keep') throw new Error('names ' + names);
  has(await tipOf(page.locator('.studio button[data-act=l-lvl][data-v="1"]')), /−16 LUFS/);
  has(await tipOf(page.locator('.studio button[data-act=l-lvl][data-v="2"]')), /−14 LUFS/);
  has(await tipOf(page.locator('.studio button[data-act=l-lvl][data-v="3"]')), /−23 LUFS/);
  const sent = [];
  await page.route('**/api/jobs', async (r) => { if (r.request().method() === 'POST') { sent.push(r.request().postDataJSON()); return r.fulfill({ status: 400, json: { detail: 'Stopped by the test.' } }); } return r.continue(); });
  try {
    for (const [name, lufs] of [['podcast', -16], ['youtube', -14], ['broadcast', -23]]) {
      await setLeveling(page, name);
      await page.getByTestId('start').click();
      for (let i = 0; i < 40 && sent.length === 0; i++) await sleep(250);
      const got = sent.pop();
      if (got?.timeline?.layers?.leveling?.[0]?.loudness_lufs !== lufs) throw new Error(`${name}: ${JSON.stringify(got?.timeline?.layers?.leveling)}`);
      await page.getByTestId('start').waitFor({ timeout: 30000 });
    }
  } finally { await page.unroute('**/api/jobs'); }
  return '-16, -14, -23 in the timeline';
});

await check('Leveling: Keep original loudness is the fifth option, Keep (the timeline says null)', async () => {
  await open('noisy.wav');
  has(await tipOf(page.locator('.studio button[data-act=l-lvl][data-v="4"]')), /^Keep loudness: evens out the volume but keeps the overall loudness as recorded\.$/);
  await setLeveling(page, 'keep');
  has((await sectionTexts('level')).join(' '), /Keep/);
  return 'Keep';
});

await check('Leveling: Leveling on its own, without noise removal, makes a paid job whose timeline has Leveling alone, at the price the server gives for it (worked out here)', async () => {
  await open('noisy.wav');
  await setSubs(page, 'off');
  await setVoice(page, 'off');
  const T = Number(await page.getByTestId('wave').getAttribute('data-seconds'));
  const est = (await apiCall(page, 'POST', 'jobs/estimate', { kind: 'denoise', durationSeconds: T, options: { audio_format: 'flac' }, timeline: { version: 1, seconds: T, layers: { leveling: [{ start: 0, end: T, loudness_lufs: -14 }] } } })).body;
  const micros = est.estCostMicros;
  if (est.lines.map((l) => l.item).join() !== 'leveling') throw new Error('lines ' + JSON.stringify(est.lines));
  const shown = micros < 1000 ? '<$0.001' : `$${(micros / 1e6).toFixed(micros >= 100000 || micros % 10000 === 0 ? 2 : 3)}`;
  await page.waitForFunction((t) => document.querySelector('[data-testid=start]')?.textContent === t, `Start · ${shown}`, { timeout: 10000 });
  return `${await mainButton(page)} (${micros} micros)`;
});

/* ================================================================================================ Subtitles */

await check('Subtitles: Off · Free in your browser · Standard · Extra accurate, with their hints as tooltips', async () => {
  await open('noisy.wav');
  const names = await page.locator('.studio button[data-act=s-q]').allInnerTexts();
  if (names.join() !== 'Off,Free,Standard,Extra accurate') throw new Error('names ' + names);
  has(await tipOf(page.locator('.studio button[data-act=s-q][data-v="0"]')), /^No subtitles or transcript for this part\.$/);
  has(await tipOf(page.locator('.studio button[data-act=s-q][data-v="1"]')), /^Made in your browser, on your own device: nothing is uploaded\. Fine for clear speech\. The first time, it downloads about 80 MB/);
  has(await tipOf(page.locator('.studio button[data-act=s-q][data-v="2"]')), /^Done on our side\. Good for most recordings, in many languages\. \$0\.004 a minute\./);
  has(await tipOf(page.locator('.studio button[data-act=s-q][data-v="3"]')), /speech partner we work with: your audio is sent to them, only for this option\./);
  return 'four options';
});

await check('Subtitles: Language is Detect automatically plus the 28 languages', async () => {
  await open('noisy.wav');
  await setSubs(page, 'standard');
  const opts = await page.locator('.studio select[data-act=s-lang] option').allInnerTexts();
  if (opts.length !== 29 || opts[0] !== 'Detect automatically' || !opts.includes('Thai') || !opts.includes('Spanish')) throw new Error(opts.length + ' ' + opts.slice(0, 4));
  return `${opts.length} entries`;
});

await check('Subtitles: also translate into up to 3 languages (Standard / Extra accurate; Free has none); the subtitle files are chosen under the list in the same card, with no note about a set per language', async () => {
  await open('noisy.wav');
  await setSubs(page, 'free');
  if (!(await page.locator('.studio select[data-act=tr-add]').isDisabled())) throw new Error('translations are offered with Free');
  await setSubs(page, 'standard');
  const add = page.locator('.studio select[data-act=tr-add]');
  for (const l of ['es', 'fr', 'de']) await add.selectOption(l);
  if ((await page.getByTestId('translation-chip').count()) !== 3) throw new Error('chips');
  if (await add.count()) throw new Error('a fourth language is offered');
  // the file boxes sit in the Subtitles card, under "Translate to"; the old note about a set of files per language is gone
  const card = page.locator('.studio [data-testid=inspector] section.l-subs');
  for (const k of ['srt', 'vtt', 'txt']) if ((await card.getByTestId(`cap-${k}`).count()) !== 1) throw new Error(`cap-${k} is not in the Subtitles card`);
  if ((await card.getByTestId('translation-chip').count()) !== 3) throw new Error('the chips are not in the Subtitles card');
  const cardText = await txt(card);
  if (!(cardText.indexOf('Translate to') > 0 && cardText.indexOf('Translate to') < cardText.indexOf('Files'))) throw new Error('the files are not under Translate to: ' + cardText);
  if (/One set of subtitle files per language/.test(await page.locator('main').innerText())) throw new Error('the note about a set of files per language is back');
  await page.getByTestId('translation-chip').first().locator('button').click();
  if ((await page.getByTestId('translation-chip').count()) !== 2) throw new Error('a language was not taken away');
  return '3 chips, none more, the file boxes in the Subtitles card';
});

await check('Subtitles: SRT / VTT / TXT are in the Subtitles card, each with its tooltip', async () => {
  await open('noisy.wav');
  await setSubs(page, 'standard');
  if (await page.getByTestId('output-pane').getByTestId('cap-srt').count()) throw new Error('the subtitle files are in the output pane');
  for (const [k, re] of [['srt', /^SRT: subtitles that most video players and editors take\.$/], ['vtt', /^VTT: subtitles for web players\.$/], ['txt', /^TXT: the transcript as plain text, without times\.$/]]) {
    has(await tipOf(page.getByTestId(`cap-${k}`).locator('xpath=..')), re);
  }
  return 'three formats';
});

await check('Subtitles: "Into the video" (MKV) is in the Subtitles card for a video; its hint is its tooltip', async () => {
  await open('speech.mp4');
  await setSubs(page, 'free');
  const box = page.getByRole('checkbox', { name: /Put the subtitles into the video/ });
  await box.waitFor();
  const tip = await tipOf(box.locator('xpath=..'));
  has(tip, /^Put into the video on your device, so nothing extra is uploaded\. Players can turn them on and off\. They go into an MKV file, the kind players show subtitles from\./);
  await box.check();
  has(await txt(box.locator('xpath=..')), /^Into the video$/);
  if ((await page.getByTestId('inspector').getByRole('checkbox', { name: /Put the subtitles into the video/ }).count()) !== 1) throw new Error('the box is not in the settings');
  has(await page.getByTestId('out-format').inputValue(), /^mkv$/);
  return 'tooltip; ticking picks MKV';
});

await check('Subtitles: Extra accurate goes to a speech partner (tooltip and All the options); there is no hint line under Translate to any more', async () => {
  await open('noisy.wav');
  await setSubs(page, 'accurate');
  has(await tipOf(page.locator('.studio button[data-act=s-q][data-v="3"]')), /with a speech partner we work with: your audio is sent to them, only for this option/);
  if (await page.getByTestId('accurate-note').count()) throw new Error('the hint line under Translate to is back');
  await page.getByTestId('info-open').click();
  has(await txt(page.getByTestId('info-modal')), /with a speech partner we work with \(your audio goes to them, only for this option\)/);
  await page.getByTestId('info-close').click();
  return 'tooltip + window, no line';
});

await check('Subtitles: mixing browser and server work (Standard subtitles with Standard or Free noise removal, Free subtitles with Standard noise removal) shows no hint lines under Translate to; the prices and the file boxes are all that is there', async () => {
  await open('noisy.wav');
  await setSubs(page, 'standard');
  const notes = ['both-note', 'browser-subs-note', 'denoise-local-subs-note', 'accurate-note'];
  const none = async (what) => { for (const id of notes) if (await page.getByTestId(id).count()) throw new Error(`${what}: ${id} is on the page`); };
  await none('Standard noise removal + Standard subtitles');
  await setKind(page, 'free');
  await setLeveling(page, 'off');
  await none('Free noise removal + Standard subtitles');
  await setKind(page, 'paid');
  await setSubs(page, 'free');
  await none('Standard noise removal + Free subtitles');
  if ((await page.getByTestId('cap-srt').count()) !== 1) throw new Error('the subtitle file boxes are gone');
  return 'no hint lines';
});

/* ================================================================================================ Choosing parts */

await check('Choosing parts: the whole file is the selection by default, drawn as one with its handles; changing one part leaves the rest', async () => {
  await open('speech.wav');
  if (!/^Whole file · 0:1[01]$/.test(await selectionTitle(page))) throw new Error(await selectionTitle(page));
  if (!(await page.getByTestId('handle-start').isVisible()) || !(await page.getByTestId('handle-end').isVisible())) throw new Error('handles are not drawn on the whole file');
  await dragRange(page, 2, 5);
  has(await selectionTitle(page), /^0:02–0:05 · 3 s$/);
  await setStrength(page, 'max');
  const v = await sectionTexts('voice');
  if (v.length !== 3 || v[1] !== 'Max' || v[0] !== 'Balanced' || v[2] !== 'Balanced') throw new Error(v.join(' | '));
  return v.join(' | ');
});

await check('Choosing parts: drag on the waveform (even past the ends), the handles with the keyboard, tap a part', async () => {
  await open('speech.wav');
  await dragRange(page, 3, 8);
  has(await selectionTitle(page), /^0:03–0:08 · 5 s$/);
  await page.getByTestId('handle-end').focus();
  await page.keyboard.press('ArrowRight');
  has(await selectionTitle(page), /^0:03–0:09 · 6 s$/);
  await setStrength(page, 'strong');
  await page.getByTestId('select-all').click();
  await sections('voice').nth(1).click();
  has(await selectionTitle(page), /^0:03–0:09 · 6 s$/); // a tap on a part selects exactly it
  const box = await page.getByTestId('waveform').boundingBox();
  await page.mouse.move(box.x + box.width * 0.5, box.y + box.height / 2);
  await page.mouse.down();
  await page.mouse.move(box.x + box.width + 80, box.y + box.height / 2, { steps: 6 }); // past the end
  await page.mouse.up();
  has(await selectionTitle(page), /–0:1[01] · /);
  return 'drag, handle key, tap, past the end';
});

await check('Choosing parts: Split at the playhead', async () => {
  await open('speech.wav');
  await page.getByTestId('waveform').click({ position: { x: 200, y: 40 } });
  await page.getByTestId('split').click();
  const v = await sectionTexts('voice');
  if (v.length !== 2) throw new Error('sections ' + v.length);
  return `${v.length} sections`;
});

await check('Choosing parts: Merge is automatic (no button): parts that end up the same join, a split stays until you change something at it', async () => {
  await open('speech.wav');
  if (await page.getByRole('button', { name: /^Merge/ }).count()) throw new Error('a Merge button');
  await dragRange(page, 2, 5);
  await setStrength(page, 'max');
  if ((await sections('voice').count()) !== 3) throw new Error('not split by a change');
  await setStrength(page, 'balanced');
  if ((await sections('voice').count()) !== 1) throw new Error('the same neighbours did not join: ' + (await sections('voice').count()));
  await page.getByTestId('select-all').click();
  await page.getByTestId('waveform').click({ position: { x: 250, y: 40 } });
  await page.getByTestId('split').click();
  if ((await sections('voice').count()) !== 2) throw new Error('split');
  await page.getByTestId('select-all').click();
  if ((await sections('voice').count()) !== 2) throw new Error('a split went away by itself');
  return 'joins by itself; a split stays';
});

await check('Choosing parts: several parts at once (Shift-drag, or Select more)', async () => {
  await open('speech.wav');
  await dragRange(page, 1, 3);
  await page.keyboard.down('Shift');
  await dragRange(page, 6, 8);
  await page.keyboard.up('Shift');
  has(await selectionTitle(page), /^2 ranges · 0:04 total$/);
  await setStrength(page, 'gentle');
  const v = await sectionTexts('voice');
  if (v.filter((t) => t === 'Gentle').length !== 2) throw new Error(v.join(' | '));
  await page.getByTestId('select-all').click();
  if (await page.getByTestId('select-more').isVisible()) throw new Error('Select more is drawn on a wide screen (the prototype has it on a phone)');
  await page.setViewportSize({ width: 390, height: 844 }); // on a phone: Select more adds the next part you tap
  await page.getByTestId('select-more').click();
  if ((await page.getByTestId('select-more').getAttribute('aria-pressed')) !== 'true') throw new Error('Select more is not pressed');
  await sections('voice').nth(0).click();
  await sections('voice').nth(2).click();
  has(await selectionTitle(page), /^2 ranges · /);
  await page.setViewportSize({ width: 1280, height: 900 });
  return v.join(' | ');
});

await check('Choosing parts: Undo and redo', async () => {
  await open('speech.wav');
  await dragRange(page, 2, 5);
  await setStrength(page, 'max');
  await page.getByTestId('undo').click();
  if ((await sections('voice').count()) !== 1) throw new Error('undo');
  await page.getByTestId('redo').click();
  if ((await sections('voice').count()) !== 3) throw new Error('redo');
  await page.keyboard.press('Control+z');
  if ((await sections('voice').count()) !== 1) throw new Error('Ctrl+Z');
  return 'buttons and Ctrl+Z';
});

await check('Choosing parts: a part is chosen on the timeline only: the handles move it with the keys, and the inspector has no Start/End card (no ±1 s buttons, Use playhead, Play part)', async () => {
  await open('speech.wav');
  await dragRange(page, 3, 8);
  has(await selectionTitle(page), /^0:03–0:08 · 5 s$/);
  for (const id of ['range-box', 'time-s', 'time-e', 'nudge-s-minus', 'nudge-s-plus', 'nudge-e-minus', 'nudge-e-plus', 'use-playhead-s', 'use-playhead-e', 'play-part']) {
    if (await page.getByTestId(id).count()) throw new Error(`${id} is back in the inspector`);
  }
  await page.getByTestId('handle-start').focus();
  await page.keyboard.press('ArrowRight');
  await page.getByTestId('handle-end').focus();
  await page.keyboard.press('ArrowLeft');
  has(await selectionTitle(page), /^0:04–0:07 · 3 s$/);
  await page.getByTestId('waveform').click({ position: { x: 330, y: 40 } });
  await page.getByTestId('play').click();
  await page.waitForFunction(() => document.querySelector('[data-testid=studio]')?.dataset.playing === 'yes', null, { timeout: 8000 });
  await page.getByTestId('play').click();
  return 'handles by keys; Play plays from the playhead; no range card';
});

await check('Choosing parts: Zoom − / + (not drawn on the prototype, drawn here as today)', async () => {
  await open('speech.wav');
  if (!(await page.getByTestId('zoom-out').isDisabled())) throw new Error('zoom out is on at the whole file');
  const w0 = (await page.getByTestId('wave').boundingBox()).width;
  await page.getByTestId('zoom-in').click();
  await page.getByTestId('zoom-in').click();
  const w1 = (await page.getByTestId('wave').boundingBox()).width;
  if (Math.abs(w1 / w0 - 4) > 0.1) throw new Error(`${w0} -> ${w1}`);
  await page.getByTestId('zoom-out').click();
  await page.getByTestId('zoom-out').click();
  const w2 = (await page.getByTestId('wave').boundingBox()).width;
  if (Math.abs(w2 - w0) > 1) throw new Error('zoom out');
  return `x1 ${Math.round(w0)}px, x4 ${Math.round(w1)}px`;
});

await check('Choosing parts: the strength slider for any strength (as on today\'s results page) is in the inspector, with its name and dB', async () => {
  await open('speech.wav');
  await setVoice(page, 'noise');
  const slider = page.getByTestId('strength-slider');
  has(await txt(page.getByTestId('strength-name')), /^Balanced \(24 dB\)$/);
  await slider.evaluate((el) => { const set = Object.getOwnPropertyDescriptor(HTMLInputElement.prototype, 'value').set; set.call(el, '1.5'); el.dispatchEvent(new Event('input', { bubbles: true })); });
  has(await txt(page.getByTestId('strength-name')), /^Between Balanced and Strong \(32 dB\)$/);
  has((await sectionTexts('voice')).join(' '), /32 dB/);
  const max = await slider.getAttribute('max');
  if (Number(max) !== 3) throw new Error('the slider runs 0 to 3 over the four names: ' + max);
  await page.locator('.studio button[data-act=v-kind][data-v=free]').click();
  if (Number(await slider.getAttribute('max')) > 2.001) throw new Error('Free goes above Strong');
  return 'a strength between two names';
});

/* ================================================================================================ The result file */

await check('The result file: name and file type are in the output pane under the editor, in view without opening anything (the type follows the name\'s ending and the other way round)', async () => {
  await open('noisy.wav');
  await page.getByTestId('output-pane').waitFor();
  if (await page.getByTestId('output-button').count() || await page.getByTestId('output-pop').count()) throw new Error('the Output menu is back');
  has(await page.getByTestId('output-pane').getByTestId('out-name').inputValue(), /^noisy-cleaned\.flac$/);
  await page.getByTestId('out-format').selectOption('mp3').catch(() => {});
  const name = await page.getByTestId('out-name').inputValue();
  await page.getByTestId('out-name').fill('talk.flac');
  if ((await page.getByTestId('out-format').inputValue()) !== 'flac') throw new Error('typing an ending did not pick the type');
  has(await page.getByTestId('out-name').inputValue(), /^talk\.flac$/);
  return `${name} -> talk.flac`;
});

await check('The result file: video size Same, 2160p, 1440p, 1080p, 720p, 480p with the size notes as the tooltip/hint', async () => {
  await open('conv720.mp4');
  const size = page.getByTestId('out-size');
  await size.waitFor();
  const opts = await size.locator('option').allInnerTexts();
  if (opts[0] !== 'Same as the original' || opts.length < 3) throw new Error(opts.join());
  has(await txt(page.getByTestId('out-size-note')), /The picture is copied as it is whenever the new file type can hold it\.|can’t re-encode/);
  return opts.join(' | ');
});

await check('The result file: this browser can\'t make some types; audio saved as a video gets a plain black picture — the notes stay', async () => {
  await open('noisy.wav');
  await page.getByTestId('out-name').fill('clip.mkv');
  has(await txt(page.getByTestId('out-black-note')), /^There’s no picture in your recording, so the video gets a plain black one\.$/);
  await page.getByTestId('out-name').fill('clip.xyz');
  has(await txt(page.getByTestId('out-hint')), /We can’t make \.xyz files/);
  return 'black picture note; unknown ending refused';
});

/* ================================================================================================ Price, credit and limits */

await check('Price, credit and limits: Start · price is worked out here (no estimate is asked of our side) and is the price list\'s', async () => {
  const asked = [];
  const onReq = (r) => { if (new URL(r.url()).pathname === '/api/jobs/estimate') asked.push(r.url()); };
  page.on('request', onReq);
  try {
    await open('noisy.wav');
    await page.waitForFunction(() => /^Start · \$0\.0\d+$/.test(document.querySelector('[data-testid=start]')?.textContent || ''), null, { timeout: 15000 });
    if (asked.length) throw new Error('the price was asked of our side: ' + asked.join(', '));
    return await mainButton(page);
  } finally { page.off('request', onReq); }
});

await check('Price, credit and limits: the price lines of Details come from the live price list (changed here to prove it)', async () => {
  const real = await pricing();
  const fake = JSON.parse(JSON.stringify(real));
  for (const r of fake.rates) if (r.product === 'denoise' && r.tier === 'standard') r.usdPerMinute = 1.2;
  await page.route('**/api/billing/pricing', (r) => r.fulfill({ json: fake }));
  try {
    await open('long10min.mp3');
    await setLeveling(page, 'off');
    await setSubs(page, 'off');
    await page.getByTestId('bar-det').click();
    const secs = Number(await page.getByTestId('wave').getAttribute('data-seconds'));
    const want = Math.ceil((1200000 * (Math.ceil(secs) + real.stepFixedSeconds.light)) / 60 - 1e-9); // billed per second of audio (a started second counts) and the step's start-up per run
    has(await txt(page.getByTestId('details')), new RegExp(`Noise removal · \\d+:\\d\\d \\+ (?:[\\d.]+ s|\\d+:\\d\\d) start-up \\$${(want / 1e6).toFixed(want >= 100000 ? 2 : 3).replace('.', '\\.')}`));
    return 'Details: ' + (await txt(page.getByTestId('details'))).slice(0, 80);
  } finally { await page.unroute('**/api/billing/pricing'); }
});

await check('Price, credit and limits: Details lists what the price is made of and says each line is its seconds of audio at that step\'s price a minute, no minimum charge', async () => {
  await open('speech.wav');
  await setSubs(page, 'standard');
  await page.getByTestId('bar-det').click();
  const d = await txt(page.getByTestId('details'));
  has(d, /Noise removal · 0:1[01] \+ (?:[\d.]+ s|\d+:\d\d) start-up \$/);
  has(d, /Leveling · 0:1[01] \+ (?:[\d.]+ s|\d+:\d\d) start-up /);
  has(d, /Subtitles, Standard · /);
  has(d, /Total/);
  has(d, /Each line is its seconds of audio, rounded up to a whole second, at that step.s price a minute\. No minimum charge\./);
  await page.locator('.studio select[data-act=tr-add]').selectOption('es');
  has(await txt(page.getByTestId('details')), /Translations · Spanish/);
  return d.slice(0, 160);
});

await check('Price, credit and limits: not enough credit, prices that cannot load, 4 h / 4 GB, 3 files at once, length unknown — the same words as today', async () => {
  // no credit: a second account
  const other = await browser.newContext({ viewport: { width: 1280, height: 900 } });
  const p = await other.newPage();
  await signIn(p, `poor-${Date.now()}@example.com`);
  const choosePaid = async () => {
    await p.goto(`${BASE}/app`);
    await p.setInputFiles('input[type=file]', join(FILES, 'noisy.wav'));
    await p.getByTestId('start').waitFor();
    await p.waitForFunction(() => !document.querySelector('[data-testid=reco-listening]'), null, { timeout: 60000 });
    await p.locator('.studio button[data-act=auto]').click().catch(() => {});
    await p.locator('.studio button[data-act=v-mode][data-v="1"]').click();
    await p.locator('.studio button[data-act=v-kind][data-v=paid]').click();
    await p.getByTestId('start').waitFor();
    await p.waitForFunction(() => /^Start · \$/.test(document.querySelector('[data-testid=start]')?.innerText.replace(/\s+/g, ' ').trim() || ''), null, { timeout: 15000 });
  };
  await choosePaid();
  if (await p.getByTestId('notes').count() && /your balance is/.test(await p.getByTestId('notes').innerText())) throw new Error('a balance warning before starting');
  if (!(await p.getByTestId('start').isDisabled())) throw new Error('Start is on for a price above a balance of $0.00'); // (the page does not start what the balance cannot pay)
  // the page told there is credit while the account has none: our side refuses it (402)
  await p.route('**/api/me', async (r) => { const res = await r.fetch(); const j = await res.json(); await r.fulfill({ response: res, json: { ...j, balance: { ...j.balance, available_micros: 5_000_000, balance_micros: 5_000_000 } } }); });
  await choosePaid();
  await p.getByTestId('start').click(); // (our side refuses it: 402)
  await p.getByTestId('job-error').waitFor({ timeout: 15000 });
  has(await p.getByTestId('job-error').innerText(), /This job costs about \$[\d.]+; your available balance is \$0\.00\./);
  await other.close();
  // prices that cannot load
  await page.route('**/api/billing/pricing', (r) => r.fulfill({ status: 500, json: { detail: 'no' } }));
  try {
    await open('noisy.wav');
    await page.getByTestId('notes').getByText(/We couldn’t load the prices just now, so this can’t start yet\./).waitFor({ timeout: 15000 });
    if (!(await page.getByTestId('notes').getByRole('button', { name: 'Try again' }).count())) throw new Error('no Try again');
    if (!(await page.getByTestId('start').isDisabled())) throw new Error('Start is on without prices');
  } finally { await page.unroute('**/api/billing/pricing'); }
  // 3 files at once
  const busy = Array.from({ length: 3 }, (_, i) => ({ id: `act${i}`, kind: 'denoise', free: false, status: 'running', inputName: 'a.wav', createdAt: now(), durationSeconds: 60, progress: 10, stage: 'Cleaning', outputs: [], costMicros: 0, expiresAt: null }));
  await page.route(/\/api\/jobs$/, (r) => (r.request().method() === 'GET' ? r.fulfill({ json: { jobs: busy, limits: { maxActiveJobs: 3, retentionSeconds: 7200 } } }) : r.continue()));
  try {
    await open('noisy.wav');
    await page.getByTestId('notes').getByText('You have 3 files being processed. Start this one when one of them is done.').waitFor({ timeout: 15000 });
  } finally { await page.unroute(/\/api\/jobs$/); }
  // length unknown (a file type the browser can't read: the length is a guess from the size)
  await open('noisy.wma');
  return 'credit, prices, 3 files; the rest in the unit suite';
});

/* ================================================================================================ Running */

const running = (extra) => ({ status: 'running', progress: 40, stage: 'Cleaning', estimateSeconds: 120, expiresAt: null, finishedAt: null, outputs: [], ...extra });

await check('Running: progress and time left: a bar and the step line with "about N min left"', async () => {
  const s = await stubJob('run1', running({ estimateSeconds: 300 }));
  try {
    await gotoJob('run1');
    const bar = page.getByTestId('job-progress');
    await bar.waitFor();
    if (Math.round(await bar.evaluate((p) => p.value)) !== 40) throw new Error('progress value');
    has(await txt(page.getByTestId('job-line')), /^Cleaning the sound · 40% · (about \d+ min left|less than a minute left)$/);
    if ((await page.getByTestId('processing').count()) !== 1) throw new Error('the main button should be the Processing one');
    has(await txt(page.getByTestId('processing')), /^Processing · 40%$/);
    return await txt(page.getByTestId('job-line'));
  } finally { await s.off(); }
});

await check('Running: the step line says uploading, in line (with the place and the wait) or cleaning', async () => {
  const lines = [];
  for (const [status, extra] of [['uploading', { progress: 0, stage: '' }], ['queued', { progress: 0, queuePosition: 2, queueWaitSeconds: 120, stage: '' }]]) {
    const s = await stubJob('run2', running({ status, ...extra }));
    try { await gotoJob('run2'); lines.push(await txt(page.getByTestId('job-line'))); } finally { await s.off(); }
  }
  has(lines[0], /^Waiting for the upload to finish$/);
  has(lines[1], /^In line: 1 ahead of you, about 2 min/);
  return lines.join(' | ');
});

await check('Running: Cancel job (while uploading or in line) asks the server to cancel, and the job is not charged', async () => {
  const s = await stubJob('run3', running({ status: 'queued', progress: 0, queuePosition: 1, queueWaitSeconds: 30, stage: '' }));
  let canceled = 0;
  await page.route('**/api/jobs/run3/cancel', (r) => { canceled += 1; return r.fulfill({ json: { ok: true } }); });
  try {
    await gotoJob('run3');
    await page.getByTestId('cancel-job').click();
    for (let i = 0; i < 20 && !canceled; i++) await sleep(150);
    if (canceled !== 1) throw new Error('cancel requests: ' + canceled);
    return 'POST …/cancel';
  } finally { await page.unroute('**/api/jobs/run3/cancel'); await s.off(); }
});

await check('Running: failed and canceled jobs say what happened and that nothing was charged; "Keep this tab open while it works." shows while the browser works', async () => {
  const bad = { id: 'bad1', kind: 'denoise', free: false, status: 'failed', inputName: 'bad1.wav', createdAt: now(), durationSeconds: 60, progress: 0, stage: '', outputs: [], costMicros: 0, error: 'We couldn’t read that file.', expiresAt: null, edit: { available: false } };
  await page.route(/\/api\/jobs$/, (r) => (r.request().method() === 'GET' ? r.fulfill({ json: { jobs: [bad, { ...bad, id: 'can1', status: 'canceled', error: 'Canceled.' }], limits: {} } }) : r.continue()));
  try {
    await page.goto(`${BASE}/app/job/bad1`);
    await page.getByTestId('job').waitFor();
    has(await txt(page.getByTestId('job')), /We couldn’t read that file\. You weren’t charged\./);
    await page.goto(`${BASE}/app/job/can1`);
    await page.getByTestId('job').waitFor();
    has(await txt(page.getByTestId('job')), /Canceled\. You weren’t charged\./);
  } finally { await page.unroute(/\/api\/jobs$/); }
  await open('long10min.mp3');
  await pick(page, 'noise', 'browser');
  await pick(page, 'subs', 'off');
  await page.getByTestId('start').click();
  await page.getByTestId('keep-open').waitFor({ timeout: 30000 });
  has(await txt(page.getByTestId('keep-open')), /^Keep this tab open while it works\.$/);
  const warned = await page.evaluate(() => { const e = new Event('beforeunload', { cancelable: true }); window.dispatchEvent(e); return e.defaultPrevented; });
  if (!warned) throw new Error('leaving the page is not warned about while the browser works');
  await page.getByTestId('cancel-job').click();
  await page.getByTestId('start').waitFor({ timeout: 30000 });
  return 'messages, keep-open note, leave-page warning, cancel returns to Start';
});

/* ================================================================================================ After processing */

const doneJob = (id, files, extra = {}) => stubJob(id, { outputs: files.map((n) => ({ name: n, kind: n.endsWith('.flac') ? 'audio' : n.endsWith('.txt') ? 'transcript' : 'captions', bytes: 4096, lang: n.endsWith('.flac') ? undefined : 'en' })), transcribeTier: files.length > 1 ? 'local' : null, ...extra });

await check('After processing: Your files lists names, labels and sizes (labels like "Cleaned audio", "Subtitles")', async () => {
  const s = await doneJob('pd1', ['pd1-cleaned.flac', 'pd1.srt', 'pd1.vtt', 'pd1-transcript.txt']);
  try {
    await gotoJob('pd1');
    await page.evaluate(() => localStorage.setItem('acw.wants.pd1', JSON.stringify({ sound: true, format: 'flac', name: 'pd1-cleaned.flac', size: 'same', subs: true, caps: { srt: true, vtt: true, txt: true }, embed: false, soundHere: false, subsHere: false })));
    await page.reload(); await page.getByTestId('studio').waitFor(); await sleep(600);
    await page.getByTestId('bar-files').click();
    const rows = (await page.getByTestId('made-file').allInnerTexts()).map((t) => t.replace(/\s+/g, ' '));
    if (rows.length !== 4) throw new Error('rows ' + JSON.stringify(rows));
    has(rows.join(' | '), /pd1-cleaned\.flac Cleaned audio 4\.0 KB/);
    has(rows.join(' | '), /pd1\.srt Subtitles/);
    has(rows.join(' | '), /pd1-transcript\.txt Transcript/);
    return rows.join(' | ');
  } finally { await s.off(); }
});

await check('After processing: Download again, and "Downloads all N files. If your browser asks, allow it." sits in Your files', async () => {
  const s = await doneJob('pd2', ['pd2-cleaned.flac', 'pd2.srt']);
  try {
    await gotoJob('pd2');
    await page.getByTestId('bar-files').click();
    has(await txt(page.getByTestId('files')), /Downloads all 2 files\. If your browser asks, allow it\./);
    await page.getByTestId('bar-files').click();
    downloads.length = 0;
    await page.getByTestId('download-again').click();
    for (let i = 0; i < 40 && downloads.length < 2; i++) await sleep(250);
    if (downloads.slice().sort().join() !== 'pd2-cleaned.flac,pd2.srt') throw new Error('downloaded ' + downloads);
    return downloads.join(', ');
  } finally { await s.off(); }
});

await check('After processing: the subtitle preview is at the top of Your files (a free run made here has the words)', async () => {
  await open('speech.wav');
  await pick(page, 'noise', 'off');
  await pick(page, 'subs', 'browser');
  await page.getByTestId('start').click();
  await waitResult(page, 'result', 180000);
  await page.getByTestId('bar-files').click();
  has(await txt(page.getByTestId('subs-preview')), /responsible|hearing|damage|exposure/i);
  const first = await page.getByTestId('files').evaluate((el) => el.firstElementChild.getAttribute('data-testid'));
  if (first !== 'subs-preview') throw new Error('the preview is not first: ' + first);
  return 'preview first';
});

await check('After processing: cost and "Files are cleared away in … (2 hours after your last change, at most 6 hours after the upload)"', async () => {
  const s = await doneJob('pd3', ['pd3-cleaned.flac'], { costMicros: 4559, expiresAt: now() + 3 * 3600 / 2 });
  try {
    await gotoJob('pd3');
    await page.getByTestId('bar-files').click();
    const t = await txt(page.getByTestId('files'));
    has(t, /Cost \$0\.005/);
    has(t, /Files are cleared away in 1 h 30 min \(2 hours after your last change, at most 6 hours after the upload\)/);
    return t.slice(0, 160);
  } finally { await s.off(); }
});

await check('After processing: Delete now asks ("They can’t be brought back."), Yes, delete / Keep them', async () => {
  const s = await doneJob('pd4', ['pd4-cleaned.flac']);
  let deletes = 0;
  await page.route('**/api/jobs/pd4', (r) => { if (r.request().method() === 'DELETE') { deletes += 1; return r.fulfill({ json: { ok: true } }); } return r.fallback(); });
  try {
    await gotoJob('pd4');
    await page.getByTestId('bar-files').click();
    await page.getByTestId('delete-now').click();
    has(await txt(page.getByTestId('delete-confirm')), /^Delete everything of this file now: your upload, the results and any other versions made from it\? They can’t be brought back\.$/);
    await page.getByRole('button', { name: 'Keep them' }).click();
    if (deletes) throw new Error('deleted without a yes');
    await page.getByTestId('delete-now').click();
    await page.getByTestId('delete-yes').click();
    for (let i = 0; i < 30 && !deletes; i++) await sleep(150);
    if (deletes !== 1) throw new Error('deletes ' + deletes);
    await page.waitForURL(/\/app$/, { timeout: 10000 });
    return 'asked, kept, deleted, back on /app';
  } finally { await page.unroute('**/api/jobs/pd4'); await s.off(); }
});

await check('After processing: a free run is kept in this tab only, with Remove', async () => {
  await open('noisy.wav');
  await pick(page, 'noise', 'browser');
  await pick(page, 'subs', 'off');
  await page.getByTestId('start').click();
  await waitResult(page, 'result', 120000);
  await page.getByTestId('bar-files').click();
  has(await txt(page.getByTestId('files')), /Kept in this tab only, until you leave or reload the page\./);
  has(await txt(page.getByTestId('remove-local')), /^Remove$/);
  await page.getByTestId('remove-local').click();
  await page.waitForURL(/\/app$/, { timeout: 10000 });
  return 'kept in this tab; Remove goes back';
});

await check('After processing: Change and save again — strength, rebuild, leveling, subtitles, type, name, files, into the video, price — are the same controls with Save changes · price', async () => {
  const s = await doneJob('pd5', ['pd5-cleaned.flac']);
  try {
    await gotoJob('pd5');
    has(await txt(page.getByTestId('download-again')), /^Download again$/);
    await setStrength(page, 'strong');
    await setLeveling(page, 'broadcast');
    await setSubs(page, 'standard');
    await page.getByTestId('save-changes').waitFor();
    await page.waitForFunction(() => /Save changes · \$0\.0\d+/.test(document.querySelector('[data-testid=save-changes]')?.textContent || ''), null, { timeout: 8000 });
    has((await sectionTexts('voice')).join(' '), /Strong/);
    has((await sectionTexts('level')).join(' '), /Broadcast/);
    await page.getByTestId('out-name').fill('again.mp3');
    has(await page.getByTestId('out-format').inputValue(), /^mp3$/);
    return await txt(page.getByTestId('save-changes'));
  } finally { await s.off(); }
});

await check('After processing: Fix a moment is this editor (a part of a finished file is chosen and changed in the same screen)', async () => {
  const s = await doneJob('pd6', ['pd6-cleaned.flac']);
  try {
    await gotoJob('pd6');
    await dragRange(page, 10, 20);
    has(await selectionTitle(page), /^0:10–0:20 · 10 s$/);
    await setStrength(page, 'max');
    const v = await sectionTexts('voice');
    if (v.length < 3) throw new Error('a part was not made: ' + v.join(' | '));
    return v.join(' | ');
  } finally { await s.off(); }
});

await check('After processing: after a reload the original video is chosen once; a server notice is shown', async () => {
  const s = await doneJob('pd7', ['pd7-cleaned.flac'], { inputName: 'pd7.mp4', displayName: 'pd7.mp4', notice: 'Only one audio track was found, so echo removal was not used.' });
  try {
    await gotoJob('pd7');
    has(await txt(page.getByTestId('job-notice')), /Only one audio track was found/);
    await page.evaluate(() => localStorage.setItem('acw.wants.pd7', JSON.stringify({ sound: true, format: 'mp4', name: 'pd7-cleaned.mp4', size: 'same', subs: false, caps: {}, embed: false, soundHere: false, subsHere: false })));
    await page.reload(); await page.getByTestId('studio').waitFor(); await sleep(800);
    has(await txt(page.getByTestId('download-again')), /^Choose the original video$/);
    has(await txt(page.getByTestId('need-video')), /^To put the picture back, choose your original video once\. It stays on your device\.$/);
    return 'notice; the button asks for the original video';
  } finally { await s.off(); }
});

console.log(logs.filter((l) => !/Failed to load resource|favicon|net::ERR|503|500/.test(l)).slice(0, 6).join('\n'));
await browser.close();
process.exit(summary().some((r) => !r.ok) ? 1 : 0);
