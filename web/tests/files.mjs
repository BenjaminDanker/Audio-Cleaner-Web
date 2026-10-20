// The editor for a file (the Cleanup page once a file is chosen): what each choice sends to the server (checked by stopping the job
// request, so nothing is made; the body is a timeline, docs/api.md section 8), the prices in the editor (the price on Start is the
// server's own estimate for the same timeline: the page asks for it, and the Details list adds up from the price list), files that cannot be
// used or that come with no credit, names shown safely, and two real paid jobs: a WAV sent as FLAC gives the same result as the raw
// upload, and echo removal with two audio tracks.
// Needs the local backend (port 8792, see helpers.mjs) with `npm run build` output and the fixtures (make-fixtures.sh).
// Usage: node tests/files.mjs
import { createHash } from 'node:crypto';
import { readFileSync, statSync } from 'node:fs';
import { join } from 'node:path';
import { BASE, FILES, SHOTS, admin, check, dragRange, dropFile, jobIdOf, launch, mainButton, pick, setLeveling, setStrength, setVoice, settleRecommendation, signIn, summary } from './helpers.mjs';

const EMAIL = `files-${Date.now()}@example.com`;
const SERVER_RATES = { denoise: 0.009, leveling: 0.001, rebuild: 0.03 }; // the served list (GET /api/billing/pricing), per minute
admin('login-link', EMAIL); // creates the user (no credit: the checks that need none must not need any)

/** fetch() with the CSRF token, run inside the page (same origin). */
const apiCall = (page, method, path, body) => page.evaluate(async ({ method, path, body }) => {
  const me = await (await fetch('/api/me', { credentials: 'same-origin' })).json();
  const res = await fetch(`/api/${path}`, {
    method, credentials: 'same-origin',
    headers: { Accept: 'application/json', 'Content-Type': 'application/json', 'X-CSRF-Token': me.csrfToken },
    body: body === undefined ? undefined : JSON.stringify(body),
  });
  const text = await res.text();
  return { status: res.status, body: text ? JSON.parse(text) : null };
}, { method, path, body });

const { browser, page, logs } = await launch();
await signIn(page, EMAIL);
await apiCall(page, 'POST', 'account/tos', { version: '2026-10-10' });

// Job requests are recorded and stopped with an error, so no job is made, unless a real run lets them through.
const posts = [];
let passThrough = false;
await page.route('**/api/jobs', (r) => {
  if (r.request().method() !== 'POST' || passThrough) return r.continue();
  posts.push(r.request().postDataJSON());
  return r.fulfill({ status: 400, json: { detail: 'Stopped by the test.' } });
});

const start = () => page.getByTestId('start');
const choose = (f) => dropFile(page, join(FILES, f));
const freshApp = async () => { await page.goto(`${BASE}/app`); await page.getByTestId('available').waitFor(); };
const withMe = (patch) => page.route('**/api/me', async (r) => { const res = await r.fetch(); const j = await res.json(); await r.fulfill({ response: res, json: patch(j) }); });
const withPricing = (patch) => page.route('**/api/billing/pricing', async (r) => { const res = await r.fetch(); const j = await res.json(); await r.fulfill({ response: res, json: patch(j) }); });
const lastPost = () => posts.at(-1);
/** Presses Start and waits until the (stopped) job request has gone out and the page has shown the error; answers the request's body. */
async function send() {
  const n = posts.length;
  await start().click();
  for (let i = 0; i < 600 && posts.length === n; i++) await page.waitForTimeout(100); // (a WAV is packed smaller first)
  if (posts.length === n) throw new Error('Start did not send anything: ' + (await page.getByTestId('notes').innerText().catch(() => '')));
  await page.getByTestId('job-error').waitFor({ timeout: 60000 });
  return lastPost();
}
const openOutput = async () => { if (!(await page.getByTestId('output-pop').count())) await page.getByTestId('output-button').click(); await page.getByTestId('output-pop').waitFor(); };
const closePop = async () => { if (await page.getByTestId('output-pop').count()) await page.getByTestId('output-button').click(); };
const seconds = async () => Number(await page.getByTestId('wave').getAttribute('data-seconds'));
const waitStartPrice = () => page.waitForFunction(() => /^Start · /.test(document.querySelector('[data-testid=start]')?.textContent || ''), null, { timeout: 15000 });

// The price list and the prototype's price words ("Free", "<$0.001", "$0.024", "$1.50"); the checks work the price out themselves.
const priceList = await (await fetch(`${BASE}/api/billing/pricing`)).json();
const perMinute = (product, tier) => priceList.rates.find((r) => r.product === product && r.tier === tier);
const perSecond = (product, tier) => Math.round(perMinute(product, tier).usdPerMinute * 1e6) / 60; // micros
const words = (m) => (m <= 0 ? 'Free' : m < 1000 ? '<$0.001' : `$${m >= 100000 || m % 10000 === 0 ? (m / 1e6).toFixed(2) : (m / 1e6).toFixed(3)}`);
/** What the price list asks for these many seconds of each (the editor's price for a file with one section per track): a started second counts, the sum is rounded up to a micro. */
const started = (s) => Math.ceil(s - 1e-6);
const listPrice = ({ noise = 0, rebuild = 0, level = 0, std = 0, ext = 0 }) => Math.ceil(started(noise) * perSecond('denoise', 'standard') + started(rebuild) * perSecond('denoise', 'rebuild') + started(level) * perSecond('leveling', 'standard')
  + started(std) * perSecond('transcribe', 'local') + started(ext) * perSecond('transcribe', 'premium') - 1e-6);
/** The request without the file's fields: what the page itself asks the estimate with. */
const withoutFile = ({ filename: _f, sizeBytes: _s, displayName: _d, ...rest }) => rest;
// Every estimate the page asked (a job on our side is priced by the server: the timeline, without the file), with the server's answer.
const estimates = [];
page.on('response', async (res) => {
  if (res.request().method() !== 'POST' || new URL(res.url()).pathname !== '/api/jobs/estimate') return;
  try { estimates.push({ body: res.request().postDataJSON(), answer: await res.json(), status: res.status() }); } catch { /* aborted */ }
});
/**
 * The price on Start is the server's estimate for the timeline in the editor: waits for it, then checks that the page's own request carried no
 * file fields, that the server says the same for the same body when asked again, that nothing charges Restore highs, and that the price is not
 * below what the price list asks for the parts (`atLeast`: the server adds each step's per-run amount). Answers {body, answer}.
 */
async function startIsServers({ atLeast = 0 } = {}) {
  await page.waitForTimeout(700); // (the page asks 400 ms after the last change; until it has the answer Start shows no price)
  await waitStartPrice();
  const label = await mainButton(page);
  for (let i = 0; i < 50; i++) {
    const e = estimates.at(-1);
    if (e && label === `Start · ${words(e.answer.estCostMicros)}`) break;
    await page.waitForTimeout(100);
  }
  const e = estimates.at(-1);
  if (!e || label !== `Start · ${words(e.answer.estCostMicros)}`) throw new Error(`Start says "${label}", the page's last estimate says ${e?.answer?.estCostMicros}`);
  for (const k of ['filename', 'displayName', 'sizeBytes']) if (k in e.body) throw new Error(`the estimate carries the file field ${k}`);
  const mine = await apiCall(page, 'POST', 'jobs/estimate', e.body);
  if (mine.status !== 200 || mine.body.estCostMicros !== e.answer.estCostMicros) throw new Error(`the page's estimate ${e.answer.estCostMicros}, the server's for the same body: ${JSON.stringify(mine).slice(0, 300)}`);
  if (mine.body.lines.some((l) => l.item === 'highs')) throw new Error('the server has a highs line: ' + JSON.stringify(mine.body.lines));
  if (mine.body.estCostMicros < atLeast) throw new Error(`the server's estimate ${mine.body.estCostMicros} is below the price list's ${atLeast}`);
  return { body: e.body, answer: mine.body };
}
/**
 * This backend has no card, so the server will not price a Voice rebuild ("paused right now"): the estimate for a timeline with a rebuild part is
 * answered as the server would with the card up (the rebuild's per-run amount, 10,591 micros, and Leveling's, 1,641: docs/api.md section 8), and a
 * real answer goes through when there is one. Undo with unrebuildPricing().
 */
async function rebuildPricing() {
  await page.route('**/api/jobs/estimate', async (r) => {
    const body = r.request().method() === 'POST' ? r.request().postDataJSON() : null;
    if (!body?.timeline?.layers?.rebuild) return r.continue();
    const real = await r.fetch();
    if (real.status() === 200) return r.fulfill({ response: real });
    const L = body.timeline.layers;
    const sec = (parts) => (parts || []).reduce((n, p) => n + started(p.end - p.start), 0);
    const micros = listPrice({ rebuild: sec(L.rebuild), noise: sec(L.noise), level: sec(L.leveling) }) + 10591 + (L.leveling ? 1641 : 0);
    return r.fulfill({ json: { estCostMicros: micros, estCostUsd: micros / 1e6, durationSeconds: body.durationSeconds, maxSeconds: 36000, lines: [{ item: 'rebuild', micros }], options: body.options, timeline: body.timeline } });
  });
}
const unrebuildPricing = () => page.unroute('**/api/jobs/estimate');
/** The estimate the page asked about is the timeline that is sent. */
const sameTimeline = (asked, sent) => {
  if (JSON.stringify(asked.body.timeline) !== JSON.stringify(sent.timeline)) throw new Error(`the price was asked for ${JSON.stringify(asked.body.timeline)}, the job sent ${JSON.stringify(sent.timeline)}`);
};
const { rate: shownRate } = await import('../src/format.js');

// What a job request must look like: a timeline job, none of the fields of the old form.
const OLD_TOP = ['denoiseTier', 'transcribeTier', 'free'];
const OLD_OPTIONS = ['atten_lim_db', 'noise', 'leveling', 'loudness_lufs', 'echo_reference_track', 'restore_highs', 'language', 'translate'];
/** Throws unless `b` is a timeline job of kind denoise whose layers are exactly `layers` (and version `version`); answers the layers. */
function shape(b, layers, version) {
  if (b.kind !== 'denoise') throw new Error('kind ' + b.kind + ': ' + JSON.stringify(b));
  for (const k of OLD_TOP) if (k in b) throw new Error(`${k} is sent: ${JSON.stringify(b)}`);
  for (const k of OLD_OPTIONS) if (k in b.options) throw new Error(`options.${k} is sent: ${JSON.stringify(b.options)}`);
  if (!b.options.audio_format || !Array.isArray(b.options.caption_formats) || typeof b.options.embed_subtitles !== 'boolean') throw new Error('options: ' + JSON.stringify(b.options));
  const tl = b.timeline;
  if (!tl || tl.version !== version || !(Math.abs(tl.seconds - b.durationSeconds) < 0.001)) throw new Error(`timeline version ${tl?.version}, ${tl?.seconds} s of ${b.durationSeconds}: ${JSON.stringify(tl)}`);
  const have = Object.keys(tl.layers).sort().join();
  if (have !== layers.split(',').sort().join()) throw new Error(`layers ${have}, wanted ${layers}: ${JSON.stringify(tl.layers)}`);
  return tl.layers;
}
const whole = (parts, T) => parts.length === 1 && parts[0].start === 0 && Math.abs(parts[0].end - T) < 0.001; // (the timeline keeps milliseconds)

// ---- files that cannot be used, and an account with no credit (this account has none yet) ----
await check('bad files: an empty file is refused with a clear message; a tiny text file is refused at once; no Start is offered for either', async () => {
  await freshApp();
  await page.setInputFiles('input[type=file]', { name: 'empty.wav', mimeType: 'audio/wav', buffer: Buffer.alloc(0) });
  await page.getByTestId('job-error').waitFor({ timeout: 5000 });
  const empty = await page.getByTestId('job-error').innerText();
  if (!/empty/.test(empty)) throw new Error(empty);
  if (await start().count()) throw new Error('Start offered for an empty file');
  await freshApp();
  await page.setInputFiles('input[type=file]', join(FILES, 'bad.txt'));
  await page.getByTestId('job-error').waitFor({ timeout: 5000 });
  if (await start().count()) throw new Error('Start offered for a 17-byte file');
  return empty;
});

await check('bad files: a file the browser cannot open gets Free noise removal greyed out with the reason, Standard as the starting choice, and an honest unknown length', async () => {
  await freshApp();
  // (not through dropFile: the starting choices are what is looked at here)
  await page.setInputFiles('input[type=file]', join(FILES, 'garbage.mp3'));
  await start().waitFor({ timeout: 30000 });
  await settleRecommendation(page);
  const free = page.locator('.studio button[data-act=v-kind][data-v=free]');
  if ((await free.getAttribute('aria-disabled')) !== 'true') throw new Error('free noise removal offered for a file the browser cannot open');
  const why = (await free.getAttribute('data-tip')).trim();
  if (why !== 'Your browser can’t open this file type. Paid Denoising can do it.') throw new Error('reason: ' + why);
  if ((await page.getByTestId('voice-noise').getAttribute('aria-checked')) !== 'true' || (await page.locator('.studio button[data-act=v-kind][data-v=paid]').getAttribute('aria-pressed')) !== 'true') throw new Error('Standard noise removal should be the starting choice');
  await page.getByTestId('length-unknown').waitFor({ timeout: 15000 });
  await page.screenshot({ path: join(SHOTS, 'cleanup-unreadable.png'), fullPage: true });
  return why;
});

await check('credit: an account with no balance cannot start a paid job (Start off, the reason, "Add credit"); free subtitles on a 10 min file start with no credit', async () => {
  await freshApp();
  await choose('noisy.wav');
  await pick(page, 'noise', 'denoise');
  const msg = page.locator('[data-testid=notes] .banner-warn', { hasText: /your balance is/ });
  await msg.waitFor();
  if (!(await msg.getByRole('link', { name: 'Add credit' }).count())) throw new Error('no Add credit link');
  if (await start().isEnabled()) throw new Error('start enabled with no credit');
  const reason = (await msg.innerText()).replace(/\s+/g, ' ').trim();
  if (!/^This costs about .+, but your balance is \$0\.00\. You can also choose the free options\. Add credit$/.test(reason)) throw new Error('reason: ' + reason);
  await freshApp();
  await choose('long10min.mp3');
  await pick(page, 'noise', 'off');
  await pick(page, 'subs', 'browser');
  await page.getByTestId('start').waitFor();
  if ((await mainButton(page)) !== 'Start · Free') throw new Error('label: ' + await mainButton(page));
  if (!(await start().isEnabled())) throw new Error('free start disabled for a free 10 min file');
  return reason;
});

// ---- what each choice sends (the balance is topped up on the page only; the job is stopped by the test) ----
await withMe((j) => ({ ...j, balance: { ...j.balance, available_micros: 5_000_000, balance_micros: 5_000_000 } }));

await check('denoising: the prices in the editor are the server\'s; Gentle sends 12 dB, the remover is Auto by default, Heavy is sent as heavy; YouTube Leveling is -14; nothing of the old request is sent', async () => {
  await freshApp();
  await choose('noisy.wav');
  await pick(page, 'noise', 'denoise');
  await setLeveling(page, 'youtube');
  const T = await seconds();
  const kindTip = await page.locator('.studio button[data-act=v-kind][data-v=paid]').getAttribute('data-tip');
  if (!kindTip.includes(`${shownRate(SERVER_RATES.denoise)} a minute`)) throw new Error('Standard noise removal price in the editor: ' + kindTip);
  const levelTip = await page.locator('.studio .l-level .cl b').getAttribute('data-tip');
  if (!levelTip.includes(`${shownRate(SERVER_RATES.leveling)} a minute`)) throw new Error('Leveling price in the editor: ' + levelTip);
  if ((await page.locator('.studio button[data-act=l-lvl][data-v="2"]').getAttribute('aria-pressed')) !== 'true') throw new Error('Leveling should be on (YouTube)');
  await setStrength(page, 'gentle');
  const asked = await startIsServers({ atLeast: listPrice({ noise: T, level: T }) });
  const want = await mainButton(page);
  posts.length = 0;
  const b = await send();
  sameTimeline(asked, b);
  const L = shape(b, 'leveling,noise', 1);
  const n = L.noise[0];
  if (!whole(L.noise, T) || n.strength_db !== 12 || n.remover !== 'auto' || 'browser' in n) throw new Error('noise: ' + JSON.stringify(L.noise));
  if (!whole(L.leveling, T) || L.leveling[0].loudness_lufs !== -14) throw new Error('leveling: ' + JSON.stringify(L.leveling));
  await page.locator('.studio button[data-act=v-rem][data-v="2"]').click();
  const askedHeavy = await startIsServers({ atLeast: listPrice({ noise: T, level: T }) });
  const h = await send();
  sameTimeline(askedHeavy, h);
  if (shape(h, 'leveling,noise', 1).noise[0].remover !== 'heavy') throw new Error('remover sent: ' + JSON.stringify(h.timeline.layers.noise));
  return `12 dB, auto, then heavy; ${want}; prices ${shownRate(SERVER_RATES.denoise)} and ${shownRate(SERVER_RATES.leveling)} a minute`;
});

await check('leveling alone, subtitles alone, and both together each send exactly what the editor says (Standard and Extra accurate are parts of the timeline, Free subtitles are a part that is made here)', async () => {
  await freshApp();
  await choose('noisy.wav');
  await pick(page, 'noise', 'off');
  await setLeveling(page, 'youtube');
  const T = await seconds();
  posts.length = 0;
  let b = await send();
  let L = shape(b, 'leveling', 1);
  if (!whole(L.leveling, T) || L.leveling[0].loudness_lufs !== -14) throw new Error('leveling only: ' + JSON.stringify(b));

  await pick(page, 'noise', 'denoise');
  await setLeveling(page, 'youtube');
  await pick(page, 'subs', 'standard');
  const asked = await startIsServers({ atLeast: listPrice({ noise: T, level: T, std: T }) });
  b = await send();
  sameTimeline(asked, b);
  L = shape(b, 'leveling,noise,subtitles', 2);
  if (!whole(L.subtitles, T) || JSON.stringify({ ...L.subtitles[0], end: 0 }) !== JSON.stringify({ start: 0, end: 0, quality: 'standard', language: null, translate: [] })) throw new Error('denoise + standard: ' + JSON.stringify(L.subtitles));
  if ('browser' in L.noise[0]) throw new Error('noise: ' + JSON.stringify(L.noise));

  await pick(page, 'subs', 'browser');
  await page.getByTestId('denoise-local-subs-note').waitFor();
  b = await send();
  L = shape(b, 'leveling,noise,subtitles', 2);
  if (L.subtitles[0].quality !== 'free' || !whole(L.subtitles, T) || 'browser' in L.noise[0]) throw new Error('denoise + free subtitles: ' + JSON.stringify(L));

  await pick(page, 'noise', 'browser');
  await pick(page, 'subs', 'standard');
  const askedFree = await startIsServers({ atLeast: listPrice({ std: T }) }); // (the Free cleanup is not charged: only the subtitles are)
  b = await send();
  sameTimeline(askedFree, b);
  L = shape(b, 'noise,subtitles', 2); // (Leveling went off with the free cleanup, as the old form did not offer it there)
  if (L.subtitles[0].quality !== 'standard' || L.noise[0].browser !== true || typeof L.noise[0].strength_db !== 'number' || 'remover' in L.noise[0]) throw new Error('free cleanup + standard: ' + JSON.stringify(L));

  await pick(page, 'noise', 'off');
  await pick(page, 'subs', 'accurate');
  const askedExt = await startIsServers({ atLeast: listPrice({ ext: T }) });
  b = await send();
  sameTimeline(askedExt, b);
  L = shape(b, 'subtitles', 2);
  if (L.subtitles[0].quality !== 'accurate' || !whole(L.subtitles, T)) throw new Error('extra accurate: ' + JSON.stringify(L));
  return 'leveling / leveling + noise + standard / + free subtitles / free noise + standard / extra accurate, as chosen';
});

await check('the line before Start names exactly what will download by itself: the cleaned file under its chosen name, and the subtitle formats ticked (with "inside the video" when that is ticked)', async () => {
  await freshApp();
  await choose('noisy.wav');
  await pick(page, 'noise', 'denoise');
  await pick(page, 'subs', 'standard');
  await openOutput();
  const line = () => page.getByTestId('will-download').innerText();
  let t = await line();
  // (the Output menu starts with the SRT only; the old form ticked all three: tick the other two to get the same line)
  if (!/noisy-cleaned\.flac/.test(t) || !/\(SRT\)/.test(t) || /VTT|TXT/.test(t)) throw new Error('default: ' + t);
  await page.getByTestId('cap-vtt').check();
  await page.getByTestId('cap-txt').check();
  t = await line();
  if (!/noisy-cleaned\.flac/.test(t) || !/SRT, VTT, TXT/.test(t)) throw new Error('all three: ' + t);
  await page.getByTestId('cap-srt').uncheck();
  await page.getByTestId('cap-vtt').uncheck();
  await page.getByTestId('out-name').fill('talk.mp3');
  t = await line();
  if (!/talk\.mp3/.test(t) || !/\(TXT\)/.test(t) || /SRT|VTT/.test(t)) throw new Error('after the changes: ' + t);
  await page.getByTestId('cap-txt').uncheck();
  await page.getByTestId('cap-srt').check();
  await closePop();
  await pick(page, 'subs', 'off');
  await openOutput();
  t = await line();
  if (!/talk\.mp3/.test(t) || /subtitles/.test(t)) throw new Error('with the subtitles off: ' + t);
  await closePop();
  return t.replace(/\s+/g, ' ');
});

await check('echo removal and translation: Echo removal is offered only with Standard noise removal (a file with a second track starts with it on), sends no echo part when off and one when on, uploads one smaller .mka; Spanish is sent as the subtitle part\'s translate', async () => {
  await page.evaluate(() => localStorage.setItem('acw.autoSettings', 'off'));
  await freshApp();
  await page.setInputFiles('input[type=file]', join(FILES, 'echo2.mp4'));
  await start().waitFor({ timeout: 30000 });
  await settleRecommendation(page);
  const sw = page.getByTestId('echo-switch');
  await sw.waitFor();
  if ((await sw.getAttribute('aria-checked')) !== 'true' || (await page.locator('.studio button[data-act=v-kind][data-v=paid]').getAttribute('aria-pressed')) !== 'true') throw new Error('a file with a second audio track should start with Standard noise removal and Echo removal on');
  await page.getByTestId('different-file').click();
  await choose('echo2.mp4'); // (the old starting point: Free cleanup, as chosen by the helper)
  const note = (await page.getByTestId('echo-browser-note').innerText()).replace(/\s+/g, ' ');
  if (!/second audio track[\s\S]*needs Denoising[\s\S]*free cleanup in your browser can.t do it/.test(note)) throw new Error('note: ' + note);
  if ((await sw.getAttribute('aria-disabled')) !== 'true' || (await sw.getAttribute('aria-checked')) !== 'false') throw new Error('echo removal offered for free cleanup');
  await sw.click({ force: true });
  if ((await sw.getAttribute('aria-checked')) !== 'false') throw new Error('echo switched on with the free cleanup');
  await pick(page, 'noise', 'denoise');
  if ((await sw.getAttribute('aria-disabled')) === 'true') throw new Error('echo removal not offered for Standard');
  if ((await sw.getAttribute('aria-checked')) !== 'false') throw new Error('echo should be off (the free cleanup turned it off)');
  const T = await seconds();
  posts.length = 0;
  const off = await send();
  if ('echo' in shape(off, 'noise', 1)) throw new Error('unticked should send no echo part: ' + JSON.stringify(off.timeline));
  await sw.click();
  if ((await sw.getAttribute('aria-checked')) !== 'true') throw new Error('echo did not switch on');
  const b = await send();
  const L = shape(b, 'echo,noise', 1);
  const size = statSync(join(FILES, 'echo2.mp4')).size;
  if (!whole(L.echo, T) || !/\.mka$/.test(b.filename) || b.sizeBytes >= size) throw new Error(JSON.stringify(b));
  await pick(page, 'noise', 'off');
  await page.getByTestId('different-file').click();
  await choose('noisy.wav');
  await pick(page, 'noise', 'denoise');
  await pick(page, 'subs', 'standard');
  await page.getByTestId('translate-add').selectOption('es');
  if ((await page.getByTestId('translation-chip').count()) !== 1 || !/Spanish/.test(await page.getByTestId('translation-chip').innerText())) throw new Error('no Spanish chip');
  const t = await send();
  const S = shape(t, 'noise,subtitles', 2).subtitles;
  if (S[0].quality !== 'standard' || JSON.stringify(S[0].translate) !== '["es"]') throw new Error('translate: ' + JSON.stringify(S));
  return `${b.filename} ${b.sizeBytes} bytes of ${size}`;
});

await check('rebuild the voice: its own price (the server\'s) and no strength/remover choices; the request has a rebuild part and no noise part', async () => {
  await withPricing((j) => ({ ...j, rebuildAvailable: true }));
  await rebuildPricing();
  try {
    await freshApp();
    await choose('noisy.wav');
    await pick(page, 'noise', 'denoise');
    await setVoice(page, 'rebuild');
    if (await page.locator('.studio button[data-act=v-str], .studio button[data-act=v-rem]').count() || await page.getByTestId('strength-slider').count()) throw new Error('strength/noise still shown with rebuild');
    const hint = await page.getByTestId('voice-rebuild').getAttribute('data-tip');
    if (!hint.includes(`${shownRate(SERVER_RATES.rebuild)} a minute`)) throw new Error('rebuild price in the editor: ' + hint);
    const T = await seconds();
    const asked = await startIsServers({ atLeast: listPrice({ rebuild: T }) });
    posts.length = 0;
    const b = await send();
    sameTimeline(asked, b);
    const L = shape(b, 'rebuild', 1);
    if (!whole(L.rebuild, T)) throw new Error('rebuild part: ' + JSON.stringify(L.rebuild));
    return `${hint.slice(-80)} (Start ${await mainButton(page)})`;
  } finally {
    await unrebuildPricing();
    await page.unroute('**/api/billing/pricing');
  }
});

// ---- Rebuild the voice with its words, Leveling on its own; Restore highs is gone ----
const REBUILD_HINT = (price) => `For an echoey room, a muffled voice, phone-quality or damaged sound, or distortion from recording too loud: the whole voice is rebuilt, which also brings back the high sounds that are missing. Keeps only speech, so game sounds, music and room sound are gone. Slower, ${price} a minute, files up to 90 minutes. On very echoey recordings it can change a word, so listen before you publish.`;

await check('the editor has no Restore highs (not in the price list, not in the editor or its "All the options" window); Voice rebuild has its words and price, in this order: Strength, Remover, Voice rebuild; with Automatic off a file starts from Standard noise removal and YouTube Leveling', async () => {
  if (priceList.rates.some((r) => r.product === 'highs')) throw new Error('the price list still has Restore highs: ' + JSON.stringify(priceList.rates.filter((r) => r.product === 'highs')));
  await withPricing((j) => ({ ...j, rebuildAvailable: true }));
  try {
    // without "Automatic" a file starts from the usual choices; this browser remembers the switch
    await freshApp();
    await page.evaluate(() => localStorage.setItem('acw.autoSettings', 'off'));
    await freshApp(); // loaded again with the switch off
    await page.setInputFiles('input[type=file]', join(FILES, 'noisy.wav'));
    await start().waitFor({ timeout: 20000 });
    await settleRecommendation(page);
    if ((await page.getByTestId('voice-noise').getAttribute('aria-checked')) !== 'true' || (await page.locator('.studio button[data-act=v-kind][data-v=paid]').getAttribute('aria-pressed')) !== 'true') throw new Error('Standard noise removal is not the starting choice');
    if ((await page.locator('.studio button[data-act=l-lvl][data-v="2"]').getAttribute('aria-pressed')) !== 'true' || (await page.locator('.studio button[data-act=s-q][data-v="0"]').getAttribute('aria-pressed')) !== 'true') throw new Error('Leveling (YouTube) should start on and the subtitles off');
    const inEditor = await page.getByTestId('studio').evaluate((el) => el.innerText + '\n' + [...el.querySelectorAll('[data-tip]')].map((e) => e.dataset.tip).join('\n'));
    await page.getByTestId('info-open').click();
    const inWindow = await page.getByTestId('info-modal').innerText();
    await page.getByTestId('info-close').click();
    if (/highs/i.test(inEditor + inWindow)) throw new Error('"highs" is in the editor: ' + (inEditor + inWindow).match(/.{0,40}highs.{0,40}/is)?.[0]);
    if (await page.locator('input[name=highs], input[name=highs-alone]').count() || await page.getByTestId('highs-hint').count()) throw new Error('a Restore highs box is in the editor');
    const rebuildHint = (await page.getByTestId('voice-rebuild').getAttribute('data-tip')).trim();
    if (rebuildHint !== REBUILD_HINT(shownRate(perMinute('denoise', 'rebuild').usdPerMinute))) throw new Error('rebuild hint: ' + rebuildHint);
    const order = await page.evaluate(() => {
      const find = (sel) => document.querySelector(`.studio ${sel}`);
      const things = [['strength', 'button[data-act=v-str]'], ['remover', 'button[data-act=v-rem]'], ['rebuild', '[data-testid=voice-rebuild]']].map(([n, sel]) => [n, find(sel)]);
      if (things.some(([, el]) => !el)) return { missing: things.filter(([, el]) => !el).map(([n]) => n) };
      const out = [];
      for (let i = 1; i < things.length; i++) out.push(`${things[i - 1][0]} before ${things[i][0]}: ${!!(things[i - 1][1].compareDocumentPosition(things[i][1]) & Node.DOCUMENT_POSITION_FOLLOWING)}`);
      return { out };
    });
    if (order.missing || order.out.some((x) => x.endsWith('false'))) throw new Error('order: ' + JSON.stringify(order));
    return `no Restore highs; "${rebuildHint.slice(-60)}"; ${order.out.map((x) => x.split(':')[0]).join(', ')}`;
  } finally {
    await page.unroute('**/api/billing/pricing');
    await page.evaluate(() => localStorage.removeItem('acw.autoSettings'));
  }
});

await check('the price on Start is exactly the server\'s own estimate for the timeline in the editor, for one section per track and for a track with more than one (the page asks for it, the same timeline without the file; Details adds it up from the price list); no restore_highs is sent', async () => {
  await freshApp();
  await choose('noisy.wav');
  await pick(page, 'noise', 'denoise');
  await setLeveling(page, 'youtube');
  const T = await seconds();
  const whole1 = await startIsServers({ atLeast: listPrice({ noise: T, level: T }) });
  await page.getByTestId('bar-det').click();
  const lines = (await page.getByTestId('details').innerText()).replace(/\s+/g, ' ');
  if ((await page.getByTestId('details-total').innerText()) !== words(whole1.answer.estCostMicros) || !/Noise removal · 0:\d\d/.test(lines) || !/Leveling · 0:\d\d/.test(lines)) throw new Error('Details: ' + lines);
  if (!/Includes a small amount for each run\. Parts are rounded up to whole seconds\./.test(lines)) throw new Error('Details does not say the price includes each run\'s amount: ' + lines);
  if (/highs/i.test(lines)) throw new Error('a highs line in Details: ' + lines);
  await page.getByTestId('bar-det').click();
  posts.length = 0;
  const base = await send();
  sameTimeline(whole1, base);
  if ('restore_highs' in base.options) throw new Error('sent restore_highs ' + base.options.restore_highs);
  shape(base, 'leveling,noise', 1);
  // a track with more than one section: the same, for the parts
  await dragRange(page, 2, 6);
  await page.locator('.studio button[data-act=v-rem][data-v="1"]').click(); // Light, for 0:02-0:06 only
  const multi = await startIsServers({ atLeast: listPrice({ noise: 2 + 4 + started(T - 6), level: T }) });
  shape({ ...multi.body, options: { audio_format: 'flac', caption_formats: [], embed_subtitles: false } }, 'leveling,noise', 1);
  const parts = multi.body.timeline.layers.noise;
  if (parts.length !== 3 || parts[1].remover !== 'light' || parts[1].start !== 2 || parts[1].end !== 6) throw new Error('timeline asked about: ' + JSON.stringify(parts));
  await page.getByTestId('bar-det').click();
  if ((await page.getByTestId('details-total').innerText()) !== words(multi.answer.estCostMicros)) throw new Error('Details total ' + await page.getByTestId('details-total').innerText());
  await page.getByTestId('bar-det').click();
  if (!(await start().isEnabled())) throw new Error('Start is off once the price is here');
  const again = await send();
  sameTimeline(multi, again);
  return `${words(whole1.answer.estCostMicros)} (${whole1.answer.estCostMicros} micros; the price list alone: ${listPrice({ noise: T, level: T })}) for ${T.toFixed(1)} s in one piece, ${words(multi.answer.estCostMicros)} (${multi.answer.estCostMicros} micros) with a Light stretch`;
});

await check('with Voice rebuild chosen the strength and remover choices go away, the rebuild is sent and charged (and the server\'s estimate has no highs line); choosing Noise removal again brings them back', async () => {
  await withPricing((j) => ({ ...j, rebuildAvailable: true }));
  await rebuildPricing();
  try {
    await freshApp();
    await choose('noisy.wav');
    await pick(page, 'noise', 'denoise');
    await setLeveling(page, 'youtube');
    const T = await seconds();
    const askedPlain = await startIsServers({ atLeast: listPrice({ noise: T, level: T }) });
    const plain = await mainButton(page);
    await setVoice(page, 'rebuild');
    if (await page.locator('.studio button[data-act=v-str]').count()) throw new Error('strength still offered with the rebuild');
    // only the rebuild (and Leveling) is charged: the price changed with the choice, and it is the server's for the new timeline
    const asked = await startIsServers({ atLeast: listPrice({ rebuild: T, level: T }) });
    if (asked.answer.estCostMicros === askedPlain.answer.estCostMicros) throw new Error('the price did not change with the rebuild');
    if (asked.body.timeline.layers.noise) throw new Error('the rebuilt voice is also priced as noise removal: ' + JSON.stringify(asked.body.timeline.layers));
    const startWith = await mainButton(page);
    posts.length = 0;
    const sent = await send();
    sameTimeline(asked, sent);
    if ('restore_highs' in sent.options) throw new Error('sent: ' + JSON.stringify(sent));
    shape(sent, 'leveling,rebuild', 1);
    const quote = await apiCall(page, 'POST', 'jobs/estimate', withoutFile(sent));
    if (quote.status === 200 && quote.body.lines.some((l) => l.item === 'highs')) throw new Error('the server charges Restore highs with the rebuild');
    await setVoice(page, 'noise');
    await page.locator('.studio button[data-act=v-str]').first().waitFor();
    return `${plain} Standard noise removal, ${startWith} rebuilt (${shownRate(perMinute('denoise', 'rebuild').usdPerMinute)} a minute)`;
  } finally {
    await unrebuildPricing();
    await page.unroute('**/api/billing/pricing');
  }
});

await check('Voice Off, Leveling on its own: nothing chosen at first (Start is off with the reason), then Leveling makes a paid job at the price the server gives for Leveling alone (its estimate has one line, Leveling), sent as a leveling part only', async () => {
  await freshApp();
  await choose('noisy.wav');
  await pick(page, 'noise', 'off');
  await page.getByText(/Choose at least one thing/).waitFor();
  if ((await page.locator('.studio button[data-act=l-lvl][data-v="0"]').getAttribute('aria-pressed')) !== 'true') throw new Error('Leveling is on with nothing else chosen');
  if (await start().isEnabled()) throw new Error('Start is on with nothing chosen');
  if (await page.locator('input[name=highs-alone]').count() || /highs/i.test(await page.getByTestId('studio').innerText())) throw new Error('a Restore highs box is in the editor');
  await setLeveling(page, 'youtube');
  const T = await seconds();
  const asked = await startIsServers({ atLeast: listPrice({ level: T }) });
  posts.length = 0;
  const b = await send();
  sameTimeline(asked, b);
  const L = shape(b, 'leveling', 1);
  if (!whole(L.leveling, T) || L.leveling[0].loudness_lufs !== -14) throw new Error('sent: ' + JSON.stringify(b));
  if (asked.answer.lines.length !== 1 || asked.answer.lines[0].item !== 'leveling') throw new Error('the server\'s estimate: ' + JSON.stringify(asked.answer));
  // a longer file makes the words precise: ten minutes of Leveling are about $0.01 by the price list, and the run's own amount (about $0.0016) comes on top
  await page.getByTestId('different-file').click();
  await choose('long10min.mp3');
  await pick(page, 'noise', 'off');
  await setLeveling(page, 'youtube');
  const T2 = await seconds();
  const long = listPrice({ level: T2 });
  const askedLong = await startIsServers({ atLeast: long });
  if (!/^\$0\.01/.test(words(long)) || askedLong.answer.estCostMicros - long > 2000) throw new Error(`ten minutes: Start ${await mainButton(page)}, the price list alone ${long} micros, the server ${askedLong.answer.estCostMicros}`);
  return `${words(asked.answer.estCostMicros)} for ${T.toFixed(1)} s, ${words(askedLong.answer.estCostMicros)} for ${Math.round(T2)} s`;
});

await check('nothing chosen: Start is off with the reason; a file name with direction-changing characters is shown with them marked, never reversed', async () => {
  await freshApp();
  await choose('noisy.wav');
  await pick(page, 'noise', 'off');
  await page.getByText(/Choose at least one thing/).waitFor();
  if (await start().isEnabled()) throw new Error('start enabled with nothing chosen');
  await page.unroute('**/api/me');
  await page.reload();
  await page.getByTestId('available').waitFor();
  const buffer = readFileSync(join(FILES, 'noisy.wav'));
  await page.setInputFiles('input[type=file]', { name: 'evil‮gpj.wav', mimeType: 'audio/wav', buffer });
  await start().waitFor({ timeout: 20000 });
  const shown = await page.getByTestId('file-name').innerText();
  if (/[‪-‮⁦-⁩]/.test(shown) || !shown.includes('□')) throw new Error(JSON.stringify(shown));
  await openOutput();
  const outName = await page.getByTestId('out-name').inputValue();
  await closePop();
  if (/[‪-‮⁦-⁩]/.test(outName)) throw new Error('the result name keeps the direction-changing character: ' + JSON.stringify(outName));
  return shown;
});

await check('prices that cannot be loaded: Start stays off and the page says so, with "Try again"', async () => {
  await page.route('**/api/billing/pricing', (r) => r.fulfill({ status: 500, json: { detail: 'broken' } }));
  try {
    await freshApp();
    await choose('noisy.wav');
    await pick(page, 'noise', 'denoise');
    await page.getByText('couldn’t load the prices').waitFor({ timeout: 5000 });
    if (!(await page.locator('[data-testid=notes]').getByRole('button', { name: 'Try again' }).count())) throw new Error('no retry button');
    if (await start().isEnabled()) throw new Error('Start is enabled without a price');
    return 'ok';
  } finally {
    await page.unroute('**/api/billing/pricing');
  }
});

// ---- real paid jobs (the test lets them through; this account gets its credit now) ----
admin('grant', EMAIL, '5', 'files test');
await page.unroute('**/api/me');
passThrough = true;
const sha256 = (b) => createHash('sha256').update(b).digest('hex');
const waitDone = async (id) => {
  for (let i = 0; i < 240; i++) {
    const j = await apiCall(page, 'GET', `jobs/${id}`);
    if (['done', 'failed', 'canceled'].includes(j.body.status)) return j.body;
    await page.waitForTimeout(2500);
  }
  throw new Error('job timeout');
};
const jobOutputHash = (id) => page.evaluate(async (id) => {
  const j = await (await fetch(`/api/jobs/${id}`)).json();
  const audio = j.outputs.find((o) => o.kind === 'audio');
  const buf = await (await fetch(`/api/jobs/${id}/outputs/${encodeURIComponent(audio.name)}`)).arrayBuffer();
  const h = await crypto.subtle.digest('SHA-256', buf);
  return { name: audio.name, bytes: buf.byteLength, sha: [...new Uint8Array(h)].map((x) => x.toString(16).padStart(2, '0')).join('') };
}, id);
/** Uploads the raw file through the API, bypassing the browser's FLAC step (the reference for "same result as before"); no audio_format, so it is the API default. `body`: the settings of the job to compare with (the page's own, with its timeline). */
const rawJob = async (file, body) => {
  const bytes = readFileSync(file);
  const made = await apiCall(page, 'POST', 'jobs', { ...body, filename: file.split('/').pop(), sizeBytes: bytes.length });
  if (made.status !== 200) throw new Error('raw create: ' + JSON.stringify(made));
  const put = await page.evaluate(async ({ url, b64 }) => {
    const me = await (await fetch('/api/me')).json();
    const bin = Uint8Array.from(atob(b64), (c) => c.charCodeAt(0));
    return (await fetch(`/${url}`, { method: 'PUT', headers: { 'X-CSRF-Token': me.csrfToken, 'Content-Type': 'application/octet-stream' }, body: bin })).status;
  }, { url: made.body.uploadUrl, b64: bytes.toString('base64') });
  if (put !== 200) throw new Error('raw upload ' + put);
  return made.body.id;
};

await check('a WAV is sent as FLAC (smaller, the declared name and size match) and the cleaned result is the same as uploading the WAV itself', async () => {
  await freshApp();
  await page.evaluate(() => {
    window.__puts = [];
    const send = XMLHttpRequest.prototype.send;
    const open = XMLHttpRequest.prototype.open;
    XMLHttpRequest.prototype.open = function (m, u, ...rest) { this.__m = m; this.__u = u; return open.call(this, m, u, ...rest); };
    XMLHttpRequest.prototype.send = function (body) { if (this.__m === 'PUT' && body && body.size != null) window.__puts.push({ size: body.size }); return send.call(this, body); };
  });
  const reqs = [];
  page.on('request', (r) => { if (r.method() === 'POST' && new URL(r.url()).pathname.endsWith('/api/jobs')) reqs.push(r.postDataJSON()); });
  await choose('noisy.wav');
  await pick(page, 'noise', 'denoise');
  await waitStartPrice();
  await start().click();
  const banner = await page.getByTestId('submitted').innerText({ timeout: 90000 });
  const m = /Compressed for upload: ([\d.]+ [KMG]?B) → ([\d.]+ [KMG]?B)/.exec(banner);
  if (!m) throw new Error('no compression note: ' + banner);
  const put = await page.evaluate(() => window.__puts.at(-1)?.size ?? -1);
  const orig = statSync(join(FILES, 'noisy.wav')).size;
  if (reqs[0].filename !== 'noisy.flac' || reqs[0].sizeBytes !== put || !(put < orig)) throw new Error(`declared ${reqs[0].filename} ${reqs[0].sizeBytes}, uploaded ${put}, original ${orig}`);
  const jobs = await apiCall(page, 'GET', 'jobs');
  const done = await waitDone(jobs.body.jobs[0].id);
  if (done.status !== 'done' || done.inputName !== 'noisy.wav') throw new Error('job ' + done.status + ' ' + (done.error || ''));
  const viaFlac = await jobOutputHash(done.id);
  // the same settings as the page chose (its timeline), so the only difference is the browser's FLAC step
  const { audio_format: _sent, ...pageOptions } = reqs[0].options;
  const { filename: _f, sizeBytes: _s, displayName: _d, ...pageBody } = reqs[0];
  const rawDone = await waitDone(await rawJob(join(FILES, 'noisy.wav'), { ...pageBody, options: pageOptions }));
  if (rawDone.status !== 'done') throw new Error('raw job ' + rawDone.status);
  const viaRaw = await jobOutputHash(rawDone.id);
  if (reqs[0].options.audio_format !== 'flac' || rawDone.options.audio_format !== 'flac' || !viaFlac.name.endsWith('.flac')) throw new Error(`FLAC is the default for server audio: page ${reqs[0].options.audio_format}, API ${rawDone.options.audio_format}, file ${viaFlac.name}`);
  if (viaFlac.sha !== viaRaw.sha) throw new Error(`outputs differ: ${viaFlac.bytes} B vs ${viaRaw.bytes} B`);
  return `${orig} B -> ${put} B (${(100 * put / orig).toFixed(0)}%); output identical (${viaFlac.bytes} B, ${sha256(Buffer.from(viaFlac.sha)).slice(0, 8)})`;
});

await check('echo removal on a real two-track file: the server finds both tracks (no "only one track" note, an echo part in the job\'s timeline) and the job finishes', async () => {
  await freshApp();
  await choose('echo2.mp4');
  await pick(page, 'noise', 'denoise');
  const sw = page.getByTestId('echo-switch');
  if ((await sw.getAttribute('aria-checked')) !== 'true') await sw.click();
  await start().click();
  await page.getByTestId('submitted').waitFor({ timeout: 60000 });
  const id = jobIdOf(page);
  for (let i = 0; i < 90; i++) {
    const j = await apiCall(page, 'GET', `jobs/${id}`);
    if (j.body?.status === 'done') {
      const tl = await apiCall(page, 'GET', `jobs/${id}/timeline`);
      if (tl.status !== 200 || !tl.body.timeline.layers.echo?.length || tl.body.secondTrack !== true || j.body.notice) throw new Error(JSON.stringify({ job: j.body, timeline: tl.body?.timeline, second: tl.body?.secondTrack }));
      return `done: ${j.body.outputs.map((o) => o.name).join(', ')}`;
    }
    if (['failed', 'canceled'].includes(j.body?.status)) throw new Error(JSON.stringify(j.body));
    await page.waitForTimeout(1000);
  }
  throw new Error('job did not finish');
});

console.log('LOGS:\n' + logs.join('\n'));
await browser.close();
process.exit(summary().some((r) => !r.ok) ? 1 : 0);
