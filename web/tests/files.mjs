// The Cleanup page for a file: what each choice sends to the server (checked by stopping the job request, so nothing is made),
// the prices on the form (the server's own list), files that cannot be used or that come with no credit, names shown safely,
// and two real paid jobs: a WAV sent as FLAC gives the same result as the raw upload, and echo removal with two audio tracks.
// Needs the local backend (port 8792, see helpers.mjs) with `npm run build` output and the fixtures (make-fixtures.sh).
// Usage: node tests/files.mjs
import { createHash } from 'node:crypto';
import { readFileSync, statSync } from 'node:fs';
import { join } from 'node:path';
import { BASE, FILES, SHOTS, admin, check, dropFile, launch, pick, signIn, summary } from './helpers.mjs';

const EMAIL = `files-${Date.now()}@example.com`;
const SERVER_RATES = { denoise: 0.009, leveling: 0.001, rebuild: 0.03 }; // the served list (GET /api/billing/pricing), per minute
const money = (usd) => `$${usd.toFixed(3).replace(/0$/, '')}`; // the form's own rounding ($0.03, $0.009)
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

const radio = (group, value) => page.locator(`input[name=${group}][value=${value}]`);
const card = (group, value) => page.locator(`label.choice:has(input[name=${group}][value=${value}])`);
const start = () => page.getByTestId('start');
const choose = (f) => dropFile(page, join(FILES, f));
const freshApp = async () => { await page.goto(`${BASE}/app`); await page.getByTestId('available').waitFor(); };
const withMe = (patch) => page.route('**/api/me', async (r) => { const res = await r.fetch(); const j = await res.json(); await r.fulfill({ response: res, json: patch(j) }); });
const withPricing = (patch) => page.route('**/api/billing/pricing', async (r) => { const res = await r.fetch(); const j = await res.json(); await r.fulfill({ response: res, json: patch(j) }); });
const lastPost = () => posts.at(-1);
const refused = async () => { await page.getByTestId('job-error').waitFor({ timeout: 60000 }); };

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

await check('bad files: a file the browser cannot open gets the free cleanup greyed out with the reason, Denoising as the starting choice, and an honest unknown length', async () => {
  await freshApp();
  await choose('garbage.mp3');
  if (!(await radio('noise', 'browser').isDisabled())) throw new Error('free noise removal offered for a file the browser cannot open');
  const why = (await card('noise', 'browser').locator('small').innerText()).trim();
  if (why !== 'Your browser can’t open this file type. Paid Denoising can do it.') throw new Error('reason: ' + why);
  if (!(await radio('noise', 'denoise').isChecked())) throw new Error('Denoising should be the starting choice');
  await page.getByTestId('length-unknown').waitFor({ timeout: 15000 });
  await page.screenshot({ path: join(SHOTS, 'cleanup-unreadable.png'), fullPage: true });
  return why;
});

await check('credit: an account with no balance cannot start a paid job (Start off, the reason, "Add credit"); free subtitles on a 10 min file start with no credit', async () => {
  await freshApp();
  await choose('noisy.wav');
  await pick(page, 'noise', 'denoise');
  const msg = page.locator('.banner-warn', { hasText: /your balance is/ });
  await msg.waitFor();
  if (!(await msg.getByRole('link', { name: 'Add credit' }).count())) throw new Error('no Add credit link');
  if (await start().isEnabled()) throw new Error('start enabled with no credit');
  const reason = (await msg.innerText()).trim();
  await freshApp();
  await choose('long10min.mp3');
  await pick(page, 'noise', 'off');
  await pick(page, 'subs', 'browser');
  await page.getByRole('button', { name: 'Start – free' }).waitFor();
  if (!(await page.getByRole('button', { name: 'Start – free' }).isEnabled())) throw new Error('free start disabled for a free 10 min file');
  return reason;
});

// ---- what each choice sends (the balance is topped up on the page only; the job is stopped by the test) ----
await withMe((j) => ({ ...j, balance: { ...j.balance, available_micros: 5_000_000, balance_micros: 5_000_000 } }));

await check('denoising: the prices on the form are the server\'s; Gentle sends 12 dB, the noise is automatic by default, Heavy noise is sent as heavy', async () => {
  await freshApp();
  await choose('noisy.wav');
  await pick(page, 'noise', 'denoise');
  const denoiseHint = await card('noise', 'denoise').locator('small').innerText();
  if (!denoiseHint.includes(money(SERVER_RATES.denoise))) throw new Error('Denoising price on the form: ' + denoiseHint);
  const lh = await page.getByTestId('leveling-hint').innerText();
  if (!lh.includes(money(SERVER_RATES.leveling))) throw new Error('Leveling price on the form: ' + lh);
  if (!(await page.locator('input[type=checkbox][name=leveling]').isChecked())) throw new Error('Leveling should be on by default');
  await page.getByTestId('strength-strength').getByText('Gentle', { exact: true }).click();
  posts.length = 0;
  await start().click();
  await refused();
  const b = lastPost();
  if (b.kind !== 'denoise' || b.denoiseTier !== 'standard' || b.options.atten_lim_db !== 12 || b.options.noise !== 'auto' || b.options.leveling !== true || 'free' in b) throw new Error(JSON.stringify(b));
  await page.locator('#noisekind').selectOption('heavy');
  await start().click();
  await refused();
  if (lastPost().options.noise !== 'heavy') throw new Error('noise sent: ' + lastPost().options.noise);
  return `12 dB, auto, then heavy; prices ${money(SERVER_RATES.denoise)} and ${money(SERVER_RATES.leveling)} a minute`;
});

await check('leveling alone, subtitles alone, and both together each send exactly what the form says (Standard = denoise_transcribe, free subtitles add nothing, Extra accurate = premium transcribe)', async () => {
  await freshApp();
  await choose('noisy.wav');
  await pick(page, 'noise', 'off');
  await page.locator('input[name=leveling-alone]').check();
  await page.locator('#loud-alone').selectOption('-14');
  await page.getByTestId('estimate').waitFor();
  posts.length = 0;
  await start().click();
  await refused();
  let b = lastPost();
  if (b.kind !== 'denoise' || b.denoiseTier !== 'off' || b.options.leveling !== true || b.options.loudness_lufs !== -14 || b.transcribeTier) throw new Error('leveling only: ' + JSON.stringify(b));

  await pick(page, 'noise', 'denoise');
  await pick(page, 'subs', 'standard');
  await page.getByTestId('estimate').waitFor();
  if (!/^Start – about \$0\.00\d+$/.test((await start().innerText()).trim())) throw new Error('label: ' + await start().innerText());
  await start().click();
  await refused();
  b = lastPost();
  if (b.kind !== 'denoise_transcribe' || b.denoiseTier !== 'standard' || b.transcribeTier !== 'local' || 'free' in b) throw new Error('denoise + standard: ' + JSON.stringify(b));

  await pick(page, 'subs', 'browser');
  await page.getByText(/the subtitles made on your device are free/).waitFor();
  await start().click();
  await refused();
  b = lastPost();
  if (b.kind !== 'denoise' || b.denoiseTier !== 'standard' || b.transcribeTier || 'free' in b) throw new Error('denoise + free subtitles: ' + JSON.stringify(b));

  await pick(page, 'noise', 'browser');
  await pick(page, 'subs', 'standard');
  await start().click();
  await refused();
  b = lastPost();
  if (b.kind !== 'transcribe' || b.denoiseTier || b.transcribeTier !== 'local' || 'free' in b) throw new Error('free cleanup + standard: ' + JSON.stringify(b));

  await pick(page, 'noise', 'off');
  await pick(page, 'subs', 'accurate');
  await page.getByTestId('estimate').waitFor();
  await start().click();
  await refused();
  b = lastPost();
  if (b.kind !== 'transcribe' || b.transcribeTier !== 'premium' || 'free' in b) throw new Error('extra accurate: ' + JSON.stringify(b));
  return 'denoise / denoise_transcribe / transcribe premium as chosen';
});

await check('the line before Start names exactly what will download by itself: the cleaned file under its chosen name, and the subtitle formats ticked (with "inside the video" when that is ticked)', async () => {
  await freshApp();
  await choose('noisy.wav');
  await pick(page, 'noise', 'denoise');
  await pick(page, 'subs', 'standard');
  const line = () => page.getByTestId('will-download').innerText();
  let t = await line();
  if (!/noisy-cleaned\.flac/.test(t) || !/SRT, VTT, TXT/.test(t)) throw new Error('default: ' + t);
  await page.locator('label.check').filter({ hasText: /^SRT$/ }).locator('input').uncheck();
  await page.locator('label.check').filter({ hasText: /^VTT$/ }).locator('input').uncheck();
  await page.getByTestId('out-name').fill('talk.mp3');
  t = await line();
  if (!/talk\.mp3/.test(t) || !/\(TXT\)/.test(t) || /SRT|VTT/.test(t)) throw new Error('after the changes: ' + t);
  await page.locator('label.check').filter({ hasText: /^TXT$/ }).locator('input').uncheck();
  await page.locator('label.check').filter({ hasText: /^SRT$/ }).locator('input').check();
  await pick(page, 'subs', 'off');
  t = await line();
  if (!/talk\.mp3/.test(t) || /subtitles/.test(t)) throw new Error('with the subtitles off: ' + t);
  return t.replace(/\s+/g, ' ');
});

await check('echo removal and translation: the echo box is offered only for Denoising, sends true/false as ticked, uploads one smaller .mka; Spanish is sent as translate', async () => {
  await freshApp();
  await choose('echo2.mp4');
  if (!(await radio('noise', 'browser').isChecked())) throw new Error('free cleanup should be the starting choice');
  const note = (await page.getByTestId('echo-browser-note').innerText()).replace(/\s+/g, ' ');
  if (!/second audio track[\s\S]*needs Denoising[\s\S]*free cleanup in your browser can.t do it/.test(note)) throw new Error('note: ' + note);
  const box = page.getByRole('checkbox', { name: /Remove speaker echo using the second audio track/ });
  if (await box.count()) throw new Error('echo box offered for free cleanup');
  await pick(page, 'noise', 'denoise');
  await box.waitFor();
  await start().click();
  await refused();
  if (lastPost().options.echo_reference_track !== false) throw new Error('unticked should send false');
  await box.check();
  await start().click();
  await refused();
  const b = lastPost();
  const size = statSync(join(FILES, 'echo2.mp4')).size;
  if (b.kind !== 'denoise' || b.options.echo_reference_track !== true || !/\.mka$/.test(b.filename) || b.sizeBytes >= size) throw new Error(JSON.stringify(b));
  await pick(page, 'noise', 'off');
  await page.getByRole('button', { name: 'Remove' }).click();
  await choose('noisy.wav');
  await pick(page, 'noise', 'denoise');
  await pick(page, 'subs', 'standard');
  await page.getByLabel('Translation language 1').selectOption('es');
  await start().click();
  await refused();
  if (JSON.stringify(lastPost().options.translate) !== '["es"]') throw new Error('translate: ' + JSON.stringify(lastPost().options));
  return `${b.filename} ${b.sizeBytes} bytes of ${size}`;
});

await check('rebuild the voice: its own price (the server\'s) and no strength/noise choices; the job says rebuild', async () => {
  await withPricing((j) => ({ ...j, rebuildAvailable: true }));
  try {
    await freshApp();
    await choose('noisy.wav');
    await pick(page, 'noise', 'denoise');
    await page.getByRole('checkbox', { name: /Rebuild the voice/ }).check();
    if (await page.locator('#noisekind').count() || await page.getByTestId('strength-strength').count()) throw new Error('strength/noise still shown with rebuild');
    const hint = await page.getByTestId('rebuild-hint').innerText();
    if (!hint.includes(money(SERVER_RATES.rebuild))) throw new Error('rebuild price on the form: ' + hint);
    await start().click();
    await refused();
    if (lastPost().denoiseTier !== 'rebuild') throw new Error('denoiseTier sent: ' + lastPost().denoiseTier);
    return hint;
  } finally {
    await page.unroute('**/api/billing/pricing');
  }
});

// ---- Rebuild the voice with its words, Leveling on its own; Restore highs is gone ----
const REBUILD_HINT = (price) => `For an echoey room, a muffled voice, phone-quality or damaged sound, or distortion from recording too loud: the whole voice is rebuilt, which also brings back the high sounds that are missing. Slower, ${price} a minute, and on very echoey recordings it can change a word, so listen before you publish.`;
const priceList = await (await fetch(`${BASE}/api/billing/pricing`)).json();
const perMinute = (product, tier) => priceList.rates.find((r) => r.product === product && r.tier === tier);
const shownRate = (usd) => `$${usd.toFixed(4).replace(/(\.\d\d\d*?)0+$/, '$1')}`;
const { money: fmtMoney } = await import('../src/format.js');

await check('the form has no Restore highs (not in the price list, not on the form); Rebuild the voice has its words and price, in this order: Strength, Background noise, Rebuild the voice', async () => {
  if (priceList.rates.some((r) => r.product === 'highs')) throw new Error('the price list still has Restore highs: ' + JSON.stringify(priceList.rates.filter((r) => r.product === 'highs')));
  await withPricing((j) => ({ ...j, rebuildAvailable: true }));
  try {
    // without "Automatic best settings" a file starts from the usual choices; this browser remembers the switch
    await freshApp();
    await page.evaluate(() => localStorage.setItem('acw.autoSettings', 'off'));
    await freshApp(); // loaded again with the switch off
    await page.setInputFiles('input[type=file]', join(FILES, 'noisy.wav'));
    await start().waitFor({ timeout: 20000 });
    if (!(await radio('noise', 'denoise').isChecked())) throw new Error('Denoising is not the starting choice');
    await page.getByTestId('rebuild-hint').waitFor();
    if (await page.locator('input[name=highs], input[name=highs-alone]').count() || await page.getByTestId('highs-hint').count()) throw new Error('a Restore highs box is on the form');
    if (/highs/i.test(await page.locator('form').innerText())) throw new Error('"highs" is on the form: ' + (await page.locator('form').innerText()).match(/.{0,40}highs.{0,40}/is)?.[0]);
    const rebuildHint = (await page.getByTestId('rebuild-hint').innerText()).trim();
    if (rebuildHint !== REBUILD_HINT(shownRate(perMinute('denoise', 'rebuild').usdPerMinute))) throw new Error('rebuild hint: ' + rebuildHint);
    const order = await page.evaluate(() => {
      const form = document.querySelector('form.job-form');
      const find = (sel) => form.querySelector(sel);
      const things = [['strength', '[data-testid=strength-strength]'], ['noise', '#noisekind'], ['rebuild', 'input[name=rebuild]']].map(([n, sel]) => [n, find(sel)]);
      if (things.some(([, el]) => !el)) return { missing: things.filter(([, el]) => !el).map(([n]) => n) };
      const out = [];
      for (let i = 1; i < things.length; i++) out.push(`${things[i - 1][0]} before ${things[i][0]}: ${!!(things[i - 1][1].compareDocumentPosition(things[i][1]) & Node.DOCUMENT_POSITION_FOLLOWING)}`);
      return { out };
    });
    if (order.missing || order.out.some((x) => x.endsWith('false'))) throw new Error('order: ' + JSON.stringify(order));
    return `no Restore highs; "${rebuildHint.slice(-40)}"; ${order.out.map((x) => x.split(':')[0]).join(', ')}`;
  } finally {
    await page.unroute('**/api/billing/pricing');
    await page.evaluate(() => localStorage.removeItem('acw.autoSettings'));
  }
});

await check('the estimate on the form is exactly the server\'s own estimate for the same choices, and no restore_highs is sent', async () => {
  await freshApp();
  await choose('noisy.wav');
  await pick(page, 'noise', 'denoise');
  await page.locator('input[name=leveling]').check();
  const startText = async () => (await start().innerText()).trim();
  posts.length = 0;
  await start().click();
  await refused();
  const base = lastPost();
  const dur = base.durationSeconds;
  const r = await apiCall(page, 'POST', 'jobs/estimate', { kind: 'denoise', denoiseTier: 'standard', durationSeconds: dur, options: { leveling: true } });
  if (r.status !== 200) throw new Error('estimate ' + JSON.stringify(r));
  if ('restore_highs' in base.options) throw new Error('sent restore_highs ' + base.options.restore_highs);
  if (r.body.lines.some((l) => l.item === 'highs')) throw new Error('the server has a highs line: ' + JSON.stringify(r.body.lines));
  if ((await startText()) !== `Start – about ${fmtMoney(r.body.estCostMicros)}`) throw new Error(`Start: "${await startText()}", the server says ${r.body.estCostMicros}`);
  const est = (await page.getByTestId('estimate').innerText()).replace(/\s+/g, ' ');
  if (!est.includes(`Estimated cost: ${fmtMoney(r.body.estCostMicros)}`)) throw new Error('estimate line: ' + est);
  if (base.denoiseTier !== 'standard') throw new Error('sent: ' + JSON.stringify(base));
  return `${fmtMoney(r.body.estCostMicros)} for ${dur.toFixed(1)} s`;
});

await check('with Voice rebuild ticked the strength and noise choices go away, the rebuild is sent and charged (and the server\'s estimate has no highs line); unticking it brings them back', async () => {
  await withPricing((j) => ({ ...j, rebuildAvailable: true }));
  try {
    await freshApp();
    await choose('noisy.wav');
    await pick(page, 'noise', 'denoise');
    const rebuildRate = perMinute('denoise', 'rebuild').usdPerMinute;
    const startText = async () => (await start().innerText()).trim();
    const plain = await startText();
    await page.getByRole('checkbox', { name: /Rebuild the voice/ }).check();
    posts.length = 0;
    await start().click();
    await refused();
    const sent = lastPost();
    if (sent.denoiseTier !== 'rebuild' || 'restore_highs' in sent.options) throw new Error('sent: ' + JSON.stringify(sent));
    // only the rebuild (and Leveling) is charged
    const quote = await apiCall(page, 'POST', 'jobs/estimate', { kind: 'denoise', denoiseTier: 'rebuild', durationSeconds: sent.durationSeconds, options: { leveling: true } });
    const startWith = await startText();
    if (quote.status === 200 && startWith !== `Start – about ${fmtMoney(quote.body.estCostMicros)}`) throw new Error(`Start with the rebuild: "${startWith}", the server says ${quote.body.estCostMicros}`);
    if (quote.status === 200 && quote.body.lines.some((l) => l.item === 'highs')) throw new Error('the server charges Restore highs with the rebuild');
    await page.getByRole('checkbox', { name: /Rebuild the voice/ }).uncheck();
    await page.getByTestId('strength-strength').waitFor();
    return `${plain} Denoising, ${startWith} rebuilt (${shownRate(rebuildRate)} a minute)`;
  } finally {
    await page.unroute('**/api/billing/pricing');
  }
});

await check('noise Off, Leveling on its own: nothing chosen at first (Start is off with the reason), then Leveling makes a paid job at the Leveling price alone, sent as denoiseTier off with restore_highs not in it', async () => {
  await freshApp();
  await choose('noisy.wav');
  await pick(page, 'noise', 'off');
  const box = page.locator('input[name=leveling-alone]');
  await box.waitFor();
  if (await box.isChecked()) throw new Error('Leveling is ticked with nothing else chosen');
  await page.getByText(/Choose at least one thing/).waitFor();
  if (await start().isEnabled()) throw new Error('Start is on with nothing chosen');
  if (await page.locator('input[name=highs-alone]').count()) throw new Error('a Restore highs box is on the form');
  await box.check();
  await page.getByTestId('estimate').waitFor();
  posts.length = 0;
  await start().click();
  await refused();
  const b = lastPost();
  if (b.kind !== 'denoise' || b.denoiseTier !== 'off' || b.options.leveling !== true || 'restore_highs' in b.options || b.transcribeTier) throw new Error('sent: ' + JSON.stringify(b));
  const quote = await apiCall(page, 'POST', 'jobs/estimate', { kind: 'denoise', denoiseTier: 'off', durationSeconds: b.durationSeconds, options: { leveling: true } });
  const micros = Math.ceil((1000 * Math.ceil(b.durationSeconds)) / 60);
  if (quote.status !== 200 || quote.body.estCostMicros !== micros || quote.body.lines.length !== 1 || quote.body.lines[0].item !== 'leveling') throw new Error('the server\'s estimate: ' + JSON.stringify(quote));
  if ((await start().innerText()).trim() !== `Start – about ${fmtMoney(micros)}`) throw new Error(`Start: "${await start().innerText()}", wanted ${fmtMoney(micros)}`);
  return `${fmtMoney(micros)} for ${b.durationSeconds.toFixed(1)} s`;
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
  const shown = await page.locator('.file-chip .name').innerText();
  if (/[‪-‮⁦-⁩]/.test(shown) || !shown.includes('□')) throw new Error(JSON.stringify(shown));
  return shown;
});

await check('prices that cannot be loaded: Start stays off and the page says so, with "Try again"', async () => {
  await page.route('**/api/billing/pricing', (r) => r.fulfill({ status: 500, json: { detail: 'broken' } }));
  try {
    await freshApp();
    await choose('noisy.wav');
    await pick(page, 'noise', 'denoise');
    await page.getByText('couldn’t load the prices').waitFor({ timeout: 5000 });
    if (!(await page.getByRole('button', { name: 'Try again' }).count())) throw new Error('no retry button');
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
/** Uploads the raw file through the API, bypassing the browser's FLAC step (the reference for "same result as before"); no audio_format, so it is the API default. `options`: the settings of the job to compare with (the page's automatic settings now depend on the file). */
const rawJob = async (file, options = { loudness_lufs: -14, atten_lim_db: 100, caption_formats: [] }) => {
  const bytes = readFileSync(file);
  const made = await apiCall(page, 'POST', 'jobs', { kind: 'denoise', denoiseTier: 'standard', filename: file.split('/').pop(), sizeBytes: bytes.length, durationSeconds: 10.6, options });
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
  await page.getByTestId('estimate').waitFor();
  await page.getByRole('button', { name: /^Start – about/ }).click();
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
  // the same settings as the page chose by itself (Automatic best settings), so the only difference is the browser's FLAC step
  const { audio_format: _sent, ...pageOptions } = reqs[0].options;
  const rawDone = await waitDone(await rawJob(join(FILES, 'noisy.wav'), pageOptions));
  if (rawDone.status !== 'done') throw new Error('raw job ' + rawDone.status);
  const viaRaw = await jobOutputHash(rawDone.id);
  if (reqs[0].options.audio_format !== 'flac' || rawDone.options.audio_format !== 'flac' || !viaFlac.name.endsWith('.flac')) throw new Error(`FLAC is the default for server audio: page ${reqs[0].options.audio_format}, API ${rawDone.options.audio_format}, file ${viaFlac.name}`);
  if (viaFlac.sha !== viaRaw.sha) throw new Error(`outputs differ: ${viaFlac.bytes} B vs ${viaRaw.bytes} B`);
  return `${orig} B -> ${put} B (${(100 * put / orig).toFixed(0)}%); output identical (${viaFlac.bytes} B, ${sha256(Buffer.from(viaFlac.sha)).slice(0, 8)})`;
});

await check('echo removal on a real two-track file: the server finds both tracks (no "only one track" note) and the job finishes', async () => {
  await freshApp();
  await choose('echo2.mp4');
  await pick(page, 'noise', 'denoise');
  await page.getByRole('checkbox', { name: /Remove speaker echo/ }).check();
  await start().click();
  await page.getByTestId('submitted').waitFor({ timeout: 60000 });
  const id = await page.locator('[data-testid=job]').first().getAttribute('data-job-id');
  for (let i = 0; i < 90; i++) {
    const j = await apiCall(page, 'GET', `jobs/${id}`);
    if (j.body?.status === 'done') {
      if (j.body.options.echo_reference_track !== true || j.body.notice) throw new Error(JSON.stringify(j.body));
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
