// States and small flows that need the local backend plus mocked answers: the price list (groups, per hour, phone width),
// a file job waiting in line, "short on storage" (HTTP 507), refund estimate with the card fee, and live: capacity used up
// ("full" with one-click alternatives), the caption choices and the language picker (only for extra accurate), bad_config.
// Needs the same local backend as tests/e2e.mjs (port 8792, see helpers.mjs) with `npm run build` output served by it.
// Usage: node tests/states.mjs [section ...]   sections: prices busy refund livefull
// Kept apart from qa-fixes.mjs on purpose: the backend rate-limits one IP's API calls and sign-ins, and the long
// upload sections there use most of that budget.
import { join } from 'node:path';
import { BASE, FILES, SHOTS, admin, check, dropFile, launch, pick, signIn, summary } from './helpers.mjs';

const only = process.argv.slice(2);
const want = (s) => only.length === 0 || only.includes(s);
const EMAIL = `states-${Date.now()}@example.com`;
const freshPage = (extraArgs = [], opts = {}) => launch(extraArgs, opts);

admin('login-link', EMAIL); // creates the user
admin('grant', EMAIL, '5', 'states test');

/** fetch() with the CSRF token, run inside the page (same origin). */
const apiCall = (page, method, path, body) => page.evaluate(async ({ method, path, body }) => {
  const me = await (await fetch('/audio/api/me', { credentials: 'same-origin' })).json();
  const res = await fetch(`/audio/api/${path}`, {
    method, credentials: 'same-origin',
    headers: { Accept: 'application/json', 'Content-Type': 'application/json', 'X-CSRF-Token': me.csrfToken },
    body: body === undefined ? undefined : JSON.stringify(body),
  });
  const text = await res.text();
  return { status: res.status, body: text ? JSON.parse(text) : null };
}, { method, path, body });

/** One sign-in shared by the newer sections below (the backend allows only a handful of sign-ins per IP in a short time). */
let sharedState = null;
async function signedInState() {
  if (!sharedState) {
    const l = await freshPage();
    await signIn(l.page, EMAIL);
    sharedState = await l.context.storageState();
    await l.browser.close();
  }
  return sharedState;
}

if (want('prices')) {
  for (const width of [375, 1280]) {
    const { browser, page } = await freshPage([], { viewport: { width, height: 800 } });
    await check(`price table at ${width}px: groups, one price column per hour, plain descriptions, no status column, unavailable rows hidden, no sideways scroll`, async () => {
      await page.route('**/api/billing/pricing', async (r) => {
        const res = await r.fetch();
        const j = await res.json();
        j.rates = j.rates.map((x) => (x.product === 'stream_captions' && x.tier === 'premium' ? { ...x, available: false } : x));
        await r.fulfill({ response: res, json: j });
      });
      await page.goto(`${BASE}/#pricing`);
      await page.waitForSelector('[data-testid=prices] tbody tr');
      const t = await page.getByTestId('prices').innerText();
      const need = ['Free', 'Files', 'Live streaming', 'Denoising', 'Leveling', '$0.54 / hour', '$0.009 / min', '$0.06 / hour', '$0.001 / min', '$0.24 / hour', 'Background noise removed, the voice kept natural', 'The voice evened out and the loudness set', 'Subtitles and transcripts (SRT, VTT, TXT)', 'Noise removal in your browser', 'Subtitles & transcripts in your browser', '$0.004 / min'];
      if (/account/i.test(t)) throw new Error('the price list should not talk about accounts: ' + t);
      for (const n of need) if (!t.includes(n)) throw new Error(`missing "${n}" in: ${t}`);
      if (/Available|Coming soon|low latency/i.test(t) || t.includes('Live captions, extra accurate')) throw new Error('should not show: ' + t);
      if (!(await page.locator('[data-testid=prices] thead').evaluate((e) => e.classList.contains('sr-only')))) throw new Error('a visible column header row (the status and per-minute columns are gone)');
      const w = await page.getByTestId('prices').evaluate((e) => { const wrap = e.parentElement; return { sw: wrap.scrollWidth, cw: wrap.clientWidth, page: document.documentElement.scrollWidth, iw: window.innerWidth }; });
      if (w.sw > w.cw + 1 || w.page > w.iw + 1) throw new Error('sideways scroll: ' + JSON.stringify(w));
      await page.getByTestId('prices').screenshot({ path: join(SHOTS, `prices-${width}.png`) });
      return `ok, wrap ${w.cw}px`;
    });
    await browser.close();
  }
}

if (want('busy')) {
  const { browser, page } = await freshPage([], { storageState: await signedInState() });
  await page.goto(`${BASE}/app`);
  await page.waitForSelector('main h1');
  await apiCall(page, 'POST', 'account/tos', { version: '2026-10-16' });
  const job = (id, extra) => ({ id, kind: 'denoise', free: false, status: 'queued', inputName: `${id}.wav`, createdAt: Math.floor(Date.now() / 1000), durationSeconds: 60, progress: 0, stage: null, outputs: [], costMicros: 0, expiresAt: null, queuePosition: null, queueWaitSeconds: null, ...extra });
  await check('queued job card: "In line: N ahead of you, about X min" / "You\'re next"', async () => {
    await page.route('**/api/jobs', (r) => (r.request().method() === 'GET' ? r.fulfill({ json: { jobs: [job('jq3', { queuePosition: 3, queueWaitSeconds: 300 }), job('jq1', { queuePosition: 1, queueWaitSeconds: 40 }), job('jq0', { queuePosition: 5, queueWaitSeconds: 5400 })] } }) : r.continue()));
    await page.route(/\/api\/jobs\/jq/, (r) => r.fulfill({ status: 404, json: { detail: 'gone' } }));
    await page.goto(`${BASE}/app`);
    await page.getByTestId('queue-note').first().waitFor();
    const notes = await page.getByTestId('queue-note').allInnerTexts();
    const want = ['In line: 2 ahead of you, about 5 min', 'You’re next, about 1 min', 'In line: 4 ahead of you, about 1 h 30 min'];
    for (const w of want) if (!notes.some((n) => n.trim() === w)) throw new Error(`missing "${w}" in ${JSON.stringify(notes)}`);
    await page.screenshot({ path: join(SHOTS, 'queue.png'), fullPage: true });
    return notes.join(' | ');
  });
  await check('job creation refused with HTTP 507: the message is shown warmly, nothing breaks', async () => {
    await page.unroute('**/api/jobs');
    await page.unroute(/\/api\/jobs\/jq/);
    await page.route('**/api/jobs', (r) => (r.request().method() === 'POST' ? r.fulfill({ status: 507, json: { detail: 'We are short on storage space right now. Please try again in a little while.' } }) : r.continue()));
    await page.goto(`${BASE}/app`);
    await page.waitForSelector('[data-testid=available]');
    await dropFile(page, join(FILES, 'noisy.mp3'));
    await pick(page, 'noise', 'denoise');
    await page.getByTestId('estimate').waitFor();
    await page.getByRole('button', { name: /^Start – about/ }).click();
    const box = page.getByTestId('job-error');
    await box.waitFor({ timeout: 30000 });
    const t = await box.innerText();
    if (!/short on storage space/.test(t) || /storage.*storage/s.test(t)) throw new Error(t); // the server's message, said once
    if ((await box.getAttribute('class')).includes('banner-error')) throw new Error('should be a calm warning, not an error banner');
    await page.screenshot({ path: join(SHOTS, 'storage-507.png') });
    return t.replace(/\s+/g, ' ');
  });
  await browser.close();
}

if (want('refund')) {
  const { browser, page } = await freshPage([], { storageState: await signedInState() });
  await check('refund: the button only while the server says some of the payment is unused (refundable_cents); confirmation shows "You\'ll get back about $X (unused $Y minus the card fee)"; success shows the exact refundedCents', async () => {
    const now = Math.floor(Date.now() / 1000);
    // pay1: unused $15 of $20; pay2: refunded before, $7 left; pay3: refunded down to a $0 balance ($4 of $10 refunded,
    // the rest spent: refunded < paid, but nothing to refund); pay4: older than 14 days (the server says 0)
    const payments = [
      { id: 'pay1', kind: 'topup', status: 'succeeded', amount_cents: 2000, refunded_cents: 0, refundable_cents: 1500, created_at: now - 3600 },
      { id: 'pay2', kind: 'topup', status: 'succeeded', amount_cents: 1000, refunded_cents: 300, refundable_cents: 700, created_at: now - 7200 },
      { id: 'pay3', kind: 'topup', status: 'succeeded', amount_cents: 1000, refunded_cents: 400, refundable_cents: 0, created_at: now - 9000 },
      { id: 'pay4', kind: 'topup', status: 'succeeded', amount_cents: 1000, refunded_cents: 0, refundable_cents: 0, created_at: now - 20 * 86400 },
    ];
    await page.route('**/api/billing/history', (r) => r.fulfill({ json: { payments, ledger: [] } }));
    await page.route('**/api/billing/pricing', async (r) => { const res = await r.fetch(); await r.fulfill({ response: res, json: { ...(await res.json()), paymentsEnabled: true } }); });
    let posted = null;
    await page.route('**/api/billing/refund', (r) => { posted = r.request().postDataJSON(); return r.fulfill({ json: { refundedCents: 4321 } }); });
    await page.goto(`${BASE}/app/billing`);
    await page.getByRole('button', { name: 'Refund unused' }).first().waitFor();
    const buttons = page.getByRole('button', { name: 'Refund unused' });
    if ((await buttons.count()) !== 2) throw new Error(`${await buttons.count()} refund buttons; want 2 (pay1, pay2)`);
    const rows = page.locator('tr', { has: page.getByTestId('pay-status') });
    for (const [i, want] of [[0, 1], [1, 1], [2, 0], [3, 0]]) {
      const n = await rows.nth(i).getByRole('button', { name: 'Refund unused' }).count();
      if (n !== want) throw new Error(`payment ${payments[i].id}: ${n} refund buttons, want ${want}`);
    }
    await buttons.first().click();
    const est = await page.getByTestId('refund-estimate').first().innerText();
    const unused = 1500; // what the server said, not the balance
    const fee = Math.ceil(2000 * 0.029 + 30);
    const money = (c) => `$${(c / 100).toFixed(2)}`;
    const want = `You’ll get back about ${money(unused - fee)} (unused ${money(unused)} minus the card fee of ${money(fee)}).`;
    if (est.trim() !== want) throw new Error(`"${est}" vs "${want}"`);
    await page.getByRole('button', { name: 'Yes, refund' }).click();
    await page.getByText('$43.21 is on its way back to your card').waitFor();
    if (posted?.paymentId !== 'pay1') throw new Error('posted ' + JSON.stringify(posted));
    // a second refund of a payment that was refunded before has no card fee: all of its unused part
    await page.reload();
    await page.getByRole('button', { name: 'Refund unused' }).nth(1).click();
    const est2 = (await page.getByTestId('refund-estimate').innerText()).trim();
    if (/card fee of/.test(est2) || !est2.includes(money(700))) throw new Error('second refund: ' + est2);
    return `${est.trim()} / ${est2}`;
  });
  await browser.close();
}

if (want('livefull')) {
  const wav = join(FILES, 'noisy.wav');
  const { browser, page } = await freshPage(['--use-fake-device-for-media-stream', '--use-fake-ui-for-media-stream', `--use-file-for-fake-audio-capture=${wav}`], { storageState: await signedInState() });
  await page.goto(`${BASE}/app`);
  await page.waitForSelector('main h1');
  await apiCall(page, 'POST', 'account/tos', { version: '2026-10-16' });
  const starts = [];
  let mode = 'full';
  const mock = async () => page.routeWebSocket(/\/api\/stream\/ws/, (ws) => {
    ws.onMessage((m) => {
      if (typeof m !== 'string') return;
      const j = JSON.parse(m);
      if (j.type === 'start') {
        starts.push(j);
        if (mode === 'full' && j.captions === 'premium') ws.send(JSON.stringify({ type: 'error', code: 'full', available: [{ captions: 'local' }, { denoise: null }], message: 'Live is busy right now. Standard captions or extra accurate captions on their own are free.' }));
        else ws.send(JSON.stringify({ type: 'ready', sessionId: 's1', sampleRate: 48000, hopSamples: 480, modelLatencyMs: 30, usdPerMinute: 0.004, maxSessionSeconds: 3600 }));
      } else if (j.type === 'stop') ws.close();
    });
  });
  await mock();
  const capCard = (v) => page.locator(`label.choice:has(input[name=live-captions][value=${v}])`);
  await check('live: no low-latency option anywhere; when live is full the message and one-click alternatives appear and retrying works', async () => {
    await page.goto(`${BASE}/app/live`);
    await page.getByTestId('live-mode').locator('label', { hasText: 'Paid, done on our side' }).click(); // the paid test (the free one runs in the browser)
    await page.getByTestId('live-price').waitFor();
    if (/low.latency|20 ms/i.test(await page.locator('main').innerText()) || (await page.locator('#live-den').count())) throw new Error('low latency is still on the page');
    await capCard('premium').click();
    await page.getByTestId('live-start').click();
    const box = page.getByTestId('live-full');
    await box.waitFor({ timeout: 20000 });
    const t = await box.innerText();
    if (!t.includes('Live is busy right now')) throw new Error(t);
    const opts = await page.getByTestId('live-full-option').allInnerTexts();
    if (opts.length !== 2 || !/standard noise removal \+ standard captions/i.test(opts[0]) || !/extra accurate captions|without noise removal/i.test(opts[1])) throw new Error(JSON.stringify(opts));
    if (starts[0].denoise !== 'light' || starts[0].captions !== 'premium') throw new Error('first start: ' + JSON.stringify(starts[0]));
    await page.getByTestId('live-full-option').first().click();
    await page.getByTestId('live-stop').waitFor({ timeout: 20000 });
    const second = starts[1];
    if (second.denoise !== 'light' || second.captions !== 'local') throw new Error('retry start: ' + JSON.stringify(second));
    if (await page.getByTestId('live-full').count()) throw new Error('full notice still shown after the retry worked');
    await page.getByTestId('live-stop').click();
    await page.getByTestId('live-start').waitFor({ timeout: 15000 });
    return opts.join(' | ');
  });
  await check('live: the language picker only appears with Extra accurate (any language); Standard says English only and sends no language', async () => {
    mode = 'ok';
    await page.goto(`${BASE}/app/live`);
    await page.getByTestId('live-mode').locator('label', { hasText: 'Paid, done on our side' }).click(); // the paid test (the free one runs in the browser)
    await page.getByTestId('live-price').waitFor();
    if (await page.locator('#live-lang').count()) throw new Error('language picker shown for standard captions');
    if (!/English only/.test(await capCard('local').innerText())) throw new Error('Standard card: ' + (await capCard('local').innerText()));
    if (!/any language/i.test(await capCard('premium').innerText())) throw new Error('Extra accurate card: ' + (await capCard('premium').innerText()));
    await page.getByTestId('live-start').click();
    await page.getByTestId('live-stop').waitFor({ timeout: 20000 });
    if (starts.at(-1).captions !== 'local' || starts.at(-1).language) throw new Error('standard start: ' + JSON.stringify(starts.at(-1)));
    await page.getByTestId('live-stop').click();
    await page.getByTestId('live-start').waitFor({ timeout: 15000 });
    await capCard('premium').click();
    await page.locator('#live-lang').selectOption('fr');
    await page.getByTestId('live-start').click();
    await page.getByTestId('live-stop').waitFor({ timeout: 20000 });
    if (starts.at(-1).captions !== 'premium' || starts.at(-1).language !== 'fr') throw new Error('extra accurate start: ' + JSON.stringify(starts.at(-1)));
    await page.getByTestId('live-stop').click();
    await page.getByTestId('live-start').waitFor({ timeout: 15000 });
    return 'ok';
  });
  await check('live: a bad_config answer shows the server\'s message', async () => {
    await page.unroute(/\/api\/stream\/ws/);
    await page.routeWebSocket(/\/api\/stream\/ws/, (ws) => ws.onMessage((m) => { if (typeof m === 'string' && JSON.parse(m).type === 'start') ws.send(JSON.stringify({ type: 'error', code: 'bad_config', message: 'Standard captions are English only.' })); }));
    await page.goto(`${BASE}/app/live`);
    await page.getByTestId('live-mode').locator('label', { hasText: 'Paid, done on our side' }).click(); // the paid test (the free one runs in the browser)
    await page.getByTestId('live-price').waitFor();
    await page.getByTestId('live-start').click();
    await page.getByTestId('live-error').waitFor({ timeout: 15000 });
    const t = await page.getByTestId('live-error').innerText();
    if (!t.includes('Standard captions are English only.')) throw new Error(t);
    return t;
  });
  await browser.close();
}

process.exit(summary().some((r) => !r.ok) ? 1 : 0);
