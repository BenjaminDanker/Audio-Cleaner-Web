// Live captions (Live page): the free test in the browser (nothing sent to us), the paid test from start to stop with a change
// made while it runs, its price against the server's list, a full service answered with one-click alternatives, premium captions
// shown as they arrive, the OBS plugin downloads, and the session charged in the ledger.
// The paid checks run against the local backend (port 8792); the full-service and captions checks use a mocked socket.
// Usage: node tests/live.mjs
import { join } from 'node:path';
import { BASE, FILES, SHOTS, check, launch, signIn, summary } from './helpers.mjs';

const EMAIL = process.env.E2E_EMAIL || 'e2e-main@example.com';
const wav = join(FILES, 'noisy.wav');
const { browser, page, logs } = await launch([
  '--use-fake-device-for-media-stream', '--use-fake-ui-for-media-stream', `--use-file-for-fake-audio-capture=${wav}`,
]);
const wsFrames = { start: null, textIn: [] };
page.on('websocket', (ws) => {
  ws.on('framesent', (f) => { if (typeof f.payload === 'string') { if (!wsFrames.start && f.payload.includes('"start"')) wsFrames.start = f.payload; wsFrames.textIn.push('>' + f.payload); } });
  ws.on('framereceived', (f) => { if (typeof f.payload === 'string') wsFrames.textIn.push('<' + f.payload.slice(0, 160)); });
});
const rate = (usd) => `$${(usd || 0).toFixed(4).replace(/(\.\d\d\d*?)0+$/, '$1')}`; // the page's own price rounding
const paidTest = async () => {
  await page.getByTestId('live-mode').locator('label', { hasText: 'Paid, done on our side' }).click();
  await page.getByTestId('live-free-hint').waitFor();
};

await signIn(page, EMAIL);

await check('live, free test: it runs in the browser with no credit and no connection to us; the speech is cleaned and captioned', async () => {
  const f = await launch(['--use-fake-device-for-media-stream', '--use-fake-ui-for-media-stream', `--use-file-for-fake-audio-capture=${join(FILES, 'speech.wav')}`]);
  try {
    let sockets = 0;
    f.page.on('websocket', () => { sockets++; });
    await signIn(f.page, EMAIL);
    await f.page.goto(`${BASE}/app/live`);
    await f.page.getByTestId('live-free-banner').waitFor();
    if (!/Cost: free/.test(await f.page.getByTestId('live-price').innerText())) throw new Error('price line');
    await f.page.getByTestId('live-start').click();
    await f.page.getByTestId('live-stop').waitFor({ timeout: 120000 });
    await f.page.waitForFunction(() => Number(document.querySelector('[data-testid=live-stats]')?.getAttribute('data-recv')) > 100, null, { timeout: 30000 });
    await f.page.getByTestId('caption-line').first().waitFor({ timeout: 90000 });
    const lines = await f.page.getByTestId('caption-line').allInnerTexts();
    const delay = Number(await f.page.getByTestId('live-stats').getAttribute('data-rtt'));
    await f.page.screenshot({ path: join(SHOTS, 'live-free.png'), fullPage: true });
    await f.page.getByTestId('live-stop').click();
    await f.page.getByTestId('live-start').waitFor({ timeout: 15000 });
    if (await f.page.getByTestId('live-error').count()) throw new Error(await f.page.getByTestId('live-error').innerText());
    if (!lines.some((l) => /responsible|hearing|damage|exposure/i.test(l))) throw new Error(`captions: ${JSON.stringify(lines)}`);
    if (sockets) throw new Error('a connection to the live service was opened');
    return `cleaning delay ${delay.toFixed(1)} ms; captions: ${JSON.stringify(lines.slice(0, 2))}`;
  } finally {
    await f.browser.close();
  }
});

await check('live, paid test: the price is the server\'s for the chosen captions; the start asks for denoise and captions; a change made while it runs goes through (price down, audio keeps flowing), and one that can\'t is put back with the reason', async () => {
  const pricing = await (await page.request.get(`${BASE}/api/billing/pricing`)).json();
  const r = (product, tier) => pricing.rates.find((x) => x.product === product && x.tier === tier).usdPerMinute;
  await page.goto(`${BASE}/app/live`);
  await paidTest();
  const priceEl = page.getByTestId('live-price');
  for (const [cap, want] of [['local', r('stream_denoise', 'light') + r('stream_captions', 'local')], ['premium', r('stream_denoise', 'light') + r('stream_captions', 'premium')]]) {
    await page.locator(`label.choice:has(input[name=live-captions][value="${cap}"])`).click();
    const usd = Number(await priceEl.getAttribute('data-usd-per-minute'));
    if (Math.abs(usd - want) > 1e-9) throw new Error(`${cap}: page ${usd} vs server ${want}`);
    if (!(await priceEl.innerText()).includes(rate(want))) throw new Error(`${cap}: text lacks ${rate(want)}: ${await priceEl.innerText()}`);
  }
  await page.locator('label.choice:has(input[name=live-captions][value="local"])').click();
  await page.getByTestId('live-start').click();
  await page.getByTestId('live-stop').waitFor({ timeout: 30000 });
  const start = JSON.parse(wsFrames.start || 'null');
  if (!start || start.sampleRate !== 48000 || start.denoise !== 'light' || start.captions !== 'local') throw new Error('start message: ' + wsFrames.start);
  const stats = page.getByTestId('live-stats');
  await page.waitForFunction(() => Number(document.querySelector('[data-testid=live-stats]')?.getAttribute('data-recv')) > 50, null, { timeout: 30000 });
  const before = Number(await priceEl.getAttribute('data-usd-per-minute'));
  const answers = () => wsFrames.textIn.filter((t) => t.startsWith('<{"type":"config"'));
  const n = answers().length;
  await page.locator('label.choice:has(input[name=live-captions][value=""])').click();
  for (let i = 0; i < 50 && answers().length === n; i++) await page.waitForTimeout(100);
  if (!answers().at(-1)?.includes('"ok":true')) throw new Error('captions off: ' + answers().at(-1));
  const after = Number(await priceEl.getAttribute('data-usd-per-minute'));
  if (!(after < before)) throw new Error(`price ${before} -> ${after}`);
  const r1 = Number(await stats.getAttribute('data-recv'));
  await page.waitForTimeout(1500);
  if (Number(await stats.getAttribute('data-recv')) <= r1) throw new Error('the audio stopped after the change');
  await page.getByRole('checkbox', { name: /Remove noise/ }).uncheck(); // no captions and no noise removal would leave nothing to do
  await page.getByTestId('live-error').waitFor({ timeout: 10000 });
  if (!(await page.getByRole('checkbox', { name: /Remove noise/ }).isChecked())) throw new Error('noise removal was not put back');
  const why = await page.getByTestId('live-error').innerText();
  await page.getByTestId('live-stop').click();
  await page.getByTestId('live-start').waitFor({ timeout: 15000 });
  return `price ${before} -> ${after} a minute; refused: ${why}`;
});

await check('live, paid: the session is charged in the ledger', async () => {
  await page.goto(`${BASE}/app/billing`);
  await page.waitForSelector('[data-testid=ledger] tbody tr');
  const rows = await page.locator('[data-testid=ledger] tbody tr').allInnerTexts();
  const live = rows.find((x) => /stream|live/i.test(x));
  if (!live) throw new Error('no live session in the ledger: ' + rows.slice(0, 2).join(' | '));
  return live.replace(/\s+/g, ' ');
});

await check('live, premium captions: a partial line is replaced in place by its final text, and the order holds (mocked socket)', async () => {
  let startMsg = null;
  const send = [];
  await page.routeWebSocket(/\/api\/stream\/ws/, (ws) => {
    ws.onMessage((m) => {
      if (typeof m !== 'string') return;
      const j = JSON.parse(m);
      if (j.type === 'start') {
        startMsg = j;
        ws.send(JSON.stringify({ type: 'ready', sessionId: 's1', sampleRate: 48000, hopSamples: 480, modelLatencyMs: 30, usdPerMinute: 0.029, maxSessionSeconds: 3600 }));
      } else if (j.type === 'stop') ws.close();
    });
    send.push((o) => ws.send(JSON.stringify({ type: 'caption', ...o })));
  });
  try {
    await page.goto(`${BASE}/app/live`);
    await paidTest();
    await page.locator('label.choice:has(input[name=live-captions][value=premium])').click();
    if (!(await page.getByRole('checkbox', { name: /Remove noise/ }).isChecked())) await page.getByRole('checkbox', { name: /Remove noise/ }).check();
    await page.getByTestId('live-start').click();
    await page.getByTestId('live-stop').waitFor({ timeout: 20000 });
    if (startMsg?.captions !== 'premium' || startMsg?.denoise !== 'light') throw new Error('start message: ' + JSON.stringify(startMsg));
    const emit = send[0];
    emit({ id: 'item_a', final: false, text: 'hello' });
    await page.getByTestId('caption-partial').first().waitFor();
    await page.evaluate(() => { document.querySelector('[data-testid=caption-partial]').__mark = 'A'; });
    emit({ id: 'item_a', final: false, text: 'hello wor' });
    await page.waitForFunction(() => document.querySelector('[data-testid=caption-partial]')?.textContent === 'hello wor');
    emit({ id: 'item_b', final: false, text: 'second' });
    await page.waitForFunction(() => document.querySelectorAll('[data-testid=caption-partial]').length === 2);
    emit({ id: 'item_a', final: true, text: 'Hello world.' });
    await page.waitForFunction(() => document.querySelectorAll('[data-testid=caption-line]').length === 1);
    const mark = await page.evaluate(() => document.querySelector('[data-testid=caption-line]').__mark);
    if (mark !== 'A') throw new Error('final did not replace the partial in place (different DOM node)');
    emit({ id: 'item_a', final: false, text: 'late partial' }); // must not reopen a finished line
    emit({ id: 'item_b', final: true, text: 'Second one.' });
    await page.waitForFunction(() => document.querySelectorAll('[data-testid=caption-line]').length === 2 && !document.querySelector('[data-testid=caption-partial]'));
    const lines = await page.getByTestId('caption-line').allInnerTexts();
    if (lines.join('|') !== 'Hello world.|Second one.') throw new Error('order/content: ' + lines.join('|'));
    await page.screenshot({ path: join(SHOTS, 'live-premium.png'), fullPage: true });
    await page.getByTestId('live-stop').click();
    await page.getByTestId('live-start').waitFor({ timeout: 15000 });
    return 'partials updated in place, finals kept order';
  } finally {
    await page.unrouteAll({ behavior: 'ignoreErrors' }).catch(() => {});
  }
});

await check('live, full: a busy service says so and offers the one-click alternatives; choosing one starts the session (mocked socket)', async () => {
  const starts = [];
  await page.routeWebSocket(/\/api\/stream\/ws/, (ws) => {
    ws.onMessage((m) => {
      if (typeof m !== 'string') return;
      const j = JSON.parse(m);
      if (j.type === 'start') {
        starts.push(j);
        if (starts.length === 1) ws.send(JSON.stringify({ type: 'error', code: 'full', available: [{ captions: 'local' }, { denoise: null }], message: 'Live is busy right now. Standard captions or extra accurate captions on their own are free.' }));
        else ws.send(JSON.stringify({ type: 'ready', sessionId: 's2', sampleRate: 48000, hopSamples: 480, modelLatencyMs: 30, usdPerMinute: 0.004, maxSessionSeconds: 3600 }));
      } else if (j.type === 'stop') ws.close();
    });
  });
  try {
    await page.goto(`${BASE}/app/live`);
    await paidTest();
    await page.locator('label.choice:has(input[name=live-captions][value=premium])').click();
    await page.getByTestId('live-start').click();
    const box = page.getByTestId('live-full');
    await box.waitFor({ timeout: 20000 });
    if (!(await box.innerText()).includes('Live is busy right now')) throw new Error(await box.innerText());
    const opts = await page.getByTestId('live-full-option').allInnerTexts();
    if (opts.length !== 2) throw new Error(JSON.stringify(opts));
    await page.getByTestId('live-full-option').first().click();
    await page.getByTestId('live-stop').waitFor({ timeout: 20000 });
    if (starts[1].captions !== 'local' || starts[1].denoise !== 'light') throw new Error('retry start: ' + JSON.stringify(starts[1]));
    await page.getByTestId('live-stop').click();
    await page.getByTestId('live-start').waitFor({ timeout: 15000 });
    return opts.join(' | ');
  } finally {
    await page.unrouteAll({ behavior: 'ignoreErrors' }).catch(() => {});
  }
});

await check('live, OBS plugin: a download button per published platform file, and a missing file says so with no button (mocked)', async () => {
  const manifest = { version: '0.1.60', files: {
    'windows-setup': { name: 'a.exe', size: 1048576, sha256: 'e'.repeat(64) }, source: { name: 's.tar.gz', size: 99999, sha256: 'f'.repeat(64) },
    'windows-x64': { name: 'a.zip', size: 5242880, sha256: 'a'.repeat(64) }, macos: { name: 'b.tar.xz', size: 4194304, sha256: 'b'.repeat(64) },
    'ubuntu-debian': { name: 'c.deb', size: 3145728, sha256: 'c'.repeat(64) }, 'linux-other': { name: 'd.tar.xz', size: 2097152, sha256: 'd'.repeat(64) } } };
  await page.route('**/api/downloads/obs-plugin/manifest.json', (r) => r.fulfill({ json: manifest }));
  try {
    await page.goto(`${BASE}/app/live`);
    const box = page.getByTestId('plugin-downloads');
    await box.getByTestId('plugin-version').waitFor();
    for (const id of ['windows-setup', 'windows', 'mac', 'ubuntu-debian', 'linux-other']) {
      const h = await box.getByTestId(`plugin-${id}`).getAttribute('href');
      if (!h.endsWith(`/api/downloads/obs-plugin/latest/${id}`)) throw new Error(`href ${id}: ${h}`);
    }
    await page.unroute('**/api/downloads/obs-plugin/manifest.json');
    await page.route('**/api/downloads/obs-plugin/manifest.json', (r) => r.fulfill({ status: 404, json: { detail: 'Not found.' } }));
    await page.goto(`${BASE}/app/live`);
    await box.getByTestId('plugin-unavailable').waitFor();
    if (await box.locator('a.btn').count()) throw new Error('download buttons without a download');
    return 'five platform buttons; "not ready" with no buttons';
  } finally {
    await page.unrouteAll({ behavior: 'ignoreErrors' }).catch(() => {});
  }
});

console.log('LOGS:\n' + logs.filter((l) => !/Failed to load resource|net::ERR_FAILED/.test(l)).join('\n'));
await browser.close();
process.exit(summary().some((r) => !r.ok) ? 1 : 0);
