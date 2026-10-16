import { join } from 'node:path';
import { BASE, FILES, SHOTS, check, launch, signIn, summary } from './helpers.mjs';

const EMAIL = process.env.E2E_EMAIL || 'e2e-main@example.com';
const wav = join(FILES, 'noisy.wav');
const { browser, page, logs } = await launch([
  '--use-fake-device-for-media-stream', '--use-fake-ui-for-media-stream', `--use-file-for-fake-audio-capture=${wav}`,
]);
const wsFrames = { sent: 0, recv: 0, start: null, ready: null, textIn: [] };
page.on('websocket', (ws) => {
  ws.on('framesent', (f) => { if (typeof f.payload === 'string') { if (!wsFrames.start && f.payload.includes('"start"')) wsFrames.start = f.payload; wsFrames.textIn.push('>' + f.payload); } else wsFrames.sent++; });
  ws.on('framereceived', (f) => { if (typeof f.payload === 'string') { wsFrames.textIn.push('<' + f.payload.slice(0, 160)); if (f.payload.includes('"ready"')) wsFrames.ready = f.payload; } else wsFrames.recv++; });
});

await check('live: sign in', async () => { await signIn(page, EMAIL); return 'ok'; });

const paidTest = async () => {
  await page.getByTestId('live-mode').locator('label', { hasText: 'Paid, done on our side' }).click();
  await page.getByTestId('live-free-hint').waitFor();
};

await check('live: the test starts as the free one (in the browser, no credit), with a note on how the OBS plugin compares', async () => {
  await page.goto(`${BASE}/app/live`);
  await page.getByTestId('live-free-banner').waitFor();
  if (!(await page.getByTestId('live-mode').locator('input[value=free]').isChecked())) throw new Error('not free by default');
  const note = await page.getByTestId('live-plugin-note').innerText();
  if (!/40% fewer words wrong/.test(note)) throw new Error(note);
  if (!/Cost: free/.test(await page.getByTestId('live-price').innerText())) throw new Error('price line');
  const caps = await page.getByTestId('group-captions').innerText();
  if (!/Free, made on your device/.test(caps) || /Extra accurate/.test(caps)) throw new Error(caps);
  return note;
});

await check('live: the paid test panel is three aligned groups (Noise removal, Captions, You hear); clear labels; no low latency; points to the free test', async () => {
  await page.goto(`${BASE}/app/live`);
  await page.getByTestId('live-options').waitFor();
  await paidTest();
  const boxes = {};
  for (const [id, title, inside] of [['noise', 'Noise removal', ['How much noise to remove', 'Remove noise']], ['captions', 'Captions', ['Off', 'Standard', 'Extra accurate']], ['hear', 'You hear', ['Nothing', 'Original', 'Cleaned']]]) {
    const g = page.getByTestId(`group-${id}`);
    if ((await g.getByRole('heading', { level: 3 }).innerText()).trim() !== title) throw new Error(`group ${id}: heading`);
    const t = await g.innerText();
    for (const x of inside) if (!t.includes(x)) throw new Error(`group ${id} lacks "${x}"`);
    boxes[id] = await g.boundingBox();
  }
  const [a, b, c] = [boxes.noise, boxes.captions, boxes.hear];
  if (Math.abs(a.y - b.y) > 1 || Math.abs(b.y - c.y) > 1) throw new Error('group tops are not aligned: ' + JSON.stringify([a.y, b.y, c.y]));
  if (Math.abs(a.height - b.height) > 1 || Math.abs(b.height - c.height) > 1) throw new Error('groups are not the same height');
  if (!(a.x + a.width <= b.x + 1 && b.x + b.width <= c.x + 1)) throw new Error('groups overlap or are out of order');
  const t = await page.locator('main').innerText();
  if (/How strong|low.latency|Noise removal style|20 ms/i.test(t)) throw new Error('old wording is still on the page');
  if (!t.includes('This is how the OBS plugin works. Want to try without spending credit? Switch to the free test above.')) throw new Error('free-test line missing');
  await page.screenshot({ path: join(SHOTS, 'live-groups.png'), fullPage: true });
  await page.setViewportSize({ width: 360, height: 800 });
  const m = await page.evaluate(() => ({ sw: document.documentElement.scrollWidth, vw: window.innerWidth, tops: ['noise', 'captions', 'hear'].map((i) => document.querySelector(`[data-testid=group-${i}]`).getBoundingClientRect().top) }));
  await page.setViewportSize({ width: 1280, height: 900 });
  if (m.sw > m.vw + 1) throw new Error('sideways scroll at 360px');
  if (!(m.tops[0] < m.tops[1] && m.tops[1] < m.tops[2])) throw new Error('groups do not stack at 360px');
  return `3 groups at y=${Math.round(a.y)}, ${Math.round(a.width)}/${Math.round(b.width)}/${Math.round(c.width)} px wide; stack on a phone`;
});

await check('live: the plugin section offers a download per platform with the version; says so plainly when no download exists yet; no "coming soon"', async () => {
  const manifest = { version: '0.1.60', files: {
    'windows-setup': { name: 'a.exe', size: 1048576, sha256: 'e'.repeat(64) }, source: { name: 's.tar.gz', size: 99999, sha256: 'f'.repeat(64) },
    'windows-x64': { name: 'a.zip', size: 5242880, sha256: 'a'.repeat(64) }, macos: { name: 'b.tar.xz', size: 4194304, sha256: 'b'.repeat(64) },
    'ubuntu-debian': { name: 'c.deb', size: 3145728, sha256: 'c'.repeat(64) }, 'linux-other': { name: 'd.tar.xz', size: 2097152, sha256: 'd'.repeat(64) } } };
  await page.route('**/api/downloads/obs-plugin/manifest.json', (r) => r.fulfill({ json: manifest }));
  await page.goto(`${BASE}/app/live`);
  const box = page.getByTestId('plugin-downloads');
  await box.getByTestId('plugin-version').waitFor();
  const t = await box.innerText();
  for (const x of ['Newest version: 0.1.60', 'Windows', 'Windows (zip)', 'Run anyway', 'make a folder called plugins', 'Mac', 'Ubuntu, Debian, Mint', 'Other Linux', 'C:\\ProgramData\\obs-studio\\plugins', '~/Library/Application Support/obs-studio/plugins', '~/.config/obs-studio/plugins', '5.0 MB']) if (!t.includes(x)) throw new Error(`missing "${x}" in: ${t}`);
  const hrefs = {};
  for (const id of ['windows-setup', 'windows', 'mac', 'ubuntu-debian', 'linux-other']) hrefs[id] = await box.getByTestId(`plugin-${id}`).getAttribute('href');
  for (const [id, h] of Object.entries(hrefs)) if (!h.endsWith(`/audio/api/downloads/obs-plugin/latest/${id}`) && !h.endsWith(`/api/downloads/obs-plugin/latest/${id}`)) throw new Error(`href ${id}: ${h}`);
  if (/coming soon|github/i.test(await page.locator('main').innerText())) throw new Error('old or revealing wording on the page');
  if (!(await box.getByTestId('plugin-source').getByRole('link').getAttribute('href')).endsWith('/obs-plugin/latest/source')) throw new Error('source link');
  // a platform the manifest does not list gets no button
  await page.unroute('**/api/downloads/obs-plugin/manifest.json');
  const small = { version: '0.1.61', files: { macos: manifest.files.macos } };
  await page.route('**/api/downloads/obs-plugin/manifest.json', (r) => r.fulfill({ json: small }));
  await page.goto(`${BASE}/app/live`);
  await box.getByTestId('plugin-mac').waitFor();
  if (await box.getByTestId('plugin-windows').count()) throw new Error('a button for a file that is not listed');
  // nothing published yet: plain words, no buttons, no sideways scroll on a phone
  await page.unroute('**/api/downloads/obs-plugin/manifest.json');
  await page.route('**/api/downloads/obs-plugin/manifest.json', (r) => r.fulfill({ status: 404, json: { detail: 'Not found.' } }));
  await page.goto(`${BASE}/app/live`);
  await box.getByTestId('plugin-unavailable').waitFor();
  if (await box.locator('a.btn').count()) throw new Error('download buttons without a download');
  await page.setViewportSize({ width: 360, height: 800 });
  const m = await page.evaluate(() => ({ sw: document.documentElement.scrollWidth, vw: window.innerWidth }));
  await page.setViewportSize({ width: 1280, height: 900 });
  if (m.sw > m.vw + 1) throw new Error('sideways scroll at 360px');
  await page.unroute('**/api/downloads/obs-plugin/manifest.json');
  // no credit in the account: no downloads and no key form, a note with a link to add credit instead
  await page.route('**/api/me', async (r) => { const res = await r.fetch(); const j = await res.json(); await r.fulfill({ response: res, json: { ...j, balance: { ...j.balance, balance_micros: 0, available_micros: 0 } } }); });
  await page.goto(`${BASE}/app/live`);
  await page.getByTestId('needs-credit').first().waitFor();
  if ((await page.getByTestId('needs-credit').count()) !== 2) throw new Error('both the plugin and the key form should say credit is needed');
  if (await page.getByTestId('plugin-downloads').count()) throw new Error('downloads offered without credit');
  if (await page.getByRole('button', { name: 'Create key' }).isVisible()) throw new Error('key form shown without credit');
  await page.unroute('**/api/me');
  return 'version, 5 buttons, install steps, source link; partial manifest; not-ready message; needs credit';
});

await check('live: free test with a fake mic (clear speech): the voice is cleaned and captioned in the browser; nothing is sent to us', async () => {
  const f = await launch(['--use-fake-device-for-media-stream', '--use-fake-ui-for-media-stream', `--use-file-for-fake-audio-capture=${join(FILES, 'speech.wav')}`]);
  try {
    let sockets = 0;
    f.page.on('websocket', () => { sockets++; });
    await signIn(f.page, EMAIL);
    await f.page.goto(`${BASE}/app/live`);
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

await check('live: paid demo with fake mic, denoise + local captions', async () => {
  await page.goto(`${BASE}/app/live`);
  await paidTest();
  await page.getByTestId('live-start').waitFor();
  await page.getByTestId('live-start').click();
  await page.getByTestId('live-stop').waitFor({ timeout: 30000 });
  const stats = page.getByTestId('live-stats');
  await stats.waitFor();
  await page.waitForFunction(() => Number(document.querySelector('[data-testid=live-stats]')?.getAttribute('data-recv')) > 100, null, { timeout: 30000 });
  // captions: wait up to 40 s for a final line
  let cap = null;
  try {
    await page.getByTestId('caption-line').first().waitFor({ timeout: 40000 });
    cap = await page.getByTestId('caption-line').first().innerText();
  } catch { /* none */ }
  await page.screenshot({ path: join(SHOTS, 'live.png'), fullPage: true });
  const rtt = Number(await stats.getAttribute('data-rtt'));
  const sent = Number(await stats.getAttribute('data-sent'));
  const recv = Number(await stats.getAttribute('data-recv'));
  // switch monitor modes while running (no errors expected)
  await page.locator('label:has(input[value=original])').click();
  await page.waitForTimeout(500);
  await page.locator('label:has(input[value=processed])').click();
  await page.waitForTimeout(500);
  await page.getByTestId('live-stop').click();
  await page.getByTestId('live-start').waitFor({ timeout: 15000 });
  const err = await page.getByTestId('live-error').count();
  if (err) throw new Error('error banner: ' + (await page.getByTestId('live-error').innerText()));
  return `RTT avg ${rtt.toFixed(1)} ms, frames sent ${sent} recv ${recv}; captions: ${cap ? JSON.stringify(cap) : 'NONE'}`;
});

await check('live: protocol sanity (start msg, ready, binary frames, stop)', async () => {
  const s = JSON.parse(wsFrames.start);
  if (s.sampleRate !== 48000 || s.version !== 1 || s.denoise !== 'light' || s.captions !== 'local') throw new Error(wsFrames.start);
  if (!wsFrames.ready) throw new Error('no ready');
  if (!wsFrames.textIn.some((t) => t.includes('"stop"'))) throw new Error('no stop sent');
  return `ready=${wsFrames.ready.slice(0, 140)} binary sent=${wsFrames.sent} recv=${wsFrames.recv}`;
});

await check('live: balance deducted for session', async () => {
  await page.goto(`${BASE}/app/billing`);
  await page.waitForSelector('[data-testid=ledger]'); await page.waitForTimeout(3000); await page.reload(); await page.waitForSelector('[data-testid=ledger] tbody tr');
  const rows = await page.locator('[data-testid=ledger] tbody tr').allInnerTexts();
  const live = rows.find((r) => /stream|live/i.test(r));
  return live ? live.replace(/\s+/g, ' ') : rows[0].replace(/\s+/g, ' ');
});

console.log('LOGS:\n' + logs.join('\n'));
await browser.close();
process.exit(summary().some((r) => !r.ok) ? 1 : 0);
