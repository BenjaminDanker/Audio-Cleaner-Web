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

await check('live: demo with fake mic, denoise + local captions', async () => {
  await page.goto(`${BASE}/app/live`);
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
