import { join } from 'node:path';
import { BASE, FILES, SHOTS, check, dropFile, launch, signIn, summary } from './helpers.mjs';

// Formats the browser can't read itself go through libav.js (src/media/libav.js): free noise removal works for them too,
// and the decoder is only downloaded when such a file is chosen.
const EMAIL = process.env.E2E_EMAIL || 'e2e-main@example.com';
const SOURCE_FRAMES = 508591; // noisy.wav at 48 kHz; the WMA and AVI fixtures are made from it (tests/make-fixtures.sh)

const { browser, page } = await launch();
const libavFetches = [];
page.on('request', (r) => { if (r.url().includes('/dfn/libav/')) libavFetches.push(new URL(r.url()).pathname.split('/').pop()); });
await signIn(page, EMAIL);

async function runFree(file) {
  await page.goto(`${BASE}/app`);
  await dropFile(page, join(FILES, file));
  await page.locator('input[name=noise][value=browser]:checked').waitFor({ timeout: 30000 });
  await page.getByRole('button', { name: 'Start - free' }).click();
  const res = page.locator('[data-testid=result]');
  await res.waitFor({ timeout: 180000 });
  return { rms: Number(await res.getAttribute('data-rms')), frames: Number(await res.getAttribute('data-frames')), channels: Number(await res.getAttribute('data-channels')) };
}

const close = (frames) => Math.abs(frames - SOURCE_FRAMES) < SOURCE_FRAMES * 0.02; // codec padding is a few ms

await check('formats: a WAV does not download the extra decoder', async () => {
  const r = await runFree('noisy.wav');
  if (libavFetches.length) throw new Error('fetched ' + libavFetches.join(', '));
  return `frames=${r.frames}`;
});

await check('formats: WMA is cleaned for free in the browser (length known up front)', async () => {
  await page.goto(`${BASE}/app`);
  await dropFile(page, join(FILES, 'noisy.wma'));
  await page.locator('.file-chip .meta').filter({ hasText: /0:1[01]/ }).waitFor({ timeout: 30000 });
  if (await page.getByTestId('length-unknown').count()) throw new Error('length shown as a guess');
  const r = await runFree('noisy.wma');
  if (!close(r.frames)) throw new Error(`frames ${r.frames}`);
  if (r.rms < 0.001) throw new Error(`silent rms=${r.rms}`);
  await page.screenshot({ path: join(SHOTS, 'formats-wma.png'), fullPage: true });
  return `frames=${r.frames} rms=${r.rms}; decoder files: ${[...new Set(libavFetches)].join(', ')}`;
});

await check('formats: AVI (AC-3 sound) is cleaned in the browser; the sound comes back as WAV, with a note why', async () => {
  await page.goto(`${BASE}/app`);
  await dropFile(page, join(FILES, 'noisy.avi'));
  await page.getByTestId('audio-only-note').waitFor({ timeout: 30000 });
  if (await page.getByRole('checkbox', { name: /Keep the video/ }).count()) throw new Error('keep-video offered');
  const r = await runFree('noisy.avi');
  if (!close(r.frames)) throw new Error(`frames ${r.frames}`);
  if (r.rms < 0.001) throw new Error(`silent rms=${r.rms}`);
  if (await page.getByTestId('dl-video').count()) throw new Error('video download offered');
  return `frames=${r.frames} channels=${r.channels}`;
});

await browser.close();
process.exit(summary().some((r) => !r.ok) ? 1 : 0);
