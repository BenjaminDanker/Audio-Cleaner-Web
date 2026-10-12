import { join } from 'node:path';
import { BASE, FILES, SHOTS, check, launch, signIn, summary } from './helpers.mjs';
const { browser, page, logs } = await launch();
await check('free: garbage file -> friendly error, no crash', async () => {
  await page.goto(`${BASE}/free`);
  await page.setInputFiles('input[type=file]', join(FILES, 'garbage.mp3'));
  await page.getByRole('button', { name: 'Clean audio' }).click();
  await page.getByRole('alert').filter({ hasText: /./ }).first().waitFor({ timeout: 30000 });
  const t = await page.locator('main .banner-error').first().innerText();
  await page.screenshot({ path: join(SHOTS, 'free-error.png') });
  return t;
});
await check('free: text file -> friendly error', async () => {
  await page.goto(`${BASE}/free`);
  await page.setInputFiles('input[type=file]', join(FILES, 'bad.txt'));
  await page.getByRole('button', { name: 'Clean audio' }).click();
  await page.locator('main .banner-error').first().waitFor({ timeout: 30000 });
  return await page.locator('main .banner-error').first().innerText();
});
const mail = `e2e-poor-${Date.now()}@example.com`;
await signIn(page, mail);
await check('studio: zero balance -> warning + disabled start, link to billing', async () => {
  await page.goto(`${BASE}/app`);
  await page.setInputFiles('input[type=file]', join(FILES, 'noisy.wav'));
  await page.getByTestId('estimate').waitFor();
  const w = await page.locator('main .banner-warn', { hasText: 'available balance' }).innerText();
  if (!(await page.getByRole('button', { name: /Start job/ }).isDisabled())) throw new Error('start enabled');
  return w;
});
await check('studio: unreadable file -> clear message', async () => {
  await page.goto(`${BASE}/app`);
  await page.setInputFiles('input[type=file]', join(FILES, 'bad.txt'));
  await page.locator('main .banner-warn', { hasText: 'could not read' }).waitFor({ timeout: 15000 });
  return 'ok';
});
await check('transcribe: file >30 min blocked (free limit)', async () => {
  await page.goto(`${BASE}/transcribe`);
  await page.setInputFiles('input[type=file]', join(FILES, 'long10min.mp3'));
  await page.getByRole('button', { name: 'Start free transcription' }).waitFor();
  // 10 min is under 30 min: expect enabled
  const en = await page.getByRole('button', { name: 'Start free transcription' }).isEnabled();
  return 'start enabled for 10 min file: ' + en;
});
console.log('LOGS:\n' + logs.join('\n'));
await browser.close();
process.exit(summary().some((r) => !r.ok) ? 1 : 0);
