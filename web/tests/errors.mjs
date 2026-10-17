// Files that cannot be used, and balances that are too low: every case says what is wrong, in plain words, and never crashes.
import { join } from 'node:path';
import { BASE, FILES, SHOTS, check, dropFile, launch, pick, signIn, summary } from './helpers.mjs';
const { browser, page, logs } = await launch();
const mail = `e2e-poor-${Date.now()}@example.com`;
await signIn(page, mail);
const radio = (group, value) => page.locator(`input[name=${group}][value=${value}]`);
await check('cleanup: a file that is not audio -> Free noise removal is greyed out with the reason, no crash', async () => {
  await page.goto(`${BASE}/app`);
  await dropFile(page, join(FILES, 'garbage.mp3'));
  if (!(await radio('noise', 'browser').isDisabled())) throw new Error('free noise removal offered for a file the browser cannot read');
  const t = await page.locator('label.choice:has(input[name=noise][value=browser]) small').innerText();
  await page.screenshot({ path: join(SHOTS, 'cleanup-garbage.png') });
  return t;
});
await check('cleanup: a tiny text file -> refused at once as too small to hold sound, no options with made-up numbers', async () => {
  await page.goto(`${BASE}/app`);
  await page.setInputFiles('input[type=file]', join(FILES, 'bad.txt'));
  const box = page.getByTestId('job-error');
  await box.waitFor({ timeout: 5000 });
  if (await page.getByTestId('start').count()) throw new Error('Start offered for a 17-byte file');
  return await box.innerText();
});
await check('cleanup: a longer text file -> free noise removal greyed out with the reason', async () => {
  await page.goto(`${BASE}/app`);
  await page.waitForSelector('input[type=file]');
  await page.setInputFiles('input[type=file]', { name: 'notes.txt', mimeType: 'text/plain', buffer: Buffer.from('this is not audio\n'.repeat(400)) });
  await page.getByTestId('start').waitFor({ timeout: 20000 });
  if (!(await radio('noise', 'browser').isDisabled())) throw new Error('free noise removal offered for a text file');
  return await page.locator('label.choice:has(input[name=noise][value=browser]) small').innerText();
});
await check('cleanup: zero balance -> warning + disabled start, link to billing', async () => {
  await page.goto(`${BASE}/app`);
  await dropFile(page, join(FILES, 'noisy.wav'));
  await pick(page, 'noise', 'denoise');
  await page.getByTestId('estimate').waitFor();
  const w = await page.locator('main .banner-warn', { hasText: 'your balance is' }).innerText();
  if (!(await page.getByTestId('start').isDisabled())) throw new Error('start enabled');
  return w;
});
await check('subtitles only, free in the browser: a 10 min file can start without credit', async () => {
  await page.goto(`${BASE}/app`);
  await dropFile(page, join(FILES, 'long10min.mp3'));
  await pick(page, 'noise', 'off');
  await pick(page, 'subs', 'browser');
  await page.getByRole('button', { name: 'Start – free' }).waitFor();
  const en = await page.getByRole('button', { name: 'Start – free' }).isEnabled();
  if (!en) throw new Error('start disabled for a free 10 min file');
  return 'start enabled for 10 min file';
});
console.log('LOGS:\n' + logs.join('\n'));
await browser.close();
process.exit(summary().some((r) => !r.ok) ? 1 : 0);
