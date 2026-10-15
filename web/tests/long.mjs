import { join } from 'node:path';
import { BASE, FILES, SHOTS, dropFile, launch, signIn } from './helpers.mjs';
const file = process.argv[2] || 'long10min.mp3';
const { browser, page, logs } = await launch(['--enable-precise-memory-info']);
await signIn(page, `long-${Date.now()}@example.com`);
await page.goto(`${BASE}/app`);
await dropFile(page, join(FILES, file));
console.log('meta:', await page.locator('.file-chip .meta').innerText(), '|', await page.getByTestId('eta').innerText().catch(() => ''));
const t0 = Date.now();
await page.getByRole('button', { name: 'Start - free' }).click();
let maxHeap = 0;
const timer = setInterval(async () => {
  try {
    const t = await page.getByTestId('progress-text').innerText({ timeout: 1000 });
    const h = await page.evaluate(() => performance.memory?.usedJSHeapSize || 0);
    maxHeap = Math.max(maxHeap, h);
    console.log(`${((Date.now() - t0) / 1000).toFixed(0)}s  ${t}  heap ${(h / 1048576).toFixed(0)} MB`);
  } catch { /* finished */ }
}, 8000);
await page.waitForSelector('[data-testid=result]', { timeout: 1200000 });
clearInterval(timer);
const res = page.locator('[data-testid=result]');
console.log('DONE in', ((Date.now() - t0) / 1000).toFixed(1), 's frames', await res.getAttribute('data-frames'), 'rms', await res.getAttribute('data-rms'), 'maxHeap', (maxHeap / 1048576).toFixed(0), 'MB', logs.join(';'));
await page.screenshot({ path: join(SHOTS, 'free-long-done.png') });
await browser.close();
