import { writeFileSync, statSync, readFileSync } from 'node:fs';
import { join } from 'node:path';
import { BASE, FILES, SHOTS, check, launch, summary } from './helpers.mjs';

const { browser, page, logs } = await launch();
const apiCalls = [];
page.on('request', (r) => { const u = new URL(r.url()); if (u.pathname.includes('/audio/api/')) apiCalls.push(`${r.method()} ${u.pathname}`); });

async function runFree(file, strengthLabel = 'Strong') {
  await page.goto(`${BASE}/free`);
  await page.setInputFiles('input[type=file]', join(FILES, file));
  await page.getByRole('button', { name: 'Clean audio' }).waitFor({ timeout: 20000 });
  await page.getByLabel(strengthLabel, { exact: false }).first().check().catch(() => {});
  const t0 = Date.now();
  await page.getByRole('button', { name: 'Clean audio' }).click();
  await page.waitForSelector('[data-testid=result]', { timeout: 180000 });
  const res = page.locator('[data-testid=result]');
  return { ms: Date.now() - t0, rms: Number(await res.getAttribute('data-rms')), peak: Number(await res.getAttribute('data-peak')), frames: Number(await res.getAttribute('data-frames')), channels: Number(await res.getAttribute('data-channels')) };
}

await check('free: mono 48k wav denoised', async () => {
  const r = await runFree('noisy.wav');
  
  if (Math.abs(r.frames - 508591) > 2) throw new Error(`frames ${r.frames}`);
  if (r.rms < 0.001) throw new Error(`silent rms=${r.rms}`);
  await page.screenshot({ path: join(SHOTS, 'free-done.png'), fullPage: true });
  return `frames=${r.frames} rms=${r.rms} peak=${r.peak} in ${r.ms} ms`;
});

await check('free: no server calls besides /api/me & /billing/pricing', async () => {
  const odd = apiCalls.filter((c) => !/api\/me$|billing\/pricing$/.test(c));
  if (odd.length) throw new Error(odd.join(', '));
  return apiCalls.join(', ');
});

await check('free: download WAV 16-bit matches length', async () => {
  const [dl] = await Promise.all([page.waitForEvent('download'), page.getByRole('link', { name: /Cleaned WAV/ }).click()]);
  const p = join(FILES, 'out-free.wav');
  await dl.saveAs(p);
  const b = readFileSync(p);
  const dataLen = b.readUInt32LE(40);
  if (b.toString('ascii', 0, 4) !== 'RIFF') throw new Error('not RIFF');
  const frames = dataLen / 2 / b.readUInt16LE(22);
  if (Math.abs(frames - 508591) > 2) throw new Error(`frames ${frames}`);
  return `${b.length} bytes, ${b.readUInt16LE(22)}ch ${b.readUInt32LE(24)} Hz`;
});

await check('free: download WAV 24-bit', async () => {
  const [dl] = await Promise.all([page.waitForEvent('download'), page.getByRole('button', { name: 'WAV (24-bit)' }).click()]);
  const p = join(FILES, 'out-free24.wav');
  await dl.saveAs(p);
  const b = readFileSync(p);
  if (b.readUInt16LE(34) !== 24) throw new Error('bits ' + b.readUInt16LE(34));
  return `${b.length} bytes`;
});

await check('free: A/B toggle keeps position', async () => {
  const orig = page.getByTestId('ab-original');
  await page.getByRole('button', { name: 'Play' }).click();
  await page.waitForTimeout(1500);
  const t1 = await orig.evaluate((e) => e.currentTime);
  const m1 = await orig.evaluate((e) => e.muted);
  await page.locator('label:has(input[value=original])').click();
  const m2 = await orig.evaluate((e) => e.muted);
  const cl = await page.getByTestId('ab-cleaned').evaluate((e) => ({ t: e.currentTime, muted: e.muted, dur: e.duration, paused: e.paused }));
  const t2 = await orig.evaluate((e) => e.currentTime);
  await page.getByRole('button', { name: 'Pause' }).click();
  if (!(t1 > 0.5)) throw new Error('not playing: ' + t1);
  if (!(m1 === true && m2 === false)) throw new Error(`mute states ${m1} ${m2}`);
  if (Math.abs(cl.t - t2) > 0.3) throw new Error(`desync ${cl.t} vs ${t2}`);
  return `t=${t1.toFixed(2)} cleaned dur ${cl.dur.toFixed(2)}`;
});

await check('free: stereo 44.1k wav (resample path)', async () => {
  const r = await runFree('stereo44.wav');
  if (r.channels !== 2) throw new Error('channels ' + r.channels);
  if (Math.abs(r.frames - 508591) > 2) throw new Error(`frames ${r.frames}`);
  if (r.rms < 0.001) throw new Error('silent');
  return `frames=${r.frames} rms=${r.rms} in ${r.ms} ms`;
});

await check('free: mp3 input', async () => {
  const r = await runFree('noisy.mp3');
  if (r.rms < 0.001) throw new Error('silent');
  return `frames=${r.frames} rms=${r.rms}`;
});

await check('free: mp4 video (aac) -> cleaned WAV + remuxed video', async () => {
  const r = await runFree('test.mp4');
  if (r.rms < 0.001) throw new Error('silent');
  await page.screenshot({ path: join(SHOTS, 'free-video-done.png'), fullPage: true });
  const btn = page.getByTestId('dl-video');
  await btn.waitFor();
  await page.waitForFunction(() => !document.querySelector('[data-testid=dl-video]').disabled, null, { timeout: 5000 });
  const [dl] = await Promise.all([page.waitForEvent('download', { timeout: 60000 }), btn.click()]);
  const p = join(FILES, 'out-free-video' + (dl.suggestedFilename().match(/\.[a-z0-9]+$/)?.[0] || '.mp4'));
  await dl.saveAs(p);
  return `${dl.suggestedFilename()} ${statSync(p).size} bytes; frames=${r.frames}`;
});

await check('free: webm video (opus)', async () => {
  const r = await runFree('test.webm');
  const btn = page.getByTestId('dl-video');
  const [dl] = await Promise.all([page.waitForEvent('download', { timeout: 60000 }), btn.click()]);
  const p = join(FILES, 'out-free-video.webm');
  await dl.saveAs(p);
  return `${dl.suggestedFilename()} ${statSync(p).size} bytes; frames=${r.frames}`;
});

await check('free: cancel works', async () => {
  await page.goto(`${BASE}/free`);
  await page.setInputFiles('input[type=file]', join(FILES, 'stereo44.wav'));
  await page.getByRole('button', { name: 'Clean audio' }).click();
  await page.getByRole('button', { name: 'Cancel' }).click({ timeout: 20000 });
  await page.getByRole('button', { name: 'Clean audio' }).waitFor({ timeout: 5000 });
  return 'back to ready state';
});

await check('free: the noise remover is fetched once, only on /free, and a stereo file (two workers) fetches nothing again', async () => {
  const ctx = await browser.newContext({ viewport: { width: 1280, height: 900 } });
  const dfn = [];
  ctx.on('request', (r) => { const u = new URL(r.url()); if (u.pathname.includes('/dfn/')) dfn.push(u.pathname.split('/').pop()); });
  const p = await ctx.newPage();
  await p.goto(`${BASE}/`); await p.waitForSelector('table tbody tr');
  await p.goto(`${BASE}/transcribe`); await p.waitForSelector('main h1');
  await p.goto(`${BASE}/desktop`); await p.waitForSelector('main h1');
  await p.waitForTimeout(500);
  if (dfn.length) throw new Error('downloaded on another page: ' + dfn.join(', '));
  await p.goto(`${BASE}/free`);
  await p.waitForFunction(() => performance.getEntriesByType('resource').some((e) => e.name.includes('DeepFilterNet3_onnx')), null, { timeout: 15000 });
  await p.setInputFiles('input[type=file]', join(FILES, 'stereo44.wav'));
  await p.getByRole('button', { name: 'Clean audio' }).click();
  await p.waitForSelector('[data-testid=result]', { timeout: 120000 });
  const count = (n) => dfn.filter((f) => f === n).length;
  if (count('df_bg.wasm') !== 1 || count('DeepFilterNet3_onnx.tar.gz') !== 1) throw new Error('downloads: ' + dfn.join(', '));
  if (count('df.js') > 3) throw new Error('df.js fetched ' + count('df.js') + ' times');
  const channels = await p.locator('[data-testid=result]').getAttribute('data-channels');
  if (channels !== '2') throw new Error('expected a stereo run, got ' + channels);
  await ctx.close();
  return `wasm x1, model x1, df.js x${count('df.js')} (stereo, 2 workers)`;
});

await check('free: starting before the download is done shows "Getting the noise remover ready… X of Y MB" and still finishes', async () => {
  const ctx = await browser.newContext({ viewport: { width: 1280, height: 900 } });
  const p = await ctx.newPage();
  await p.route(/DeepFilterNet3_onnx\.tar\.gz/, async (route) => {
    const res = await route.fetch();
    await new Promise((r) => setTimeout(r, 6000)); // a slow connection
    await route.fulfill({ response: res });
  });
  await p.goto(`${BASE}/free`);
  await p.setInputFiles('input[type=file]', join(FILES, 'noisy.wav'));
  await p.getByRole('button', { name: 'Clean audio' }).waitFor();
  const eta = await p.getByTestId('eta').innerText().catch(() => '');
  await p.getByRole('button', { name: 'Clean audio' }).click();
  await p.waitForFunction(() => /^Getting the noise remover ready/.test(document.querySelector('[data-testid=stage]')?.textContent || ''), null, { timeout: 15000 });
  const stage = await p.getByTestId('stage').innerText();
  if (!/^Getting the noise remover ready… [\d.]+ of [\d.]+ MB$/.test(stage)) throw new Error('stage: ' + stage);
  const text = await p.getByTestId('progress-text').innerText();
  await p.waitForSelector('[data-testid=result]', { timeout: 120000 });
  await ctx.close();
  return `${stage}; "${text}"; before starting: "${eta}"`;
});

console.log('LOGS:\n' + logs.join('\n'));
await browser.close();
const bad = summary().filter((r) => !r.ok);
process.exit(bad.length ? 1 : 0);
