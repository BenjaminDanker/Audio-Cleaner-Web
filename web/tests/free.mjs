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

console.log('LOGS:\n' + logs.join('\n'));
await browser.close();
const bad = summary().filter((r) => !r.ok);
process.exit(bad.length ? 1 : 0);
