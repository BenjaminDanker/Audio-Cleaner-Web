// Free noise removal in the browser (Cleanup page): what it makes and how it is downloaded, the file types it reads itself
// (WAV, MP3, WMA and AVI through libav.js, stereo at 44.1 kHz, MP4 video), that nothing about the file goes to our server, that the
// noise remover is fetched once, and the free + free combination (browser noise removal + free subtitles, no job at all).
// Needs the local backend (port 8792, see helpers.mjs) with `npm run build` output and the fixtures (make-fixtures.sh).
// Usage: node tests/free.mjs
import { readFileSync, statSync } from 'node:fs';
import { join } from 'node:path';
import { BASE, FILES, SHOTS, check, dropFile, launch, pick, settleRecommendation, signIn, skip, summary } from './helpers.mjs';

const EMAIL = process.env.E2E_EMAIL || 'e2e-main@example.com'; // the free option needs an account
const SOURCE_FRAMES = 508591; // noisy.wav at 48 kHz; the other fixtures are made from it (tests/make-fixtures.sh)
const close = (frames) => Math.abs(frames - SOURCE_FRAMES) < SOURCE_FRAMES * 0.02; // codec padding is a few ms

const { browser, page, logs } = await launch();
const apiCalls = [];
const libavFetches = [];
page.on('request', (r) => {
  const u = new URL(r.url());
  if (u.pathname.includes('/audio/api/')) apiCalls.push(`${r.method()} ${u.pathname}`);
  if (u.pathname.includes('/dfn/libav/')) libavFetches.push(u.pathname.split('/').pop());
});
await signIn(page, EMAIL);
apiCalls.length = 0;

async function runFree(file, strengthLabel = 'Strong') {
  await page.goto(`${BASE}/app`);
  await dropFile(page, join(FILES, file)); // "Free, in your browser" is the default whenever the browser can open the file
  await page.getByTestId('strength-lstrength').getByText(strengthLabel, { exact: true }).click();
  const t0 = Date.now();
  await page.getByRole('button', { name: 'Start – free' }).click();
  await page.waitForSelector('[data-testid=result]', { timeout: 180000 });
  const res = page.locator('[data-testid=result]');
  return { ms: Date.now() - t0, rms: Number(await res.getAttribute('data-rms')), peak: Number(await res.getAttribute('data-peak')), frames: Number(await res.getAttribute('data-frames')), channels: Number(await res.getAttribute('data-channels')) };
}

await check('free: a mono 48 kHz WAV is cleaned in the browser (length kept, not silent), and the cleaned WAV downloads with the same length', async () => {
  const r = await runFree('noisy.wav');
  if (!close(r.frames)) throw new Error(`frames ${r.frames}`);
  if (r.rms < 0.001) throw new Error(`silent rms=${r.rms}`);
  if (libavFetches.length) throw new Error('a WAV fetched the extra decoder: ' + libavFetches.join(', '));
  const [dl] = await Promise.all([page.waitForEvent('download'), page.getByRole('link', { name: /Cleaned WAV/ }).click()]);
  const p = join(FILES, 'out-free.wav');
  await dl.saveAs(p);
  const b = readFileSync(p);
  if (b.toString('ascii', 0, 4) !== 'RIFF') throw new Error('not RIFF');
  const got = b.readUInt32LE(40) / 2 / b.readUInt16LE(22);
  if (!close(got)) throw new Error(`downloaded frames ${got}`);
  await page.screenshot({ path: join(SHOTS, 'free-done.png'), fullPage: true });
  return `frames=${r.frames} rms=${r.rms} in ${r.ms} ms; download ${b.length} bytes`;
});

await check('free: nothing about the file is sent to our server (only the account, prices and the job list are read)', async () => {
  const odd = apiCalls.filter((c) => !/api\/me$|billing\/pricing$|^GET .*api\/jobs$/.test(c));
  if (odd.length) throw new Error(odd.join(', '));
  return apiCalls.join(', ');
});

await check('free: other inputs clean in the browser: 44.1 kHz stereo WAV (two channels kept), MP3', async () => {
  const st = await runFree('stereo44.wav');
  if (st.channels !== 2 || !close(st.frames) || st.rms < 0.001) throw new Error(JSON.stringify(st));
  const mp3 = await runFree('noisy.mp3');
  if (mp3.rms < 0.001) throw new Error('mp3 silent');
  return `stereo: frames=${st.frames}; mp3: frames=${mp3.frames}`;
});

await check('free: WMA (read by libav.js) shows its real length and is cleaned; AVI (AC-3 sound) gives the sound back as WAV with a note why', async () => {
  await page.goto(`${BASE}/app`);
  await dropFile(page, join(FILES, 'noisy.wma'));
  await page.locator('.file-chip .meta').filter({ hasText: /0:1[01]/ }).waitFor({ timeout: 30000 });
  if (await page.getByTestId('length-unknown').count()) throw new Error('length shown as a guess');
  const w = await runFree('noisy.wma');
  if (!close(w.frames) || w.rms < 0.001) throw new Error(`wma ${JSON.stringify(w)}`);
  await page.goto(`${BASE}/app`);
  await dropFile(page, join(FILES, 'noisy.avi'));
  await page.getByTestId('audio-only-note').waitFor({ timeout: 30000 });
  if (await page.getByRole('checkbox', { name: /Keep the video/ }).count()) throw new Error('keep-video offered');
  const a = await runFree('noisy.avi');
  if (!close(a.frames) || a.rms < 0.001) throw new Error(`avi ${JSON.stringify(a)}`);
  if (await page.getByTestId('dl-video').count()) throw new Error('video download offered for an AVI');
  return `wma frames=${w.frames}; avi frames=${a.frames}; decoder files: ${[...new Set(libavFetches)].join(', ')}`;
});

await check('free: an MP4 is cleaned and the video comes back with the cleaned sound', async () => {
  const r = await runFree('test.mp4');
  if (r.rms < 0.001) throw new Error('silent');
  const btn = page.getByTestId('dl-video');
  await btn.waitFor();
  await page.waitForFunction(() => !document.querySelector('[data-testid=dl-video]').disabled, null, { timeout: 5000 });
  const [dl] = await Promise.all([page.waitForEvent('download', { timeout: 60000 }), btn.click()]);
  const p = join(FILES, 'out-free-video' + (dl.suggestedFilename().match(/\.[a-z0-9]+$/)?.[0] || '.mp4'));
  await dl.saveAs(p);
  return `${dl.suggestedFilename()} ${statSync(p).size} bytes`;
});

await check('free: the noise remover is fetched once, only after a file is chosen (not on opening the page), and a stereo file uses the same copy', async () => {
  const ctx = await browser.newContext({ viewport: { width: 1280, height: 900 } });
  const dfn = [];
  ctx.on('request', (r) => { const u = new URL(r.url()); if (u.pathname.includes('/dfn/')) dfn.push(u.pathname.split('/').pop()); });
  const p = await ctx.newPage();
  try {
    await signIn(p, EMAIL);
    await p.goto(`${BASE}/app`); await p.waitForSelector('main h1');
    await p.waitForTimeout(1500);
    if (dfn.length) throw new Error('downloaded before a file was chosen: ' + dfn.join(', '));
    await p.setInputFiles('input[type=file]', join(FILES, 'stereo44.wav'));
    await p.getByTestId('start').waitFor({ timeout: 20000 });
    await settleRecommendation(p);
    await pick(p, 'noise', 'browser');
    await p.getByRole('button', { name: 'Start – free' }).waitFor({ timeout: 20000 });
    await p.getByRole('button', { name: 'Start – free' }).click();
    await p.waitForSelector('[data-testid=result]', { timeout: 120000 });
    const count = (n) => dfn.filter((f) => f === n).length;
    if (count('df_bg.wasm') !== 1 || count('DeepFilterNet3_onnx.tar.gz') !== 1) throw new Error('downloads: ' + dfn.join(', '));
    const channels = await p.locator('[data-testid=result]').getAttribute('data-channels');
    if (channels !== '2') throw new Error('expected a stereo run, got ' + channels);
    return 'wasm x1, model x1 (stereo)';
  } finally {
    await ctx.close();
  }
});

skip('free: when the browser asks to save data, nothing is downloaded before Start',
  'the recommendation quick listen loads the noise remover on file choice even with navigator.connection.saveData on (product promise: nothing before Start)');

await check('free + free: browser noise removal AND free subtitles at the same time, no job is sent at all', async () => {
  const posts = [];
  await page.route('**/api/jobs', (r) => { if (r.request().method() === 'POST') posts.push(r.request().postDataJSON()); return r.continue(); });
  await page.goto(`${BASE}/app`);
  await dropFile(page, join(FILES, 'noisy.wav'));
  await pick(page, 'subs', 'browser');
  const label = (await page.getByTestId('start').innerText()).trim();
  if (label !== 'Start – free') throw new Error('label: ' + label);
  await page.getByTestId('start').click();
  await page.waitForSelector('[data-testid=result]', { timeout: 180000 });
  await page.getByTestId('subs-result').waitFor({ timeout: 300000 });
  await page.unroute('**/api/jobs');
  if (posts.length) throw new Error('a job was sent: ' + JSON.stringify(posts.at(-1)));
  const [dl] = await Promise.all([page.waitForEvent('download'), page.getByRole('link', { name: /Cleaned WAV/ }).click()]);
  await page.screenshot({ path: join(SHOTS, 'free-free-combo.png'), fullPage: true });
  return `download ${dl.suggestedFilename()} + subtitles (${await page.getByTestId('subs-result').getAttribute('data-cues')} cues)`;
});

console.log('LOGS:\n' + logs.join('\n'));
await browser.close();
process.exit(summary().some((r) => !r.ok) ? 1 : 0);
